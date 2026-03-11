# -*- coding: utf-8 -*-
"""
ActiveSpatial Ray Server
Flask HTTP 服务器，提供 REST API 接口

架构：
┌─────────────────────────────────────────────────────────────────┐
│  Training Process                                                │
│    └── BatchEnvClient (HTTP)                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓ HTTP
┌─────────────────────────────────────────────────────────────────┐
│  ActiveSpatialServer (本文件)                                    │
│    Flask App                                                     │
│      /environments  →  create_environments_batch                 │
│      /batch/reset   →  reset_batch                               │
│      /batch/step    →  step_batch                                │
│      /batch/close   →  close_batch                               │
│      /health        →  health_check                              │
│                                                                  │
│    ActiveSpatialRayService (@ray.remote)                        │
│      └── ActiveSpatialActorPool (@ray.remote)                   │
│            ├── GaussianRendererActor (cuda:0)                   │
│            ├── GaussianRendererActor (cuda:1)                   │
│            └── ...                                               │
└─────────────────────────────────────────────────────────────────┘
"""
import os
import sys
import time
import logging
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional
from flask import Flask, request, jsonify
import threading

import ray

from .ray_actor_pool import ActiveSpatialActorPool, create_actor_pool
from .ray_service import (
    ActiveSpatialRayService,
    create_ray_service,
    deserialize_observation,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Time Metrics
# =============================================================================

class TimeMetrics:
    """追踪和报告操作耗时"""
    
    def __init__(self, max_history: int = 1000, report_interval: int = 50):
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.report_interval = report_interval
        self.last_report_time = time.time()
    
    def record(self, operation: str, duration: float) -> None:
        """记录一次操作耗时"""
        self.metrics[operation].append(duration)
        self.counters[operation] += 1
        
        if self.counters[operation] % self.report_interval == 0:
            self._report_metrics()
    
    def _report_metrics(self) -> None:
        """打印指标报告"""
        current_time = time.time()
        time_since_last = current_time - self.last_report_time
        
        report = f"\n[METRICS] Time metrics report (last {time_since_last:.1f}s):\n"
        for operation, times in self.metrics.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                count = self.counters[operation]
                report += (
                    f"  {operation}: count={count}, "
                    f"avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s\n"
                )
        
        logger.info(report)
        self.last_report_time = current_time


# =============================================================================
# ActiveSpatial HTTP Server
# =============================================================================

class ActiveSpatialServer:
    """
    ActiveSpatial HTTP 服务器
    
    提供与 VAGEN BatchEnvClient 兼容的 REST API
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化服务器
        
        Args:
            config: 服务器配置，包含：
                - host: 监听地址
                - port: 监听端口
                - debug: 是否启用调试模式
                - num_gpus: GPU 数量
                - gpu_plan: 每个 GPU 上的 Renderer 数量
                - gs_root: GS 数据根目录
        """
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 5001)
        self.debug = config.get('debug', False)
        self.config = config
        
        # 创建 Flask app
        self.app = Flask(__name__)
        
        # 初始化 Ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                logging_level=logging.INFO,
            )
        
        # 创建 Actor Pool
        pool_config = {
            'logical_to_physical': config.get('logical_to_physical', list(range(config.get('num_gpus', 4)))),
            'gpu_plan': config.get('gpu_plan', [2] * config.get('num_gpus', 4)),
            'gs_root': config.get('gs_root', ''),
            'resource_prefix': config.get('resource_prefix', 'active_spatial'),
        }
        
        logger.info(f"[Server] Initializing Actor Pool with config: {pool_config}")
        self.actor_pool = create_actor_pool(pool_config)
        
        # 创建 Service
        service_config = {
            'save_dir': config.get('save_dir'),
            'gs_root': config.get('gs_root'),
        }
        self.service = create_ray_service(service_config, self.actor_pool)
        
        # 指标收集
        self.metrics = TimeMetrics(max_history=1000, report_interval=20)
        
        # 设置路由
        self._setup_routes()
        
        # 服务器状态
        self.is_running = False
        self.server_thread = None
        
        logger.info(f"[Server] ActiveSpatial Server initialized on {self.host}:{self.port}")
    
    def _setup_routes(self) -> None:
        """设置 HTTP 路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            try:
                stats = ray.get(self.service.get_stats.remote())
                return jsonify({
                    "status": "ok",
                    "message": "ActiveSpatial server is running",
                    "stats": stats,
                }), 200
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e),
                }), 500
        
        @self.app.route('/environments', methods=['POST'])
        def create_environments_batch():
            """创建环境"""
            data = request.json
            if not data or 'ids2configs' not in data:
                return jsonify({"error": "Missing required parameter: ids2configs"}), 400
            
            try:
                start_time = time.time()
                ids2configs = data['ids2configs']
                
                ray.get(self.service.create_environments_batch.remote(ids2configs))
                
                self.metrics.record('create_environments', time.time() - start_time)
                return jsonify({"success": True}), 200
                
            except Exception as e:
                logger.error(f"Error creating environments: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/batch/reset', methods=['POST'])
        def reset_batch():
            """批量重置环境"""
            data = request.json
            if not data or 'ids2seeds' not in data:
                return jsonify({"error": "Missing required parameter: ids2seeds"}), 400
            
            try:
                start_time = time.time()
                ids2seeds = data['ids2seeds']
                
                # 获取 futures
                futures = ray.get(self.service.reset_batch.remote(ids2seeds))
                
                # 并行等待所有结果
                results = {}
                for env_id, future in futures.items():
                    obs, info = ray.get(future)
                    results[env_id] = (obs, info)
                
                self.metrics.record('reset_batch', time.time() - start_time)
                return jsonify({"results": results}), 200
                
            except Exception as e:
                logger.error(f"Error resetting environments: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/batch/step', methods=['POST'])
        def step_batch():
            """批量执行动作"""
            data = request.json
            if not data or 'ids2actions' not in data:
                return jsonify({"error": "Missing required parameter: ids2actions"}), 400
            
            try:
                start_time = time.time()
                ids2actions = data['ids2actions']
                
                # 获取 futures
                futures = ray.get(self.service.step_batch.remote(ids2actions))
                
                # 并行等待所有结果
                results = {}
                for env_id, future in futures.items():
                    obs, reward, done, info = ray.get(future)
                    results[env_id] = (obs, reward, done, info)
                
                self.metrics.record('step_batch', time.time() - start_time)
                return jsonify({"results": results}), 200
                
            except Exception as e:
                logger.error(f"Error stepping environments: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/batch/compute_reward', methods=['POST'])
        def compute_reward_batch():
            """批量计算奖励"""
            data = request.json
            if not data or 'env_ids' not in data:
                return jsonify({"error": "Missing required parameter: env_ids"}), 400
            
            try:
                env_ids = data['env_ids']
                
                futures = ray.get(self.service.compute_reward_batch.remote(env_ids))
                
                results = {}
                for env_id, future in futures.items():
                    results[env_id] = ray.get(future)
                
                return jsonify({"results": results}), 200
                
            except Exception as e:
                logger.error(f"Error computing rewards: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/batch/close', methods=['POST'])
        def close_batch():
            """批量关闭环境"""
            data = request.json
            if not data or 'env_ids' not in data:
                return jsonify({"error": "Missing required parameter: env_ids"}), 400
            
            try:
                env_ids = data['env_ids']
                ray.get(self.service.close_batch.remote(env_ids))
                return jsonify({"success": True}), 200
                
            except Exception as e:
                logger.error(f"Error closing environments: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """获取统计信息"""
            try:
                stats = ray.get(self.service.get_stats.remote())
                return jsonify(stats), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def run(self, threaded: bool = True) -> None:
        """
        运行服务器
        
        Args:
            threaded: 是否在后台线程运行
        """
        if threaded:
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
            )
            self.server_thread.start()
            self.is_running = True
            logger.info(f"[Server] Started in background thread")
        else:
            self._run_server()
    
    def _run_server(self) -> None:
        """实际运行服务器"""
        self.is_running = True
        self.app.run(
            host=self.host,
            port=self.port,
            debug=self.debug,
            threaded=True,
            use_reloader=False,
        )
    
    def shutdown(self) -> None:
        """关闭服务器"""
        self.is_running = False
        # Flask 在调试模式下需要特殊处理
        logger.info("[Server] Shutting down...")


# =============================================================================
# 启动脚本
# =============================================================================

def main():
    """服务器启动入口"""
    parser = argparse.ArgumentParser(description='ActiveSpatial Ray Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5001, help='Port number')
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--renderers-per-gpu', type=int, default=2, help='Renderers per GPU')
    parser.add_argument('--gs-root', type=str, required=True, help='GS data root directory')
    parser.add_argument('--gpu-ids', type=str, default='', help='Comma-separated GPU IDs (e.g., 0,1,2,3)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 解析 GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = list(range(args.num_gpus))
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # =========================================================================
    # 关键：渲染服务器不使用 Ray！
    # =========================================================================
    # Ray 会导致与训练进程冲突。我们使用原有的 ThreadPoolExecutor 方案。
    # =========================================================================
    
    logger.info("=" * 60)
    logger.info("  ActiveSpatial Render Server (Non-Ray Mode)")
    logger.info("=" * 60)
    logger.info("NOTE: This server does NOT use Ray to avoid conflicts")
    logger.info("      with training Ray cluster.")
    logger.info(f"GPU IDs: {gpu_ids}")
    logger.info(f"GS Root: {args.gs_root}")
    logger.info("=" * 60)
    
    # 使用原有的 service.py 中的 ActiveSpatialService
    from .service import ActiveSpatialService
    from .service_config import ActiveSpatialServiceConfig
    
    service_config = ActiveSpatialServiceConfig(
        max_workers=args.num_gpus * args.renderers_per_gpu,
        devices=gpu_ids,
        render_backend="local",
        gs_root=args.gs_root,
    )
    
    service = ActiveSpatialService(service_config)
    
    # 简单的 Flask 服务器
    from flask import Flask, request, jsonify
    import traceback
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'ok', 
            'mode': 'non-ray',
            'num_envs': len(service.environments),
            'gpu_ids': gpu_ids,
        })
    
    @app.route('/environments', methods=['POST'])
    def create_environments():
        try:
            data = request.json
            ids2configs = data.get('ids2configs', data)
            service.create_environments_batch(ids2configs)
            return jsonify({'success': True, 'created': list(ids2configs.keys()), 'status': 'ok'})
        except Exception as e:
            logger.error(f"Error creating environments: {e}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/batch/reset', methods=['POST'])
    def batch_reset():
        try:
            data = request.json
            ids2seeds = data.get('ids2seeds', data)
            results = service.reset_batch(ids2seeds)
            # 序列化观察
            serialized = {}
            for env_id, (obs, info) in results.items():
                serialized[env_id] = (obs, info)
            return jsonify({'results': serialized})
        except Exception as e:
            logger.error(f"Error in batch_reset: {e}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/batch/step', methods=['POST'])
    def batch_step():
        try:
            data = request.json
            ids2actions = data.get('ids2actions', data)
            results = service.step_batch(ids2actions)
            # 序列化结果
            serialized = {}
            for env_id, (obs, reward, done, info) in results.items():
                serialized[env_id] = (obs, reward, done, info)
            return jsonify({'results': serialized})
        except Exception as e:
            logger.error(f"Error in batch_step: {e}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/batch/close', methods=['POST'])
    def batch_close():
        try:
            data = request.json
            env_ids = data.get('env_ids', data)
            service.close_batch(env_ids)
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Error in batch_close: {e}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/batch/system_prompt', methods=['POST'])
    def batch_system_prompt():
        try:
            data = request.json
            env_ids = data.get('env_ids', [])
            results = service.get_system_prompts_batch(env_ids)
            return jsonify({'system_prompts': results})
        except Exception as e:
            logger.error(f"Error in batch_system_prompt: {e}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/batch/reward', methods=['POST'])
    def batch_reward():
        try:
            data = request.json
            env_ids = data.get('env_ids', [])
            results = service.compute_reward_batch(env_ids)
            return jsonify({'rewards': results})
        except Exception as e:
            logger.error(f"Error in batch_reward: {e}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
