# -*- coding: utf-8 -*-
"""
ActiveSpatial Ray Service
基于 Ray Actor 的环境服务，实现真正的多进程并行渲染

架构：
- ActiveSpatialEnvActor: 单个环境实例（Ray Actor）
- ActiveSpatialRayService: 管理所有环境的服务（Ray Actor）
"""
import ray
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
from PIL import Image

from .env import ActiveSpatialEnv
from .env_config import ActiveSpatialEnvConfig
from .ray_actor_pool import ActiveSpatialActorPool, GaussianRendererActor

logger = logging.getLogger(__name__)


def serialize_observation(obs: Dict) -> Dict:
    """序列化观察，将 PIL Image 转为 bytes"""
    result = {}
    for key, value in obs.items():
        if key == 'multi_modal_data':
            result[key] = {}
            for mm_key, mm_value in value.items():
                if isinstance(mm_value, list):
                    serialized_list = []
                    for item in mm_value:
                        if isinstance(item, Image.Image):
                            import io
                            buf = io.BytesIO()
                            item.save(buf, format='PNG')
                            serialized_list.append({
                                '_type': 'PIL.Image',
                                'bytes': buf.getvalue(),
                                'mode': item.mode,
                                'size': item.size,
                            })
                        else:
                            serialized_list.append(item)
                    result[key][mm_key] = serialized_list
                else:
                    result[key][mm_key] = mm_value
        else:
            result[key] = value
    return result


def deserialize_observation(obs: Dict) -> Dict:
    """反序列化观察"""
    result = {}
    for key, value in obs.items():
        if key == 'multi_modal_data':
            result[key] = {}
            for mm_key, mm_value in value.items():
                if isinstance(mm_value, list):
                    deserialized_list = []
                    for item in mm_value:
                        if isinstance(item, dict) and item.get('_type') == 'PIL.Image':
                            import io
                            buf = io.BytesIO(item['bytes'])
                            img = Image.open(buf)
                            deserialized_list.append(img)
                        else:
                            deserialized_list.append(item)
                    result[key][mm_key] = deserialized_list
                else:
                    result[key][mm_key] = mm_value
        else:
            result[key] = value
    return result


# =============================================================================
# Environment Actor
# =============================================================================

@ray.remote
class ActiveSpatialEnvActor:
    """
    单个 ActiveSpatial 环境 Actor
    
    特点：
    - 独立进程运行
    - 持有 Renderer 引用
    - 实现 reset/step/close 接口
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        renderer: GaussianRendererActor,
    ):
        """
        初始化环境 Actor
        
        Args:
            config: 环境配置字典
            renderer: 渲染器 Actor 引用
        """
        self.config = ActiveSpatialEnvConfig(**config)
        self.renderer = renderer
        
        # 创建环境实例，但禁用其内部渲染器
        self.config.render_backend = None  # 使用外部渲染器
        self.env = ActiveSpatialEnv(self.config)
        
        self._current_scene_id: Optional[str] = None
    
    def reset(self, seed: int = None) -> Tuple[Dict, Dict]:
        """
        重置环境
        
        Returns:
            (observation, info)
        """
        # 调用环境 reset（不渲染）
        self.env.reset(seed=seed)
        
        # 获取场景信息
        if self.env.current_item:
            scene_id = self.env.current_item.get('scene_id')
            if scene_id and scene_id != self._current_scene_id:
                # 切换场景
                ray.get(self.renderer.set_scene.remote(scene_id))
                self._current_scene_id = scene_id
        
        # 渲染初始帧
        obs = self._render_and_build_observation(init_obs=True)
        
        info = {
            "scene_id": self._current_scene_id,
            "object_label": self.env.current_item.get('object_label', ''),
            "preset": self.env.current_item.get('preset', ''),
        }
        
        return serialize_observation(obs), info
    
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 动作字符串（LLM 输出）
            
        Returns:
            (observation, reward, done, info)
        """
        # 执行动作（不渲染）
        _, reward, done, info = self.env.step(action)
        
        # 渲染当前帧
        obs = self._render_and_build_observation(init_obs=False)
        
        return serialize_observation(obs), reward, done, info
    
    def _render_and_build_observation(self, init_obs: bool = False) -> Dict:
        """渲染并构建观察"""
        # 获取相机参数
        camera_intrinsics = self.env.camera_intrinsics
        camera_extrinsics = self.env.view_engine.get_pose()
        
        # 调用渲染器
        img_array = ray.get(self.renderer.render.remote(
            camera_intrinsics,
            camera_extrinsics,
            self.config.render_width,
            self.config.render_height,
        ))
        
        # 转换为 PIL Image
        if isinstance(img_array, np.ndarray):
            img = Image.fromarray(img_array)
        else:
            img = img_array
        
        # 构建观察
        return self.env._build_observation_from_image(img, init_obs)
    
    def get_system_prompt(self) -> str:
        """获取系统提示"""
        return self.env.system_prompt()
    
    def compute_reward(self) -> float:
        """计算最终奖励"""
        return self.env.compute_reward()
    
    def close(self) -> None:
        """关闭环境"""
        self.env.close()


# =============================================================================
# Ray Service
# =============================================================================

@ray.remote
class ActiveSpatialRayService:
    """
    ActiveSpatial Ray 服务
    
    功能：
    - 管理多个 EnvActor 的生命周期
    - 通过 ActorPool 获取/释放渲染器
    - 实现批量 reset/step/close
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        actor_pool: ActiveSpatialActorPool,
    ):
        """
        初始化服务
        
        Args:
            config: 服务配置
            actor_pool: 渲染器资源池
        """
        self.config = config
        self.actor_pool = actor_pool
        
        # 环境追踪
        self.actors: Dict[str, ActiveSpatialEnvActor] = {}
        self.resources: Dict[str, Tuple[str, str, Any]] = {}  # env_id -> (rid, cache_key, renderer)
        
        # 获取资源前缀
        self.resource_prefix = ray.get(actor_pool.get_resource_prefix.remote())
        
        logger.info(f"[RayService] Initialized with resource prefix: {self.resource_prefix}")
    
    def create_environments_batch(
        self,
        ids2configs: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        批量创建环境
        
        Args:
            ids2configs: env_id -> 配置
        """
        if not ids2configs:
            return
        
        # 提取 cache_keys (scene_id)
        cache_keys = []
        for env_id, cfg in ids2configs.items():
            env_config = cfg.get('env_config', cfg)
            scene_id = env_config.get('scene_id', 'unknown')
            cache_keys.append(scene_id)
        
        # 从资源池获取渲染器
        sims = ray.get(self.actor_pool.acquire_many.remote(cache_keys))
        
        # 创建环境 Actors
        for (env_id, cfg), cache_key, (rid, renderer) in zip(
            ids2configs.items(),
            cache_keys,
            sims,
        ):
            env_config = cfg.get('env_config', cfg)
            
            # 创建 EnvActor
            actor = ActiveSpatialEnvActor.remote(env_config, renderer)
            
            self.actors[env_id] = actor
            self.resources[env_id] = (rid, cache_key, renderer)
        
        logger.info(f"[RayService] Created {len(ids2configs)} environments")
    
    def reset_batch(
        self,
        ids2seeds: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        批量重置环境
        
        返回 Ray futures（不阻塞）
        """
        futures = {}
        for env_id, seed in ids2seeds.items():
            if env_id in self.actors:
                futures[env_id] = self.actors[env_id].reset.remote(seed)
            else:
                logger.warning(f"[RayService] Environment {env_id} not found")
        
        return futures
    
    def step_batch(
        self,
        ids2actions: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        批量执行动作
        
        返回 Ray futures（不阻塞）
        """
        futures = {}
        for env_id, action in ids2actions.items():
            if env_id in self.actors:
                futures[env_id] = self.actors[env_id].step.remote(action)
            else:
                logger.warning(f"[RayService] Environment {env_id} not found")
        
        return futures
    
    def compute_reward_batch(
        self,
        env_ids: List[str],
    ) -> Dict[str, Any]:
        """批量计算奖励"""
        futures = {}
        for env_id in env_ids:
            if env_id in self.actors:
                futures[env_id] = self.actors[env_id].compute_reward.remote()
        return futures
    
    def close_batch(
        self,
        env_ids: Optional[List[str]] = None,
    ) -> None:
        """
        批量关闭环境，释放渲染器资源
        """
        if env_ids is None:
            env_ids = list(self.actors.keys())
        
        # 收集要释放的资源
        rids = []
        cache_keys = []
        
        for env_id in env_ids:
            if env_id in self.resources:
                rid, cache_key, _ = self.resources.pop(env_id)
                rids.append(rid)
                cache_keys.append(cache_key)
            
            if env_id in self.actors:
                # 关闭 Actor
                ray.get(self.actors[env_id].close.remote())
                del self.actors[env_id]
        
        # 释放渲染器回资源池
        if rids:
            ray.get(self.actor_pool.release_many.remote(rids, cache_keys))
        
        logger.info(f"[RayService] Closed {len(env_ids)} environments")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pool_stats = ray.get(self.actor_pool.get_stats.remote())
        return {
            "active_environments": len(self.actors),
            "pool_stats": pool_stats,
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_ray_service(
    config: Dict[str, Any],
    actor_pool: ActiveSpatialActorPool,
) -> ActiveSpatialRayService:
    """
    创建 Ray 服务
    
    Args:
        config: 服务配置
        actor_pool: 渲染器资源池
        
    Returns:
        Ray 服务 Actor
    """
    service = ActiveSpatialRayService.remote(config, actor_pool)
    return service
