# -*- coding: utf-8 -*-
"""
ActiveSpatial Ray Actor Pool
本地 GPU 渲染的资源池管理

核心功能：
1. 管理多个 GaussianRenderer Actor
2. 实现 cache_key (scene_id) 机制，避免重复加载场景
3. 每个 Actor 独占一个 GPU，实现真正的多进程并行
"""
import os
import ray
import asyncio
import uuid
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActorPoolConfig:
    """Actor Pool 配置"""
    # GPU 配置
    num_gpus: int = 4
    logical_to_physical: Optional[List[int]] = None  # 逻辑 GPU ID -> 物理 GPU ID
    gpu_plan: Optional[List[int]] = None  # 每个逻辑 GPU 上的 Renderer 数量
    
    # GS 渲染配置
    gs_root: str = ""  # Gaussian Splatting 数据根目录
    render_width: int = 512
    render_height: int = 512
    
    # 资源命名
    resource_prefix: str = "active_spatial"
    
    def __post_init__(self):
        if self.logical_to_physical is None:
            self.logical_to_physical = list(range(self.num_gpus))
        if self.gpu_plan is None:
            # 默认每个 GPU 上放 2 个 Renderer
            self.gpu_plan = [2] * self.num_gpus


# =============================================================================
# Renderer Actor (每个独占一个 GPU)
# =============================================================================

@ray.remote
class GaussianRendererActor:
    """
    单个 Gaussian Splatting 渲染器 Actor
    
    特点：
    - 手动指定 GPU（不依赖 Ray GPU 调度）
    - 缓存当前加载的场景
    - 场景切换时才重新加载
    
    重要：此 Actor 不使用 Ray 的 GPU 资源调度 (@ray.remote(num_gpus=1))
    而是通过 CUDA_VISIBLE_DEVICES 手动控制 GPU
    这样不会与训练进程的 Ray 调度冲突
    """
    
    VERBOSE = os.environ.get('GS_RENDERER_VERBOSE', '0') == '1'
    
    def __init__(self, gpu_id: int, gs_root: str, logical_gpu_id: int = 0):
        """
        初始化渲染器
        
        Args:
            gpu_id: 物理 GPU ID（用于 CUDA_VISIBLE_DEVICES）
            gs_root: GS 数据根目录
            logical_gpu_id: 逻辑 GPU ID（用于日志标识）
        """
        # 不使用 CUDA_VISIBLE_DEVICES 隔离，因为可能影响其他进程
        # 而是直接指定 gpu_device 参数
        self.physical_gpu_id = gpu_id
        self.logical_gpu_id = logical_gpu_id
        self.gs_root = gs_root
        self._current_scene_id: Optional[str] = None
        self._renderer = None
        self._is_initialized = False
        
        if self.VERBOSE:
            print(f"[GaussianRendererActor] Initialized for GPU {gpu_id} (logical: {logical_gpu_id})")
    
    def get_current_scene(self) -> Optional[str]:
        """获取当前加载的场景 ID"""
        return self._current_scene_id
    
    def set_scene(self, scene_id: str) -> bool:
        """
        设置/切换场景
        
        Args:
            scene_id: 场景 ID
            
        Returns:
            True if scene was changed, False if already loaded
        """
        if scene_id == self._current_scene_id and self._renderer is not None:
            if self.VERBOSE:
                print(f"[GaussianRendererActor] Scene {scene_id} already loaded, skipping")
            return False  # 已加载，跳过
        
        # 需要加载新场景
        try:
            from .render.gs_render_local import GaussianRenderer
            
            # 查找 PLY 文件
            ply_path = self._find_ply_path(scene_id)
            
            if self.VERBOSE:
                print(f"[GaussianRendererActor] Loading scene {scene_id} from {ply_path}")
            
            # 直接指定物理 GPU ID，不使用 CUDA_VISIBLE_DEVICES
            # 这样不会影响其他进程，也不依赖 Ray 的 GPU 调度
            self._renderer = GaussianRenderer(ply_path, gpu_device=self.physical_gpu_id)
            self._current_scene_id = scene_id
            self._is_initialized = True
            
            if self.VERBOSE:
                print(f"[GaussianRendererActor] Scene {scene_id} loaded successfully on GPU {self.physical_gpu_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"[GaussianRendererActor] Failed to load scene {scene_id}: {e}")
            raise
    
    def _find_ply_path(self, scene_id: str) -> str:
        """查找场景的 PLY 文件路径"""
        candidates = [
            f"{self.gs_root}/{scene_id}/3dgs_compressed.ply",
            f"{self.gs_root}/{scene_id}.ply",
            f"{self.gs_root}/{scene_id}/gaussian.ply",
            f"{self.gs_root}/{scene_id}/point_cloud.ply",
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"Could not find PLY file for scene {scene_id}. Tried: {candidates}"
        )
    
    def render(
        self,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        width: int = 512,
        height: int = 512,
    ) -> np.ndarray:
        """
        渲染当前视角
        
        Args:
            camera_intrinsics: 3x3 或 4x4 相机内参
            camera_extrinsics: 4x4 相机外参 (world-to-camera)
            width: 输出宽度
            height: 输出高度
            
        Returns:
            RGB 图像 (H, W, 3) uint8
        """
        if self._renderer is None:
            raise RuntimeError("Scene not loaded. Call set_scene() first.")
        
        # 调用 gsplat 渲染
        img = self._renderer.render_image_from_cam_param(
            camera_intrinsics,
            camera_extrinsics,
            width,
            height,
        )
        
        # 确保返回 numpy array
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        if isinstance(img, np.ndarray) and img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        return img
    
    def render_batch(
        self,
        camera_params_list: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        """
        批量渲染（同一场景，多个视角）
        
        Args:
            camera_params_list: List of dicts with 'intrinsics', 'extrinsics', 'width', 'height'
            
        Returns:
            List of RGB images
        """
        results = []
        for params in camera_params_list:
            img = self.render(
                params['intrinsics'],
                params['extrinsics'],
                params.get('width', 512),
                params.get('height', 512),
            )
            results.append(img)
        return results
    
    def close(self):
        """释放资源"""
        self._renderer = None
        self._current_scene_id = None
        self._is_initialized = False


# =============================================================================
# Actor Pool (资源池管理)
# =============================================================================

@ray.remote
class ActiveSpatialActorPool:
    """
    ActiveSpatial 渲染器资源池
    
    核心功能：
    1. 管理多个 GaussianRendererActor
    2. cache_key 机制：优先分配已加载目标场景的 Renderer
    3. 异步获取/释放资源
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化资源池
        
        Args:
            config: 配置字典，包含：
                - logical_to_physical: GPU 映射
                - gpu_plan: 每个 GPU 上的 Renderer 数量
                - gs_root: GS 数据根目录
                - resource_prefix: 资源命名前缀
        """
        self.logical_to_physical = config.get('logical_to_physical', [0, 1, 2, 3])
        self.gpu_plan = config.get('gpu_plan', [2, 2, 2, 2])
        self.gs_root = config.get('gs_root', '')
        self.resource_prefix = config.get('resource_prefix', 'active_spatial')
        
        # 异步条件变量
        self.condition = asyncio.Condition()
        
        # 资源追踪
        self.resources: Dict[str, Any] = {}  # rid -> RendererActor
        self.in_use: set = set()  # 正在使用的 resource IDs
        self.free_by_key: Dict[str, deque] = defaultdict(deque)  # cache_key -> deque of rids
        
        # 统计
        self.max_in_use = 0
        self.total_acquires = 0
        self.cache_hits = 0
        
        # 启动状态监控
        asyncio.get_event_loop().create_task(self._periodic_status())
    
    async def init_pool(self) -> int:
        """
        初始化渲染器池
        
        Returns:
            创建的渲染器数量
        """
        async with self.condition:
            initial_key = str(uuid.uuid4())  # 初始 cache_key
            
            for logical_gpu, count in enumerate(self.gpu_plan):
                physical_gpu = self.logical_to_physical[logical_gpu]
                
                for i in range(count):
                    # 创建 Ray Actor
                    # 注意：不使用 Ray 的 GPU 资源调度 (resources={})
                    # 而是通过 physical_gpu 参数让 Actor 内部直接访问指定 GPU
                    # 这样不会与训练进程的 Ray GPU 调度冲突
                    renderer = GaussianRendererActor.options(
                        lifetime="detached",
                        # 不声明任何资源，避免与 Ray 调度冲突
                    ).remote(physical_gpu, self.gs_root, logical_gpu)
                    
                    rid = f"{self.resource_prefix}_{logical_gpu}_{i}_{uuid.uuid4().hex[:8]}"
                    self.resources[rid] = renderer
                    self.free_by_key[initial_key].append(rid)
                    
                    logger.info(
                        f"[ActorPool] Created Renderer {rid} on GPU {physical_gpu}"
                    )
            
            self.condition.notify_all()
            total = len(self.resources)
            logger.info(f"[ActorPool] Initialized {total} renderers")
            return total
    
    async def acquire_many(self, cache_keys: List[str]) -> List[Tuple[str, Any]]:
        """
        批量获取渲染器
        
        优先匹配 cache_key（场景已加载），避免重复加载。
        
        Args:
            cache_keys: 每个请求的场景 ID (cache_key)
            
        Returns:
            List of (resource_id, renderer_actor) tuples
        """
        if not cache_keys:
            return []
        
        async with self.condition:
            while True:
                acquired: List[Tuple[Optional[str], Optional[str]]] = []
                local_cache_hits = 0
                
                # 第一轮：尝试精确匹配 cache_key
                for key in cache_keys:
                    queue = self.free_by_key.get(key, deque())
                    if queue:
                        rid = queue.popleft()
                        if not self.free_by_key[key]:
                            del self.free_by_key[key]
                        self.in_use.add(rid)
                        acquired.append((rid, key))
                        local_cache_hits += 1
                    else:
                        acquired.append((None, None))
                
                # 第二轮：Fallback - 使用任何可用资源
                for i, (rid, matched_key) in enumerate(acquired):
                    if rid is not None:
                        continue
                    
                    # 查找任何可用的资源
                    for alt_key in list(self.free_by_key.keys()):
                        queue = self.free_by_key[alt_key]
                        if queue:
                            rid = queue.popleft()
                            if not queue:
                                del self.free_by_key[alt_key]
                            self.in_use.add(rid)
                            acquired[i] = (rid, alt_key)
                            break
                
                # 检查是否全部满足
                if all(rid is not None for rid, _ in acquired):
                    self.max_in_use = max(self.max_in_use, len(self.in_use))
                    self.total_acquires += len(cache_keys)
                    self.cache_hits += local_cache_hits
                    
                    # 返回 (rid, renderer) 对
                    return [
                        (rid, self.resources[rid])
                        for rid, _ in acquired
                    ]
                
                # 资源不足，回滚并等待
                for rid, key in acquired:
                    if rid is not None:
                        self.in_use.remove(rid)
                        self.free_by_key[key].appendleft(rid)
                
                logger.debug("[ActorPool] Not enough resources, waiting...")
                await self.condition.wait()
    
    async def release_many(
        self,
        resource_ids: List[str],
        cache_keys: List[str],
    ) -> None:
        """
        释放资源，并用新的 cache_key 标记
        
        Args:
            resource_ids: 要释放的资源 ID 列表
            cache_keys: 每个资源当前加载的场景 ID
        """
        assert len(resource_ids) == len(cache_keys), \
            f"Length mismatch: {len(resource_ids)} vs {len(cache_keys)}"
        
        async with self.condition:
            for rid, key in zip(resource_ids, cache_keys):
                if rid in self.in_use:
                    self.in_use.remove(rid)
                    self.free_by_key[key].append(rid)
            
            self.condition.notify_all()
    
    def get_resource_prefix(self) -> str:
        """获取资源前缀"""
        return self.resource_prefix
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_resources": len(self.resources),
            "in_use": len(self.in_use),
            "free": len(self.resources) - len(self.in_use),
            "max_in_use": self.max_in_use,
            "total_acquires": self.total_acquires,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_acquires),
        }
    
    async def _periodic_status(self, interval: int = 30) -> None:
        """定期打印状态"""
        while True:
            await asyncio.sleep(interval)
            stats = self.get_stats()
            logger.info(
                f"[ActorPool] Total: {stats['total_resources']}, "
                f"In use: {stats['in_use']}, Free: {stats['free']}, "
                f"Cache hit rate: {stats['cache_hit_rate']:.2%}"
            )


# =============================================================================
# 工厂函数
# =============================================================================

def create_actor_pool(config: Dict[str, Any]) -> ActiveSpatialActorPool:
    """
    创建并初始化 Actor Pool
    
    Args:
        config: 配置字典
        
    Returns:
        初始化好的 ActorPool actor
    """
    pool = ActiveSpatialActorPool.remote(config)
    ray.get(pool.init_pool.remote())
    return pool
