# Active Spatial 3D渲染设置指南

## 架构概述

ViewSuite 的 Gaussian Splatting 渲染采用 **客户端-服务器分离架构**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            渲染架构图                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   VAGEN 训练节点 (CPU/多GPU)              GPU 渲染服务器                    │
│   ┌─────────────────────────┐            ┌─────────────────────────┐        │
│   │                         │            │                         │        │
│   │  ┌─────────────────┐    │  WebSocket │  ┌─────────────────┐    │        │
│   │  │ ActiveSpatialEnv│    │ ─────────▶ │  │ GS Render Server│    │        │
│   │  │ (环境逻辑)      │    │ ◀───────── │  │ (FastAPI+Uvicorn│    │        │
│   │  └────────┬────────┘    │   图像数据  │  └────────┬────────┘    │        │
│   │           │             │            │           │             │        │
│   │           ▼             │            │           ▼             │        │
│   │  ┌─────────────────┐    │            │  ┌─────────────────┐    │        │
│   │  │ UnifiedRenderGS │    │            │  │ GaussianRenderer│    │        │
│   │  │ (client mode)   │    │            │  │ (gsplat+CUDA)   │    │        │
│   │  └─────────────────┘    │            │  └─────────────────┘    │        │
│   │                         │            │                         │        │
│   └─────────────────────────┘            └─────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 为什么需要服务器分离？

1. **GPU 资源管理**：Gaussian Splatting 渲染需要 GPU，训练也需要 GPU。分离可以更好地管理资源。
2. **多环境并发**：一个渲染服务器可以为多个并发的训练环境提供服务。
3. **缓存优化**：服务器可以缓存加载的 3D 场景，避免重复加载。
4. **灵活部署**：渲染服务器可以部署在专用 GPU 节点上。

## 设置步骤

### 步骤 1: 准备 3D 数据

Gaussian Splatting 数据格式：
```
/path/to/gs_data/
├── scene_001.ply          # 场景1的高斯点云
├── scene_002.ply          # 场景2的高斯点云
└── ...
```

每个 `.ply` 文件包含：
- 点位置 (positions)
- 球谐函数系数 (SH coefficients) 
- 缩放参数 (scales)
- 旋转四元数 (rotations)
- 不透明度 (opacities)

### 步骤 2: 启动渲染服务器

在 GPU 节点上启动渲染服务器：

```bash
# 方式1: 使用 ViewSuite 自带的服务
cd /path/to/ViewSuite
python -m view_suite.interiorGS.service.gs_render_service \
    --interiorgs_root=/path/to/gs_data \
    --num_shards=8 \
    --max_renderers_per_worker=8 \
    --host=0.0.0.0 \
    --port=8777

# 方式2: 使用 SLURM 作业
sbatch render_server.sbatch
```

**render_server.sbatch 示例：**
```bash
#!/bin/bash
#SBATCH --job-name=gs_render
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

source ~/.bashrc
conda activate viewsuite

python -m view_suite.interiorGS.service.gs_render_service \
    --interiorgs_root=/scratch/data/interior_gs \
    --num_shards=8 \
    --port=8777
```

### 步骤 3: 配置 VAGEN 环境

在训练配置中设置渲染后端：

```yaml
# env_config.yaml
env:
  env_name: active_spatial
  env_config:
    max_steps: 20
    render_backend: "client"  # 使用远程渲染服务器
    client_url: "ws://gpu-node:8777/render/interiorgs"  # 服务器地址
    client_origin: null
    gs_root: null  # client模式不需要本地路径
    image_width: 512
    image_height: 512
```

### 步骤 4: 准备训练数据

训练数据格式 (JSONL):
```json
{
  "scene_id": "scene_001",
  "question": "Navigate to see the red sofa from the front",
  "target_pose": [1.0, 2.0, 1.5, 0, 45, 0],
  "initial_pose": [0, 0, 1.5, 0, 0, 0],
  "answer": "I can see the red sofa from the front view",
  "camera_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
}
```

## 本地渲染模式（用于调试）

如果要在本地渲染（需要 GPU）：

```yaml
# env_config.yaml
env:
  env_name: active_spatial
  env_config:
    render_backend: "local"
    gs_root: "/path/to/gs_data"
    # client_url 不需要
```

**依赖安装：**
```bash
# 安装 gsplat（需要CUDA）
pip install gsplat

# 或从源码编译
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
pip install -e .
```

## 渲染流程详解

### 1. 环境初始化
```python
# ActiveSpatialEnv.reset()
async def reset(self, seed=None):
    # 设置场景
    self.renderer.set_scene(self.scene_id)
    
    # 初始化相机位姿
    self.current_pose = self.initial_pose.copy()
    
    # 渲染初始视图
    image = await self.renderer.render_image_from_cam_param(
        self.camera_intrinsics,
        self._pose_to_extrinsics(self.current_pose),
        self.image_width,
        self.image_height
    )
    return image, info
```

### 2. 执行动作
```python
# ActiveSpatialEnv.step()
async def step(self, action):
    # 解析动作
    action_type, params = parse_action(action)
    
    # 更新相机位姿
    if action_type == "move_forward":
        self.current_pose = self.manipulator.move_forward(
            self.current_pose, params["distance"]
        )
    
    # 重新渲染
    image = await self.renderer.render_image_from_cam_param(
        self.camera_intrinsics,
        self._pose_to_extrinsics(self.current_pose),
        self.image_width,
        self.image_height
    )
    
    # 计算奖励
    reward = self.compute_pose_reward(self.current_pose, self.target_pose)
    
    return image, reward, done, truncated, info
```

### 3. 相机参数说明

**相机位姿 (6-DoF)：**
```python
pose = [tx, ty, tz, rx, ry, rz]
# tx, ty, tz: 世界坐标系中的位置
# rx, ry, rz: 欧拉角（度），世界坐标系 Z 轴向上
```

**相机内参 (Intrinsics)：**
```python
K = [
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
]
# fx, fy: 焦距（像素）
# cx, cy: 主点（图像中心）
```

**相机外参 (Extrinsics)：**
```python
E = np.eye(4)  # 4x4 变换矩阵
E[:3, :3] = R  # 旋转矩阵
E[:3, 3] = t   # 平移向量
# 从世界坐标系到相机坐标系的变换
```

## 常见问题

### Q1: WebSocket 连接失败
```
确保：
1. 渲染服务器正在运行
2. 防火墙允许端口 8777
3. URL 格式正确：ws://host:port/render/interiorgs
```

### Q2: GPU 内存不足
```
减少 max_renderers_per_worker 或 num_shards
或使用更小的图像尺寸
```

### Q3: PLY 文件找不到
```
确保 gs_root 路径正确，且包含 {scene_id}.ply 文件
```

## 性能优化建议

1. **批量渲染**：使用 `render_tasks()` 批量渲染多个视图
2. **场景缓存**：服务器自动缓存加载的场景，首次加载后更快
3. **分片策略**：根据场景数量调整 `num_shards`
4. **图像大小**：训练时使用较小图像（256x256），评估时使用较大图像（512x512）
