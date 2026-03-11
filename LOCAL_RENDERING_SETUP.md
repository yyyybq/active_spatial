# 🎨 本地 Gaussian Splatting 渲染设置指南

## 📋 前提条件

- ✅ CUDA GPU (用于 gsplat 渲染)
- ✅ Gaussian Splatting PLY 文件 (.ply 格式)
- ✅ Python 环境已激活 (vagen)

---

## 🚀 快速开始

### 步骤 1: 安装渲染依赖

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN
bash scripts/examples/vagen_base/active_spatial/install_render_deps.sh
```

这会安装：
- **gsplat**: GPU 加速的 Gaussian Splatting 渲染库
- **ply_gaussian_loader**: PLY 文件加载器（从 ViewSuite 复制）

### 步骤 2: 验证 PLY 文件路径

检查你的 Gaussian Splatting 文件结构：

```bash
# 应该看到类似这样的结构：
ls -la /scratch/by2593/project/Active_Spatial/InteriorGS/
# 0001_839920/3dgs_compressed.ply
# 0002_839955/3dgs_compressed.ply
# ...
```

### 步骤 3: 配置环境

`env_config.yaml` 已经配置为使用本地渲染：

```yaml
env1:
    env_name: active_spatial
    env_config:
        jsonl_path: "/scratch/by2593/project/Active_Spatial/VAGEN/data/interior_data_gen/interior_train_data.jsonl"
        
        # ✓ 本地 GPU 渲染
        render_backend: local
        gs_root: "/scratch/by2593/project/Active_Spatial/InteriorGS"
        image_width: 512
        image_height: 512
```

### 步骤 4: 启动训练

```bash
bash scripts/examples/vagen_base/active_spatial/run.sh
```

---

## 🔍 渲染模式说明

### 1. **本地渲染 (render_backend: local)** ⭐ 当前配置

**优点**：
- ✅ 不需要额外的渲染服务器
- ✅ 简单直接，适合单机训练
- ✅ 低延迟

**缺点**：
- ⚠️ 每个训练进程都会加载完整的 GS 模型到 GPU
- ⚠️ GPU 内存开销大
- ⚠️ 不适合多 GPU 分布式训练

**适用场景**：
- 单 GPU 训练或调试
- 小规模数据集
- 不需要高并发渲染

**配置示例**：
```yaml
render_backend: local
gs_root: "/path/to/gaussian_ply_files"  # 包含 {scene_id}/3dgs_compressed.ply
```

---

### 2. **远程渲染 (render_backend: client)**

**优点**：
- ✅ 将渲染负载转移到专用 GPU 服务器
- ✅ 训练进程不占用 GPU 内存
- ✅ 适合多 GPU 分布式训练
- ✅ 可以多个训练任务共享一个渲染服务

**缺点**：
- ⚠️ 需要启动独立的渲染服务器
- ⚠️ 网络延迟（WebSocket 通信）
- ⚠️ 配置稍复杂

**适用场景**：
- 多 GPU 分布式训练
- 大规模训练任务
- GPU 内存受限

**配置步骤**：

#### 2.1 启动渲染服务器

在一个独立的终端/节点上：

```bash
# 激活环境
conda activate vagen

# 启动渲染服务（占用 1 个 GPU）
cd /scratch/by2593/project/Active_Spatial/ViewSuite
python -m view_suite.interiorGS.service.gs_render_service \
    --host 0.0.0.0 \
    --port 8777 \
    --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS
```

#### 2.2 修改训练配置

```yaml
render_backend: client
client_url: "ws://localhost:8777/render/interiorgs"  # 如果在同一节点
# client_url: "ws://gpu-node-hostname:8777/render/interiorgs"  # 如果在不同节点
```

#### 2.3 启动训练

```bash
bash scripts/examples/vagen_base/active_spatial/run.sh
```

---

### 3. **预渲染图像 (render_backend: null)**

**优点**：
- ✅ 无渲染开销
- ✅ 最快的训练速度
- ✅ 不需要 GPU 渲染

**缺点**：
- ⚠️ 需要预先渲染所有图像
- ⚠️ 存储开销大
- ⚠️ 灵活性差（固定视角）

**适用场景**：
- 固定视角的训练任务
- 渲染计算受限

**配置示例**：
```yaml
render_backend: null
dataset_root: "/path/to/images"  # 包含预渲染的图像
# JSONL 中需要包含 "image_path" 字段
```

---

## 🧪 测试渲染

测试本地渲染是否工作：

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN

python -c "
from vagen.env.active_spatial.render.unified_renderer import UnifiedRenderGS
import asyncio
import numpy as np

async def test():
    renderer = UnifiedRenderGS(
        render_backend='local',
        gs_root='/scratch/by2593/project/Active_Spatial/InteriorGS',
        scene_id='0001_839920'
    )
    
    # 测试相机参数
    K = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]], dtype=np.float32)
    E = np.eye(4, dtype=np.float32)
    E[2, 3] = 2.0  # Z=2 米
    
    img = await renderer.render_image_from_cam_param(K, E, 512, 512)
    img.save('test_render.png')
    print('✓ 渲染成功! 输出: test_render.png')

asyncio.run(test())
"
```

---

## ⚠️ 常见问题

### 问题 1: ModuleNotFoundError: No module named 'gsplat'

**解决**：
```bash
pip install gsplat
```

### 问题 2: ModuleNotFoundError: No module named 'ply_gaussian_loader'

**解决**：
```bash
bash scripts/examples/vagen_base/active_spatial/install_render_deps.sh
```

### 问题 3: FileNotFoundError: Could not find PLY file

**解决**：
检查 PLY 文件路径和 scene_id 是否匹配：
```bash
# 检查 JSONL 中的 scene_id
head -1 data/interior_data_gen/interior_train_data.jsonl | python -m json.tool | grep scene_id

# 检查对应的 PLY 文件是否存在
ls -la /scratch/by2593/project/Active_Spatial/InteriorGS/{scene_id}/3dgs_compressed.ply
```

### 问题 4: CUDA out of memory

**原因**: 本地渲染会为每个场景加载 GS 模型到 GPU

**解决方案**：
1. 减少训练 batch size
2. 使用远程渲染模式 (render_backend: client)
3. 增加 GPU offloading

---

## 📊 性能对比

| 渲染模式 | GPU 内存 | 训练速度 | 适用场景 |
|---------|---------|---------|---------|
| local | 高 | 快 | 单 GPU |
| client | 低 | 中等 | 多 GPU |
| null | 无 | 最快 | 预渲染 |

---

## 🎯 推荐配置

### 单 GPU 训练 (当前)
```yaml
render_backend: local
gs_root: "/scratch/by2593/project/Active_Spatial/InteriorGS"
```

### 多 GPU 训练 (4+ GPUs)
```yaml
render_backend: client
client_url: "ws://localhost:8777/render/interiorgs"
```
+ 在单独的 GPU 上运行渲染服务

---

## ✅ 总结

1. ✅ 安装依赖: `bash install_render_deps.sh`
2. ✅ 验证 PLY 文件路径
3. ✅ 配置已设置为本地渲染
4. ✅ 运行训练: `bash run.sh`

训练开始时会看到：
```
[GaussianRenderer] Loading model from /path/to/scene.ply
[GaussianRenderer] Loaded 123456 Gaussians on cuda
```

祝训练顺利！🚀
