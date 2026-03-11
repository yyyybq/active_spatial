# Active Spatial 训练快速启动指南 🚀

## 前置条件检查 ✅

已完成项目：
- ✅ 所有环境代码实现完成
- ✅ 通过 10/10 单元测试
- ✅ 依赖问题已修复
- ✅ 配置文件准备就绪

---

## 训练模式选择

Active Spatial 支持两种渲染模式，请根据你的场景选择：

| 模式 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **模式 A: 本地渲染** | 单机、小批量 | 简单，无需额外启动 | 串行渲染，速度较慢 |
| **模式 B: Ray 渲染服务** | 多 GPU、大批量 | 并行渲染，7x 加速 | 需要先启动服务器 |

---

## 模式 A: 本地渲染（简单模式）

### 直接运行训练

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN

# 一条命令启动训练
bash scripts/examples/vagen_base/active_spatial/run.sh
```

**当前配置**（`run.sh` 默认）：
- `rollout_manager.use_service=False` - 不使用服务
- `render_backend: local` - 本地渲染
- 每个 step 串行渲染，batch_size=16 时约需 85 秒

---

## 模式 B: Ray 渲染服务（推荐，高性能）

### 架构说明

```
┌─────────────────────────────────────────────────────────────────┐
│  终端 1: Ray 渲染服务器                                          │
│  (独立 Ray 集群 @ port 6380, 不与训练冲突)                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Flask Server (port 5001)                                  │  │
│  │    └── ActiveSpatialRayService                            │  │
│  │          └── GaussianRendererActor × 8 (4 GPU × 2)        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                ↓ HTTP
┌─────────────────────────────────────────────────────────────────┐
│  终端 2: 训练进程                                                │
│  (默认 Ray 集群 @ port 6379)                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  VeRL PPO Trainer                                          │  │
│  │    └── BatchEnvClient → http://localhost:5001              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 步骤 1: 启动 Ray 渲染服务器（终端 1）

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN

# 启动渲染服务器（独立 Ray 集群，不影响训练）
./vagen/env/active_spatial/start_ray_server.sh \
    --gs-root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --num-gpus 4 \
    --renderers-per-gpu 2 \
    --port 5001 \
    --ray-port 6380

# 预期输出:
# ============================================
#   ActiveSpatial Ray Server
# ============================================
#   Port:              5001
#   Num GPUs:          4
#   GPU IDs:           0,1,2,3
#   Renderers/GPU:     2
#   Ray Port:          6380 (isolated cluster)
# ============================================
#   NOTE: This server uses an ISOLATED Ray cluster
#         It will NOT interfere with training Ray
# ============================================
# Starting ActiveSpatial Ray Server...
#  * Running on http://0.0.0.0:5001
```

**验证服务器运行**:
```bash
curl http://localhost:5001/health
# 应返回: {"status": "ok", ...}
```

### 步骤 2: 修改训练配置

修改 `run.sh` 中的参数（或创建新的脚本）：

```bash
# 将这两行:
rollout_manager.use_service=False \
rollout_manager.timeout=600 \

# 改为:
rollout_manager.use_service=True \
rollout_manager.base_url="http://localhost:5001" \
rollout_manager.timeout=600 \
```

同时修改 `env_config.yaml`：

```yaml
env1:
    env_name: active_spatial
    env_config:
        # 渲染模式改为 null（由服务器处理）
        render_backend: null
        # 其他配置保持不变...
```

### 步骤 3: 启动训练（终端 2）

```bash
cd /scratch/by2593/project/Active_Spatial/VAGEN

# 确保渲染服务器已运行后，再启动训练
bash scripts/examples/vagen_base/active_spatial/run.sh
```

### 性能对比

| 场景 | 模式 A (本地) | 模式 B (Ray 服务) | 提升 |
|------|---------------|-------------------|------|
| 16 不同场景 step | ~85s | ~12s | **7x** |
| 16 相同场景 step | ~85s | ~3s | **28x** |

---

## 步骤详解（通用）

### 步骤 1: 准备训练数据

**数据格式要求（JSONL）**：

```json
{
  "scene_id": "scene_001",
  "object_label": "chair",
  "preset": "front",
  "distance": 2.0,
  "init_camera": {
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsics": [[...], [...], [...], [...]]
  },
  "target_position": [x, y, z],
  "camera_params": {
    "forward": [dx, dy, dz]
  }
}
```

当前配置使用的数据文件：
```
/scratch/by2593/project/Active_Spatial/VAGEN/data/interior_data_gen/interior_train_data_tiny.jsonl
```

### 步骤 2: 创建训练数据集

`run.sh` 会自动执行此步骤：

```bash
python -m vagen.env.create_dataset \
    --yaml_path scripts/examples/vagen_base/active_spatial/env_config.yaml \
    --train_path data/active_spatial_ppo/train.parquet \
    --test_path data/active_spatial_ppo/test.parquet
```

### 步骤 3: 检查 GPU 和资源

```bash
# 检查 GPU 可用性
nvidia-smi

# 检查磁盘空间
df -h /scratch/by2593

# 检查 Python 环境
conda activate vagen
python --version  # 应该是 3.10+
```

### 步骤 4: 启动训练

参见上面的 **模式 A** 或 **模式 B**。

---

## 完整启动流程（模式 B 推荐）

```bash
# ========================================
# 终端 1: 启动渲染服务器
# ========================================
cd /scratch/by2593/project/Active_Spatial/VAGEN

./vagen/env/active_spatial/start_ray_server.sh \
    --gs-root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --num-gpus 4 \
    --port 5001

# 保持运行，不要关闭

# ========================================
# 终端 2: 启动训练
# ========================================
cd /scratch/by2593/project/Active_Spatial/VAGEN

# 激活环境
conda activate vagen

# 确保已修改 use_service=True，然后启动训练
bash scripts/examples/vagen_base/active_spatial/run.sh
```

---

## SLURM 集群提交

如果在 SLURM 集群上运行，创建两个任务：

### 任务 1: 渲染服务器

创建 `submit_render_server.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=render_server
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/render_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vagen

cd /scratch/by2593/project/Active_Spatial/VAGEN

./vagen/env/active_spatial/start_ray_server.sh \
    --gs-root /scratch/by2593/project/Active_Spatial/InteriorGS \
    --num-gpus 4 \
    --port 5001
```

### 任务 2: 训练进程

创建 `submit_training.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=vagen_training
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --dependency=afterok:${RENDER_JOB_ID}  # 等待渲染服务器启动

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vagen

cd /scratch/by2593/project/Active_Spatial/VAGEN

# 等待渲染服务器就绪
echo "Waiting for render server..."
until curl -s http://localhost:5001/health > /dev/null; do
    sleep 5
done
echo "Render server is ready!"

bash scripts/examples/vagen_base/active_spatial/run.sh
```

**提交顺序**:
```bash
mkdir -p logs

# 先提交渲染服务器
RENDER_JOB=$(sbatch --parsable submit_render_server.sbatch)
echo "Render server job: $RENDER_JOB"

# 再提交训练任务（依赖渲染服务器）
sbatch --dependency=afterok:$RENDER_JOB submit_training.sbatch
```

---

## 监控训练

### 查看日志

```bash
# 实时查看训练日志
tail -f active_spatial_ppo.log

# 查看渲染服务器日志
tail -f logs/render_*.out
```

### 关键指标

| 指标 | 说明 | 期望趋势 |
|------|------|----------|
| `success_rate` | 成功到达目标率 | ↑ 上升 |
| `mean_reward` | 平均奖励 | ↑ 上升 |
| `format_correct_rate` | 格式正确率 | 保持 > 0.8 |
| `action_validity_rate` | 动作有效率 | ↑ 上升 |
| `mean_episode_length` | 平均步数 | ↓ 下降 |

---

## 常见问题排查 🔧

### 问题 1: 渲染服务器连接失败

**错误**: `Connection refused: http://localhost:5001`

**解决**:
```bash
# 检查服务器是否运行
curl http://localhost:5001/health

# 如果没有，启动服务器
./vagen/env/active_spatial/start_ray_server.sh --gs-root /path/to/data
```

### 问题 2: Ray 集群冲突

**错误**: `Ray cluster already running on port 6379`

**解决**:
```bash
# 渲染服务器使用独立端口 6380，不应冲突
# 如果仍有问题，手动指定端口:
./start_ray_server.sh --ray-port 6381 ...
```

### 问题 3: CUDA 内存不足

**错误**: `OutOfMemoryError: CUDA out of memory`

**解决**:
```yaml
# 减小 batch size
data.train_batch_size=8  # 从 16 减到 8

# 减少每 GPU 渲染器数量
./start_ray_server.sh --renderers-per-gpu 1 ...

# 减少 GPU 内存利用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.05
```

### 问题 4: 找不到场景文件

**错误**: `FileNotFoundError: Could not find PLY file for scene_xxx`

**解决**:
```bash
# 检查 GS 数据目录
ls /scratch/by2593/project/Active_Spatial/InteriorGS/

# 确保场景文件存在:
# {gs_root}/{scene_id}/3dgs_compressed.ply
# 或 {gs_root}/{scene_id}.ply
```

### 问题 5: 训练不收敛

**症状**: reward 不上升或震荡

**解决**:
```yaml
# 调整学习率
actor_rollout_ref.actor.optim.lr=5e-7

# 调整奖励权重
format_reward: 0.1
success_reward: 2.0
```

---

## 训练完成后

### 检查点位置

```
outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/
```

### 停止渲染服务器

```bash
# 如果不再需要，可以停止服务器
# Ctrl+C 在终端 1
# 或:
ray stop --address=localhost:6380
```

---

## 配置参考

### env_config.yaml 关键字段

```yaml
env1:
    env_name: active_spatial
    env_config:
        jsonl_path: "..."           # 训练数据路径
        render_backend: local|null  # local=本地渲染, null=使用服务
        gs_root: "..."              # GS 数据目录
        image_width: 512
        image_height: 512
        step_translation: 0.1       # 移动步长（米）
        step_rotation_deg: 5.0      # 旋转步长（度）
        format_reward: 0.2          # 格式奖励
        success_reward: 1.0         # 成功奖励
        max_episode_steps: 50       # 最大步数
    train_size: 40
    test_size: 20
```

### run.sh 关键参数

```bash
# 数据
data.train_batch_size=16
data.max_prompt_length=2048
data.max_response_length=512

# 模型
actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct
actor_rollout_ref.actor.optim.lr=1e-6

# 渲染服务（模式 B）
rollout_manager.use_service=True
rollout_manager.base_url="http://localhost:5001"

# 训练
trainer.total_training_steps=500
trainer.save_freq=100
```

---

**祝训练顺利！** 🚀
