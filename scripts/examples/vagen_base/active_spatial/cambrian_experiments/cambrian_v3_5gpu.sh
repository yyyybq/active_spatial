# cambrian_v3_5gpu: v3 bi-level GAE 的 5卡 H200 版本
# =============================================================================
# 启动命令 (从 VAGEN/ 目录):
#   cd /scratch/by2593/project/Active_Spatial/VAGEN && \
#   export CAMBRIAN_MODEL_PATH=/scratch/by2593/hf_cache/cambrian-s-7b && \
#   export CAMBRIAN_SRC=/scratch/by2593/project/Active_Spatial/cambrian-s && \
#   nohup bash scripts/examples/vagen_base/active_spatial/run_cambrian_experiment.sh \
#     cambrian_v3_5gpu.sh > cambrian_v3_5gpu.log 2>&1 &
#
# 硬件: 5x H200 (4 train + 1 render)
# 相比 v3 (6+1 GPU) 的改动:
#   - NUM_TRAIN_GPUS: 6 → 4
#   - RENDERING_GPU:  6 → 4  (第5张卡，index=4)
#   - TRAIN_BATCH_SIZE: 30 → 24  (保持每卡 6 samples: 24/4=6)
#   - MINI_BATCH_SIZE: 6 → 4     (必须整除 TRAIN_BATCH_SIZE: 24/4=6 mini-batches)
#   - VAL_BATCH_SIZE: 6 → 4
#   - 其余参数与 v3 完全一致
# =============================================================================
EXPERIMENT_NAME="cambrian_v3_5gpu"
ENV_CONFIG="env_config_v2.yaml"

# --- 5卡 GPU 配置 ---
NUM_TRAIN_GPUS=4               # 6→4  训练卡
RENDERING_GPU=4                # 6→4  渲染卡 (第5张)
TRAIN_BATCH_SIZE=24            # 30→24  保持每卡 6 samples
MINI_BATCH_SIZE=4              # 6→4   整除 24 得 6 个 mini-batch
VAL_BATCH_SIZE=4               # 6→4

# --- v3 核心改动: Advantage Estimator ---
ADV_ESTIMATOR="bi_level_gae"   # masked_gae → bi_level_gae
HIGH_LEVEL_GAMMA="0.95"        # turn 级别折扣

# --- v3 核心改动: 减少截断 ---
MAX_TRAJECTORY_LENGTH=24000    # 18000 → 24000  减少 left truncation

# --- 继承 v2 的所有稳定化参数 ---

# Critic 稳定
CRITIC_WARMUP=30
CRITIC_LR="5e-6"
CRITIC_GRAD_CLIP="10.0"

# KL 保护
USE_KL_LOSS="True"
KL_LOSS_COEF="0.02"
KL_COEF="0.01"

# Entropy 保护
ENTROPY_COEFF="0.01"

# 采样多样性
TEMPERATURE="0.8"

# Optimizer offload (防 OOM)
ACTOR_OPTIMIZER_OFFLOAD="True"
CRITIC_OPTIMIZER_OFFLOAD="True"

# 保存频率
SAVE_FREQ=50
TEST_FREQ=50
