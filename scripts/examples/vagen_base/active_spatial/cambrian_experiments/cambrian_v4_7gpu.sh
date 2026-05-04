# cambrian_v4_7gpu: v4 稳定性收紧版 (7x H200)
# =============================================================================
# 启动命令 (从 VAGEN/ 目录):
#   cd /scratch/by2593/project/Active_Spatial/VAGEN && \
#   export CAMBRIAN_MODEL_PATH=/scratch/by2593/hf_cache/cambrian-s-7b && \
#   export CAMBRIAN_SRC=/scratch/by2593/project/Active_Spatial/cambrian-s && \
#   nohup bash scripts/examples/vagen_base/active_spatial/run_cambrian_experiment.sh \
#     cambrian_v4_7gpu.sh > cambrian_v4_7gpu.log 2>&1 &
#
# 硬件: 7x H200 (6 train + 1 render)
# =============================================================================
# 目标:
#   - 在 v4_5gpu 基础上扩展到 7 卡 (6 train + 1 render)，同步增大 batch size
# 主要改动 (相比 v4_5gpu):
#   - NUM_TRAIN_GPUS:   4  -> 6
#   - RENDERING_GPU:    4  -> 6   (第7张卡，index=6)
#   - TRAIN_BATCH_SIZE: 20 -> 30  (保持每卡 5 samples: 30/6=5)
#   - MINI_BATCH_SIZE:  2  -> 3   (保持 10 个 mini-batch: 30/3=10)
#   - VAL_BATCH_SIZE:   2  -> 3
# 其余参数与 v4_5gpu 完全一致。
# =============================================================================

EXPERIMENT_NAME="cambrian_v4_7gpu"
ENV_CONFIG="env_config_v2.yaml"

# --- 7卡 GPU 配置 ---
NUM_TRAIN_GPUS=6
RENDERING_GPU=6
TRAIN_BATCH_SIZE=30
MINI_BATCH_SIZE=3
VAL_BATCH_SIZE=3

# --- Advantage Estimator ---
ADV_ESTIMATOR="bi_level_gae"
HIGH_LEVEL_GAMMA="0.95"

# --- 截断长度 ---
MAX_TRAJECTORY_LENGTH=22000

# --- Critic 保守配置 ---
CRITIC_WARMUP=60
CRITIC_LR="3e-6"
CRITIC_GRAD_CLIP="5.0"

# --- KL 保护 ---
USE_KL_LOSS="True"
KL_LOSS_COEF="0.02"
KL_COEF="0.01"

# --- Entropy 保护 ---
ENTROPY_COEFF="0.01"

# --- 采样多样性 ---
TEMPERATURE="0.8"

# --- Optimizer offload (防 OOM) ---
ACTOR_OPTIMIZER_OFFLOAD="True"
CRITIC_OPTIMIZER_OFFLOAD="True"

# --- 保存频率 ---
SAVE_FREQ=50
TEST_FREQ=50
