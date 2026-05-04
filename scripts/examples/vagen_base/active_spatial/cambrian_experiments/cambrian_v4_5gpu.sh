# cambrian_v4_5gpu: v4 稳定性收紧版 (5x H200)
# =============================================================================
# 目标:
#   - 在 v3_5gpu 基础上做一档保守收紧，降低 OOM 与数值不稳定风险
# 主要改动:
#   - TRAIN_BATCH_SIZE: 24 -> 20
#   - MINI_BATCH_SIZE:  4  -> 2
#   - VAL_BATCH_SIZE:   4  -> 2
#   - CRITIC_LR:        5e-6 -> 3e-6
#   - CRITIC_WARMUP:    30   -> 60
#   - CRITIC_GRAD_CLIP: 10.0 -> 5.0
#   - MAX_TRAJECTORY_LENGTH: 24000 -> 22000
# 其余保持与 v3_5gpu 一致，便于做对照。
# =============================================================================

EXPERIMENT_NAME="cambrian_v4_5gpu"
ENV_CONFIG="env_config_v2.yaml"

# --- 5卡 GPU 配置 ---
NUM_TRAIN_GPUS=4
RENDERING_GPU=4
TRAIN_BATCH_SIZE=20
MINI_BATCH_SIZE=2
VAL_BATCH_SIZE=2

# --- Advantage Estimator ---
ADV_ESTIMATOR="bi_level_gae"
HIGH_LEVEL_GAMMA="0.95"

# --- 截断长度（轻微收紧） ---
MAX_TRAJECTORY_LENGTH=22000

# --- Critic 更保守 ---
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
