# v12: MAX_TURNS 12 → 50（核心修复：给模型足够的动作预算）
#
# ============================================================
# v10/v11 崩溃根本原因
# ============================================================
# [根本问题] max_actions_per_step=1 + max_turns=12 = 每集仅 12 步动作
#   导航任务通常需要 20-50 步才能到达目标
#   → 模型从未收到 success_reward=1.0
#   → 唯一稳定的正奖励是 format_reward=0.01（输出 done）或 delta_score≈0
#
# v10 崩溃：输出 done（12步预算内永远不成功，干脆立即结束）
# v11 崩溃：循环 move_backward（n_trajectory=2 的对比优势误导）
#
# ============================================================
# v12 核心修复：MAX_TURNS=50（50步×1动作=50总动作，接近v9的12×5=60）
# ============================================================
# [关键变化]
#   MAX_TURNS:              ★ 12 → 50  → 模型现在有足够动作步数到达目标
#   MAX_TRAJECTORY_LENGTH:  ★ 16000 → 26000  → 支持 50 轮对话的 token 预算
#                            估算: 50轮 × ~500 tokens/轮 = 25,000 tokens
#
# [env 设定]（env_config_single_action.yaml 已有 max_episode_steps=50）
#   max_episode_steps=50 与 max_turns=50 对齐
#   → 50 turns × 1 action/turn = 最多 50 个环境步骤
#
# [其余全部继承 v11]
#   N_TRAJECTORY=2, TRAIN_BATCH_SIZE=12（12×2=24总轨迹）
#   PPO_MINI_BATCH_SIZE=12, MINI_BATCH_SIZE=8
#   所有稳定性超参不变
# ============================================================

EXPERIMENT_NAME="v12_max_turns50"
ENV_CONFIG="env_config_single_action_256.yaml"  # 256×256: max_turns=50时ViT不OOM
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === 继承 v11 稳定性参数 ===
ENTROPY_COEFF="0.008"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.015"
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.5

# === Critic：继承 v11 ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=40
CLIPRANGE_VALUE="0.8"

# === 梯度：继承 v11 ===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === ★ 核心修复：MAX_TURNS 12 → 50，扩大 MAX_TRAJECTORY_LENGTH ===
MAX_TURNS=50               # ★ 12 → 50：每集最多 50 步，让模型有机会到达目标
MAX_TRAJECTORY_LENGTH=26000 # ★ 16000 → 26000：50轮×~500token/轮≈25000，留余量
MAX_RESPONSE_LENGTH=512    # 不变
MAX_PROMPT_LENGTH=2048     # 不变
WINDOW_SIZE=1              # 由于会OOM，这里改成1

# === 继承 v11 的 N=2 设置（总轨迹数 12×2=24 不变）===
N_TRAJECTORY=2
TRAIN_BATCH_SIZE=12        # 12×2=24 总轨迹（与 v10 的 24×1 相同）
VAL_BATCH_SIZE=4

# === PPO mini-batch：不变（总轨迹数不变）===
PPO_MINI_BATCH_SIZE=12     # 24/12=2 mini-batches; 12/4=3 per GPU ✓
MINI_BATCH_SIZE=8          # 24/8=3 rollout mini-batches ✓

# === 训练参数 ===
SAVE_FREQ=100
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数：继承 v11 ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
