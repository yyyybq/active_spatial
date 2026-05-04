# v11: N_TRAJECTORY=2（双轨迹 Dense Reward），基于 v10 改进
#
# ============================================================
# v10 问题分析
# ============================================================
# [问题1] N_TRAJECTORY=1 的局限性
#   每个 prompt 只看到 1 条轨迹 → critic 梯度来自单一路径
#   advantage 方差大，critic vf_explained_var 学习慢
#
# [问题2] 为何 v10 不直接开 N=2
#   N=2 + TRAIN_BATCH_SIZE=24 → 总 batch = 48 条轨迹
#   critic 内部 use_remove_padding=True：
#     per GPU mini-batch = 48/4/(24/12) = 6 sequences packed
#     packed length ≈ 6 × 14000 = 84K tokens
#     flash-attn memory ∝ L² → 84K² × 28 layers → OOM
#   即使 max_turns=12（已比 v9 的 max_turns=20 更安全），
#   TRAIN_BATCH_SIZE 不变时 packed length 依然翻倍
#
# ============================================================
# v11 核心设计：TRAIN_BATCH_SIZE / 2 → 总轨迹数不变
# ============================================================
# [核心变化] N_TRAJECTORY=2 + TRAIN_BATCH_SIZE=12
#   12 个 unique prompts × 2 轨迹 = 24 总轨迹
#   与 v10 的 24 × 1 = 24 总轨迹 完全相同：
#     - vLLM KV cache 峰值：由 MINI_BATCH_SIZE=8 控制，不变 ✓
#     - Critic per-GPU packed length：24/4/2 = 3 sequences，与 v10 一致 ✓
#     - Actor mini-batch per GPU：12/4=3，与 v10 一致 ✓
#     - 每步计算量（FLOPs）：相同 ✓
#
# [N=2 的收益]
#   同一 prompt 出现 2 条独立轨迹：
#   1. Critic 多样性：相同起点的不同路径 → critic 对 return 的估计更稳健
#      vf_explained_var 预计从 v10 的 ~0.1 提升至 ~0.2–0.35
#   2. Actor gradient 信号更丰富：2 条路径的优劣对比明确
#      advantage 的信噪比提升（同 prompt 内隐式对照）
#   3. 为未来 GRPO-style 组内归一化奠基：
#      切换 adv_estimator="grpo" 只需改 1 个参数
#
# [为何不用 use_dynamic_bsz + ppo_max_token_len_per_gpu]
#   dynamic_bsz 通过运行时按 token 数分配 micro-batch，理论上可支持 N=2
#   但 VAGEN 当前未经测试（seqlen balancing 与 multi-turn 轨迹的兼容性不明）
#   TRAIN_BATCH_SIZE/2 方案零代码改动，风险最低
#
# ============================================================
# v11 参数总结（相对 v10 的变化用 ★ 标注）
# ============================================================
#   N_TRAJECTORY:   ★ 1 → 2
#   TRAIN_BATCH_SIZE: ★ 24 → 12  (12×2=24 总轨迹，与 v10 相同)
#   VAL_BATCH_SIZE:  ★ 8 → 4    (整除 4)
#   PPO_MINI_BATCH_SIZE: 12  (24/12=2 mini-batches; 12/4=3 per GPU ✓ 不变)
#   MINI_BATCH_SIZE: 8       (24/8=3 rollout mini-batches ✓ 不变)
#   ENV_CONFIG:      env_config_single_action.yaml（继承 v10）
#   MAX_TURNS:       12      (继承 v10)
#   MAX_TRAJECTORY_LENGTH: 16000 (继承 v10)
#   MAX_RESPONSE_LENGTH: 512 (继承 v10 修复)
#   RESUME:          disable（从头训练；v10 仅 step 1-2 且 prompt 格式刚修复）
#   其余所有参数继承 v10
# ============================================================

EXPERIMENT_NAME="v11_n2_trajectory"
ENV_CONFIG="env_config_single_action.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === 继承 v10 稳定性参数 ===
ENTROPY_COEFF="0.008"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.015"
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.5

# === Critic：继承 v10 ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=40
CLIPRANGE_VALUE="0.8"

# === 梯度：继承 v10 ===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === 轨迹参数：继承 v10 ===
MAX_TURNS=12
WINDOW_SIZE=5
MAX_TRAJECTORY_LENGTH=16000
MAX_RESPONSE_LENGTH=512    # v10 已修复冷启动截断 bug，保留 512
MAX_PROMPT_LENGTH=2048

# === ★ 核心变化：N=2 + TRAIN_BATCH_SIZE/2 ===
N_TRAJECTORY=2             # ★ 1 → 2：每个 prompt 采样 2 条轨迹
TRAIN_BATCH_SIZE=12        # ★ 24 → 12：12×2=24 总轨迹，与 v10 的 24×1 相同
                           # 保证 critic packed length per GPU 不变 → 无 OOM
VAL_BATCH_SIZE=4           # ★ 8 → 4：整除 4 ✓

# === PPO mini-batch：不变（因为总轨迹数不变）===
PPO_MINI_BATCH_SIZE=12     # 24 总轨迹 / 12 = 2 mini-batches; 12/4=3 per GPU ✓
MINI_BATCH_SIZE=8          # 24 总轨迹 / 8 = 3 rollout mini-batches; 整除 4 ✓

# === 训练参数 ===
SAVE_FREQ=100
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数：继承 v10 ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
