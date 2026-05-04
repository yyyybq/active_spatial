# cambrian_v3: Bi-Level GAE + 减少截断
# =============================================================================
# v2 诊断 (step 41/2000):
#   - Policy 未 collapse ✓  输出连贯无乱码 ✓  action_is_valid 稳步上升 ✓
#   - 但 adv_estimator=masked_gae 完全忽略 high_level_gamma=0.95
#     → gamma=1.0 + lam=1.0 无折扣，所有 turn 等权回传，advantage 方差极大
#   - Left truncation 频繁: 12 rewards → 5 positions (58% reward 信号丢失)
#     → max_trajectory_length=18000 不够装 12 turn 含图像的完整 trajectory
#   - vf_explained_var 报告值 (-50~-130) 比实际差:
#     metric 用 response_mask 计算，包含大量 loss_mask=0 位置 (returns=0, values≠0)
#     critic 实际训练只在 loss_mask=1 位置，损失函数是正确的
#   - success=5.3%, potential_score=0.40，训练才开始，趋势向好
#
# v3 改动 (在 v2 基础上):
#   1. adv_estimator → bi_level_gae
#      - 两层 MDP: turn 级别用 high_level_gamma=0.95 折扣
#        token 级别用 gamma=1.0 不折扣
#      - 解决 masked_gae 忽略 high_level_gamma 的问题
#      - 远期 turn reward 被折扣，减少 advantage 方差
#   2. max_trajectory_length → 24000
#      - 减少 left truncation 频率，保留更多 reward 信号
#      - 18000→24000 (33% 增加)，OOM 时可回退到 20000
# =============================================================================
EXPERIMENT_NAME="cambrian_v3_bilevel"
ENV_CONFIG="env_config_v2.yaml"

# --- 核心改动: Advantage Estimator ---
ADV_ESTIMATOR="bi_level_gae"   # masked_gae → bi_level_gae
HIGH_LEVEL_GAMMA="0.95"        # turn 级别折扣 (v2 中被忽略，现在生效)

# --- 核心改动: 减少截断 ---
MAX_TRAJECTORY_LENGTH=24000    # 18000 → 24000  减少 left truncation

# --- 继承 v2 的所有稳定化参数 (无改动) ---

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

# 保存频率
SAVE_FREQ=20
TEST_FREQ=20
