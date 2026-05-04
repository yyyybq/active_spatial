# cambrian_v2: 防 Policy Collapse - Critic Warmup + KL 保护 + 梯度稳定
# =============================================================================
# v1 诊断:
#   - Critic score head 随机初始化 → vf_explained_var = -1881 (step 1)
#   - critic grad_norm 847-932，被 grad_clip=1.0 裁到 1/900，学不动
#   - 无 critic_warmup → actor 一开始就吃噪声 advantage
#   - use_kl_loss=False + kl_coef=0.001 → 策略无锚点，自由漂移
#   - Step 1-11 靠预训练惯性维持，OOM restart 后 policy 不可逆崩溃
#   - Step 30: entropy 0.5→3.5, valid 55%→0%, 纯乱码
#
# v2 改动:
#   1. critic_warmup=30       让 Critic 先学 30 步，建立合理 baseline
#   2. use_kl_loss=True       开启显式 KL 锚定，防止策略远离预训练分布
#      kl_loss_coef=0.02      足够强的 KL 拉力
#   3. kl_ctrl.kl_coef=0.01   外部 KL penalty 也加强 10x
#   4. critic_lr=5e-6         降低 critic LR (1e-5→5e-6)，防止过拟合噪声 reward
#   5. critic_grad_clip=10.0  放宽到 10 (初始 grad_norm 800+, clip=1 等于没学)
#   6. entropy_coeff=0.01     10x entropy bonus，抵抗 entropy 坍塌
#   7. temperature=0.8        轻微提高采样多样性
#   8. save_freq=20           更频繁保存，方便 A/B 对比中间状态
# =============================================================================
EXPERIMENT_NAME="cambrian_v2_stabilize"
ENV_CONFIG="env_config_v2.yaml"

# --- Critic 稳定 ---
CRITIC_WARMUP=30             # 0→30   先让 Critic 收敛再更新 Actor
CRITIC_LR="5e-6"             # 1e-5→5e-6  降低 LR 防过拟合
CRITIC_GRAD_CLIP="10.0"      # 1.0→10.0  放宽裁剪，让初始大梯度能学习

# --- KL 保护 (防 policy drift) ---
USE_KL_LOSS="True"           # False→True  开启显式 KL loss
KL_LOSS_COEF="0.02"          # 0.001→0.02  20x 加强
KL_COEF="0.01"               # 0.001→0.01  外部 penalty 也加强

# --- Entropy 保护 (防 entropy collapse) ---
ENTROPY_COEFF="0.01"         # 0.001→0.01  10x entropy bonus

# --- 采样多样性 ---
TEMPERATURE="0.8"            # 0.7→0.8

# --- 保存频率 ---
SAVE_FREQ=20                 # 50→20  更频繁保存，便于分析
TEST_FREQ=20                 # 50→20
