# cambrian_v1: Baseline - 等价于 run_cambrian_7gpu_v2.sh 的原始配置
# 结果: Step 1-11 正常 (valid=55-65%, entropy=0.3-0.5), Step 30+ 完全崩溃
#        (valid=0%, entropy=3.5, 纯乱码输出). Critic grad_norm 初始 847-932.
#        OOM crash at step ~11, resume 后 policy 已不可逆损坏.
# 根因: Critic 随机初始化 + 无 warmup + KL 约束太弱 → PPO policy collapse
EXPERIMENT_NAME="cambrian_v1_baseline"
ENV_CONFIG="env_config_v2.yaml"

# 所有值和 run_cambrian_7gpu_v2.sh 一致，无需 override
