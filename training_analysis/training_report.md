# 训练分析报告

**项目**: vagen_active_spatial
**实验**: active_spatial_ppo_4gpu_warmer
**Run ID**: gynymtij
**状态**: running
**总训练步数**: 100

## 1. 核心性能指标

### Training Score
- 最新值: 0.9509
- 平均值: 0.3979
- 最大值: 1.0319
- 最小值: -0.3308
- 趋势: 上升 ↑ (前期均值: -0.0060, 后期均值: 0.8130)

### Success Rate
- 最新值: 0.0000
- 平均值: 0.0003

## 2. PPO算法指标

### Policy Gradient Loss
- 最新值: -0.193886
- 平均值: 0.009403

### Value Function Loss
- 最新值: 0.199604
- 平均值: 0.311114

### Entropy
- 最新值: 0.091533
- 平均值: 0.360187

### Gradient Norms
- actor: 均值=6.5113, 最大=64.2507
- critic: 均值=51.6611, 最大=492.2041

## 3. 序列长度统计
- Response Length (mean): 601.7 tokens
- Prompt Length (mean): 4357.9 tokens

## 4. 训练效率
- 每步平均时间: 693.31秒
- 预估总训练时间: 385.17小时 (2000 steps)
- 生成时间占比: 90.1%