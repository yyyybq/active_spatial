#!/usr/bin/env python3
"""
训练曲线分析脚本 - 从wandb下载数据并生成分析报告
项目: vagen_active_spatial
实验: active_spatial_ppo_4gpu_warmer
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 配置
PROJECT_NAME = "vagen_active_spatial"
EXPERIMENT_NAME = "active_spatial_ppo_4gpu_warmer"
# 当前正在运行的run ID
RUN_ID = "gynymtij"
ENTITY = "nyu-visionx"
OUTPUT_DIR = Path("training_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# 设置中文字体（如果可用）
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def download_wandb_data():
    """从wandb下载训练数据"""
    print(f"连接wandb项目: {ENTITY}/{PROJECT_NAME}...")
    api = wandb.Api()
    
    # 直接使用指定的run ID
    run = api.run(f"{ENTITY}/{PROJECT_NAME}/{RUN_ID}")
    print(f"找到run: {run.name} (id: {run.id})")
    print(f"状态: {run.state}, 总步数: {run.summary.get('_step', 'N/A')}")
    
    # 下载历史数据
    print("下载训练历史数据...")
    history = run.history(samples=10000)  # 获取所有数据点
    
    # 保存为CSV
    csv_path = OUTPUT_DIR / "training_history.csv"
    history.to_csv(csv_path, index=False)
    print(f"数据已保存到: {csv_path}")
    
    return history, run

def analyze_core_metrics(df):
    """分析核心训练指标"""
    print("\n" + "="*60)
    print("核心训练指标分析")
    print("="*60)
    
    # 关键指标列表
    core_metrics = {
        'train_score': 'train/score/ActiveSpatialEnvConfig(render_backend=local,step_translation=0.2,step_rotation_deg=10.0,max_actions_per_step=5)',
        'train_success': 'train/success/ActiveSpatialEnvConfig(render_backend=local,step_translation=0.2,step_rotation_deg=10.0,max_actions_per_step=5)',
        'potential_score': 'train/potential_score/ActiveSpatialEnvConfig(render_backend=local,step_translation=0.2,step_rotation_deg=10.0,max_actions_per_step=5)',
    }
    
    # 简化列名查找
    for key, full_name in core_metrics.items():
        if full_name not in df.columns:
            # 尝试匹配部分名称
            matches = [c for c in df.columns if key.replace('train_', 'train/') in c.lower() or key in c.lower()]
            if matches:
                core_metrics[key] = matches[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Score曲线
    ax = axes[0, 0]
    score_cols = [c for c in df.columns if 'score' in c.lower() and 'potential' not in c.lower()]
    for col in score_cols[:3]:  # 最多显示3条
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label=col.split('/')[-1][:30], alpha=0.8)
    ax.set_title('Training Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Score')
    ax.legend(fontsize=8)
    
    # 2. Success Rate
    ax = axes[0, 1]
    success_cols = [c for c in df.columns if 'success' in c.lower()]
    for col in success_cols[:3]:
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label=col.split('/')[-1][:30], alpha=0.8)
    ax.set_title('Success Rate', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Success Rate')
    ax.legend(fontsize=8)
    
    # 3. Potential Score
    ax = axes[1, 0]
    potential_cols = [c for c in df.columns if 'potential' in c.lower()]
    for col in potential_cols[:3]:
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label=col.split('/')[-1][:30], alpha=0.8)
    ax.set_title('Potential Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Potential Score')
    ax.legend(fontsize=8)
    
    # 4. Steps per Episode
    ax = axes[1, 1]
    step_cols = [c for c in df.columns if 'train/step/' in c]
    for col in step_cols[:3]:
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label='steps_per_episode', alpha=0.8)
    ax.set_title('Steps per Episode', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Steps')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_core_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: 1_core_metrics.png")

def analyze_ppo_metrics(df):
    """分析PPO算法相关指标"""
    print("\n" + "="*60)
    print("PPO算法指标分析")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Actor Loss (Policy Gradient Loss)
    ax = axes[0, 0]
    pg_loss_cols = [c for c in df.columns if 'pg_loss' in c.lower()]
    for col in pg_loss_cols:
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label='pg_loss', alpha=0.8)
    ax.set_title('Policy Gradient Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    
    # 2. Entropy Loss
    ax = axes[0, 1]
    entropy_cols = [c for c in df.columns if 'entropy' in c.lower()]
    for col in entropy_cols:
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label='entropy_loss', alpha=0.8)
    ax.set_title('Entropy Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    
    # 3. Value Function Loss
    ax = axes[0, 2]
    vf_loss_cols = [c for c in df.columns if 'vf_loss' in c.lower()]
    for col in vf_loss_cols:
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label='vf_loss', alpha=0.8)
    ax.set_title('Value Function Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    
    # 4. KL Divergence
    ax = axes[1, 0]
    kl_cols = [c for c in df.columns if '/kl' in c.lower() and 'coeff' not in c.lower()]
    for col in kl_cols:
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label=col.replace('/', '_')[-20:], alpha=0.8)
    ax.set_title('KL Divergence', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    
    # 5. Gradient Norm
    ax = axes[1, 1]
    grad_cols = [c for c in df.columns if 'grad_norm' in c.lower()]
    for col in grad_cols:
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label=col.split('/')[0], alpha=0.8)
    ax.set_title('Gradient Norm', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    
    # 6. Clip Fraction
    ax = axes[1, 2]
    clip_cols = [c for c in df.columns if 'clipfrac' in c.lower()]
    for col in clip_cols:
        data = df[col].dropna()
        if len(data) > 0:
            ax.plot(data.index, data.values, label=col.split('/')[0], alpha=0.8)
    ax.set_title('Clip Fraction', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_ppo_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: 2_ppo_metrics.png")

def analyze_critic_metrics(df):
    """分析Critic相关指标"""
    print("\n" + "="*60)
    print("Critic指标分析")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = [
        ('critic/rewards/mean', 'Rewards Mean'),
        ('critic/advantages/mean', 'Advantages Mean'),
        ('critic/returns/mean', 'Returns Mean'),
        ('critic/values/mean', 'Values Mean'),
        ('critic/vf_explained_var', 'VF Explained Variance'),
        ('critic/vpred_mean', 'Value Prediction Mean'),
    ]
    
    for idx, (metric_pattern, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        cols = [c for c in df.columns if metric_pattern in c]
        if not cols:
            cols = [c for c in df.columns if metric_pattern.split('/')[-1] in c.lower()]
        
        for col in cols[:2]:
            data = df[col].dropna()
            if len(data) > 0:
                ax.plot(data.index, data.values, alpha=0.8)
                
                # 添加滑动平均
                if len(data) > 10:
                    window = min(20, len(data) // 5)
                    smoothed = data.rolling(window=window, min_periods=1).mean()
                    ax.plot(smoothed.index, smoothed.values, '--', alpha=0.6, label='MA')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_critic_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: 3_critic_metrics.png")

def analyze_sequence_length(df):
    """分析序列长度相关指标"""
    print("\n" + "="*60)
    print("序列长度分析")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Response Length
    ax = axes[0, 0]
    resp_cols = [c for c in df.columns if 'response_length' in c.lower()]
    for col in resp_cols:
        data = df[col].dropna()
        if len(data) > 0:
            label = 'mean' if 'mean' in col else ('max' if 'max' in col else 'min')
            ax.plot(data.index, data.values, label=label, alpha=0.8)
    ax.set_title('Response Length', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens')
    ax.legend()
    
    # 2. Prompt Length
    ax = axes[0, 1]
    prompt_cols = [c for c in df.columns if 'prompt_length' in c.lower()]
    for col in prompt_cols:
        data = df[col].dropna()
        if len(data) > 0:
            label = 'mean' if 'mean' in col else ('max' if 'max' in col else 'min')
            ax.plot(data.index, data.values, label=label, alpha=0.8)
    ax.set_title('Prompt Length', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens')
    ax.legend()
    
    # 3. Global Sequence Length
    ax = axes[1, 0]
    seqlen_cols = [c for c in df.columns if 'global_seqlen' in c.lower()]
    for col in seqlen_cols:
        data = df[col].dropna()
        if len(data) > 0:
            label = col.split('/')[-1]
            ax.plot(data.index, data.values, label=label, alpha=0.8)
    ax.set_title('Global Sequence Length', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens')
    ax.legend()
    
    # 4. Sequence Length Distribution (histogram of mean)
    ax = axes[1, 1]
    mean_col = [c for c in df.columns if 'global_seqlen/mean' in c]
    if mean_col:
        data = df[mean_col[0]].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.0f}')
    ax.set_title('Sequence Length Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_sequence_length.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: 4_sequence_length.png")

def analyze_timing(df):
    """分析训练时间相关指标"""
    print("\n" + "="*60)
    print("训练效率分析")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Step Timing
    ax = axes[0, 0]
    timing_cols = [c for c in df.columns if 'timing_s/' in c.lower()]
    key_timings = ['gen', 'update_actor', 'update_critic', 'step']
    for timing in key_timings:
        cols = [c for c in timing_cols if timing in c]
        for col in cols:
            data = df[col].dropna()
            if len(data) > 0:
                ax.plot(data.index, data.values, label=timing, alpha=0.8)
    ax.set_title('Training Time per Step (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Seconds')
    ax.legend()
    
    # 2. Per-token timing
    ax = axes[0, 1]
    token_timing = [c for c in df.columns if 'timing_per_token' in c.lower()]
    for col in token_timing:
        data = df[col].dropna()
        if len(data) > 0:
            label = col.split('/')[-1]
            ax.plot(data.index, data.values, label=label, alpha=0.8)
    ax.set_title('Time per Token (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('ms/token')
    ax.legend()
    
    # 3. MFU (Model FLOPS Utilization)
    ax = axes[1, 0]
    mfu_cols = [c for c in df.columns if 'mfu' in c.lower()]
    for col in mfu_cols:
        data = df[col].dropna()
        if len(data) > 0:
            label = col.split('/')[-1]
            ax.plot(data.index, data.values, label=label, alpha=0.8)
    ax.set_title('MFU (Model FLOPS Utilization)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('MFU')
    ax.legend()
    
    # 4. Time breakdown pie chart
    ax = axes[1, 1]
    time_components = {}
    for timing in ['gen', 'update_actor', 'update_critic', 'old_log_prob', 'ref', 'values', 'adv']:
        cols = [c for c in df.columns if f'timing_s/{timing}' in c]
        if cols:
            data = df[cols[0]].dropna()
            if len(data) > 0:
                time_components[timing] = data.mean()
    
    if time_components:
        labels = list(time_components.keys())
        sizes = list(time_components.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Average Time Breakdown', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_timing_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: 5_timing_analysis.png")

def generate_summary_report(df, run=None):
    """生成文字分析报告"""
    print("\n" + "="*60)
    print("生成分析报告")
    print("="*60)
    
    report = []
    report.append("# 训练分析报告")
    report.append(f"\n**项目**: {PROJECT_NAME}")
    report.append(f"**实验**: {EXPERIMENT_NAME}")
    if run:
        report.append(f"**Run ID**: {run.id}")
        report.append(f"**状态**: {run.state}")
    report.append(f"**总训练步数**: {len(df)}")
    report.append("")
    
    # 性能指标
    report.append("## 1. 核心性能指标")
    
    score_cols = [c for c in df.columns if 'score' in c.lower() and 'potential' not in c.lower() and 'ActiveSpatial' in c]
    if score_cols:
        score_data = df[score_cols[0]].dropna()
        if len(score_data) > 0:
            report.append(f"\n### Training Score")
            report.append(f"- 最新值: {score_data.iloc[-1]:.4f}")
            report.append(f"- 平均值: {score_data.mean():.4f}")
            report.append(f"- 最大值: {score_data.max():.4f}")
            report.append(f"- 最小值: {score_data.min():.4f}")
            # 趋势
            if len(score_data) > 10:
                early = score_data.iloc[:len(score_data)//3].mean()
                late = score_data.iloc[-len(score_data)//3:].mean()
                trend = "上升 ↑" if late > early else "下降 ↓" if late < early else "稳定 →"
                report.append(f"- 趋势: {trend} (前期均值: {early:.4f}, 后期均值: {late:.4f})")
    
    success_cols = [c for c in df.columns if 'success' in c.lower()]
    if success_cols:
        success_data = df[success_cols[0]].dropna()
        if len(success_data) > 0:
            report.append(f"\n### Success Rate")
            report.append(f"- 最新值: {success_data.iloc[-1]:.4f}")
            report.append(f"- 平均值: {success_data.mean():.4f}")
    
    # PPO指标
    report.append("\n## 2. PPO算法指标")
    
    ppo_metrics = {
        'pg_loss': 'Policy Gradient Loss',
        'vf_loss': 'Value Function Loss',
        'entropy': 'Entropy',
    }
    
    for key, name in ppo_metrics.items():
        cols = [c for c in df.columns if key in c.lower()]
        if cols:
            data = df[cols[0]].dropna()
            if len(data) > 0:
                report.append(f"\n### {name}")
                report.append(f"- 最新值: {data.iloc[-1]:.6f}")
                report.append(f"- 平均值: {data.mean():.6f}")
    
    # 梯度范数
    grad_cols = [c for c in df.columns if 'grad_norm' in c.lower()]
    if grad_cols:
        report.append("\n### Gradient Norms")
        for col in grad_cols:
            data = df[col].dropna()
            if len(data) > 0:
                component = 'actor' if 'actor' in col else 'critic'
                report.append(f"- {component}: 均值={data.mean():.4f}, 最大={data.max():.4f}")
    
    # 序列长度
    report.append("\n## 3. 序列长度统计")
    
    resp_mean = [c for c in df.columns if 'response_length/mean' in c]
    if resp_mean:
        data = df[resp_mean[0]].dropna()
        if len(data) > 0:
            report.append(f"- Response Length (mean): {data.mean():.1f} tokens")
    
    prompt_mean = [c for c in df.columns if 'prompt_length/mean' in c]
    if prompt_mean:
        data = df[prompt_mean[0]].dropna()
        if len(data) > 0:
            report.append(f"- Prompt Length (mean): {data.mean():.1f} tokens")
    
    # 训练效率
    report.append("\n## 4. 训练效率")
    
    step_time = [c for c in df.columns if 'timing_s/step' in c]
    if step_time:
        data = df[step_time[0]].dropna()
        if len(data) > 0:
            report.append(f"- 每步平均时间: {data.mean():.2f}秒")
            report.append(f"- 预估总训练时间: {(data.mean() * 2000 / 3600):.2f}小时 (2000 steps)")
    
    gen_time = [c for c in df.columns if 'timing_s/gen' in c]
    if gen_time:
        data = df[gen_time[0]].dropna()
        if len(data) > 0:
            report.append(f"- 生成时间占比: {(data.mean() / df[[c for c in df.columns if 'timing_s/step' in c][0]].dropna().mean() * 100):.1f}%")
    
    # 保存报告
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "training_report.md"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"报告已保存到: {report_path}")
    print("\n" + report_text)

def list_all_metrics(df):
    """列出所有可用指标"""
    print("\n" + "="*60)
    print("可用指标列表")
    print("="*60)
    
    metrics_path = OUTPUT_DIR / "available_metrics.txt"
    with open(metrics_path, 'w') as f:
        for col in sorted(df.columns):
            if not col.startswith('_'):
                f.write(f"{col}\n")
    print(f"指标列表已保存到: {metrics_path}")
    print(f"共 {len([c for c in df.columns if not c.startswith('_')])} 个指标")

def main():
    print("="*60)
    print("训练曲线分析工具")
    print("="*60)
    
    try:
        result = download_wandb_data()
        if result is None:
            print("\n无法获取数据，请检查wandb配置")
            return
        
        df, run = result
        print(f"\n成功获取 {len(df)} 条记录, {len(df.columns)} 个指标")
        
        # 列出所有指标
        list_all_metrics(df)
        
        # 生成分析图表
        analyze_core_metrics(df)
        analyze_ppo_metrics(df)
        analyze_critic_metrics(df)
        analyze_sequence_length(df)
        analyze_timing(df)
        
        # 生成文字报告
        generate_summary_report(df, run)
        
        print("\n" + "="*60)
        print("分析完成!")
        print(f"所有结果已保存到: {OUTPUT_DIR.absolute()}")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
