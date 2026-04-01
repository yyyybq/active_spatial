#!/usr/bin/env python3
"""
Wandb 训练曲线获取与分析脚本

用法:
  # 分析指定 run
  python scripts/analyze_wandb_run.py --run_id yr5xf955

  # 自定义项目
  python scripts/analyze_wandb_run.py --run_id yr5xf955 --project nyu-visionx/vagen_active_spatial

  # 列出项目中所有 run
  python scripts/analyze_wandb_run.py --list

  # 导出 CSV
  python scripts/analyze_wandb_run.py --run_id yr5xf955 --export csv

  # 保存图表
  python scripts/analyze_wandb_run.py --run_id yr5xf955 --save_plots
"""

import argparse
import sys
import os

import wandb
import pandas as pd


DEFAULT_PROJECT = "nyu-visionx/vagen_active_spatial"

# Active Spatial 环境配置字符串（用于过滤列名）
ENV_CONFIG_PATTERN = "ActiveSpatialEnvConfig"


def get_env_suffix(columns):
    """自动检测环境配置后缀"""
    for c in columns:
        if ENV_CONFIG_PATTERN in c:
            start = c.index(ENV_CONFIG_PATTERN)
            return c[start:]
    return None


def list_runs(project):
    """列出项目中所有 run"""
    api = wandb.Api()
    runs = api.runs(project, order="-created_at", per_page=50)
    print(f"{'Run ID':>12} | {'Name':>50} | {'State':>10} | {'Steps':>6} | {'Score':>8}")
    print("-" * 100)
    for run in runs:
        steps = run.summary.get("_step", "N/A")
        score = run.summary.get("critic/score/mean", None)
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"{run.id:>12} | {run.name:>50} | {run.state:>10} | {str(steps):>6} | {score_str:>8}")


def fetch_history(run_id, project):
    """从 wandb 获取训练历史"""
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    print(f"Run: {run.name} (ID: {run.id})")
    print(f"State: {run.state}")
    print(f"Created: {run.created_at}")
    runtime = run.summary.get("_runtime", 0)
    print(f"Runtime: {runtime / 3600:.1f}h ({runtime:.0f}s)")
    print()

    history = run.history(samples=2000)
    return run, history


def build_metrics_df(history):
    """从 history 中提取关键指标，构建 DataFrame"""
    env_suffix = get_env_suffix(history.columns)
    if env_suffix is None:
        print("WARNING: 未检测到 ActiveSpatialEnvConfig 列，仅提取通用指标")

    metric_map = {
        "_step": "step",
        # critic & actor
        "critic/score/mean": "score_mean",
        "critic/score/max": "score_max",
        "critic/score/min": "score_min",
        "critic/vf_loss": "vf_loss",
        "critic/vf_explained_var": "vf_exp_var",
        "critic/grad_norm": "crit_grad",
        "critic/rewards/mean": "rewards_mean",
        "critic/advantages/mean": "adv_mean",
        "actor/pg_loss": "pg_loss",
        "actor/grad_norm": "act_grad",
        "actor/entropy_loss": "entropy",
        "actor/pg_clipfrac": "clipfrac",
        "actor/ppo_kl": "ppo_kl",
        # lengths
        "response_length/mean": "resp_len",
        "prompt_length/mean": "prompt_len",
        # timing
        "timing_s/step": "step_time_s",
        "timing_s/gen": "gen_time_s",
    }

    # 环境指标
    if env_suffix:
        env_metrics = {
            f"train/score/{env_suffix}": "train_score",
            f"train/success/{env_suffix}": "train_success",
            f"train/done/{env_suffix}": "train_done",
            f"train/action_is_valid/{env_suffix}": "train_valid",
            f"train/action_is_effective/{env_suffix}": "train_effective",
            f"train/collision_count/{env_suffix}": "train_collision",
            f"train/potential_score/{env_suffix}": "train_potential",
            f"train/total_collisions/{env_suffix}": "train_total_collisions",
            f"train/step/{env_suffix}": "train_env_steps",
            f"val/score/{env_suffix}": "val_score",
            f"val/success/{env_suffix}": "val_success",
            f"val/done/{env_suffix}": "val_done",
            f"val/potential_score/{env_suffix}": "val_potential",
            f"val/collision_count/{env_suffix}": "val_collision",
        }
        metric_map.update(env_metrics)

    available = {k: v for k, v in metric_map.items() if k in history.columns}
    df = history[list(available.keys())].rename(columns=available)
    # 只保留有 train_score 或 score_mean 的行
    score_col = "train_score" if "train_score" in df.columns else "score_mean"
    df = df.dropna(subset=[score_col])
    return df


def print_summary_table(df):
    """打印关键指标摘要表"""
    n = len(df)
    if n == 0:
        print("No data available.")
        return

    print(f"{'='*80}")
    print(f"  训练摘要 (共 {n} steps)")
    print(f"{'='*80}")

    # 分阶段统计
    phases = []
    step_col = "step" if "step" in df.columns else df.index
    steps = df["step"].values if "step" in df.columns else range(n)
    max_step = int(max(steps))

    boundaries = [0, max_step // 4, max_step // 2, 3 * max_step // 4, max_step + 1]
    phase_names = ["初始阶段", "早期", "中期", "后期"]

    score_col = "train_score" if "train_score" in df.columns else "score_mean"

    for i, name in enumerate(phase_names):
        mask = (df["step"] >= boundaries[i]) & (df["step"] < boundaries[i + 1])
        sub = df[mask]
        if len(sub) == 0:
            continue
        phases.append({
            "阶段": name,
            "Steps": f"{boundaries[i]}-{boundaries[i+1]-1}",
            "Score": f"{sub[score_col].mean():.3f}",
            "Success": f"{sub['train_success'].mean():.1%}" if "train_success" in sub else "N/A",
            "Valid": f"{sub['train_valid'].mean():.1%}" if "train_valid" in sub else "N/A",
            "VF_loss": f"{sub['vf_loss'].mean():.3f}" if "vf_loss" in sub else "N/A",
            "ActGrad": f"{sub['act_grad'].mean():.2f}" if "act_grad" in sub else "N/A",
        })

    phase_df = pd.DataFrame(phases)
    print("\n📊 分阶段统计:")
    print(phase_df.to_string(index=False))

    # 关键指标趋势
    print(f"\n📈 关键指标变化:")
    first5 = df.head(5)
    last5 = df.tail(5)

    trend_metrics = [
        ("Score", score_col),
        ("Success", "train_success"),
        ("Done", "train_done"),
        ("Valid", "train_valid"),
        ("Collision", "train_collision"),
        ("Potential", "train_potential"),
        ("VF Loss", "vf_loss"),
        ("Actor Grad", "act_grad"),
        ("Entropy", "entropy"),
        ("Resp Len", "resp_len"),
    ]

    print(f"  {'Metric':>14} | {'First 5 avg':>12} | {'Last 5 avg':>12} | {'Change':>10} | {'Trend'}")
    print(f"  {'-'*70}")
    for name, col in trend_metrics:
        if col not in df.columns:
            continue
        v1 = first5[col].mean()
        v2 = last5[col].mean()
        if pd.isna(v1) or pd.isna(v2):
            continue
        change = v2 - v1
        arrow = "↑" if change > 0.001 else ("↓" if change < -0.001 else "→")
        print(f"  {name:>14} | {v1:>12.4f} | {v2:>12.4f} | {change:>+10.4f} | {arrow}")

    # Validation
    val_cols = [c for c in df.columns if c.startswith("val_")]
    if val_cols:
        val_df = df[["step"] + val_cols].dropna(subset=val_cols, how="all")
        if not val_df.empty:
            print(f"\n📋 Validation 指标:")
            pd.set_option("display.float_format", "{:.4f}".format)
            print(val_df.to_string(index=False))


def print_detailed_table(df, interval=10):
    """打印详细指标表"""
    n = len(df)
    cols_to_show = ["step"]
    for c in ["train_score", "score_mean", "train_success", "train_done", "train_valid",
              "train_collision", "train_potential", "pg_loss", "vf_loss", "act_grad",
              "entropy", "vf_exp_var", "resp_len"]:
        if c in df.columns:
            cols_to_show.append(c)

    subset = df.iloc[::interval]
    # Always include last row
    if df.index[-1] not in subset.index:
        subset = pd.concat([subset, df.iloc[[-1]]])

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"\n{'='*80}")
    print(f"  详细指标 (每 {interval} steps)")
    print(f"{'='*80}")
    print(subset[cols_to_show].to_string(index=False))


def print_diagnosis(df):
    """自动诊断训练问题"""
    print(f"\n{'='*80}")
    print("  🔍 自动诊断")
    print(f"{'='*80}")

    issues = []
    good = []

    score_col = "train_score" if "train_score" in df.columns else "score_mean"
    last10 = df.tail(10)
    first10 = df.head(10)

    # 1. Score 趋势
    score_change = last10[score_col].mean() - first10[score_col].mean()
    if score_change > 0.3:
        good.append(f"Score 显著上升 ({first10[score_col].mean():.3f} → {last10[score_col].mean():.3f})")
    elif score_change > 0:
        good.append(f"Score 小幅上升 ({first10[score_col].mean():.3f} → {last10[score_col].mean():.3f})")
    else:
        issues.append(f"⚠️ Score 未上升或下降 ({first10[score_col].mean():.3f} → {last10[score_col].mean():.3f})")

    # 2. Success rate
    if "train_success" in df.columns:
        final_success = last10["train_success"].mean()
        if final_success > 0.3:
            good.append(f"Success rate 较好 ({final_success:.1%})")
        elif final_success > 0:
            good.append(f"Success rate 开始提升 ({final_success:.1%}), 但仍有提升空间")
        else:
            issues.append(f"⚠️ Success rate 仍为 0, 模型尚未学会完成目标任务")

    # 3. Action validity
    if "train_valid" in df.columns:
        final_valid = last10["train_valid"].mean()
        if final_valid > 0.95:
            good.append(f"Action validity 优秀 ({final_valid:.1%})")
        elif final_valid > 0.8:
            good.append(f"Action validity 良好 ({final_valid:.1%})")
        else:
            issues.append(f"⚠️ Action validity 偏低 ({final_valid:.1%}), 格式学习不充分")

    # 4. VF explained var
    if "vf_exp_var" in df.columns:
        final_vf = last10["vf_exp_var"].mean()
        if final_vf > 0:
            good.append(f"Critic 预测有效 (VF explained var: {final_vf:.3f})")
        elif final_vf > -5:
            issues.append(f"⚠️ Critic 预测能力一般 (VF explained var: {final_vf:.3f})")
        else:
            issues.append(f"⚠️ Critic 预测能力差 (VF explained var: {final_vf:.3f}), advantage 估计不准")

    # 5. Actor grad norm
    if "act_grad" in df.columns:
        max_grad = df["act_grad"].max()
        if max_grad > 50:
            issues.append(f"⚠️ Actor 梯度存在 spike (最大: {max_grad:.1f}), 训练可能不稳定")
        else:
            good.append(f"Actor 梯度稳定 (最大: {max_grad:.1f})")

    # 6. Entropy
    if "entropy" in df.columns:
        entropy_drop = first10["entropy"].mean() - last10["entropy"].mean()
        if entropy_drop > 0.3:
            issues.append(f"⚠️ Entropy 大幅下降 ({first10['entropy'].mean():.3f} → {last10['entropy'].mean():.3f}), 可能过早收敛")
        elif entropy_drop > 0.1:
            good.append(f"Entropy 正常下降 ({first10['entropy'].mean():.3f} → {last10['entropy'].mean():.3f})")

    # 7. Collision
    if "train_collision" in df.columns:
        final_coll = last10["train_collision"].mean()
        if final_coll < 0.05:
            good.append(f"碰撞率低 ({final_coll:.3f})")
        elif final_coll < 0.2:
            good.append(f"碰撞率中等 ({final_coll:.3f})")
        else:
            issues.append(f"⚠️ 碰撞率偏高 ({final_coll:.3f})")

    print("\n✅ 正面信号:")
    for g in good:
        print(f"   {g}")
    if issues:
        print("\n⚠️ 需要关注:")
        for i in issues:
            print(f"   {i}")
    else:
        print("\n   训练整体健康！")

    # 建议
    print("\n💡 建议:")
    n_steps = len(df)
    if n_steps < 200:
        print(f"   - 当前仅训练 {n_steps} steps, 建议至少训练 500+ steps 再评估效果")
    if "train_success" in df.columns and last10["train_success"].mean() < 0.5:
        print("   - 可以尝试增大 success_reward 权重或降低任务难度")
    if "vf_exp_var" in df.columns and last10["vf_exp_var"].mean() < -3:
        print("   - Critic 学习较慢, 考虑增大 critic lr 或减小 batch size")
    if "act_grad" in df.columns and df["act_grad"].max() > 100:
        print("   - 建议添加 gradient clipping 或降低 actor lr")


def save_plots(df, run_id, output_dir="analysis_plots"):
    """保存训练曲线图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装, 跳过绘图。请安装: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    score_col = "train_score" if "train_score" in df.columns else "score_mean"
    steps = df["step"].values

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f"Training Analysis: {run_id}", fontsize=14)

    plots = [
        (0, 0, score_col, "Score", "blue"),
        (0, 1, "train_success", "Success Rate", "green"),
        (0, 2, "train_valid", "Action Validity", "orange"),
        (1, 0, "vf_loss", "Critic VF Loss", "red"),
        (1, 1, "act_grad", "Actor Grad Norm", "purple"),
        (1, 2, "entropy", "Entropy", "brown"),
        (2, 0, "train_collision", "Collision Count", "darkred"),
        (2, 1, "train_potential", "Potential Score", "teal"),
        (2, 2, "vf_exp_var", "VF Explained Var", "navy"),
    ]

    for r, c, col, title, color in plots:
        ax = axes[r][c]
        if col in df.columns and df[col].notna().any():
            ax.plot(steps, df[col].values, color=color, alpha=0.6, linewidth=0.8)
            # 滑动平均
            window = min(10, len(df) // 3) if len(df) > 10 else 1
            if window > 1:
                smoothed = df[col].rolling(window=window, min_periods=1).mean()
                ax.plot(steps, smoothed.values, color=color, linewidth=2, label=f"MA-{window}")
                ax.legend(fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=11, color="gray")

    plt.tight_layout()
    path = os.path.join(output_dir, f"{run_id}_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 图表已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="Wandb 训练曲线获取与分析")
    parser.add_argument("--run_id", type=str, help="Wandb Run ID (例如 yr5xf955)")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Wandb project (entity/project)")
    parser.add_argument("--list", action="store_true", help="列出项目中所有 run")
    parser.add_argument("--export", choices=["csv", "json"], help="导出数据格式")
    parser.add_argument("--save_plots", action="store_true", help="保存训练曲线图")
    parser.add_argument("--interval", type=int, default=10, help="详细表格显示间隔 (默认 10)")
    parser.add_argument("--output_dir", type=str, default="analysis_plots", help="图表输出目录")
    args = parser.parse_args()

    if args.list:
        list_runs(args.project)
        return

    if not args.run_id:
        parser.error("请指定 --run_id 或使用 --list 查看可用 run")

    # 获取数据
    run, history = fetch_history(args.run_id, args.project)
    df = build_metrics_df(history)

    if len(df) == 0:
        print("ERROR: 未获取到训练数据")
        sys.exit(1)

    # 打印分析
    print_summary_table(df)
    print_detailed_table(df, interval=args.interval)
    print_diagnosis(df)

    # 导出
    if args.export == "csv":
        path = f"{args.run_id}_metrics.csv"
        df.to_csv(path, index=False)
        print(f"\n📁 数据已导出: {path}")
    elif args.export == "json":
        path = f"{args.run_id}_metrics.json"
        df.to_json(path, orient="records", indent=2)
        print(f"\n📁 数据已导出: {path}")

    # 绘图
    if args.save_plots:
        save_plots(df, args.run_id, args.output_dir)


if __name__ == "__main__":
    main()
