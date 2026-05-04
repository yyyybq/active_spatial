#!/usr/bin/env python3
"""
Reusable analyzer for VAGEN experiment logs + local wandb run logs.

What it does:
1) Detects the matching local wandb run from an experiment log.
2) Parses step metrics from wandb output.log (fallback: experiment log).
3) Checks potential garbled outputs from response previews.
4) Summarizes early/mid/late RL metric behavior.
5) Writes a markdown report for later comparison.

Usage example:
  python scripts/analyze_experiment_log_wandb.py \
      --log cambrian_v4_7gpu.log
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\b")
KV_NUM_RE = re.compile(r"\s*([^:]+):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
RUN_DIR_RE = re.compile(r"run-\d{8}_\d{6}-[a-z0-9]+")
RUN_ID_RE = re.compile(r"\b([a-z0-9]{8})\b")
ENV_SUFFIX_RE = re.compile(r"/ActiveSpatialEnvConfig\(.*\)$")
ESCAPED_HEX_RE = re.compile(r"\\x[0-9a-fA-F]{2}")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def read_lines(path: Path) -> List[str]:
    # Read with replacement to avoid parser interruption on any odd bytes.
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def normalize_metric_key(key: str) -> str:
    k = key.strip()
    k = ENV_SUFFIX_RE.sub("", k)
    return k


@dataclass
class RunMatch:
    run_id: Optional[str]
    run_dir_name: Optional[str]
    run_dir_path: Optional[Path]
    confidence: str


def infer_wandb_run(log_lines: List[str], wandb_root: Path) -> RunMatch:
    run_dir_name: Optional[str] = None
    run_id: Optional[str] = None

    for raw in log_lines:
        line = strip_ansi(raw)
        if "wandb:" not in line.lower():
            continue

        m_dir = RUN_DIR_RE.search(line)
        if m_dir:
            run_dir_name = m_dir.group(0)

        # Common wandb URL format includes /runs/<run_id>
        if "/runs/" in line:
            rid = line.rsplit("/runs/", 1)[-1].strip().split()[0]
            rid = rid.strip("/)")
            if re.fullmatch(r"[a-z0-9]{8}", rid):
                run_id = rid

    if run_dir_name and not run_id:
        # Extract run id from run-YYYYMMDD_HHMMSS-<id>
        parts = run_dir_name.split("-")
        if len(parts) >= 3:
            run_id = parts[-1]

    run_dir_path = None
    confidence = "low"
    if run_dir_name:
        candidate = wandb_root / run_dir_name
        if candidate.exists():
            run_dir_path = candidate
            confidence = "high"
    elif run_id:
        hits = sorted(wandb_root.glob(f"run-*-{run_id}"))
        if hits:
            run_dir_path = hits[-1]
            run_dir_name = run_dir_path.name
            confidence = "medium"

    return RunMatch(
        run_id=run_id,
        run_dir_name=run_dir_name,
        run_dir_path=run_dir_path,
        confidence=confidence,
    )


def parse_step_metrics(lines: List[str]) -> Dict[int, Dict[str, float]]:
    by_step: Dict[int, Dict[str, float]] = {}

    for raw in lines:
        line = strip_ansi(raw)
        step_match = STEP_RE.search(line)
        if not step_match:
            continue

        step = int(step_match.group(1))
        if step not in by_step:
            by_step[step] = {}

        # Keep only content after "step:<n> -" when available.
        pivot = line.find(f"step:{step}")
        tail = line[pivot:]
        if " - " in tail:
            tail = tail.split(" - ", 1)[1]
        else:
            continue

        parts = tail.split(" - ")
        for part in parts:
            m = KV_NUM_RE.match(part)
            if not m:
                continue
            key_raw, val_raw = m.group(1), m.group(2)
            key = normalize_metric_key(key_raw)
            try:
                val = float(val_raw)
            except ValueError:
                continue
            by_step[step][key] = val

    return by_step


def pick_metric_key(metric_map: Dict[str, float], candidates: List[str], contains: List[str]) -> Optional[str]:
    keys = list(metric_map.keys())

    for c in candidates:
        if c in metric_map:
            return c

    for k in keys:
        lk = k.lower()
        if all(token in lk for token in contains):
            return k
    return None


def extract_series(by_step: Dict[int, Dict[str, float]], preferred_keys: List[str], contains: List[str]) -> List[Tuple[int, float]]:
    all_keys: Dict[str, int] = {}
    for step_vals in by_step.values():
        for k in step_vals:
            all_keys[k] = all_keys.get(k, 0) + 1

    resolved = pick_metric_key(all_keys, preferred_keys, contains)
    if not resolved:
        return []

    series: List[Tuple[int, float]] = []
    for s in sorted(by_step.keys()):
        if resolved in by_step[s]:
            series.append((s, by_step[s][resolved]))
    return series


def summarize_three_phases(series: List[Tuple[int, float]]) -> Optional[Dict[str, float]]:
    if not series:
        return None

    values = [v for _, v in series]
    n = len(values)
    first_end = max(1, n // 3)
    second_end = max(first_end + 1, (2 * n) // 3)

    early = values[:first_end]
    mid = values[first_end:second_end] or values[first_end:first_end + 1]
    late = values[second_end:] or values[-1:]

    early_m = mean(early)
    mid_m = mean(mid)
    late_m = mean(late)
    delta = late_m - early_m

    if delta > 1e-9:
        trend = "up"
    elif delta < -1e-9:
        trend = "down"
    else:
        trend = "flat"

    return {
        "early_mean": early_m,
        "mid_mean": mid_m,
        "late_mean": late_m,
        "delta_late_vs_early": delta,
        "trend": trend,
    }


def find_response_preview_samples(log_lines: List[str], limit_each_side: int = 3) -> Tuple[List[str], List[str]]:
    previews: List[str] = []
    for raw in log_lines:
        line = strip_ansi(raw)
        marker = "response_preview="
        if marker not in line:
            continue
        text = line.split(marker, 1)[1].strip()
        previews.append(text)

    if not previews:
        return [], []

    head = previews[:limit_each_side]
    tail = previews[-limit_each_side:] if len(previews) > limit_each_side else []
    return head, tail


def count_garbled_signals(log_text: str, preview_texts: List[str]) -> Dict[str, int]:
    replacement = log_text.count("\ufffd")
    escaped_hex = len(ESCAPED_HEX_RE.findall(log_text))
    mojibake_markers = sum(log_text.count(x) for x in ["Ã", "Â"])
    trunc_warnings = len(re.findall(r"\[WARNING\]\s+Left truncation", log_text))

    suspicious_preview = 0
    for text in preview_texts:
        if "\ufffd" in text or ESCAPED_HEX_RE.search(text):
            suspicious_preview += 1

    return {
        "replacement_char_count": replacement,
        "escaped_hex_count": escaped_hex,
        "mojibake_marker_count": mojibake_markers,
        "left_truncation_warning_count": trunc_warnings,
        "suspicious_preview_count": suspicious_preview,
    }


def maybe_get_metrics_source(run_match: RunMatch, exp_log_path: Path) -> Path:
    if run_match.run_dir_path:
        output_log = run_match.run_dir_path / "files" / "output.log"
        if output_log.exists():
            return output_log
    return exp_log_path


def format_metric_line(name: str, summary: Optional[Dict[str, float]]) -> str:
    if not summary:
        return f"- {name}: N/A"
    return (
        f"- {name}: early={summary['early_mean']:.4f}, "
        f"mid={summary['mid_mean']:.4f}, "
        f"late={summary['late_mean']:.4f}, "
        f"delta={summary['delta_late_vs_early']:+.4f}, trend={summary['trend']}"
    )


def build_report(
    exp_log_path: Path,
    run_match: RunMatch,
    metrics_source: Path,
    by_step: Dict[int, Dict[str, float]],
    metric_summaries: Dict[str, Optional[Dict[str, float]]],
    garble_counts: Dict[str, int],
    preview_head: List[str],
    preview_tail: List[str],
) -> str:
    steps_sorted = sorted(by_step.keys())
    step_info = (
        f"{steps_sorted[0]} -> {steps_sorted[-1]} (count={len(steps_sorted)})"
        if steps_sorted
        else "N/A"
    )

    lines: List[str] = []
    lines.append("# Experiment + W&B Analysis Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- experiment_log: {exp_log_path}")
    lines.append(f"- wandb_root: {exp_log_path.parent / 'wandb'}")
    lines.append(f"- matched_run_dir: {run_match.run_dir_path}")
    lines.append(f"- matched_run_id: {run_match.run_id}")
    lines.append(f"- run_match_confidence: {run_match.confidence}")
    lines.append(f"- metrics_source: {metrics_source}")
    lines.append("")
    lines.append("## Metric Coverage")
    lines.append(f"- parsed_steps: {step_info}")
    lines.append("")
    lines.append("## RL Metric Trends (Early/Mid/Late)")
    lines.append(format_metric_line("train/score", metric_summaries["train_score"]))
    lines.append(format_metric_line("train/success", metric_summaries["train_success"]))
    lines.append(format_metric_line("critic/vf_loss", metric_summaries["critic_vf_loss"]))
    lines.append(format_metric_line("actor/pg_loss", metric_summaries["actor_pg_loss"]))
    lines.append(format_metric_line("actor/entropy_loss", metric_summaries["actor_entropy_loss"]))
    lines.append(format_metric_line("kl (ppo/actor)", metric_summaries["kl"]))
    lines.append(format_metric_line("train/total_collisions", metric_summaries["total_collisions"]))
    lines.append(format_metric_line("response_length/mean", metric_summaries["response_length_mean"]))
    lines.append(format_metric_line("reward-related", metric_summaries["reward_related"]))
    lines.append("")
    lines.append("## Garbled Output Check")
    lines.append(f"- replacement_char_count: {garble_counts['replacement_char_count']}")
    lines.append(f"- mojibake_marker_count(Ã/Â): {garble_counts['mojibake_marker_count']}")
    lines.append(f"- escaped_hex_count(\\xNN): {garble_counts['escaped_hex_count']}")
    lines.append(f"- suspicious_preview_count: {garble_counts['suspicious_preview_count']}")
    lines.append(f"- left_truncation_warning_count: {garble_counts['left_truncation_warning_count']}")
    lines.append("")
    if (
        garble_counts["replacement_char_count"] == 0
        and garble_counts["mojibake_marker_count"] == 0
        and garble_counts["escaped_hex_count"] == 0
        and garble_counts["suspicious_preview_count"] == 0
    ):
        lines.append("- conclusion: no obvious text-encoding garble found in log samples.")
    else:
        lines.append("- conclusion: possible garble indicators found; inspect preview samples and raw bytes.")
    lines.append("")
    lines.append("## response_preview Samples")
    if not preview_head and not preview_tail:
        lines.append("- no response_preview lines found")
    else:
        for i, text in enumerate(preview_head, start=1):
            lines.append(f"- head[{i}]: {text[:220]}")
        for i, text in enumerate(preview_tail, start=1):
            lines.append(f"- tail[{i}]: {text[:220]}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment log + local wandb run metrics.")
    parser.add_argument("--log", required=True, help="Path to experiment log file")
    parser.add_argument(
        "--wandb-root",
        default=None,
        help="Path to local wandb root directory (default: <log_dir>/wandb)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output markdown report path (default: <log_dir>/<log_stem>_analysis.md)",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional output json summary path",
    )
    args = parser.parse_args()

    exp_log_path = Path(args.log).expanduser().resolve()
    if not exp_log_path.exists():
        raise FileNotFoundError(f"Log file not found: {exp_log_path}")

    wandb_root = Path(args.wandb_root).expanduser().resolve() if args.wandb_root else exp_log_path.parent / "wandb"
    if not wandb_root.exists():
        # Keep fallback behavior; the script can still parse experiment log directly.
        wandb_root = exp_log_path.parent / "wandb"

    log_lines = read_lines(exp_log_path)
    run_match = infer_wandb_run(log_lines, wandb_root)
    metrics_source = maybe_get_metrics_source(run_match, exp_log_path)
    metrics_lines = read_lines(metrics_source)
    by_step = parse_step_metrics(metrics_lines)

    metric_summaries: Dict[str, Optional[Dict[str, float]]] = {}
    metric_summaries["train_score"] = summarize_three_phases(
        extract_series(by_step, ["train/score", "critic/score/mean"], ["train", "score"])
        or extract_series(by_step, ["critic/score/mean"], ["critic", "score", "mean"])
    )
    metric_summaries["train_success"] = summarize_three_phases(
        extract_series(by_step, ["train/success"], ["train", "success"])
    )
    metric_summaries["critic_vf_loss"] = summarize_three_phases(
        extract_series(by_step, ["critic/vf_loss"], ["vf_loss"])
    )
    metric_summaries["actor_pg_loss"] = summarize_three_phases(
        extract_series(by_step, ["actor/pg_loss"], ["pg_loss"])
    )
    metric_summaries["actor_entropy_loss"] = summarize_three_phases(
        extract_series(by_step, ["actor/entropy_loss"], ["entropy"])
    )
    metric_summaries["kl"] = summarize_three_phases(
        extract_series(by_step, ["actor/ppo_kl", "actor/kl_loss"], ["kl"])
    )
    metric_summaries["total_collisions"] = summarize_three_phases(
        extract_series(by_step, ["train/total_collisions"], ["total", "collision"])
    )
    metric_summaries["response_length_mean"] = summarize_three_phases(
        extract_series(by_step, ["response_length/mean"], ["response_length", "mean"])
    )
    metric_summaries["reward_related"] = summarize_three_phases(
        extract_series(by_step, ["critic/rewards/mean"], ["reward"])
    )

    head_previews, tail_previews = find_response_preview_samples(log_lines)
    all_previews = head_previews + tail_previews
    full_log_text = "\n".join(log_lines)
    garble_counts = count_garbled_signals(full_log_text, all_previews)

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else exp_log_path.with_name(f"{exp_log_path.stem}_analysis.md")
    )
    report_text = build_report(
        exp_log_path=exp_log_path,
        run_match=run_match,
        metrics_source=metrics_source,
        by_step=by_step,
        metric_summaries=metric_summaries,
        garble_counts=garble_counts,
        preview_head=head_previews,
        preview_tail=tail_previews,
    )

    report_path.write_text(report_text, encoding="utf-8")

    json_path: Optional[Path] = None
    if args.json:
        json_path = Path(args.json).expanduser().resolve()
        payload = {
            "experiment_log": str(exp_log_path),
            "wandb_root": str(wandb_root),
            "run_match": {
                "run_id": run_match.run_id,
                "run_dir_name": run_match.run_dir_name,
                "run_dir_path": str(run_match.run_dir_path) if run_match.run_dir_path else None,
                "confidence": run_match.confidence,
            },
            "metrics_source": str(metrics_source),
            "parsed_step_count": len(by_step),
            "metric_summaries": metric_summaries,
            "garble_counts": garble_counts,
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Analysis completed.")
    print(f"- report: {report_path}")
    if json_path:
        print(f"- json: {json_path}")
    print(f"- matched run: {run_match.run_dir_name} (id={run_match.run_id}, confidence={run_match.confidence})")
    print(f"- metrics source: {metrics_source}")
    print(f"- parsed steps: {len(by_step)}")


if __name__ == "__main__":
    main()
