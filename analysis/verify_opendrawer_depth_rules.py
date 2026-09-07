"""Verify two exploratory episode-level OpenDrawer depth rules.

Rules are evaluated without using operational onset timestamps:

1. episode median of ``feat_depth_change_ratio_prev`` < 0.22 -> failure;
2. fraction of steps with ``feat_depth_change_ratio_prev`` < 0.10 > 0.14
   -> failure.

The thresholds are supplied by the user and are not learned from the labels
in this script.  The output is therefore a transparent retrospective check,
not a validated online detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


DEFAULT_INPUT = Path("data/formal_features/step_features_trace_depth.csv")
DEFAULT_OUTPUT = Path("data/opendrawer_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--median-threshold", type=float, default=0.22)
    parser.add_argument("--low-ratio-threshold", type=float, default=0.10)
    parser.add_argument("--low-fraction-threshold", type=float, default=0.14)
    return parser.parse_args()


def episode_summary(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_key, episode in data.groupby("episode_key", sort=True):
        values = episode["feat_depth_change_ratio_prev"].dropna().to_numpy(float)
        if len(values) == 0:
            continue
        median = float(np.median(values))
        low_fraction = float(np.mean(values < args.low_ratio_threshold))
        actual_failure = int(episode["label_episode_failure"].iloc[0])
        pred_median = int(median < args.median_threshold)
        pred_low_fraction = int(low_fraction > args.low_fraction_threshold)
        rows.append(
            {
                "task": "open_drawer",
                "condition_id": int(episode["condition_id"].iloc[0]),
                "episode_key": episode_key,
                "episode_success": int(episode["label_episode_success"].iloc[0]),
                "episode_failure": actual_failure,
                "num_valid_steps": int(len(values)),
                "episode_median": median,
                "low_ratio_threshold": args.low_ratio_threshold,
                "low_ratio_fraction": low_fraction,
                "median_rule_failure": pred_median,
                "low_fraction_rule_failure": pred_low_fraction,
                "median_rule_correct": int(pred_median == actual_failure),
                "low_fraction_rule_correct": int(pred_low_fraction == actual_failure),
            }
        )
    return pd.DataFrame(rows)


def rule_stats(summary: pd.DataFrame, prediction: str) -> dict[str, object]:
    actual = summary["episode_failure"].to_numpy(int)
    predicted = summary[prediction].to_numpy(int)
    tp = int(((predicted == 1) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    return {
        "rule": prediction,
        "n_episodes": len(summary),
        "failure_n": int(actual.sum()),
        "success_n": int((actual == 0).sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / len(actual)),
        "sensitivity_failure_recall": float(tp / max(1, tp + fn)),
        "specificity_success_recall": float(tn / max(1, tn + fp)),
        "failure_precision": float(tp / max(1, tp + fp)),
    }


def metric_stats(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    failure = summary.loc[summary["episode_failure"].eq(1)]
    success = summary.loc[summary["episode_success"].eq(1)]
    for metric, score in [
        ("episode_median", -summary["episode_median"]),
        ("low_ratio_fraction", summary["low_ratio_fraction"]),
    ]:
        f = score.loc[summary["episode_failure"].eq(1)]
        s = score.loc[summary["episode_success"].eq(1)]
        labels = np.r_[np.ones(len(f)), np.zeros(len(s))]
        scores = np.r_[f.to_numpy(float), s.to_numpy(float)]
        rows.append(
            {
                "metric": metric,
                "failure_mean": float(summary.loc[failure.index, metric].mean()),
                "failure_median": float(summary.loc[failure.index, metric].median()),
                "success_mean": float(summary.loc[success.index, metric].mean()),
                "success_median": float(summary.loc[success.index, metric].median()),
                "failure_high_auc": float(roc_auc_score(labels, scores)),
                "mann_whitney_p": float(mannwhitneyu(f, s, alternative="two-sided").pvalue),
            }
        )
    return pd.DataFrame(rows)


def plot_rules(summary: pd.DataFrame, args: argparse.Namespace, output: Path) -> None:
    colors = np.where(summary["episode_failure"].eq(1), "#d1495b", "#777777")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, x, threshold, xlabel in [
        (
            axes[0],
            summary["episode_median"],
            args.median_threshold,
            "Episode median of depth-change ratio",
        ),
        (
            axes[1],
            summary["low_ratio_fraction"],
            args.low_fraction_threshold,
            "Fraction with depth-change ratio < 0.10",
        ),
    ]:
        axis.scatter(x, summary["condition_id"], c=colors, s=55, alpha=0.9)
        axis.axvline(threshold, color="#2563eb", linestyle="--", linewidth=1.5)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Condition ID")
        axis.grid(alpha=0.25)
    fig.suptitle("OpenDrawer exploratory depth rules | no onset used")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    stats: pd.DataFrame,
    metric_summary: pd.DataFrame,
    args: argparse.Namespace,
    output: Path,
) -> None:
    lines = [
        "# OpenDrawer exploratory depth-rule verification",
        "",
        f"Rules: episode median `< {args.median_threshold}`; fraction with "
        f"depth-change ratio `< {args.low_ratio_threshold}` greater than "
        f"`{args.low_fraction_threshold}`.",
        "",
        stats.to_csv(index=False),
        "",
        metric_summary.to_csv(index=False),
        "",
        "Both rules are evaluated retrospectively on complete episodes and use "
        "no operational onset. They are not cross-validated and the perfect "
        "separation on this 20-episode sample must not be treated as a "
        "generalization result.",
        "",
        "Closest threshold margins: ",
        f"- median: failure minimum margin = "
        f"{args.median_threshold - summary.loc[summary.episode_failure.eq(1), 'episode_median'].max():.4f}; "
        f"success minimum margin = "
        f"{summary.loc[summary.episode_success.eq(1), 'episode_median'].min() - args.median_threshold:.4f}",
        f"- low-fraction: failure minimum margin = "
        f"{summary.loc[summary.episode_failure.eq(1), 'low_ratio_fraction'].min() - args.low_fraction_threshold:.4f}; "
        f"success minimum margin = "
        f"{args.low_fraction_threshold - summary.loc[summary.episode_success.eq(1), 'low_ratio_fraction'].max():.4f}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    data = data.loc[data["task"].eq("open_drawer")].copy()
    data["feat_depth_change_ratio_prev"] = pd.to_numeric(
        data["feat_depth_change_ratio_prev"], errors="coerce"
    )
    summary = episode_summary(data, args)
    stats = pd.DataFrame(
        [
            rule_stats(summary, "median_rule_failure"),
            rule_stats(summary, "low_fraction_rule_failure"),
        ]
    )
    metric_summary = metric_stats(summary)
    summary_path = args.output_dir / "opendrawer_depth_rule_episode.csv"
    stats_path = args.output_dir / "opendrawer_depth_rule_stats.csv"
    metric_path = args.output_dir / "opendrawer_depth_rule_metric_summary.csv"
    plot_path = args.output_dir / "opendrawer_depth_rule_verification.png"
    report_path = args.output_dir / "opendrawer_depth_rule_verification_report.md"
    summary.to_csv(summary_path, index=False)
    stats.to_csv(stats_path, index=False)
    metric_summary.to_csv(metric_path, index=False)
    plot_rules(summary, args, plot_path)
    write_report(summary, stats, metric_summary, args, report_path)
    print(stats.to_string(index=False))
    print(metric_summary.to_string(index=False))
    print(summary_path)
    print(plot_path)


if __name__ == "__main__":
    main()
