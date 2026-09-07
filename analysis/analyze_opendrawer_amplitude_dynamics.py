"""Evaluate amplitude-thresholded depth-change dynamics.

No operational onset is used.  For a depth-change sequence y, let
delta[t] = y[t] - y[t-1].  A turning event requires both adjacent changes to
have magnitude greater than epsilon and to have opposite signs.  The primary
turning rate is conditional on amplitude-qualified adjacent pairs; an
all-pairs denominator is also retained to make the denominator choice
explicit.  Prominent peak rate and median inter-peak interval use the same
epsilon as a peak-prominence threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


DEFAULT_INPUT = Path("data/formal_features/step_features_trace_depth.csv")
DEFAULT_OUTPUT = Path("data/opendrawer_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epsilon", type=float, default=0.05)
    return parser.parse_args()


def episode_metrics(values: np.ndarray, epsilon: float) -> dict[str, float]:
    deltas = np.diff(values)
    eligible = (np.abs(deltas[:-1]) > epsilon) & (np.abs(deltas[1:]) > epsilon)
    turns = eligible & (deltas[:-1] * deltas[1:] < 0)
    qualified_pairs = int(eligible.sum())
    num_turns = int(turns.sum())

    peaks, _ = find_peaks(values, prominence=epsilon)
    intervals = np.diff(peaks)
    return {
        "num_eligible_adjacent_pairs": qualified_pairs,
        "num_amplitude_turns": num_turns,
        "amplitude_thresholded_turning_rate": (
            num_turns / qualified_pairs if qualified_pairs else np.nan
        ),
        "amplitude_thresholded_turning_rate_all_pairs": num_turns
        / max(1, len(values) - 2),
        "prominent_peak_count": int(len(peaks)),
        "prominent_peak_rate_per10": len(peaks) / len(values) * 10,
        "median_inter_peak_interval": (
            float(np.median(intervals)) if len(intervals) else np.nan
        ),
    }


def load_summary(args: argparse.Namespace) -> pd.DataFrame:
    data = pd.read_csv(args.input)
    data = data.loc[data["task"].eq("open_drawer")].copy()
    data["feat_depth_change_ratio_prev"] = pd.to_numeric(
        data["feat_depth_change_ratio_prev"], errors="coerce"
    )
    rows: list[dict[str, object]] = []
    for episode_key, episode in data.groupby("episode_key", sort=True):
        values = episode["feat_depth_change_ratio_prev"].dropna().to_numpy(float)
        if len(values) < 3:
            continue
        row: dict[str, object] = {
            "task": "open_drawer",
            "condition_id": int(episode["condition_id"].iloc[0]),
            "episode_key": episode_key,
            "episode_success": int(episode["label_episode_success"].iloc[0]),
            "episode_failure": int(episode["label_episode_failure"].iloc[0]),
            "num_valid_steps": int(len(values)),
            "epsilon": args.epsilon,
        }
        row.update(episode_metrics(values, args.epsilon))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_groups(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "amplitude_thresholded_turning_rate",
        "amplitude_thresholded_turning_rate_all_pairs",
        "prominent_peak_rate_per10",
        "median_inter_peak_interval",
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        failure = summary.loc[summary["episode_failure"].eq(1), metric].dropna()
        success = summary.loc[summary["episode_success"].eq(1), metric].dropna()
        labels = np.r_[np.ones(len(failure)), np.zeros(len(success))]
        scores = np.r_[failure, success]
        rows.append(
            {
                "metric": metric,
                "failure_n": len(failure),
                "success_n": len(success),
                "failure_mean": float(failure.mean()),
                "failure_median": float(failure.median()),
                "success_mean": float(success.mean()),
                "success_median": float(success.median()),
                "failure_minus_success_mean": float(failure.mean() - success.mean()),
                "failure_high_auc": float(roc_auc_score(labels, scores)),
                "mann_whitney_p": float(
                    mannwhitneyu(failure, success, alternative="two-sided").pvalue
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("amplitude_thresholded_turning_rate", "Turning rate | qualified pairs"),
        ("prominent_peak_rate_per10", "Prominent peaks per 10 steps"),
        ("median_inter_peak_interval", "Median inter-peak interval (steps)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    colors = {0: "#777777", 1: "#d1495b"}
    labels = {0: "success", 1: "failure"}
    for axis, (metric, title) in zip(axes, metrics):
        for outcome in [0, 1]:
            group = summary.loc[summary["episode_failure"].eq(outcome), metric].dropna()
            x = np.full(len(group), outcome + 1, dtype=float)
            jitter = np.linspace(-0.08, 0.08, len(group)) if len(group) > 1 else np.zeros(len(group))
            axis.scatter(
                x + jitter,
                group,
                color=colors[outcome],
                s=48,
                alpha=0.85,
                label=labels[outcome] if metric == metrics[0][0] else None,
            )
        axis.set_xticks([1, 2], ["success", "failure"])
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("OpenDrawer amplitude-thresholded depth dynamics | epsilon=0.05")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, group_summary: pd.DataFrame, args: argparse.Namespace, output: Path) -> None:
    lines = [
        "# OpenDrawer amplitude-thresholded depth dynamics",
        "",
        "No operational onset is used. The primary turning rate counts a sign "
        "reversal only when both adjacent depth changes exceed the amplitude "
        f"threshold epsilon = `{args.epsilon}`; its denominator is the number "
        "of amplitude-qualified adjacent pairs.",
        "",
        group_summary.to_csv(index=False),
        "",
        "The all-pairs turning-rate denominator is retained because it answers a "
        "different question: how many large turning events occur per episode "
        "step. Prominent peak rate and inter-peak interval are amplitude-filtered "
        "and should be preferred over raw local maxima when small token-level "
        "wiggles dominate.",
        "",
        "These are exploratory episode-level summaries (8 failures and 12 "
        "successes), not cross-validated predictors.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args)
    group_summary = summarize_groups(summary)
    summary_path = args.output_dir / "opendrawer_amplitude_dynamics_episode.csv"
    group_path = args.output_dir / "opendrawer_amplitude_dynamics_group_summary.csv"
    plot_path = args.output_dir / "opendrawer_amplitude_dynamics.png"
    report_path = args.output_dir / "opendrawer_amplitude_dynamics_report.md"
    summary.to_csv(summary_path, index=False)
    group_summary.to_csv(group_path, index=False)
    plot_summary(summary, plot_path)
    write_report(summary, group_summary, args, report_path)
    print(group_summary.to_string(index=False))
    print(summary_path)
    print(plot_path)


if __name__ == "__main__":
    main()
