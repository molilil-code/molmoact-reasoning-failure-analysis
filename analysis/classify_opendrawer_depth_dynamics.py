"""Classify OpenDrawer episode-level depth-change dynamics.

This is a retrospective, failure-only descriptive classification.  It does
not use operational onset timestamps.  The provisional labels follow the
requested visual taxonomy:

* stagnation-like: a large fraction of low-change steps and a sustained
  low-change run, with relatively small total variation;
* oscillation/burst-like: larger total variation and/or frequent peak-valley
  switching, with less persistent low-change behavior.

Thresholds are explicit command-line arguments because this eight-episode
classification is exploratory and must not be presented as a validated
general-purpose detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/formal_features/step_features_trace_depth.csv")
DEFAULT_OUTPUT = Path("data/opendrawer_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--low-change-threshold", type=float, default=0.10)
    parser.add_argument("--stagnation-fraction-threshold", type=float, default=0.40)
    parser.add_argument("--stagnation-run-threshold", type=float, default=0.10)
    parser.add_argument("--burst-turning-threshold", type=float, default=0.65)
    parser.add_argument("--burst-variation-threshold", type=float, default=13.0)
    return parser.parse_args()


def longest_true_run(values: np.ndarray) -> int:
    current = 0
    longest = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def turning_rate(values: np.ndarray) -> float:
    differences = np.diff(values)
    signs = np.sign(differences)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    return float(np.mean(signs[1:] != signs[:-1]))


def episode_features(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_key, episode in data.groupby("episode_key", sort=True):
        values = episode["feat_depth_change_ratio_prev"].dropna().to_numpy(float)
        if len(values) < 3:
            continue
        low = values < args.low_change_threshold
        longest = longest_true_run(low)
        variation = float(np.abs(np.diff(values)).sum())
        low_fraction = float(np.mean(low))
        run_ratio = float(longest / len(values))
        turn = turning_rate(values)

        stagnation_evidence = int(low_fraction >= args.stagnation_fraction_threshold) + int(
            run_ratio >= args.stagnation_run_threshold
        )
        burst_evidence = int(turn >= args.burst_turning_threshold) + int(
            variation >= args.burst_variation_threshold
        )
        if stagnation_evidence > burst_evidence:
            mode = "stagnation-like"
        elif burst_evidence > stagnation_evidence:
            mode = "oscillation/burst-like"
        else:
            # The tie-break makes the high-variation C10 profile burst-like,
            # while preserving an explicit and auditable rule.
            mode = (
                "oscillation/burst-like"
                if variation >= args.burst_variation_threshold
                else "stagnation-like"
            )

        rows.append(
            {
                "task": "open_drawer",
                "condition_id": int(episode["condition_id"].iloc[0]),
                "episode_key": episode_key,
                "episode_success": int(episode["label_episode_success"].iloc[0]),
                "episode_failure": int(episode["label_episode_failure"].iloc[0]),
                "num_valid_steps": int(len(values)),
                "low_change_threshold": args.low_change_threshold,
                "low_change_fraction": low_fraction,
                "longest_low_change_run_ratio": run_ratio,
                "turning_rate": turn,
                "total_variation": variation,
                "total_variation_per_step": variation / max(1, len(values) - 1),
                "episode_mean": float(np.mean(values)),
                "episode_median": float(np.median(values)),
                "episode_p90": float(np.quantile(values, 0.90)),
                "stagnation_evidence": stagnation_evidence,
                "burst_evidence": burst_evidence,
                "depth_dynamics_class": mode if not episode["label_episode_success"].iloc[0] else "success",
            }
        )
    return pd.DataFrame(rows)


def plot_features(summary: pd.DataFrame, output: Path) -> None:
    failure = summary.loc[summary["episode_failure"].eq(1)].copy()
    success = summary.loc[summary["episode_success"].eq(1)].copy()
    colors = {
        "stagnation-like": "#3b82f6",
        "oscillation/burst-like": "#d1495b",
        "success": "#777777",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    specs = [
        ("low_change_fraction", "Low-change fraction", None),
        ("longest_low_change_run_ratio", "Longest low-change run ratio", None),
        ("total_variation", "Total variation", "turning_rate"),
    ]
    for axis, (xcol, xlabel, ycol) in zip(axes, specs):
        if ycol is None:
            for mode, group in failure.groupby("depth_dynamics_class", sort=True):
                axis.scatter(
                    group[xcol],
                    group["turning_rate"],
                    color=colors[mode],
                    s=55,
                    label=mode,
                )
            axis.set_ylabel("Turning rate")
        else:
            for mode, group in failure.groupby("depth_dynamics_class", sort=True):
                axis.scatter(
                    group[xcol],
                    group[ycol],
                    color=colors[mode],
                    s=55,
                    label=mode,
                )
            axis.set_ylabel("Turning rate")
        axis.scatter(
            success[xcol],
            success["turning_rate"],
            color=colors["success"],
            marker="x",
            s=40,
            alpha=0.75,
            label="success" if axis is axes[0] else None,
        )
        for _, row in summary.iterrows():
            if row["episode_failure"]:
                axis.annotate(
                    f"C{int(row['condition_id']):02d}",
                    (row[xcol], row["turning_rate"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("OpenDrawer depth-change dynamics | no onset used", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, args: argparse.Namespace, output: Path) -> None:
    failure = summary.loc[summary["episode_failure"].eq(1)]
    groups = failure.groupby("depth_dynamics_class")
    rows = []
    for mode, group in groups:
        rows.append(
            f"| {mode} | {len(group)} | {group['low_change_fraction'].mean():.3f} | "
            f"{group['longest_low_change_run_ratio'].mean():.3f} | "
            f"{group['turning_rate'].mean():.3f} | {group['total_variation'].mean():.2f} |"
        )
    table = [
        "| class | episodes | low-change fraction | longest low run | turning rate | total variation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
    ]
    lines = [
        "# OpenDrawer depth-change dynamics classification",
        "",
        "This retrospective classification uses only the complete "
        "`feat_depth_change_ratio_prev` trajectory. Operational onset is not used.",
        "",
        f"Low-change threshold: `{args.low_change_threshold}`.",
        "",
        *table,
        "",
        "The stagnation-like group is characterized primarily by a larger "
        "low-change fraction and longer low-change runs. The oscillation/burst-like "
        "group is characterized primarily by larger total variation. Turning rate "
        "overlaps between the groups and should not be treated as a standalone "
        "separator. These thresholds are exploratory and were not cross-validated.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    data = data.loc[data["task"].eq("open_drawer")].copy()
    data["step_id"] = pd.to_numeric(data["step_id"], errors="raise").astype(int)
    data["feat_depth_change_ratio_prev"] = pd.to_numeric(
        data["feat_depth_change_ratio_prev"], errors="coerce"
    )
    summary = episode_features(data, args)
    summary_path = args.output_dir / "opendrawer_depth_dynamics_episode.csv"
    failure_path = args.output_dir / "opendrawer_depth_dynamics_failure_modes.csv"
    plot_path = args.output_dir / "opendrawer_depth_dynamics_classification.png"
    report_path = args.output_dir / "opendrawer_depth_dynamics_report.md"
    summary.to_csv(summary_path, index=False)
    summary.loc[summary["episode_failure"].eq(1)].to_csv(failure_path, index=False)
    plot_features(summary, plot_path)
    write_report(summary, args, report_path)
    print(summary.loc[summary["episode_failure"].eq(1)].to_string(index=False))
    print(summary_path)
    print(plot_path)


if __name__ == "__main__":
    main()
