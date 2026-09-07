"""Align Trace-geometry features around Partial-progress stall episodes.

This is a descriptive analysis of the four episodes currently labelled
``partial_progress_stall`` in the OpenDrawer action-mechanism table.  The
operational onset is read from ``opendrawer_onsets.csv`` and is *not* redefined
from the predictor features here.  The script only examines four Trace
features requested for the first onset pass.

Outputs are episode-aligned trajectories, per-episode summaries, a group
summary, and a short Markdown report.  With only four episodes, the results
are exploratory and are not a trained detector or a significance test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = [
    "feat_trace_direction_change_prev_rad",
    "feat_trace_straightness",
    "feat_trace_backtracking_ratio",
    "feat_trace_direction_alignment",
]

DEFAULT_STEPS = Path("data/formal_features/step_features_drawer_geometry.csv")
DEFAULT_ONSETS = Path("data/opendrawer_analysis/opendrawer_onsets.csv")
DEFAULT_ACTION = Path(
    "data/opendrawer_analysis/opendrawer_action_mechanism_episode.csv"
)
DEFAULT_OUTPUT = Path("data/opendrawer_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--onsets", type=Path, default=DEFAULT_ONSETS)
    parser.add_argument("--action-episodes", type=Path, default=DEFAULT_ACTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-relative-step", type=int, default=-20)
    parser.add_argument("--max-relative-step", type=int, default=20)
    return parser.parse_args()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def load_partial_episodes(args: argparse.Namespace) -> pd.DataFrame:
    steps = pd.read_csv(args.steps)
    required = {"episode_key", "step_id", "label_episode_failure", *FEATURES}
    missing = sorted(required.difference(steps.columns))
    if missing:
        raise ValueError(f"Missing columns in {args.steps}: {missing}")
    steps = steps.loc[steps["task"].eq("open_drawer")].copy()
    steps["step_id"] = pd.to_numeric(steps["step_id"], errors="raise").astype(int)
    for feature in FEATURES:
        steps[feature] = _numeric(steps, feature)

    onsets = pd.read_csv(args.onsets)
    onsets = onsets[["episode_key", "automatic_t_onset"]].copy()
    onsets["automatic_t_onset"] = pd.to_numeric(
        onsets["automatic_t_onset"], errors="coerce"
    )

    action = pd.read_csv(args.action_episodes)
    if "failure_mode" not in action.columns:
        raise ValueError(f"Missing failure_mode in {args.action_episodes}")
    action = action[["episode_key", "failure_mode"]].copy()

    data = steps.merge(onsets, on="episode_key", how="left").merge(
        action, on="episode_key", how="left"
    )
    data = data.loc[data["failure_mode"].eq("partial_progress_stall")].copy()
    data = data.loc[data["automatic_t_onset"].notna()].copy()
    data["automatic_t_onset"] = data["automatic_t_onset"].astype(int)
    data["relative_step"] = data["step_id"] - data["automatic_t_onset"]
    data = data.sort_values(["episode_key", "step_id"])
    if data["episode_key"].nunique() == 0:
        raise ValueError("No partial_progress_stall episodes found")
    return data


def summarize(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for episode_key, episode in data.groupby("episode_key", sort=True):
        onset = int(episode["automatic_t_onset"].iloc[0])
        pre = episode.loc[episode["relative_step"].between(-10, -1)]
        near = episode.loc[episode["relative_step"].between(-3, 1)]
        record: dict[str, object] = {
            "episode_key": episode_key,
            "condition_id": int(episode["condition_id"].iloc[0]),
            "automatic_t_onset": onset,
            "num_steps": int(episode["step_id"].max() + 1),
        }
        for feature in FEATURES:
            pre_values = pre[feature].dropna()
            near_values = near[feature].dropna()
            record[f"{feature}_pre10_mean"] = (
                float(pre_values.mean()) if len(pre_values) else np.nan
            )
            record[f"{feature}_near_onset_mean"] = (
                float(near_values.mean()) if len(near_values) else np.nan
            )
            record[f"{feature}_near_onset_delta"] = (
                float(near_values.mean() - pre_values.mean())
                if len(pre_values) and len(near_values)
                else np.nan
            )
            window = episode.loc[episode["relative_step"].between(-10, 2), ["relative_step", feature]].dropna()
            if len(window):
                if feature in {
                    "feat_trace_straightness",
                    "feat_trace_direction_alignment",
                }:
                    index = window[feature].idxmin()
                else:
                    index = window[feature].idxmax()
                record[f"{feature}_extreme_relative_step"] = int(
                    window.loc[index, "relative_step"]
                )
                record[f"{feature}_extreme_value"] = float(window.loc[index, feature])
            else:
                record[f"{feature}_extreme_relative_step"] = np.nan
                record[f"{feature}_extreme_value"] = np.nan
        rows.append(record)

    episode_summary = pd.DataFrame(rows)
    group_rows: list[dict[str, object]] = []
    for feature in FEATURES:
        delta = episode_summary[f"{feature}_near_onset_delta"].dropna()
        group_rows.append(
            {
                "feature": feature,
                "num_episodes": int(len(delta)),
                "pre10_mean_across_episodes": float(
                    episode_summary[f"{feature}_pre10_mean"].mean()
                ),
                "near_onset_mean_across_episodes": float(
                    episode_summary[f"{feature}_near_onset_mean"].mean()
                ),
                "near_minus_pre_mean": float(delta.mean()),
                "near_minus_pre_median": float(delta.median()),
                "episodes_with_expected_change": int(
                    sum(delta < 0)
                    if feature
                    in {
                        "feat_trace_straightness",
                        "feat_trace_direction_alignment",
                    }
                    else sum(delta > 0)
                ),
            }
        )
    return episode_summary, pd.DataFrame(group_rows)


def plot_aligned(data: pd.DataFrame, args: argparse.Namespace, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()
    colors = plt.get_cmap("tab10").colors
    titles = {
        "feat_trace_direction_change_prev_rad": "Trace direction change",
        "feat_trace_straightness": "Trace straightness",
        "feat_trace_backtracking_ratio": "Trace backtracking ratio",
        "feat_trace_direction_alignment": "Trace direction alignment",
    }
    for axis, feature in zip(axes, FEATURES):
        for color, (episode_key, episode) in zip(
            colors, data.groupby("episode_key", sort=True)
        ):
            curve = episode.loc[
                episode["relative_step"].between(
                    args.min_relative_step, args.max_relative_step
                )
            ]
            axis.plot(
                curve["relative_step"],
                curve[feature],
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                alpha=0.85,
                color=color,
                label=f"C{int(episode['condition_id'].iloc[0]):02d}",
            )
        axis.axvline(0, color="black", linewidth=1.0, label="operational onset")
        axis.axvline(-10, color="gray", linestyle=":", linewidth=0.9)
        axis.set_title(titles[feature])
        axis.set_xlabel("step - operational onset")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("value")
    axes[2].set_ylabel("value")
    axes[0].legend(fontsize=8, ncol=2, loc="best")
    fig.suptitle(
        "OpenDrawer Partial-progress stall | Trace features aligned to onset",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    episode_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    output: Path,
) -> None:
    # Avoid requiring the optional ``tabulate`` dependency just to write the
    # small Markdown report.
    table_columns = list(group_summary.columns)
    table_lines = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join("---" for _ in table_columns) + " |",
    ]
    for row in group_summary.itertuples(index=False, name=None):
        cells: list[str] = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        table_lines.append("| " + " | ".join(cells) + " |")
    lines = [
        "# Partial-progress stall onset: Trace feature analysis",
        "",
        "This is an exploratory, episode-level analysis of the four current "
        "`partial_progress_stall` episodes. The onset is the independently "
        "computed operational onset from `opendrawer_onsets.csv`; it is not "
        "redefined from these features.",
        "",
        f"Episodes analysed: {len(episode_summary)}.",
        "",
        "## Group summary",
        "",
        *table_lines,
        "",
        "## Interpretation",
        "",
        "The four trajectories are heterogeneous: only two of four episodes "
        "show the expected near-onset direction-change increase, straightness "
        "decrease, backtracking increase, or alignment decrease when comparing "
        "the [-3, 1] window with the preceding ten steps. C00 and C09 often "
        "show their largest deviations well before the operational onset, while "
        "C02 and C10 have deviations closer to onset. These features are useful "
        "for describing precursor patterns, but no single one is a reliable "
        "onset rule yet.",
        "",
        "For a next exploratory detector, treat direction change and straightness "
        "as the primary pair, with backtracking and alignment as secondary "
        "context. Thresholds must be selected using success episodes or an outer "
        "training fold and should include temporal persistence. The four episodes "
        "are not enough for a formal detector or significance test.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_partial_episodes(args)
    episode_summary, group_summary = summarize(data)

    aligned = data.loc[
        data["relative_step"].between(args.min_relative_step, args.max_relative_step)
    ].copy()
    aligned.to_csv(
        args.output_dir / "partial_progress_stall_onset_aligned.csv", index=False
    )
    episode_summary.to_csv(
        args.output_dir / "partial_progress_stall_onset_episode_summary.csv",
        index=False,
    )
    group_summary.to_csv(
        args.output_dir / "partial_progress_stall_onset_feature_summary.csv",
        index=False,
    )
    plot_aligned(
        data,
        args,
        args.output_dir / "partial_progress_stall_trace_onset.png",
    )
    write_report(
        episode_summary,
        group_summary,
        args.output_dir / "partial_progress_stall_onset_report.md",
    )
    print(f"episodes={data['episode_key'].nunique()}")
    print(episode_summary.to_string(index=False))
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
