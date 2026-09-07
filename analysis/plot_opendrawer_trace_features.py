"""Plot Trace geometry trajectories for all OpenDrawer episodes.

The plot is descriptive: it uses the existing extracted Trace features and
the independently computed operational onsets. It does not select a
predictor threshold or redefine an onset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FEATURES = {
    "feat_trace_direction_change_prev_rad": "Trace direction change (rad)",
    "feat_trace_straightness": "Trace straightness",
    "feat_trace_backtracking_ratio": "Trace backtracking ratio",
}

DEFAULT_STEPS = Path("data/formal_features/step_features_drawer_geometry.csv")
DEFAULT_ONSETS = Path("data/opendrawer_analysis/opendrawer_onsets.csv")
DEFAULT_OUTPUT = Path("data/opendrawer_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--onsets", type=Path, default=DEFAULT_ONSETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    data = pd.read_csv(args.steps)
    data = data.loc[data["task"].eq("open_drawer")].copy()
    data["step_id"] = pd.to_numeric(data["step_id"], errors="raise").astype(int)
    data["success"] = pd.to_numeric(
        data["label_episode_success"], errors="raise"
    ).astype(int)
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    onsets = pd.read_csv(args.onsets)[["episode_key", "automatic_t_onset"]]
    onsets["automatic_t_onset"] = pd.to_numeric(
        onsets["automatic_t_onset"], errors="coerce"
    )
    data = data.merge(onsets, on="episode_key", how="left")
    return data.sort_values(["episode_key", "step_id"])


def padded_limits(data: pd.DataFrame, feature: str) -> tuple[float, float]:
    values = data[feature].dropna().to_numpy(dtype=float)
    low, high = float(np.min(values)), float(np.max(values))
    pad = max((high - low) * 0.06, 0.02)
    return low - pad, high + pad


def plot_overlay(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    failure_color = "#d1495b"
    success_color = "#6f6f6f"
    onset_color = "#e67e22"

    for axis, (feature, title) in zip(axes, FEATURES.items()):
        for _, episode in data.groupby("episode_key", sort=True):
            success = bool(episode["success"].iloc[0])
            color = success_color if success else failure_color
            axis.plot(
                episode["step_id"],
                episode[feature],
                color=color,
                linewidth=0.9 if success else 1.25,
                alpha=0.42 if success else 0.85,
            )
            if not success and pd.notna(episode["automatic_t_onset"].iloc[0]):
                onset = int(episode["automatic_t_onset"].iloc[0])
                axis.axvline(onset, color=onset_color, alpha=0.16, linewidth=0.8)
                onset_row = episode.loc[episode["step_id"].eq(onset)]
                if not onset_row.empty and pd.notna(onset_row[feature].iloc[0]):
                    axis.scatter(
                        [onset],
                        [onset_row[feature].iloc[0]],
                        color=onset_color,
                        s=18,
                        zorder=4,
                    )
        axis.set_title(title)
        axis.set_ylabel("value")
        axis.set_ylim(*padded_limits(data, feature))
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("step")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=success_color, linewidth=1.5, label="success"),
            Line2D([0], [0], color=failure_color, linewidth=1.8, label="failure"),
            Line2D(
                [0],
                [0],
                color=onset_color,
                marker="o",
                linewidth=1.0,
                label="failure operational onset",
            ),
        ],
        loc="upper right",
        ncol=3,
    )
    fig.suptitle("OpenDrawer Trace feature trajectories | all episodes", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_facets(data: pd.DataFrame, output: Path) -> None:
    episodes = list(data.groupby("episode_key", sort=True))
    ncols = 3
    nrows = len(episodes)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, max(2.0 * nrows, 8)),
        squeeze=False,
        sharex=False,
    )
    colors = ["#d1495b" if not bool(ep["success"].iloc[0]) else "#6f6f6f" for _, ep in episodes]
    for row, ((episode_key, episode), color) in enumerate(zip(episodes, colors)):
        condition = int(episode["condition_id"].iloc[0])
        outcome = "success" if bool(episode["success"].iloc[0]) else "failure"
        for col, feature in enumerate(FEATURES):
            axis = axes[row, col]
            axis.plot(episode["step_id"], episode[feature], color=color, linewidth=1.0)
            if outcome == "failure" and pd.notna(episode["automatic_t_onset"].iloc[0]):
                onset = int(episode["automatic_t_onset"].iloc[0])
                axis.axvline(onset, color="#e67e22", alpha=0.65, linestyle="--", linewidth=0.8)
            if row == 0:
                axis.set_title(FEATURES[feature])
            if col == 0:
                axis.set_ylabel(f"C{condition:02d}\n{outcome}")
            axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("step")
    fig.suptitle("OpenDrawer Trace feature trajectories | episode facets", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    overlay = args.output_dir / "opendrawer_trace_features_all_episodes.png"
    facets = args.output_dir / "opendrawer_trace_features_episode_facets.png"
    plot_overlay(data, overlay)
    plot_facets(data, facets)
    print(f"episodes={data['episode_key'].nunique()}")
    print(f"success={int(data.groupby('episode_key')['success'].first().sum())}")
    print(f"failure={int(data.groupby('episode_key')['success'].first().eq(0).sum())}")
    print(overlay)
    print(facets)


if __name__ == "__main__":
    main()
