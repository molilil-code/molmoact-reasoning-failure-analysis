"""Plot raw OpenDrawer task-state and reasoning trajectories.

The plots are descriptive.  They do not use risk scores to define onset or
choose a threshold.  Failure onset markers are the independently constructed
progress-stagnation labels from ``opendrawer_onsets.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STEP_FEATURES = Path("data/formal_features/step_features.csv")
DEFAULT_EPISODE_FEATURES = Path("data/formal_features/episode_features.csv")
DEFAULT_ONSETS = Path("data/opendrawer_analysis/opendrawer_onsets.csv")
DEFAULT_OUTPUT_DIR = Path("data/opendrawer_analysis")


TRAJECTORIES = {
    "qpos": {
        "column": "post_drawer_qpos",
        "title": "OpenDrawer drawer qpos trajectories",
        "ylabel": "drawer qpos",
        "filename": "opendrawer_qpos_trajectories.png",
    },
    "temporal_change_ratio": {
        "column": "feat_depth_change_ratio_prev",
        "title": "OpenDrawer temporal depth change ratio",
        "ylabel": "Depth change ratio vs. previous step",
        "filename": "opendrawer_temporal_change_ratio_trajectories.png",
    },
    "num_points": {
        "column": "feat_trace_num_points",
        "title": "OpenDrawer Trace point-count trajectories",
        "ylabel": "Trace points",
        "filename": "opendrawer_trace_num_points_trajectories.png",
    },
    "endpoint_drift": {
        "column": "feat_trace_endpoint_drift_prev",
        "title": "OpenDrawer Trace endpoint-drift trajectories",
        "ylabel": "Endpoint drift (pixels)",
        "filename": "opendrawer_endpoint_drift_trajectories.png",
    },
    "model_action_jump": {
        "column": "feat_model_action_jump_prev_l2",
        "title": "OpenDrawer model-action jump trajectories",
        "ylabel": "Model action jump (L2)",
        "filename": "opendrawer_model_action_jump_trajectories.png",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-features", type=Path, default=DEFAULT_STEP_FEATURES)
    parser.add_argument("--episode-features", type=Path, default=DEFAULT_EPISODE_FEATURES)
    parser.add_argument("--onsets", type=Path, default=DEFAULT_ONSETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    steps = pd.read_csv(args.step_features)
    steps = steps.loc[steps["task"].eq("open_drawer")].copy()
    steps["step_id"] = pd.to_numeric(steps["step_id"], errors="raise").astype(int)
    steps["label_episode_success"] = pd.to_numeric(
        steps["label_episode_success"], errors="raise"
    ).astype(int)
    onsets = pd.read_csv(args.onsets)
    episodes = pd.read_csv(args.episode_features)
    episodes = episodes.loc[episodes["task"].eq("open_drawer")].copy()
    return steps.sort_values(["condition_id", "step_id"]), episodes, onsets


def _episode_title(episode: pd.DataFrame) -> str:
    condition_id = int(episode["condition_id"].iloc[0])
    success = bool(int(episode["label_episode_success"].iloc[0]))
    return f"C{condition_id:02d} | {'success' if success else 'failure'}"


def _onset_map(onsets: pd.DataFrame) -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    for row in onsets.itertuples(index=False):
        raw = row.automatic_t_onset
        values[row.episode_key] = None if pd.isna(raw) or raw == "" else int(raw)
    return values


def _global_ylim(steps: pd.DataFrame, column: str) -> tuple[float, float]:
    """Return one shared y-axis range for all episode panels of a metric."""
    values = pd.to_numeric(steps[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if values.empty:
        return (0.0, 1.0)
    lo = float(values.min())
    hi = float(values.max())
    # These task-state/reasoning quantities are non-negative.  Anchoring at
    # zero keeps panels comparable instead of letting each episode autoscale.
    lo = min(0.0, lo)
    if hi <= lo:
        pad = 0.5 if hi == 0 else abs(hi) * 0.1
        return (lo - pad, hi + pad)
    pad = 0.05 * (hi - lo)
    return (lo, hi + pad)


def plot_trajectory(
    steps: pd.DataFrame,
    onsets: pd.DataFrame,
    spec: dict[str, str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    column = spec["column"]
    if column not in steps.columns:
        raise ValueError(f"Missing requested trajectory column: {column}")
    onset_by_episode = _onset_map(onsets)
    episodes = [g for _, g in steps.groupby("episode_key", sort=False)]
    y_limits = _global_ylim(steps, column)
    ncols = 4
    nrows = int(np.ceil(len(episodes) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, max(8, nrows * 2.6)),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    axes_flat = axes.flat
    for ax, episode in zip(axes_flat, episodes):
        episode = episode.sort_values("step_id")
        x = episode["step_id"].to_numpy(dtype=int)
        y = pd.to_numeric(episode[column], errors="coerce").to_numpy(dtype=float)
        success = bool(int(episode["label_episode_success"].iloc[0]))
        color = "#777777" if success else "#D1495B"
        ax.plot(x, y, color=color, linewidth=1.35)
        onset = onset_by_episode.get(episode["episode_key"].iloc[0])
        if onset is not None:
            ax.axvline(onset, color="#D1495B", linestyle="-", linewidth=0.85, alpha=0.9)
            ax.axvline(onset - 10, color="#F28E2B", linestyle=":", linewidth=0.8, alpha=0.9)
        ax.set_title(_episode_title(episode), fontsize=9)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel(spec["ylabel"], fontsize=8)
        ax.set_ylim(*y_limits)
    for ax in list(axes_flat)[len(episodes) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"{spec['title']} | red=operational onset, orange=onset−10",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_episode_maxima(episodes: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    required = {"condition_id", "label_episode_success", "label_drawer_max_qpos", "label_drawer_max_progress"}
    missing = required.difference(episodes.columns)
    if missing:
        raise ValueError(f"Missing episode summary columns: {sorted(missing)}")
    data = episodes.sort_values("condition_id").copy()
    x = data["condition_id"].to_numpy(dtype=int)
    success = data["label_episode_success"].to_numpy(dtype=int)
    colors = np.where(success == 1, "#777777", "#D1495B")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    specs = [
        ("label_drawer_max_qpos", "Maximum drawer qpos", "max qpos"),
        ("label_drawer_max_progress", "Maximum normalized drawer progress", "max progress [0, 1]"),
    ]
    for ax, (column, title, ylabel) in zip(axes, specs):
        y = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        ax.scatter(x, y, c=colors, s=48, edgecolor="white", linewidth=0.7, zorder=3)
        ax.plot(x, y, color="#aaaaaa", linewidth=0.7, alpha=0.55, zorder=1)
        ax.set_title(title)
        ax.set_xlabel("condition id")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#D1495B", markersize=7, label="failure"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777", markersize=7, label="success"),
    ]
    fig.suptitle("OpenDrawer episode-level drawer maxima", fontsize=14, y=0.995)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps, episodes, onsets = read_inputs(args)
    for spec in TRAJECTORIES.values():
        plot_trajectory(steps, onsets, spec, args.output_dir / spec["filename"])
    plot_episode_maxima(episodes, args.output_dir / "opendrawer_drawer_max_qpos_progress.png")
    print(f"wrote {len(TRAJECTORIES) + 1} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
