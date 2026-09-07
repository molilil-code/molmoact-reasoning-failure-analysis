"""Analyze whether OpenDrawer failures are associated with unstable actions.

This is a descriptive mechanism analysis, not a causal identification.  It
separates Cartesian translation instability, rotational instability, and
command magnitude, then relates each to drawer qpos response.  Force/contact
signals are not present in the rollout logs, so command magnitude must not be
called physical force.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STEPS = Path("data/formal_features/step_features.csv")
DEFAULT_ONSETS = Path("data/opendrawer_analysis/opendrawer_onsets.csv")
DEFAULT_OUTPUT = Path("data/opendrawer_analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=Path, default=DEFAULT_STEPS)
    p.add_argument("--onsets", type=Path, default=DEFAULT_ONSETS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def read_steps(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int | None]]:
    df = pd.read_csv(args.steps)
    df = df.loc[df["task"].eq("open_drawer")].copy()
    df["step_id"] = pd.to_numeric(df["step_id"], errors="raise").astype(int)
    df["success"] = pd.to_numeric(df["label_episode_success"], errors="raise").astype(int)
    df = df.sort_values(["episode_key", "step_id"])
    for col in [
        "post_drawer_qpos",
        "label_drawer_progress",
        "feat_model_action_jump_prev_l2",
        "feat_model_translation_jump_prev_l2",
        "feat_model_rotation_jump_prev_l2",
        "feat_model_translation_cosine_prev",
        "feat_model_translation_direction_flip",
        "feat_model_translation_norm",
        "feat_model_rotation_norm",
        "feat_env_action_jump_prev_l2",
        "feat_env_translation_jump_prev_l2",
        "feat_env_rotation_jump_prev_l2",
        "feat_env_translation_cosine_prev",
        "feat_env_translation_direction_flip",
        "feat_env_translation_norm",
        "feat_env_rotation_norm",
    ]:
        if col in df:
            df[col] = _numeric(df, col)
    df["qpos_delta"] = df.groupby("episode_key")["post_drawer_qpos"].diff()
    df["progress_delta"] = df.groupby("episode_key")["label_drawer_progress"].diff()
    # Full action jump includes the gripper channel.  Keep a separate
    # translation+rotation jump so a gripper toggle is not mistaken for an
    # end-effector direction or magnitude change.
    df["model_motion_jump_no_gripper"] = np.sqrt(
        df["feat_model_translation_jump_prev_l2"].pow(2)
        + df["feat_model_rotation_jump_prev_l2"].pow(2)
    )
    df["env_motion_jump_no_gripper"] = np.sqrt(
        df["feat_env_translation_jump_prev_l2"].pow(2)
        + df["feat_env_rotation_jump_prev_l2"].pow(2)
    )
    onsets = pd.read_csv(args.onsets)
    onset_map: dict[str, int | None] = {}
    for row in onsets.itertuples(index=False):
        value = row.automatic_t_onset
        onset_map[row.episode_key] = None if pd.isna(value) else int(value)
    return df, onset_map


def _window(g: pd.DataFrame, onset: int | None, start_offset: int, end_offset: int) -> pd.DataFrame:
    if onset is None:
        return g.iloc[0:0]
    lo = onset + start_offset
    hi = onset + end_offset
    return g.loc[g["step_id"].between(lo, hi)]


def _summarize_window(g: pd.DataFrame, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    metrics = {
        "qpos_delta": "qpos_delta",
        "progress_delta": "progress_delta",
        "action_jump": "feat_model_action_jump_prev_l2",
        "motion_jump_no_gripper": "model_motion_jump_no_gripper",
        "translation_jump": "feat_model_translation_jump_prev_l2",
        "rotation_jump": "feat_model_rotation_jump_prev_l2",
        "gripper_change": "feat_model_gripper_change_prev",
        "translation_cosine": "feat_model_translation_cosine_prev",
        "direction_flip": "feat_model_translation_direction_flip",
        "translation_norm": "feat_model_translation_norm",
        "rotation_norm": "feat_model_rotation_norm",
        "env_action_jump": "feat_env_action_jump_prev_l2",
        "env_motion_jump_no_gripper": "env_motion_jump_no_gripper",
    }
    for name, col in metrics.items():
        values = _numeric(g, col).dropna()
        if values.empty:
            out[f"{prefix}_{name}_mean"] = np.nan
            out[f"{prefix}_{name}_p90"] = np.nan
        else:
            out[f"{prefix}_{name}_mean"] = float(values.mean())
            out[f"{prefix}_{name}_p90"] = float(values.quantile(0.9))
    qd = _numeric(g, "qpos_delta").dropna()
    out[f"{prefix}_qpos_stall_rate"] = float((qd.abs() < 1e-4).mean()) if len(qd) else np.nan
    out[f"{prefix}_qpos_positive_rate"] = float((qd > 1e-4).mean()) if len(qd) else np.nan
    # This is an empirical command-to-motion ratio, not a force estimate.
    tn = _numeric(g, "feat_model_translation_norm").dropna()
    motion = qd.clip(lower=0).sum() if len(qd) else np.nan
    out[f"{prefix}_positive_qpos_per_translation"] = (
        float(motion / tn.sum()) if len(tn) and tn.sum() > 0 else np.nan
    )
    out[f"{prefix}_n_steps"] = float(len(g))
    return out


def build_episode_table(df: pd.DataFrame, onset_map: dict[str, int | None]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_key, g in df.groupby("episode_key", sort=False):
        g = g.sort_values("step_id")
        condition_id = int(g["condition_id"].iloc[0])
        success = int(g["success"].iloc[0])
        onset = onset_map.get(episode_key)
        row: dict[str, object] = {
            "episode_key": episode_key,
            "condition_id": condition_id,
            "success": success,
            "failure": 1 - success,
            "failure_mode": (
                "success"
                if success
                else "no_progress"
                if float(_numeric(g, "label_drawer_progress").max()) < 0.01
                else "partial_progress_stall"
            ),
            "n_steps": len(g),
            "onset": onset,
            "max_qpos": float(_numeric(g, "post_drawer_qpos").max()),
            "max_progress": float(_numeric(g, "label_drawer_progress").max()),
        }
        row.update(_summarize_window(g, "all"))
        if onset is not None:
            row.update(_summarize_window(_window(g, onset, -20, -11), "baseline"))
            row.update(_summarize_window(_window(g, onset, -10, -1), "pre_onset"))
            row.update(_summarize_window(_window(g, onset, 0, 9), "post_onset"))
        else:
            row.update(_summarize_window(g.iloc[0:0], "baseline"))
            row.update(_summarize_window(g.iloc[0:0], "pre_onset"))
            row.update(_summarize_window(g.iloc[0:0], "post_onset"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("condition_id")


def build_step_table(df: pd.DataFrame, onset_map: dict[str, int | None]) -> pd.DataFrame:
    out = df.copy()
    out["onset"] = out["episode_key"].map(onset_map)
    out["relative_onset"] = out["step_id"] - out["onset"]
    out["phase"] = np.where(
        out["onset"].notna() & out["relative_onset"].between(-10, -1),
        "failure_pre_onset",
        np.where(out["onset"].notna() & out["relative_onset"].between(-20, -11), "failure_baseline", "other"),
    )
    keep = [
        "episode_key", "condition_id", "step_id", "success", "onset", "relative_onset", "phase",
        "post_drawer_qpos", "qpos_delta", "label_drawer_progress", "progress_delta",
        "feat_model_action_jump_prev_l2", "feat_model_translation_jump_prev_l2",
        "feat_model_rotation_jump_prev_l2", "feat_model_translation_cosine_prev",
        "feat_model_translation_direction_flip", "feat_model_gripper_change_prev",
        "feat_model_translation_norm", "feat_model_rotation_norm",
        "model_motion_jump_no_gripper",
        "feat_env_action_jump_prev_l2", "feat_env_translation_jump_prev_l2", "feat_env_rotation_jump_prev_l2",
    ]
    return out[[c for c in keep if c in out.columns]]


def plot_outputs(episode: pd.DataFrame, step: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = np.where(episode["success"].to_numpy() == 1, "#777777", "#D1495B")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), squeeze=False)
    panels = [
        ("all_motion_jump_no_gripper_p90", "max_progress", "Motion jump p90 (translation+rotation)", "Maximum drawer progress"),
        ("all_direction_flip_mean", "max_progress", "Translation direction-flip rate", "Maximum drawer progress"),
        ("all_rotation_jump_p90", "max_progress", "Rotation jump p90", "Maximum drawer progress"),
        ("all_gripper_change_mean", "max_progress", "Gripper-change rate", "Maximum drawer progress"),
    ]
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flat, panels):
        x = pd.to_numeric(episode[xcol], errors="coerce")
        y = pd.to_numeric(episode[ycol], errors="coerce")
        ax.scatter(x, y, c=colors, s=58, edgecolor="white", linewidth=0.7)
        for _, r in episode.iterrows():
            if pd.notna(r[xcol]) and pd.notna(r[ycol]):
                ax.annotate(f"C{int(r.condition_id):02d}", (r[xcol], r[ycol]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle("OpenDrawer action instability vs. final drawer progress", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "opendrawer_action_mechanism_episode_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    failures = step.loc[step["phase"].eq("failure_pre_onset")].copy()
    if failures.empty:
        return
    failures["relative_onset"] = pd.to_numeric(failures["relative_onset"], errors="coerce")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, squeeze=False)
    lines = [
        ("post_drawer_qpos", "drawer qpos"),
        ("feat_model_action_jump_prev_l2", "model action jump (includes gripper)"),
        ("model_motion_jump_no_gripper", "motion jump (translation+rotation)"),
        ("feat_model_translation_direction_flip", "translation direction flip"),
        ("feat_model_rotation_jump_prev_l2", "rotation jump"),
        ("feat_model_gripper_change_prev", "gripper change"),
    ]
    for ax, (col, ylabel) in zip(axes.flat, lines):
        for _, g in failures.groupby("episode_key", sort=False):
            g = g.sort_values("relative_onset")
            ax.plot(g["relative_onset"], _numeric(g, col), marker="o", markersize=2.5, linewidth=1.0, alpha=0.8, label=f"C{int(g.condition_id.iloc[0]):02d}")
        ax.axvline(0, color="#D1495B", linewidth=1.0)
        ax.axvline(-10, color="#F28E2B", linestyle=":", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel("relative step (0 = operational onset)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=4, frameon=False)
    fig.suptitle("Failure episodes: signals around the qpos-stagnation onset", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_dir / "opendrawer_action_mechanism_pre_onset.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, onset_map = read_steps(args)
    episode = build_episode_table(df, onset_map)
    step = build_step_table(df, onset_map)
    episode.to_csv(args.output_dir / "opendrawer_action_mechanism_episode.csv", index=False)
    step.to_csv(args.output_dir / "opendrawer_action_mechanism_steps.csv", index=False)
    group_cols = ["success"]
    summary = episode.groupby(group_cols, dropna=False).agg({
        "max_progress": ["count", "median"],
        "all_action_jump_p90": "median",
        "all_motion_jump_no_gripper_p90": "median",
        "all_translation_jump_p90": "median",
        "all_rotation_jump_p90": "median",
        "all_direction_flip_mean": "median",
        "all_gripper_change_mean": "median",
        "all_translation_norm_p90": "median",
        "all_qpos_positive_rate": "median",
        "all_qpos_stall_rate": "median",
        "pre_onset_action_jump_p90": "median",
        "pre_onset_motion_jump_no_gripper_p90": "median",
        "pre_onset_translation_jump_p90": "median",
        "pre_onset_rotation_jump_p90": "median",
        "pre_onset_direction_flip_mean": "median",
        "pre_onset_gripper_change_mean": "median",
        "pre_onset_qpos_stall_rate": "median",
    })
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "opendrawer_action_mechanism_group_summary.csv", index=False)
    mode_summary = episode.groupby("failure_mode", dropna=False).agg({
        "condition_id": "count",
        "max_progress": "median",
        "all_motion_jump_no_gripper_p90": "median",
        "all_rotation_jump_p90": "median",
        "all_direction_flip_mean": "median",
        "all_gripper_change_mean": "median",
        "all_qpos_stall_rate": "median",
        "pre_onset_motion_jump_no_gripper_p90": "median",
        "pre_onset_rotation_jump_p90": "median",
        "pre_onset_direction_flip_mean": "median",
        "pre_onset_gripper_change_mean": "median",
        "pre_onset_qpos_stall_rate": "median",
    }).rename(columns={"condition_id": "n_episodes"})
    mode_summary.reset_index().to_csv(
        args.output_dir / "opendrawer_action_mechanism_failure_mode_summary.csv",
        index=False,
    )
    plot_outputs(episode, step, args.output_dir)
    print(f"wrote mechanism tables and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
