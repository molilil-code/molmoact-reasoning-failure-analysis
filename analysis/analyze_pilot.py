#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_STEP = "/root/autodl-tmp/molmoact_project/data/derived/pilot_features/step_features.csv"
DEFAULT_EPISODE = "/root/autodl-tmp/molmoact_project/data/derived/pilot_features/episode_features.csv"
DEFAULT_OUTPUT = "/root/autodl-tmp/molmoact_project/data/derived/pilot_analysis"

CURVE_FEATURES = [
    "feat_depth_change_ratio_prev",
    "feat_trace_num_points",
    "feat_trace_single_point",
    "feat_trace_shape_drift_prev",
    "feat_model_action_jump_prev_l2",
]

CURVE_LABELS = {
    "feat_depth_change_ratio_prev": "Depth token change ratio vs previous step",
    "feat_trace_num_points": "Number of generated trace points",
    "feat_trace_single_point": "Single-point trace indicator",
    "feat_trace_shape_drift_prev": "Trace shape drift vs previous step",
    "feat_model_action_jump_prev_l2": "Model action jump L2 vs previous step",
}

EP_SPECS = {
    "feat_depth_change_ratio_prev": ["mean", "p90"],
    "feat_trace_endpoint_drift_prev": ["mean", "p90"],
    "feat_trace_shape_drift_prev": ["mean", "p90"],
    "feat_model_action_jump_prev_l2": ["mean", "p90"],
    "feat_model_translation_jump_prev_l2": ["mean", "p90"],
}


def parse_args():
    p = argparse.ArgumentParser(description="Pilot analysis for MolmoAct failure precursor features.")
    p.add_argument("--step-features", default=DEFAULT_STEP)
    p.add_argument("--episode-features", default=DEFAULT_EPISODE)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--degenerate-run", type=int, default=3)
    a = p.parse_args()
    if a.bins < 5:
        p.error("--bins must be >= 5")
    if a.bootstrap < 100:
        p.error("--bootstrap must be >= 100")
    if a.degenerate_run < 1:
        p.error("--degenerate-run must be >= 1")
    return a


def require(df: pd.DataFrame, cols: Sequence[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_mean(s: pd.Series) -> float:
    s = numeric(s).dropna()
    return float(s.mean()) if len(s) else math.nan


def safe_p90(s: pd.Series) -> float:
    s = numeric(s).dropna()
    return float(np.quantile(s, 0.90)) if len(s) else math.nan


def longest_streak(values) -> int:
    best = cur = 0
    for v in values:
        if int(v) == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def first_run(values, run_len: int) -> Optional[int]:
    cur = 0
    for i, v in enumerate(values):
        if int(v) == 1:
            cur += 1
            if cur >= run_len:
                return i - run_len + 1
        else:
            cur = 0
    return None


def bootstrap_diff(sv, fv, n_boot, rng) -> Tuple[float, float, float]:
    sv = np.asarray(sv, dtype=float)
    fv = np.asarray(fv, dtype=float)
    sv = sv[np.isfinite(sv)]
    fv = fv[np.isfinite(fv)]
    if len(sv) == 0 or len(fv) == 0:
        return math.nan, math.nan, math.nan

    effect = float(fv.mean() - sv.mean())
    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sb = rng.choice(sv, size=len(sv), replace=True)
        fb = rng.choice(fv, size=len(fv), replace=True)
        boot[i] = fb.mean() - sb.mean()

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return effect, float(lo), float(hi)


def load_data(step_path: Path, ep_path: Path):
    step = pd.read_csv(step_path)
    ep = pd.read_csv(ep_path)

    require(
        step,
        [
            "task", "condition_id", "episode_key", "step_id",
            "label_episode_success", "label_episode_failure",
            "label_episode_num_steps", "feat_depth_token_count",
            "feat_trace_num_points",
        ],
        "step_features.csv",
    )
    require(
        ep,
        [
            "task", "condition_id", "episode_key",
            "label_episode_success", "label_episode_failure",
        ],
        "episode_features.csv",
    )

    for c in ["condition_id", "step_id", "label_episode_success", "label_episode_failure"]:
        step[c] = numeric(step[c]).astype(int)

    for c in ["condition_id", "label_episode_success", "label_episode_failure"]:
        ep[c] = numeric(ep[c]).astype(int)

    step = step.sort_values(["task", "condition_id", "step_id"]).reset_index(drop=True)

    step["feat_trace_single_point"] = (numeric(step["feat_trace_num_points"]) == 1).astype(int)
    step["feat_trace_missing"] = (numeric(step["feat_trace_num_points"]) == 0).astype(int)

    denom = (numeric(step["label_episode_num_steps"]) - 1).clip(lower=1)
    step["analysis_time_frac"] = (step["step_id"] / denom).clip(0, 1)

    # Online-safe consecutive single-point trace streak.
    streak = np.zeros(len(step), dtype=int)
    for _, idx in step.groupby("episode_key", sort=False).groups.items():
        cur = 0
        for pos in list(idx):
            if int(step.at[pos, "feat_trace_single_point"]) == 1:
                cur += 1
            else:
                cur = 0
            streak[pos] = cur
    step["feat_trace_degenerate_streak"] = streak

    return step, ep


def build_episode_metrics(step: pd.DataFrame, ep: pd.DataFrame, run_len: int):
    lookup = ep.drop_duplicates("episode_key").set_index("episode_key")
    rows: List[Dict[str, object]] = []

    for key, g in step.groupby("episode_key", sort=True):
        g = g.sort_values("step_id")
        task = str(g["task"].iloc[0])
        cid = int(g["condition_id"].iloc[0])
        success = int(g["label_episode_success"].iloc[0])
        failure = int(g["label_episode_failure"].iloc[0])
        n_steps = int(g["label_episode_num_steps"].iloc[0])

        row: Dict[str, object] = {
            "task": task,
            "condition_id": cid,
            "episode_key": key,
            "success": success,
            "failure": failure,
            "num_steps": n_steps,
        }

        for feat, aggs in EP_SPECS.items():
            if feat not in g.columns:
                continue
            for agg in aggs:
                row[f"ep_{agg}__{feat}"] = safe_mean(g[feat]) if agg == "mean" else safe_p90(g[feat])

        single = g["feat_trace_single_point"].to_numpy(dtype=int)
        row["ep_trace_single_point_fraction"] = float(single.mean())
        row["ep_trace_single_point_max_streak"] = longest_streak(single)

        start_pos = first_run(single, run_len)
        if start_pos is None:
            row["ep_trace_sustained_degeneracy"] = 0
            row["ep_trace_degeneracy_onset_step"] = math.nan
            row["ep_trace_degeneracy_onset_frac"] = math.nan
            row["ep_trace_degeneracy_lead_to_end_steps"] = math.nan
        else:
            onset_step = int(g.iloc[start_pos]["step_id"])
            row["ep_trace_sustained_degeneracy"] = 1
            row["ep_trace_degeneracy_onset_step"] = onset_step
            row["ep_trace_degeneracy_onset_frac"] = onset_step / max(n_steps - 1, 1)
            row["ep_trace_degeneracy_lead_to_end_steps"] = n_steps - 1 - onset_step

        if key in lookup.index:
            e = lookup.loc[key]
            for col in [
                "label_drawer_max_qpos",
                "label_drawer_max_progress",
                "label_coke_ever_grasped",
                "label_coke_ever_lifted",
                "label_coke_ever_lifted_significantly",
                "label_coke_max_stage",
                "label_stop_reason",
                "label_elapsed_s",
            ]:
                if col in e.index:
                    row[col] = e[col]

        rows.append(row)

    return pd.DataFrame(rows)


def task_outcomes(epm: pd.DataFrame):
    rows = []
    for task, g in epm.groupby("task"):
        n = len(g)
        s = int(g["success"].sum())
        rows.append(
            {
                "task": task,
                "N": n,
                "success": s,
                "failure": n - s,
                "success_rate": s / n if n else math.nan,
            }
        )
    return pd.DataFrame(rows)


def trace_degeneracy_summary(epm: pd.DataFrame):
    rows = []
    for (task, success), g in epm.groupby(["task", "success"]):
        sustained = numeric(g["ep_trace_sustained_degeneracy"]).fillna(0)
        rows.append(
            {
                "task": task,
                "success": int(success),
                "N_episodes": len(g),
                "episodes_with_sustained_trace_degeneracy": int(sustained.sum()),
                "fraction_with_sustained_trace_degeneracy": float(sustained.mean()),
                "mean_single_point_fraction": safe_mean(g["ep_trace_single_point_fraction"]),
                "mean_max_single_point_streak": safe_mean(g["ep_trace_single_point_max_streak"]),
                "mean_degeneracy_lead_to_end_steps": safe_mean(g["ep_trace_degeneracy_lead_to_end_steps"]),
            }
        )
    return pd.DataFrame(rows)


def success_failure_comparison(epm: pd.DataFrame, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    metrics = [
        c for c in epm.columns
        if c.startswith("ep_mean__")
        or c.startswith("ep_p90__")
        or c in {
            "ep_trace_single_point_fraction",
            "ep_trace_single_point_max_streak",
        }
    ]

    rows = []

    for task, g in epm.groupby("task"):
        s = g[g["success"] == 1]
        f = g[g["success"] == 0]

        for metric in metrics:
            sv = numeric(s[metric]).to_numpy(dtype=float)
            fv = numeric(f[metric]).to_numpy(dtype=float)
            sv = sv[np.isfinite(sv)]
            fv = fv[np.isfinite(fv)]

            effect, lo, hi = bootstrap_diff(sv, fv, n_boot, rng)

            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "n_success": len(sv),
                    "n_failure": len(fv),
                    "success_mean": float(sv.mean()) if len(sv) else math.nan,
                    "failure_mean": float(fv.mean()) if len(fv) else math.nan,
                    "failure_minus_success": effect,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "descriptive_only_small_n": int(min(len(sv), len(fv)) < 3),
                }
            )

    return pd.DataFrame(rows)


def normalized_curves(step: pd.DataFrame, bins: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    work = step.copy()
    work["analysis_time_bin"] = np.minimum(
        (work["analysis_time_frac"] * bins).astype(int),
        bins - 1,
    )

    rows = []

    for feat in [f for f in CURVE_FEATURES if f in work.columns]:
        tmp = work[
            [
                "task", "episode_key", "label_episode_success",
                "analysis_time_bin", feat,
            ]
        ].copy()

        tmp[feat] = numeric(tmp[feat])

        per_ep = (
            tmp.groupby(
                [
                    "task", "episode_key", "label_episode_success",
                    "analysis_time_bin",
                ],
                as_index=False,
            )[feat]
            .mean()
        )

        for (task, success, b), g in per_ep.groupby(
            ["task", "label_episode_success", "analysis_time_bin"]
        ):
            vals = numeric(g[feat]).dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue

            boot = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                boot[i] = rng.choice(vals, size=len(vals), replace=True).mean()

            lo, hi = np.quantile(boot, [0.025, 0.975])

            rows.append(
                {
                    "task": task,
                    "feature": feat,
                    "success": int(success),
                    "time_bin": int(b),
                    "time_frac_center": (int(b) + 0.5) / bins,
                    "N_episodes": len(vals),
                    "mean": float(vals.mean()),
                    "bootstrap_ci_low": float(lo),
                    "bootstrap_ci_high": float(hi),
                }
            )

    return pd.DataFrame(rows)


def plot_curves(curves: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    if curves.empty:
        return

    for (task, feat), g in curves.groupby(["task", "feature"]):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))

        for success, gg in g.groupby("success"):
            gg = gg.sort_values("time_frac_center")
            label = "Success episodes" if int(success) == 1 else "Failure episodes"
            ax.plot(
                gg["time_frac_center"],
                gg["mean"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=label,
            )

        ax.set_xlabel("Normalized episode time")
        ax.set_ylabel(CURVE_LABELS.get(feat, feat))
        ax.set_title(f"{task}: {CURVE_LABELS.get(feat, feat)}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{task}__{feat}.png", dpi=180)
        plt.close(fig)


def plot_drawer_progress(step: pd.DataFrame, fig_dir: Path, bins: int):
    if "label_drawer_progress" not in step.columns:
        return

    d = step[step["task"] == "open_drawer"].copy()
    if d.empty:
        return

    d["label_drawer_progress"] = numeric(d["label_drawer_progress"])
    d = d.dropna(subset=["label_drawer_progress"])
    if d.empty:
        return

    d["analysis_time_bin"] = np.minimum(
        (d["analysis_time_frac"] * bins).astype(int),
        bins - 1,
    )

    per_ep = (
        d.groupby(
            ["episode_key", "label_episode_success", "analysis_time_bin"],
            as_index=False,
        )["label_drawer_progress"]
        .mean()
    )

    grouped = (
        per_ep.groupby(
            ["label_episode_success", "analysis_time_bin"],
            as_index=False,
        )["label_drawer_progress"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for success, g in grouped.groupby("label_episode_success"):
        label = "Success episodes" if int(success) == 1 else "Failure episodes"
        ax.plot(
            (g["analysis_time_bin"] + 0.5) / bins,
            g["label_drawer_progress"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
        )

    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="Success threshold")
    ax.set_xlabel("Normalized episode time")
    ax.set_ylabel("Drawer progress (qpos / 0.15, clipped)")
    ax.set_title("Open Drawer: task progress over episode")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "drawer__progress.png", dpi=180)
    plt.close(fig)


def plot_condition3(step: pd.DataFrame, fig_dir: Path):
    case = step[
        (step["task"] == "pick_coke_can")
        & (step["condition_id"] == 3)
    ].copy()

    if case.empty:
        return

    case = case.sort_values("step_id")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(
        case["step_id"],
        case["feat_trace_num_points"],
        marker="o",
        markersize=2.5,
        linewidth=1.2,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Generated trace points")
    ax.set_title("Coke condition 0003: trace-point degeneration case")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "coke_condition_0003__trace_points.png", dpi=180)
    plt.close(fig)

    feat = "feat_model_action_jump_prev_l2"
    if feat in case.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        ax.plot(
            case["step_id"],
            numeric(case[feat]),
            marker="o",
            markersize=2.5,
            linewidth=1.2,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Model action jump L2")
        ax.set_title("Coke condition 0003: action temporal change")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "coke_condition_0003__action_jump.png", dpi=180)
        plt.close(fig)


def drawer_table(epm: pd.DataFrame):
    d = epm[epm["task"] == "open_drawer"].copy()
    wanted = [
        "task", "condition_id", "episode_key", "success", "failure",
        "num_steps", "label_drawer_max_qpos", "label_drawer_max_progress",
        "ep_trace_single_point_fraction",
        "ep_trace_single_point_max_streak",
        "ep_trace_sustained_degeneracy",
        "ep_trace_degeneracy_onset_step",
        "ep_trace_degeneracy_lead_to_end_steps",
    ]
    return d[[c for c in wanted if c in d.columns]]


def write_report(
    path: Path,
    outcomes: pd.DataFrame,
    trace_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    step: pd.DataFrame,
    run_len: int,
):
    lines: List[str] = []

    lines += [
        "MolmoAct Pilot Analysis",
        "=" * 72,
        "",
        "Exploratory only. Timesteps are not treated as independent samples.",
        "",
        "Task outcomes",
        "-" * 72,
    ]

    for _, r in outcomes.iterrows():
        lines.append(
            f"{r['task']}: N={int(r['N'])}, success={int(r['success'])}, "
            f"failure={int(r['failure'])}, SR={float(r['success_rate']):.3f}"
        )

    total = len(step)
    missing = int(step["feat_trace_missing"].sum())
    single = int(step["feat_trace_single_point"].sum())

    lines += [
        "",
        "Trace parser / degeneracy",
        "-" * 72,
        f"trace missing: {missing}/{total} ({missing/total:.2%})",
        f"single-point traces: {single}/{total} ({single/total:.2%})",
        f"sustained trace degeneracy: >= {run_len} consecutive single-point traces",
        "",
    ]

    if not trace_summary.empty:
        for _, r in trace_summary.iterrows():
            label = "success" if int(r["success"]) == 1 else "failure"
            lines.append(
                f"{r['task']} / {label}: "
                f"{int(r['episodes_with_sustained_trace_degeneracy'])}/"
                f"{int(r['N_episodes'])} episodes with sustained degeneracy "
                f"(fraction={float(r['fraction_with_sustained_trace_degeneracy']):.3f})"
            )

    lines += [
        "",
        "Largest descriptive episode-level differences",
        "-" * 72,
        "Effect = failure mean - success mean. Positive means larger in failures.",
    ]

    if not comparison.empty:
        temp = comparison.copy()
        temp["abs_effect"] = numeric(temp["failure_minus_success"]).abs()

        for task, g in temp.groupby("task"):
            lines.append("")
            lines.append(task)

            for _, r in g.sort_values("abs_effect", ascending=False).head(5).iterrows():
                lines.append(
                    f"  {r['metric']}: effect={r['failure_minus_success']:.6g}, "
                    f"95% bootstrap CI=[{r['bootstrap_ci_low']:.6g}, "
                    f"{r['bootstrap_ci_high']:.6g}], "
                    f"nS={int(r['n_success'])}, nF={int(r['n_failure'])}, "
                    f"small_n={bool(r['descriptive_only_small_n'])}"
                )

    lines += [
        "",
        "Guardrails",
        "-" * 72,
        "- Do not claim causality from Depth -> Trace -> Action from this pilot.",
        "- Do not randomly split timestep rows for train/test; split by episode.",
        "- Coke has very few success episodes, so its estimates are descriptive.",
        "- Use the pilot only to decide whether signals justify formal collection.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    a = parse_args()

    step_path = Path(a.step_features).expanduser().resolve()
    ep_path = Path(a.episode_features).expanduser().resolve()
    out = Path(a.output_dir).expanduser().resolve()
    fig_dir = out / "figures"

    if not step_path.exists():
        raise FileNotFoundError(step_path)
    if not ep_path.exists():
        raise FileNotFoundError(ep_path)

    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    step, ep = load_data(step_path, ep_path)

    print("===== PILOT DATA =====")
    print("episodes:", step["episode_key"].nunique())
    print("steps:", len(step))
    print("trace missing:", int(step["feat_trace_missing"].sum()))
    print("single-point trace steps:", int(step["feat_trace_single_point"].sum()))

    if int(step["feat_trace_missing"].sum()) > 0:
        print(
            "WARNING: trace-missing steps remain. If these are from the old "
            "single-point parser bug, fix parse_trace() and rerun extraction first."
        )

    epm = build_episode_metrics(step, ep, a.degenerate_run)
    outcomes = task_outcomes(epm)
    trace_summary = trace_degeneracy_summary(epm)
    comparison = success_failure_comparison(epm, a.bootstrap, a.seed)
    curves = normalized_curves(step, a.bins, a.bootstrap, a.seed + 1)
    drawer = drawer_table(epm)

    outcomes.to_csv(out / "task_outcomes.csv", index=False)
    epm.to_csv(out / "episode_metrics.csv", index=False)
    comparison.to_csv(out / "success_failure_episode_comparison.csv", index=False)
    curves.to_csv(out / "normalized_time_curves.csv", index=False)
    trace_summary.to_csv(out / "trace_degeneracy_summary.csv", index=False)
    drawer.to_csv(out / "drawer_episode_progress.csv", index=False)

    plot_curves(curves, fig_dir)
    plot_drawer_progress(step, fig_dir, a.bins)
    plot_condition3(step, fig_dir)

    write_report(
        out / "pilot_report.txt",
        outcomes,
        trace_summary,
        comparison,
        step,
        a.degenerate_run,
    )

    print("\n===== TASK OUTCOMES =====")
    print(outcomes.to_string(index=False))

    print("\n===== TRACE DEGENERACY =====")
    print(trace_summary.to_string(index=False))

    print("\n===== TOP DESCRIPTIVE DIFFERENCES =====")
    if not comparison.empty:
        temp = comparison.copy()
        temp["abs_effect"] = numeric(temp["failure_minus_success"]).abs()
        cols = [
            "metric", "n_success", "n_failure", "success_mean", "failure_mean",
            "failure_minus_success", "bootstrap_ci_low", "bootstrap_ci_high",
            "descriptive_only_small_n",
        ]
        for task, g in temp.groupby("task"):
            print(f"\n[{task}]")
            print(
                g.sort_values("abs_effect", ascending=False)
                .head(5)[cols]
                .to_string(index=False)
            )

    print("\nSaved analysis to:", out)
    print("Figures:", fig_dir)
    print("Report:", out / "pilot_report.txt")


if __name__ == "__main__":
    main()

