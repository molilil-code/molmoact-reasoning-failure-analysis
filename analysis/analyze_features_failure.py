#!/usr/bin/env python3
"""
Full-featureset failure-correlation analysis over the merged formal features.

Design-doc discipline (experiment_design.md §6.1 / §9):
- Episode is the analysis unit, not the step.  We aggregate per-episode
  (mean / max / p90) and estimate effect sizes + bootstrap CIs at the
  episode level.
- This is EXPLORATORY / retrospective association only: worst-case
  (max) aggregation uses the back of the episode, so nothing here is an
  online "prediction before execution" claim.  Feature selection for the
  online model still comes from the pre-registered §4 set.
- Per-task separation is reported (episode lengths / failure rates differ),
  plus a cross-task sign-consistency check.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_PREFIX = "feat_"
N_BOOT = 2000
SEED = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step-features", required=True)
    p.add_argument("--episode-features", required=True)
    p.add_argument("--out", required=True, help="output report path (.txt)")
    return p.parse_args()


def safe_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan}
    return {"n": len(s), "mean": float(s.mean()), "std": float(s.std())}


def cohen_d(s_succ: pd.Series, s_fail: pd.Series) -> float:
    """Pooled Cohen's d, positive means failures are LARGER."""
    s = pd.to_numeric(s_succ, errors="coerce").dropna()
    f = pd.to_numeric(s_fail, errors="coerce").dropna()
    if len(s) < 2 or len(f) < 2:
        return np.nan
    sp = np.sqrt(((len(s) - 1) * s.var() + (len(f) - 1) * f.var()) / (len(s) + len(f) - 2))
    if sp == 0 or np.isnan(sp):
        return np.nan
    return float((f.mean() - s.mean()) / sp)


def bootstrap_ci_effect(rows, feature_col, n_boot, seed):
    """Episode-grouped bootstrap CI for (fail_mean - succ_mean), larger=more in failures."""
    rng = np.random.default_rng(seed)
    succ = rows.loc[rows["label_episode_success"] == 1, feature_col].dropna().to_numpy()
    fail = rows.loc[rows["label_episode_success"] == 0, feature_col].dropna().to_numpy()
    if len(succ) < 2 or len(fail) < 2:
        return np.nan, np.nan
    diffs = []
    for _ in range(n_boot):
        bs = rng.choice(succ, size=len(succ), replace=True)
        bf = rng.choice(fail, size=len(fail), replace=True)
        diffs.append(bf.mean() - bs.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi


def main():
    args = parse_args()
    st = pd.read_csv(args.step_features)
    ep = pd.read_csv(args.episode_features)

    feat_cols = [c for c in st.columns if c.startswith(FEATURE_PREFIX)]
    prior_cols = [c for c in st.columns if c.startswith("feat_trace_relplan_")
                  or c.startswith("feat_trace_spread") or c.startswith("feat_trace_curvature")
                  or c.startswith("feat_trace_startpoint_")
                  or c.endswith("_h1") or c.endswith("_cos_h1")]
    added_cols = ["feat_trace_spread", "feat_trace_curvature", "feat_trace_startpoint_drift_prev",
                  "feat_trace_relplan_drift_prev", "feat_trace_relplan_drift_norm_prev", "feat_trace_relplan_cos_prev",
                  "feat_trace_endpoint_drift_h1", "feat_trace_endpoint_drift_h1_norm", "feat_trace_endpoint_cos_h1"]

    # ---------------- Episode aggregates ----------------
    # mean / max / p90 per episode across all feat cols.
    agg = {}
    for stat, fn in [("mean", "mean"), ("max", "max"), ("p90", lambda x: np.nanpercentile(x, 90))]:
        g = st.groupby(["task", "condition_id"])[feat_cols].agg(fn)
        g = g.rename(columns={c: f"ep_{stat}__{c}" for c in feat_cols})
        agg[stat] = g

    ep_full = ep.copy()
    for stat, g in agg.items():
        ep_full = ep_full.merge(g.reset_index(), on=["task", "condition_id"], how="left")

    ep_full["label_episode_success"] = ep_full["label_episode_success"].astype(int)

    # ---------------- Analysis ---------------
    lines = []
    lines.append("=" * 90)
    lines.append("FULL-FEATURESET FAILURE-ASSOCIATION SCAN (exploratory, episode-level)")
    lines.append("=" * 90)
    lines.append("Units: episodes (NOT steps). 0 rows already aggregated.")
    lines.append("Effect = Cohen's d on episode aggregate (fail - succ), positive => larger in failures.")
    lines.append("CI = 95% bootstrap of mean difference (fail - succ); excl. 0 => reliable sign.")
    lines.append("")

    # Guardrail counts
    lines.append("Tasks / episodes / failure rate:")
    for t in ep_full["task"].unique():
        sub = ep_full[ep_full["task"] == t]
        n = len(sub)
        f = int(sub["label_episode_failure"].sum())
        lines.append(f"  {t:18s} N={n:3d} fail={f:3d} ({100*f/n:.0f}%)")
    lines.append("")

    # Per-task: list features with large |d| and consistent sign across >=2 tasks.
    tasks = sorted(ep_full["task"].unique())
    results = []
    for col in feat_cols:
        for stat in ["mean", "max", "p90"]:
            epcol = f"ep_{stat}__{col}"
            ds = [cohen_d(
                ep_full.loc[(ep_full["task"] == t) & (ep_full["label_episode_success"] == 1), epcol],
                ep_full.loc[(ep_full["task"] == t) & (ep_full["label_episode_success"] == 0), epcol],
            ) for t in tasks]
            results.append({"col": col, "stat": stat, "ds": ds})

    # Rank by mean |d| across tasks.
    rankable = [r for r in results if all(not np.isnan(d) for d in r["ds"])]
    rankable.sort(key=lambda r: -np.mean([abs(d) for d in r["ds"]]))

    lines.append("Top-ranked features by avg |Cohen's d| across tasks (mean / max / p90 aggregation):")
    lines.append("")
    hdr = f"{'feature':34s} {'agg':5s} " + "".join(f"{t[:1]:>9s} (d)" for t in tasks)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rankable[:25]:
        lines.append(f"{r['col']:34s} {r['stat']:5s} " + "".join(f"{d:>+9.2f}" for d in r["ds"]))
    lines.append("")

    # New/added features specifically, with CIs on the mean aggregation.
    lines.append("NEW / ADDED features: mean-aggregation effect + 95% bootstrap CI (per task):")
    lines.append("")
    lines.append(f"{'feature':38s} " + "".join(f"{t[:1]:>20s}" for t in tasks))
    lines.append("-" * 100)
    for col in added_cols:
        if col not in st.columns:
            continue
        epcol = f"ep_mean__{col}"
        if epcol not in ep_full.columns:
            continue
        cells = []
        ok = True
        for t in tasks:
            sub = ep_full[ep_full["task"] == t]
            lo, hi = bootstrap_ci_effect(sub, epcol, N_BOOT, SEED)
            d = cohen_d(sub[sub["label_episode_success"] == 1][epcol],
                        sub[sub["label_episode_success"] == 0][epcol])
            if np.isnan(lo):
                cells.append("n/a")
                ok = False
            else:
                cells.append(f"d={d:+.2f} CI[{lo:+.2f},{hi:+.2f}]")
        lines.append(f"{col:38s} " + "".join(f"{c:>20s}" for c in cells))
    lines.append("")

    # ---------------- Cross-task sign consistency ---------------
    lines.append("Cross-task sign consistency (features with |d|>=0.4 in >=2 tasks and NO sign conflicts):")
    lines.append("")
    for r in rankable:
        ds = r["ds"]
        big = [d for d in ds if abs(d) >= 0.4]
        if len(big) < 2:
            continue
        signs = {1 if d > 0 else (-1 if d < 0 else 0) for d in ds}
        if signs == {1}:
            tag = "ALL-FAIL>SUCC"
        elif signs == {-1}:
            tag = "ALL-FAIL<SUCC"
        else:
            continue
        lines.append(f"  {tag:15s} {r['col']:38s} " + "".join(f"{d:>+9.2f}" for d in ds))
    lines.append("")

    # ---------------- Sign-conflated action / depth sanity ----------
    lines.append("Note on leak discipline:")
    lines.append("  - ep_max__* aggregates use the whole playing episode (inc. back end).")
    lines.append("    They are RETROSPECTIVE descriptors, not online predictors.")
    lines.append("  - ep_mean__/ep_p90__* are less back-leaky but still aggregate over the full rollout.")
    lines.append("  - Only per-step feat_* with prefix evaluation can support online claims (later stage).")
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
