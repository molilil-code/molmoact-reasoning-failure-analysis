#!/usr/bin/env python3
"""
Backfill the document-specified H1 "relative planning-vector drift" feature
on top of the already-extracted formal_features.

Motivation
----------
The design doc (experiment_design.md, section 4B, H1) defines the *primary*
signal as:

    v_t     = p_t^end - p_t^start          (planning vector, endpoints)
    D_t^plan = || v_t - v_{t-1} ||_2        (vector drift between replans)

The current extract_features.py exports raw endpoint / centroid / shape drifts
(e.g. feat_trace_endpoint_drift_prev = ||p_t^end - p_{t-1}^end||_2), which is
the drift of the *absolute endpoint image position*.  H1 instead explicitly
removes the whole-image translation of the gripper (the scene / visual-trace
frame moves together) and measures the drift of the *translation-invariant
planning vector*.

This script recomputes that feature (and a couple of close relatives) from
the raw trace arrays stored in steps.jsonl, then back-fills three new columns
into diff'ing format:

    feat_trace_relplan_drift_prev          # D_t^plan, raw vector drift
    feat_trace_relplan_drift_norm_prev     # D_t^plan / ||v_t||
    feat_trace_relplan_cos_prev            # cosine(v_t, v_{t-1})

Only feat_* columns are consumed by the online predictors (per design doc),
so these new columns are valid online inputs and are candidates for the
pre-registered feature set.  ep_* aggregates are written to episode by the
caller's aggregation stage.

Usage
-----
python analysis/add_h1_endpoint_drift.py \
  --input-root results/formal \
  --features-path data/formal_features/step_features.csv \
  --output-path data/formal_features/step_features.csv \
  [--rename-backfill]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Trace parsing (identical logic to extract_features.parse_trace)
# ----------------------------------------------------------------------

def parse_trace(trace):
    if trace is None:
        return None
    try:
        arr = np.asarray(trace, dtype=np.float64)
    except Exception:
        return None
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
        return None
    return arr.astype(np.float64)


def planning_vector(trace):
    """v_t = p_end - p_start (H1)."""
    if trace is None or len(trace) < 2:
        return None
    return trace[-1] - trace[0]


def cosine(a, b):
    if a is None or b is None:
        return np.nan
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    # Returns 1.0 for identical vectors (dot / (na*nb)), matching extract_features.
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_steps(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--features-path", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument(
        "--rename-backfill",
        action="store_true",
        help=(
            "Also overwrite the existing *_endpoint_drift_prev columns with "
            "the H1 relative metric so downstream models see the true H1 "
            "updates in-place. Without this, keep the old columns and only "
            "add new ones."
        ),
    )
    args = ap.parse_args()

    features_path = Path(args.features_path)
    output_path = Path(args.output_path)
    df = pd.read_csv(features_path)

    # A row is identified by (task, condition_id, step_id) in the features.
    key = set(zip(df["task"], df["condition_id"], df["step_id"]))

    prev_plan = {}
    new_rows = []

    for ep_path in sorted(Path(args.input_root).rglob("steps.jsonl")):
        ep_dir = ep_path.parent
        # Recompute the identity fields the same way extract_features does.
        # Prefer summary.json fields, fall back to first step.
        summary = {}
        if (ep_dir / "summary.json").exists():
            summary = read_json(ep_dir / "summary.json")
        steps = load_steps(ep_path)
        if not steps:
            continue

        cond = int(summary.get("condition_id", steps[0].get("condition_id", -1)))
        task = str(summary.get("task", steps[0].get("task", "unknown")))
        last_cond_plan = None

        for item in steps:
            sid = int(item.get("step_id", -1))
            pre = item.get("pre_action") or {}

            trace = parse_trace(pre.get("trace"))
            cur_plan = planning_vector(trace)

            drift = np.nan
            drift_norm = np.nan
            cos = np.nan
            if cur_plan is not None and last_cond_plan is not None:
                diff = cur_plan - last_cond_plan
                drift = float(np.linalg.norm(diff))
                vnorm = float(np.linalg.norm(cur_plan))
                drift_norm = drift / vnorm if vnorm > 0 else np.nan
                cos = cosine(cur_plan, last_cond_plan)

            new_rows.append(
                {
                    "task": task,
                    "condition_id": cond,
                    "step_id": sid,
                    "feat_trace_relplan_drift_prev": drift,
                    "feat_trace_relplan_drift_norm_prev": drift_norm,
                    "feat_trace_relplan_cos_prev": cos,
                }
            )
            last_cond_plan = cur_plan

    if not new_rows:
        sys.exit("No steps found under input-root")

    nf = pd.DataFrame(new_rows)
    if len(nf) != len(df):
        sys.exit(
            f"Mismatch: features has {len(df)} rows, scan built {len(nf)}. "
            f"Refusing to backfill to avoid misalignment."
        )

    # Sanity: identity keys must line up with the features file.
    mism = [
        i
        for i, row in enumerate(zip(df["task"], df["condition_id"], df["step_id"]))
        if row != (nf.iloc[i]["task"], nf.iloc[i]["condition_id"], nf.iloc[i]["step_id"])
    ]
    if mism:
        sys.exit(
            f"Row identity mismatch at {len(mism)} rows "
            f"(e.g. first at {mism[0]}); refusing to backfill."
        )

    new_cols = ["feat_trace_relplan_drift_prev", "feat_trace_relplan_drift_norm_prev", "feat_trace_relplan_cos_prev"]

    if args.rename_backfill:
        # Replace the old absolute endpoint-drift columns with the H1 metric.
        df.rename(
            columns={
                "feat_trace_endpoint_drift_prev": "feat_trace_abs_endpoint_drift_prev",
            },
            inplace=True,
        )
        df["feat_trace_endpoint_drift_prev"] = nf["feat_trace_relplan_drift_prev"].values

    for c in new_cols:
        df[c] = nf[c].values

    df.rename(
        columns={
            "feat_trace_relplan_drift_prev": "feat_trace_endpoint_drift_h1",
            "feat_trace_relplan_drift_norm_prev": "feat_trace_endpoint_drift_h1_norm",
            "feat_trace_relplan_cos_prev": "feat_trace_endpoint_cos_h1",
        },
        inplace=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Report missingness / basic stats of the new columns.
    print("Backfilled to:", output_path)
    for c in ["feat_trace_endpoint_drift_h1", "feat_trace_endpoint_drift_h1_norm", "feat_trace_endpoint_cos_h1"]:
        col = df[c].astype(float)
        n = int(col.notna().sum())
        print(
            f"  {c:38s} n={n:5d}/{len(df)} "
            f"mean={col.mean():.4f} std={col.std():.4f}"
        )
    print("Rows:", len(df), "Columns:", len(df.columns))


if __name__ == "__main__":
    main()
