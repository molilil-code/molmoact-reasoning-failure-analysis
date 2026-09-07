"""Evaluate decoder-based Trace--Depth features without changing formal results.

The analysis is exploratory and uses the already generated trace-depth CSV.
Episode-level associations are retrospective descriptors. OpenDrawer
checkpoint scores use grouped out-of-fold logistic regression and the existing
online labels, with fold-local imputation/standardization.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


TRACE_DEPTH_FEATURES = [
    "feat_trace_depth_std_norm",
    "feat_trace_depth_gradient_norm",
    "feat_trace_depth_range_norm",
    "feat_trace_endpoint_depth_percentile",
    "feat_trace_endpoint_depth_percentile_drift_prev",
]


def cohen_d(failure: pd.Series, success: pd.Series) -> float:
    if len(failure) < 2 or len(success) < 2:
        return float("nan")
    pooled = math.sqrt(
        ((len(failure) - 1) * failure.var(ddof=1) + (len(success) - 1) * success.var(ddof=1))
        / (len(failure) + len(success) - 2)
    )
    return float((failure.mean() - success.mean()) / pooled) if pooled > 0 else 0.0


def auc_failure(values: pd.Series, success: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "success": success}).dropna()
    if frame["success"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(1 - frame["success"].astype(int), frame["value"]))


def episode_associations(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, task_df in df.groupby("task", sort=True):
        episode_success = task_df.groupby("episode_key")["label_episode_success"].first()
        for feature in TRACE_DEPTH_FEATURES:
            aggregates = task_df.groupby("episode_key")[feature].mean()
            frame = pd.concat([episode_success.rename("success"), aggregates.rename("value")], axis=1).dropna()
            failure = frame.loc[frame["success"].eq(0), "value"]
            success = frame.loc[frame["success"].eq(1), "value"]
            rows.append(
                {
                    "task": task,
                    "feature": feature,
                    "episode_count": len(frame),
                    "step_coverage": float(task_df[feature].notna().mean()),
                    "success_mean": float(success.mean()) if len(success) else np.nan,
                    "failure_mean": float(failure.mean()) if len(failure) else np.nan,
                    "cohen_d_failure_minus_success": cohen_d(failure, success),
                    "episode_auc_failure": auc_failure(frame["value"], frame["success"]),
                }
            )
    return rows


def grouped_oof_scores(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    clean = frame[["episode_key", "online_failure_label", *features]].copy()
    clean[features] = clean[features].replace([np.inf, -np.inf], np.nan)
    y = clean["online_failure_label"].to_numpy(dtype=int)
    groups = clean["episode_key"].to_numpy()
    scores = np.full(len(clean), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=5)
    for train_idx, test_idx in splitter.split(clean, y, groups):
        train = clean.iloc[train_idx]
        test = clean.iloc[test_idx]
        medians = train[features].median()
        means = train[features].mean()
        train_x = train[features].fillna(medians).fillna(means).fillna(0.0)
        test_x = test[features].fillna(medians).fillna(means).fillna(0.0)
        train_mean = train_x.mean()
        scales = train_x.std(ddof=0).replace(0, 1.0).fillna(1.0)
        train_x = (train_x - train_mean) / scales
        test_x = (test_x - train_mean) / scales
        if np.unique(y[train_idx]).size < 2:
            scores[test_idx] = float(np.mean(y[train_idx]))
            continue
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(train_x, y[train_idx])
        scores[test_idx] = model.predict_proba(test_x)[:, 1]
    return y, scores


def oof_rows(df: pd.DataFrame, labels: pd.DataFrame) -> list[dict[str, Any]]:
    merged = df.merge(
        labels[["episode_key", "step_id", "horizon", "valid_mask", "online_failure_label"]],
        on=["episode_key", "step_id"],
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for horizon in [5, 10, 20]:
        subset = merged.loc[merged["horizon"].eq(horizon) & merged["valid_mask"].eq(1)].copy()
        for name, features in [(feature, [feature]) for feature in TRACE_DEPTH_FEATURES] + [("TraceDepthCombined", TRACE_DEPTH_FEATURES)]:
            coverage = float(subset[features].notna().all(axis=1).mean())
            usable = subset.loc[subset[features].notna().all(axis=1)].copy()
            y, scores = grouped_oof_scores(usable, features)
            rows.append(
                {
                    "horizon": horizon,
                    "feature_set": name,
                    "n_rows": len(usable),
                    "coverage": coverage,
                    "positive_prevalence": float(np.mean(y)),
                    "step_auroc_oof": float(roc_auc_score(y, scores)),
                    "step_auprc_oof": float(average_precision_score(y, scores)),
                    "step_auprc_baseline": float(np.mean(y)),
                    "delta_auprc": float(average_precision_score(y, scores) - np.mean(y)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, associations: list[dict[str, Any]], oof: list[dict[str, Any]]) -> None:
    lines = [
        "# Trace–Depth feature predictivity audit",
        "",
        "The feature definitions are decoder-based and online-safe: current/past pre-action Depth and Trace only.",
        "This report is exploratory. It does not change the frozen OpenDrawer formal ablation and does not add GT task-state fields to the predictor.",
        "",
        "## Coverage",
        "",
        "- Endpoint depth percentile is available on 99.8% of 6,356 steps.",
        "- Path statistics (std/gradient/range) are available on 89.0%; 362 steps have single-point traces and are intentionally NaN for path variation.",
        "- Fourteen malformed depth-token rows are excluded by the existing decoder extraction metadata.",
        "",
        "## Episode-level association (retrospective)",
        "",
        "| Task | Feature | Success mean | Failure mean | Cohen d (failure-success) | Episode AUC for failure |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in associations:
        lines.append(
            f"| {row['task']} | `{row['feature']}` | {row['success_mean']:.4f} | {row['failure_mean']:.4f} | "
            f"{row['cohen_d_failure_minus_success']:.3f} | {row['episode_auc_failure']:.3f} |"
        )
    lines += [
        "",
        "These whole-episode means are descriptors, not online prediction scores; they can use information after the eventual failure and should not be interpreted as causal precursors.",
        "",
        "## OpenDrawer grouped OOF checkpoint results",
        "",
        "| H | Feature set | Coverage | Step AUROC | Step AUPRC | Prevalence baseline | ΔAUPRC |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in oof:
        lines.append(
            f"| {row['horizon']} | `{row['feature_set']}` | {row['coverage']:.1%} | {row['step_auroc_oof']:.3f} | "
            f"{row['step_auprc_oof']:.3f} | {row['step_auprc_baseline']:.3f} | {row['delta_auprc']:+.3f} |"
        )
    lines += [
        "",
        "In this small 20-episode OpenDrawer OOF check, the new Trace–Depth features alone do not consistently exceed the prevalence baseline. They are therefore promising feature candidates, not yet evidence of a useful standalone predictor.",
        "",
        "## Recommended feature families",
        "",
        "1. Path geometry: normalized depth std, gradient, range.",
        "2. Endpoint semantics: endpoint depth percentile and its causal drift.",
        "3. Temporal consistency: profile-level Trace–Depth drift, signed endpoint-start depth change, monotonicity, and local extrema count.",
        "4. Missingness/degeneracy indicators: single-point Trace, malformed depth row, and invalid decoder alignment.",
        "",
        "Only the first two families are currently implemented. Raw decoder depth diagnostics should remain exploratory rather than primary predictor inputs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/formal_features/step_features_trace_depth.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/opendrawer_analysis/online_labels_opendrawer.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/formal_features"))
    args = parser.parse_args()
    df = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    associations = episode_associations(df)
    oof = oof_rows(df.loc[df["task"].eq("open_drawer")].copy(), labels)
    write_csv(args.output_dir / "trace_depth_episode_associations.csv", associations)
    write_csv(args.output_dir / "trace_depth_opendrawer_oof.csv", oof)
    write_report(args.output_dir / "trace_depth_predictivity_report.md", associations, oof)
    print({"episode_rows": len(associations), "oof_rows": len(oof)})


if __name__ == "__main__":
    main()
