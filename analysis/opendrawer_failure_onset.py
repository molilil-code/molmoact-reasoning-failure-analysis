"""OpenDrawer onset validation, online labels, and grouped ablations.

This module is deliberately self-contained so the frozen onset definition can
be rerun from the checked-in formal feature table without touching raw data.
It produces:

* a 10-episode validation table and progress plot;
* automatic onset timestamps for all OpenDrawer episodes;
* horizon-aware online labels (H=5, 10, 20);
* episode-grouped out-of-fold A0--A4 ablation metrics.

The formal table contains no action log-probability column.  Consequently A0
is reported as ``A0_action_magnitude_proxy`` rather than silently claiming to
be the preregistered log-probability baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


W_DEFAULT = 10
EPS_DEFAULT = 0.02
MIN_START_DEFAULT = 20
HORIZONS = (5, 10, 20)
BOOTSTRAP_DEFAULT = 1000


def normalize_drawer_progress(values: Sequence[float], raw_scale: float | None = None) -> np.ndarray:
    """Return progress in [0, 1] without double-normalising the formal table.

    ``label_drawer_progress`` in the checked-in formal features is already
    normalised.  ``raw_scale`` is available for callers passing raw qpos.
    """

    x = np.asarray(values, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    if raw_scale is not None:
        if raw_scale <= 0:
            raise ValueError("raw_scale must be positive")
        x = x / raw_scale
    return np.clip(x, 0.0, 1.0)


def compute_onset_open_drawer(
    drawer_progress: Sequence[float],
    W: int = W_DEFAULT,
    eps: float = EPS_DEFAULT,
    min_start: int = MIN_START_DEFAULT,
) -> int | None:
    """Detect the first *irreversible* best-so-far progress plateau.

    A candidate t must have less than ``eps`` best-so-far progress in the next
    W steps.  The second condition prevents calling a temporary pause an onset
    when the trajectory later makes a meaningful recovery.
    """

    if W < 1 or eps <= 0 or min_start < 0:
        raise ValueError("W >= 1, eps > 0, and min_start >= 0 are required")
    p = normalize_drawer_progress(drawer_progress)
    if len(p) <= min_start + W:
        return None
    best = np.maximum.accumulate(p)
    for t in range(min_start, len(best) - W):
        if best[t + W] - best[t] < eps:
            future_max = float(np.max(p[t + W :])) if t + W < len(p) else float(best[t])
            if future_max - best[t] < 2.0 * eps:
                return int(t)
    return None


def _read_open_drawer(step_features: Path) -> pd.DataFrame:
    df = pd.read_csv(step_features)
    required = {"task", "episode_key", "step_id", "label_episode_success", "label_drawer_progress"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.loc[df["task"].eq("open_drawer")].copy()
    if df.empty:
        raise ValueError("No OpenDrawer rows found")
    df["step_id"] = pd.to_numeric(df["step_id"], errors="raise").astype(int)
    df["label_drawer_progress"] = pd.to_numeric(df["label_drawer_progress"], errors="coerce").fillna(0.0)
    df["label_episode_success"] = pd.to_numeric(df["label_episode_success"], errors="raise").astype(int)
    return df.sort_values(["episode_key", "step_id"]).reset_index(drop=True)


def episode_onsets(df: pd.DataFrame, W: int, eps: float, min_start: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, g in df.groupby("episode_key", sort=True):
        g = g.sort_values("step_id")
        p = normalize_drawer_progress(g["label_drawer_progress"].to_numpy())
        onset = compute_onset_open_drawer(p, W=W, eps=eps, min_start=min_start)
        success = int(g["label_episode_success"].iloc[0])
        rows.append(
            {
                "task": "open_drawer",
                "episode_key": key,
                "condition_id": int(g["condition_id"].iloc[0]) if "condition_id" in g else "",
                "success": success,
                "failure": 1 - success,
                "num_steps": len(g),
                "max_progress": float(p.max(initial=0.0)),
                "automatic_t_onset": onset if onset is not None else "",
                "automatic_onset_frac": (onset / len(g)) if onset is not None else "",
                "rule_W": W,
                "rule_eps": eps,
                "rule_min_start": min_start,
            }
        )
    return pd.DataFrame(rows)


# These are an auditable first-pass manual curve review.  The values are
# intentionally kept separate from the automatic rule and are only used for
# validation, never for tuning the classifier.
MANUAL_VALIDATION_ONSETS: Mapping[str, int | None] = {
    "open_drawer/condition_0000": 22,
    "open_drawer/condition_0001": None,
    "open_drawer/condition_0004": None,
    "open_drawer/condition_0007": 20,
    "open_drawer/condition_0008": 20,
    "open_drawer/condition_0009": 46,
    "open_drawer/condition_0010": 48,
    "open_drawer/condition_0014": None,
    "open_drawer/condition_0017": None,
    "open_drawer/condition_0019": None,
}


def make_validation_table(onsets: pd.DataFrame) -> pd.DataFrame:
    selected = onsets[onsets["episode_key"].isin(MANUAL_VALIDATION_ONSETS)].copy()
    selected["manual_t_onset"] = selected["episode_key"].map(MANUAL_VALIDATION_ONSETS)
    selected["manual_t_onset"] = selected["manual_t_onset"].where(selected["manual_t_onset"].notna(), "")
    selected["onset_match"] = selected.apply(
        lambda r: (r["manual_t_onset"] == "" and r["automatic_t_onset"] == "")
        or (r["manual_t_onset"] != "" and r["manual_t_onset"] == r["automatic_t_onset"]),
        axis=1,
    )
    selected["annotation_note"] = np.where(
        selected["success"].eq(1),
        "success trajectory reaches completion; no failure onset",
        "first visually sustained irreversible plateau",
    )
    return selected.sort_values("episode_key")


def make_online_labels(df: pd.DataFrame, onsets: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    onset_map = onsets.set_index("episode_key")["automatic_t_onset"].to_dict()
    rows: list[pd.DataFrame] = []
    base_cols = ["task", "condition_id", "episode_key", "step_id", "label_episode_success", "label_episode_failure"]
    for key, g in df.groupby("episode_key", sort=True):
        g = g.sort_values("step_id").copy()
        onset_raw = onset_map[key]
        onset = None if onset_raw == "" or pd.isna(onset_raw) else int(onset_raw)
        for H in horizons:
            valid = np.ones(len(g), dtype=bool) if onset is None else g["step_id"].to_numpy() < onset
            y = np.zeros(len(g), dtype=int)
            if onset is not None:
                step = g["step_id"].to_numpy()
                y[valid] = ((step[valid] + int(H)) >= onset).astype(int)
            out = g[base_cols].copy()
            out["t_onset"] = onset if onset is not None else ""
            out["horizon"] = int(H)
            out["valid_mask"] = valid.astype(int)
            out["online_failure_label"] = y
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


FEATURE_GROUPS: Mapping[str, list[str]] = {
    "action_proxy": ["feat_model_translation_norm"],
    "action_structured": [
        "feat_model_translation_norm",
        "feat_model_rotation_norm",
        "feat_model_gripper",
        "feat_model_action_jump_prev_l2",
        "feat_model_translation_jump_prev_l2",
        "feat_model_rotation_jump_prev_l2",
        "feat_model_translation_cosine_prev",
        "feat_model_translation_direction_flip",
        "feat_model_gripper_change_prev",
        "feat_model_action_jump_rollmean",
    ],
    "trace": [
        "feat_trace_num_points",
        "feat_trace_polyline_length",
        "feat_trace_straightness",
        "feat_trace_mean_turn_angle_rad",
        "feat_trace_spread",
        "feat_trace_curvature",
        "feat_trace_endpoint_drift_prev",
        "feat_trace_shape_drift_prev",
        "feat_trace_direction_change_prev_rad",
        "feat_trace_shape_drift_rollmean",
    ],
    "depth": [
        "feat_depth_token_count",
        "feat_depth_unique_count",
        "feat_depth_token_entropy",
        "feat_depth_spatial_boundary_ratio",
        "feat_depth_hamming_prev",
        "feat_depth_change_ratio_prev",
        "feat_depth_change_ratio_rollmean",
    ],
}


ABLATIONS: Mapping[str, tuple[str, ...]] = {
    "A0_action_magnitude_proxy": ("action_proxy",),
    "A1_action_structured": ("action_structured",),
    "A2_trace_only": ("trace",),
    "A3_depth_trace": ("depth", "trace"),
    "A4_depth_trace_action": ("depth", "trace", "action_structured"),
}

# Registered modality ablation.  Unlike the exploratory A0--A4 feature-family
# ladder above, every condition here contains the same action modality and
# adds/removes Depth and Trace explicitly.
MODALITY_ABLATIONS: Mapping[str, tuple[str, ...]] = {
    "Action": ("action_structured",),
    "Action+Depth": ("action_structured", "depth"),
    "Action+Trace": ("action_structured", "trace"),
    "Action+Depth+Trace": ("action_structured", "depth", "trace"),
    "Depth+Trace": ("depth", "trace"),
}


def _feature_columns(df: pd.DataFrame, groups: Sequence[str]) -> list[str]:
    cols: list[str] = []
    for group in groups:
        cols.extend(c for c in FEATURE_GROUPS[group] if c in df.columns and c not in cols)
    if not cols:
        raise ValueError(f"None of the requested features are present: {groups}")
    return cols


def threshold_at_episode_far(success_scores: np.ndarray, max_far: float = 0.10) -> float:
    """Choose the lowest observed threshold whose empirical FAR is allowed.

    Searching observed scores (instead of using a nominal quantile) matters for
    the 12-success-episode dataset: a 90th percentile can still count 2/12
    episodes because of finite-sample interpolation or ties.
    """

    if not 0 <= max_far <= 1:
        raise ValueError("max_far must lie in [0, 1]")
    scores = np.asarray(success_scores, dtype=float)
    if scores.size == 0:
        return float("nan")
    allowed = int(np.floor(max_far * scores.size + 1e-12))
    candidates = np.unique(scores)
    candidates = np.concatenate([candidates, [np.inf]])
    valid = [float(t) for t in candidates if int(np.sum(scores >= t)) <= allowed]
    return min(valid) if valid else float("inf")


def _make_model(random_state: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
    )


def _inner_fold_threshold(data: pd.DataFrame, train_idx: np.ndarray, y: np.ndarray, groups: np.ndarray, cols: Sequence[str], random_state: int) -> float:
    """Select a FAR threshold using only inner OOF predictions in train_idx."""

    from sklearn.model_selection import GroupKFold

    train = data.iloc[train_idx]
    local_y = y[train_idx]
    local_groups = groups[train_idx]
    n_inner = min(3, len(np.unique(local_groups)))
    if n_inner < 2:
        model = _make_model(random_state)
        model.fit(train[list(cols)], local_y)
        pred = model.predict_proba(train[list(cols)])[:, 1]
    else:
        pred = np.full(len(train), np.nan, dtype=float)
        splitter = GroupKFold(n_splits=n_inner)
        for inner_tr, inner_te in splitter.split(train, local_y, local_groups):
            model = _make_model(random_state)
            model.fit(train.iloc[inner_tr][list(cols)], local_y[inner_tr])
            pred[inner_te] = model.predict_proba(train.iloc[inner_te][list(cols)])[:, 1]
    scored = train[["episode_key", "label_episode_success"]].copy()
    scored["score"] = pred
    ep = scored.groupby("episode_key").agg(success=("label_episode_success", "first"), score=("score", "max"))
    return threshold_at_episode_far(ep.loc[ep.success.eq(1), "score"].to_numpy(), max_far=0.10)


def _cluster_bootstrap_ci(frame: pd.DataFrame, score_col: str, label_col: str, metric: str, n_bootstrap: int, random_state: int) -> tuple[float, float]:
    """95% episode-cluster bootstrap interval for step or episode metrics."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    keys = frame["episode_key"].drop_duplicates().to_numpy()
    by_key = {k: g for k, g in frame.groupby("episode_key", sort=False)}
    rng = np.random.default_rng(random_state)
    vals: list[float] = []
    for _ in range(n_bootstrap):
        selected = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([by_key[k] for k in selected], ignore_index=True)
        y = sample[label_col].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        score = sample[score_col].to_numpy(dtype=float)
        vals.append(float(roc_auc_score(y, score) if metric == "auroc" else average_precision_score(y, score)))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _risk_group(row: pd.Series, H: int) -> str:
    if int(row["label_episode_success"]) == 1:
        return "success"
    onset = int(row["t_onset"])
    return "failure_pre_window" if int(row["step_id"]) < onset - H else "failure_h_window"


def _risk_distribution_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (horizon, ablation, group), g in scores.groupby(["horizon", "ablation", "risk_group"], sort=True):
        values = g["score"].to_numpy(dtype=float)
        rows.append(
            {
                "horizon": int(horizon),
                "ablation": ablation,
                "risk_group": group,
                "n_steps": len(values),
                "n_episodes": g["episode_key"].nunique(),
                "score_mean": float(np.mean(values)),
                "score_median": float(np.median(values)),
                "score_p10": float(np.percentile(values, 10)),
                "score_p90": float(np.percentile(values, 90)),
                "score_max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def save_risk_distribution_plot(scores: pd.DataFrame, path: Path, specs: Mapping[str, tuple[str, ...]] = ABLATIONS) -> None:
    import matplotlib.pyplot as plt

    horizons = sorted(scores["horizon"].unique())
    ablations = list(specs)
    group_order = ["success", "failure_pre_window", "failure_h_window"]
    fig, axes = plt.subplots(len(horizons), len(ablations), figsize=(18, 3.8 * len(horizons)), squeeze=False, sharey=True)
    for i, h in enumerate(horizons):
        for j, ablation in enumerate(ablations):
            ax = axes[i, j]
            parts = [scores.loc[(scores.horizon.eq(h)) & (scores.ablation.eq(ablation)) & (scores.risk_group.eq(group)), "score"].to_numpy() for group in group_order]
            ax.boxplot(parts, tick_labels=["success", "pre", "H-window"], showfliers=False)
            ax.set_title(f"H={h} | {ablation.replace('_', ' ')}")
            ax.grid(axis="y", alpha=0.25)
            if j == 0:
                ax.set_ylabel("OOF risk score")
            ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle("OpenDrawer risk-score distributions", y=1.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def premature_alert_distances(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return event-level alerts, first-crossing episode rows, and summaries.

    A premature alert is an OOF score crossing the fold-specific threshold at
    ``t < t_onset - H``.  The distance ``d=t_onset-t`` is therefore strictly
    greater than H and cannot be relabelled as an H-window true positive.
    """

    records: list[dict[str, object]] = []
    first_records: list[dict[str, object]] = []
    keys = list(scores[["horizon", "ablation"]].drop_duplicates().itertuples(index=False))
    for horizon, ablation in keys:
        subset = scores.loc[scores.horizon.eq(horizon) & scores.ablation.eq(ablation)]
        for episode_key, g in subset.groupby("episode_key", sort=True):
            if int(g["label_episode_success"].iloc[0]) == 1:
                continue
            onset = int(g["t_onset"].iloc[0])
            threshold = float(g["fold_threshold"].iloc[0])
            early = g.loc[(g["score"] >= threshold) & (g["step_id"] < onset - int(horizon))].sort_values("step_id")
            if not early.empty:
                first = early.iloc[0]
                first_step = int(first["step_id"])
                first_records.append(
                    {
                        "horizon": int(horizon),
                        "ablation": ablation,
                        "episode_key": episode_key,
                        "first_premature_step": first_step,
                        "t_onset": onset,
                        "first_distance_to_onset": onset - first_step,
                        "first_distance_over_horizon": onset - first_step - int(horizon),
                        "score": float(first["score"]),
                        "threshold": threshold,
                    }
                )
            for _, row in early.iterrows():
                step = int(row["step_id"])
                distance = onset - step
                records.append(
                    {
                        "horizon": int(horizon),
                        "ablation": ablation,
                        "episode_key": episode_key,
                        "step_id": step,
                        "t_onset": onset,
                        "distance_to_onset": distance,
                        "distance_over_horizon": distance - int(horizon),
                        "score": float(row["score"]),
                        "threshold": threshold,
                    }
                )
    detail = pd.DataFrame(records, columns=[
        "horizon", "ablation", "episode_key", "step_id", "t_onset",
        "distance_to_onset", "distance_over_horizon", "score", "threshold",
    ])
    summary_rows: list[dict[str, object]] = []
    for horizon, ablation in keys:
        sub = detail.loc[detail.horizon.eq(horizon) & detail.ablation.eq(ablation)] if not detail.empty else detail
        source = scores.loc[scores.horizon.eq(horizon) & scores.ablation.eq(ablation)]
        n_failures = int(source.loc[source.label_episode_success.eq(0), "episode_key"].nunique())
        event_distances = sub["distance_to_onset"].to_numpy(dtype=float) if not sub.empty else np.array([], dtype=float)
        first_sub = pd.DataFrame(first_records)
        if not first_sub.empty:
            first_sub = first_sub.loc[first_sub.horizon.eq(horizon) & first_sub.ablation.eq(ablation)]
        first_distances = first_sub["first_distance_to_onset"].to_numpy(dtype=float) if not first_sub.empty else np.array([], dtype=float)
        summary_rows.append(
            {
                "horizon": int(horizon),
                "ablation": ablation,
                "n_failure_episodes": n_failures,
                "episodes_with_premature_alert": int(sub["episode_key"].nunique()) if not sub.empty else 0,
                "premature_episode_rate": (sub["episode_key"].nunique() / max(1, n_failures)) if not sub.empty else 0.0,
                "n_premature_alert_events": int(len(event_distances)),
                "event_distance_median": float(np.median(event_distances)) if len(event_distances) else float("nan"),
                "event_distance_mean": float(np.mean(event_distances)) if len(event_distances) else float("nan"),
                "event_distance_p90": float(np.percentile(event_distances, 90)) if len(event_distances) else float("nan"),
                "episode_first_distance_min": float(np.min(first_distances)) if len(first_distances) else float("nan"),
                "episode_first_distance_median": float(np.median(first_distances)) if len(first_distances) else float("nan"),
                "episode_first_distance_mean": float(np.mean(first_distances)) if len(first_distances) else float("nan"),
                "episode_first_distance_p90": float(np.percentile(first_distances, 90)) if len(first_distances) else float("nan"),
                "episode_first_distance_max": float(np.max(first_distances)) if len(first_distances) else float("nan"),
            }
        )
    first_columns = [
        "horizon", "ablation", "episode_key", "first_premature_step", "t_onset",
        "first_distance_to_onset", "first_distance_over_horizon", "score", "threshold",
    ]
    first = pd.DataFrame(first_records, columns=first_columns)
    return detail, first, pd.DataFrame(summary_rows)


DETECTORS: Mapping[str, str] = {
    "T0_raw": "raw risk threshold crossing",
    "T1_rollmean3": "causal 3-step rolling mean threshold crossing",
    "T2_consecutive2": "two consecutive raw steps over threshold",
}


def _detector_alert_mask(raw_scores: np.ndarray, threshold: float, detector: str) -> np.ndarray:
    """Apply a fixed detector without fitting or tuning on held-out steps."""

    raw_scores = np.asarray(raw_scores, dtype=float)
    above = raw_scores >= threshold
    if detector == "T0_raw":
        return above
    if detector == "T1_rollmean3":
        smooth = pd.Series(raw_scores).rolling(window=3, min_periods=3).mean().to_numpy()
        return np.isfinite(smooth) & (smooth >= threshold)
    if detector == "T2_consecutive2":
        return above & np.concatenate(([False], above[:-1]))
    raise ValueError(f"Unknown detector: {detector}")


def evaluate_detectors(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate T0/T1/T2 using the already fold-internal raw thresholds."""

    result_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for horizon in sorted(scores.horizon.unique()):
        for ablation in scores.loc[scores.horizon.eq(horizon), "ablation"].drop_duplicates():
            for detector in DETECTORS:
                subset = scores.loc[scores.horizon.eq(horizon) & scores.ablation.eq(ablation)].copy()
                if subset.empty:
                    continue
                subset["detector_alert"] = False
                for key, g in subset.groupby("episode_key", sort=True):
                    idx = g.sort_values("step_id").index
                    threshold = float(g["fold_threshold"].iloc[0])
                    subset.loc[idx, "detector_alert"] = _detector_alert_mask(g.sort_values("step_id")["score"].to_numpy(), threshold, detector)
                n_success = int(subset.loc[subset.label_episode_success.eq(1), "episode_key"].nunique())
                n_failure = int(subset.loc[subset.label_episode_success.eq(0), "episode_key"].nunique())
                false_alarm = 0
                detected = 0
                premature_episodes: set[str] = set()
                first_distances: list[float] = []
                lead_times: list[float] = []
                local_event_count = 0
                for key, g in subset.groupby("episode_key", sort=True):
                    g = g.sort_values("step_id")
                    alerts = g.loc[g["detector_alert"], "step_id"].astype(int)
                    if int(g["label_episode_success"].iloc[0]) == 1:
                        if not alerts.empty:
                            false_alarm += 1
                        continue
                    onset = int(g["t_onset"].iloc[0])
                    positive_start = onset - int(horizon)
                    window = alerts[(alerts >= positive_start) & (alerts < onset)]
                    premature = alerts[alerts < positive_start]
                    if not window.empty:
                        detected += 1
                        lead_times.append(float(onset - int(window.min())))
                    if not premature.empty:
                        premature_episodes.add(key)
                        first_distances.append(float(onset - int(premature.min())))
                        for step in premature:
                            local_event_count += 1
                            event_rows.append(
                                {
                                    "horizon": int(horizon),
                                    "ablation": ablation,
                                    "detector": detector,
                                    "episode_key": key,
                                    "step_id": int(step),
                                    "t_onset": onset,
                                    "distance_to_onset": onset - int(step),
                                }
                            )
                result_rows.append(
                    {
                        "horizon": int(horizon),
                        "ablation": ablation,
                        "detector": detector,
                        "detector_definition": DETECTORS[detector],
                        "n_success_episodes": n_success,
                        "n_failure_episodes": n_failure,
                        "success_false_alarm_count": false_alarm,
                        "far_episode": false_alarm / max(1, n_success),
                        "failure_detection_count_h_window": detected,
                        "failure_detection_rate_h_window": detected / max(1, n_failure),
                        "premature_episode_count": len(premature_episodes),
                        "premature_episode_rate": len(premature_episodes) / max(1, n_failure),
                        "premature_event_count": local_event_count,
                        "first_premature_distance_median": float(np.median(first_distances)) if first_distances else float("nan"),
                        "first_premature_distance_mean": float(np.mean(first_distances)) if first_distances else float("nan"),
                        "mean_lead_time_steps_h_window": float(np.mean(lead_times)) if lead_times else float("nan"),
                        "n_failures_with_h_window_alert": len(lead_times),
                    }
                )
    return pd.DataFrame(result_rows), pd.DataFrame(event_rows)


def save_risk_trajectory_plots(scores: pd.DataFrame, path_pattern: str, specs: Mapping[str, tuple[str, ...]]) -> list[Path]:
    """Plot OOF trajectories for all failures and four representative successes."""

    import matplotlib.pyplot as plt

    failures = sorted(scores.loc[scores.label_episode_success.eq(0), "episode_key"].unique())
    all_success = sorted(scores.loc[scores.label_episode_success.eq(1), "episode_key"].unique())
    successes = all_success[:4]
    episode_keys = failures + successes
    outputs: list[Path] = []
    for horizon in sorted(scores["horizon"].unique()):
        subset = scores.loc[scores.horizon.eq(horizon)]
        names = list(specs)
        fig, axes = plt.subplots(len(episode_keys), len(names), figsize=(4.2 * len(names), 2.0 * len(episode_keys)), squeeze=False, sharey=True)
        for i, episode_key in enumerate(episode_keys):
            for j, name in enumerate(names):
                ax = axes[i, j]
                g = subset.loc[subset.episode_key.eq(episode_key) & subset.ablation.eq(name)].sort_values("step_id")
                if g.empty:
                    ax.set_axis_off()
                    continue
                x = g["step_id"].to_numpy(dtype=int)
                ax.plot(x, g["score"].to_numpy(dtype=float), color="tab:red" if int(g["label_episode_success"].iloc[0]) == 0 else "0.45", lw=1.5)
                ax.axhline(float(g["fold_threshold"].iloc[0]), color="black", ls="--", lw=0.8, alpha=0.8)
                onset_raw = g["t_onset"].iloc[0]
                if int(g["label_episode_success"].iloc[0]) == 0 and pd.notna(onset_raw):
                    onset = int(onset_raw)
                    ax.axvline(onset, color="tab:red", ls="-", lw=1.0)
                    ax.axvline(onset - int(horizon), color="tab:orange", ls=":", lw=1.0)
                if i == 0:
                    ax.set_title(name.replace("_", " "))
                if j == 0:
                    ax.set_ylabel(episode_key.split("/")[-1])
                if i == len(episode_keys) - 1:
                    ax.set_xlabel("step")
                ax.set_ylim(-0.02, 1.02)
                ax.grid(alpha=0.2)
        fig.suptitle(f"OpenDrawer OOF risk trajectories | H={horizon} | dashed=fold threshold, red=onset, orange=onset-H", y=0.998)
        fig.tight_layout()
        path = Path(path_pattern.format(horizon=horizon))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def run_ablations(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    onsets: pd.DataFrame,
    random_state: int = 0,
    n_bootstrap: int = BOOTSTRAP_DEFAULT,
    specs: Mapping[str, tuple[str, ...]] = ABLATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run grouped OOF A0--A4 for H=5,10,20 with train-fold thresholds."""

    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    onset_map = onsets.set_index("episode_key")["automatic_t_onset"].to_dict()
    result_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    max_horizons = list(HORIZONS)

    for horizon in max_horizons:
        lab = labels.loc[(labels["horizon"].eq(horizon)) & (labels["valid_mask"].eq(1))].copy()
        data = df.merge(lab[["episode_key", "step_id", "t_onset", "online_failure_label"]], on=["episode_key", "step_id"], how="inner")
        groups = data["episode_key"].to_numpy()
        y = data["online_failure_label"].to_numpy(dtype=int)
        n_splits = min(5, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits)

        for ablation_index, (name, group_names) in enumerate(specs.items()):
            cols = _feature_columns(data, group_names)
            oof = np.full(len(data), np.nan, dtype=float)
            fold_threshold = np.full(len(data), np.nan, dtype=float)
            fold_id = np.full(len(data), -1, dtype=int)
            for fold, (train_idx, test_idx) in enumerate(splitter.split(data, y, groups)):
                threshold = _inner_fold_threshold(data, train_idx, y, groups, cols, random_state + fold)
                model = _make_model(random_state + fold)
                model.fit(data.iloc[train_idx][cols], y[train_idx])
                oof[test_idx] = model.predict_proba(data.iloc[test_idx][cols])[:, 1]
                fold_threshold[test_idx] = threshold
                fold_id[test_idx] = fold

            scored = data[["episode_key", "step_id", "label_episode_success", "t_onset", "online_failure_label"]].copy()
            scored["score"] = oof
            scored["fold_threshold"] = fold_threshold
            scored["fold_id"] = fold_id
            scored["horizon"] = horizon
            scored["ablation"] = name
            scored["risk_group"] = scored.apply(lambda r: _risk_group(r, horizon), axis=1)
            score_frames.append(scored)

            step_auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) == 2 else float("nan")
            step_ap = float(average_precision_score(y, oof)) if y.sum() else float("nan")
            step_frame = scored[["episode_key", "score", "online_failure_label"]].rename(columns={"online_failure_label": "label"})
            step_auc_ci = _cluster_bootstrap_ci(step_frame, "score", "label", "auroc", n_bootstrap, random_state + 1000 + horizon + ablation_index)
            step_ap_ci = _cluster_bootstrap_ci(step_frame, "score", "label", "auprc", n_bootstrap, random_state + 2000 + horizon + ablation_index)

            ep_rows: list[dict[str, object]] = []
            for key, g in scored.groupby("episode_key", sort=True):
                g = g.sort_values("step_id")
                values = g["score"].to_numpy(dtype=float)
                ep_rows.append(
                    {
                        "episode_key": key,
                        "success": int(g["label_episode_success"].iloc[0]),
                        "score_max_valid": float(np.max(values)),
                        "score_mean_valid": float(np.mean(values)),
                        "score_last_valid": float(values[-1]),
                        "fold_threshold": float(g["fold_threshold"].iloc[0]),
                    }
                )
            ep = pd.DataFrame(ep_rows)
            episode_metrics: dict[str, dict[str, float]] = {}
            for agg in ("max_valid", "mean_valid", "last_valid"):
                score_col = f"score_{agg}"
                ep_label = (1 - ep["success"]).astype(int)
                episode_metrics[agg] = {
                    "auroc": float(roc_auc_score(ep_label, ep[score_col])),
                    "auprc": float(average_precision_score(ep_label, ep[score_col])),
                }
            ep_boot_frame = ep.rename(columns={"success": "success_label"}).copy()
            ep_boot_frame["label"] = 1 - ep_boot_frame["success_label"]
            ep_ci: dict[str, tuple[float, float]] = {}
            for agg in ("max_valid", "mean_valid", "last_valid"):
                ep_ci[f"{agg}_auroc"] = _cluster_bootstrap_ci(ep_boot_frame[["episode_key", f"score_{agg}", "label"]].rename(columns={f"score_{agg}": "score"}), "score", "label", "auroc", n_bootstrap, random_state + 3000 + horizon + ablation_index)
                ep_ci[f"{agg}_auprc"] = _cluster_bootstrap_ci(ep_boot_frame[["episode_key", f"score_{agg}", "label"]].rename(columns={f"score_{agg}": "score"}), "score", "label", "auprc", n_bootstrap, random_state + 4000 + horizon + ablation_index)

            false_alarm_flags = ep.loc[ep.success.eq(1), "score_max_valid"].to_numpy() >= ep.loc[ep.success.eq(1), "fold_threshold"].to_numpy()
            false_alarms = int(false_alarm_flags.sum())
            detected = 0
            premature = 0
            lead_times: list[float] = []
            for key, g in scored.groupby("episode_key", sort=True):
                if int(g["label_episode_success"].iloc[0]) == 1:
                    continue
                onset = int(onset_map[key])
                threshold = float(g["fold_threshold"].iloc[0])
                alert_steps = g.loc[g["score"].ge(threshold), "step_id"].astype(int)
                positive_start = onset - horizon
                window_alerts = alert_steps[(alert_steps >= positive_start) & (alert_steps < onset)]
                premature_alerts = alert_steps[alert_steps < positive_start]
                if not window_alerts.empty:
                    detected += 1
                    lead_times.append(float(onset - int(window_alerts.min())))
                if not premature_alerts.empty:
                    premature += 1

            thresholds = ep["fold_threshold"].to_numpy(dtype=float)
            result_rows.append(
                {
                    "horizon": horizon,
                    "ablation": name,
                    "features": ";".join(cols),
                    "n_features": len(cols),
                    "n_steps": len(data),
                    "n_episodes": len(ep),
                    "positive_steps": int(y.sum()),
                    "positive_prevalence": float(y.mean()),
                    "step_auroc_oof": step_auc,
                    "step_auroc_ci_low": step_auc_ci[0],
                    "step_auroc_ci_high": step_auc_ci[1],
                    "step_auprc_oof": step_ap,
                    "step_auprc_baseline": float(y.mean()),
                    "step_auprc_ci_low": step_ap_ci[0],
                    "step_auprc_ci_high": step_ap_ci[1],
                    "episode_auroc_max_valid_retro": episode_metrics["max_valid"]["auroc"],
                    "episode_auroc_max_ci_low": ep_ci["max_valid_auroc"][0],
                    "episode_auroc_max_ci_high": ep_ci["max_valid_auroc"][1],
                    "episode_auprc_max_valid_retro": episode_metrics["max_valid"]["auprc"],
                    "episode_auprc_max_baseline": float(ep_label.mean()),
                    "episode_auprc_max_ci_low": ep_ci["max_valid_auprc"][0],
                    "episode_auprc_max_ci_high": ep_ci["max_valid_auprc"][1],
                    "episode_auroc_mean_valid_retro": episode_metrics["mean_valid"]["auroc"],
                    "episode_auroc_last_valid_retro": episode_metrics["last_valid"]["auroc"],
                    "episode_auprc_mean_valid_retro": episode_metrics["mean_valid"]["auprc"],
                    "episode_auprc_last_valid_retro": episode_metrics["last_valid"]["auprc"],
                    "episode_score_aggregation_primary": "max_valid (retrospective any-alert score)",
                    "threshold_selection": "inner OOF within each outer train fold",
                    "threshold_median_train_fold": float(np.median(thresholds)),
                    "target_episode_far": 0.10,
                    "success_false_alarm_count": false_alarms,
                    "far_episode": false_alarms / max(1, int(ep.success.sum())),
                    "failure_detection_count_h_window": detected,
                    "failure_detection_rate_h_window": detected / max(1, int((ep.success.eq(0)).sum())),
                    "failure_premature_alert_count": premature,
                    "failure_premature_alert_rate": premature / max(1, int((ep.success.eq(0)).sum())),
                    "mean_lead_time_steps_h_window": float(np.mean(lead_times)) if lead_times else float("nan"),
                    "median_lead_time_steps_h_window": float(np.median(lead_times)) if lead_times else float("nan"),
                    "n_failures_with_h_window_alert": len(lead_times),
                }
            )

    scores = pd.concat(score_frames, ignore_index=True)
    risk_summary = _risk_distribution_summary(scores)
    meta = {
        "label_horizons": HORIZONS,
        "grouping": "episode_key",
        "cv": "5-fold outer GroupKFold; threshold from inner OOF predictions inside each outer train fold",
        "bootstrap": {"unit": "episode_key", "n": n_bootstrap, "ci": "percentile 95%"},
        "episode_score_note": "max_valid is retrospective any-alert aggregation; H-window detection and lead time only count alerts in [t_onset-H, t_onset-1].",
        "a0_note": "No action_logprob column exists in formal step_features.csv; A0 is a translation-magnitude proxy.",
        "feature_groups": FEATURE_GROUPS,
        "ablations": specs,
    }
    return pd.DataFrame(result_rows), scores, risk_summary, meta


def save_validation_plot(df: pd.DataFrame, validation: pd.DataFrame, path: Path, W: int, eps: float, min_start: int) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharey=True)
    for ax, (_, row) in zip(axes.flat, validation.iterrows()):
        g = df.loc[df["episode_key"].eq(row["episode_key"])].sort_values("step_id")
        x = g["step_id"].to_numpy()
        p = normalize_drawer_progress(g["label_drawer_progress"])
        ax.plot(x, p, lw=1.8, label="drawer progress")
        ax.plot(x, np.maximum.accumulate(p), lw=1.0, ls="--", alpha=0.7, label="best-so-far")
        auto = row["automatic_t_onset"]
        manual = row["manual_t_onset"]
        if auto != "":
            ax.axvline(float(auto), color="tab:red", ls="--", label=f"auto={int(auto)}")
        if manual != "":
            ax.axvline(float(manual), color="tab:green", ls=":", label=f"manual={int(manual)}")
        ax.set_title(f"{row['episode_key']} | {'success' if row['success'] else 'failure'}")
        ax.set_xlabel("step")
        ax.set_ylabel("progress")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle(f"OpenDrawer onset validation (W={W}, eps={eps}, min_start={min_start})", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(step_features: Path, output_dir: Path, n_bootstrap: int = BOOTSTRAP_DEFAULT) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _read_open_drawer(step_features)
    onsets = episode_onsets(df, W_DEFAULT, EPS_DEFAULT, MIN_START_DEFAULT)
    validation = make_validation_table(onsets)
    labels = make_online_labels(df, onsets, HORIZONS)
    ablations, oof_scores, risk_summary, metadata = run_ablations(df, labels, onsets, n_bootstrap=n_bootstrap)
    modalities, modality_scores, modality_risk_summary, modality_metadata = run_ablations(
        df, labels, onsets, n_bootstrap=n_bootstrap, specs=MODALITY_ABLATIONS
    )
    premature_detail, premature_episode, premature_summary = premature_alert_distances(modality_scores)
    detector_results, detector_premature_events = evaluate_detectors(modality_scores)

    outputs = {
        "onsets": output_dir / "opendrawer_onsets.csv",
        "validation": output_dir / "opendrawer_onset_validation.csv",
        "plot": output_dir / "opendrawer_onset_validation.png",
        "labels": output_dir / "online_labels_opendrawer.csv",
        "ablation": output_dir / "ablation_results_opendrawer.csv",
        "ablation_txt": output_dir / "ablation_results_opendrawer.txt",
        "oof_scores": output_dir / "opendrawer_oof_scores.csv",
        "risk_summary": output_dir / "risk_score_distributions_opendrawer.csv",
        "risk_plot": output_dir / "risk_score_distributions_opendrawer.png",
        "modality": output_dir / "modality_results_opendrawer.csv",
        "modality_txt": output_dir / "modality_results_opendrawer.txt",
        "modality_oof_scores": output_dir / "modality_oof_scores_opendrawer.csv",
        "modality_risk_summary": output_dir / "modality_risk_score_distributions_opendrawer.csv",
        "modality_risk_plot": output_dir / "modality_risk_score_distributions_opendrawer.png",
        "premature_detail": output_dir / "premature_alert_distances_opendrawer.csv",
        "premature_episode": output_dir / "premature_episode_first_crossing_opendrawer.csv",
        "premature_summary": output_dir / "premature_alert_distance_summary_opendrawer.csv",
        "detectors": output_dir / "detector_results_opendrawer.csv",
        "detector_premature_events": output_dir / "detector_premature_alert_events_opendrawer.csv",
        "metadata": output_dir / "opendrawer_analysis_metadata.json",
    }
    onsets.to_csv(outputs["onsets"], index=False)
    validation.to_csv(outputs["validation"], index=False)
    labels.to_csv(outputs["labels"], index=False)
    ablations.to_csv(outputs["ablation"], index=False)
    oof_scores.to_csv(outputs["oof_scores"], index=False)
    risk_summary.to_csv(outputs["risk_summary"], index=False)
    modalities.to_csv(outputs["modality"], index=False)
    outputs["modality_txt"].write_text(modalities.to_string(index=False) + "\n", encoding="utf-8")
    modality_scores.to_csv(outputs["modality_oof_scores"], index=False)
    modality_risk_summary.to_csv(outputs["modality_risk_summary"], index=False)
    premature_detail.to_csv(outputs["premature_detail"], index=False)
    premature_episode.to_csv(outputs["premature_episode"], index=False)
    premature_summary.to_csv(outputs["premature_summary"], index=False)
    detector_results.to_csv(outputs["detectors"], index=False)
    detector_premature_events.to_csv(outputs["detector_premature_events"], index=False)
    save_validation_plot(df, validation, outputs["plot"], W_DEFAULT, EPS_DEFAULT, MIN_START_DEFAULT)
    save_risk_distribution_plot(oof_scores, outputs["risk_plot"])
    save_risk_distribution_plot(modality_scores, outputs["modality_risk_plot"], specs=MODALITY_ABLATIONS)
    trajectory_paths = save_risk_trajectory_plots(
        modality_scores,
        str(output_dir / "risk_trajectories_modality_H{horizon}.png"),
        MODALITY_ABLATIONS,
    )
    for horizon, trajectory_path in zip(HORIZONS, trajectory_paths):
        outputs[f"risk_trajectory_H{horizon}"] = trajectory_path
    outputs["ablation_txt"].write_text(ablations.to_string(index=False) + "\n", encoding="utf-8")
    metadata.update(
        {
            "step_features": str(step_features),
            "parameters": {"W": W_DEFAULT, "eps": EPS_DEFAULT, "min_start": MIN_START_DEFAULT, "horizons": HORIZONS, "n_bootstrap": n_bootstrap},
            "n_open_drawer_steps": int(len(df)),
            "n_open_drawer_episodes": int(df["episode_key"].nunique()),
            "validation_episodes": list(MANUAL_VALIDATION_ONSETS),
            "manual_annotation_note": "Initial curve review; values are frozen and never tuned on AUROC.",
            "formal_modality_ablations": modality_metadata,
            "premature_alert_definition": "score >= outer-fold threshold at t < t_onset-H; distance_to_onset=t_onset-t",
            "detectors": DETECTORS,
            "detector_note": "T0/T1/T2 apply fixed raw-score thresholds selected in the training fold; no smoothing or persistence parameter is tuned on held-out episodes.",
        }
    )
    outputs["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-features", type=Path, default=Path("data/formal_features/step_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/opendrawer_analysis"))
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_DEFAULT)
    args = parser.parse_args()
    outputs = run(args.step_features, args.output_dir, n_bootstrap=args.n_bootstrap)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
