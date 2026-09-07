#!/usr/bin/env python3
"""
Extract online-safe temporal features from MolmoAct rollout logs.

Reads rollout directories containing:
    condition_XXXX/
        steps.jsonl
        summary.json
        condition.json   (optional)

Writes:
    step_features.csv     # one row per control step
    episode_features.csv  # one row per rollout
    feature_metadata.json

Important:
- `feat_*` columns are computed only from the current pre-action output and
  previous timesteps, so they are candidates for online failure prediction.
- `label_*` and `post_*` columns are supervision / analysis fields and should
  NOT be used as online predictor inputs.

Example:
python analysis/extract_features.py \
  --input-root /root/autodl-tmp/molmoact_project/results/debug \
  --output-dir /root/autodl-tmp/molmoact_project/data/derived/pilot_features
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DRAWER_SUCCESS_QPOS = 0.15
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Depth / Trace / Action temporal features from MolmoAct rollouts."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-window", type=int, default=3)
    parser.add_argument("--trace-resample-points", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.rolling_window <= 0:
        parser.error("--rolling-window must be > 0")
    if args.trace_resample_points < 2:
        parser.error("--trace-resample-points must be >= 2")
    return args


def warn(msg: str) -> None:
    print(f"WARN: {msg}")


def safe_float(x: Any) -> float:
    if x is None:
        return math.nan
    try:
        arr = np.asarray(x)
        if arr.size != 1:
            return math.nan
        return float(arr.reshape(-1)[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return math.nan


def safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return None


def flatten_singletons(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        return None
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None or a.shape != b.shape:
        return math.nan
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return math.nan
    return float(np.dot(a, b) / (na * nb))


def rolling_mean(values: deque) -> float:
    valid = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(valid)) if valid else math.nan


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        warn(f"No rows to write: {path}")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------- Depth ---------------------------------

def parse_depth_tokens(depth: Any) -> List[int]:
    if depth is None:
        return []

    if isinstance(depth, list):
        if not depth:
            return []
        if all(isinstance(x, (int, np.integer)) for x in depth):
            return [int(x) for x in depth]
        if len(depth) == 1 and isinstance(depth[0], str):
            depth = depth[0]
        elif all(isinstance(x, str) for x in depth):
            depth = "".join(depth)

    if isinstance(depth, str):
        return [int(x) for x in re.findall(r"<DEPTH_(\d+)>", depth)]

    return []


def categorical_entropy(tokens: Sequence[int]) -> float:
    if not tokens:
        return math.nan
    _, counts = np.unique(np.asarray(tokens, dtype=np.int64), return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-(p * np.log2(p + EPS)).sum())


def infer_grid_hw(n: int) -> Optional[Tuple[int, int]]:
    if n <= 0:
        return None
    r = int(math.isqrt(n))
    if r * r == n:
        return r, r
    for h in range(r, 0, -1):
        if n % h == 0:
            return h, n // h
    return None


def depth_spatial_boundary_ratio(tokens: Sequence[int]) -> float:
    if not tokens:
        return math.nan
    shape = infer_grid_hw(len(tokens))
    if shape is None:
        return math.nan

    h, w = shape
    grid = np.asarray(tokens, dtype=np.int64).reshape(h, w)

    changed = 0
    total = 0

    if w > 1:
        x = grid[:, 1:] != grid[:, :-1]
        changed += int(x.sum())
        total += int(x.size)

    if h > 1:
        x = grid[1:, :] != grid[:-1, :]
        changed += int(x.sum())
        total += int(x.size)

    return float(changed / total) if total else 0.0


def depth_change_metrics(
    current: Sequence[int],
    previous: Optional[Sequence[int]],
) -> Tuple[float, float]:
    if previous is None or not current or not previous:
        return math.nan, math.nan
    if len(current) != len(previous):
        return math.nan, math.nan

    a = np.asarray(current, dtype=np.int64)
    b = np.asarray(previous, dtype=np.int64)
    diff = a != b
    return float(diff.sum()), float(diff.mean())


# ----------------------------- Trace ---------------------------------

def parse_trace(trace: Any) -> Optional[np.ndarray]:
    if trace is None:
        return None

    try:
        arr = np.asarray(trace, dtype=np.float64)
    except Exception:
        return None

    # MolmoAct often returns:
    # [[[x1, y1], [x2, y2], ...]]
    #
    # Remove only outer wrapper dimensions.
    # Do NOT squeeze a valid one-point trace (1, 2) into (2,).
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    # Defensive support for [x, y]
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)

    if arr.ndim != 2:
        return None

    if arr.shape[1] != 2:
        return None

    if arr.shape[0] < 1:
        return None

    return arr.astype(np.float64)

def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def trace_straightness(points: np.ndarray) -> float:
    if len(points) < 2:
        return math.nan
    path_len = polyline_length(points)
    if path_len < EPS:
        return math.nan
    endpoint_dist = float(np.linalg.norm(points[-1] - points[0]))
    return endpoint_dist / path_len


def polygon_area(points: np.ndarray) -> float:
    """Absolute shoelace area of the trace. Captures 'trace_spread'."""
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def trace_spread(points: np.ndarray) -> float:
    """Spread of trace points as area of their bounding-rectangle diagonal."""
    if len(points) < 2:
        return 0.0
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    d = mx - mn
    return float(np.linalg.norm(d))


def trace_curvature(points: np.ndarray) -> float:
    """Path length / endpoint straight-line distance (>1 means curved)."""
    if len(points) < 2:
        return math.nan
    path_len = polyline_length(points)
    if path_len < EPS:
        return math.nan
    endpoint_dist = float(np.linalg.norm(points[-1] - points[0]))
    if endpoint_dist < EPS:
        return math.nan
    return path_len / endpoint_dist


def mean_turn_angle(points: np.ndarray) -> float:
    if len(points) < 3:
        return math.nan

    seg = points[1:] - points[:-1]
    angles: List[float] = []

    for i in range(len(seg) - 1):
        a, b = seg[i], seg[i + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < EPS or nb < EPS:
            continue
        cosv = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        angles.append(float(np.arccos(cosv)))

    return float(np.mean(angles)) if angles else math.nan


def planning_vector(points: np.ndarray) -> Optional[np.ndarray]:
    """v_t = p^end - p^start (H1 relative planning vector)."""
    if points is None or len(points) < 2:
        return None
    return (points[-1] - points[0]).astype(np.float64)


def trace_direction_angle(points: np.ndarray) -> float:
    if len(points) < 2:
        return math.nan
    vec = points[-1] - points[0]
    if np.linalg.norm(vec) < EPS:
        return math.nan
    return float(np.arctan2(vec[1], vec[0]))


def wrapped_angle_difference(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return math.nan
    d = a - b
    return float((d + math.pi) % (2 * math.pi) - math.pi)


def resample_polyline(points: np.ndarray, n: int) -> Optional[np.ndarray]:
    if len(points) == 0:
        return None
    if len(points) == 1:
        return np.repeat(points, n, axis=0)

    seg_len = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])

    if total < EPS:
        return np.repeat(points[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float64)

    for j, t in enumerate(targets):
        idx = np.searchsorted(cum, t, side="right") - 1
        idx = min(max(idx, 0), len(points) - 2)

        left, right = cum[idx], cum[idx + 1]
        alpha = 0.0 if right - left < EPS else (t - left) / (right - left)

        out[j] = (1.0 - alpha) * points[idx] + alpha * points[idx + 1]

    return out


def trace_temporal_metrics(
    current: Optional[np.ndarray],
    previous: Optional[np.ndarray],
    resample_points: int,
) -> Dict[str, float]:
    default = {
        "endpoint_drift": math.nan,
        "centroid_drift": math.nan,
        "shape_drift": math.nan,
        "direction_change_rad": math.nan,
        "startpoint_drift": math.nan,
        "relplan_drift": math.nan,
        "relplan_drift_norm": math.nan,
        "relplan_cos": math.nan,
    }

    if current is None or previous is None:
        return default

    endpoint_drift = float(np.linalg.norm(current[-1] - previous[-1]))
    startpoint_drift = float(np.linalg.norm(current[0] - previous[0]))
    centroid_drift = float(
        np.linalg.norm(current.mean(axis=0) - previous.mean(axis=0))
    )

    cr = resample_polyline(current, resample_points)
    pr = resample_polyline(previous, resample_points)
    shape_drift = (
        float(np.linalg.norm(cr - pr, axis=1).mean())
        if cr is not None and pr is not None
        else math.nan
    )

    ca = trace_direction_angle(current)
    pa = trace_direction_angle(previous)
    direction_change = abs(wrapped_angle_difference(ca, pa))

    # H1: relative planning-vector drift (v_t = p^end - p^start), which
    # removes the whole-image translation of the gripper/scene frame.
    cv = planning_vector(current)
    pv = planning_vector(previous)
    relplan_drift = math.nan
    relplan_drift_norm = math.nan
    relplan_cos = math.nan
    if cv is not None and pv is not None:
        diff = cv - pv
        relplan_drift = float(np.linalg.norm(diff))
        vnorm = float(np.linalg.norm(cv))
        relplan_drift_norm = relplan_drift / vnorm if vnorm > EPS else math.nan
        relplan_cos = cosine_similarity(cv, pv)

    return {
        "endpoint_drift": endpoint_drift,
        "centroid_drift": centroid_drift,
        "shape_drift": shape_drift,
        "direction_change_rad": direction_change,
        "startpoint_drift": startpoint_drift,
        "relplan_drift": relplan_drift,
        "relplan_drift_norm": relplan_drift_norm,
        "relplan_cos": relplan_cos,
    }


# ----------------------------- Action --------------------------------

def parse_action(action: Any) -> Optional[np.ndarray]:
    arr = flatten_singletons(action)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return arr if arr.size else None


def split_7d_action(
    action: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    if action is None or len(action) < 7:
        return None, None, math.nan
    return action[:3].copy(), action[3:6].copy(), float(action[6])


def action_static_metrics(action: Optional[np.ndarray]) -> Dict[str, float]:
    trans, rot, grip = split_7d_action(action)
    return {
        "translation_norm": (
            float(np.linalg.norm(trans)) if trans is not None else math.nan
        ),
        "rotation_norm": (
            float(np.linalg.norm(rot)) if rot is not None else math.nan
        ),
        "gripper": grip,
    }


def action_temporal_metrics(
    current: Optional[np.ndarray],
    previous: Optional[np.ndarray],
) -> Dict[str, float]:
    default = {
        "full_jump_l2": math.nan,
        "translation_jump_l2": math.nan,
        "rotation_jump_l2": math.nan,
        "translation_cosine_prev": math.nan,
        "translation_direction_flip": math.nan,
        "gripper_change": math.nan,
    }

    if current is None or previous is None:
        return default

    if len(current) < 7 or len(previous) < 7:
        if current.shape == previous.shape:
            default["full_jump_l2"] = float(np.linalg.norm(current - previous))
        return default

    c_t, p_t = current[:3], previous[:3]
    c_r, p_r = current[3:6], previous[3:6]

    cosv = cosine_similarity(c_t, p_t)

    return {
        "full_jump_l2": float(np.linalg.norm(current - previous)),
        "translation_jump_l2": float(np.linalg.norm(c_t - p_t)),
        "rotation_jump_l2": float(np.linalg.norm(c_r - p_r)),
        "translation_cosine_prev": cosv,
        "translation_direction_flip": (
            float(cosv < 0.0) if not math.isnan(cosv) else math.nan
        ),
        "gripper_change": float(abs(float(current[6]) - float(previous[6]))),
    }


# ----------------------- Task-specific labels -------------------------

def extract_post_labels(post: Dict[str, Any]) -> Dict[str, Any]:
    info = post.get("info") or {}

    out: Dict[str, Any] = {
        "post_reward": safe_float(post.get("reward")),
        "post_terminated": safe_bool(post.get("terminated")),
        "post_truncated": safe_bool(post.get("truncated")),
        "post_predicted_terminated": safe_bool(post.get("predicted_terminated")),
        "post_env_success": safe_bool(info.get("success")),
    }

    qpos = safe_float(info.get("qpos"))
    out["post_drawer_qpos"] = qpos
    out["label_drawer_progress"] = (
        float(np.clip(qpos / DRAWER_SUCCESS_QPOS, 0.0, 1.0))
        if not math.isnan(qpos)
        else math.nan
    )

    out["post_is_grasped"] = safe_bool(info.get("is_grasped"))
    out["post_consecutive_grasp"] = safe_bool(info.get("consecutive_grasp"))
    out["post_lifted_object"] = safe_bool(info.get("lifted_object"))
    out["post_lifted_object_significantly"] = safe_bool(
        info.get("lifted_object_significantly")
    )

    coke_stage = 0
    if out["post_is_grasped"]:
        coke_stage = max(coke_stage, 1)
    if out["post_lifted_object"]:
        coke_stage = max(coke_stage, 2)
    if out["post_lifted_object_significantly"]:
        coke_stage = max(coke_stage, 3)
    if out["post_env_success"]:
        coke_stage = max(coke_stage, 4)

    out["label_coke_stage"] = coke_stage
    return out


# ----------------------------- I/O ------------------------------------

def rollout_dirs(input_root: Path) -> List[Path]:
    dirs = []
    for steps_path in input_root.rglob("steps.jsonl"):
        ep_dir = steps_path.parent
        if ep_dir.name.startswith("condition_"):
            dirs.append(ep_dir)
    return sorted(set(dirs))


def load_steps(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_no}: {exc}"
                ) from exc
    return rows


def extract_episode(
    ep_dir: Path,
    rolling_window: int,
    trace_resample_points: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary_path = ep_dir / "summary.json"
    steps_path = ep_dir / "steps.jsonl"

    summary = read_json(summary_path) if summary_path.exists() else {}
    steps = load_steps(steps_path)

    if not steps:
        raise ValueError(f"No valid steps in {steps_path}")

    first = steps[0]
    task = str(summary.get("task", first.get("task", "unknown")))
    condition_id = int(
        summary.get("condition_id", first.get("condition_id", -1))
    )

    final_success = bool(summary.get("success", False))
    final_failure = int(not final_success)
    num_steps = int(summary.get("num_steps", len(steps)))
    max_steps = int(summary.get("max_steps", num_steps))

    previous_depth: Optional[List[int]] = None
    previous_trace: Optional[np.ndarray] = None
    previous_parsed_action: Optional[np.ndarray] = None
    previous_env_action: Optional[np.ndarray] = None

    depth_change_hist: deque = deque(maxlen=rolling_window)
    trace_shape_hist: deque = deque(maxlen=rolling_window)
    action_jump_hist: deque = deque(maxlen=rolling_window)

    step_rows: List[Dict[str, Any]] = []

    max_drawer_qpos = -math.inf
    max_drawer_progress = -math.inf

    ever_grasped = False
    ever_lifted = False
    ever_lifted_significant = False
    max_coke_stage = 0

    for local_idx, item in enumerate(steps):
        pre = item.get("pre_action") or {}
        post = item.get("post_action") or {}
        step_id = int(item.get("step_id", local_idx))

        # Depth
        depth = parse_depth_tokens(pre.get("depth"))
        depth_hamming, depth_change_ratio = depth_change_metrics(
            depth, previous_depth
        )

        # Trace
        trace = parse_trace(pre.get("trace"))
        if trace is None:
            trace_n = 0
            trace_start_x = trace_start_y = math.nan
            trace_end_x = trace_end_y = math.nan
            trace_len = trace_straight = trace_turn = trace_angle = math.nan
            trace_spread_v = trace_curv_v = math.nan
        else:
            trace_n = int(len(trace))
            trace_start_x, trace_start_y = map(float, trace[0])
            trace_end_x, trace_end_y = map(float, trace[-1])
            trace_len = polyline_length(trace)
            trace_straight = trace_straightness(trace)
            trace_turn = mean_turn_angle(trace)
            trace_angle = trace_direction_angle(trace)
            trace_spread_v = trace_spread(trace)
            trace_curv_v = trace_curvature(trace)

        trace_temporal = trace_temporal_metrics(
            trace, previous_trace, trace_resample_points
        )

        # Actions
        parsed_action = parse_action(pre.get("parsed_action"))
        env_action = parse_action(pre.get("env_action"))

        parsed_static = action_static_metrics(parsed_action)
        parsed_temporal = action_temporal_metrics(
            parsed_action, previous_parsed_action
        )

        env_static = action_static_metrics(env_action)
        env_temporal = action_temporal_metrics(
            env_action, previous_env_action
        )

        # Rolling histories use only current/past values.
        depth_change_hist.append(depth_change_ratio)
        trace_shape_hist.append(trace_temporal["shape_drift"])
        action_jump_hist.append(env_temporal["full_jump_l2"])

        # Post-action supervision
        post_labels = extract_post_labels(post)

        qpos = post_labels["post_drawer_qpos"]
        if not math.isnan(qpos):
            max_drawer_qpos = max(max_drawer_qpos, qpos)

        drawer_progress = post_labels["label_drawer_progress"]
        if not math.isnan(drawer_progress):
            max_drawer_progress = max(max_drawer_progress, drawer_progress)

        ever_grasped |= bool(post_labels.get("post_is_grasped"))
        ever_lifted |= bool(post_labels.get("post_lifted_object"))
        ever_lifted_significant |= bool(
            post_labels.get("post_lifted_object_significantly")
        )
        max_coke_stage = max(
            max_coke_stage,
            int(post_labels.get("label_coke_stage", 0)),
        )

        row = {
            # Identity / grouping
            "task": task,
            "condition_id": condition_id,
            "episode_key": f"{task}/condition_{condition_id:04d}",
            "step_id": step_id,
            "source_dir": str(ep_dir),

            # Episode labels (future supervision only)
            "label_episode_success": int(final_success),
            "label_episode_failure": final_failure,
            "label_episode_num_steps": num_steps,
            "label_episode_max_steps": max_steps,

            # Depth features
            "feat_depth_token_count": len(depth),
            "feat_depth_unique_count": len(set(depth)) if depth else 0,
            "feat_depth_token_entropy": categorical_entropy(depth),
            "feat_depth_spatial_boundary_ratio": depth_spatial_boundary_ratio(
                depth
            ),
            "feat_depth_hamming_prev": depth_hamming,
            "feat_depth_change_ratio_prev": depth_change_ratio,
            "feat_depth_change_ratio_rollmean": rolling_mean(
                depth_change_hist
            ),

            # Trace features
            "feat_trace_num_points": trace_n,
            "feat_trace_start_x": trace_start_x,
            "feat_trace_start_y": trace_start_y,
            "feat_trace_endpoint_x": trace_end_x,
            "feat_trace_endpoint_y": trace_end_y,
            "feat_trace_polyline_length": trace_len,
            "feat_trace_straightness": trace_straight,
            "feat_trace_mean_turn_angle_rad": trace_turn,
            "feat_trace_direction_angle_rad": trace_angle,
            "feat_trace_spread": trace_spread_v,
            "feat_trace_curvature": trace_curv_v,
            "feat_trace_endpoint_drift_prev": trace_temporal[
                "endpoint_drift"
            ],
            "feat_trace_startpoint_drift_prev": trace_temporal[
                "startpoint_drift"
            ],
            "feat_trace_centroid_drift_prev": trace_temporal[
                "centroid_drift"
            ],
            "feat_trace_shape_drift_prev": trace_temporal["shape_drift"],
            "feat_trace_direction_change_prev_rad": trace_temporal[
                "direction_change_rad"
            ],
            "feat_trace_relplan_drift_prev": trace_temporal["relplan_drift"],
            "feat_trace_relplan_drift_norm_prev": trace_temporal[
                "relplan_drift_norm"
            ],
            "feat_trace_relplan_cos_prev": trace_temporal["relplan_cos"],
            "feat_trace_shape_drift_rollmean": rolling_mean(
                trace_shape_hist
            ),

            # Model/raw action features
            "feat_model_translation_norm": parsed_static["translation_norm"],
            "feat_model_rotation_norm": parsed_static["rotation_norm"],
            "feat_model_gripper": parsed_static["gripper"],
            "feat_model_action_jump_prev_l2": parsed_temporal[
                "full_jump_l2"
            ],
            "feat_model_translation_jump_prev_l2": parsed_temporal[
                "translation_jump_l2"
            ],
            "feat_model_rotation_jump_prev_l2": parsed_temporal[
                "rotation_jump_l2"
            ],
            "feat_model_translation_cosine_prev": parsed_temporal[
                "translation_cosine_prev"
            ],
            "feat_model_translation_direction_flip": parsed_temporal[
                "translation_direction_flip"
            ],
            "feat_model_gripper_change_prev": parsed_temporal[
                "gripper_change"
            ],

            # Executed action features
            "feat_env_translation_norm": env_static["translation_norm"],
            "feat_env_rotation_norm": env_static["rotation_norm"],
            "feat_env_gripper": env_static["gripper"],
            "feat_env_action_jump_prev_l2": env_temporal["full_jump_l2"],
            "feat_env_translation_jump_prev_l2": env_temporal[
                "translation_jump_l2"
            ],
            "feat_env_rotation_jump_prev_l2": env_temporal[
                "rotation_jump_l2"
            ],
            "feat_env_translation_cosine_prev": env_temporal[
                "translation_cosine_prev"
            ],
            "feat_env_translation_direction_flip": env_temporal[
                "translation_direction_flip"
            ],
            "feat_env_gripper_change_prev": env_temporal["gripper_change"],
            "feat_env_action_jump_rollmean": rolling_mean(
                action_jump_hist
            ),

            # Diagnostics
            "diag_inference_time_s": safe_float(
                pre.get("inference_time_s")
            ),

            # Post-action supervision / task progress
            **post_labels,
        }

        step_rows.append(row)

        previous_depth = list(depth)
        previous_trace = trace.copy() if trace is not None else None
        previous_parsed_action = (
            parsed_action.copy() if parsed_action is not None else None
        )
        previous_env_action = (
            env_action.copy() if env_action is not None else None
        )

    if max_drawer_qpos == -math.inf:
        max_drawer_qpos = math.nan
    if max_drawer_progress == -math.inf:
        max_drawer_progress = math.nan

    episode_row = {
        "task": task,
        "condition_id": condition_id,
        "episode_key": f"{task}/condition_{condition_id:04d}",
        "source_dir": str(ep_dir),
        "label_episode_success": int(final_success),
        "label_episode_failure": final_failure,
        "label_episode_num_steps": num_steps,
        "label_episode_max_steps": max_steps,
        "label_stop_reason": summary.get("stop_reason"),
        "label_elapsed_s": safe_float(summary.get("elapsed_s")),
        "label_drawer_max_qpos": max_drawer_qpos,
        "label_drawer_max_progress": max_drawer_progress,
        "label_coke_ever_grasped": int(ever_grasped),
        "label_coke_ever_lifted": int(ever_lifted),
        "label_coke_ever_lifted_significantly": int(
            ever_lifted_significant
        ),
        "label_coke_max_stage": max_coke_stage,
    }

    return step_rows, episode_row


def summarize_dataset(
    step_rows: List[Dict[str, Any]],
    episode_rows: List[Dict[str, Any]],
) -> None:
    print("\n===== FEATURE EXTRACTION SUMMARY =====")
    print("episodes:", len(episode_rows))
    print("steps:", len(step_rows))

    tasks = sorted(set(x["task"] for x in episode_rows))

    for task in tasks:
        eps = [x for x in episode_rows if x["task"] == task]
        n = len(eps)
        success = sum(int(x["label_episode_success"]) for x in eps)
        failure = n - success
        print(
            f"{task:20s} "
            f"N={n:4d} "
            f"success={success:4d} "
            f"failure={failure:4d} "
            f"SR={(success / n if n else 0):.3f}"
        )

    if step_rows:
        depth_missing = sum(
            int(x["feat_depth_token_count"] == 0)
            for x in step_rows
        )
        trace_missing = sum(
            int(x["feat_trace_num_points"] == 0)
            for x in step_rows
        )
        print("\nParser health:")
        print(f"depth missing: {depth_missing}/{len(step_rows)}")
        print(f"trace missing: {trace_missing}/{len(step_rows)}")


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = rollout_dirs(input_root)

    print("Input root:", input_root)
    print("Found rollout directories:", len(episode_dirs))

    if not episode_dirs:
        raise RuntimeError(
            f"No condition_*/steps.jsonl found below {input_root}"
        )

    all_step_rows: List[Dict[str, Any]] = []
    all_episode_rows: List[Dict[str, Any]] = []
    skipped = 0

    for i, ep_dir in enumerate(episode_dirs, start=1):
        try:
            step_rows, episode_row = extract_episode(
                ep_dir=ep_dir,
                rolling_window=args.rolling_window,
                trace_resample_points=args.trace_resample_points,
            )

            all_step_rows.extend(step_rows)
            all_episode_rows.append(episode_row)

            print(
                f"[{i}/{len(episode_dirs)}] "
                f"{episode_row['episode_key']} "
                f"steps={episode_row['label_episode_num_steps']} "
                f"success={bool(episode_row['label_episode_success'])}"
            )

        except Exception as exc:
            skipped += 1
            msg = (
                f"Skipping {ep_dir}: "
                f"{type(exc).__name__}: {exc}"
            )
            if args.strict:
                raise RuntimeError(msg) from exc
            warn(msg)

    step_csv = output_dir / "step_features.csv"
    episode_csv = output_dir / "episode_features.csv"

    write_csv(step_csv, all_step_rows)
    write_csv(episode_csv, all_episode_rows)

    metadata = {
        "input_root": str(input_root),
        "num_rollouts_found": len(episode_dirs),
        "num_rollouts_extracted": len(all_episode_rows),
        "num_rollouts_skipped": skipped,
        "num_steps_extracted": len(all_step_rows),
        "rolling_window": args.rolling_window,
        "trace_resample_points": args.trace_resample_points,
        "drawer_success_qpos": DRAWER_SUCCESS_QPOS,
        "feature_prefix": "feat_",
        "label_prefix": "label_",
        "post_prefix": "post_",
        "note": (
            "Only feat_* columns are intended as online predictor inputs. "
            "label_* and post_* columns are supervision/analysis fields."
        ),
    }

    with (output_dir / "feature_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    summarize_dataset(all_step_rows, all_episode_rows)

    print("\nSaved:")
    print(" ", step_csv)
    print(" ", episode_csv)
    print(" ", output_dir / "feature_metadata.json")


if __name__ == "__main__":
    main()
