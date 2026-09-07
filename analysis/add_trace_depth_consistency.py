#!/usr/bin/env python3
"""
Add decoder-based Trace–Depth consistency features to an existing
step_features.csv without modifying the source CSV.

Design
------
For every rollout step:

    100 generated <DEPTH_i> tokens
        -> 10x10 VQ code grid
        -> official MolmoAct VQ-VAE decoder
        -> raw reconstructed relative-depth map [1, 1, 320, 320]
        -> sample depth along the predicted 2D Trace

Important:
- Raw VQ token IDs are categorical codebook indices and are NOT treated as
  continuous depth values.
- We use the decoder output BEFORE the official reconstruction script's
  per-image min-max normalization.
- Trace path features use 20-point arc-length resampling.
- A single-point Trace is treated as structural degeneration:
  path std/gradient/range are NaN, while endpoint features remain valid.
- "*_drift_prev" is always computed against the immediately preceding step
  t-1 in the SAME episode. If t-1 is unavailable/invalid, drift is NaN.
- The source step_features.csv is never overwritten.

Main added online-safe exploratory features
-------------------------------------------
feat_trace_depth_std_norm
    std(depth values sampled along resampled Trace) / std(full depth map)

feat_trace_depth_gradient_norm
    mean absolute depth difference between adjacent resampled Trace points)
    / std(full depth map)

feat_trace_depth_range_norm
    (max(trace depth) - min(trace depth)) / std(full depth map)

feat_trace_endpoint_depth_percentile
    Percentile rank of the Trace endpoint depth within the current decoded
    relative-depth map, in [0, 1].

feat_trace_endpoint_depth_percentile_drift_prev
    Absolute change of endpoint depth percentile from t-1 to t.

Diagnostics (not intended as primary predictor inputs)
------------------------------------------------------
diag_trace_endpoint_depth_raw
diag_trace_endpoint_depth_raw_drift_prev

Example
-------
python analysis/add_trace_depth_consistency.py \
  --step-features /root/autodl-tmp/molmoact_project/data/derived/formal_features/step_features.csv \
  --output /root/autodl-tmp/molmoact_project/data/derived/formal_features/step_features_trace_depth.csv \
  --ckpt-path /root/autodl-tmp/molmoact_project/models/depth_vqvae/vae-final.pt \
  --molmoact-scripts /root/autodl-tmp/molmoact_project/code/molmoact/scripts \
  --batch-size 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


FEATURE_COLUMNS = [
    "feat_trace_depth_std_norm",
    "feat_trace_depth_gradient_norm",
    "feat_trace_depth_range_norm",
    "feat_trace_endpoint_depth_percentile",
    "feat_trace_endpoint_depth_percentile_drift_prev",
]

DIAG_COLUMNS = [
    "diag_trace_endpoint_depth_raw",
    "diag_trace_endpoint_depth_raw_drift_prev",
]

ALL_NEW_COLUMNS = FEATURE_COLUMNS + DIAG_COLUMNS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Decode MolmoAct DEPTH tokens with the official VQ-VAE and add "
            "Trace-Depth consistency features to step_features.csv."
        )
    )
    p.add_argument(
        "--step-features",
        required=True,
        type=Path,
        help="Existing step_features.csv. This file is never modified.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path. Must differ from --step-features.",
    )
    p.add_argument(
        "--ckpt-path",
        required=True,
        type=Path,
        help="Path to vae-final.pt.",
    )
    p.add_argument(
        "--molmoact-scripts",
        required=True,
        type=Path,
        help=(
            "Official MolmoAct scripts directory containing "
            "reconstruct_from_tokens.py and vqvae.py."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of 10x10 code grids decoded per VQ-VAE batch. Default: 16.",
    )
    p.add_argument(
        "--resample-points",
        type=int,
        default=20,
        help="Arc-length-resampled Trace point count. Default: 20.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon for normalization. Default: 1e-8.",
    )
    p.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "If the output feature names already exist in the INPUT CSV, "
            "drop and recompute them. Without this flag, the script refuses."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Abort immediately on the first malformed/missing rollout step.",
    )
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    return value


def parse_trace(trace: Any) -> Optional[np.ndarray]:
    """
    Accept common MolmoAct trace shapes:
      [[[x,y], ...]]
      [[x,y], ...]
      [x,y]                 # one-point trace after singleton squeezing
    Returns float64 array [N,2], N>=1, or None.
    """
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

    if not np.isfinite(arr).all():
        return None

    return arr.astype(np.float64, copy=False)


def remove_consecutive_duplicate_points(
    points: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    if len(points) <= 1:
        return points

    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > eps
    return points[keep]


def resample_trace_arc_length(
    points: np.ndarray, n_points: int, eps: float = 1e-8
) -> Optional[np.ndarray]:
    """
    Resample a polyline to n_points equally spaced in arc length.

    Returns None for single-point / zero-length traces because path variation,
    gradient and range are undefined for structural degeneration.
    """
    if points is None or len(points) < 2:
        return None

    points = remove_consecutive_duplicate_points(points, eps=eps)
    if len(points) < 2:
        return None

    seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(seg_len.sum())
    if not np.isfinite(total) or total <= eps:
        return None

    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    targets = np.linspace(0.0, total, n_points)

    out = np.empty((n_points, 2), dtype=np.float64)
    j = 0
    for i, s in enumerate(targets):
        while j < len(seg_len) - 1 and s > cum[j + 1]:
            j += 1

        denom = max(float(seg_len[j]), eps)
        alpha = float((s - cum[j]) / denom)
        out[i] = points[j] + alpha * (points[j + 1] - points[j])

    return out


def bilinear_sample_255_coords(
    depth_map: np.ndarray, points_xy: np.ndarray
) -> np.ndarray:
    """
    Sample decoded depth using Trace coordinates defined on [0,255]^2.

    Trace x,y -> decoded map u,v:
        u = x / 255 * (W-1)
        v = y / 255 * (H-1)

    Uses vectorized bilinear interpolation.
    """
    if points_xy is None or len(points_xy) == 0:
        return np.empty((0,), dtype=np.float64)

    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D depth map, got {depth.shape}")

    h, w = depth.shape

    x = np.clip(points_xy[:, 0], 0.0, 255.0)
    y = np.clip(points_xy[:, 1], 0.0, 255.0)

    u = x / 255.0 * (w - 1)
    v = y / 255.0 * (h - 1)

    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    dx = u - x0
    dy = v - y0

    vals = (
        depth[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + depth[y0, x1] * dx * (1.0 - dy)
        + depth[y1, x0] * (1.0 - dx) * dy
        + depth[y1, x1] * dx * dy
    )

    return vals.astype(np.float64, copy=False)


def endpoint_percentile(depth_map: np.ndarray, endpoint_depth: float) -> float:
    """
    Percentile rank of endpoint_depth in the CURRENT relative-depth map.

    This is intentionally per-frame and rank-based, making temporal endpoint
    comparison more robust than comparing raw decoder values directly.
    """
    flat = np.asarray(depth_map, dtype=np.float64).ravel()
    finite = np.isfinite(flat)
    if not np.any(finite) or not np.isfinite(endpoint_depth):
        return math.nan
    return float(np.mean(flat[finite] <= endpoint_depth))


def extract_depth_text(depth_field: Any) -> str:
    if isinstance(depth_field, list):
        return " ".join(map(str, depth_field))
    return str(depth_field)


def load_episode_steps(source_dir: Path) -> Dict[int, Dict[str, Any]]:
    path = source_dir / "steps.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    out: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            step_id = row.get("step_id", line_no)
            try:
                step_id = int(step_id)
            except Exception:
                step_id = line_no
            out[step_id] = row
    return out


def build_rollout_payloads(
    df: pd.DataFrame,
    parse_depth_tokens,
    strict: bool,
) -> Tuple[List[Optional[List[int]]], List[Optional[np.ndarray]], List[str]]:
    """
    Align raw rollout data to rows in step_features.csv using source_dir + step_id.
    """
    n = len(df)
    tokens_by_row: List[Optional[List[int]]] = [None] * n
    trace_by_row: List[Optional[np.ndarray]] = [None] * n
    errors: List[str] = []

    grouped = df.groupby("source_dir", sort=False).groups

    for episode_i, (source_dir_str, row_indices) in enumerate(grouped.items(), start=1):
        source_dir = Path(str(source_dir_str))

        try:
            step_map = load_episode_steps(source_dir)
        except Exception as e:
            msg = f"{source_dir}: {type(e).__name__}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            errors.append(msg)
            continue

        for row_idx in row_indices:
            step_id = int(df.at[row_idx, "step_id"])

            if step_id not in step_map:
                msg = f"{source_dir}: missing step_id={step_id}"
                if strict:
                    raise RuntimeError(msg)
                errors.append(msg)
                continue

            raw_step = step_map[step_id]
            pre_action = raw_step.get("pre_action", {})
            depth_field = pre_action.get("depth", None)
            trace_field = pre_action.get("trace", None)

            if depth_field is None:
                msg = f"{source_dir}: step={step_id}: missing pre_action.depth"
                if strict:
                    raise RuntimeError(msg)
                errors.append(msg)
                continue

            try:
                token_ids = parse_depth_tokens(extract_depth_text(depth_field))
            except Exception as e:
                msg = (
                    f"{source_dir}: step={step_id}: depth parse failed: "
                    f"{type(e).__name__}: {e}"
                )
                if strict:
                    raise RuntimeError(msg) from e
                errors.append(msg)
                continue

            if len(token_ids) != 100:
                msg = (
                    f"{source_dir}: step={step_id}: expected 100 depth tokens, "
                    f"got {len(token_ids)}"
                )
                if strict:
                    raise RuntimeError(msg)
                errors.append(msg)
                continue

            if min(token_ids) < 0 or max(token_ids) >= 128:
                msg = (
                    f"{source_dir}: step={step_id}: depth code out of [0,127], "
                    f"min={min(token_ids)}, max={max(token_ids)}"
                )
                if strict:
                    raise RuntimeError(msg)
                errors.append(msg)
                continue

            tokens_by_row[row_idx] = token_ids
            trace_by_row[row_idx] = parse_trace(trace_field)

        if episode_i % 10 == 0 or episode_i == len(grouped):
            print(
                f"[load] episodes {episode_i}/{len(grouped)} "
                f"(errors so far: {len(errors)})"
            )

    return tokens_by_row, trace_by_row, errors


def compute_temporal_drifts(
    out_df: pd.DataFrame,
) -> None:
    """
    Compute strictly adjacent t-1 drift within each episode.

    We do NOT carry the last valid value across missing steps.
    """
    pct_col = "feat_trace_endpoint_depth_percentile"
    raw_col = "diag_trace_endpoint_depth_raw"

    out_df["feat_trace_endpoint_depth_percentile_drift_prev"] = np.nan
    out_df["diag_trace_endpoint_depth_raw_drift_prev"] = np.nan

    for _, idxs in out_df.groupby("episode_key", sort=False).groups.items():
        idxs = list(idxs)
        g = out_df.loc[idxs, ["step_id", pct_col, raw_col]].copy()
        g["_original_index"] = idxs
        g = g.sort_values("step_id")

        prev_step = g["step_id"].shift(1)
        adjacent = (g["step_id"] - prev_step) == 1

        pct_prev = g[pct_col].shift(1)
        raw_prev = g[raw_col].shift(1)

        pct_drift = (g[pct_col] - pct_prev).abs().where(adjacent)
        raw_drift = (g[raw_col] - raw_prev).abs().where(adjacent)

        # If either current or previous is NaN, pandas naturally leaves NaN.
        out_df.loc[
            g["_original_index"],
            "feat_trace_endpoint_depth_percentile_drift_prev",
        ] = pct_drift.to_numpy()

        out_df.loc[
            g["_original_index"],
            "diag_trace_endpoint_depth_raw_drift_prev",
        ] = raw_drift.to_numpy()


def main() -> None:
    args = parse_args()

    step_features = args.step_features.expanduser().resolve()
    output = args.output.expanduser().resolve()
    ckpt_path = args.ckpt_path.expanduser().resolve()
    scripts_dir = args.molmoact_scripts.expanduser().resolve()

    if step_features == output:
        raise ValueError(
            "--output must differ from --step-features. "
            "This script intentionally refuses in-place modification."
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.resample_points < 2:
        raise ValueError("--resample-points must be >= 2")

    for required in [
        step_features,
        ckpt_path,
        scripts_dir / "reconstruct_from_tokens.py",
        scripts_dir / "vqvae.py",
    ]:
        if not required.exists():
            raise FileNotFoundError(required)

    # Import the exact official MolmoAct reconstruction helpers requested by user.
    sys.path.insert(0, str(scripts_dir))
    from reconstruct_from_tokens import load_vae, parse_depth_tokens  # type: ignore

    print("=" * 80)
    print("TRACE-DEPTH CONSISTENCY FEATURE EXTRACTION")
    print("=" * 80)
    print("input :", step_features)
    print("output:", output)
    print("VAE   :", ckpt_path)
    print("scripts:", scripts_dir)
    print("batch_size:", args.batch_size)
    print("resample_points:", args.resample_points)
    print()

    df = pd.read_csv(step_features)

    required_cols = {
        "episode_key",
        "step_id",
        "source_dir",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Input CSV missing columns: {missing_cols}")

    existing_new = [c for c in ALL_NEW_COLUMNS if c in df.columns]
    if existing_new and not args.replace_existing:
        raise ValueError(
            "Input CSV already contains Trace-Depth columns:\n  "
            + "\n  ".join(existing_new)
            + "\nUse --replace-existing only if you intentionally want to recompute them."
        )

    if existing_new:
        df = df.drop(columns=existing_new)

    # Make row index positional and stable for payload alignment.
    df = df.reset_index(drop=True)

    print(
        f"rows={len(df)}, episodes={df['episode_key'].nunique()}, "
        f"tasks={df['task'].nunique() if 'task' in df.columns else 'NA'}"
    )

    tokens_by_row, trace_by_row, errors = build_rollout_payloads(
        df=df,
        parse_depth_tokens=parse_depth_tokens,
        strict=args.strict,
    )

    valid_rows = [i for i, tok in enumerate(tokens_by_row) if tok is not None]
    print(
        f"\nrollout alignment: valid depth rows={len(valid_rows)}/{len(df)}, "
        f"errors={len(errors)}"
    )

    # Initialize output arrays.
    std_norm = np.full(len(df), np.nan, dtype=np.float64)
    grad_norm = np.full(len(df), np.nan, dtype=np.float64)
    range_norm = np.full(len(df), np.nan, dtype=np.float64)
    endpoint_pct = np.full(len(df), np.nan, dtype=np.float64)
    endpoint_raw = np.full(len(df), np.nan, dtype=np.float64)

    single_point_count = 0
    path_valid_count = 0
    endpoint_valid_count = 0
    flat_frame_count = 0

    print("\nLoading VQ-VAE once...")
    vae = load_vae(str(ckpt_path))
    device = next(vae.parameters()).device
    print("VQ-VAE device:", device)

    # Batch decode all valid rows.
    n_batches = math.ceil(len(valid_rows) / args.batch_size) if valid_rows else 0

    with torch.inference_mode():
        for batch_no, start in enumerate(
            range(0, len(valid_rows), args.batch_size), start=1
        ):
            batch_indices = valid_rows[start : start + args.batch_size]

            codes_np = np.asarray(
                [tokens_by_row[i] for i in batch_indices],
                dtype=np.int64,
            ).reshape(-1, 10, 10)

            codes = torch.as_tensor(
                codes_np,
                dtype=torch.long,
                device=device,
            )

            decoded = vae.decode(codes)

            if decoded.ndim != 4 or decoded.shape[1] != 1:
                raise RuntimeError(
                    f"Unexpected VQ-VAE output shape {tuple(decoded.shape)}; "
                    "expected [B,1,H,W]."
                )

            decoded_np = (
                decoded[:, 0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            for local_i, row_idx in enumerate(batch_indices):
                raw = decoded_np[local_i]

                if not np.isfinite(raw).all():
                    msg = (
                        f"row={row_idx}: decoded depth contains non-finite values"
                    )
                    if args.strict:
                        raise RuntimeError(msg)
                    errors.append(msg)
                    continue

                frame_std = float(np.std(raw))
                trace = trace_by_row[row_idx]

                if trace is None or len(trace) == 0:
                    continue

                # Endpoint remains well-defined for one-point traces.
                ep_depth_arr = bilinear_sample_255_coords(raw, trace[-1:])
                if len(ep_depth_arr) == 1 and np.isfinite(ep_depth_arr[0]):
                    ep_depth = float(ep_depth_arr[0])
                    endpoint_raw[row_idx] = ep_depth
                    endpoint_pct[row_idx] = endpoint_percentile(raw, ep_depth)
                    endpoint_valid_count += 1

                # Structural degeneration: do not encode as zero path variation.
                if len(trace) == 1:
                    single_point_count += 1
                    continue

                sampled = resample_trace_arc_length(
                    trace,
                    n_points=args.resample_points,
                    eps=args.eps,
                )
                if sampled is None:
                    continue

                z = bilinear_sample_255_coords(raw, sampled)
                if len(z) != args.resample_points or not np.isfinite(z).all():
                    continue

                if frame_std <= args.eps:
                    flat_frame_count += 1
                    continue

                trace_std = float(np.std(z))
                trace_grad = float(np.mean(np.abs(np.diff(z))))
                trace_range = float(np.max(z) - np.min(z))

                std_norm[row_idx] = trace_std / frame_std
                grad_norm[row_idx] = trace_grad / frame_std
                range_norm[row_idx] = trace_range / frame_std
                path_valid_count += 1

            if (
                batch_no == 1
                or batch_no % 10 == 0
                or batch_no == n_batches
            ):
                print(
                    f"[decode] batch {batch_no}/{n_batches} "
                    f"({min(start + args.batch_size, len(valid_rows))}/"
                    f"{len(valid_rows)} rows)"
                )

    out_df = df.copy()
    out_df["feat_trace_depth_std_norm"] = std_norm
    out_df["feat_trace_depth_gradient_norm"] = grad_norm
    out_df["feat_trace_depth_range_norm"] = range_norm
    out_df["feat_trace_endpoint_depth_percentile"] = endpoint_pct
    out_df["diag_trace_endpoint_depth_raw"] = endpoint_raw

    # Strictly t vs t-1 within episode.
    compute_temporal_drifts(out_df)

    # Safety checks.
    if len(out_df) != len(df):
        raise RuntimeError("Row count changed unexpectedly.")

    if out_df[FEATURE_COLUMNS + DIAG_COLUMNS].columns.duplicated().any():
        raise RuntimeError("Duplicate Trace-Depth output columns detected.")

    # Sanity: percentile must be in [0,1] wherever finite.
    finite_pct = out_df["feat_trace_endpoint_depth_percentile"].dropna()
    if len(finite_pct) and (
        (finite_pct < 0.0).any() or (finite_pct > 1.0).any()
    ):
        raise RuntimeError("Endpoint percentile outside [0,1].")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)

    # Reproducibility / definitions metadata.
    metadata_path = output.with_suffix(".metadata.json")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_step_features": str(step_features),
        "source_step_features_sha256": sha256_file(step_features),
        "output_csv": str(output),
        "vae_checkpoint": str(ckpt_path),
        "vae_checkpoint_sha256": sha256_file(ckpt_path),
        "molmoact_scripts": str(scripts_dir),
        "num_rows": int(len(out_df)),
        "num_episodes": int(out_df["episode_key"].nunique()),
        "num_rollout_depth_rows_valid": int(len(valid_rows)),
        "num_endpoint_features_valid": int(endpoint_valid_count),
        "num_path_features_valid": int(path_valid_count),
        "num_single_point_traces": int(single_point_count),
        "num_flat_depth_maps": int(flat_frame_count),
        "num_errors": int(len(errors)),
        "batch_size": int(args.batch_size),
        "trace_resample_points": int(args.resample_points),
        "trace_coordinate_system": "[0,255]^2",
        "decoder_output": "raw VQ-VAE reconstructed relative depth, before per-image min-max visualization normalization",
        "feature_definitions": {
            "feat_trace_depth_std_norm": (
                "std(depth sampled at arc-length-resampled Trace points) / "
                "std(full decoded depth map)"
            ),
            "feat_trace_depth_gradient_norm": (
                "mean(abs(diff(depth along arc-length-resampled Trace))) / "
                "std(full decoded depth map)"
            ),
            "feat_trace_depth_range_norm": (
                "(max(trace depth)-min(trace depth)) / std(full decoded depth map)"
            ),
            "feat_trace_endpoint_depth_percentile": (
                "percentile rank of the Trace endpoint raw decoded depth within "
                "the current decoded depth map"
            ),
            "feat_trace_endpoint_depth_percentile_drift_prev": (
                "absolute difference in endpoint-depth percentile between t and "
                "the immediately preceding step t-1 in the same episode"
            ),
            "diag_trace_endpoint_depth_raw": (
                "raw decoder depth sampled at current Trace endpoint; diagnostic"
            ),
            "diag_trace_endpoint_depth_raw_drift_prev": (
                "absolute change of raw endpoint decoder value from t-1; "
                "diagnostic/exploratory because raw decoder scale may shift "
                "between frames"
            ),
        },
        "single_point_policy": (
            "For one-point Trace: path std/gradient/range are NaN; endpoint "
            "percentile/raw features remain valid."
        ),
        "errors_preview": errors[:100],
        "note": (
            "Trace-Depth features are exploratory and online-safe (current/past "
            "pre-action information only). Raw diagnostic columns should not be "
            "treated as primary predictor inputs."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=json_safe)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("output CSV     :", output)
    print("metadata       :", metadata_path)
    print("rows           :", len(out_df))
    print("episodes       :", out_df["episode_key"].nunique())
    print("valid endpoints:", endpoint_valid_count)
    print("valid paths    :", path_valid_count)
    print("single-point   :", single_point_count)
    print("errors         :", len(errors))

    print("\nNew feature missingness:")
    for col in FEATURE_COLUMNS:
        n_missing = int(out_df[col].isna().sum())
        print(f"  {col:52s} {n_missing:5d}/{len(out_df)}")

    if errors:
        print("\nFirst errors:")
        for e in errors[:10]:
            print("  -", e)

    print(
        "\nSource CSV was NOT modified. "
        "Use the new output CSV for Trace-Depth exploratory analysis."
    )


if __name__ == "__main__":
    main()
