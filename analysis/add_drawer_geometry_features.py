"""Add drawer-specific Trace/Depth features.

The Trace features are extracted directly from pre-action rollout outputs.
Depth features require the official MolmoAct VQ-VAE decoder because VQ token
IDs are categorical and must not be treated as numeric depth values.

The default canonical image-plane pull direction is estimated from successful
OpenDrawer traces.  This is appropriate for exploratory analysis only.  A
formal OOF run must estimate the direction inside each training fold.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DEFAULT_STEPS = Path("data/formal_features/step_features.csv")
DEFAULT_OUTPUT = Path("data/formal_features/step_features_drawer_geometry.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--step-features", type=Path, default=DEFAULT_STEPS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--task", default="open_drawer")
    p.add_argument("--canonical-dx", type=float, default=None)
    p.add_argument("--canonical-dy", type=float, default=None)
    p.add_argument("--ckpt-path", type=Path, default=None)
    p.add_argument("--molmoact-scripts", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--resample-points", type=int, default=20)
    p.add_argument("--border-frac", type=float, default=0.10)
    p.add_argument("--patch-radius-frac", type=float, default=0.035)
    p.add_argument("--depth-diff-eps-frac", type=float, default=0.02)
    return p.parse_args()


def parse_trace(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        return None
    return arr


def parse_depth_tokens(value: Any) -> list[int]:
    if isinstance(value, list):
        value = "".join(map(str, value))
    return [int(x) for x in re.findall(r"<DEPTH_(\d+)>", str(value))]


def load_raw_steps(source_dirs: list[str]) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for source in sorted(set(source_dirs)):
        path = Path(source) / "steps.jsonl"
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    rows[(source, int(item.get("step_id", line_no)))] = item
    return rows


def trace_vector(trace: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if trace is None or len(trace) < 2:
        return None
    v = trace[-1] - trace[0]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else None


def estimate_canonical_direction(df: pd.DataFrame, raw: dict[tuple[str, int], dict[str, Any]]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for row in df.loc[df["label_episode_success"].astype(int).eq(1)].itertuples(index=False):
        item = raw.get((str(row.source_dir), int(row.step_id)))
        if item is None:
            continue
        u = trace_vector(parse_trace((item.get("pre_action") or {}).get("trace")))
        if u is not None:
            vectors.append(u)
    if not vectors:
        raise ValueError("Cannot estimate canonical direction: no valid successful traces")
    # Mean unit vector is stable here; normalize after aggregation.
    u = np.mean(np.asarray(vectors), axis=0)
    n = float(np.linalg.norm(u))
    if n <= 1e-8:
        raise ValueError("Successful trace directions cancel; pass --canonical-dx/--canonical-dy")
    return u / n


def resample_trace(trace: np.ndarray, n: int) -> np.ndarray:
    if len(trace) == 1:
        return np.repeat(trace, n, axis=0)
    seg = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] <= 1e-8:
        return np.repeat(trace[:1], n, axis=0)
    out = np.empty((n, 2), dtype=np.float64)
    for j, t in enumerate(np.linspace(0.0, float(cum[-1]), n)):
        k = min(max(int(np.searchsorted(cum, t, side="right") - 1), 0), len(seg) - 1)
        a = 0.0 if cum[k + 1] - cum[k] <= 1e-8 else (t - cum[k]) / (cum[k + 1] - cum[k])
        out[j] = trace[k] * (1.0 - a) + trace[k + 1] * a
    return out


def trace_features(trace: Optional[np.ndarray], canonical: np.ndarray) -> dict[str, float]:
    out = {
        "feat_trace_backtracking_ratio": math.nan,
        "feat_trace_direction_alignment": math.nan,
    }
    if trace is None or len(trace) < 2:
        return out
    d = np.diff(trace, axis=0)
    proj = d @ canonical
    denom = float(np.abs(proj).sum())
    out["feat_trace_backtracking_ratio"] = float(np.abs(proj[proj < 0]).sum() / denom) if denom > 1e-8 else math.nan
    v = trace_vector(trace)
    if v is not None:
        out["feat_trace_direction_alignment"] = float(np.dot(v, canonical))
    return out


def bilinear_sample(depth: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    x = np.clip(points[:, 0], 0.0, 255.0) / 255.0 * (w - 1)
    y = np.clip(points[:, 1], 0.0, 255.0) / 255.0 * (h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    return (
        depth[y0, x0] * (1 - dx) * (1 - dy)
        + depth[y0, x1] * dx * (1 - dy)
        + depth[y1, x0] * (1 - dx) * dy
        + depth[y1, x1] * dx * dy
    )


def patch_median(depth: np.ndarray, center: np.ndarray, radius_frac: float) -> float:
    h, w = depth.shape
    cx, cy = center[0] / 255.0 * (w - 1), center[1] / 255.0 * (h - 1)
    radius = radius_frac * min(h, w)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    vals = depth[mask]
    return float(np.median(vals)) if len(vals) else math.nan


def depth_features(depth: np.ndarray, trace: Optional[np.ndarray], args: argparse.Namespace) -> dict[str, float]:
    out = {
        "feat_trace_depth_sign_change_count": math.nan,
        "feat_trace_endpoint_vs_background_depth_contrast": math.nan,
        "feat_trace_endpoint_vs_drawer_front_depth_contrast": math.nan,
    }
    if trace is None or len(trace) < 2:
        return out
    sample = bilinear_sample(depth, resample_trace(trace, args.resample_points))
    std = float(np.std(depth))
    if std <= 1e-8:
        return out
    diffs = np.diff(sample)
    eps = args.depth_diff_eps_frac * std
    signs = np.sign(diffs)
    signs[np.abs(diffs) <= eps] = 0.0
    nz = signs[signs != 0]
    out["feat_trace_depth_sign_change_count"] = float(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0.0

    endpoint = patch_median(depth, trace[-1], args.patch_radius_frac)
    h, w = depth.shape
    border = int(args.border_frac * min(h, w))
    ring = np.ones((h, w), dtype=bool)
    ring[border : h - border, border : w - border] = False
    bg = float(np.median(depth[ring])) if ring.any() else math.nan
    interior = trace[1:-1] if len(trace) > 2 else trace[:-1]
    front_vals = [patch_median(depth, p, args.patch_radius_frac) for p in interior]
    front_vals = [v for v in front_vals if np.isfinite(v)]
    front = float(np.median(front_vals)) if front_vals else math.nan
    out["feat_trace_endpoint_vs_background_depth_contrast"] = (endpoint - bg) / std if np.isfinite(endpoint) and np.isfinite(bg) else math.nan
    out["feat_trace_endpoint_vs_drawer_front_depth_contrast"] = (endpoint - front) / std if np.isfinite(endpoint) and np.isfinite(front) else math.nan
    return out


def decode_depths(
    df: pd.DataFrame,
    raw: dict[tuple[str, int], dict[str, Any]],
    args: argparse.Namespace,
) -> dict[int, np.ndarray]:
    """Decode the 10x10 depth-token grids with the official VQ-VAE."""
    if args.ckpt_path is None or args.molmoact_scripts is None:
        return {}
    scripts = args.molmoact_scripts.expanduser().resolve()
    ckpt = args.ckpt_path.expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    sys.path.insert(0, str(scripts))
    from reconstruct_from_tokens import load_vae  # type: ignore
    import torch

    payloads: list[tuple[int, list[int]]] = []
    for i, row in enumerate(df.itertuples(index=False)):
        item = raw[(str(row.source_dir), int(row.step_id))]
        tokens = parse_depth_tokens((item.get("pre_action") or {}).get("depth"))
        if len(tokens) == 100 and min(tokens) >= 0 and max(tokens) < 128:
            payloads.append((i, tokens))
    if not payloads:
        return {}

    vae = load_vae(str(ckpt))
    device = next(vae.parameters()).device
    decoded_by_row: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(payloads), max(1, args.batch_size)):
            batch = payloads[start : start + max(1, args.batch_size)]
            codes = torch.as_tensor(
                np.asarray([tokens for _, tokens in batch], dtype=np.int64).reshape(-1, 10, 10),
                dtype=torch.long,
                device=device,
            )
            decoded = vae.decode(codes)
            if decoded.ndim != 4 or decoded.shape[1] != 1:
                raise RuntimeError(f"Unexpected VQ-VAE output shape: {tuple(decoded.shape)}")
            arr = decoded[:, 0].detach().float().cpu().numpy()
            for (row_idx, _), depth in zip(batch, arr):
                decoded_by_row[row_idx] = depth
    return decoded_by_row


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.step_features)
    df = df.loc[df["task"].eq(args.task)].copy().reset_index(drop=True)
    raw = load_raw_steps(df["source_dir"].astype(str).tolist())
    if args.canonical_dx is None or args.canonical_dy is None:
        canonical = estimate_canonical_direction(df, raw)
        direction_source = "successful traces in this input table (exploratory; not fold-safe)"
    else:
        canonical = np.asarray([args.canonical_dx, args.canonical_dy], dtype=float)
        canonical /= np.linalg.norm(canonical)
        direction_source = "user supplied"

    decoded_by_row = decode_depths(df, raw, args)
    out = {"feat_trace_backtracking_ratio": [], "feat_trace_direction_alignment": [], "feat_trace_depth_sign_change_count": [], "feat_trace_endpoint_vs_background_depth_contrast": [], "feat_trace_endpoint_vs_drawer_front_depth_contrast": []}
    for i, row in enumerate(df.itertuples(index=False)):
        item = raw[(str(row.source_dir), int(row.step_id))]
        trace = parse_trace((item.get("pre_action") or {}).get("trace"))
        tf = trace_features(trace, canonical)
        for key, value in tf.items():
            out[key].append(value)
        dfm = depth_features(decoded_by_row[i], trace, args) if i in decoded_by_row else {
            "feat_trace_depth_sign_change_count": math.nan,
            "feat_trace_endpoint_vs_background_depth_contrast": math.nan,
            "feat_trace_endpoint_vs_drawer_front_depth_contrast": math.nan,
        }
        for key, value in dfm.items():
            out[key].append(value)

    for key, values in out.items():
        df[key] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    metadata = {
        "task": args.task,
        "canonical_direction_xy": canonical.tolist(),
        "canonical_direction_source": direction_source,
        "trace_backtracking_definition": "sum(abs(negative segment projections on canonical direction)) / sum(abs(all segment projections))",
        "direction_alignment_definition": "cosine(trace endpoint-start vector, canonical direction)",
        "drawer_front_definition": "median depth of local patches around interior Trace points (proxy; not GT handle/front ROI)",
        "background_definition": "median depth of a 10% image border ring",
        "depth_note": "Depth features are NaN until the official VQ-VAE decoder is supplied; raw VQ IDs are not treated as numeric depth.",
        "decoded_depth_rows": int(len(decoded_by_row)),
    }
    with args.output.with_suffix(".metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(df)} {args.task} rows to {args.output}")
    print("canonical_direction_xy:", canonical.tolist())
    if args.ckpt_path is None:
        print("NOTE: --ckpt-path not supplied; depth-derived columns are NaN by design.")


if __name__ == "__main__":
    main()
