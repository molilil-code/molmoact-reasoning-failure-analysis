#!/usr/bin/env python3
"""Compare static MolmoAct outputs from HF and vLLM JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True, help="HF JSONL from dump_static_batch.py")
    p.add_argument("--vllm", required=True, help="vLLM JSONL from dump_static_batch.py")
    p.add_argument("--output-csv", default=None)
    p.add_argument("--action-atol", type=float, default=1e-8)
    p.add_argument("--trace-atol", type=float, default=1e-6)
    return p.parse_args()


def load_jsonl(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {path}:{line_no}: {e}") from e
            if "step" not in item:
                raise KeyError(f"Missing step: {path}:{line_no}")
            step = int(item["step"])
            if step in rows:
                raise ValueError(f"Duplicate step={step} in {path}")
            rows[step] = item
    return rows


def parse_depth(depth: Any):
    if depth is None:
        return []
    if isinstance(depth, list):
        if not depth:
            return []
        if all(isinstance(x, int) for x in depth):
            return [int(x) for x in depth]
        if len(depth) == 1 and isinstance(depth[0], str):
            depth = depth[0]
        elif all(isinstance(x, str) for x in depth):
            depth = "".join(depth)
    if isinstance(depth, str):
        return [int(x) for x in re.findall(r"<DEPTH_(\d+)>", depth)]
    return []


def squeeze_leading_singletons(x: Any):
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64)
    while a.ndim > 0 and a.shape[0] == 1:
        a = a[0]
    return a


def parse_trace(x: Any):
    a = squeeze_leading_singletons(x)
    if a is None or a.ndim != 2 or a.shape[-1] != 2:
        return None
    return a


def parse_action(x: Any):
    a = squeeze_leading_singletons(x)
    if a is None:
        return None
    return np.asarray(a, dtype=np.float64).reshape(-1)


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def fmt(x, digits=4):
    if isinstance(x, (float, np.floating)):
        if math.isnan(float(x)):
            return "nan"
        return f"{float(x):.{digits}f}"
    return str(x)


def pct(n, d):
    return "n/a" if d == 0 else f"{100*n/d:.1f}%"


def mean_nan(xs):
    a = np.asarray(xs, dtype=float)
    return math.nan if len(a) == 0 or np.isnan(a).all() else float(np.nanmean(a))


def min_nan(xs):
    a = np.asarray(xs, dtype=float)
    return math.nan if len(a) == 0 or np.isnan(a).all() else float(np.nanmin(a))


def max_nan(xs):
    a = np.asarray(xs, dtype=float)
    return math.nan if len(a) == 0 or np.isnan(a).all() else float(np.nanmax(a))


def compare_step(h, v, action_atol, trace_atol):
    hd, vd = parse_depth(h.get("depth")), parse_depth(v.get("depth"))
    if hd and vd and len(hd) == len(vd):
        hd_a, vd_a = np.asarray(hd), np.asarray(vd)
        depth_agreement = float((hd_a == vd_a).mean())
        depth_diff_tokens = int((hd_a != vd_a).sum())
        depth_exact = bool(np.array_equal(hd_a, vd_a))
    else:
        depth_agreement = math.nan
        depth_diff_tokens = math.nan
        depth_exact = False

    ht, vt = parse_trace(h.get("trace")), parse_trace(v.get("trace"))
    if ht is not None and vt is not None and ht.shape == vt.shape:
        d = np.linalg.norm(ht - vt, axis=1)
        trace_exact = bool(np.array_equal(ht, vt))
        trace_near = bool(np.allclose(ht, vt, atol=trace_atol, rtol=0.0))
        trace_mean = float(d.mean())
        trace_max = float(d.max())
        trace_end = float(np.linalg.norm(ht[-1] - vt[-1]))
    else:
        trace_exact = trace_near = False
        trace_mean = trace_max = trace_end = math.nan

    ha, va = parse_action(h.get("parsed_action")), parse_action(v.get("parsed_action"))
    if ha is not None and va is not None and ha.shape == va.shape:
        diff = ha - va
        action_exact = bool(np.array_equal(ha, va))
        action_near = bool(np.allclose(ha, va, atol=action_atol, rtol=0.0))
        action_l2 = float(np.linalg.norm(diff))
        action_max_abs = float(np.abs(diff).max())
    else:
        action_exact = action_near = False
        action_l2 = action_max_abs = math.nan

    hf_time = safe_float(h.get("inference_time_s"))
    vl_time = safe_float(v.get("inference_time_s"))
    speedup = hf_time / vl_time if not math.isnan(hf_time) and not math.isnan(vl_time) and vl_time > 0 else math.nan

    return {
        "generated_text_exact": h.get("generated_text") == v.get("generated_text"),
        "depth_hf_count": len(hd),
        "depth_vllm_count": len(vd),
        "depth_agreement": depth_agreement,
        "depth_diff_tokens": depth_diff_tokens,
        "depth_exact": depth_exact,
        "trace_exact": trace_exact,
        "trace_near_equal": trace_near,
        "trace_mean_dist": trace_mean,
        "trace_max_dist": trace_max,
        "trace_endpoint_dist": trace_end,
        "action_exact": action_exact,
        "action_near_equal": action_near,
        "action_l2": action_l2,
        "action_max_abs_diff": action_max_abs,
        "hf_time_s": hf_time,
        "vllm_time_s": vl_time,
        "speedup_hf_over_vllm": speedup,
    }


def main():
    args = parse_args()
    hf_path, vl_path = Path(args.hf), Path(args.vllm)
    hf, vl = load_jsonl(hf_path), load_jsonl(vl_path)

    common = sorted(set(hf) & set(vl))
    only_hf = sorted(set(hf) - set(vl))
    only_vl = sorted(set(vl) - set(hf))

    print("===== STATIC BATCH HF vs vLLM =====")
    print("HF steps:", sorted(hf))
    print("vLLM steps:", sorted(vl))
    print("Common steps:", common)
    if only_hf:
        print("Only in HF:", only_hf)
    if only_vl:
        print("Only in vLLM:", only_vl)
    if not common:
        raise RuntimeError("No common step ids")

    rows = []
    for step in common:
        r = {"step": step, **compare_step(hf[step], vl[step], args.action_atol, args.trace_atol)}
        rows.append(r)

    print("\n===== PER-STEP COMPARISON =====")
    header = f"{'step':>5} {'D_agree':>9} {'D_diff':>7} {'T_mean':>10} {'T_end':>10} {'A_L2':>11} {'HF_s':>9} {'vLLM_s':>9} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['step']:5d} "
            f"{fmt(r['depth_agreement'],3):>9} "
            f"{fmt(r['depth_diff_tokens'],0):>7} "
            f"{fmt(r['trace_mean_dist'],3):>10} "
            f"{fmt(r['trace_endpoint_dist'],3):>10} "
            f"{fmt(r['action_l2'],6):>11} "
            f"{fmt(r['hf_time_s'],3):>9} "
            f"{fmt(r['vllm_time_s'],3):>9} "
            f"{fmt(r['speedup_hf_over_vllm'],3):>9}"
        )

    n = len(rows)
    print("\n===== AGGREGATE =====")
    print("Compared frames:", n)
    print("Generated-text exact-match rate:", pct(sum(r['generated_text_exact'] for r in rows), n))
    print("Depth exact-match rate:", pct(sum(r['depth_exact'] for r in rows), n))
    print("Mean Depth token agreement:", fmt(mean_nan([r['depth_agreement'] for r in rows]), 4))
    print("Minimum Depth token agreement:", fmt(min_nan([r['depth_agreement'] for r in rows]), 4))
    print("Trace exact-match rate:", pct(sum(r['trace_exact'] for r in rows), n))
    print("Trace near-equal rate:", pct(sum(r['trace_near_equal'] for r in rows), n))
    print("Mean trace point distance:", fmt(mean_nan([r['trace_mean_dist'] for r in rows]), 4))
    print("Mean trace endpoint distance:", fmt(mean_nan([r['trace_endpoint_dist'] for r in rows]), 4))
    print("Max trace endpoint distance:", fmt(max_nan([r['trace_endpoint_dist'] for r in rows]), 4))
    print("Action exact-match rate:", pct(sum(r['action_exact'] for r in rows), n))
    print(f"Action near-equal rate (atol={args.action_atol:g}):", pct(sum(r['action_near_equal'] for r in rows), n))
    print("Mean action L2:", fmt(mean_nan([r['action_l2'] for r in rows]), 8))
    print("Max action L2:", fmt(max_nan([r['action_l2'] for r in rows]), 8))
    mean_hf = mean_nan([r['hf_time_s'] for r in rows])
    mean_vl = mean_nan([r['vllm_time_s'] for r in rows])
    print("Mean HF latency (s):", fmt(mean_hf, 4))
    print("Mean vLLM latency (s):", fmt(mean_vl, 4))
    print("Speedup (mean HF / mean vLLM):", fmt(mean_hf / mean_vl if mean_vl > 0 else math.nan, 4))

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("\nSaved CSV:", out)


if __name__ == "__main__":
    main()
