#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.instrumented_molmoact import InstrumentedMolmoAct


def flatten_depth(depth):
    if depth is None:
        return []

    if isinstance(depth, list):
        depth = depth[0]

    import re
    return [
        int(x)
        for x in re.findall(r"<DEPTH_(\d+)>", depth)
    ]


def trace_array(trace):
    if trace is None:
        return None

    arr = np.asarray(trace, dtype=np.float32)

    while arr.ndim > 2:
        arr = arr[0]

    return arr


def action_array(x):
    if x is None:
        return None

    arr = np.asarray(x, dtype=np.float32)

    while arr.ndim > 1:
        arr = arr[0]

    return arr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)

    args = parser.parse_args()

    image = np.asarray(
        Image.open(args.image).convert("RGB"),
        dtype=np.uint8,
    )

    results = {}

    for backend in ["hf", "vllm"]:

        print(f"\n===== {backend.upper()} =====")

        model = InstrumentedMolmoAct(
            saved_model_path=args.checkpoint,
            policy_setup="google_robot",
            backend=backend,
            quiet=True,
        )

        model.reset(args.instruction)

        raw_action, action, _ = model.step(
            image,
            args.instruction,
        )

        results[backend] = dict(model.last_debug)

        print(
            "time:",
            results[backend]["inference_time_s"]
        )

        # 释放后再加载另一个 backend
        del model

        import torch
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # Depth
    # --------------------------------------------------

    d_hf = flatten_depth(results["hf"]["depth"])
    d_vl = flatten_depth(results["vllm"]["depth"])

    if len(d_hf) == len(d_vl) and len(d_hf) > 0:

        depth_agreement = np.mean(
            np.asarray(d_hf) == np.asarray(d_vl)
        )

        print(
            "\nDepth token agreement:",
            depth_agreement
        )

    else:
        print(
            "\nDepth token count:",
            len(d_hf),
            len(d_vl),
        )

    # --------------------------------------------------
    # Trace
    # --------------------------------------------------

    t_hf = trace_array(results["hf"]["trace"])
    t_vl = trace_array(results["vllm"]["trace"])

    if (
        t_hf is not None
        and t_vl is not None
        and t_hf.shape == t_vl.shape
    ):

        trace_distance = np.linalg.norm(
            t_hf - t_vl,
            axis=1,
        ).mean()

        print(
            "Mean trace-point distance:",
            trace_distance
        )

    # --------------------------------------------------
    # Action
    # --------------------------------------------------

    a_hf = action_array(
        results["hf"]["parsed_action"]
    )

    a_vl = action_array(
        results["vllm"]["parsed_action"]
    )

    if a_hf is not None and a_vl is not None:

        print(
            "Action L2:",
            np.linalg.norm(a_hf - a_vl)
        )

        print(
            "HF action:",
            a_hf
        )

        print(
            "vLLM action:",
            a_vl
        )

    print(
        "\nHF time:",
        results["hf"]["inference_time_s"]
    )

    print(
        "vLLM time:",
        results["vllm"]["inference_time_s"]
    )


if __name__ == "__main__":
    main()