import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(
    "/root/autodl-tmp/molmoact_project/results/backend_test"
)

TASK = "pick_coke_can"
CONDITION = "condition_0001"


HF_PATH = (
    ROOT
    / "hf"
    / TASK
    / CONDITION
    / "steps.jsonl"
)

VLLM_PATH = (
    ROOT
    / "vllm"
    / TASK
    / CONDITION
    / "steps.jsonl"
)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def parse_depth(depth):
    if depth is None:
        return []

    if isinstance(depth, list):
        if not depth:
            return []
        depth = depth[0]

    return [
        int(x)
        for x in re.findall(
            r"<DEPTH_(\d+)>",
            depth,
        )
    ]


def arr(x):
    if x is None:
        return None

    x = np.asarray(x, dtype=np.float64)

    while x.ndim > 1:
        # parsed_action 常见 [[...]]
        if x.shape[0] == 1:
            x = x[0]
        else:
            break

    return x


hf = load_jsonl(HF_PATH)
vl = load_jsonl(VLLM_PATH)

n = min(len(hf), len(vl))

print("HF steps:", len(hf))
print("vLLM steps:", len(vl))
print("compare first:", n)
print()


first_rgb_diff = None
first_depth_diff = None
first_trace_diff = None
first_parsed_action_diff = None
first_env_action_diff = None


for t in range(n):

    hp = hf[t]["pre_action"]
    vp = vl[t]["pre_action"]

    # --------------------------------------------------
    # RGB
    # --------------------------------------------------

    h_rgb = hp.get("rgb_sha1")
    v_rgb = vp.get("rgb_sha1")

    rgb_same = h_rgb == v_rgb

    if not rgb_same and first_rgb_diff is None:
        first_rgb_diff = t


    # --------------------------------------------------
    # Depth
    # --------------------------------------------------

    hd = parse_depth(hp.get("depth"))
    vd = parse_depth(vp.get("depth"))

    depth_same = hd == vd

    if not depth_same and first_depth_diff is None:
        first_depth_diff = t


    # --------------------------------------------------
    # Trace
    # --------------------------------------------------

    ht = np.asarray(
        hp.get("trace"),
        dtype=np.float64,
    )

    vt = np.asarray(
        vp.get("trace"),
        dtype=np.float64,
    )

    trace_same = (
        ht.shape == vt.shape
        and np.array_equal(ht, vt)
    )

    if not trace_same and first_trace_diff is None:
        first_trace_diff = t


    # --------------------------------------------------
    # Parsed action
    # --------------------------------------------------

    ha = arr(hp.get("parsed_action"))
    va = arr(vp.get("parsed_action"))

    parsed_action_same = (
        ha is not None
        and va is not None
        and ha.shape == va.shape
        and np.array_equal(ha, va)
    )

    if (
        not parsed_action_same
        and first_parsed_action_diff is None
    ):
        first_parsed_action_diff = t


    # --------------------------------------------------
    # Actual action sent to environment
    # --------------------------------------------------

    he = np.asarray(
        hp.get("env_action"),
        dtype=np.float64,
    )

    ve = np.asarray(
        vp.get("env_action"),
        dtype=np.float64,
    )

    env_action_same = (
        he.shape == ve.shape
        and np.array_equal(he, ve)
    )

    if (
        not env_action_same
        and first_env_action_diff is None
    ):
        first_env_action_diff = t


    print(
        f"step={t:02d} "
        f"RGB={'=' if rgb_same else 'X'} "
        f"D={'=' if depth_same else 'X'} "
        f"T={'=' if trace_same else 'X'} "
        f"A={'=' if parsed_action_same else 'X'} "
        f"ENV_A={'=' if env_action_same else 'X'}"
    )

    # 一旦实际环境 action 不同，
    # 后续闭环 RGB 很可能开始逐渐分叉。
    if not env_action_same:

        print("\n===== FIRST ACTION DIVERGENCE =====")
        print("step:", t)

        print("\nHF parsed action:")
        print(ha)

        print("\nvLLM parsed action:")
        print(va)

        print("\nparsed action L2:")
        print(np.linalg.norm(ha - va))

        print("\nHF env action:")
        print(he)

        print("\nvLLM env action:")
        print(ve)

        print("\nenv action L2:")
        print(np.linalg.norm(he - ve))

        break


print("\n===== SUMMARY =====")

print(
    "first RGB difference:",
    first_rgb_diff,
)

print(
    "first Depth difference:",
    first_depth_diff,
)

print(
    "first Trace difference:",
    first_trace_diff,
)

print(
    "first parsed-action difference:",
    first_parsed_action_diff,
)

print(
    "first env-action difference:",
    first_env_action_diff,
)