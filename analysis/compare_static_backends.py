import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(
    "/root/autodl-tmp/molmoact_project/results/backend_test"
)

HF_PATH = ROOT / "static_hf.json"
VLLM_PATH = ROOT / "static_vllm.json"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_depth(depth):
    if depth is None:
        return []

    if isinstance(depth, list):
        if len(depth) == 0:
            return []
        depth = depth[0]

    return [
        int(x)
        for x in re.findall(r"<DEPTH_(\d+)>", depth)
    ]


def parse_trace(trace):
    if trace is None:
        return None

    x = np.asarray(trace, dtype=np.float32)

    while x.ndim > 2:
        x = x[0]

    return x


def parse_action(action):
    if action is None:
        return None

    x = np.asarray(action, dtype=np.float32)

    while x.ndim > 1:
        x = x[0]

    return x


hf = load(HF_PATH)
vl = load(VLLM_PATH)


print("===== STATIC HF vs vLLM =====")


# -------------------------------------------------------
# Generated text
# -------------------------------------------------------

text_hf = hf.get("generated_text")
text_vl = vl.get("generated_text")

print("\n[Generated text]")
print("exact match:", text_hf == text_vl)


# -------------------------------------------------------
# Depth
# -------------------------------------------------------

d_hf = parse_depth(hf.get("depth"))
d_vl = parse_depth(vl.get("depth"))

print("\n[Depth]")
print("HF token count:", len(d_hf))
print("vLLM token count:", len(d_vl))

if len(d_hf) == len(d_vl) and len(d_hf) > 0:
    dh = np.asarray(d_hf)
    dv = np.asarray(d_vl)

    agreement = np.mean(dh == dv)

    print("token agreement:", agreement)
    print("different tokens:", np.sum(dh != dv))
else:
    print("Cannot compute token agreement.")


# -------------------------------------------------------
# Trace
# -------------------------------------------------------

t_hf = parse_trace(hf.get("trace"))
t_vl = parse_trace(vl.get("trace"))

print("\n[Trace]")
print("HF:", t_hf)
print("vLLM:", t_vl)

if t_hf is not None and t_vl is not None:

    print("HF shape:", t_hf.shape)
    print("vLLM shape:", t_vl.shape)

    if t_hf.shape == t_vl.shape:

        point_dist = np.linalg.norm(
            t_hf - t_vl,
            axis=1,
        )

        print(
            "mean point distance:",
            float(point_dist.mean()),
        )

        print(
            "max point distance:",
            float(point_dist.max()),
        )

        print(
            "endpoint distance:",
            float(
                np.linalg.norm(
                    t_hf[-1] - t_vl[-1]
                )
            ),
        )


# -------------------------------------------------------
# Action
# -------------------------------------------------------

a_hf = parse_action(hf.get("parsed_action"))
a_vl = parse_action(vl.get("parsed_action"))

print("\n[Action]")
print("HF:", a_hf)
print("vLLM:", a_vl)

if a_hf is not None and a_vl is not None:

    diff = a_hf - a_vl

    print(
        "L2 difference:",
        float(np.linalg.norm(diff)),
    )

    print(
        "max abs difference:",
        float(np.abs(diff).max()),
    )

    print(
        "per-dim difference:",
        diff,
    )


# -------------------------------------------------------
# Time
# -------------------------------------------------------

hf_time = hf.get("inference_time_s")
vl_time = vl.get("inference_time_s")

print("\n[Speed]")
print("HF:", hf_time)
print("vLLM:", vl_time)

if hf_time and vl_time:
    print(
        "speedup HF/vLLM:",
        hf_time / vl_time,
    )