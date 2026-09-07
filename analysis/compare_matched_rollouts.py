import json
from pathlib import Path


ROOT = Path(
    "/root/autodl-tmp/molmoact_project/results/backend_test/matched"
)

tasks = [
    "pick_coke_can",
    "open_drawer",
    "move_near",
]


pairs = []


for task in tasks:

    hf_root = ROOT / "hf" / task
    vl_root = ROOT / "vllm" / task

    if not hf_root.exists():
        continue

    for hf_path in sorted(
        hf_root.glob("condition_*/summary.json")
    ):

        condition = hf_path.parent.name

        vl_path = (
            vl_root
            / condition
            / "summary.json"
        )

        if not vl_path.exists():
            continue

        with open(hf_path) as f:
            hf = json.load(f)

        with open(vl_path) as f:
            vl = json.load(f)

        rgb_same = (
            hf["initial_rgb_sha1"]
            == vl["initial_rgb_sha1"]
        )

        outcome_same = (
            bool(hf["success"])
            == bool(vl["success"])
        )

        pairs.append(
            {
                "task": task,
                "condition": condition,

                "rgb_same": rgb_same,

                "hf_success": hf["success"],
                "vllm_success": vl["success"],

                "outcome_same": outcome_same,

                "hf_steps": hf["num_steps"],
                "vllm_steps": vl["num_steps"],

                "hf_time": hf["elapsed_s"],
                "vllm_time": vl["elapsed_s"],
            }
        )


print(
    f"{'task':18s} "
    f"{'condition':14s} "
    f"{'RGB':5s} "
    f"{'HF':6s} "
    f"{'vLLM':6s} "
    f"{'same':6s} "
    f"{'HFstep':7s} "
    f"{'VLstep':7s}"
)

print("-" * 80)


for x in pairs:

    print(
        f"{x['task']:18s} "
        f"{x['condition']:14s} "
        f"{str(x['rgb_same']):5s} "
        f"{str(x['hf_success']):6s} "
        f"{str(x['vllm_success']):6s} "
        f"{str(x['outcome_same']):6s} "
        f"{x['hf_steps']:7d} "
        f"{x['vllm_steps']:7d}"
    )


valid = [
    x for x in pairs
    if x["rgb_same"]
]

agree = sum(
    x["outcome_same"]
    for x in valid
)


print("\n===== SUMMARY =====")

print("valid matched pairs:", len(valid))

if valid:
    print(
        "outcome agreement:",
        f"{agree}/{len(valid)}",
        f"= {agree / len(valid):.1%}",
    )

    hf_total_time = sum(
        x["hf_time"]
        for x in valid
    )

    vl_total_time = sum(
        x["vllm_time"]
        for x in valid
    )

    print(
        "overall speedup:",
        hf_total_time / vl_total_time
    )