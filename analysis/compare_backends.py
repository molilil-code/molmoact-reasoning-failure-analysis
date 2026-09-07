import json
from pathlib import Path

root = Path(
    "/root/autodl-tmp/molmoact_project/results/backend_test"
)

task = "pick_coke_can"
condition = "condition_0001"

rows = {}

for backend in ["hf", "vllm"]:

    path = (
        root
        / backend
        / task
        / condition
        / "summary.json"
    )

    with open(path, "r") as f:
        rows[backend] = json.load(f)


print("===== CLOSED-LOOP COMPARISON =====")

for backend in ["hf", "vllm"]:

    x = rows[backend]

    print(f"\n{backend.upper()}")
    print("success:", x["success"])
    print("steps:", x["num_steps"])
    print("time:", x["elapsed_s"])
    print("stop:", x["stop_reason"])


hf_t = rows["hf"]["elapsed_s"]
vl_t = rows["vllm"]["elapsed_s"]

print("\nSpeedup HF/vLLM:", hf_t / vl_t)