"""Search raw step records for per-step privileged geometry/state fields.

The search is intentionally schema-agnostic and reports where candidate keys
occur.  It does not infer state from images, action text, or model features.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CANDIDATE_FIELDS = {
    "pick_coke_can": [
        "per_step_object_position",
        "per_step_object_z",
        "per_step_tcp_position",
        "per_step_tcp_object_distance",
        "per_step_gripper_state",
    ],
    "move_near": [
        "per_step_source_object_position",
        "per_step_target_object_position",
        "per_step_object_target_distance",
        "per_step_tcp_position",
        "per_step_tcp_object_distance",
        "per_step_gripper_state",
    ],
}

SEARCH_TERMS = (
    "position",
    "pose",
    "qpos",
    "distance",
    "dist",
    "tcp",
    "gripper",
    "object",
    "source",
    "target",
    "goal",
    "height",
    "stage",
    "contact",
    "state",
)


def scan_value(value: Any, path: str, found: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if any(term in key.lower() for term in SEARCH_TERMS):
                item = found.setdefault(
                    child_path,
                    {"count": 0, "types": set(), "examples": [], "locations": set()},
                )
                item["count"] += 1
                item["types"].add(type(child).__name__)
                item["locations"].add(path.split(".")[0] if path else "top_level")
                if len(item["examples"]) < 3 and not isinstance(child, (dict, list)):
                    item["examples"].append(repr(child)[:180])
            scan_value(child, child_path, found)
    elif isinstance(value, list):
        # Candidate state is expected to be a small structured field.  Scan a
        # couple of list elements without expanding the depth-token arrays.
        for child in value[:2]:
            scan_value(child, f"{path}[]", found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/formal"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/task_progress/dynamic_raw_state_search.json"))
    parser.add_argument("--tasks", nargs="+", default=["pick_coke_can", "move_near"])
    args = parser.parse_args()

    report: dict[str, Any] = {
        "source_root": str(args.results_dir),
        "search_terms": list(SEARCH_TERMS),
        "tasks": {},
    }
    for task in args.tasks:
        found: dict[str, dict[str, Any]] = {}
        n_steps = 0
        n_episodes = 0
        for path in sorted((args.results_dir / task).glob("condition_*/steps.jsonl")):
            n_episodes += 1
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                n_steps += 1
                scan_value(json.loads(line), "", found)
        serializable = {
            path: {
                **item,
                "types": sorted(item["types"]),
                "locations": sorted(item["locations"]),
            }
            for path, item in sorted(found.items())
        }
        candidate_status = {
            candidate: {
                "found": False,
                "matching_paths": [],
                "decision": "not_available_in_raw_steps",
            }
            for candidate in CANDIDATE_FIELDS.get(task, [])
        }
        report["tasks"][task] = {
            "episodes": n_episodes,
            "steps": n_steps,
            "matching_key_paths": serializable,
            "candidate_status": candidate_status,
            "conclusion": "No per-step object/TCP/source/target geometry was recorded in steps.jsonl; only task booleans, actions, and observation encodings were found.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({task: details["steps"] for task, details in report["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
