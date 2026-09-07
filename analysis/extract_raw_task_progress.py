"""Extract per-step task-state GT from raw formal rollout logs.

This script deliberately does *not* use derived feature CSVs, define failure
onset, construct predictor features, or train a model.  It records the
environment-side fields available in ``steps.jsonl`` and the episode-initial
poses stored in each episode's ``summary.json``.  Initial poses are repeated
on each step with an explicit ``initial_only`` note in the inventory; they are
not treated as per-step trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_STEP_FIELDS: dict[str, dict[str, str]] = {
    "pick_coke_can": {
        "gt_is_grasped": "is_grasped",
        "gt_consecutive_grasp": "consecutive_grasp",
        "gt_lifted_object": "lifted_object",
        "gt_lifted_object_significantly": "lifted_object_significantly",
        "gt_success": "success",
    },
    "move_near": {
        "gt_all_obj_keep_height": "all_obj_keep_height",
        "gt_moved_correct_obj": "moved_correct_obj",
        "gt_moved_wrong_obj": "moved_wrong_obj",
        "gt_near_tgt_obj": "near_tgt_obj",
        "gt_is_closest_to_tgt": "is_closest_to_tgt",
        "gt_success": "success",
    },
    "open_drawer": {
        "gt_drawer_qpos": "qpos",
        "gt_success": "success",
    },
}

COMMON_STEP_FIELDS = {
    "gt_reward": ("post_action", "reward"),
    "gt_terminated": ("post_action", "terminated"),
    "gt_truncated": ("post_action", "truncated"),
    "gt_elapsed_steps": ("post_action", "info", "elapsed_steps"),
}

COMMON_STEP_SOURCES = {
    "gt_reward": "steps.post_action.reward",
    "gt_terminated": "steps.post_action.terminated",
    "gt_truncated": "steps.post_action.truncated",
    "gt_elapsed_steps": "steps.post_action.info.elapsed_steps",
}


def nested_get(value: Any, path: tuple[str, ...], default: Any = None) -> Any:
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def flatten_pose(prefix: str, pose: Any) -> dict[str, Any]:
    """Return only the initial position vector from a serialized SAPIEN pose."""
    if not isinstance(pose, dict):
        return {}
    position = pose.get("p")
    if not isinstance(position, list) or len(position) < 3:
        return {}
    return {f"{prefix}_{axis}": position[idx] for idx, axis in enumerate(("x", "y", "z"))}


def initial_fields(task: str, summary: dict[str, Any]) -> dict[str, Any]:
    reset = summary.get("reset_info", {})
    if task == "pick_coke_can":
        return flatten_pose(
            "gt_object_init",
            reset.get("obj_init_pose_wrt_robot_base"),
        )
    if task == "move_near":
        return {
            **flatten_pose(
                "gt_source_init",
                reset.get("episode_source_obj_init_pose_wrt_robot_base"),
            ),
            **flatten_pose(
                "gt_target_init",
                reset.get("episode_target_obj_init_pose_wrt_robot_base"),
            ),
        }
    if task == "open_drawer":
        return {
            **flatten_pose("gt_drawer_init", reset.get("drawer_pose_wrt_robot_base")),
            **flatten_pose("gt_cabinet_init", reset.get("cabinet_pose_wrt_robot_base")),
        }
    return {}


def normalize_csv_value(value: Any) -> Any:
    # Keep raw booleans and numerics typed in the generated CSV rather than
    # converting them to prose labels.  None becomes an empty CSV cell.
    return "" if value is None else value


def read_episode(task: str, steps_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition_dir = steps_path.parent
    summary_path = condition_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    rows = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    condition_id = summary.get("condition_id", rows[0].get("condition_id") if rows else None)
    episode_key = f"{task}/condition_{int(condition_id):04d}" if condition_id is not None else f"{task}/{condition_dir.name}"
    num_steps = summary.get("num_steps", len(rows))
    episode_success = summary.get("success")
    if episode_success is None and rows:
        episode_success = nested_get(rows[-1], ("post_action", "info", "success"))
    initial = initial_fields(task, summary)
    output: list[dict[str, Any]] = []
    for row in rows:
        post = row.get("post_action", {})
        info = post.get("info", {}) if isinstance(post, dict) else {}
        record: dict[str, Any] = {
            "task": task,
            "episode_key": episode_key,
            "condition_id": condition_id,
            "step_id": row.get("step_id"),
            "episode_success": episode_success,
            "num_steps": num_steps,
            "max_steps": summary.get("max_steps"),
            **initial,
        }
        for output_name, path in COMMON_STEP_FIELDS.items():
            record[output_name] = nested_get(row, path)
        for output_name, raw_name in TASK_STEP_FIELDS[task].items():
            record[output_name] = info.get(raw_name)
        output.append(record)

    meta = {
        "task": task,
        "episode_key": episode_key,
        "condition_id": condition_id,
        "steps_path": str(steps_path),
        "summary_path": str(summary_path),
        "n_steps": len(output),
        "episode_success": episode_success,
        "initial_fields": sorted(initial),
    }
    return output, meta


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def build_inventory(task: str, rows: list[dict[str, Any]], metas: list[dict[str, Any]]) -> dict[str, Any]:
    fields = sorted({key for row in rows for key in row if key.startswith("gt_")})
    episodes = sorted({row["episode_key"] for row in rows})
    field_inventory: list[dict[str, Any]] = []
    for field in fields:
        values = [row.get(field) for row in rows if row.get(field) != ""]
        episode_values: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            if row.get(field) != "":
                episode_values[row["episode_key"]].append(row[field])
        distinct = []
        for value in values:
            if value not in distinct:
                distinct.append(value)
        varying_eps = sum(len({repr(value) for value in vals}) > 1 for vals in episode_values.values())
        if field.startswith("gt_object_init_") or field.startswith("gt_source_init_") or field.startswith("gt_target_init_") or field.startswith("gt_drawer_init_") or field.startswith("gt_cabinet_init_"):
            scope = "episode_initial_only"
            source = "summary.reset_info"
        elif field == "gt_drawer_qpos":
            scope = "per_step"
            source = "steps.post_action.info.qpos"
        elif field.startswith("gt_"):
            scope = "per_step"
            raw_name = TASK_STEP_FIELDS[task].get(field)
            if raw_name:
                source = f"steps.post_action.info.{raw_name}"
            else:
                source = COMMON_STEP_SOURCES.get(field, "steps.post_action")
        else:
            scope = "unknown"
            source = "unknown"
        field_inventory.append(
            {
                "name": field,
                "source": source,
                "scope": scope,
                # Initial poses are repeated in the CSV for episode joins, but
                # are not per-step observations; report their temporal
                # coverage as null rather than implying a trajectory exists.
                "step_coverage": (
                    round(len(values) / len(rows), 6) if rows and scope == "per_step" else None
                ),
                "row_coverage": round(len(values) / len(rows), 6) if rows else 0.0,
                "episode_coverage": round(len(episode_values) / len(episodes), 6) if episodes else 0.0,
                "episodes_with_temporal_variation": varying_eps,
                "recorded_as_repeated_column": scope == "episode_initial_only",
                "types": sorted({type_name(value) for value in values}),
                "distinct_values_sample": distinct[:20],
            }
        )

    missing_candidates = {
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
        "open_drawer": [
            "per_step_tcp_position",
            "per_step_handle_position",
        ],
    }
    return {
        "task": task,
        "episodes": len(episodes),
        "steps": len(rows),
        "successful_episodes": sum(meta["episode_success"] is True for meta in metas),
        "failed_or_truncated_episodes": sum(meta["episode_success"] is not True for meta in metas),
        "fields": field_inventory,
        "missing_candidate_fields": missing_candidates[task],
        "progress_definition": None,
        "onset_definition": None,
        "notes": [
            "This first-pass extraction is raw task-state GT only.",
            "No predictor features, progress transforms, onset labels, threshold selection, or model training are performed.",
            "Episode-initial poses are explicitly marked episode_initial_only and are not per-step trajectories.",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    # Put identifiers and time columns first, then raw GT fields.
    preferred = [
        "task",
        "episode_key",
        "condition_id",
        "step_id",
        "episode_success",
        "num_steps",
        "max_steps",
    ]
    fieldnames = [field for field in preferred if field in fieldnames] + [
        field for field in fieldnames if field not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_csv_value(value) for key, value in row.items()})


def write_availability_report(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# Task-state GT availability",
        "",
        "Source: raw `results/formal/*/condition_*/steps.jsonl` plus episode `summary.json` reset metadata.",
        "This is a raw-state inventory only. Progress transforms, onset labels, predictor features, and model fitting are intentionally undefined.",
        "",
    ]
    for task, details in inventory["tasks"].items():
        lines += [
            f"## {task}",
            "",
            f"Episodes: {details['episodes']}  ",
            f"Steps: {details['steps']}  ",
            f"Success episodes: {details['successful_episodes']}  ",
            f"Failure/truncated episodes: {details['failed_or_truncated_episodes']}",
            "",
            "| Field | Source | Scope | Per-step coverage | Episode coverage | Temporal variation |",
            "|---|---|---|---:|---:|---:|",
        ]
        for field in details["fields"]:
            step_coverage = "—" if field["step_coverage"] is None else f"{field['step_coverage']:.1%}"
            lines.append(
                f"| `{field['name']}` | `{field['source']}` | {field['scope']} | "
                f"{step_coverage} | {field['episode_coverage']:.1%} | "
                f"{field['episodes_with_temporal_variation']} episodes |"
            )
        lines += [
            "",
            "Missing candidate fields:",
            "",
        ]
        lines.extend(f"- `{field}`" for field in details["missing_candidate_fields"])
        lines += ["", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/formal"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/task_progress"))
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["pick_coke_can", "move_near", "open_drawer"],
        choices=["pick_coke_can", "move_near", "open_drawer"],
    )
    args = parser.parse_args()

    inventory: dict[str, Any] = {
        "source_root": str(args.results_dir),
        "generated_by": "analysis/extract_raw_task_progress.py",
        "scope": "raw task-state GT extraction only; onset/progress intentionally undefined",
        "tasks": {},
    }
    for task in args.tasks:
        all_rows: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []
        for steps_path in sorted((args.results_dir / task).glob("condition_*/steps.jsonl")):
            rows, meta = read_episode(task, steps_path)
            all_rows.extend(rows)
            metas.append(meta)
        output_name = {
            "pick_coke_can": "pick_coke_task_state.csv",
            "move_near": "move_near_task_state.csv",
            "open_drawer": "open_drawer_task_state.csv",
        }[task]
        write_csv(args.output_dir / output_name, all_rows)
        inventory["tasks"][task] = build_inventory(task, all_rows, metas)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_state_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_availability_report(args.output_dir / "task_state_availability.md", inventory)
    print(json.dumps({task: inventory["tasks"][task]["steps"] for task in args.tasks}, indent=2))


if __name__ == "__main__":
    main()
