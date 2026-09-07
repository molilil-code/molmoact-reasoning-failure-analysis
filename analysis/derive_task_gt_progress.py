"""Derive task-progress and failure-mode GT from raw task-state CSVs.

Inputs are produced by ``extract_raw_task_progress.py``.  The outputs remain
privileged GT analysis artifacts: they are not predictor features and do not
define failure onset or train a detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any


def parse_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    return int(number) if number is not None else None


def first_true(rows: list[dict[str, Any]], field: str) -> int | None:
    for row in rows:
        if parse_bool(row.get(field)) is True:
            return parse_int(row.get("step_id"))
    return None


def count_true(rows: list[dict[str, Any]], field: str) -> int:
    return sum(parse_bool(row.get(field)) is True for row in rows)


def with_common_episode_fields(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
    first_steps: dict[str, int | None],
) -> None:
    for name, step in first_steps.items():
        row[f"gt_first_{name}_step"] = step


def add_pick_progress(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return rows, {}
    success = parse_bool(rows[0].get("episode_success")) is True
    first_steps = {
        "grasp": first_true(rows, "gt_is_grasped"),
        "lift": first_true(rows, "gt_lifted_object"),
        "significant_lift": first_true(rows, "gt_lifted_object_significantly"),
        "success": first_true(rows, "gt_success"),
    }
    totals = {
        "grasp": count_true(rows, "gt_is_grasped"),
        "lift": count_true(rows, "gt_lifted_object"),
        "significant_lift": count_true(rows, "gt_lifted_object_significantly"),
    }
    if success:
        failure_mode = "success"
    elif first_steps["grasp"] is None:
        failure_mode = "never_grasped"
    elif first_steps["lift"] is None:
        failure_mode = "grasped_but_not_lifted"
    elif first_steps["significant_lift"] is None:
        failure_mode = "lifted_but_not_significantly_lifted"
    else:
        failure_mode = "significant_lift_but_not_success"

    previous_phase: int | None = None
    grasp_streak = 0
    lift_streak = 0
    init_x = parse_float(rows[0].get("gt_object_init_x"))
    init_y = parse_float(rows[0].get("gt_object_init_y"))
    xy_radius = math.hypot(init_x, init_y) if init_x is not None and init_y is not None else None
    for row in rows:
        grasped = parse_bool(row.get("gt_is_grasped")) is True
        lifted = parse_bool(row.get("gt_lifted_object")) is True
        significant = parse_bool(row.get("gt_lifted_object_significantly")) is True
        is_success = parse_bool(row.get("gt_success")) is True
        phase = 4 if is_success else 3 if significant else 2 if lifted else 1 if grasped else 0
        grasp_streak = grasp_streak + 1 if grasped else 0
        lift_streak = lift_streak + 1 if lifted else 0
        step = parse_int(row.get("step_id"))
        row.update(
            {
                "gt_pick_phase": phase,
                "gt_pick_phase_change": int(previous_phase is not None and phase != previous_phase),
                "gt_grasp_streak": grasp_streak,
                "gt_lift_streak": lift_streak,
                "gt_time_since_first_grasp": (
                    step - first_steps["grasp"]
                    if step is not None and first_steps["grasp"] is not None and step >= first_steps["grasp"]
                    else ""
                ),
                "gt_grasp_duration_steps": totals["grasp"],
                "gt_lift_duration_steps": totals["lift"],
                "gt_significant_lift_duration_steps": totals["significant_lift"],
                "control_init_object_xy_radius": xy_radius if xy_radius is not None else "",
                "failure_mode_primary": failure_mode,
                "failure_mode_never_grasped": int(failure_mode == "never_grasped"),
                "failure_mode_grasped_but_not_lifted": int(failure_mode == "grasped_but_not_lifted"),
                "failure_mode_lifted_but_not_significantly_lifted": int(
                    failure_mode == "lifted_but_not_significantly_lifted"
                ),
                "failure_mode_significant_lift_but_not_success": int(
                    failure_mode == "significant_lift_but_not_success"
                ),
            }
        )
        row.update({f"gt_first_{name}_step": value if value is not None else "" for name, value in first_steps.items()})
        previous_phase = phase

    episode = {
        "task": "pick_coke_can",
        "episode_key": rows[0].get("episode_key"),
        "condition_id": rows[0].get("condition_id"),
        "episode_success": int(success),
        "num_steps": len(rows),
        **{f"gt_first_{name}_step": value if value is not None else "" for name, value in first_steps.items()},
        **{f"gt_{name}_duration_steps": value for name, value in totals.items()},
        "failure_mode_primary": failure_mode,
        "failure_mode_never_grasped": int(failure_mode == "never_grasped"),
        "failure_mode_grasped_but_not_lifted": int(failure_mode == "grasped_but_not_lifted"),
        "failure_mode_lifted_but_not_significantly_lifted": int(
            failure_mode == "lifted_but_not_significantly_lifted"
        ),
        "failure_mode_significant_lift_but_not_success": int(
            failure_mode == "significant_lift_but_not_success"
        ),
        "control_init_object_xy_radius": xy_radius if xy_radius is not None else "",
    }
    return rows, episode


def add_move_progress(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return rows, {}
    success = parse_bool(rows[0].get("episode_success")) is True
    first_steps = {
        "correct_move": first_true(rows, "gt_moved_correct_obj"),
        "near_target": first_true(rows, "gt_near_tgt_obj"),
        "success": first_true(rows, "gt_success"),
        "height_violation": next(
            (parse_int(row.get("step_id")) for row in rows if parse_bool(row.get("gt_all_obj_keep_height")) is False),
            None,
        ),
    }
    correct_total = count_true(rows, "gt_moved_correct_obj")
    near_total = count_true(rows, "gt_near_tgt_obj")
    height_violated = first_steps["height_violation"] is not None
    failure_no_correct = (not success) and first_steps["correct_move"] is None
    failure_transport = (not success) and first_steps["correct_move"] is not None and first_steps["near_target"] is None
    failure_height = (not success) and height_violated
    if success:
        failure_mode = "success"
    elif height_violated:
        failure_mode = "height_violation"
    elif first_steps["correct_move"] is None:
        failure_mode = "no_correct_move"
    elif first_steps["near_target"] is None:
        failure_mode = "transport_not_completed"
    else:
        failure_mode = "other_failure"

    source_x = parse_float(rows[0].get("gt_source_init_x"))
    source_y = parse_float(rows[0].get("gt_source_init_y"))
    source_z = parse_float(rows[0].get("gt_source_init_z"))
    target_x = parse_float(rows[0].get("gt_target_init_x"))
    target_y = parse_float(rows[0].get("gt_target_init_y"))
    target_z = parse_float(rows[0].get("gt_target_init_z"))
    delta = (
        (target_x - source_x, target_y - source_y, target_z - source_z)
        if None not in (source_x, source_y, source_z, target_x, target_y, target_z)
        else None
    )
    init_distance = math.sqrt(sum(value * value for value in delta)) if delta is not None else None
    init_xy_distance = math.hypot(delta[0], delta[1]) if delta is not None else None

    previous_phase: int | None = None
    correct_streak = 0
    height_violation_streak = 0
    cumulative_correct = 0
    for index, row in enumerate(rows):
        moved = parse_bool(row.get("gt_moved_correct_obj")) is True
        near = parse_bool(row.get("gt_near_tgt_obj")) is True
        is_success = parse_bool(row.get("gt_success")) is True
        keep_height = parse_bool(row.get("gt_all_obj_keep_height")) is True
        phase = 3 if is_success else 2 if near else 1 if moved else 0
        correct_streak = correct_streak + 1 if moved else 0
        height_violation_streak = height_violation_streak + 1 if not keep_height else 0
        cumulative_correct += int(moved)
        step = parse_int(row.get("step_id"))
        row.update(
            {
                "gt_move_phase": phase,
                "gt_move_phase_change": int(previous_phase is not None and phase != previous_phase),
                "gt_correct_move_streak": correct_streak,
                "gt_correct_move_fraction": cumulative_correct / (index + 1),
                "gt_time_since_first_correct_move": (
                    step - first_steps["correct_move"]
                    if step is not None
                    and first_steps["correct_move"] is not None
                    and step >= first_steps["correct_move"]
                    else ""
                ),
                "gt_height_violation": int(not keep_height),
                "gt_height_violation_streak": height_violation_streak,
                "gt_correct_move_duration_steps": correct_total,
                "gt_near_target_duration_steps": near_total,
                "gt_first_correct_move_step": first_steps["correct_move"] if first_steps["correct_move"] is not None else "",
                "gt_first_near_target_step": first_steps["near_target"] if first_steps["near_target"] is not None else "",
                "gt_first_success_step": first_steps["success"] if first_steps["success"] is not None else "",
                "gt_first_height_violation_step": first_steps["height_violation"] if first_steps["height_violation"] is not None else "",
                "gt_correct_move_fraction_episode": correct_total / len(rows),
                "control_init_source_target_dx": delta[0] if delta is not None else "",
                "control_init_source_target_dy": delta[1] if delta is not None else "",
                "control_init_source_target_dz": delta[2] if delta is not None else "",
                "control_init_source_target_distance": init_distance if init_distance is not None else "",
                "control_init_source_target_xy_distance": init_xy_distance if init_xy_distance is not None else "",
                "failure_mode_primary": failure_mode,
                # Failure-mode flags are not mutually exclusive: a transport
                # failure can also violate the height constraint.  Keep the
                # primary mode for compact summaries, but preserve all flags.
                "failure_mode_no_correct_move": int(failure_no_correct),
                "failure_mode_transport_not_completed": int(failure_transport),
                "failure_mode_height_violation": int(failure_height),
            }
        )
        previous_phase = phase

    episode = {
        "task": "move_near",
        "episode_key": rows[0].get("episode_key"),
        "condition_id": rows[0].get("condition_id"),
        "episode_success": int(success),
        "num_steps": len(rows),
        "gt_first_correct_move_step": first_steps["correct_move"] if first_steps["correct_move"] is not None else "",
        "gt_first_near_target_step": first_steps["near_target"] if first_steps["near_target"] is not None else "",
        "gt_first_success_step": first_steps["success"] if first_steps["success"] is not None else "",
        "gt_first_height_violation_step": first_steps["height_violation"] if first_steps["height_violation"] is not None else "",
        "gt_correct_move_duration_steps": correct_total,
        "gt_near_target_duration_steps": near_total,
        "gt_correct_move_fraction_episode": correct_total / len(rows),
        "gt_height_violation": int(height_violated),
        "failure_mode_primary": failure_mode,
        "failure_mode_no_correct_move": int(failure_no_correct),
        "failure_mode_transport_not_completed": int(failure_transport),
        "failure_mode_height_violation": int(failure_height),
        "control_init_source_target_dx": delta[0] if delta is not None else "",
        "control_init_source_target_dy": delta[1] if delta is not None else "",
        "control_init_source_target_dz": delta[2] if delta is not None else "",
        "control_init_source_target_distance": init_distance if init_distance is not None else "",
        "control_init_source_target_xy_distance": init_xy_distance if init_xy_distance is not None else "",
    }
    return rows, episode


def read_grouped(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(row["episode_key"], []).append(row)
    return groups


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = ["task", "episode_key", "condition_id", "step_id", "episode_success", "num_steps"]
    fields = sorted({key for row in rows for key in row})
    fields = [key for key in preferred if key in fields] + [key for key in fields if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/derived/task_progress"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/task_progress"))
    args = parser.parse_args()

    metadata: dict[str, Any] = {
        "scope": "privileged GT progress/milestone and failure-mode derivation only",
        "onset_definition": None,
        "predictor_features": None,
        "tasks": {},
    }
    configs = {
        "pick_coke_can": ("pick_coke_task_state.csv", "pick_coke_gt_progress.csv", "pick_coke_gt_episode_summary.csv", add_pick_progress),
        "move_near": ("move_near_task_state.csv", "move_near_gt_progress.csv", "move_near_gt_episode_summary.csv", add_move_progress),
    }
    for task, (input_name, output_name, episode_name, transform) in configs.items():
        groups = read_grouped(args.input_dir / input_name)
        step_rows: list[dict[str, Any]] = []
        episode_rows: list[dict[str, Any]] = []
        for rows in groups.values():
            transformed, episode = transform(rows)
            step_rows.extend(transformed)
            episode_rows.append(episode)
        write_csv(args.output_dir / output_name, step_rows)
        write_csv(args.output_dir / episode_name, episode_rows)
        metadata["tasks"][task] = {
            "input": input_name,
            "step_output": output_name,
            "episode_output": episode_name,
            "episodes": len(episode_rows),
            "steps": len(step_rows),
            "failure_mode_counts": {
                mode: sum(row.get("failure_mode_primary") == mode for row in episode_rows)
                for mode in sorted({row.get("failure_mode_primary") for row in episode_rows})
            },
            "failure_mode_flag_counts": {
                "failure_mode_no_correct_move": sum(
                    row.get("failure_mode_no_correct_move") == 1 for row in episode_rows
                ),
                "failure_mode_transport_not_completed": sum(
                    row.get("failure_mode_transport_not_completed") == 1 for row in episode_rows
                ),
                "failure_mode_height_violation": sum(
                    row.get("failure_mode_height_violation") == 1 for row in episode_rows
                ),
            },
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_gt_progress_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["tasks"], indent=2))


if __name__ == "__main__":
    main()
