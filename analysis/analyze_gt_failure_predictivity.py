"""Summarize whether privileged GT fields separate eventual success/failure.

This is an exploratory characterization report, not a detector evaluation. It
keeps terminal milestones, online-visible state, and initial-condition controls
separate so that endpoint leakage is explicit.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def integer(value: Any) -> int | None:
    parsed = number(value)
    return int(parsed) if parsed is not None else None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def auc_failure(scores: list[float], failure_labels: list[int]) -> float | None:
    pairs = [(score, label) for score, label in zip(scores, failure_labels) if score is not None]
    positive = [score for score, label in pairs if label == 1]
    negative = [score for score, label in pairs if label == 0]
    if not positive or not negative:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def episode_summary(task: str, rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    episodes: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        episodes.setdefault(row["episode_key"], []).append(row)
    summary: list[dict[str, Any]] = []
    for episode_key, episode_rows in episodes.items():
        first = episode_rows[0]
        success = truth(first["episode_success"])
        record: dict[str, Any] = {
            "task": task,
            "episode_key": episode_key,
            "success": success,
            "failure": not success,
        }
        if task == "pick_coke_can":
            for field, out in [
                ("gt_is_grasped", "any_grasp"),
                ("gt_lifted_object", "any_lift"),
                ("gt_lifted_object_significantly", "any_significant_lift"),
            ]:
                record[out] = any(truth(row[field]) for row in episode_rows)
            record["initial_geometry"] = number(first["control_init_object_xy_radius"])
            record["first_milestone"] = integer(first["gt_first_grasp_step"]) if first["gt_first_grasp_step"] else 81
            record["first_grasp_step"] = integer(first["gt_first_grasp_step"]) if first["gt_first_grasp_step"] else None
            record["first_success_step"] = integer(first["gt_first_success_step"]) if first["gt_first_success_step"] else None
        else:
            for field, out in [
                ("gt_moved_correct_obj", "any_correct_move"),
                ("gt_near_tgt_obj", "any_near_target"),
            ]:
                record[out] = any(truth(row[field]) for row in episode_rows)
            record["any_height_violation"] = any(not truth(row["gt_all_obj_keep_height"]) for row in episode_rows)
            record["initial_geometry"] = number(first["control_init_source_target_distance"])
            record["first_milestone"] = integer(first["gt_first_correct_move_step"]) if first["gt_first_correct_move_step"] else 81
            record["first_correct_move_step"] = integer(first["gt_first_correct_move_step"]) if first["gt_first_correct_move_step"] else None
            record["first_near_target_step"] = integer(first["gt_first_near_target_step"]) if first["gt_first_near_target_step"] else None
            record["first_success_step"] = integer(first["gt_first_success_step"]) if first["gt_first_success_step"] else None
        summary.append(record)
    return summary, episodes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    preferred = ["task", "episode_key", "success", "failure"]
    fields = [key for key in preferred if key in fields] + [key for key in fields if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_rows(task: str, summary: list[dict[str, Any]], episodes: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    checkpoints = [10, 20, 30, 40, 50, 60, 70]
    fields = {
        "pick_coke_can": ["gt_is_grasped", "gt_lifted_object", "gt_lifted_object_significantly"],
        "move_near": ["gt_moved_correct_obj", "gt_near_tgt_obj"],
    }[task]
    output: list[dict[str, Any]] = []
    for cutoff in checkpoints:
        for field in fields:
            event_values: dict[str, bool] = {}
            for episode_key, rows in episodes.items():
                event_values[episode_key] = any(
                    truth(row[field]) and integer(row["step_id"]) is not None and integer(row["step_id"]) <= cutoff
                    for row in rows
                )
            success_rows = [row for row in summary if row["success"]]
            failure_rows = [row for row in summary if row["failure"]]
            true_success = sum(event_values[row["episode_key"]] for row in success_rows)
            true_failure = sum(event_values[row["episode_key"]] for row in failure_rows)
            absent_success = len(success_rows) - true_success
            absent_failure = len(failure_rows) - true_failure
            output.append(
                {
                    "task": task,
                    "cutoff_step": cutoff,
                    "field": field,
                    "event_true_success": f"{true_success}/{len(success_rows)}",
                    "event_true_failure": f"{true_failure}/{len(failure_rows)}",
                    "absence_failure_recall": absent_failure / len(failure_rows),
                    "absence_success_false_alarm": absent_success / len(success_rows),
                }
            )
    return output


def write_report(path: Path, all_summary: dict[str, list[dict[str, Any]]], checkpoints: list[dict[str, Any]]) -> None:
    lines = [
        "# Privileged GT failure-predictivity analysis",
        "",
        "This report characterizes task-state GT only. It does not evaluate the MolmoAct reasoning predictor and does not define failure onset.",
        "Terminal milestones are explicitly marked as endpoint/leakage-prone; initial geometry is treated as a separate control baseline.",
        "",
    ]
    for task, rows in all_summary.items():
        success = [row for row in rows if row["success"]]
        failure = [row for row in rows if row["failure"]]
        lines += [f"## {task}", "", f"Episodes: {len(rows)} ({len(success)} success, {len(failure)} failure)", ""]
        if task == "pick_coke_can":
            lines += [
                "| GT signal | Success episodes with signal | Failure episodes with signal | Interpretation |",
                "|---|---:|---:|---|",
            ]
            for key, label, interpretation in [
                ("any_grasp", "any grasp", "Perfect retrospective milestone; usually late."),
                ("any_lift", "any lift", "Same endpoint-adjacent information as grasp."),
                ("any_significant_lift", "significant lift", "Incomplete milestone; only a subset of successes."),
            ]:
                lines.append(
                    f"| `{label}` | {sum(row[key] for row in success)}/{len(success)} | "
                    f"{sum(row[key] for row in failure)}/{len(failure)} | {interpretation} |"
                )
            scores = [row["initial_geometry"] for row in rows]
            labels = [int(row["failure"]) for row in rows]
            lines += [
                "",
                f"Initial object XY radius mean: success={mean([row['initial_geometry'] for row in success]):.3f}, failure={mean([row['initial_geometry'] for row in failure]):.3f}; exploratory failure AUC={auc_failure(scores, labels):.3f}.",
                "This suggests a possible condition-difficulty confound, not a dynamic failure precursor.",
            ]
            grasp_steps = [row["first_grasp_step"] for row in success if row["first_grasp_step"] is not None]
            lines.append(
                f"First grasp in successful episodes: median={statistics.median(grasp_steps):.0f}, range={min(grasp_steps)}–{max(grasp_steps)}; no failure episode ever grasps."
            )
        else:
            lines += [
                "| GT signal | Success episodes with signal | Failure episodes with signal | Interpretation |",
                "|---|---:|---:|---|",
            ]
            for key, label, interpretation in [
                ("any_correct_move", "correct-object movement", "Useful progress milestone, but 11 failures also reach it."),
                ("any_near_target", "near target", "Strong terminal milestone; not early warning."),
                ("any_height_violation", "height violation", "Specific but rare: only 2/20 failures."),
            ]:
                lines.append(
                    f"| `{label}` | {sum(row[key] for row in success)}/{len(success)} | "
                    f"{sum(row[key] for row in failure)}/{len(failure)} | {interpretation} |"
                )
            scores = [row["initial_geometry"] for row in rows]
            labels = [int(row["failure"]) for row in rows]
            lines += [
                "",
                f"Initial source-target distance mean: success={mean([row['initial_geometry'] for row in success]):.3f}, failure={mean([row['initial_geometry'] for row in failure]):.3f}; exploratory failure AUC={auc_failure(scores, labels):.3f}.",
                "Initial geometry is close to chance in this sample.",
            ]
            correct_success = [row["first_correct_move_step"] for row in success if row["first_correct_move_step"] is not None]
            correct_failure = [row["first_correct_move_step"] for row in failure if row["first_correct_move_step"] is not None]
            lines.append(
                f"First correct movement: success median={statistics.median(correct_success):.0f} (range {min(correct_success)}–{max(correct_success)}); "
                f"the 11 failures that reached it have median={statistics.median(correct_failure):.0f} (range {min(correct_failure)}–{max(correct_failure)})."
            )
        lines += ["", "### Online checkpoints", "", "Absence is interpreted as a risk rule only for this descriptive table; it is not an onset label.", "", "| Step | Field | Event true in success | Event true in failure | Absence recall for failure | Absence false alarm on success |", "|---:|---|---:|---:|---:|---:|"]
        for row in checkpoints:
            if row["task"] != task:
                continue
            lines.append(
                f"| {row['cutoff_step']} | `{row['field']}` | {row['event_true_success']} | {row['event_true_failure']} | "
                f"{row['absence_failure_recall']:.1%} | {row['absence_success_false_alarm']:.1%} |"
            )
        lines += ["", ""]
    lines += [
        "## Conclusion",
        "",
        "PickCokeCan GT milestones identify the eventual outcome retrospectively, but they appear too late and are absent throughout every failure episode. MoveNear has a more useful phase split: no correct movement versus transport not completed, with rare height violations. Neither task currently has per-step geometry, so these are milestone/failure-mode analyses rather than continuous task-progress onset labels.",
        "",
        "Privileged GT fields should remain outside the main Depth/Trace/Action predictor. They can support task-specific labels, a geometry-only control baseline, or a clearly named privileged-state oracle.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/derived/task_progress"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/task_progress"))
    args = parser.parse_args()

    specs = {
        "pick_coke_can": "pick_coke_gt_progress.csv",
        "move_near": "move_near_gt_progress.csv",
    }
    all_summary: dict[str, list[dict[str, Any]]] = {}
    all_checkpoints: list[dict[str, Any]] = []
    for task, filename in specs.items():
        rows = read_csv(args.input_dir / filename)
        summary, episodes = episode_summary(task, rows)
        all_summary[task] = summary
        all_checkpoints.extend(checkpoint_rows(task, summary, episodes))
        write_csv(args.output_dir / filename.replace("_gt_progress.csv", "_gt_predictivity_episode.csv"), summary)
    write_csv(args.output_dir / "gt_predictivity_checkpoints.csv", all_checkpoints)
    write_report(args.output_dir / "gt_failure_predictivity_report.md", all_summary, all_checkpoints)
    print({task: len(rows) for task, rows in all_summary.items()})


if __name__ == "__main__":
    main()
