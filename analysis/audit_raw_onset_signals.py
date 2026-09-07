"""Audit raw steps.jsonl for task-level failure-onset supervision.

This intentionally inspects the environment fields recorded after each action,
not the derived feature CSVs.  It is meant to answer whether an episode has a
defensible per-step outcome/state signal from which an onset can be labelled.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


TASK_KEYS = {
    "pick_coke_can": [
        "is_grasped",
        "consecutive_grasp",
        "lifted_object",
        "lifted_object_significantly",
        "success",
    ],
    "move_near": [
        "moved_correct_obj",
        "moved_wrong_obj",
        "near_tgt_obj",
        "is_closest_to_tgt",
        "all_obj_keep_height",
        "success",
    ],
}


def first_true(values: list[object]) -> int | None:
    for idx, value in enumerate(values):
        if value is True:
            return idx
    return None


def transitions(values: list[object]) -> int:
    bool_values = [v for v in values if isinstance(v, bool)]
    return sum(a != b for a, b in zip(bool_values, bool_values[1:]))


def audit_task(root: Path, task: str) -> tuple[list[dict], Counter, dict[str, set[str]]]:
    episode_rows: list[dict] = []
    field_counts: Counter = Counter()
    field_values: dict[str, set[str]] = defaultdict(set)
    for path in sorted((root / task).glob("condition_*/steps.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        infos = [row.get("post_action", {}).get("info", {}) for row in rows]
        condition = path.parent.name
        outcome = "success" if infos and infos[-1].get("success") is True else "failure_truncated"
        record = {
            "task": task,
            "condition": condition,
            "n_steps": len(rows),
            "outcome": outcome,
            "terminated": rows[-1].get("post_action", {}).get("terminated") if rows else None,
            "truncated": rows[-1].get("post_action", {}).get("truncated") if rows else None,
        }
        for key in TASK_KEYS[task]:
            values = [info.get(key) for info in infos]
            for info in infos:
                if key in info:
                    field_counts[key] += 1
                    field_values[key].add(repr(info[key]))
            record[f"{key}_first_true_step"] = first_true(values)
            record[f"{key}_true_steps"] = sum(value is True for value in values)
            record[f"{key}_transitions"] = transitions(values)
            record[f"{key}_final"] = values[-1] if values else None
        episode_rows.append(record)
    return episode_rows, field_counts, field_values


def write_outputs(root: Path, out_dir: Path, tasks: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    inventory_rows: list[dict] = []
    task_stats: dict[str, dict] = {}
    for task in tasks:
        rows, counts, values = audit_task(root, task)
        all_rows.extend(rows)
        task_stats[task] = {
            "episodes": len(rows),
            "successes": sum(row["outcome"] == "success" for row in rows),
            "failures": sum(row["outcome"] != "success" for row in rows),
            "lengths": [row["n_steps"] for row in rows],
        }
        for field in TASK_KEYS[task]:
            inventory_rows.append(
                {
                    "task": task,
                    "field": f"post_action.info.{field}",
                    "present_steps": counts[field],
                    "distinct_scalar_values": "|".join(sorted(values[field])),
                }
            )

    episode_path = out_dir / "raw_onset_episode_summary.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in all_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    inventory_path = out_dir / "raw_onset_field_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)

    lines = [
        "# Raw onset-signal audit: PickCokeCan and MoveNear",
        "",
        "This report is based directly on `results/formal/*/condition_*/steps.jsonl`, not on derived CSVs.",
        "The decision criterion is whether a recorded environment signal identifies an intermediate,",
        "time-localized failure state before the episode terminates.",
        "",
        "## Episode counts",
        "",
        "| Task | Episodes | Success | Failure/truncated | Step count (min/median/max) |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, stats in task_stats.items():
        lengths = stats["lengths"]
        lines.append(
            f"| {task} | {stats['episodes']} | {stats['successes']} | {stats['failures']} | "
            f"{min(lengths)}/{median(lengths):g}/{max(lengths)} |"
        )

    lines += [
        "",
        "## Raw per-step fields",
        "",
        "The only task-state fields in `post_action.info` are booleans plus `elapsed_steps`; there are",
        "no per-step object poses, source-target distances, qpos, stage IDs, contact forces, or failure",
        "reasons. `reset_info` in each episode summary contains initial poses only and cannot provide a",
        "time series. `pre_action.trace` is the model-provided 2-D trajectory input, not environment ground truth.",
        "",
        "### PickCokeCan",
        "",
        "- `is_grasped` and `lifted_object` become true only in successful episodes; their first true step can precede terminal success.",
        "- `lifted_object_significantly` is true in only a subset of successful episodes; it is not a complete success signal.",
        "- Failure episodes remain `False` for all task-state booleans and end by `truncated=true` at step 80 (except shorter successful runs).",
        "- Therefore the raw log has no observed failure transition/onset; assigning onset at truncation would be an artificial censoring convention.",
        "",
        "### MoveNear",
        "",
        "- `near_tgt_obj` becomes true only in successful episodes (sometimes several steps before terminal success) and never becomes true in failures.",
        "- `moved_correct_obj` can become true in both successful and failed episodes (11/20 failures), so it is an intermediate-progress signal, not failure onset.",
        "- `is_closest_to_tgt` is already true at step 0 in many episodes and is therefore not a monotone temporal milestone.",
        "- Failure episodes end by `truncated=true` at step 80 with `near_tgt_obj=false`; no distance or failure event is recorded.",
        "- Therefore the raw log supports terminal success/progress analysis, but not a defensible per-step failure onset.",
        "",
        "## Decision",
        "",
        "**Do not define an onset-based AUROC/AUPRC for PickCokeCan or MoveNear from these raw logs.**",
        "Keep them as episode-level outcome/progress analyses unless new rollouts record a per-step",
        "distance/state margin or an explicit failure event. If a future run adds such a signal, define",
        "onset as the first sustained crossing of that signal and audit it before computing lead time.",
        "",
        "Generated files: `raw_onset_episode_summary.csv` and `raw_onset_field_inventory.csv`.",
    ]
    (out_dir / "raw_onset_signal_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/formal"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_onset_audit"))
    parser.add_argument("--tasks", nargs="+", default=list(TASK_KEYS), choices=list(TASK_KEYS))
    args = parser.parse_args()
    write_outputs(args.results_dir, args.output_dir, args.tasks)


if __name__ == "__main__":
    main()
