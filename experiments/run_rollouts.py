#!/usr/bin/env python3
"""
Controlled MolmoAct rollout collection for SimplerEnv.

Expected project layout
-----------------------
molmoact-reasoning-failure-analysis/
├── src/
│   ├── instrumented_molmoact.py
│   └── task_configs.py
└── experiments/
    └── run_rollouts.py

This runner intentionally separates:
1) condition sampling      -> src/task_configs.py
2) MolmoAct instrumentation -> src/instrumented_molmoact.py
3) rollout execution/logging -> this file

Formal-design principles
------------------------
- Do NOT generate episodes only by looping over random seeds.
- Every rollout is tied to an explicit, reproducible task condition.
- Pre-action signals and post-action labels are stored separately.
- Success follows SimplerEnv's evaluator convention: terminated/done == True.
- Model is loaded once and reused across all conditions.
- Environments are reused while their build-level configuration is unchanged.
- Partial runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from sapien.core import Pose
from transforms3d.euler import euler2quat

from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict,
)


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.instrumented_molmoact import (  # noqa: E402
    InstrumentedMolmoAct,
    RobustJSONEncoder,
    to_jsonable,
)
from src.task_configs import get_conditions  # noqa: E402


ROBOT_NAME = "google_robot_static"
POLICY_NAME = "molmoact"
CONTROL_FREQ = 3
SIM_FREQ = 513


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect controlled MolmoAct rollouts in SimplerEnv."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Local MolmoAct checkpoint path.",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "pick_coke_can",
            "move_near",
            "open_drawer",
        ],
        help="Task family whose explicit conditions are defined in task_configs.py.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "vllm"],
        help="MolmoAct inference backend. Use hf first for debugging; vllm for large runs after validation.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this index in the task's condition list.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most N conditions after start-index. Omit to run all.",
    )

    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Run explicit condition indices, e.g. "
            "--indices 25 29 37 45 49. "
            "Useful for stratified pilot sampling."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Root directory for rollout outputs.",
    )

    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save every pre-action RGB frame plus the final RGB frame.",
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save an annotated rollout video for every completed episode.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip conditions that already contain summary.json; rerun incomplete conditions.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rerun an existing condition directory.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record an error and continue to the next condition instead of stopping.",
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help=(
            "Optional deterministic Gymnasium seed. "
            "If set, condition i uses base_seed + condition_id. "
            "Leave unset to follow the official-style explicit-condition reset."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected conditions without loading the model or simulator.",
    )

    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("--start-index must be >= 0")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")

    if args.indices is not None:
        if args.start_index != 0:
            parser.error("--indices cannot be combined with non-zero --start-index")
        if args.limit is not None:
            parser.error("--indices cannot be combined with --limit")
        for i in args.indices:
            if i < 0:
                parser.error(f"--indices contains negative value: {i}")

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")

    return args


# ---------------------------------------------------------------------
# JSON / hashing helpers
# ---------------------------------------------------------------------

def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            to_jsonable(obj),
            f,
            ensure_ascii=False,
            indent=2,
            cls=RobustJSONEncoder,
        )


def json_line(obj: Any) -> str:
    return json.dumps(
        to_jsonable(obj),
        ensure_ascii=False,
        separators=(",", ":"),
        cls=RobustJSONEncoder,
    )


def image_sha1(image: np.ndarray) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()


def scalar_float(x: Any) -> float:
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"Expected scalar reward, got shape={arr.shape}")
    return float(arr.reshape(-1)[0])


# ---------------------------------------------------------------------
# Condition handling
# ---------------------------------------------------------------------

def select_conditions(
    all_conditions: List[Dict[str, Any]],
    start_index: int,
    limit: Optional[int],
    indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    if indices is not None:
        selected = []

        seen = set()

        for idx in indices:
            if idx < 0 or idx >= len(all_conditions):
                raise ValueError(
                    f"condition index {idx} out of range "
                    f"[0, {len(all_conditions) - 1}]"
                )

            if idx in seen:
                continue

            seen.add(idx)
            selected.append(all_conditions[idx])

        return selected

    if start_index >= len(all_conditions):
        raise ValueError(
            f"start-index={start_index} is outside condition list "
            f"(num_conditions={len(all_conditions)})"
        )

    selected = all_conditions[start_index:]

    if limit is not None:
        selected = selected[:limit]

    return selected


def env_build_key(condition: Dict[str, Any]) -> str:
    """
    Conditions with the same key can reuse one already-created environment.

    Robot/object reset poses are deliberately excluded because they belong to
    reset(), not environment construction.
    """
    payload = {
        "env_name": condition["env_name"],
        "scene_name": condition["scene_name"],
        "max_steps": condition["max_steps"],
        "env_kwargs": condition.get("env_kwargs", {}),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def condition_dir_name(condition: Dict[str, Any]) -> str:
    return f"condition_{int(condition['condition_id']):04d}"


# ---------------------------------------------------------------------
# Environment construction/reset
# ---------------------------------------------------------------------

def build_env(condition: Dict[str, Any]):
    control_mode = get_robot_control_mode(
        ROBOT_NAME,
        POLICY_NAME,
    )

    kwargs: Dict[str, Any] = {
        "obs_mode": "rgbd",
        "robot": ROBOT_NAME,
        "sim_freq": SIM_FREQ,
        "control_mode": control_mode,
        "control_freq": CONTROL_FREQ,
        "max_episode_steps": int(condition["max_steps"]),
        "scene_name": condition["scene_name"],
        "camera_cfgs": {"add_segmentation": True},
    }

    # Build-level task variants, e.g.:
    # Coke: lr_switch/upright/laid_vertically
    # Drawer: shader_dir="rt"
    kwargs.update(condition.get("env_kwargs", {}))

    env = build_maniskill2_env(
        condition["env_name"],
        **kwargs,
    )
    return env


def build_robot_quaternion(condition: Dict[str, Any]) -> np.ndarray:
    """
    Reproduce SimplerEnv's quaternion composition style:
        Pose(q=euler2quat(r,p,y)) * Pose(q=center)

    Our base task configs currently use roll=pitch=0 and task-specific yaw.
    """
    if "robot_rot_quat_center" not in condition:
        raise KeyError(
            "Condition is missing 'robot_rot_quat_center'. "
            "Define it explicitly in src/task_configs.py."
        )

    center = np.asarray(
        condition["robot_rot_quat_center"],
        dtype=np.float64,
    )

    roll = float(condition.get("robot_roll", 0.0))
    pitch = float(condition.get("robot_pitch", 0.0))
    yaw = float(condition.get("robot_yaw", 0.0))

    delta_pose = Pose(q=euler2quat(roll, pitch, yaw))
    center_pose = Pose(q=center)

    return np.asarray((delta_pose * center_pose).q, dtype=np.float64)


def reset_env(
    env,
    condition: Dict[str, Any],
    seed: Optional[int],
):
    robot_quat = build_robot_quaternion(condition)

    reset_options: Dict[str, Any] = {
        "robot_init_options": {
            "init_xy": np.asarray(
                [
                    condition["robot_x"],
                    condition["robot_y"],
                ],
                dtype=np.float64,
            ),
            "init_rot_quat": robot_quat,
        }
    }

    obj_mode = condition["obj_mode"]

    if obj_mode == "xy":
        reset_options["obj_init_options"] = {
            "init_xy": np.asarray(
                [
                    condition["obj_x"],
                    condition["obj_y"],
                ],
                dtype=np.float64,
            )
        }

    elif obj_mode == "episode":
        reset_options["obj_init_options"] = {
            "episode_id": int(condition["obj_episode_id"])
        }

    else:
        raise ValueError(f"Unsupported obj_mode: {obj_mode}")

    if seed is None:
        obs, info = env.reset(options=reset_options)
    else:
        obs, info = env.reset(
            seed=int(seed),
            options=reset_options,
        )

    return obs, info, robot_quat


# ---------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------

def run_episode(
    env,
    model: InstrumentedMolmoAct,
    condition: Dict[str, Any],
    episode_dir: Path,
    seed: Optional[int],
    save_images: bool,
    save_video: bool,
) -> Dict[str, Any]:
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Keep the exact condition beside the raw step log.
    json_dump(episode_dir / "condition.json", condition)

    obs, reset_info, robot_quat = reset_env(
        env=env,
        condition=condition,
        seed=seed,
    )

    task_description = env.get_language_instruction()
    initial_task_description = task_description
    model.reset(task_description)

    image = get_image_from_maniskill2_obs_dict(env, obs)
    initial_rgb_sha1 = image_sha1(image)

    annotated_frames: List[np.ndarray] = []

    terminated = False
    truncated = False
    predicted_terminated = False

    timestep = 0
    max_steps = int(condition["max_steps"])

    final_info: Dict[str, Any] = {}
    final_reward: Optional[float] = None

    episode_start = time.perf_counter()

    steps_path = episode_dir / "steps.jsonl"

    # "w" guarantees a fresh trajectory when rerunning an incomplete condition.
    # flush after every line so a machine/SSH failure does not erase prior steps.
    with steps_path.open("w", encoding="utf-8", buffering=1) as step_file:

        while (
            not terminated
            and not truncated
            and not predicted_terminated
            and timestep < max_steps
        ):
            current_instruction = task_description

            rgb_hash = image_sha1(image)

            rgb_relpath: Optional[str] = None
            if save_images:
                rgb_name = f"rgb_{timestep:03d}.jpg"
                Image.fromarray(image).save(
                    episode_dir / rgb_name,
                    quality=90,
                )
                rgb_relpath = rgb_name

            # -------------------------------------------------------------
            # PRE-ACTION: information that is legally available to an
            # online failure predictor at time t.
            # -------------------------------------------------------------
            raw_action, action, annotated_image = model.step(
                image,
                current_instruction,
            )

            debug = dict(model.last_debug)

            env_action = np.concatenate(
                [
                    np.asarray(action["world_vector"]).reshape(-1),
                    np.asarray(action["rot_axangle"]).reshape(-1),
                    np.asarray(action["gripper"]).reshape(-1),
                ]
            ).astype(np.float32)

            if env_action.shape != (7,):
                raise ValueError(
                    f"Expected 7D env action, got shape={env_action.shape}: "
                    f"{env_action}"
                )

            predicted_terminated_this_step = bool(
                np.asarray(
                    action.get("terminate_episode", [0.0])
                ).reshape(-1)[0] > 0
            )

            # Official evaluator supports multi-subtask environments.
            # These three base tasks are normally single-subtask, but keep
            # compatible behavior for future extensions.
            if predicted_terminated_this_step:
                if (
                    hasattr(env, "is_final_subtask")
                    and not env.is_final_subtask()
                ):
                    predicted_terminated_this_step = False
                    if hasattr(env, "advance_to_next_subtask"):
                        env.advance_to_next_subtask()

            if save_video:
                annotated_frames.append(annotated_image)

            # -------------------------------------------------------------
            # ACTION EXECUTION
            # -------------------------------------------------------------
            next_obs, reward, terminated, truncated, info = env.step(
                env_action
            )

            reward_value = scalar_float(reward)
            final_reward = reward_value
            final_info = to_jsonable(info)

            # -------------------------------------------------------------
            # POST-ACTION: labels / environment feedback.
            # Do not use this block as an online predictor input.
            # -------------------------------------------------------------
            step_record = {
                "condition_id": int(condition["condition_id"]),
                "task": condition["task"],
                "step_id": timestep,
                "seed": seed,
                "instruction": current_instruction,

                "pre_action": {
                    "rgb_sha1": rgb_hash,
                    "rgb_path": rgb_relpath,

                    "generated_text": debug.get("generated_text"),
                    "depth": debug.get("depth"),
                    "trace": debug.get("trace"),
                    "parsed_action": debug.get("parsed_action"),

                    "raw_action": debug.get("raw_action"),
                    "processed_action": debug.get("processed_action"),
                    "env_action": env_action.tolist(),

                    "inference_time_s": debug.get("inference_time_s"),
                },

                "post_action": {
                    "reward": reward_value,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "predicted_terminated": bool(
                        predicted_terminated_this_step
                    ),
                    "info": final_info,
                },
            }

            step_file.write(json_line(step_record) + "\n")
            step_file.flush()

            # Next observation/instruction
            image = get_image_from_maniskill2_obs_dict(
                env,
                next_obs,
            )

            new_task_description = env.get_language_instruction()
            if new_task_description != task_description:
                task_description = new_task_description

            predicted_terminated = predicted_terminated_this_step
            timestep += 1

    elapsed_s = time.perf_counter() - episode_start

    final_rgb_sha1 = image_sha1(image)

    if save_images:
        Image.fromarray(image).save(
            episode_dir / "rgb_final.jpg",
            quality=90,
        )

    # SimplerEnv evaluator convention:
    # done/terminated -> success
    # timeout/truncation without terminated -> failure
    success = bool(terminated)

    if success:
        stop_reason = "success"
    elif truncated:
        stop_reason = "truncated"
    elif predicted_terminated:
        stop_reason = "policy_terminated"
    elif timestep >= max_steps:
        stop_reason = "max_steps"
    else:
        stop_reason = "unknown"

    summary = {
        "condition_id": int(condition["condition_id"]),
        "task": condition["task"],
        "condition": condition,

        "seed": seed,

        "env_name": condition["env_name"],
        "scene_name": condition["scene_name"],

        "robot_init_quat": robot_quat.tolist(),

        "instruction_initial": initial_task_description,
        "instruction_final": task_description,

        "num_steps": timestep,
        "max_steps": max_steps,

        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "predicted_terminated": bool(predicted_terminated),
        "stop_reason": stop_reason,

        "initial_rgb_sha1": initial_rgb_sha1,
        "final_rgb_sha1": final_rgb_sha1,

        "final_reward": final_reward,
        "final_info": final_info,
        "reset_info": to_jsonable(reset_info),

        "elapsed_s": elapsed_s,
    }

    json_dump(
        episode_dir / "summary.json",
        summary,
    )

    if save_video and annotated_frames:
        from simpler_env.utils.visualization import write_video

        write_video(
            str(episode_dir / "rollout.mp4"),
            annotated_frames,
            fps=5,
        )

    return summary


# ---------------------------------------------------------------------
# Aggregate outputs
# ---------------------------------------------------------------------

def rebuild_task_summary(task_dir: Path) -> Tuple[int, int, int]:
    summaries: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for condition_dir in sorted(task_dir.glob("condition_*")):
        summary_path = condition_dir / "summary.json"
        error_path = condition_dir / "error.json"

        loaded = False
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summaries.append(json.load(f))
                loaded = True
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"    WARN: skipping corrupted {summary_path} "
                    f"({type(exc).__name__}: {exc})"
                )

        if not loaded and error_path.exists():
            try:
                with error_path.open("r", encoding="utf-8") as f:
                    errors.append(json.load(f))
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"    WARN: skipping corrupted {error_path} "
                    f"({type(exc).__name__}: {exc})"
                )

    summaries.sort(key=lambda x: int(x.get("condition_id", -1)))
    errors.sort(key=lambda x: int(x.get("condition_id", -1)))

    with (task_dir / "summary.jsonl").open("w", encoding="utf-8") as f:
        for item in summaries:
            f.write(json_line(item) + "\n")

    with (task_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
        for item in errors:
            f.write(json_line(item) + "\n")

    num_success = sum(bool(x.get("success", False)) for x in summaries)
    num_failure = len(summaries) - num_success
    num_error = len(errors)

    aggregate = {
        "num_completed": len(summaries),
        "num_success": num_success,
        "num_failure": num_failure,
        "num_error": num_error,
        "success_rate": (
            num_success / len(summaries)
            if summaries
            else None
        ),
    }
    json_dump(task_dir / "aggregate.json", aggregate)

    return num_success, num_failure, num_error



# ---------------------------------------------------------------------
# SimplerEnv working-directory compatibility
# ---------------------------------------------------------------------

def ensure_simpler_env_working_directory() -> Optional[Path]:
    """
    The current official MolmoAct wrapper reads dataset_statistics.json from
    a relative path:
        ./simpler_env/policies/molmoact/dataset_statistics.json

    When this research repository is a sibling of SimplerEnv, running the
    script from the research repo would otherwise raise FileNotFoundError.

    We therefore:
    1) keep cwd if the relative file already exists;
    2) otherwise try PROJECT_ROOT.parent / "SimplerEnv";
    3) fail with a clear message if neither layout is valid.

    Output/checkpoint paths are resolved before calling this function.
    """
    rel_stats = Path(
        "simpler_env/policies/molmoact/dataset_statistics.json"
    )

    if rel_stats.exists():
        return Path.cwd()

    sibling = PROJECT_ROOT.parent / "SimplerEnv"
    sibling_stats = sibling / rel_stats

    if sibling_stats.exists():
        os.chdir(sibling)
        return sibling

    raise FileNotFoundError(
        "Cannot locate SimplerEnv's dataset_statistics.json. "
        "Expected either ./simpler_env/policies/molmoact/"
        "dataset_statistics.json from the current directory, or a sibling "
        f"SimplerEnv repository at: {sibling}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    all_conditions = get_conditions(args.task)
    selected_conditions = select_conditions(
        all_conditions=all_conditions,
        start_index=args.start_index,
        limit=args.limit,
        indices=args.indices,
    )

    selected_ids = [
        int(c["condition_id"])
        for c in selected_conditions
    ]

    print(
        f"Task: {args.task}\n"
        f"Total available conditions: {len(all_conditions)}\n"
        f"Selected conditions: {len(selected_conditions)}\n"
        f"Selected condition ids: {selected_ids}"
    )

    if args.dry_run:
        for i, condition in enumerate(selected_conditions):
            print(
                f"[{i:03d}] "
                f"condition_id={condition['condition_id']} "
                f"env={condition['env_name']} "
                f"scene={condition['scene_name']} "
                f"obj_mode={condition['obj_mode']} "
                f"max_steps={condition['max_steps']} "
                f"env_kwargs={condition.get('env_kwargs', {})}"
            )
        return

    output_root = Path(args.output_dir).expanduser().resolve()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if checkpoint_path.exists():
        checkpoint_for_model = str(checkpoint_path.resolve())
    else:
        checkpoint_for_model = args.checkpoint

    task_dir = output_root / args.task
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save selected conditions and run settings before expensive model loading.
    json_dump(
        task_dir / "run_config.json",
        {
            "checkpoint": checkpoint_for_model,
            "task": args.task,
            "backend": args.backend,
            "start_index": args.start_index,
            "limit": args.limit,
            "indices": args.indices,
            "save_images": args.save_images,
            "save_video": args.save_video,
            "resume": args.resume,
            "overwrite": args.overwrite,
            "continue_on_error": args.continue_on_error,
            "base_seed": args.base_seed,
            "robot": ROBOT_NAME,
            "control_freq": CONTROL_FREQ,
            "sim_freq": SIM_FREQ,
        },
    )
    json_dump(
        task_dir / "selected_conditions.json",
        selected_conditions,
    )

    simpler_env_cwd = ensure_simpler_env_working_directory()
    print(f"Using SimplerEnv working directory: {simpler_env_cwd}")

    pending: List[Tuple[int, Dict[str, Any], Path]] = []
    for _idx, condition in enumerate(selected_conditions, start=1):
        _cid = int(condition["condition_id"])
        _episode_dir = task_dir / condition_dir_name(condition)
        _summary_path = _episode_dir / "summary.json"

        if _summary_path.exists() and args.resume and not args.overwrite:
            print(
                f"[pre-check] condition={_cid}: already complete, "
                f"will skip (--resume)"
            )
            continue

        if _summary_path.exists() and not args.overwrite and not args.resume:
            raise FileExistsError(
                f"{_summary_path} already exists. "
                "Use --resume to skip it or --overwrite to rerun it."
            )

        if _episode_dir.exists() and not _summary_path.exists():
            if not (args.resume or args.overwrite):
                raise FileExistsError(
                    f"Incomplete directory exists: {_episode_dir}. "
                    "Use --resume or --overwrite."
                )

        pending.append((_idx, condition, _episode_dir))

    if not pending:
        print(
            "\nAll conditions are already complete under --resume. "
            "Skipping model load and environment build."
        )
        success_n, failure_n, error_n = rebuild_task_summary(task_dir)
        completed_n = success_n + failure_n
        success_rate = (
            success_n / completed_n if completed_n > 0 else float("nan")
        )
        print("\n===== RUN SUMMARY =====")
        print(f"completed: {completed_n}")
        print(f"success:   {success_n}")
        print(f"failure:   {failure_n}")
        print(f"errors:    {error_n}")
        if completed_n > 0:
            print(f"success rate: {success_rate:.3f}")
        print(f"results: {task_dir}")
        return

    print(
        f"\n{len(pending)} condition(s) pending out of "
        f"{len(selected_conditions)}; loading model..."
    )

    print("\nLoading MolmoAct once...")
    model = InstrumentedMolmoAct(
        saved_model_path=checkpoint_for_model,
        policy_setup="google_robot",
        backend=args.backend,
        quiet=True,
    )

    current_env = None
    current_env_key: Optional[str] = None

    try:
        for local_idx, condition, episode_dir in pending:
            cid = int(condition["condition_id"])
            summary_path = episode_dir / "summary.json"

            if summary_path.exists():
                if args.resume:
                    print(
                        f"[{local_idx}/{len(selected_conditions)}] "
                        f"condition={cid}: already complete, skip"
                    )
                    continue

                if args.overwrite:
                    shutil.rmtree(episode_dir)
                else:
                    raise FileExistsError(
                        f"{summary_path} already exists. "
                        "Use --resume to skip it or --overwrite to rerun it."
                    )

            elif episode_dir.exists():
                if args.resume or args.overwrite:
                    shutil.rmtree(episode_dir)
                else:
                    raise FileExistsError(
                        f"Incomplete directory exists: {episode_dir}. "
                        "Use --resume or --overwrite."
                    )

            # Rebuild environment only when build-level configuration changes.
            wanted_key = env_build_key(condition)
            if current_env is None or wanted_key != current_env_key:
                if current_env is not None:
                    current_env.close()

                print(
                    "\nBuilding environment:\n"
                    f"  env_name={condition['env_name']}\n"
                    f"  scene_name={condition['scene_name']}\n"
                    f"  max_steps={condition['max_steps']}\n"
                    f"  env_kwargs={condition.get('env_kwargs', {})}"
                )

                current_env = build_env(condition)
                current_env_key = wanted_key

            seed = (
                None
                if args.base_seed is None
                else int(args.base_seed) + cid
            )

            print(
                f"[{local_idx}/{len(selected_conditions)}] "
                f"running condition={cid} ..."
            )

            try:
                summary = run_episode(
                    env=current_env,
                    model=model,
                    condition=condition,
                    episode_dir=episode_dir,
                    seed=seed,
                    save_images=args.save_images,
                    save_video=args.save_video,
                )

                print(
                    f"    success={summary['success']} "
                    f"steps={summary['num_steps']} "
                    f"stop={summary['stop_reason']} "
                    f"time={summary['elapsed_s']:.1f}s"
                )

            except Exception as exc:
                episode_dir.mkdir(parents=True, exist_ok=True)

                error_record = {
                    "condition_id": cid,
                    "task": args.task,
                    "condition": condition,
                    "seed": seed,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                }
                json_dump(
                    episode_dir / "error.json",
                    error_record,
                )

                print(
                    f"    ERROR: {type(exc).__name__}: {exc}"
                )

                if not args.continue_on_error:
                    raise

    finally:
        if current_env is not None:
            current_env.close()

        success_n, failure_n, error_n = rebuild_task_summary(
            task_dir
        )

        completed_n = success_n + failure_n
        success_rate = (
            success_n / completed_n
            if completed_n > 0
            else float("nan")
        )

        print("\n===== RUN SUMMARY =====")
        print(f"completed: {completed_n}")
        print(f"success:   {success_n}")
        print(f"failure:   {failure_n}")
        print(f"errors:    {error_n}")
        if completed_n > 0:
            print(f"success rate: {success_rate:.3f}")
        print(f"results: {task_dir}")


if __name__ == "__main__":
    main()
