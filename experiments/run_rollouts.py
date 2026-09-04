import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import simpler_env
from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict
)
from simpler_env.utils.visualization import write_video


# 让 Python 能找到项目 src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.instrumented_molmoact import (
    InstrumentedMolmoAct,
    to_jsonable,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument(
        "--env-name",
        required=True,
        choices=[
            "google_robot_pick_coke_can",
            "google_robot_open_drawer",
            "google_robot_move_near",
        ],
    )

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)

    parser.add_argument(
        "--backend",
        choices=["hf", "vllm"],
        default="hf",
    )

    parser.add_argument("--start-seed", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        default="results",
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
    )

    parser.add_argument(
        "--save-images",
        action="store_true",
    )

    return parser.parse_args()


def append_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                to_jsonable(data),
                ensure_ascii=False,
            )
            + "\n"
        )

        # 每个 step 立即写磁盘。
        # 实验中途断电也不会丢整局数据。
        f.flush()


def run_episode(
    env,
    model,
    episode_id,
    seed,
    env_name,
    max_steps,
    task_dir,
    save_video=False,
    save_images=False,
):
    episode_dir = task_dir / f"episode_{episode_id:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # -------- reset --------

    obs, reset_info = env.reset(seed=seed)

    instruction = env.get_language_instruction()

    image = get_image_from_maniskill2_obs_dict(
        env,
        obs,
    )

    initial_image_hash = hashlib.sha1(
        image.tobytes()
    ).hexdigest()

    model.reset(instruction)

    annotated_frames = []

    terminated = False
    truncated = False

    step_id = 0

    step_log_path = episode_dir / "steps.jsonl"

    # -------- rollout --------

    while (
        not terminated
        and not truncated
        and step_id < max_steps
    ):
        # ==========================================
        # t 时刻：所有 failure predictor 可用信息
        # ==========================================

        current_instruction = instruction

        if save_images:
            Image.fromarray(image).save(
                episode_dir / f"rgb_{step_id:03d}.jpg",
                quality=90,
            )

        raw_action, action, annotated_image = model.step(
            image,
            current_instruction,
        )

        debug = dict(model.last_debug)

        env_action = np.concatenate(
            [
                action["world_vector"],
                action["rot_axangle"],
                action["gripper"],
            ]
        ).astype(np.float32)

        if save_video:
            annotated_frames.append(annotated_image)

        # ==========================================
        # 执行动作
        # ==========================================

        next_obs, reward, terminated, truncated, info = env.step(
            env_action
        )

        # ==========================================
        # 保存该 step
        #
        # 注意：
        # pre_action 以后允许进入 failure predictor
        # post_action 只能作为标签 / 事后分析
        # ==========================================

        step_record = {
            "episode_id": episode_id,
            "step_id": step_id,
            "seed": seed,
            "env_name": env_name,
            "instruction": current_instruction,

            "pre_action": {
                "generated_text": debug.get("generated_text"),
                "depth": debug.get("depth"),
                "trace": debug.get("trace"),
                "parsed_action": debug.get("parsed_action"),

                "raw_action": debug.get("raw_action"),
                "processed_action": debug.get(
                    "processed_action"
                ),

                "env_action": env_action.tolist(),

                "inference_time_s": debug.get(
                    "inference_time_s"
                ),
            },

            "post_action": {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": to_jsonable(info),
            },
        }

        append_jsonl(
            step_log_path,
            step_record,
        )

        # -------- next timestep --------

        image = get_image_from_maniskill2_obs_dict(
            env,
            next_obs,
        )

        new_instruction = env.get_language_instruction()

        if new_instruction != instruction:
            instruction = new_instruction

        step_id += 1

    # ==========================================
    # episode label
    # ==========================================

    success = bool(terminated)

    summary = {
        "episode_id": episode_id,
        "env_name": env_name,
        "seed": seed,
        "num_steps": step_id,
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "initial_image_sha1": initial_image_hash,
        "reset_info": to_jsonable(reset_info),
    }

    with open(
        episode_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    if save_video and len(annotated_frames) > 0:
        write_video(
            str(episode_dir / "rollout.mp4"),
            annotated_frames,
            fps=5,
        )

    append_jsonl(
        task_dir / "summary.jsonl",
        summary,
    )

    return summary


def main():
    args = parse_args()

    output_root = Path(args.output_dir)

    task_dir = output_root / args.env_name
    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading MolmoAct...")

    model = InstrumentedMolmoAct(
        saved_model_path=args.checkpoint,
        policy_setup="google_robot",
        backend=args.backend,
        quiet=True,
    )

    print("Creating SimplerEnv...")

    # 一个 task 只创建一次 env。
    # 不要每个 episode 重建 Vulkan renderer。
    env = simpler_env.make(
        args.env_name,
        max_episode_steps=args.max_steps,
    )

    try:
        for i in range(args.episodes):

            seed = args.start_seed + i

            summary = run_episode(
                env=env,
                model=model,
                episode_id=i,
                seed=seed,
                env_name=args.env_name,
                max_steps=args.max_steps,
                task_dir=task_dir,
                save_video=args.save_video,
                save_images=args.save_images,
            )

            print(
                f"[{i + 1}/{args.episodes}] "
                f"success={summary['success']} "
                f"steps={summary['num_steps']} "
                f"seed={seed}"
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()