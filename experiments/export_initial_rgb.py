import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 复用我们正式 rollout 里的环境构建/reset 逻辑
from experiments.run_rollouts import build_env, reset_env
from src.task_configs import get_conditions

from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict,
)


TASK = "pick_coke_can"
CONDITION_ID = 1

OUT = Path(
    "/root/autodl-tmp/molmoact_project/results/"
    "backend_test/static_inputs/coke_condition_0001_rgb0.png"
)


def main():
    conditions = get_conditions(TASK)
    condition = conditions[CONDITION_ID]

    print("condition:")
    print(condition)

    print("\nBuilding environment...")
    env = build_env(condition)

    try:
        # 和你之前正式 rollout 一样，不额外设 seed
        obs, info, robot_quat = reset_env(
            env=env,
            condition=condition,
            seed=None,
        )

        image = get_image_from_maniskill2_obs_dict(
            env,
            obs,
        )

        OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.fromarray(image).save(OUT)

        print("\nimage shape:", image.shape)
        print("saved:", OUT)

        import hashlib
        sha1 = hashlib.sha1(
            image.tobytes()
        ).hexdigest()

        print("rgb sha1:", sha1)

    finally:
        env.close()


if __name__ == "__main__":
    main()