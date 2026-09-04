# src/task_configs.py

import itertools
import numpy as np


def linspace(start, end, n):
    return [float(x) for x in np.linspace(start, end, n)]


# ============================================================
# Pick Coke Can
# ============================================================

def build_coke_conditions():
    """
    Official-style base conditions:
        object x: 5
        object y: 5
        orientation: 3

    total = 5 * 5 * 3 = 75
    """

    xs = linspace(-0.35, -0.12, 5)
    ys = linspace(-0.02, 0.42, 5)

    orientations = [
        {
            "name": "horizontal",
            "env_kwargs": {"lr_switch": True},
        },
        {
            "name": "upright",
            "env_kwargs": {"upright": True},
        },
        {
            "name": "vertical",
            "env_kwargs": {"laid_vertically": True},
        },
    ]

    conditions = []

    condition_id = 0

    for ori in orientations:
        for x, y in itertools.product(xs, ys):

            conditions.append(
                {
                    "condition_id": condition_id,
                    "task": "pick_coke_can",

                    "env_name": "GraspSingleOpenedCokeCanInScene-v0",
                    "scene_name": "google_pick_coke_can_1_v4",

                    "robot_x": 0.35,
                    "robot_y": 0.20,

                    # 跟官方脚本保持一致
                    "robot_rot_quat_center": [0, 0, 0, 1],
                    "robot_yaw": 0.0,

                    "obj_mode": "xy",
                    "obj_x": x,
                    "obj_y": y,

                    "orientation": ori["name"],

                    "env_kwargs": dict(ori["env_kwargs"]),

                    "max_steps": 80,
                    "raytracing": False,
                }
            )

            condition_id += 1

    return conditions


# ============================================================
# Move Near
# ============================================================

def build_move_near_conditions():
    """
    Official base evaluation uses 60 predefined object episodes.
    """

    conditions = []

    for episode_id in range(60):

        conditions.append(
            {
                "condition_id": episode_id,
                "task": "move_near",

                "env_name": "MoveNearGoogleInScene-v0",
                "scene_name": "google_pick_coke_can_1_v4",

                "robot_x": 0.35,
                "robot_y": 0.21,

                "robot_rot_quat_center": [0, 0, 0, 1],
                "robot_yaw": -0.09,

                "obj_mode": "episode",
                "obj_episode_id": episode_id,

                "env_kwargs": {},

                "max_steps": 80,
                "raytracing": False,
            }
        )

    return conditions


# ============================================================
# Drawer
# ============================================================

def build_drawer_conditions():
    """
    Base OPEN drawer conditions.

    3 drawer levels
    × 3 robot x positions
    × 3 robot y positions
    = 27 conditions
    """

    envs = [
        ("top", "OpenTopDrawerCustomInScene-v0"),
        ("middle", "OpenMiddleDrawerCustomInScene-v0"),
        ("bottom", "OpenBottomDrawerCustomInScene-v0"),
    ]

    robot_xs = linspace(0.65, 0.85, 3)
    robot_ys = linspace(-0.2, 0.2, 3)

    conditions = []

    condition_id = 0

    for (drawer_level, env_name), x, y in itertools.product(
        envs,
        robot_xs,
        robot_ys,
    ):

        conditions.append(
            {
                "condition_id": condition_id,
                "task": "open_drawer",

                "env_name": env_name,
                "scene_name": "frl_apartment_stage_simple",

                "robot_x": x,
                "robot_y": y,

                "robot_rot_quat_center": [0, 0, 0, 1],
                "robot_yaw": 0.0,

                "drawer_level": drawer_level,

                "obj_mode": "xy",
                "obj_x": 0.0,
                "obj_y": 0.0,

                "env_kwargs": {
                    "shader_dir": "rt",
                },

                "max_steps": 113,
                "raytracing": True,
            }
        )

        condition_id += 1

    return conditions


def get_conditions(task):
    if task == "pick_coke_can":
        return build_coke_conditions()

    if task == "move_near":
        return build_move_near_conditions()

    if task == "open_drawer":
        return build_drawer_conditions()

    raise ValueError(f"Unknown task: {task}")