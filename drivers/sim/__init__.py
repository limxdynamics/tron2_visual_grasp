"""Hardware-free demo: a simulated TRON2 and wrist D405 sharing one rendered world.

`./run.sh demo` runs scripts/squat_grasp.py with --sim, which passes the config
through sim_config(): the camera and robot types switch to the simulated drivers
below (picked by make_camera / make_robot like any other hardware) and the
posture waits are shortened. Perception, grasp solving and the mission logic are
the same code that runs on the robot.
"""

from __future__ import annotations

import copy
from typing import Any

from .camera import SimCamera
from .robot import SimTron2Driver
from .world import HAND_EYE, SimWorld, get_world, reset_world

__all__ = [
    "CAMERA_TYPE", "ROBOT_TYPE", "HAND_EYE", "SimCamera", "SimTron2Driver", "SimWorld",
    "get_world", "reset_world", "sim_config",
]

CAMERA_TYPE = "sim_d405"
ROBOT_TYPE = "tron2_sim"


def sim_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy of cfg that selects the simulated camera and robot and applies sim.overrides."""
    out = copy.deepcopy(cfg)
    out.setdefault("camera", {})["type"] = CAMERA_TYPE
    out.setdefault("robot", {})["type"] = ROBOT_TYPE
    mission = out.setdefault("mission", {})
    mission["watch_camera"] = {"type": CAMERA_TYPE}

    def merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)

    merge(out, ((out.get("sim") or {}).get("overrides")) or {})
    return out
