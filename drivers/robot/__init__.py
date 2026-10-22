from __future__ import annotations

from .base import RobotDriver
from .piperx_driver import PiperDriver
from .tron2_driver import Tron2Driver

__all__ = [
    "RobotDriver",
    "PiperDriver",
    "Tron2Driver",
    "make_robot",
]


def make_robot(cfg: dict) -> RobotDriver:
    """Create a robot driver from config/default.yaml."""
    robot_cfg = cfg.get("robot", {})
    robot_type = str(robot_cfg.get("type", "")).lower()

    if robot_type == "tron2_sim":
        from drivers.sim import SimTron2Driver, get_world
        return SimTron2Driver(robot_cfg, get_world(cfg), cfg.get("sim"))
    if "tron2" in robot_type:
        return Tron2Driver(robot_cfg)
    if "piper" in robot_type:
        return PiperDriver(robot_cfg)

    raise ValueError(
        f"Unsupported robot type: {robot_type!r}\n"
        f"Set robot.type in config/default.yaml to:\n"
        f"  tron2 | piperx"
    )
