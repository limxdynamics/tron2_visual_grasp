"""Move the arm back to a resting pose, then exit.

Usage:
    ./run.sh home
    ./run.sh home --pose ready
    ./run.sh home --open-gripper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.robot import make_robot  # noqa: E402
from utils.camera_utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send the arm to a resting pose and exit")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--pose",
        choices=("park", "ready", "watch"),
        default="park",
        help="park = folded stow pose (default); ready = look-down observe pose; "
             "watch = mission.watch_joints",
    )
    parser.add_argument("--open-gripper", action="store_true", help="Open the gripper first")
    return parser.parse_args()


def _move(robot: Any, cfg: dict[str, Any], pose: str) -> bool:
    if pose == "park":
        return robot.move_park()
    if pose == "ready":
        return robot.move_ready((cfg.get("robot") or {}).get("ready_pose") or {})
    joints = (cfg.get("mission") or {}).get("watch_joints")
    if joints is None:
        raise ValueError("mission.watch_joints is required for --pose watch")
    return robot.move_joints(np.asarray(joints, dtype=float).reshape(6))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)

    robot = make_robot(cfg)
    robot.start()
    try:
        if args.open_gripper:
            print("[Home] Open gripper")
            robot.open_gripper()
        print(f"[Home] Move the arm to the {args.pose} pose")
        if not _move(robot, cfg, args.pose):
            print(f"[Home] Unable to reach the {args.pose} pose")
            return 1
        print(f"[Home] Arm at the {args.pose} pose")
        return 0
    finally:
        # park=False: the arm is already where the caller asked for.
        robot.end(park=False)


if __name__ == "__main__":
    raise SystemExit(main())
