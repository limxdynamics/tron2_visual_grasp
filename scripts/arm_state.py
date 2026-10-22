"""Print the TRON2 arm state without commanding any motion.

Usage:
    ./run.sh pose
    ./run.sh pose --watch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drivers.robot.tron2_driver import Tron2Driver  # noqa: E402
from utils.camera_utils import load_config  # noqa: E402
from utils.transforms import rotation_matrix_to_euler_zyx  # noqa: E402


class ReadOnlyTron2Driver(Tron2Driver):
    """Skip the servop stream so connecting never commands a target."""

    def _spawn(self, target, name):
        if name == "servop":
            return
        super()._spawn(target, name)


def _print_state(robot: ReadOnlyTron2Driver) -> None:
    joints = robot._wait_joints()
    pose = robot.get_tcp_pose()
    position = pose[:3, 3]
    rpy = rotation_matrix_to_euler_zyx(pose[:3, :3])

    if joints is not None:
        print("joints rad: [" + ", ".join(f"{v:+.4f}" for v in joints) + "]")
        print("joints deg: [" + ", ".join(f"{np.degrees(v):+7.2f}" for v in joints) + "]")
    print(f"tcp xyz  m: [{position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f}]")
    print(f"tcp rpy rad: [{rpy[0]:+.4f}, {rpy[1]:+.4f}, {rpy[2]:+.4f}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read the TRON2 arm state, read-only")
    parser.add_argument("--watch", action="store_true", help="keep printing until Ctrl+C")
    parser.add_argument("--interval", type=float, default=0.5, metavar="SECONDS")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config" / "default.yaml")
    robot = ReadOnlyTron2Driver(cfg.get("robot") or {})
    robot.start()
    try:
        if not args.watch:
            _print_state(robot)
            return 0
        while True:
            _print_state(robot)
            print("-" * 46)
            time.sleep(max(0.05, args.interval))
    finally:
        robot.end(park=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
