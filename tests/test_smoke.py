"""Smoke tests: the project installs, imports, loads its config and its entry points start."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

MODULES = [
    "calibration.aruco_pose", "calibration.hand_eye",
    "drivers.camera", "drivers.camera.realsense",
    "drivers.robot", "drivers.robot.tron2_driver", "drivers.robot.piperx_driver", "drivers.sim",
    "utils.camera_utils", "utils.common_utils", "utils.detection", "utils.opencv_tennis",
    "utils.ordinary_grasp", "utils.targets", "utils.transforms", "utils.vision_tuning",
    "utils.visual_servo", "utils.web_debug",
]

# Commands whose --help must print usage without touching hardware.
HELP_COMMANDS = ["grasp", "dry-run", "demo","calib", "verify", "home", "pose"]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_config_loads(cfg):
    assert cfg["robot"]["type"] in ("tron2", "piperx")
    assert cfg["robot"]["tron2"]["accid"] is None, "ship accid as null so it is learned from the robot"
    assert cfg["detection"]["tennis"]["hsv_lower"] and cfg["detection"]["tennis"]["hsv_upper"]


def test_run_sh_help(project_root):
    out = subprocess.run(["bash", str(project_root / "run.sh"), "--help"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0 and "Usage: ./run.sh" in out.stdout


@pytest.mark.parametrize("command", HELP_COMMANDS)
def test_run_sh_dispatches_help(project_root, command):
    env = {**os.environ, "TVG_PYTHON": sys.executable}
    out = subprocess.run(["bash", str(project_root / "run.sh"), command, "--help"],
                         capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout.lower()


def test_run_sh_refuses_missing_interpreter(project_root):
    env = {**os.environ, "TVG_PYTHON": "/nonexistent/python"}
    out = subprocess.run(["bash", str(project_root / "run.sh"), "pose"],
                         capture_output=True, text=True, timeout=30, env=env)
    assert out.returncode != 0 and "TVG_PYTHON" in out.stderr
