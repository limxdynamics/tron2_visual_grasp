"""Hardware-free demo (./run.sh demo): simulated camera and TRON2, and the full mission on them."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import yaml


@pytest.fixture
def sim(cfg):
    from drivers.camera import make_camera
    from drivers.robot import make_robot
    from drivers.sim import reset_world, sim_config

    reset_world()
    scfg = sim_config(cfg)
    scfg["sim"]["seed"] = 7
    camera, robot = make_camera(scfg), make_robot(scfg)
    camera.open()
    robot.start()
    yield scfg, camera, robot
    robot.end(park=False)
    camera.close()
    reset_world()


def test_sim_config_only_swaps_hardware(cfg):
    from drivers.sim import sim_config
    scfg = sim_config(cfg)
    assert scfg["camera"]["type"] == "sim_d405" and scfg["robot"]["type"] == "tron2_sim"
    assert scfg["mission"]["approach"] == cfg["mission"]["approach"]
    assert scfg["grasp_pipeline"] == cfg["grasp_pipeline"]
    assert cfg["robot"]["type"] == "tron2", "sim_config must not modify the loaded config"


def test_sim_hand_eye_file_matches_mount(project_root, cfg):
    from drivers.sim import HAND_EYE
    from utils.camera_utils import load_hand_eye
    T, mode = load_hand_eye(project_root, "sim_d405", cfg)
    assert mode == "eye_in_hand" and np.allclose(T, HAND_EYE)


def test_watch_camera_sees_the_ball(sim):
    from utils.detection import detect_objects, load_detector
    scfg, camera, robot = sim
    assert robot.move_joints(scfg["mission"]["watch_joints"])
    bgr, depth = camera.get_frame()
    assert bgr.shape == (720, 1280, 3) and depth.dtype == np.uint16
    model, opts = load_detector(scfg)
    _, dets = detect_objects(model, bgr, opts)
    assert len(dets) == 1


def test_robot_contract(sim):
    _, _, robot = sim
    robot.set_base_twist(5.0, 0.0, 0.0)                     # clipped to max_twist_mps
    assert robot._world.twist[0] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        robot.move_to_pose(float("nan"), 0, 0, 0, 0, 0)
    assert not robot.move_to_pose(2.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # out of reach
    T = robot.get_tcp_pose()
    assert robot.move_to_pose(T[0, 3] + 0.05, T[1, 3], T[2, 3], 0.0, 0.0, 0.0, duration=0.2)
    assert np.allclose(robot.get_tcp_pose()[:3, 3], T[:3, 3] + [0.05, 0, 0], atol=2e-3)
    assert not robot.grasp()                                # nothing between the jaws
    assert robot.move_park()


def test_full_mission_picks_two_balls(project_root, tmp_path):
    """The real squat_grasp.py mission, end to end on the simulator."""
    cfg = yaml.safe_load((project_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    cfg["sim"]["seed"] = 1
    config = tmp_path / "demo.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(project_root)}
    out = subprocess.run(
        [sys.executable, "scripts/squat_grasp.py", "--sim", "--cycles", "2", "--web-port", "0",
         "--config", str(config)],
        cwd=project_root, capture_output=True, text=True, timeout=180, env=env,
    )
    log = out.stdout + out.stderr
    assert out.returncode == 0, log[-3000:]
    assert log.count("[Mission] Grasp succeeded") == 2, log[-3000:]
    assert "[Approach] Arrived" in log
