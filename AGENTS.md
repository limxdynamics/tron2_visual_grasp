# AGENTS.md

Visual grasping stack for mobile manipulators: an AgileX **Piper/PiperX** arm and a
**TRON2** whole-body robot, driven by a wrist RealSense D405 RGB-D camera. The pipeline detects tennis balls (OpenCV HSV + contour), estimates grasp poses, transforms
them to the robot base frame via hand-eye calibration, and drives the arm. The TRON2
mission ([scripts/squat_grasp.py](scripts/squat_grasp.py)) watches for a target,
visual-servos the base up to it (`mission.approach`), squats, grasps, and stands up.

> The project/repo is **`tron2_visual_grasp`**; the checkout folder name may differ.

Read [README.md](README.md) (or [README_zh.md](README_zh.md)) first — it is the
authoritative guide for setup, hardware wiring, per-script usage, and the full
`config/default.yaml` reference. Do not duplicate it here; link to it.

## Architecture

Hardware differences are isolated behind driver layers so task, perception, and
coordinate code stay hardware-independent. **Never branch on robot/camera model
inside scripts** — add new hardware behind these interfaces instead:

- `make_robot(cfg)` in [drivers/robot/__init__.py](drivers/robot/__init__.py)
  selects a `RobotDriver` by `robot.type` (`tron2` | `piperx`). Contract:
  [drivers/robot/base.py](drivers/robot/base.py).
- `make_camera(cfg)` selects a `CameraDriver` by `camera.type`. Contract:
  [drivers/camera/base.py](drivers/camera/base.py).
- Perception/geometry live in [utils/](utils) (`opencv_tennis`, `detection`, `ordinary_grasp`,
  `targets`, `transforms`, `visual_servo`).
  Runnable entry points live in [scripts/](scripts).

## Running

Always launch through [run.sh](run.sh), **not** raw `python scripts/...`. It
activates the conda env before dispatching. See
`./run.sh --help` for all commands (`grasp`, `dry-run`, `calib`, `verify`,
`detect`, `home`, `pose`).

`--dry-run` is perception-only and never touches actuators — use it to validate
perception changes safely (`./run.sh dry-run`).

`./run.sh demo` (`squat_grasp.py --sim`) runs the whole mission with no hardware:
`drivers/sim` provides a kinematic TRON2 (`robot.type: tron2_sim`) and a rendered
wrist camera (`camera.type: sim_d405`) sharing one `SimWorld`, selected through the
normal factories. `drivers/sim/view3d.py` adds an optional third-person MuJoCo view of
the WFYG_TRON2A model (fetched by `drivers/sim/model.py` at a pinned commit, never
vendored here). Use it to check mission-flow changes end to end before a robot run;
it is not a substitute for one. Keep it working when changing the `RobotDriver` or
`CameraDriver` contract.

## Testing

Hardware-free tests live in [tests/](tests/) and run in CI
([.github/workflows/ci.yml](.github/workflows/ci.yml)) on every push and pull request:

```bash
pip install -r tests/requirements.txt   # inside the tron2_visual_grasp env
python -m pytest tests
```

- `test_smoke.py`: imports, config, `run.sh` dispatch and `--help`.
- `test_perception.py`: detector, grasp geometry, hand-eye and ArUco on synthetic images.
- `test_tron2_driver.py`: the TRON2 driver against `MockTron2` (conftest.py), a mock of
  the upper-layer WebSocket protocol, including malformed replies, NaN inputs, stale
  feedback, link loss and the reconnect after a WF stand.
- `test_web_debug.py`: the web console endpoints.
- `test_sim_demo.py`: the simulated drivers, and the full mission picking two balls in the simulator.
- `test_sim_view3d.py`: the 3D view poses the model consistently with the simulator
  (skipped without MuJoCo or the model; CI renders under xvfb).

When changing the driver or the protocol, extend `MockTron2` and add a test. The tests
do not replace a run on the real robot: still check `./run.sh dry-run` and
`./run.sh detect` on hardware before merging motion changes.

## Configuration

- [config/default.yaml](config/default.yaml) is the single source of truth, loaded by
  `load_config` in [utils/camera_utils.py](utils/camera_utils.py). Add config keys
  instead of hardcoding values in scripts.
- The extensive inline comments are **load-bearing experiment notes**. Preserve
  them — they come from measured runs.

## Conventions

- New scripts prepend the project root to `sys.path`
  (`PROJECT_ROOT = Path(__file__).resolve().parent.parent`), then import from
  `drivers.*` / `utils.*`. Keep this pattern.
- Units are **meters and radians** everywhere; orientations are **intrinsic ZYX**
  (roll, pitch, yaw); poses are 4x4 homogeneous matrices named `T_<from>2<to>`
  (see [utils/transforms.py](utils/transforms.py)).
- Python 3.10, `from __future__ import annotations`, type hints, `numpy<2`.

## Detectors

The only detector is the CPU OpenCV tennis detector
([utils/opencv_tennis.py](utils/opencv_tennis.py)), built by `load_detector` in
[utils/detection.py](utils/detection.py). Its results mimic the ultralytics OBB
result API so the grasp geometry code reads them unchanged. No neural model, torch,
or GPU is used.
