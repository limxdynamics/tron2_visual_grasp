"""Demo 3D view: the TRON2 model is posed consistently with the simulator.

Skipped when MuJoCo is missing or the model cannot be downloaded; the render
test also skips without an OpenGL context (CI runs it under xvfb).
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def xml():
    from drivers.sim.model import ensure_model
    path = ensure_model(quiet=True)
    if path is None:
        pytest.skip("TRON2 model not available (offline?)")
    return path


@pytest.fixture
def scene(cfg, xml):
    from drivers.sim import SimTron2Driver, SimWorld, sim_config
    from drivers.sim.view3d import Robot3DView
    scfg = sim_config(cfg)
    world = SimWorld(seed=0)
    robot = SimTron2Driver(scfg["robot"], world)
    robot.start()
    view = Robot3DView(world, xml, 320, 240)
    yield scfg, world, robot, view
    view.close()


def _body(view, name):
    return mujoco.mj_name2id(view.model, mujoco.mjtObj.mjOBJ_BODY, name)


def _pose(view, name):
    b = _body(view, name)
    T = np.eye(4)
    T[:3, :3] = view.data.xmat[b].reshape(3, 3)
    T[:3, 3] = view.data.xpos[b]
    return T


def test_sim_constants_match_the_model(scene):
    from drivers.sim.world import TOOL_OFFSET, TORSO_TO_ARM_BASE
    _, _, _, view = scene
    view.data.qpos[:] = view.model.qpos0
    mujoco.mj_kinematics(view.model, view.data)
    torso_to_arm = np.linalg.inv(_pose(view, "base_Link")) @ _pose(view, "arm_base_Link")
    assert np.allclose(torso_to_arm, TORSO_TO_ARM_BASE, atol=1e-3)
    flange_to_pick = np.linalg.inv(_pose(view, "arm6_Link")) @ _pose(view, "gripper_pick")
    assert np.allclose(flange_to_pick[:3, 3], TOOL_OFFSET[:3, 3], atol=2e-3)


def test_model_arm_follows_the_simulated_tcp(scene):
    from drivers.sim.world import TOOL_OFFSET
    scfg, world, robot, view = scene
    for q in (scfg["mission"]["watch_joints"], scfg["robot"]["tron2"]["observe_joints"]):
        q = np.asarray(q, float)
        world.arm_q_hint = q.copy()
        world.tcp = world.tcp_goal = robot._fk(q)
        view.pose()
        target = world.base_world() @ world.tcp @ np.linalg.inv(TOOL_OFFSET)
        assert np.linalg.norm(view.data.xpos[_body(view, "arm6_Link")] - target[:3, 3]) < 2e-3


@pytest.mark.parametrize("kneel", [False, True])
def test_wheels_stay_on_the_floor(scene, kneel):
    from drivers.sim.world import KNEEL
    _, world, _, view = scene
    if kneel:
        world.height = world.goal_height = KNEEL["height"]
        world.forward = world.goal_forward = KNEEL["forward"]
        world.pitch = world.goal_pitch = 0.49
        world.kneeling = True
    view.pose()
    for side in "LR":
        assert view.data.xpos[_body(view, f"wheel_{side}_Link")][2] == pytest.approx(0.1, abs=2e-3)
        if kneel:
            assert view.data.xpos[_body(view, f"knee_{side}_Link")][2] == pytest.approx(0.06, abs=2e-3)


def test_legs_fold_smoothly(scene):
    """Kneeling and standing again move the leg joints continuously, with no branch flip."""
    from drivers.sim.world import KNEEL, STAND
    _, world, _, view = scene
    prev = None
    for f in list(np.linspace(0.0, 1.0, 41)) + list(np.linspace(1.0, 0.0, 41)):
        world.height = world.goal_height = STAND["height"] + f * (KNEEL["height"] - STAND["height"])
        view.pose()
        legs = np.array(view._legs)
        if prev is not None:
            assert np.max(np.abs(legs - prev)) < 0.1, (f, prev, legs)
        prev = legs
        assert view.data.xpos[_body(view, "wheel_L_Link")][2] == pytest.approx(0.1, abs=2e-3)


def test_render(scene):
    _, _, _, view = scene
    try:
        img = view.render()
    except Exception as exc:
        pytest.skip(f"no OpenGL context: {exc}")
    assert img.shape == (240, 320, 3) and img.std() > 5
