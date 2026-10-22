"""TRON2 driver against the mock upper-layer server: normal motion and fault handling."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest


def _finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


# Normal operation

def test_start_learns_accid_and_streams(mock_tron2, make_driver):
    d = make_driver()
    d.start()
    t0 = time.monotonic()
    time.sleep(0.5)
    assert d.is_running and d._accid == "MOCK"
    assert len(mock_tron2.messages("request_yg_servop", t0)) > 30   # ~100 Hz stream


def test_joint_pose_and_servo_moves(make_driver):
    d = make_driver()
    d.start()
    assert d.move_joints([0.3, 1.0, -1.0, 0.5, 0.0, 0.2])
    assert d.move_park()
    T = d.get_tcp_pose()
    x, y, z = T[:3, 3]
    assert d.move_to_pose(x + 0.05, y, z - 0.03, 0.0, 0.0, 0.0, duration=1.0)
    d.servo_pose(x + 0.10, y + 0.02, z, 0.0, 0.0, 0.0, max_speed_m_s=0.3, max_rot_speed_rad_s=2.0)
    time.sleep(1.5)
    assert np.linalg.norm(d.get_tcp_pose()[:3, 3] - [x + 0.10, y + 0.02, z]) < 0.01
    d.stop_servo()


def test_posture_twist_gripper(mock_tron2, make_driver):
    d = make_driver()
    d.start()
    d.kneel()
    d.set_body_pitch(0.3)
    d.reset_body_pitch()
    d.stand()
    t0 = time.monotonic()
    d.set_base_twist(0.3, 0.0, 0.2)
    d.open_gripper()
    assert d.grasp()
    time.sleep(0.1)
    assert mock_tron2.messages("request_twist", t0)[-1]["data"] == {"x": 0.3, "y": 0.0, "z": 0.2}
    assert mock_tron2.messages("request_set_yg_claw_cmd", t0)


# Faults

def test_malformed_replies_are_ignored(mock_tron2, make_driver):
    d = make_driver()
    d.start()
    for text in ["not json", "[1, 2, 3]",
                 json.dumps({"title": "response_get_yg_arm_pose", "data": "oops"}),
                 json.dumps({"title": "response_get_yg_arm_state",
                             "data": {"result": "success", "q": [None] * 6}})]:
        mock_tron2.push_raw(text)
    for mode in ["nan", "short", "zeroquat", "null"]:
        mock_tron2.mode["pose"] = mode
        time.sleep(0.25)
    mock_tron2.mode["pose"] = "ok"
    time.sleep(0.3)
    assert all(t.is_alive() for t in d._threads)
    assert _finite(d._measured[0]) and _finite(d._measured[1])
    assert all(_finite(m["data"]["pos"]) for m in mock_tron2.messages("request_yg_servop"))


@pytest.mark.parametrize("call", [
    lambda d: d.move_to_pose(float("nan"), 0, 0, 0, 0, 0),
    lambda d: d.servo_pose(0.3, 0, 0.2, 0, 0, 0, max_speed_m_s=-1),
    lambda d: d.servo_pose(float("inf"), 0, 0.2, 0, 0, 0),
    lambda d: d.move_joints([0, 0, 0, 0, 0]),
    lambda d: d.move_joints([0, 0, float("nan"), 0, 0, 0]),
    lambda d: d.set_gripper(float("nan")),
    lambda d: d.set_base_twist(float("nan"), 0, 0),
], ids=["pose-nan", "servo-neg-speed", "servo-inf", "joints-5", "joints-nan", "gripper-nan", "twist-nan"])
def test_invalid_inputs_raise(make_driver, call):
    d = make_driver()
    d.start()
    with pytest.raises(ValueError):
        call(d)


def test_twist_is_capped(mock_tron2, make_driver):
    d = make_driver()
    d.start()
    t0 = time.monotonic()
    d.set_base_twist(2.0, -2.0, 3.0)
    time.sleep(0.1)
    assert mock_tron2.messages("request_twist", t0)[-1]["data"] == {"x": 0.8, "y": -0.8, "z": 1.0}


def test_stale_feedback_pauses_cartesian_stream(mock_tron2, make_driver):
    d = make_driver()
    d.start()
    mock_tron2.mode["pose"] = "silent"
    time.sleep(1.2)
    t0 = time.monotonic()
    time.sleep(0.5)
    assert mock_tron2.messages("request_yg_servop", t0) == []
    mock_tron2.mode["pose"] = "ok"
    time.sleep(0.4)
    t0 = time.monotonic()
    time.sleep(0.5)
    assert len(mock_tron2.messages("request_yg_servop", t0)) > 30


def test_posture_rejected_and_timeout(mock_tron2, make_driver):
    d = make_driver()
    d.start()
    mock_tron2.mode["posture"] = "fail_busy"
    with pytest.raises(RuntimeError, match="rejected"):
        d.kneel()
    mock_tron2.mode["posture"] = "silent"
    with pytest.raises(RuntimeError, match="no response"):
        d.kneel()


def test_link_loss_raises_and_restart_recovers(mock_tron2, make_driver):
    d = make_driver()
    d.start()
    mock_tron2.drop_clients()
    time.sleep(0.6)
    with pytest.raises(RuntimeError, match="lost its connection"):
        d.set_base_twist(0.1, 0, 0)
    t0 = time.monotonic()
    time.sleep(0.3)
    assert mock_tron2.messages("request_yg_servop", t0) == []
    d.end(park=False)
    d.start()
    assert d.is_running


def test_reconnects_when_robot_drops_link_on_stand(mock_tron2, make_driver):
    mock_tron2.mode["drop_on_stand"] = True
    d = make_driver()
    d.start()
    d.kneel()
    d.stand()
    time.sleep(0.4)
    assert d.move_joints([0.2, 0.8, -0.8, 0.4, 0.0, 0.0])
    assert mock_tron2.connections == 2


def test_failed_start_cleans_up(mock_tron2, make_driver):
    mock_tron2.mode["pose"] = "silent"
    d = make_driver()
    with pytest.raises(RuntimeError):
        d.start()
    assert d._ws is None and not d._threads and not d.is_running


def test_unreachable_robot_fails_fast(make_driver):
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]          # nothing listens here once closed
    d = make_driver(port=port, connect_timeout_s=1.0)
    with pytest.raises(RuntimeError, match="Cannot reach"):
        d.start()
    assert d._ws is None
