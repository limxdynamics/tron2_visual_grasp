"""Simulated TRON2: the RobotDriver contract of Tron2Driver on top of a SimWorld.

Motion is kinematic and blocking calls take real time, so the mission sees the
same sequence it sees on the robot: watch, approach (base twist), kneel, body
pitch, observe pose, tracked grasp, park, release, stand.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import cv2
import numpy as np

from drivers.robot.base import RobotDriver
from utils.transforms import pose6d_to_mat4

from .world import ARM_REACH_M, KNEEL, STAND, TOOL_OFFSET, SimWorld


def _finite(name: str, *values: Any) -> list[float]:
    out = [float(v) for v in values]
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite, got {values}")
    return out


class SimTron2Driver(RobotDriver):
    """Drop-in for Tron2Driver in ./run.sh demo; never touches a network or a device."""

    def __init__(self, robot_cfg: dict[str, Any], world: SimWorld,
                 sim_cfg: Optional[dict[str, Any]] = None) -> None:
        cfg = robot_cfg.get("tron2") or {}
        self._world = world
        self._sim_cfg = sim_cfg or {}
        self._views: Optional[list[tuple[str, str, Any, float]]] = None
        self._observe = np.asarray(cfg["observe_joints"], float).reshape(6)
        self._park = np.asarray(cfg["park_joints"], float).reshape(6)
        self._joint_speed = float(cfg.get("joint_speed_rad_s", 3.0)) * float(cfg.get("speed_percent", 100)) / 100.0
        self._max_twist = (float(cfg.get("max_twist_mps", 0.8)), float(cfg.get("max_twist_rps", 1.0)))
        self._q = self._park.copy()
        self._running = False
        self._fk_arm: Any = None

    # Lifecycle

    def start(self) -> None:
        with self._world.lock:
            self._world.tcp = self._fk(self._park)
            self._world.tcp_goal = self._world.tcp.copy()
        self._running = True
        print("[SimTron2] Simulated TRON2 ready (no hardware)")

    def end(self, park: bool = True) -> None:
        if not self._running:
            return
        if park:
            self.move_park()
        self.set_base_twist(0.0, 0.0, 0.0)
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def _ensure_running(self) -> None:
        if not self._running:
            raise RuntimeError("Simulated TRON2 is not started")

    # Kinematics

    def _fk(self, q: np.ndarray) -> np.ndarray:
        """TCP (fingertip point) pose in the arm base frame, via piper_x flange FK."""
        if self._fk_arm is None:
            from pyAgxArm import AgxArmFactory, PiperFW, create_agx_arm_config
            self._fk_arm = AgxArmFactory.create_arm(create_agx_arm_config(
                robot="piper_x", firmeware_version=PiperFW.DEFAULT, channel="can0"))
        flange = pose6d_to_mat4(*np.asarray(self._fk_arm.fk(list(map(float, q))), float).reshape(6))
        return flange @ TOOL_OFFSET

    def _move_tcp(self, T_goal: np.ndarray, duration: float) -> bool:
        w = self._world
        with w.lock:
            w.step()
            dist = float(np.linalg.norm(T_goal[:3, 3] - w.tcp[:3, 3]))
            ang = float(np.linalg.norm(cv2.Rodrigues(T_goal[:3, :3] @ w.tcp[:3, :3].T)[0]))
            t = max(float(duration), dist / 0.6, ang / 2.5, 0.05)
            w.lin_speed, w.rot_speed = max(dist / t, 1e-3), max(ang / t, 1e-3)
            w.tcp_goal = T_goal.copy()
        deadline = time.monotonic() + t + 2.0
        while time.monotonic() < deadline:
            time.sleep(0.01)
            w.step()
            if w.arm_settled():
                return True
        return False

    # Motion

    def move_to_pose(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                     duration: float = 2.0) -> bool:
        self._ensure_running()
        x, y, z, roll, pitch, yaw = _finite("pose", x, y, z, roll, pitch, yaw)
        if math.hypot(x, y, z) > ARM_REACH_M:
            print(f"[SimTron2] Target ({x:+.3f},{y:+.3f},{z:+.3f}) is out of reach")
            return False
        return self._move_tcp(pose6d_to_mat4(x, y, z, roll, pitch, yaw), duration)

    def servo_pose(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                   max_speed_m_s: float = 0.2, max_rot_speed_rad_s: float = 1.0) -> None:
        self._ensure_running()
        x, y, z, roll, pitch, yaw = _finite("servo pose", x, y, z, roll, pitch, yaw)
        v, w_rot = _finite("servo speed", max_speed_m_s, max_rot_speed_rad_s)
        if v <= 0.0 or w_rot <= 0.0:
            raise ValueError("servo speeds must be positive")
        p = np.array([x, y, z])
        norm = float(np.linalg.norm(p))
        if norm > ARM_REACH_M:
            p *= ARM_REACH_M / norm                    # the arm stops at the edge of its reach
        T = pose6d_to_mat4(*p, roll, pitch, yaw)
        with self._world.lock:
            self._world.step()
            self._world.lin_speed, self._world.rot_speed = v, w_rot
            self._world.tcp_goal = T

    def stop_servo(self) -> None:
        with self._world.lock:
            self._world.step()
            self._world.tcp_goal = self._world.tcp.copy()

    def get_tcp_pose(self) -> np.ndarray:
        self._ensure_running()
        with self._world.lock:
            self._world.step()
            return self._world.tcp.copy()

    def move_joints(self, q_target: Any, duration: float = 0.0) -> bool:
        self._ensure_running()
        q = np.asarray(q_target, float).reshape(-1)
        if q.size != 6 or not np.all(np.isfinite(q)):
            raise ValueError(f"move_joints needs 6 finite joint angles, got {q_target}")
        t = max(float(duration), float(np.max(np.abs(q - self._q))) / max(self._joint_speed, 1e-3))
        self._world.arm_q_hint = q.copy()          # lets the 3D view keep the same IK branch
        ok = self._move_tcp(self._fk(q), t)
        if ok:
            self._q = q.copy()
        return ok

    def move_ready(self, ready_cfg: dict[str, Any]) -> bool:
        # Like Tron2Driver: the observe joints, with duration as a minimum.
        return self.move_joints(self._observe, float(ready_cfg.get("duration", 0.0)))

    def move_park(self) -> bool:
        self._ensure_running()
        if float(np.max(np.abs(self._q - self._park))) <= 1e-3 and self._world.arm_settled():
            return True
        return self.move_joints(self._park)

    # Base and posture

    def set_base_twist(self, x: float, y: float, yaw: float) -> None:
        self._ensure_running()
        x, y, yaw = _finite("twist", x, y, yaw)
        lin, rot = self._max_twist
        with self._world.lock:
            self._world.step()
            self._world.twist = np.array([np.clip(x, -lin, lin), np.clip(y, -lin, lin), np.clip(yaw, -rot, rot)])
            self._world.twist_t = time.monotonic()

    def _posture(self, target: dict[str, float], kneeling: bool) -> None:
        self._ensure_running()
        with self._world.lock:
            self._world.step()
            self._world.goal_height, self._world.goal_forward = target["height"], target["forward"]
            self._world.kneeling = kneeling
            if not kneeling:
                self._world.goal_pitch = 0.0      # the WF stand levels the body

    def kneel(self) -> None:
        print("[SimTron2] Kneel")
        self._posture(KNEEL, True)

    def stand(self) -> None:
        print("[SimTron2] Stand")
        self._posture(STAND, False)

    def set_body_pitch(self, pitch: float) -> None:
        self._ensure_running()
        (pitch,) = _finite("body pitch", pitch)
        with self._world.lock:
            self._world.goal_pitch = pitch

    def reset_body_pitch(self) -> None:
        self.set_body_pitch(0.0)

    # Gripper

    def set_gripper(self, distance_m: float, force: Optional[float] = None) -> None:
        self._ensure_running()
        (distance_m,) = _finite("gripper distance", distance_m)
        if distance_m > 0.01:
            self._world.release()

    def open_gripper(self, distance_m: Optional[float] = None, timeout: float = 3.0) -> None:
        self.set_gripper(0.09 if distance_m is None else distance_m)

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        self._ensure_running()
        time.sleep(0.15)
        return self._world.try_grasp()

    def release_gripper(self, timeout: float = 4.0) -> None:
        self._ensure_running()
        self._world.release()
        time.sleep(0.15)

    # Web console

    def scene_view(self) -> np.ndarray:
        """Top-down map of the simulated scene."""
        return self._world.scene_view()

    def debug_views(self) -> list[tuple[str, str, Any, float]]:
        """Extra web views as (name, label, render(), hz): the 3D robot view when
        MuJoCo and the model are available, and the top-down map."""
        if self._views is None:
            from .view3d import make_view3d
            self._views = []
            view3d = make_view3d(self._world, self._sim_cfg)
            if view3d is not None:
                hz = float((self._sim_cfg.get("view3d") or {}).get("hz", 15.0))
                self._views.append(("robot3d", "Robot (3D simulation)", view3d.render, hz))
            self._views.append(("scene", "Top view (simulation)", self.scene_view, 10.0))
        return self._views
