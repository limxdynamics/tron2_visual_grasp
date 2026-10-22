"""Shared state of the hardware-free demo: a TRON2 on a flat floor with tennis balls.

The simulated robot (drivers/sim/robot.py) and the simulated wrist camera
(drivers/sim/camera.py) both read and write one SimWorld, so what the camera
renders always follows what the robot did. The model is kinematic only and aims
at running the real mission code end to end, not at matching the real robot:

  world    floor plane z=0; x/y follow the robot's starting heading
  body     planar base pose (x, y, yaw) driven by request_twist-like commands,
           plus a torso posture (height, forward shift, pitch about the torso
           origin) for stand/kneel/pitch
  arm      a TCP pose in the arm-base frame that slews toward its goal; joint
           goals go through the piper_x forward kinematics from pyAgxArm, and the
           TCP is the gripper fingertip point, TOOL_OFFSET beyond the flange
  camera   mounted on the TCP by HAND_EYE, i.e. the same eye-in-hand transform
           the mission loads from config/calibration/sim_d405/hand_eye.npz

Body dimensions follow the WFYG_TRON2A model in limxdynamics/tron2-robot-description,
which drivers/sim/view3d.py renders when MuJoCo is installed.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from utils.transforms import pose6d_to_mat4

BALL_RADIUS_M = 0.033
# From the WFYG_TRON2A model: torso (base_Link) -> arm base, and flange -> the
# point between the gripper fingertips (gripper_pick).
TORSO_TO_ARM_BASE = np.array([[1.0, 0, 0, 0.01], [0, 1.0, 0, 0], [0, 0, 1.0, 0.2692], [0, 0, 0, 1.0]])
TOOL_OFFSET = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0.1354], [0, 0, 0, 1.0]])
WHEEL_RADIUS_M = 0.1

# Camera -> TCP (eye-in-hand): the D405 sits on the wrist, behind and above the
# fingertips, looking along the tool +Z axis tilted down by CAMERA_TILT_DEG.
CAMERA_TILT_DEG = 25.0
_TILT = math.radians(-CAMERA_TILT_DEG)
HAND_EYE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, math.cos(_TILT), -math.sin(_TILT), -0.06],
    [0.0, math.sin(_TILT), math.cos(_TILT), -0.11],
    [0.0, 0.0, 0.0, 1.0],
])
TWIST_TIMEOUT_S = 0.2          # the base stops when commands go quiet, like the real controller
ARM_REACH_M = 0.75             # TCP farther than this from the arm base is unreachable
GRASP_TOLERANCE_M = 0.05       # TCP within this of the ball centre closes on the ball

# Torso posture targets: base_Link height above the floor and forward shift.
# Standing keeps the knees slightly bent; kneeling folds the legs with the knees
# on the floor. The mission adds the body pitch on top while kneeling.
STAND = {"height": 0.68, "forward": 0.0}
KNEEL = {"height": 0.24, "forward": 0.1}
POSTURE_RATE_M_S = 0.35
PITCH_RATE_RAD_S = 0.6


def _rot_toward(R: np.ndarray, R_goal: np.ndarray, max_angle: float) -> tuple[np.ndarray, float]:
    """Rotate R toward R_goal by at most max_angle; returns (new R, remaining angle)."""
    rvec = cv2.Rodrigues(R_goal @ R.T)[0].reshape(3)
    angle = float(np.linalg.norm(rvec))
    if angle <= max_angle or angle < 1e-9:
        return R_goal.copy(), 0.0
    step = cv2.Rodrigues(rvec * (max_angle / angle))[0]
    return step @ R, angle - max_angle


@dataclass
class Ball:
    pos: np.ndarray                    # centre, world frame
    held: bool = False


@dataclass
class SimWorld:
    seed: Optional[int] = None
    spawn_distance_m: tuple[float, float] = (1.0, 1.6)
    spawn_lateral_m: tuple[float, float] = (-0.6, 0.6)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        # Base pose on the floor and the commanded twist.
        self.base = np.zeros(3)                       # x, y, yaw
        self.twist = np.zeros(3)
        self.twist_t = 0.0
        # Posture, current and goal.
        self.height, self.forward, self.pitch = STAND["height"], STAND["forward"], 0.0
        self.goal_height, self.goal_forward, self.goal_pitch = self.height, self.forward, 0.0
        self.kneeling = False
        # Arm TCP in the base frame, current and goal, with slew rates.
        self.tcp = np.eye(4)
        self.tcp_goal = np.eye(4)
        self.lin_speed = 0.5
        self.rot_speed = 2.0
        self.gripper_open = True
        self.picked = 0
        self.ball = Ball(self._spawn_point())
        self._t = time.monotonic()

    # Time

    def step(self, now: Optional[float] = None) -> None:
        """Advance base, posture and arm to `now`."""
        with self.lock:
            now = time.monotonic() if now is None else now
            dt = min(max(now - self._t, 0.0), 0.1)
            self._t = now
            if dt <= 0.0:
                return
            if now - self.twist_t <= TWIST_TIMEOUT_S and not self.kneeling:
                vx, vy, wz = self.twist
                yaw = self.base[2]
                self.base[0] += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
                self.base[1] += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
                self.base[2] = (yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi
            self.height += float(np.clip(self.goal_height - self.height, -POSTURE_RATE_M_S * dt, POSTURE_RATE_M_S * dt))
            self.forward += float(np.clip(self.goal_forward - self.forward, -POSTURE_RATE_M_S * dt, POSTURE_RATE_M_S * dt))
            self.pitch += float(np.clip(self.goal_pitch - self.pitch, -PITCH_RATE_RAD_S * dt, PITCH_RATE_RAD_S * dt))
            delta = self.tcp_goal[:3, 3] - self.tcp[:3, 3]
            dist = float(np.linalg.norm(delta))
            if dist > 1e-9:
                self.tcp[:3, 3] += delta * min(1.0, self.lin_speed * dt / dist)
            self.tcp[:3, :3], _ = _rot_toward(self.tcp[:3, :3], self.tcp_goal[:3, :3], self.rot_speed * dt)
            if self.ball.held:
                self.ball.pos = self.tcp_world()[:3, 3].copy()

    # Frames

    def torso_world(self) -> np.ndarray:
        """T_torso2world: floor pose, then posture (forward shift, height, pitch)."""
        with self.lock:
            x, y, yaw = self.base
            T = pose6d_to_mat4(x, y, 0.0, 0.0, 0.0, yaw)
            return T @ pose6d_to_mat4(self.forward, 0.0, self.height, 0.0, self.pitch, 0.0)

    def base_world(self) -> np.ndarray:
        """T_base2world of the arm base, the frame the mission's poses are in."""
        return self.torso_world() @ TORSO_TO_ARM_BASE

    def kneel_fraction(self) -> float:
        """0 standing .. 1 fully knelt, from the torso height."""
        with self.lock:
            span = STAND["height"] - KNEEL["height"]
            return float(np.clip((STAND["height"] - self.height) / span, 0.0, 1.0))

    def tcp_world(self) -> np.ndarray:
        with self.lock:
            return self.base_world() @ self.tcp

    def camera_world(self) -> np.ndarray:
        return self.tcp_world() @ HAND_EYE

    # Arm

    def arm_settled(self, tol_m: float = 1e-3, tol_rad: float = 1e-2) -> bool:
        with self.lock:
            dp = float(np.linalg.norm(self.tcp_goal[:3, 3] - self.tcp[:3, 3]))
            dr = float(np.linalg.norm(cv2.Rodrigues(self.tcp_goal[:3, :3] @ self.tcp[:3, :3].T)[0]))
            return dp <= tol_m and dr <= tol_rad

    # Balls

    def _spawn_point(self) -> np.ndarray:
        d = self.rng.uniform(*self.spawn_distance_m)
        lat = self.rng.uniform(*self.spawn_lateral_m)
        x, y, yaw = getattr(self, "base", np.zeros(3))
        return np.array([x + d * math.cos(yaw) - lat * math.sin(yaw),
                         y + d * math.sin(yaw) + lat * math.cos(yaw), BALL_RADIUS_M])

    def try_grasp(self) -> bool:
        with self.lock:
            self.step()
            self.gripper_open = False
            tcp = self.tcp_world()[:3, 3]
            if not self.ball.held and np.linalg.norm(tcp - self.ball.pos) <= GRASP_TOLERANCE_M:
                self.ball.held = True
            return self.ball.held

    def release(self) -> None:
        """Open the jaws; a held ball goes into the basket and a new one appears."""
        with self.lock:
            self.gripper_open = True
            if self.ball.held:
                self.picked += 1
                self.ball = Ball(self._spawn_point())

    # Top-down map for the web console

    def scene_view(self, size: int = 480, px_per_m: float = 110.0) -> np.ndarray:
        """Map centred on the robot, forward is up."""
        with self.lock:
            self.step()
            img = np.full((size, size, 3), (38, 34, 30), np.uint8)
            x0, y0, yaw = self.base
            origin = np.array([size / 2, size * 0.78])

            def to_px(p: np.ndarray) -> tuple[int, int]:
                dx, dy = p[0] - x0, p[1] - y0
                fx = dx * math.cos(yaw) + dy * math.sin(yaw)     # robot forward
                fy = -dx * math.sin(yaw) + dy * math.cos(yaw)    # robot left
                return int(origin[0] - fy * px_per_m), int(origin[1] - fx * px_per_m)

            # Floor grid fixed in the world, so base motion shows.
            span = size / px_per_m
            for g in np.arange(math.floor(x0 - span), math.ceil(x0 + span) + 1, 0.5):
                cv2.line(img, to_px(np.array([g, y0 - span])), to_px(np.array([g, y0 + span])), (58, 54, 50), 1)
            for g in np.arange(math.floor(y0 - span), math.ceil(y0 + span) + 1, 0.5):
                cv2.line(img, to_px(np.array([x0 - span, g])), to_px(np.array([x0 + span, g])), (58, 54, 50), 1)

            # Camera footprint: where the image corners hit the floor (clipped at 4 m).
            cam = self.camera_world()
            corners = []
            for u, v in [(-0.95, 0.95), (0.95, 0.95), (0.95, -0.55), (-0.95, -0.55)]:
                d = cam[:3, :3] @ np.array([u, v * 0.56, 1.0])
                t = -cam[2, 3] / d[2] if d[2] < -1e-6 else 4.0
                corners.append(to_px(cam[:3, 3] + d * min(t, 4.0)))
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(corners, np.int32)], (90, 120, 150))
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

            # Robot body and heading.
            body = [np.array([x0, y0]) + np.array([[math.cos(yaw), -math.sin(yaw)],
                                                    [math.sin(yaw), math.cos(yaw)]]) @ c
                    for c in ([0.25, 0.22], [0.25, -0.22], [-0.2, -0.22], [-0.2, 0.22])]
            cv2.fillPoly(img, [np.array([to_px(b) for b in body], np.int32)], (200, 200, 200))
            tcp = self.tcp_world()[:3, 3]
            cv2.line(img, to_px(self.base_world()[:3, 3]), to_px(tcp), (60, 60, 230), 3)
            cv2.circle(img, to_px(tcp), 5, (60, 60, 230), -1)

            ball_colour = (40, 210, 200)
            cv2.circle(img, to_px(self.ball.pos), max(6, int(BALL_RADIUS_M * px_per_m)), ball_colour, -1)

            posture = "kneeling" if self.kneeling else "standing"
            lines = [
                f"{posture}  pitch {math.degrees(self.pitch):+.0f} deg",
                f"gripper {'open' if self.gripper_open else 'closed'}"
                f"{'  holding ball' if self.ball.held else ''}",
                f"balls picked: {self.picked}",
            ]
            for i, text in enumerate(lines):
                cv2.putText(img, text, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            cv2.putText(img, "SIMULATION", (size - 118, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 200, 255), 1)
            return img


_WORLD: Optional[SimWorld] = None
_WORLD_LOCK = threading.Lock()


def get_world(cfg: Optional[dict[str, Any]] = None) -> SimWorld:
    """The process-wide world shared by the simulated camera and robot."""
    global _WORLD
    with _WORLD_LOCK:
        if _WORLD is None:
            sim_cfg = (cfg or {}).get("sim") or {}
            _WORLD = SimWorld(
                seed=sim_cfg.get("seed"),
                spawn_distance_m=tuple(sim_cfg.get("spawn_distance_m", (1.0, 1.6))),
                spawn_lateral_m=tuple(sim_cfg.get("spawn_lateral_m", (-0.6, 0.6))),
            )
        return _WORLD


def reset_world() -> None:
    """Drop the shared world (tests start each case from a fresh one)."""
    global _WORLD
    with _WORLD_LOCK:
        _WORLD = None
