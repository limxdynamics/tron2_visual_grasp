"""TRON2 arm driver over the upper-layer WebSocket JSON protocol.

Only the documented request/response pairs are used:
  request_yg_servop        streaming ee_link target, data.pos = [x,y,z,qw,qx,qy,qz]
  request_yg_servoj        streaming joint target, data.q = six joint angles
  request_get_yg_arm_pose  data.position = [x,y,z], data.quat = [w,x,y,z]
  request_get_yg_arm_state data.q = six encoder joint angles, for joint moves and FK fallback
  request_set_yg_claw_cmd  data.position (normalised opening), data.max_effort
  request_adjust_head_pitch body pitch, or data.reset to level it
  request_wf_stand_kneel    WF posture, data.action = kneel | stand
  request_twist            base velocity, data = {x, y, z(yaw)}

The arm servo controller behind this protocol is a streaming controller, not a task API: it
has no command id, no accept response and no completion event. Two consequences
shape this driver:

  * The stream must never pause for longer than the controller command_timeout
    (0.2 s) or the arm latches a hold, so a publisher thread keeps resending the
    current target.
  * Unreachable targets are not rejected. The arm stretches toward them and
    stalls against a joint limit while the controller still reports success, so
    every streamed Cartesian target is clamped to a small lead over the measured pose and
    arrival is judged from feedback plus a stall check.

Defensive behaviour, since the robot moves on whatever this driver sends:

  * Every outgoing number is checked to be finite before it is sent; a NaN or inf
    target raises ValueError instead of reaching the robot. Base twist is capped
    at max_twist_mps / max_twist_rps.
  * Incoming poses and joint states are validated (length, finite values,
    non-zero quaternion). A malformed reply is reported and ignored, never used
    as the measured pose, and one bad frame never stops the receive loop.
  * Cartesian targets are only streamed while pose feedback is fresh; if it goes
    stale the stream pauses, so the controller latches a hold instead of chasing
    a target computed from an old pose.
  * A dropped connection is reported, stops the stream, and makes the next
    command raise instead of silently timing out.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from typing import Any, Optional

import numpy as np

from .base import RobotDriver


# Controller-side command_timeout is 0.2 s; keep at least 2x timing margin.
_CMD_TIMEOUT_S = 0.2

# Target constraint, mirroring the controller's own max lead in position / rotation.
_MAX_LEAD_POS_M = 0.05
_MAX_LEAD_ROT_RAD = 0.3

# Arrival test, taken from the controller's own unreachable_pos_tol /
# unreachable_rot_tol. Its soft gains (kp=20) plus the output low-pass leave a
# steady-state error of roughly a centimetre, so a tighter test never passes.
_POS_TOL_M = 0.02
_ROT_TOL_RAD = 0.2

# Feedback movement below this counts as settled.
_SETTLE_POS_M = 0.0015
_SETTLE_ROT_RAD = 0.008
_SETTLE_SAMPLES = 6

# Joint-space (request_yg_servoj) motion tolerances.
_JOINT_TOL_RAD = math.radians(3.0)
_JOINT_SETTLE_RAD = math.radians(0.5)

# piper_x travel, from pyAgxArm ROBOT_JOINT_JOINT_LIMIT_PRESET_RAD. Feedback can
# read a hair past these, so a pose recorded off the real arm is not necessarily
# a pose the arm will accept back.
_JOINT_LIMITS_RAD = (
    (-2.617994, 2.617994),
    (0.0, 3.141593),
    (-2.967060, 0.0),
    (-1.553344, 1.553344),
    (-1.553344, 1.553344),
    (-3.141593, 3.141593),
)

# Look-down "observe" joints (piper_x). Joints=0 aims the wrist camera straight
# forward (flange +Z ~ [1,0,0]), so table objects land far ahead where a
# straight-down grasp exceeds the j4/j5 (+/-89 deg) wrist limits and the arm
# stalls at a side angle. This pose (measured on the real arm) tilts the camera
# ~45 deg down so top-down grasps stay reachable.
# j4 kept just inside the 1.5533 rad (89 deg) wrist limit.
_LOOKDOWN_JOINTS = (-0.046, 1.238, -1.931, 1.530, -0.023, 0.046)

_DEFAULTS = {
    "robot_ip": "10.192.1.2",
    "port": 5000,
    "accid": None,
    "stream_hz": 100.0,
    "feedback_hz": 30.0,
    "connect_timeout_s": 10.0,
    "feedback_timeout_s": 2.0,
    "posture_timeout_s": 30.0,
    "arm_reacquire_timeout_s": 15.0,
    "motion_timeout_extra_s": 4.0,
    "speed_percent": 50,
    "joint_speed_rad_s": 0.9,
    # Driver-side cap on request_twist, independent of the mission's own limits.
    "max_twist_mps": 0.8,
    "max_twist_rps": 1.0,
    "observe_joints": list(_LOOKDOWN_JOINTS),
    "park_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

_CLAW_DEFAULTS = {
    "max_distance_m": 0.09,
    "scale": 1.0,
    "invert": False,
    "default_effort": 10.0,
    "max_effort": 10.0,
    "grasp_min_width_m": 0.004,
    "settle_s": 1.0,
}


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic ZYX euler angles to a wxyz quaternion."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], float)
    return q / (np.linalg.norm(q) + 1e-12)


def _finite_floats(name: str, *values: Any) -> list[float]:
    """Convert values to float, raising ValueError on anything non-finite."""
    out = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: not a number: {value!r}") from None
        if not math.isfinite(f):
            raise ValueError(f"{name}: non-finite value {value!r}")
        out.append(f)
    return out


def _finite_vector(value: Any, size: int) -> Optional[np.ndarray]:
    """Return value as a finite float vector of the given size, or None."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != size or not np.all(np.isfinite(arr)):
        return None
    return arr


def _unit_quat(value: Any) -> Optional[np.ndarray]:
    """Return a normalised wxyz quaternion, or None if it is malformed or zero."""
    q = _finite_vector(value, 4)
    if q is None:
        return None
    n = float(np.linalg.norm(q))
    if n < 1e-6:
        return None
    return q / n


def _joint_gap_text(measured: Optional[np.ndarray], goal: np.ndarray) -> str:
    """Per-joint shortfall, so a stall points at the joint that stopped."""
    if measured is None:
        return ""
    gaps = [
        f"j{i + 1} {math.degrees(measured[i]):+.1f} vs {math.degrees(goal[i]):+.1f}"
        for i in range(6)
        if abs(measured[i] - goal[i]) > _JOINT_TOL_RAD
    ]
    return f" ({'; '.join(gaps)})" if gaps else ""


def _clamp_joints(q: np.ndarray) -> np.ndarray:
    """Hold a joint target inside the arm's travel, warning about what moved.

    The controller will not honour a target it cannot reach, so an out-of-range
    waypoint reads back as "unreachable" even when only one joint is a degree
    past its stop.
    """
    lo = np.array([limit[0] for limit in _JOINT_LIMITS_RAD])
    hi = np.array([limit[1] for limit in _JOINT_LIMITS_RAD])
    clamped = np.clip(q, lo, hi)
    off = np.flatnonzero(np.abs(clamped - q) > 1e-9)
    if off.size:
        detail = ", ".join(
            f"j{int(i) + 1} {math.degrees(q[i]):+.2f}->{math.degrees(clamped[i]):+.2f}deg"
            for i in off
        )
        print(f"[Tron2Driver] Joint target outside travel, clamped: {detail}")
    return clamped


def _quat_to_mat3(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions (a then b, i.e. a * b)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], float)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (== inverse for a unit quaternion) of a wxyz quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], float)


def _quat_angle(qa: np.ndarray, qb: np.ndarray) -> float:
    d = abs(float(np.dot(qa, qb)) / ((np.linalg.norm(qa) * np.linalg.norm(qb)) + 1e-12))
    return 2.0 * math.acos(min(1.0, d))


def _slerp(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    qa = qa / (np.linalg.norm(qa) + 1e-12)
    qb = qb / (np.linalg.norm(qb) + 1e-12)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        return (qa + t * (qb - qa)) / (np.linalg.norm(qa + t * (qb - qa)) + 1e-12)
    theta = math.acos(min(1.0, dot))
    s = math.sin(theta)
    return (math.sin((1 - t) * theta) / s) * qa + (math.sin(t * theta) / s) * qb


class Tron2Driver(RobotDriver):
    """PiperX arm reached through the TRON2 upper-layer WebSocket API."""

    def __init__(self, robot_cfg: Optional[dict[str, Any]] = None) -> None:
        robot_cfg = robot_cfg or {}
        cfg = {**_DEFAULTS, **(robot_cfg.get("tron2") or {})}
        ccfg = {**_CLAW_DEFAULTS, **((robot_cfg.get("gripper") or {}).get("tron2") or {})}

        self._url = f"ws://{cfg['robot_ip']}:{int(cfg['port'])}"
        self._accid = cfg["accid"]
        self._stream_dt = 1.0 / max(float(cfg["stream_hz"]), 1.0)
        self._feedback_dt = 1.0 / max(float(cfg["feedback_hz"]), 1.0)
        self._connect_timeout_s = float(cfg["connect_timeout_s"])
        self._feedback_timeout_s = float(cfg["feedback_timeout_s"])
        self._posture_timeout_s = float(cfg["posture_timeout_s"])
        if not math.isfinite(self._posture_timeout_s) or self._posture_timeout_s <= 0:
            raise ValueError("posture_timeout_s must be finite and positive")
        self._arm_reacquire_timeout_s = float(cfg["arm_reacquire_timeout_s"])
        self._motion_timeout_extra_s = float(cfg["motion_timeout_extra_s"])
        # speed_percent scales every move duration: 50% -> twice as long -> half speed.
        self._speed_scale = 100.0 / float(np.clip(int(cfg["speed_percent"]), 1, 100))
        self._joint_speed_rad_s = float(cfg["joint_speed_rad_s"])
        if not math.isfinite(self._joint_speed_rad_s) or self._joint_speed_rad_s <= 0.0:
            raise ValueError("robot.tron2.joint_speed_rad_s must be positive")
        self._max_twist_mps, self._max_twist_rps = _finite_floats(
            "robot.tron2.max_twist_mps/max_twist_rps", cfg["max_twist_mps"], cfg["max_twist_rps"])
        if self._max_twist_mps <= 0.0 or self._max_twist_rps <= 0.0:
            raise ValueError("robot.tron2.max_twist_mps and max_twist_rps must be positive")
        self._observe_joints = np.asarray(cfg["observe_joints"], float).reshape(6)
        self._park_joints = np.asarray(cfg["park_joints"], float).reshape(6)
        self._arm_model = str(
            cfg.get("arm_model") or (robot_cfg.get("piper") or {}).get("model", "piper_x")
        ).upper()
        # The arm servo controller reports/accepts ee poses RELATIVE to prepare_ee_pose (a
        # position offset along base axes plus an orientation right-multiplied onto
        # prepare). get_tcp_pose/move_to_pose compose this back to an absolute base
        # pose so hand-eye calibration and grasp math share one rigid frame. Only the
        # prepare ORIENTATION changes results; the position cancels between read-back
        # and command, so an auto-derived (FK) position is fine.
        self._apply_prepare_offset = bool(cfg.get("apply_prepare_offset", True))
        pj = cfg.get("prepare_joints")
        self._prepare_joints = np.asarray(
            pj if pj is not None else cfg["park_joints"], float).reshape(6)
        self._prepare_cfg = self._parse_prepare_ee_pose(cfg.get("prepare_ee_pose"))
        self._prepare_p: Optional[np.ndarray] = None
        self._prepare_q: Optional[np.ndarray] = None
        if self._stream_dt > _CMD_TIMEOUT_S / 2.0:
            raise ValueError(
                f"robot.tron2.stream_hz must be at least {2.0 / _CMD_TIMEOUT_S:.0f} Hz "
                "to stay inside the controller command timeout"
            )

        self.MAX_DISTANCE_M = float(ccfg["max_distance_m"])
        self._claw_scale = float(ccfg["scale"])
        self._claw_invert = bool(ccfg["invert"])
        self._default_effort = float(ccfg["default_effort"])
        self._max_effort = float(ccfg["max_effort"])
        self._grasp_min_width_m = float(ccfg["grasp_min_width_m"])
        self._claw_settle_s = float(ccfg["settle_s"])

        self._ws: Any = None
        self._connected = threading.Event()
        self._running = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._target: Optional[np.ndarray] = None
        self._target_is_joint = False
        self._servo_goal: Optional[np.ndarray] = None
        self._servo_speed = (0.2, 1.0)
        self._measured: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._measured_t = 0.0
        self._measured_evt = threading.Event()
        self._measured_q: Optional[np.ndarray] = None
        self._joint_evt = threading.Event()
        self._pose_error: Optional[str] = None
        self._pose_api_unavailable = False
        self._pitch_result: Optional[str] = None
        self._pitch_evt = threading.Event()
        self._posture_lock = threading.Lock()
        self._posture_guid: Optional[str] = None
        self._posture_result: Optional[str] = None
        self._posture_evt = threading.Event()
        # Set by a WF stand, which takes the arm away from this stream.
        self._arm_taken_by_stand = False
        self._robot_status: Optional[str] = None
        self._fk_robot: Any = None
        self._fk_notice_printed = False
        self._fk_warning_printed = False
        self._threads: list[threading.Thread] = []
        self._link_lost = False
        self._stream_paused = False
        self._warned: set[str] = set()

    # Lifecycle

    def start(self) -> None:
        if self._running:
            return
        try:
            import websocket
        except ImportError as e:
            raise RuntimeError(
                f"websocket-client is not installed: {e}\n  pip install websocket-client"
            ) from e

        self._stop.clear()
        self._connected.clear()
        self._link_lost = False
        self._stream_paused = False
        self._ws = websocket.WebSocketApp(
            self._url, on_open=self._on_open, on_message=self._on_message,
            on_close=self._on_close, on_error=self._on_error,
        )
        self._spawn(self._ws.run_forever, "ws")
        try:
            if not self._connected.wait(self._connect_timeout_s):
                raise RuntimeError(f"Cannot reach the robot at {self._url}")
            self._wait_accid()

            self._spawn(self._feedback_loop, "feedback")
            pose = self._wait_pose()
            if pose is None:
                raise RuntimeError(
                    "TRON2 arm pose is unavailable from both "
                    "request_get_yg_arm_pose and request_get_yg_arm_state"
                    f"{f' ({self._pose_error})' if self._pose_error else ''}\n"
                    "  Check the upper-layer arm service and its joint-state source."
                )
            self._resolve_prepare_pose()
        except BaseException:
            # Do not leave the socket and feedback thread running after a failed start.
            self._shutdown_threads()
            raise
        # Anchor on the measured pose so the first streamed frame does not jump.
        self._target = np.concatenate(pose)
        self._running = True
        self._spawn(self._stream_loop, "servop")
        print(f"[Tron2Driver] Ready ({self._url}, accid={self._accid}, "
              f"stream={1.0 / self._stream_dt:.0f}Hz)")

    def end(self, park: bool = True) -> None:
        if self._ws is None:
            return
        if self._running and park:
            try:
                self.move_park()
            except Exception as exc:
                print(f"[Tron2Driver] Park failed: {exc}")
        # Dropping the stream is the documented way to stop: the controller
        # latches a rigid hold on the measured joints after command_timeout.
        self._shutdown_threads()

    def _shutdown_threads(self) -> None:
        self._stop.set()
        self._running = False
        for t in self._threads:
            if t.name != "ws":
                t.join(timeout=1.0)
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        self._ws = None
        self._threads.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    # Motion

    def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        duration: float = 2.0,
    ) -> bool:
        self._ensure_running()
        x, y, z, roll, pitch, yaw, duration = _finite_floats(
            "move_to_pose", x, y, z, roll, pitch, yaw, duration)
        if duration < 0.0:
            raise ValueError("move_to_pose: duration must not be negative")
        goal_p = np.array([x, y, z])
        goal_q = _rpy_to_quat(roll, pitch, yaw)
        # Callers work in the absolute base frame; the servop stream is
        # prepare-relative, so map the goal back before streaming/arrival checks.
        if self._prepare_q is not None and not self._pose_api_unavailable:
            goal_p, goal_q = self._servo_from_base(goal_p, goal_q)
        duration = float(duration) * self._speed_scale

        start = self._wait_pose()
        if start is None:
            print("[Tron2Driver] No arm pose feedback; refusing to move")
            return False
        p0, q0 = start

        steps = max(int(float(duration) / self._stream_dt), 1)
        t0 = time.monotonic()
        for i in range(1, steps + 1):
            if self._stop.is_set():
                return False
            s = i / steps
            self._set_target((1.0 - s) * p0 + s * goal_p, _slerp(q0, goal_q, s))
            time.sleep(self._stream_dt)

        self._set_target(goal_p, goal_q)
        t1 = time.monotonic()
        ok = self._wait_arrival(goal_p, goal_q, float(duration) + self._motion_timeout_extra_s)
        print(f"[Tron2Driver] Pose move: stream {t1 - t0:.2f}s + settle "
              f"{time.monotonic() - t1:.2f}s")
        self._latch_hold()
        return ok

    def servo_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        max_speed_m_s: float = 0.2,
        max_rot_speed_rad_s: float = 1.0,
    ) -> None:
        self._ensure_running()
        x, y, z, roll, pitch, yaw, max_speed_m_s, max_rot_speed_rad_s = _finite_floats(
            "servo_pose", x, y, z, roll, pitch, yaw, max_speed_m_s, max_rot_speed_rad_s)
        if max_speed_m_s <= 0.0 or max_rot_speed_rad_s <= 0.0:
            raise ValueError("servo_pose: speed limits must be positive")
        goal_p = np.array([x, y, z])
        goal_q = _rpy_to_quat(roll, pitch, yaw)
        if self._prepare_q is not None and not self._pose_api_unavailable:
            goal_p, goal_q = self._servo_from_base(goal_p, goal_q)
        with self._lock:
            if self._target is None or self._target_is_joint:
                if self._measured is None:
                    return
                self._target = np.concatenate(self._measured)
                self._target_is_joint = False
            self._servo_speed = (float(max_speed_m_s), float(max_rot_speed_rad_s))
            self._servo_goal = np.concatenate([goal_p, goal_q])

    def stop_servo(self) -> None:
        self._latch_hold()

    def move_joints(self, q_target: Any, duration: float = 0.0) -> bool:
        """Interpolate all joints to q_target and stream request_yg_servoj.

        duration is a floor, not a target: the move is never faster than
        joint_speed_rad_s. After a WF stand the arm ignores this stream until
        it is reconnected (a program restart recovered it), so a failed move
        then reconnects and retries until arm_reacquire_timeout_s.
        """
        ok = self._move_joints_after_stand(q_target, duration)
        if ok:
            self._arm_taken_by_stand = False
        if ok or not self._arm_taken_by_stand:
            return ok
        deadline = time.monotonic() + self._arm_reacquire_timeout_s
        while not ok and time.monotonic() < deadline:
            print("[Tron2Driver] Arm ignored the move after the WF stand; "
                  "reconnecting the arm stream")
            try:
                self.end(park=False)
                time.sleep(0.5)
                self.start()
            except Exception as exc:
                print(f"[Tron2Driver] Reconnect failed: {exc}")
                continue
            ok = self._move_joints_after_stand(q_target, duration)
        if ok:
            print("[Tron2Driver] Arm control regained after the WF stand")
            self._arm_taken_by_stand = False
        else:
            print(f"[Tron2Driver] Arm still ignored commands "
                  f"{self._arm_reacquire_timeout_s:g}s after the WF stand")
        return ok

    def _move_joints_after_stand(self, q_target: Any, duration: float) -> bool:
        """_move_joints_once, but a link lost after a WF stand counts as a failed move.

        If the robot drops the connection when it stands, the reconnect-and-retry
        path in move_joints must still run instead of the error escaping it.
        """
        try:
            return self._move_joints_once(q_target, duration)
        except (RuntimeError, ConnectionError):
            if self._arm_taken_by_stand and self._link_lost:
                return False
            raise

    def _move_joints_once(self, q_target: Any, duration: float) -> bool:
        self._ensure_running()
        q_raw = _finite_vector(q_target, 6)
        if q_raw is None:
            raise ValueError(f"move_joints: expected 6 finite joint angles, got {q_target!r}")
        (duration,) = _finite_floats("move_joints duration", duration)
        q_goal = _clamp_joints(q_raw)
        q0 = self._wait_joints()
        if q0 is None:
            print("[Tron2Driver] No joint feedback; refusing to move")
            return False
        max_delta = float(np.max(np.abs(q_goal - q0)))
        duration = max(float(duration), max_delta / self._joint_speed_rad_s) * self._speed_scale
        steps = max(int(duration / self._stream_dt), 1)
        t0 = time.monotonic()
        for i in range(1, steps + 1):
            if self._stop.is_set():
                return False
            s = i / steps
            self._set_joint_target((1.0 - s) * q0 + s * q_goal)
            time.sleep(self._stream_dt)
        self._set_joint_target(q_goal)
        t1 = time.monotonic()
        ok = self._wait_joint_arrival(q_goal, duration + self._motion_timeout_extra_s)
        print(f"[Tron2Driver] Joint move {math.degrees(max_delta):.0f}deg: "
              f"stream {t1 - t0:.2f}s + settle {time.monotonic() - t1:.2f}s")
        self._latch_hold()
        return ok

    def move_ready(self, ready_cfg: dict[str, Any]) -> bool:
        """Move to the look-down observe pose, reached via request_yg_servoj.

        Joints=0 aims the wrist camera forward, which puts table objects far
        ahead where a straight-down grasp exceeds the wrist limits and the arm
        stalls at a side angle. The observe pose tilts the camera down so
        top-down grasps stay reachable.
        """
        duration = float((ready_cfg or {}).get("duration", 3.0))
        ok = self.move_joints(self._observe_joints, duration=duration)
        if ok:
            try:
                p = self.get_tcp_pose()[:3, 3]
                print(f"[Tron2Driver] Ready at observe pose, tcp "
                      f"xyz=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
            except Exception:
                pass
        return ok

    def move_park(self) -> bool:
        """Fold the arm to the stowed joints, by default the zero pose."""
        self._ensure_running()
        q = self._wait_joints()
        if q is not None and float(np.max(np.abs(q - self._park_joints))) <= _JOINT_TOL_RAD:
            return True
        print("[Tron2Driver] Fold the arm to the park joints")
        return self.move_joints(self._park_joints)

    # Prepare-relative <-> absolute base-frame mapping

    @staticmethod
    def _parse_prepare_ee_pose(value: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Parse an explicit prepare_ee_pose config value into (position, wxyz quat)."""
        if value is None:
            return None
        arr = np.asarray(value, float).reshape(-1)
        if not np.all(np.isfinite(arr)):
            raise ValueError("robot.tron2.prepare_ee_pose must contain only finite numbers")
        if arr.size == 7:
            q = _unit_quat(arr[3:])
            if q is None:
                raise ValueError("robot.tron2.prepare_ee_pose quaternion must be non-zero")
            return arr[:3].copy(), q
        if arr.size == 6:
            return arr[:3].copy(), _rpy_to_quat(*arr[3:])
        raise ValueError(
            "robot.tron2.prepare_ee_pose must be [x,y,z,qw,qx,qy,qz] or "
            "[x,y,z,roll,pitch,yaw]"
        )

    def _resolve_prepare_pose(self) -> None:
        """Resolve prepare_ee_pose so prepare-relative servo poses map to the base
        frame. Explicit config wins; otherwise derive it from FK at the prepare
        joints. Only the orientation must be exact -- the position cancels between
        read-back and command.
        """
        if not self._apply_prepare_offset:
            print("[Tron2Driver] Prepare-offset composition disabled; arm poses used raw")
            return
        if self._prepare_cfg is not None:
            self._prepare_p, self._prepare_q = self._prepare_cfg
            print("[Tron2Driver] Using configured prepare_ee_pose for base<->servo mapping")
            return
        saved_err = self._pose_error
        fk = self._pose_from_joint_state(self._prepare_joints, quiet=True)
        self._pose_error = saved_err
        if fk is None:
            print("[Tron2Driver] WARNING: prepare_ee_pose unresolved (FK unavailable); "
                  "arm poses stay in the raw prepare-relative frame and hand-eye "
                  "translation will be wrong. Set robot.tron2.prepare_ee_pose.")
            return
        self._prepare_p, self._prepare_q = fk[0], fk[1]
        print("[Tron2Driver] Derived prepare_ee_pose from FK at prepare_joints "
              "(assumes ee_link shares the flange orientation)")

    def _base_from_servo(
        self, p_rel: np.ndarray, q_rel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compose a prepare-relative (position, wxyz quat) into the base frame."""
        return self._prepare_p + p_rel, _quat_mul(self._prepare_q, q_rel)

    def _servo_from_base(
        self, p_abs: np.ndarray, q_abs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of _base_from_servo: absolute base pose -> prepare-relative."""
        return p_abs - self._prepare_p, _quat_mul(_quat_conj(self._prepare_q), q_abs)

    def get_tcp_pose(self) -> np.ndarray:
        self._ensure_running()
        pose = self._wait_pose()
        if pose is None:
            raise RuntimeError("TRON2 arm pose feedback is not ready")
        p, q = pose
        # The API pose is prepare-relative; compose it to an absolute base pose.
        # The FK fallback is already absolute (flange), so leave it untouched.
        if self._prepare_q is not None and not self._pose_api_unavailable:
            p, q = self._base_from_servo(p, q)
        T = np.eye(4)
        T[:3, :3] = _quat_to_mat3(q)
        T[:3, 3] = p
        return T

    @property
    def tcp_pose_api_available(self) -> bool:
        """True while get_tcp_pose() reports the controller TCP frame.

        Once request_get_yg_arm_pose returns fail_no_data the driver falls back
        to request_get_yg_arm_state + FK, and get_tcp_pose() then reports the
        FLANGE frame instead, so poses before and after the switch are not comparable.
        """
        return not self._pose_api_unavailable

    def set_base_twist(self, x: float, y: float, yaw: float) -> None:
        self._ensure_running()
        x, y, yaw = _finite_floats("set_base_twist", x, y, yaw)
        cx = float(np.clip(x, -self._max_twist_mps, self._max_twist_mps))
        cy = float(np.clip(y, -self._max_twist_mps, self._max_twist_mps))
        cyaw = float(np.clip(yaw, -self._max_twist_rps, self._max_twist_rps))
        if (cx, cy, cyaw) != (x, y, yaw):
            self._warn_once("twist_cap", f"Base twist ({x:+.2f}, {y:+.2f}, {yaw:+.2f}) capped to "
                            f"({cx:+.2f}, {cy:+.2f}, {cyaw:+.2f}) by max_twist_mps/max_twist_rps")
        self._send("request_twist", {"x": cx, "y": cy, "z": cyaw})

    def kneel(self) -> None:
        self._set_wf_posture("kneel")

    def stand(self) -> None:
        self._set_wf_posture("stand")
        self._arm_taken_by_stand = True

    def _set_wf_posture(self, action: str) -> None:
        self._ensure_running()
        with self._posture_lock:
            guid = str(uuid.uuid4())
            with self._lock:
                self._posture_guid = guid
                self._posture_result = None
                self._posture_evt.clear()
            try:
                self._send("request_wf_stand_kneel", {"action": action}, guid=guid)
                if not self._posture_evt.wait(self._posture_timeout_s):
                    raise RuntimeError(
                        f"WF {action}: no response within {self._posture_timeout_s:g}s"
                    )
                with self._lock:
                    result = self._posture_result
                if result != "success":
                    raise RuntimeError(f"WF {action} rejected: {result}")
                print(f"[Tron2Driver] WF {action}: success")
            finally:
                with self._lock:
                    self._posture_guid = None
                    self._posture_result = None

    def set_body_pitch(self, pitch: float) -> None:
        self._ensure_running()
        if not 0.0 <= float(pitch) <= 0.5:
            raise ValueError("Body pitch must be between 0.0 and 0.5 radians")
        self._send_body_pitch({"pitch": float(pitch)})

    def reset_body_pitch(self) -> None:
        """Level the body. Pitch is relative, so only reset returns it to zero."""
        self._ensure_running()
        self._send_body_pitch({"reset": True})

    def _send_body_pitch(self, data: dict[str, Any]) -> None:
        self._pitch_result = None
        self._pitch_evt.clear()
        self._send("request_adjust_head_pitch", data)
        # The controller answers success even when it will not move, so report
        # the raw reply instead of assuming the body followed.
        if not self._pitch_evt.wait(self._feedback_timeout_s):
            print("[Tron2Driver] Body pitch command got no reply")
        elif self._pitch_result != "success":
            print(f"[Tron2Driver] Body pitch rejected: {self._pitch_result}")
        else:
            print(f"[Tron2Driver] Body pitch {data} acknowledged (no motion feedback)")

    # Gripper

    def set_gripper(self, distance_m: float, force: Optional[float] = None) -> None:
        self._ensure_running()
        (distance_m,) = _finite_floats("set_gripper distance_m", distance_m)
        if force is not None:
            (force,) = _finite_floats("set_gripper force", force)
        frac = float(np.clip(distance_m / max(self.MAX_DISTANCE_M, 1e-6), 0.0, 1.0))
        if self._claw_invert:
            frac = 1.0 - frac
        self._send("request_set_yg_claw_cmd", {
            "position": frac * self._claw_scale,
            "max_effort": self._clamp_effort(force),
        })

    def open_gripper(self, distance_m: Optional[float] = None, timeout: float = 3.0) -> None:
        self.set_gripper(self.MAX_DISTANCE_M if distance_m is None else distance_m)
        time.sleep(min(self._claw_settle_s, max(0.0, timeout)))

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        self.set_gripper(0.0, force=force)
        time.sleep(min(self._claw_settle_s, max(0.0, timeout)))
        # The protocol has no jaw width feedback, so an empty grasp cannot be detected.
        return True

    def release_gripper(self, timeout: float = 4.0) -> None:
        self.open_gripper(timeout=min(2.0, timeout))

    # Internals

    def _ensure_running(self) -> None:
        if self._link_lost:
            raise RuntimeError(f"Tron2Driver lost its connection to {self._url}")
        if not self._running:
            raise RuntimeError("Tron2Driver is not started; call start() first")

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[Tron2Driver] {text}")

    def _spawn(self, fn: Any, name: str) -> None:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _clamp_effort(self, force: Optional[float]) -> float:
        value = self._default_effort if force is None else float(force)
        return float(np.clip(abs(value), 0.0, self._max_effort))

    # Protocol

    def _on_open(self, _ws: Any) -> None:
        self._connected.set()

    def _on_close(self, _ws: Any, code: Any = None, reason: Any = None) -> None:
        was_open = self._connected.is_set()
        self._connected.clear()
        if was_open and not self._stop.is_set():
            self._link_lost = True
            print(f"[Tron2Driver] Connection to {self._url} closed"
                  f"{f' ({code} {reason})' if code or reason else ''}; streaming stopped")

    def _on_error(self, _ws: Any, error: Any) -> None:
        if not self._stop.is_set():
            print(f"[Tron2Driver] WebSocket error: {error}")

    def _on_message(self, _ws: Any, message: str) -> None:
        # One malformed frame must never stop the receive loop.
        try:
            self._handle_message(message)
        except Exception as exc:
            self._warn_once("bad_message",
                            f"Ignored a malformed message ({type(exc).__name__}: {exc})")

    def _handle_message(self, message: str) -> None:
        try:
            root = json.loads(message)
        except (ValueError, TypeError):
            self._warn_once("bad_json", "Ignored a message that is not valid JSON")
            return
        if not isinstance(root, dict):
            self._warn_once("bad_root", "Ignored a message that is not a JSON object")
            return
        accid = root.get("accid")
        if self._accid is None and isinstance(accid, (str, int)) and accid != "":
            self._accid = accid
        title = root.get("title")
        raw_data = root.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        if title == "notify_robot_info":
            # No posture-done event exists; status changes are the only hint of
            # when a kneel/stand actually finishes.
            status = data.get("status") if isinstance(data, dict) else None
            if status is not None and status != self._robot_status:
                if self._robot_status is not None:
                    print(f"[Tron2Driver] Robot status {self._robot_status} -> {status}")
                self._robot_status = status
            return
        if title in ("response_wf_stand_kneel", "notify_invalid_request"):
            with self._lock:
                if self._posture_guid is not None and root.get("guid") == self._posture_guid:
                    self._posture_result = (
                        str(data.get("result", "missing result"))
                        if title == "response_wf_stand_kneel" and isinstance(data, dict)
                        else f"invalid response: {raw_data}"
                    )
                    self._posture_evt.set()
            if title == "response_wf_stand_kneel":
                return
        if title == "response_get_yg_arm_pose":
            if data.get("result") == "success":
                p = _finite_vector(data.get("position"), 3)
                q = _unit_quat(data.get("quat"))
                if p is None or q is None:
                    self._pose_error = "malformed arm pose in response_get_yg_arm_pose"
                    self._warn_once("bad_pose", f"{self._pose_error}; ignored")
                else:
                    self._measured = (p, q)
                    self._measured_t = time.monotonic()
                    self._pose_error = None
            else:
                self._pose_error = data.get("result")
                if self._pose_error == "fail_no_data":
                    self._pose_api_unavailable = True
            self._measured_evt.set()
        elif title == "response_get_yg_arm_state":
            q = _finite_vector(data.get("q"), 6) if data.get("result") == "success" else None
            if data.get("result") == "success" and q is None:
                self._warn_once("bad_joints", "Malformed joint state in "
                                "response_get_yg_arm_state; ignored")
            if q is not None:
                self._measured_q = q
                if self._pose_api_unavailable:
                    pose = self._pose_from_joint_state(self._measured_q)
                    if pose is not None:
                        self._measured = pose
                        self._measured_t = time.monotonic()
                        self._pose_error = None
                    self._measured_evt.set()
            elif self._pose_api_unavailable:
                self._pose_error = data.get("result", "arm state unavailable")
                self._measured_evt.set()
            self._joint_evt.set()
        elif title == "notify_invalid_request":
            print(f"[Tron2Driver] Request rejected: {raw_data}")
        elif title == "response_adjust_head_pitch":
            self._pitch_result = str(data.get("result", "no result"))
            self._pitch_evt.set()
        elif title and title.startswith("response_") \
                and str(data.get("result", "")).startswith("fail"):
            print(f"[Tron2Driver] {title} failed: {data.get('result')}")

    def _send(
        self, title: str, data: Optional[dict] = None, guid: Optional[str] = None,
    ) -> None:
        ws = self._ws
        if ws is None or self._link_lost:
            raise ConnectionError(f"not connected to {self._url}")
        ws.send(json.dumps({
            "accid": self._accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": guid if guid is not None else str(uuid.uuid4()),
            "data": data or {},
        }))

    def _wait_accid(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self._accid is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._accid is None:
            print("[Tron2Driver] No notify_robot_info yet; sending with accid=None")

    # Loops

    def _feedback_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._send(
                    "request_get_yg_arm_state"
                    if self._pose_api_unavailable
                    else "request_get_yg_arm_pose"
                )
            except Exception as exc:
                self._report_stream_failure("feedback", exc)
                return
            self._stop.wait(self._feedback_dt)

    def _stream_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if self._servo_goal is not None and self._target is not None \
                        and not self._target_is_joint:
                    self._target = self._slew_toward_goal(self._target)
                is_joint = self._target_is_joint
                target = None if self._target is None else self._target.copy()
            if target is not None and not is_joint and not self._feedback_fresh():
                # Never chase a target computed from an old pose: pause the
                # stream so the controller latches a hold until feedback returns.
                if not self._stream_paused:
                    self._stream_paused = True
                    print("[Tron2Driver] Arm pose feedback stale; pausing the servo stream")
                target = None
            elif target is not None and self._stream_paused:
                self._stream_paused = False
                print("[Tron2Driver] Arm pose feedback back; servo stream resumed")
            if target is not None:
                try:
                    if is_joint:
                        msg = ("request_yg_servoj", {"q": [float(v) for v in target]})
                    else:
                        msg = ("request_yg_servop",
                               {"pos": [float(v) for v in self._clamp_lead(target)]})
                    values = msg[1].get("q") or msg[1].get("pos")
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError(f"non-finite {msg[0]} target {values}")
                    self._send(*msg)
                except ValueError as exc:
                    self._warn_once("bad_target", f"Dropped a servo target: {exc}")
                    with self._lock:
                        self._servo_goal = None
                        self._target = None
                except Exception as exc:
                    self._report_stream_failure("servo stream", exc)
                    return
            self._stop.wait(self._stream_dt)

    def _feedback_fresh(self) -> bool:
        return time.monotonic() - self._measured_t <= self._feedback_timeout_s

    def _report_stream_failure(self, what: str, exc: Exception) -> None:
        if self._stop.is_set():
            return
        if self._link_lost and isinstance(exc, ConnectionError):
            return  # the close itself was already reported
        self._link_lost = True
        self._warn_once(f"fail_{what}", f"{what} stopped: {type(exc).__name__}: {exc}")

    # Target handling

    def _slew_toward_goal(self, target: np.ndarray) -> np.ndarray:
        """Advance the streamed target one stream tick toward the servo goal."""
        goal = self._servo_goal
        max_v, max_w = self._servo_speed
        p, q = target[:3], target[3:]
        dp = goal[:3] - p
        n = float(np.linalg.norm(dp))
        step = max_v * self._stream_dt
        p = goal[:3].copy() if n <= step else p + dp * (step / n)
        ang = _quat_angle(q, goal[3:])
        rot_step = max_w * self._stream_dt
        q = goal[3:].copy() if ang <= rot_step else _slerp(q, goal[3:], rot_step / ang)
        return np.concatenate([p, q])

    def _set_target(self, p: np.ndarray, q: np.ndarray) -> None:
        with self._lock:
            self._servo_goal = None
            self._target_is_joint = False
            self._target = np.concatenate([p, q / (np.linalg.norm(q) + 1e-12)])

    def _set_joint_target(self, q: np.ndarray) -> None:
        with self._lock:
            self._servo_goal = None
            self._target_is_joint = True
            self._target = np.asarray(q, float).copy()

    def _latch_hold(self) -> None:
        # Stop the stream so the controller latches a rigid hold on the measured
        # joints (its documented behaviour after command_timeout). Feeding a soft
        # servo a static setpoint forever makes it hunt and visibly shake, so
        # ServoJ is stopped once settled.
        with self._lock:
            self._servo_goal = None
            self._target = None

    def _clamp_lead(self, target: np.ndarray) -> np.ndarray:
        """Bound the streamed target to a small lead over the measured pose."""
        measured = self._measured
        if measured is None:
            return target
        p_m, q_m = measured
        p_d, q_d = target[:3], target[3:]
        dp = p_d - p_m
        n = float(np.linalg.norm(dp))
        if n > _MAX_LEAD_POS_M:
            p_d = p_m + dp * (_MAX_LEAD_POS_M / n)
        ang = _quat_angle(q_m, q_d)
        if ang > _MAX_LEAD_ROT_RAD:
            q_d = _slerp(q_m, q_d, _MAX_LEAD_ROT_RAD / ang)
        return np.concatenate([p_d, q_d])

    def _wait_pose(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        self._measured_evt.clear()
        if self._pose_api_unavailable:
            try:
                self._send("request_get_yg_arm_state")
            except Exception:
                return None
        if self._measured_evt.wait(self._feedback_timeout_s) \
                and self._pose_error is None:
            return self._measured
        if self._pose_api_unavailable:
            self._measured_evt.clear()
            try:
                self._send("request_get_yg_arm_state")
            except Exception:
                return None
            if self._measured_evt.wait(self._feedback_timeout_s) \
                    and self._pose_error is None:
                return self._measured
        return None

    def _pose_from_joint_state(
        self, q: np.ndarray, quiet: bool = False
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        # quiet suppresses the "pose api unavailable" notice: an explicit FK
        # request (e.g. deriving prepare_ee_pose) is not a fallback and must not claim one.
        try:
            if self._fk_robot is None:
                from pyAgxArm import (
                    AgxArmFactory,
                    ArmModel,
                    PiperFW,
                    create_agx_arm_config,
                )

                model = getattr(ArmModel, self._arm_model)
                fk_cfg = create_agx_arm_config(
                    robot=model,
                    firmeware_version=PiperFW.DEFAULT,
                    channel="can0",
                )
                self._fk_robot = AgxArmFactory.create_arm(fk_cfg)
            flange = np.asarray(self._fk_robot.fk(q.tolist()), float).reshape(-1)
            if flange.size != 6:
                self._pose_error = "joint-state FK returned an invalid pose"
                return None
            if not quiet and not self._fk_notice_printed:
                print(
                    "[Tron2Driver] request_get_yg_arm_pose unavailable; "
                    f"using request_get_yg_arm_state + {self._arm_model} FK"
                )
                self._fk_notice_printed = True
            return flange[:3], _rpy_to_quat(*flange[3:])
        except Exception as exc:
            self._pose_error = f"joint-state FK failed: {exc}"
            if not quiet and not self._fk_warning_printed:
                print(f"[Tron2Driver] {self._pose_error}")
                self._fk_warning_printed = True
            return None

    def _wait_joints(self) -> Optional[np.ndarray]:
        self._joint_evt.clear()
        try:
            self._send("request_get_yg_arm_state")
        except Exception:
            return self._measured_q
        self._joint_evt.wait(self._feedback_timeout_s)
        return self._measured_q

    def _wait_joint_arrival(self, q_goal: np.ndarray, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        prev: Optional[np.ndarray] = None
        settled = 0
        err = float("nan")
        last_q: Optional[np.ndarray] = None
        while time.monotonic() < deadline and not self._stop.is_set():
            q = self._wait_joints()
            if q is not None:
                last_q = q
                err = float(np.max(np.abs(q - q_goal)))
                if err <= _JOINT_TOL_RAD:
                    return True
                if prev is not None and float(np.max(np.abs(q - prev))) < _JOINT_SETTLE_RAD:
                    settled += 1
                else:
                    settled = 0
                prev = q
                if settled >= _SETTLE_SAMPLES:
                    print(f"[Tron2Driver] Joint target unreachable: "
                          f"{math.degrees(err):.0f}deg away{_joint_gap_text(last_q, q_goal)}")
                    return False
            time.sleep(self._feedback_dt)
        print(f"[Tron2Driver] Joint motion timed out {math.degrees(err):.0f}deg away"
              f"{_joint_gap_text(last_q, q_goal)}")
        return False

    def _wait_arrival(self, goal_p: np.ndarray, goal_q: np.ndarray, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        prev: Optional[tuple[np.ndarray, np.ndarray]] = None
        settled = 0
        ep = er = float("nan")
        while time.monotonic() < deadline and not self._stop.is_set():
            pose = self._wait_pose()
            if pose is not None:
                p, q = pose
                ep = float(np.linalg.norm(goal_p - p))
                er = _quat_angle(q, goal_q)
                if ep <= _POS_TOL_M and er <= _ROT_TOL_RAD:
                    return True
                if prev is not None \
                        and float(np.linalg.norm(p - prev[0])) < _SETTLE_POS_M \
                        and _quat_angle(q, prev[1]) < _SETTLE_ROT_RAD:
                    settled += 1
                else:
                    settled = 0
                prev = (p, q)
                # An unreachable target is never rejected: the arm stalls against
                # a joint limit while the controller still reports success.
                if settled >= _SETTLE_SAMPLES:
                    print(f"[Tron2Driver] Target unreachable: settled "
                          f"{ep * 1000:.0f}mm / {math.degrees(er):.0f}deg away")
                    return False
            time.sleep(self._feedback_dt)
        print(f"[Tron2Driver] Motion timed out {ep * 1000:.0f}mm / "
              f"{math.degrees(er):.0f}deg away")
        return False
