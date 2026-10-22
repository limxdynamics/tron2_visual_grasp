"""AgileX Piper / PiperX arm driver built on the pyAgxArm SDK.

The SDK owns CAN transport, kinematics, and Cartesian planning; this module only
maps it onto the RobotDriver contract used by the vision grasping scripts.

pyAgxArm reports and accepts poses as [x, y, z, roll, pitch, yaw] in meters and
radians with a ZYX RPY convention, which already matches this project.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .base import RobotDriver

try:
    from ...utils.transforms import pose6d_to_mat4
except ImportError:
    from utils.transforms import pose6d_to_mat4


# ArmMsgFeedbackStatus.motion_status: target reached.
_MOTION_REACHED = 0x00

# ArmMsgFeedbackStatus.arm_status values worth reporting when a motion times out.
# The controller latches these until the next successful motion, so they are only
# diagnostics; the official wait pattern keys off motion_status alone.
_ARM_FAULTS = {
    0x01: "emergency stop",
    0x02: "no IK solution",
    0x03: "singularity",
    0x04: "target exceeds joint limit",
    0x05: "joint communication error",
    0x06: "joint brake not released",
    0x07: "collision detected",
}

# Time the controller needs to apply a leader/follower mode message.
_MODE_SETTLE_S = 0.3

# move_p accepts pitch only within this range.
_MAX_PITCH_RAD = np.pi / 2.0

_DEFAULTS = {
    "model": "piper_x",
    "firmware": "default",
    "interface": "socketcan",
    "channel": "can0",
    "bitrate": 1000000,
    "speed_percent": 50,
    "tcp_offset_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "enable_timeout_s": 10.0,
    "motion_settle_s": 0.5,
    "motion_timeout_extra_s": 3.0,
    "feedback_timeout_s": 2.0,
}

_GRIPPER_DEFAULTS = {
    "max_distance_m": 0.07,
    "default_force_n": 1.0,
    "max_force_n": 3.0,
    "grasp_min_width_m": 0.004,
    "settle_s": 0.8,
}


class PiperDriver(RobotDriver):
    """Piper / PiperX arm with an AgxGripper end effector."""

    def __init__(self, robot_cfg: Optional[dict[str, Any]] = None) -> None:
        robot_cfg = robot_cfg or {}
        cfg = {**_DEFAULTS, **(robot_cfg.get("piper") or {})}
        gcfg = {**_GRIPPER_DEFAULTS, **((robot_cfg.get("gripper") or {}).get("piper") or {})}

        self._model = str(cfg["model"])
        self._firmware = str(cfg["firmware"])
        self._interface = str(cfg["interface"])
        self._channel = str(cfg["channel"])
        self._bitrate = int(cfg["bitrate"])
        self._speed_percent = int(np.clip(int(cfg["speed_percent"]), 1, 100))
        self._tcp_offset = [float(v) for v in cfg["tcp_offset_m"]]
        if len(self._tcp_offset) != 6:
            raise ValueError("robot.piper.tcp_offset_m must have 6 values [x,y,z,roll,pitch,yaw]")
        self._enable_timeout_s = float(cfg["enable_timeout_s"])
        self._motion_settle_s = float(cfg["motion_settle_s"])
        self._motion_timeout_extra_s = float(cfg["motion_timeout_extra_s"])
        self._feedback_timeout_s = float(cfg["feedback_timeout_s"])
        self._park_pose = robot_cfg.get("park_pose")

        self.MAX_DISTANCE_M = float(gcfg["max_distance_m"])
        self._default_force = float(gcfg["default_force_n"])
        self._max_force = float(gcfg["max_force_n"])
        self._grasp_min_width_m = float(gcfg["grasp_min_width_m"])
        self._gripper_settle_s = float(gcfg["settle_s"])

        self._arm: Any = None
        self._gripper: Any = None
        self._running = False

    # Lifecycle

    def start(self) -> None:
        if self._running:
            return
        try:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config
        except ImportError as e:
            raise RuntimeError(
                f"pyAgxArm is not installed: {e}\n"
                "  pip install 'git+https://github.com/agilexrobotics/pyAgxArm.git'"
            ) from e

        config = create_agx_arm_config(
            robot=self._model,
            firmeware_version=self._firmware,  # SDK spells the keyword this way
            interface=self._interface,
            channel=self._channel,
            bitrate=self._bitrate,
        )
        self._arm = AgxArmFactory.create_arm(config)
        # The effector must be created before connect() so its frames get parsed.
        self._gripper = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._arm.connect()

        self._wait_enabled()
        # Leader zero-force drag is stored in the controller and outlives the process;
        # follower is the plain controlled state and must be restored after enable(),
        # because the controller ignores config messages while the arm is disabled.
        self._leave_leader_mode()
        self._arm.set_speed_percent(self._speed_percent)
        self._arm.set_tcp_offset(self._tcp_offset)
        self._running = True

        self._warn_on_firmware_mismatch()
        print(
            f"[PiperDriver] Ready ({self._model} on {self._interface}:{self._channel}, "
            f"speed={self._speed_percent}%)"
        )

    def end(self, park: bool = True) -> None:
        if self._arm is None:
            return
        # No disable()/reset() here: both cut power and a raised arm would drop.
        if self._running and park:
            try:
                self.move_park()
            except Exception as exc:
                print(f"[PiperDriver] Park failed: {exc}")
        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._running = False
        self._arm = None
        self._gripper = None

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
        if abs(float(pitch)) > _MAX_PITCH_RAD:
            print(f"[PiperDriver] Pitch {pitch:+.3f} rad is outside the +/-pi/2 range accepted by move_p")
            return False

        tcp_pose = [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]
        flange_pose = self._arm.get_tcp2flange_pose(tcp_pose)
        self._arm.move_p(flange_pose)
        return self._wait_motion_done(duration)

    def get_tcp_pose(self) -> np.ndarray:
        self._ensure_running()
        msg = self._wait_feedback(self._arm.get_tcp_pose, "TCP pose")
        x, y, z, roll, pitch, yaw = msg.msg
        return pose6d_to_mat4(x, y, z, roll, pitch, yaw)

    def move_park(self) -> bool:
        if not self._park_pose:
            return False
        return self.move_ready(self._park_pose)

    # Gripper

    def set_gripper(self, distance_m: float, force: Optional[float] = None) -> None:
        self._ensure_running()
        width = float(np.clip(float(distance_m), 0.0, self.MAX_DISTANCE_M))
        self._gripper.move_gripper_m(value=width, force=self._clamp_force(force))

    def open_gripper(self, distance_m: Optional[float] = None, timeout: float = 3.0) -> None:
        target = self.MAX_DISTANCE_M if distance_m is None else distance_m
        self.set_gripper(target)
        time.sleep(min(self._gripper_settle_s, max(0.0, timeout)))

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        self.set_gripper(0.0, force=force)
        time.sleep(min(self._gripper_settle_s, max(0.0, timeout)))
        # The jaws stall on the payload, so a residual opening means we hold something.
        width = self.get_gripper_width()
        if width is None:
            print("[PiperDriver] No gripper feedback; treating as an empty grasp")
            return False
        return width > self._grasp_min_width_m

    def release_gripper(self, timeout: float = 4.0) -> None:
        self.open_gripper(timeout=min(2.0, timeout))
        self.set_gripper(0.0)

    def get_gripper_width(self) -> Optional[float]:
        """Return the current jaw opening in meters, or None when feedback is missing."""
        self._ensure_running()
        status = self._gripper.get_gripper_status()
        if status is None or status.msg.mode != "width":
            return None
        return float(status.msg.value)

    # Hand guiding

    def enter_drag_mode(self) -> None:
        self._ensure_running()
        self._arm.set_leader_mode()

    def exit_drag_mode(self) -> None:
        self._ensure_running()
        self._leave_leader_mode()
        angles = self._arm.get_joint_angles()
        if angles is not None:
            # Commanding the measured angles leaves zero-force mode without moving the arm.
            self._arm.move_j(list(angles.msg))

    # Internals

    def _ensure_running(self) -> None:
        if not self._running:
            raise RuntimeError("PiperDriver is not started; call start() first")

    def _clamp_force(self, force: Optional[float]) -> float:
        value = self._default_force if force is None else float(force)
        return float(np.clip(abs(value), 0.0, self._max_force))

    def _leave_leader_mode(self) -> None:
        """Return to follower (plain controlled) mode; linkage_config 0x00 is a no-op."""
        self._arm.set_follower_mode()
        time.sleep(_MODE_SETTLE_S)

    def _wait_enabled(self) -> None:
        deadline = time.monotonic() + self._enable_timeout_s
        while time.monotonic() < deadline:
            if self._arm.enable():
                return
            time.sleep(0.01)
        raise RuntimeError(
            f"Piper arm did not enable within {self._enable_timeout_s:.0f}s\n"
            f"  Check CAN bring-up on {self._channel}, arm power, and joint error codes."
        )

    def _wait_feedback(self, getter: Any, what: str) -> Any:
        deadline = time.monotonic() + self._feedback_timeout_s
        while time.monotonic() < deadline:
            msg = getter()
            if msg is not None:
                return msg
            time.sleep(0.02)
        raise RuntimeError(f"Piper {what} feedback is not ready")

    def _wait_motion_done(self, duration: float) -> bool:
        # Let the controller latch the new target before polling, otherwise the
        # previous motion still reports "reached".
        time.sleep(self._motion_settle_s)
        deadline = time.monotonic() + float(duration) + self._motion_timeout_extra_s
        status = None
        while time.monotonic() < deadline:
            status = self._arm.get_arm_status()
            if status is not None and int(status.msg.motion_status) == _MOTION_REACHED:
                return True
            time.sleep(0.05)
        print(
            "[PiperDriver] Motion did not report arrival before the timeout"
            f"{self._fault_hint(status)}"
        )
        return False

    @staticmethod
    def _fault_hint(status: Any) -> str:
        if status is None:
            return ""
        parts = []
        fault = _ARM_FAULTS.get(int(status.msg.arm_status))
        if fault:
            parts.append(fault)
        past_limit = [
            str(i)
            for i in range(1, 7)
            if getattr(status.msg.err_status, f"joint_{i}_angle_limit", False)
        ]
        if past_limit:
            parts.append(f"joints {','.join(past_limit)} past angle limit")
        return f" (arm reports: {'; '.join(parts)})" if parts else ""

    def _warn_on_firmware_mismatch(self) -> None:
        try:
            from pyAgxArm import resolve_firmware_profile

            info = self._arm.get_firmware()
            if not info:
                return
            version = info.get("software_version")
            expected = resolve_firmware_profile(self._model, version)
        except Exception:
            return
        if expected and expected != self._firmware:
            print(
                f"[PiperDriver] Firmware {version} expects robot.piper.firmware "
                f"'{expected}', config says '{self._firmware}'"
            )
