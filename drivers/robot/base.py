"""Shared robot arm interface.

RobotDriver:
  start() / end(): connect and enable, or park and release the arm.
  move_to_pose(): blocking Cartesian TCP move; False when the target is unreachable.
  get_tcp_pose(): current TCP pose as a 4x4 matrix (T_tcp2base).
  move_joints() / move_park(): optional joint-space moves.
  set_gripper() / open_gripper() / grasp() / release_gripper(): jaw control.

Units are meters and radians everywhere; orientations are intrinsic ZYX
(roll, pitch, yaw), matching utils.transforms.pose6d_to_mat4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class RobotDriver(ABC):
    """Minimal arm and gripper contract used by the vision grasping scripts."""

    MAX_DISTANCE_M: float = 0.09

    # Lifecycle

    @abstractmethod
    def start(self) -> None:
        """Connect, enable the arm, and make the gripper ready."""

    @abstractmethod
    def end(self, park: bool = True) -> None:
        """Park the arm if possible and release the connection. Safe to call twice.

        Set park=False when the caller only borrowed the connection and must not
        move the arm.
        """

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """True once start() succeeded and end() has not run."""

    # Motion

    @abstractmethod
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
        """Move the TCP to a base-frame pose and block until it arrives.

        Args:
            x, y, z: target translation in meters.
            roll, pitch, yaw: intrinsic ZYX Euler angles in radians.
            duration: expected motion time in seconds, used to size the wait.

        Returns:
            True when the target is reached; False when it is unreachable or times out.
        """

    @abstractmethod
    def get_tcp_pose(self) -> np.ndarray:
        """Return the current TCP pose as a (4, 4) T_tcp2base matrix."""

    def move_ready(self, ready_cfg: dict[str, Any]) -> bool:
        """Move to the configured ready pose."""
        return self.move_to_pose(
            x=float(ready_cfg.get("x", 0.25)),
            y=float(ready_cfg.get("y", 0.0)),
            z=float(ready_cfg.get("z", 0.35)),
            roll=float(ready_cfg.get("roll", 0.0)),
            pitch=float(ready_cfg.get("pitch", 1.2)),
            yaw=float(ready_cfg.get("yaw", 0.0)),
            duration=float(ready_cfg.get("duration", 3.0)),
        )

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
        """Set a streamed TCP goal without blocking; the driver slews toward it at the
        given speeds. Call stop_servo() when done."""
        raise NotImplementedError(f"{type(self).__name__} does not support streamed servoing")

    def stop_servo(self) -> None:
        """End a servo_pose() stream and hold the current pose."""
        raise NotImplementedError(f"{type(self).__name__} does not support streamed servoing")

    # Joint space

    def move_joints(self, q_target: Any, duration: float = 0.0) -> bool:
        """Blocking joint-space move to q_target (6,) radians; duration is a minimum."""
        raise NotImplementedError(f"{type(self).__name__} does not support joint moves")

    def move_park(self) -> bool:
        """Fold the arm to its stowed pose before the body moves or shuts down.

        False means the driver has no stowed pose configured, so the caller can
        tell "folded" apart from "left exactly where it was".
        """
        return False

    # Mobile base and body posture, used by whole-body task pipelines

    def kneel(self) -> None:
        """Request the kneeling posture and wait for a successful response."""
        raise NotImplementedError(f"{type(self).__name__} does not support kneeling")

    def stand(self) -> None:
        """Request the standing posture and wait for a successful response."""
        raise NotImplementedError(f"{type(self).__name__} does not support standing")

    def set_base_twist(self, x: float, y: float, yaw: float) -> None:
        """Command mobile-base forward, lateral, and yaw velocities."""
        raise NotImplementedError(f"{type(self).__name__} does not support mobile-base control")

    def set_body_pitch(self, pitch: float) -> None:
        """Set the body pitch in radians."""
        raise NotImplementedError(f"{type(self).__name__} does not support body-pitch control")

    def reset_body_pitch(self) -> None:
        """Level the body again."""
        raise NotImplementedError(f"{type(self).__name__} does not support body-pitch control")

    # Gripper

    @abstractmethod
    def set_gripper(self, distance_m: float, force: Optional[float] = None) -> None:
        """Command a jaw opening in meters without waiting for it to settle."""

    @abstractmethod
    def open_gripper(self, distance_m: Optional[float] = None, timeout: float = 3.0) -> None:
        """Open the jaws to a distance in meters and wait for them to settle."""

    @abstractmethod
    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        """Close the jaws with force control.

        Returns:
            True when the jaws stopped on an object; False on an empty grasp.
        """

    @abstractmethod
    def release_gripper(self, timeout: float = 4.0) -> None:
        """Open the jaws to drop the payload and return them to rest."""

    # Hand guiding, used by hand-eye calibration in manual mode

    def enter_drag_mode(self) -> None:
        """Let the operator move the arm by hand."""
        raise NotImplementedError(f"{type(self).__name__} does not support hand guiding")

    def exit_drag_mode(self) -> None:
        """Leave hand-guiding and hold the current pose."""
        raise NotImplementedError(f"{type(self).__name__} does not support hand guiding")
