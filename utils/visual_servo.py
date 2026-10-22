"""Visual-servo control used to drive the base up to a target before grasping."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .targets import Target


@dataclass(frozen=True)
class ApproachControlConfig:
    target_distance_m: float = 0.45
    distance_tolerance_m: float = 0.06
    center_tolerance: float = 0.10
    center_offset: float = 0.0
    forward_gain: float = 0.4
    yaw_gain: float = 0.9
    min_forward_mps: float = 0.2
    max_forward_mps: float = 0.8
    max_yaw_rps: float = 0.3
    creep_forward_mps: float = 0.12
    search_yaw_rps: float = 0.0
    arrive_frames: int = 5
    commit_distance_m: float = 0.9
    lost_frames: int = 5
    # Keep the last forward speed (yaw zeroed) through this many missed frames
    # instead of stopping on every detection dropout; 0 stops at once.
    hold_lost_frames: int = 0
    stop_bottom_ratio: Optional[float] = None
    edge_margin_px: int = 12
    # Commands below this do not move the base at all, so a correction is either
    # lifted clear of it or dropped to zero; anything in between is a lie.
    deadzone_mps: float = 0.0
    yaw_deadzone_rps: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ApproachControlConfig":
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for name, default in vars(defaults).items():
            if name not in values:
                continue
            value = values[name]
            if name == "stop_bottom_ratio":
                kwargs[name] = None if value is None else float(value)
            else:
                kwargs[name] = type(default)(value)
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        if self.target_distance_m <= 0.0:
            raise ValueError("target_distance_m must be positive")
        if self.distance_tolerance_m <= 0.0:
            raise ValueError("distance_tolerance_m must be positive")
        if not 0.0 < self.center_tolerance <= 1.0:
            raise ValueError("center_tolerance must be in (0, 1]")
        if not -1.0 < self.center_offset < 1.0:
            raise ValueError("center_offset must be in (-1, 1)")
        if not 0.0 <= self.min_forward_mps <= self.max_forward_mps:
            raise ValueError("min_forward_mps must be between zero and max_forward_mps")
        if self.max_yaw_rps < 0.0:
            raise ValueError("max_yaw_rps cannot be negative")
        if self.yaw_deadzone_rps > self.max_yaw_rps:
            raise ValueError("yaw_deadzone_rps cannot exceed max_yaw_rps")
        if self.arrive_frames < 1 or self.lost_frames < 1:
            raise ValueError("arrive_frames and lost_frames must be positive")
        if self.hold_lost_frames < 0:
            raise ValueError("hold_lost_frames cannot be negative")
        if self.stop_bottom_ratio is not None and not 0.0 < self.stop_bottom_ratio <= 1.0:
            raise ValueError("stop_bottom_ratio must be in (0, 1]")
        if self.edge_margin_px < 0:
            raise ValueError("edge_margin_px cannot be negative")


def clear_deadzone(value: float, deadzone: float) -> float:
    """Lift a non-zero command past the dead zone, leaving zero alone."""
    if value == 0.0 or deadzone <= 0.0:
        return value
    return float(np.copysign(max(abs(value), deadzone), value))


def aim_column_px(frame_width: int, config: ApproachControlConfig) -> float:
    """Image column the robot steers onto; center_offset cancels camera mounting bias."""
    return (frame_width / 2.0) * (1.0 + config.center_offset)


def horizontal_error(target: Target, frame_width: int, config: ApproachControlConfig) -> float:
    """Normalised offset of the target from the aim column; positive means right."""
    return (target.center_px[0] - aim_column_px(frame_width, config)) / (frame_width / 2.0)


def compute_approach_twist(
    target: Optional[Target],
    frame_width: int,
    config: ApproachControlConfig,
    frame_height: Optional[int] = None,
) -> tuple[tuple[float, float, float], bool]:
    """Return ((forward, lateral, yaw), arrived) for one camera frame."""
    if target is None:
        yaw = float(np.clip(config.search_yaw_rps, -config.max_yaw_rps, config.max_yaw_rps))
        return (0.0, 0.0, clear_deadzone(yaw, config.yaw_deadzone_rps)), False
    if frame_width <= 0:
        raise ValueError("frame_width must be positive")

    h_error = horizontal_error(target, frame_width, config)
    # Inside the tolerance the aim is good enough; outside it the gain alone can
    # ask for a turn slower than the base will make, so it is lifted clear.
    if abs(h_error) <= config.center_tolerance:
        yaw = 0.0
    else:
        yaw = float(np.clip(-config.yaw_gain * h_error, -config.max_yaw_rps, config.max_yaw_rps))
        yaw = clear_deadzone(yaw, config.yaw_deadzone_rps)
    if config.stop_bottom_ratio is not None:
        if frame_height is None or frame_height <= 0:
            raise ValueError("frame_height must be positive in 2D approach mode")
        effective_ratio = min(
            config.stop_bottom_ratio,
            max(0.0, (frame_height - config.edge_margin_px) / float(frame_height)),
        )
        bottom_ratio = target.detection.bbox_xyxy[3] / float(frame_height)
        if bottom_ratio >= effective_ratio or config.creep_forward_mps == 0.0:
            forward = 0.0
        else:
            # Scale with the gap still left to the stop line so the base coasts
            # in instead of running flat out into it.
            remaining = (effective_ratio - bottom_ratio) / effective_ratio
            forward = config.min_forward_mps + remaining * (
                config.max_forward_mps - config.min_forward_mps
            )
        arrived = abs(h_error) < config.center_tolerance and bottom_ratio >= effective_ratio
    elif target.depth_m is None:
        forward = config.creep_forward_mps
        arrived = False
    else:
        distance_error = target.depth_m - config.target_distance_m
        if abs(distance_error) <= config.distance_tolerance_m:
            forward = 0.0
        else:
            proportional = config.forward_gain * distance_error
            forward = float(np.copysign(max(abs(proportional), config.min_forward_mps), proportional))
        arrived = (
            abs(h_error) < config.center_tolerance
            and abs(distance_error) < config.distance_tolerance_m
        )

    forward = float(np.clip(forward, -config.max_forward_mps, config.max_forward_mps))
    return (clear_deadzone(forward, config.deadzone_mps), 0.0, yaw), arrived


class ArrivalTracker:
    """Require stable arrival, and accept losing a target the camera is already on top of."""

    def __init__(self, config: ApproachControlConfig) -> None:
        self._config = config
        self.stable_frames = 0
        self.lost_frames = 0
        self.last_depth_m: Optional[float] = None
        self._arrival_latched = False

    def update(self, target: Optional[Target], arrived: bool) -> tuple[bool, str]:
        if target is None:
            self.lost_frames += 1
        else:
            self.lost_frames = 0
            if target.depth_m is not None:
                self.last_depth_m = target.depth_m

        if self._config.stop_bottom_ratio is not None:
            if arrived:
                self._arrival_latched = True
            elif target is not None:
                self._arrival_latched = False
            evidence = arrived or (target is None and self._arrival_latched)
        else:
            evidence = arrived
        self.stable_frames = self.stable_frames + 1 if evidence else 0
        if self.stable_frames >= self._config.arrive_frames:
            return True, "stable"
        if (
            self._config.stop_bottom_ratio is None
            and target is None
            and self.last_depth_m is not None
            and self.last_depth_m <= self._config.commit_distance_m
            and self.lost_frames >= self._config.lost_frames
        ):
            return True, "near-field target loss"
        return False, ""


class BaseTwistStreamer:
    """Keep resending the latest base command faster than the controller timeout."""

    def __init__(self, robot: Any, frequency_hz: float, enabled: bool) -> None:
        self._robot = robot
        self._period = 1.0 / max(float(frequency_hz), 10.0)
        self._enabled = enabled
        self._command = (0.0, 0.0, 0.0)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._run, name="base-twist", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set(self, command: tuple[float, float, float]) -> None:
        with self._lock:
            self._command = tuple(float(value) for value in command)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Base twist streaming failed: {self._error}") from self._error

    def close(self) -> None:
        self.set((0.0, 0.0, 0.0))
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._enabled:
            for _ in range(3):
                try:
                    self._robot.set_base_twist(0.0, 0.0, 0.0)
                except Exception:
                    break
                time.sleep(self._period)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._enabled:
                with self._lock:
                    command = self._command
                try:
                    self._robot.set_base_twist(*command)
                except Exception as exc:
                    self._error = exc
                    self._stop.set()
                    return
            self._stop.wait(self._period)
