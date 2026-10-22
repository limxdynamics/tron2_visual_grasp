"""Intel RealSense camera driver.

RealsenseCamera:
  open() / close(): manage the pyrealsense2 pipeline.
  get_frame(): return color-aligned BGR and depth-mm frames.
  K / D: camera matrix and distortion coefficients.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .base import CameraDriver, CameraFrameError


class RealsenseCamera(CameraDriver):
    """Intel RealSense RGB-D camera driver."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        calib_dir: Optional[str] = None,
        serial: Optional[str] = None,
        model: Optional[str] = None,
        auto_exposure: bool = True,
        exposure_us: Optional[float] = None,
    ) -> None:
        self._w = width
        self._h = height
        self._fps = fps
        self._calib_dir = Path(calib_dir) if calib_dir else None
        self._serial = str(serial) if serial else None
        self._model = (model or "").lower()
        self._auto_exposure = bool(auto_exposure)
        self._exposure_us = float(exposure_us) if exposure_us is not None else None

        self._pipeline = None
        self._align = None
        self._rs = None
        self._color_sensor = None
        self._depth_scale_mm: float = 1.0   # raw uint16 to mm
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._aruco = None
        self._reset_frame_failures()

    # Lifecycle

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(f"pyrealsense2 is not installed: {e}") from e

        errors = []
        serial = self._resolve_serial(rs)
        for width, height, fps in self._profile_candidates():
            pipeline = rs.pipeline()
            config = rs.config()
            if serial:
                # Without this librealsense grabs whichever device enumerates first,
                # which silently picks the wrong camera on a multi-camera host.
                config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                profile = pipeline.start(config)
                self._pipeline = pipeline
                self._align = rs.align(rs.stream.color)
                self._w, self._h, self._fps = width, height, fps
                self._reset_frame_failures()
                break
            except RuntimeError as e:
                errors.append(f"{width}x{height}@{fps}: {e}")
                try:
                    pipeline.stop()
                except Exception:
                    pass
        else:
            raise RuntimeError(
                "RealSense camera not found or no RGB-D profile is available:\n  "
                + "\n  ".join(errors)
                + "\n  Check USB connection or other camera clients."
            )

        # Intrinsics
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        cx = getattr(intr, "ppx", getattr(intr, "cx", None))
        cy = getattr(intr, "ppy", getattr(intr, "cy", None))
        if cx is None or cy is None:
            raise RuntimeError(f"Invalid RealSense intrinsics: {intr!r}")
        self._K = np.array([
            [intr.fx, 0,        cx],
            [0,        intr.fy, cy],
            [0,        0,       1      ],
        ], dtype=np.float64)

        # raw uint16 * depth_scale (m/unit) * 1000 -> mm
        dev = profile.get_device()
        ds = dev.first_depth_sensor().get_depth_scale()
        self._depth_scale_mm = ds * 1000.0
        self._rs = rs
        self._color_sensor = self._find_color_sensor(rs, dev)
        exposure_status = self._configure_color_exposure()

        self._D = self._load_distortion()
        print(f"[RealsenseCamera] Ready ({dev.get_info(rs.camera_info.name)} "
              f"SN={dev.get_info(rs.camera_info.serial_number)}, "
              f"{intr.width}x{intr.height}, depth_scale={ds:.6f} m/unit, "
              f"exposure={exposure_status})")

    def _resolve_serial(self, rs) -> Optional[str]:
        """Serial of the device to open, from camera.serial or camera.type."""
        # Create devices one by one: a wedged unit (e.g. the robot's chest D435i
        # failing set_xu with EPROTO) must not hide the healthy wrist camera.
        device_list = rs.context().query_devices()
        devices = []
        for index in range(device_list.size()):
            try:
                device = device_list[index]
                devices.append((
                    device.get_info(rs.camera_info.name),
                    device.get_info(rs.camera_info.serial_number),
                ))
            except RuntimeError as e:
                print(f"[RealsenseCamera] Skipping unreadable RealSense device #{index}: {e}")
        if self._serial:
            if not any(sn == self._serial for _, sn in devices):
                raise RuntimeError(
                    f"RealSense serial {self._serial} not connected. Found: "
                    + (", ".join(f"{n} ({s})" for n, s in devices) or "none")
                )
            return self._serial

        wanted = "d405" if "d405" in self._model else None
        if wanted is None:
            return None
        matches = [(n, s) for n, s in devices if wanted in n.lower().replace(" ", "")]
        if not matches:
            raise RuntimeError(
                f"No RealSense {wanted.upper()} connected (camera.type={self._model!r}). Found: "
                + (", ".join(f"{n} ({s})" for n, s in devices) or "none")
                + "\n  Fix camera.type or set camera.serial in config/default.yaml."
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Several RealSense {wanted.upper()} units connected; set camera.serial. "
                + ", ".join(f"{n} ({s})" for n, s in matches)
            )
        return matches[0][1]

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._color_sensor = None

    # Frames

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._pipeline is None:
            return None, None
        try:
            frames = self._pipeline.wait_for_frames(500)
            aligned = self._align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                self._record_frame_failure("missing color or depth frame")
                return None, None

            color_bgr = np.asanyarray(cf.get_data())
            depth_raw = np.asanyarray(df.get_data())   # uint16, depth units
            depth_mm  = (depth_raw * self._depth_scale_mm).astype(np.uint16)
            self._reset_frame_failures()
            return color_bgr, depth_mm
        except CameraFrameError:
            raise
        except RuntimeError as exc:
            self._record_frame_failure(str(exc))
            return None, None

    # Intrinsics

    @property
    def K(self) -> np.ndarray:
        if self._K is None:
            raise RuntimeError("Camera is not open")
        return self._K

    @property
    def D(self) -> np.ndarray:
        if self._D is None:
            raise RuntimeError("Camera is not open")
        return self._D

    # Exposure

    def get_exposure_controls(self) -> Optional[dict]:
        sensor = self._require_color_sensor()
        exposure = self._rs.option.exposure
        automatic = self._rs.option.enable_auto_exposure
        if not sensor.supports(exposure) or not sensor.supports(automatic):
            return None
        limits = sensor.get_option_range(exposure)
        return {
            "auto": bool(sensor.get_option(automatic)),
            "value": float(sensor.get_option(exposure)),
            "min": float(limits.min), "max": float(limits.max),
            "step": float(limits.step) or 1.0,
        }

    @property
    def auto_exposure(self) -> bool:
        return self._auto_exposure

    @property
    def exposure_us(self) -> Optional[float]:
        return self._exposure_us

    @property
    def exposure_range_us(self) -> Tuple[float, float]:
        sensor = self._require_color_sensor()
        limits = sensor.get_option_range(self._rs.option.exposure)
        return float(limits.min), float(limits.max)

    def set_auto_exposure(self, enabled: bool) -> None:
        sensor = self._require_color_sensor()
        option = self._rs.option.enable_auto_exposure
        if not sensor.supports(option):
            raise RuntimeError("RealSense color sensor does not support automatic exposure control")
        sensor.set_option(option, 1.0 if enabled else 0.0)
        self._auto_exposure = bool(enabled)

    def set_exposure_us(self, exposure_us: float) -> None:
        sensor = self._require_color_sensor()
        option = self._rs.option.exposure
        if not sensor.supports(option):
            raise RuntimeError("RealSense color sensor does not support manual exposure control")
        value = float(exposure_us)
        low, high = self.exposure_range_us
        if not low <= value <= high:
            raise ValueError(
                f"exposure_us={value:g} is outside the supported range {low:g}..{high:g} us"
            )
        if self._auto_exposure:
            # librealsense keeps overriding the manual value while auto exposure runs.
            self.set_auto_exposure(False)
        sensor.set_option(option, value)
        self._exposure_us = value

    # Internals

    def _find_color_sensor(self, rs, device):
        sensor = next(
            (
                candidate
                for candidate in device.query_sensors()
                if any(profile.stream_type() == rs.stream.color
                       for profile in candidate.get_stream_profiles())
            ),
            None,
        )
        if sensor is None:
            raise RuntimeError("RealSense color stream sensor is unavailable; cannot configure exposure")
        return sensor

    def _require_color_sensor(self):
        if self._rs is None or self._color_sensor is None:
            raise RuntimeError("Camera is not open")
        return self._color_sensor

    def _configure_color_exposure(self) -> str:
        self.set_auto_exposure(self._auto_exposure)
        if self._auto_exposure:
            return "auto"
        if self._exposure_us is None:
            raise ValueError("camera.exposure_us is required when camera.auto_exposure is false")
        self.set_exposure_us(self._exposure_us)
        return f"manual {self._exposure_us:g} us"

    def _load_distortion(self) -> np.ndarray:
        if self._calib_dir is not None:
            npz_path = self._calib_dir / "intrinsics.npz"
            if npz_path.exists():
                try:
                    data = np.load(str(npz_path))
                    D = data["dist_coeffs"].flatten()
                    if abs(D[0]) > 5.0:
                        print(f"[RealsenseCamera] Invalid k1={D[0]:.2f}; using zero distortion")
                        return np.zeros((1, 5), dtype=np.float64)
                    return D.reshape(1, -1)
                except Exception:
                    pass
        return np.zeros((1, 5), dtype=np.float64)

    def _profile_candidates(self) -> list[tuple[int, int, int]]:
        requested = (int(self._w), int(self._h), int(self._fps))
        candidates = [
            requested,
            (1280, 720, 15),
            (848, 480, 30),
            (640, 480, 30),
            (640, 480, 15),
        ]
        unique = []
        seen = set()
        for item in candidates:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique
