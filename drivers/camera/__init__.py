from __future__ import annotations

from pathlib import Path

from .base import CameraDriver, CameraFrameError
from .realsense import RealsenseCamera

__all__ = [
    "CameraDriver",
    "CameraFrameError",
    "RealsenseCamera",
    "make_camera",
]


def make_camera(cfg: dict) -> CameraDriver:
    """Create a camera driver from config/default.yaml."""
    cam_cfg  = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "").lower()
    w   = cam_cfg.get("color_width",  1280)
    h   = cam_cfg.get("color_height", 720)
    fps = cam_cfg.get("fps", 30)

    _root     = Path(__file__).resolve().parent.parent.parent
    calib_dir = str(_root / "config" / "calibration" / cam_type)

    if cam_type.startswith("sim"):
        from drivers.sim import get_world, SimCamera
        return SimCamera(get_world(cfg), w, h, fps)
    if "realsense" in cam_type:
        return RealsenseCamera(w, h, fps, calib_dir=calib_dir,
                               serial=cam_cfg.get("serial"), model=cam_type,
                               auto_exposure=cam_cfg.get("auto_exposure", True),
                               exposure_us=cam_cfg.get("exposure_us"))
    else:
        raise ValueError(
            f"Unsupported camera type: {cam_type!r}\n"
            f"Set camera.type in config/default.yaml to:\n"
            f"  realsense_d405  (or sim_d405 for the hardware-free demo)"
        )
