"""Shared camera/config helpers for scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

try:
    from .transforms import pose6d_to_mat4
except ImportError:
    from utils.transforms import pose6d_to_mat4


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def hand_eye_rotation_compensation_matrix(cfg: dict[str, Any]) -> np.ndarray:
    calibration = cfg.get("calibration") or {}
    compensation = calibration.get("hand_eye_rotation_compensation_rpy_rad") or {}
    rx = float(compensation.get("roll", compensation.get("x", 0.0)))
    ry = float(compensation.get("pitch", compensation.get("y", 0.0)))
    rz = float(compensation.get("yaw", compensation.get("z", 0.0)))
    return pose6d_to_mat4(0.0, 0.0, 0.0, rx, ry, rz)


def apply_hand_eye_compensation(T_hand_eye: np.ndarray, cfg: dict[str, Any] | None) -> np.ndarray:
    T = np.asarray(T_hand_eye, dtype=np.float64)
    if cfg is None:
        return T
    return hand_eye_rotation_compensation_matrix(cfg) @ T


def load_hand_eye(project_root: str | Path, cam_type: str, cfg: dict[str, Any] | None = None) -> tuple[Optional[np.ndarray], Optional[str]]:
    calib_root = Path(project_root) / "config" / "calibration"
    wanted = str(cam_type).lower()
    hand_eye_path = calib_root / wanted / "hand_eye.npz"
    if not hand_eye_path.exists():
        # Directory casing varies between setups (e.g. Realsense_D405 vs
        # realsense_d405); match it case-insensitively so a capitalised folder
        # is still found on a case-sensitive filesystem.
        hand_eye_path = None
        if calib_root.is_dir():
            for child in calib_root.iterdir():
                if child.is_dir() and child.name.lower() == wanted:
                    candidate = child / "hand_eye.npz"
                    if candidate.exists():
                        hand_eye_path = candidate
                        break
        if hand_eye_path is None:
            return None, None

    data = np.load(str(hand_eye_path), allow_pickle=False)
    T = apply_hand_eye_compensation(data["T_result"].astype(np.float64), cfg)
    mode = str(data["mode"][0])
    return T, mode


def hand_eye_compensation_matrix(cfg: dict[str, Any]) -> np.ndarray:
    calibration = cfg.get("calibration") or {}
    compensation = calibration.get("hand_eye_compensation_m") or {}
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [
        float(compensation.get("x", 0.0)),
        float(compensation.get("y", 0.0)),
        float(compensation.get("z", 0.0)),
    ]
    return T


def compose_cam_to_base_transform(T_tcp2base: np.ndarray, T_hand_eye: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    T_compensation = hand_eye_compensation_matrix(cfg)
    return T_compensation @ np.asarray(T_tcp2base, dtype=np.float64) @ np.asarray(T_hand_eye, dtype=np.float64)
