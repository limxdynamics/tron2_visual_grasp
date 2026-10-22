"""Detector loading and detection parsing for the OpenCV tennis detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .common_utils import class_name, clip_bbox, detection_count
from .opencv_tennis import TennisDetector


@dataclass
class Detection:
    """One detection; `mask` is the full-frame uint8 ball silhouette."""

    result_index: int
    detection_index: int
    class_name: str
    conf: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None


def load_detector(cfg: dict[str, Any]) -> tuple[TennisDetector, dict[str, Any]]:
    """Build the tennis detector; its results follow the ultralytics OBB API."""
    det_cfg = cfg.get("detection", {})
    model = TennisDetector(det_cfg["tennis"])
    print("Loading OpenCV tennis detector (HSV + contour geometry, CPU)")
    return model, {
        "device": "cpu",
        "model_name": "opencv_tennis",
        "conf": float(det_cfg.get("conf_threshold", 0.35)),
    }


def detect_objects(
    model: TennisDetector,
    color_bgr: np.ndarray,
    detector_opts: dict[str, Any],
) -> tuple[list[Any], list[Detection]]:
    results = model.predict(color_bgr, conf=float(detector_opts.get("conf", 0.35)))
    image_shape = color_bgr.shape[:2]
    detections: list[Detection] = []
    for result_index, result in enumerate(results):
        obb = result.obb
        for i in range(detection_count(result)):
            detections.append(Detection(
                result_index, i,
                class_name(result.names, int(obb.cls[i])),
                float(obb.conf[i]),
                clip_bbox(obb.xyxy[i], image_shape),
                result.masks.data[i],
            ))
    return results, detections
