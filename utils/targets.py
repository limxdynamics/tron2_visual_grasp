"""Pick which detected object is the grasp target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .detection import Detection
from .ordinary_grasp import get_depth_mm


@dataclass(frozen=True)
class Target:
    detection: Detection
    center_px: tuple[int, int]
    depth_m: Optional[float]


def normalize_target_classes(value: Any) -> tuple[str, ...]:
    """Accept one class name or a list of them; order sets the 'order' priority."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        names = [str(item).strip().lower() for item in value]
    else:
        names = [str(value).strip().lower()]
    return tuple(name for name in names if name)


def match_target_class(name: str, wanted: Sequence[str]) -> Optional[int]:
    """Index of the first wanted class this detection satisfies, or None."""
    if not wanted:
        return 0
    lowered = name.strip().lower()
    for index, candidate in enumerate(wanted):
        if candidate in lowered or lowered in candidate:
            return index
    return None


def select_target(
    detections: list[Detection],
    depth_mm: Optional[np.ndarray],
    target_class: Any = None,
    priority: str = "nearest",
    frame_width: Optional[int] = None,
    depth_roi_size: int = 9,
) -> Optional[Target]:
    """Choose which detected object to grasp.

    nearest:    closest object of any wanted class.
    confidence: strongest detection.
    centered:   detection closest to the image centre column.
    order:      first class listed that is visible, then the closest of that class.
    """
    wanted = normalize_target_classes(target_class)
    candidates: list[tuple[int, Target]] = []
    for detection in detections:
        rank = match_target_class(detection.class_name, wanted)
        if rank is None:
            continue
        x1, y1, x2, y2 = detection.bbox_xyxy
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        depth = get_depth_mm(depth_mm, center[0], center[1], depth_roi_size) if depth_mm is not None else 0.0
        candidates.append((
            rank,
            Target(
                detection=detection,
                center_px=center,
                depth_m=depth / 1000.0 if depth > 0.0 else None,
            ),
        ))

    if not candidates:
        return None
    if priority == "order":
        best_rank = min(rank for rank, _ in candidates)
        candidates = [item for item in candidates if item[0] == best_rank]
    elif priority not in ("nearest", "confidence", "centered"):
        raise ValueError(
            "mission.target_priority must be 'nearest', 'confidence', 'centered' or 'order'"
        )

    targets = [target for _, target in candidates]
    if priority == "confidence":
        return max(targets, key=lambda candidate: candidate.detection.conf)
    if priority == "centered":
        if frame_width is None or frame_width <= 0:
            raise ValueError("frame_width is required for the 'centered' priority")
        aim = frame_width / 2.0
        return min(
            targets,
            key=lambda candidate: (
                abs(candidate.center_px[0] - aim), -candidate.detection.conf
            ),
        )
    with_depth = [candidate for candidate in targets if candidate.depth_m is not None]
    if with_depth:
        return min(with_depth, key=lambda candidate: float(candidate.depth_m))
    return max(targets, key=lambda candidate: candidate.detection.conf)
