"""Classical colour and shape detection with the existing OBB result contract."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np


class _Array(np.ndarray):
    """CPU array compatible with grasp helpers expecting tensor.cpu().numpy()."""

    def cpu(self) -> _Array:
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self)


class _OBB(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.conf)


class TennisDetector:
    """Reject non-round colour blobs; no neural model or actuator dependencies."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        self._hsv_lock = threading.RLock()
        self._last_frame: tuple[np.ndarray, float, float] | None = None
        self.set_hsv_bounds(options["hsv_lower"], options["hsv_upper"])
        kernel_size = int(options["morph_kernel_px"])
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("tennis morph_kernel_px must be a positive odd integer")
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def get_hsv_bounds(self) -> tuple[list[int], list[int]]:
        with self._hsv_lock:
            return self.lower.tolist(), self.upper.tolist()

    @staticmethod
    def validate_hsv_bounds(lower: Any, upper: Any) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or np.any(lower != np.floor(lower)) or np.any(upper != np.floor(upper))
                or np.any(lower < 0) or np.any(upper > [179, 255, 255])
                or np.any(lower > upper)):
            raise ValueError("tennis HSV bounds must be ordered H:0..179, S/V:0..255")
        return lower.astype(np.uint8), upper.astype(np.uint8)

    def set_hsv_bounds(self, lower: Any, upper: Any) -> None:
        lower, upper = self.validate_hsv_bounds(lower, upper)
        with self._hsv_lock:
            self.lower = lower
            self.upper = upper

    def last_frame(self) -> tuple[np.ndarray, float, float] | None:
        """Most recent (BGR frame, time.time() stamp, conf) passed to predict()."""
        return self._last_frame

    def hsv_masks(self, image: np.ndarray, lower: np.ndarray,
                  upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw inRange mask and the morphologically cleaned mask."""
        hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
        colour = cv2.inRange(hsv, lower, upper)
        cleaned = cv2.morphologyEx(colour, cv2.MORPH_CLOSE, self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.kernel)
        return colour, cleaned

    def predict(self, image: np.ndarray, hsv: tuple[Any, Any] | None = None,
                **kwargs: Any) -> list[Any]:
        """Detect balls; ``hsv`` overrides the bounds for a preview without caching the frame."""
        o = self.options
        h, w = image.shape[:2]
        if hsv is None:
            self._last_frame = (image.copy(), time.time(), float(kwargs.get("conf", 0.0)))
            with self._hsv_lock:
                lower, upper = self.lower, self.upper
        else:
            lower, upper = self.validate_hsv_bounds(*hsv)
        colour, cleaned = self.hsv_masks(image, lower, upper)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not float(o["min_area_px"]) <= area <= float(o["max_area_fraction"]) * h * w:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            # An incomplete silhouette gives an unreliable centre and grasp size.
            if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
                continue
            aspect = min(bw, bh) / max(bw, bh)
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = min(1.0, 4.0 * np.pi * area / perimeter**2)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / hull_area if hull_area > 0 else 0.0
            (_, _), radius = cv2.minEnclosingCircle(contour)
            fill = min(1.0, area / (np.pi * radius**2)) if radius > 0 else 0.0
            if (aspect < float(o["min_aspect_ratio"])
                    or circularity < float(o["min_circularity"])
                    or solidity < float(o["min_solidity"])
                    or fill < float(o["min_circle_fill"])):
                continue
            local = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(local, [contour - [x, y]], -1, 1, cv2.FILLED)
            pixels = local > 0
            colour_fraction = float(np.count_nonzero(colour[y:y+bh, x:x+bw][pixels])) / np.count_nonzero(pixels)
            if colour_fraction < float(o["min_colour_fraction"]):
                continue
            # A shape quality score, not a calibrated class probability.
            score = float(np.mean([aspect, circularity, solidity, fill, colour_fraction]))
            if score < float(kwargs.get("conf", 0.0)):
                continue
            # Axis-aligned bounds avoid unstable contour rotation on a round ball.
            points = np.array([[x, y], [x+bw-1, y], [x+bw-1, y+bh-1], [x, y+bh-1]], dtype=np.float32)
            candidates.append((score, points, (x, y, x+bw-1, y+bh-1), local))
        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[:max(1, int(o["max_det"]))]
        count = len(candidates)
        masks = np.zeros((count, h, w), dtype=np.uint8)
        for i, (_, _, (x1, y1, x2, y2), local) in enumerate(candidates):
            masks[i, y1:y2+1, x1:x2+1] = local
        obb = _OBB(
            cls=np.zeros(count, dtype=np.float32),
            conf=np.asarray([c[0] for c in candidates], dtype=np.float32),
            xyxy=np.asarray([c[2] for c in candidates], dtype=np.float32).reshape(count, 4),
            xyxyxyxy=np.asarray([c[1] for c in candidates], dtype=np.float32).reshape(count, 4, 2),
        )
        return [SimpleNamespace(obb=obb, boxes=[None] * count,
                                masks=SimpleNamespace(data=masks.view(_Array)),
                                names={0: str(o["class_name"])}, orig_shape=(h, w))]
