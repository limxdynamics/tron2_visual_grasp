"""Simulated wrist RGB-D camera: ray-casts the floor and the ball from the TCP pose."""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from drivers.camera.base import CameraDriver

from .world import BALL_RADIUS_M, SimWorld

_HFOV_DEG = 87.0               # D405 colour field of view
_MIN_DEPTH_M = 0.07            # D405 minimum range: closer reads as 0 (no depth)
_MAX_DEPTH_M = 4.0
_BALL_HSV = (35, 200, 225)     # inside the shipped tennis HSV range


class SimCamera(CameraDriver):
    """Renders BGR + depth-mm frames of a SimWorld at a fixed frame rate."""

    def __init__(self, world: SimWorld, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self._world = world
        self._w, self._h, self._period = int(width), int(height), 1.0 / max(1, int(fps))
        f = (self._w / 2.0) / math.tan(math.radians(_HFOV_DEG / 2.0))
        self._K = np.array([[f, 0.0, self._w / 2.0], [0.0, f, self._h / 2.0], [0.0, 0.0, 1.0]])
        self._D = np.zeros((1, 5))
        u, v = np.meshgrid(np.arange(self._w, dtype=np.float32), np.arange(self._h, dtype=np.float32))
        # Ray directions with z = 1, so the ray parameter is the depth along the optical axis.
        self._rays = np.stack([(u - self._K[0, 2]) / f, (v - self._K[1, 2]) / f,
                               np.ones_like(u)], axis=-1)
        self._rays_small = np.ascontiguousarray(self._rays[::4, ::4])
        bgr = cv2.cvtColor(np.uint8([[_BALL_HSV]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
        self._ball_bgr = bgr
        # Fixed sensor grain, applied with saturating uint8 adds.
        grain = np.random.default_rng(1).normal(0.0, 3.0, (self._h, self._w))
        grain = np.repeat(grain[..., None], 3, axis=2)
        self._grain_up = np.clip(grain, 0, 255).astype(np.uint8)
        self._grain_down = np.clip(-grain, 0, 255).astype(np.uint8)
        self._open = False
        self._next_t = 0.0

    def open(self) -> None:
        self._open = True
        self._next_t = time.monotonic()
        print(f"[SimCamera] Rendering {self._w}x{self._h} @ {1.0 / self._period:.0f} fps")

    def close(self) -> None:
        self._open = False

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def D(self) -> np.ndarray:
        return self._D

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._open:
            return None, None
        now = time.monotonic()
        if now < self._next_t:
            time.sleep(self._next_t - now)
        self._next_t = max(self._next_t + self._period, time.monotonic())
        with self._world.lock:
            self._world.step()
            T = self._world.camera_world()
            ball = self._world.ball.pos.copy()
        return self.render(T, ball)

    def render(self, T_cam2world: np.ndarray, ball_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R, c = T_cam2world[:3, :3].astype(np.float32), T_cam2world[:3, 3].astype(np.float32)

        # Floor z = 0 at quarter resolution, then upscaled: grey with a 0.5 m checker
        # and a soft ball shadow. Inverse depth is linear in the image for a plane,
        # so interpolating it keeps the floor depth exact.
        ds = self._rays_small @ R.T                              # world ray directions
        dz = ds[..., 2]
        floor = dz < -1e-6
        t_floor = np.where(floor, -c[2] / np.where(floor, dz, -1.0), 0.0)
        px = c[0] + t_floor * ds[..., 0]
        py = c[1] + t_floor * ds[..., 1]
        checker = ((np.floor(px * 2.0) + np.floor(py * 2.0)) % 2.0).astype(np.float32)
        shade = 105.0 + 22.0 * checker
        r2 = (px - ball_center[0]) ** 2 + (py - ball_center[1]) ** 2
        shadow_r = BALL_RADIUS_M * (1.4 + 4.0 * max(0.0, ball_center[2] - BALL_RADIUS_M))
        shade *= 1.0 - 0.45 * np.exp(-r2 / (shadow_r ** 2))
        haze = np.clip((t_floor - 2.5) / 6.0, 0.0, 0.6)          # fade toward the horizon
        shade = np.where(floor, shade * (1.0 - haze) + 150.0 * haze, 160.0)
        with np.errstate(divide="ignore"):
            inv_depth = np.where(floor, 1.0 / np.where(floor, t_floor, 1.0), 0.0).astype(np.float32)
        size = (self._w, self._h)
        shade = cv2.resize(np.clip(shade, 0, 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
        inv_depth = cv2.resize(inv_depth, size, interpolation=cv2.INTER_LINEAR)
        # Depth in mm straight from inverse depth; out of range reads 0 like the D405.
        with np.errstate(divide="ignore"):
            depth_m = np.where(inv_depth > 1.0 / _MAX_DEPTH_M, 1.0 / np.maximum(inv_depth, 1e-6), 0.0)
        depth_mm = (depth_m * 1000.0).astype(np.uint16)
        bgr = cv2.cvtColor(shade, cv2.COLOR_GRAY2BGR)

        # Ball: exact sphere intersection at full resolution inside its projected box.
        self._draw_ball(T_cam2world, R, c, ball_center, depth_mm, bgr)

        bgr = cv2.subtract(cv2.add(bgr, self._grain_up), self._grain_down)
        depth_mm[depth_mm < int(_MIN_DEPTH_M * 1000)] = 0
        return bgr, depth_mm

    def _draw_ball(self, T: np.ndarray, R: np.ndarray, c: np.ndarray, s: np.ndarray,
                   depth_mm: np.ndarray, bgr: np.ndarray) -> None:
        p_cam = np.linalg.inv(T) @ np.array([*s, 1.0])
        if p_cam[2] <= BALL_RADIUS_M * 0.5:
            return
        f = self._K[0, 0]
        uc = f * p_cam[0] / p_cam[2] + self._K[0, 2]
        vc = f * p_cam[1] / p_cam[2] + self._K[1, 2]
        rad = f * BALL_RADIUS_M / max(p_cam[2] - BALL_RADIUS_M, 1e-3) + 2
        u0, u1 = int(max(0, uc - rad)), int(min(self._w, uc + rad + 1))
        v0, v1 = int(max(0, vc - rad)), int(min(self._h, vc + rad + 1))
        if u0 >= u1 or v0 >= v1:
            return
        dd = self._rays[v0:v1, u0:u1] @ R.T
        oc = (c - s.astype(np.float32))
        a = np.einsum("...i,...i", dd, dd)
        b = 2.0 * dd @ oc
        disc = b * b - 4.0 * a * (float(oc @ oc) - BALL_RADIUS_M ** 2)
        hit = disc >= 0.0
        t = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a), np.inf)
        roi_depth = depth_mm[v0:v1, u0:u1]
        behind = np.where(roi_depth > 0, roi_depth.astype(np.float32) / 1000.0, np.inf)
        hit &= (t > 0.0) & (t < behind) & (t <= _MAX_DEPTH_M)
        if not hit.any():
            return
        th = t[hit]
        normal = (c + th[:, None] * dd[hit] - s.astype(np.float32)) / BALL_RADIUS_M
        light = np.array([0.3, -0.2, 0.93], np.float32)
        lambert = np.clip(normal @ (light / np.linalg.norm(light)), 0.0, 1.0)
        roi_depth[hit] = (th * 1000.0).astype(np.uint16)
        shaded = self._ball_bgr * (0.55 + 0.45 * lambert)[:, None]
        bgr[v0:v1, u0:u1][hit] = np.clip(shaded, 0, 255).astype(np.uint8)
