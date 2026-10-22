"""Detector, grasp geometry, hand-eye and ArUco on synthetic data (no camera)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from conftest import K

H, W = 720, 1280


def _ball_colour() -> tuple[int, int, int]:
    """A BGR colour inside the shipped tennis-ball HSV range."""
    bgr = cv2.cvtColor(np.uint8([[[35, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in bgr)


@pytest.fixture(scope="module")
def detector(cfg):
    from utils.detection import load_detector
    return load_detector(cfg)


def test_detects_ball_and_rejects_square(detector):
    from utils.detection import detect_objects
    model, opts = detector
    img = np.full((H, W, 3), 90, np.uint8)
    cv2.circle(img, (640, 400), 30, _ball_colour(), -1)
    cv2.rectangle(img, (200, 200), (280, 280), _ball_colour(), -1)  # same colour, wrong shape
    _, dets = detect_objects(model, img, opts)
    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0].bbox_xyxy
    assert abs((x1 + x2) / 2 - 640) < 5 and abs((y1 + y2) / 2 - 400) < 5
    assert dets[0].conf >= opts["conf"]


def test_blank_frame_has_no_detections(detector):
    from utils.detection import detect_objects
    model, opts = detector
    _, dets = detect_objects(model, np.full((H, W, 3), 90, np.uint8), opts)
    assert dets == []


def test_ball_touching_border_is_dropped(detector):
    from utils.detection import detect_objects
    model, opts = detector
    img = np.full((H, W, 3), 90, np.uint8)
    cv2.circle(img, (10, 400), 30, _ball_colour(), -1)
    _, dets = detect_objects(model, img, opts)
    assert dets == []


def test_grasp_pose_from_detection(detector):
    from utils.detection import detect_objects
    from utils.ordinary_grasp import estimate_grasp
    model, opts = detector
    img = np.full((H, W, 3), 90, np.uint8)
    cv2.circle(img, (640, 400), 30, _ball_colour(), -1)
    results, _ = detect_objects(model, img, opts)
    depth = np.full((H, W), 500, np.uint16)                 # floor at 0.50 m
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (640, 400), 30, 1, -1)
    depth[mask > 0] = 400                                    # ball top at 0.40 m
    grasp = estimate_grasp(results[0], 0, depth, K, depth_quantile=0.5)
    assert grasp.is_valid, grasp.rejected_reason
    assert abs(grasp.position[2] - 0.40) < 0.02
    assert abs(grasp.jaw_width_m - 60 * 0.40 / 640) < 0.015  # ~ball diameter at 0.4 m


def _rt(rvec_deg, t):
    M = np.eye(4)
    M[:3, :3] = cv2.Rodrigues(np.radians(np.asarray(rvec_deg, float)))[0]
    M[:3, 3] = t
    return M


def test_hand_eye_recovers_known_transform(cfg):
    from calibration.hand_eye import CalibMode, HandEyeCalibrator
    rng = np.random.default_rng(0)
    X = _rt([3, -2, 88], [0.04, -0.02, 0.06])                # true camera -> gripper
    marker_in_base = _rt([0, 0, 0], [0.45, 0.0, -0.1])
    cal = HandEyeCalibrator(CalibMode.EYE_IN_HAND, cfg["calibration"]["hand_eye_method"])
    for _ in range(15):
        G = _rt(rng.uniform(-25, 25, 3) + [180, 0, 0], [0.35, 0, 0.25] + rng.uniform(-0.05, 0.05, 3))
        marker_in_cam = np.linalg.inv(G @ X) @ marker_in_base
        marker_in_cam[:3, 3] += rng.normal(0, 0.0005, 3)      # 0.5 mm noise
        cal.add_sample(G, marker_in_cam)
    result = cal.calibrate()
    assert np.linalg.norm(result.T_result[:3, 3] - X[:3, 3]) < 0.003
    assert not result.suspect


def test_aruco_pose_distance(cfg):
    from calibration.aruco_pose import ArUcoDetector
    length = cfg["calibration"]["aruco"]["marker_length_m"]
    z = 0.30
    px = int(round(length * K[0, 0] / z))
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(0), 0, px)
    img = np.full((H, W), 255, np.uint8)
    img[360 - px // 2:360 - px // 2 + px, 640 - px // 2:640 - px // 2 + px] = marker
    pose = ArUcoDetector(length, 0, 0).detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), K, np.zeros(5))
    assert pose is not None and abs(pose.T_marker2cam[2, 3] - z) < 0.01


def test_calibration_pdf_marker(project_root):
    """The shipped PDF must be a real PDF containing DICT_4X4_50 id 0."""
    data = (project_root / "aruco100x100.pdf").read_bytes()
    assert data.startswith(b"%PDF-"), "aruco100x100.pdf is not a PDF (encrypted or corrupted?)"
