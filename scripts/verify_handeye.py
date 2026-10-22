"""Verify eye-in-hand calibration with the fixed-marker invariance test.

The ArUco marker is fixed on the table and the camera rides on the wrist, so
    T_marker2base = T_tcp2base @ T_cam2gripper @ T_marker2cam
must be the SAME at every robot pose when the hand-eye matrix is correct. This
script traverses several poses around an anchor, recomputes the marker position
in the base frame at each, and reports how much those points spread. The spread
is a direct, contact-free measure of calibration accuracy:

    max deviation  < 5 mm   excellent
                   < 10 mm  good
                   < 20 mm  marginal (retake calibration or add compensation)
                   >= 20 mm poor (recalibrate)

The base-frame translation compensation (calibration.hand_eye_compensation_m)
only shifts every point by the same vector, so it does not change the spread;
this test therefore checks the hand-eye matrix with the rotation compensation
(calibration.hand_eye_rotation_compensation_rpy_rad) applied.

Usage:
    ./run.sh verify
    ./run.sh verify --max-poses 10 --web-port 0
"""

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # reuse the collector

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import numpy as np

from drivers.camera import make_camera
from drivers.robot import RobotDriver, make_robot
from utils.camera_utils import load_config, load_hand_eye
from utils.web_debug import WebDebug
from collect_handeye_eih import (
    AUTO_MARKER_STABLE_FRAMES,
    AUTO_MARKER_TIMEOUT_S,
    AUTO_MOVE_DURATION_S,
    AUTO_SETTLE_EXTRA_S,
    CALIB_MOVES,
    DEFAULT_PIVOT_M,
    _depth_at_marker,
    build_calib_poses,
    wait_for_marker,
)


def _mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of rotation matrices via SVD projection onto SO(3)."""
    M = np.mean(np.stack(rotations, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _geodesic_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _verdict(max_dev_mm: float) -> str:
    if max_dev_mm < 5.0:
        return "EXCELLENT"
    if max_dev_mm < 10.0:
        return "GOOD"
    if max_dev_mm < 20.0:
        return "MARGINAL"
    return "POOR"


def check_pose_source_switch(robot, state: dict) -> str | None:
    """Detect get_tcp_pose() silently switching gripper frame mid-run.

    Tron2Driver falls back from the controller TCP-pose API to joint-state FK
    (the FLANGE frame, a different physical frame) when the API stops responding,
    and never falls back the other way. Marker positions computed before and
    after that flip are not in the same frame, so their spread is meaningless.
    Returns an error message once per flip, else None.
    """
    api = getattr(robot, "tcp_pose_api_available", None)
    if api is None:
        return None
    prev = state.get("api")
    state["api"] = api
    if prev is None or prev == api:
        return None
    prev_name, cur_name = ("API/TCP", "FK/flange") if prev else ("FK/flange", "API/TCP")
    return (
        f"TCP pose source switched mid-run ({prev_name} -> {cur_name}); samples "
        "captured before and after this point are in different gripper frames "
        "and cannot be combined."
    )


def sample_marker_at_pose(cam, web):
    """Wait briefly for a stable marker detection; return (pose, depth_z_m).

    depth_z_m is the RGB-D depth at the marker centre (meters) or None; it is an
    independent, marker_length-free reference for the solvePnP distance.
    """
    deadline = time.monotonic() + AUTO_MARKER_TIMEOUT_S
    stable = 0
    latest = None
    latest_depth_z = None
    while time.monotonic() < deadline:
        bgr, depth = cam.get_frame()
        if bgr is None:
            continue
        pose = cam.detect_aruco(bgr)
        if web is not None:
            web.set_frame(cam.draw_aruco(bgr))
        if pose is not None:
            latest = pose
            latest_depth_z = _depth_at_marker(depth, pose)
            stable += 1
            if stable >= AUTO_MARKER_STABLE_FRAMES:
                return pose, latest_depth_z
        else:
            stable = 0
        time.sleep(0.03)
    return (latest, latest_depth_z) if stable > 0 else (None, None)


def report(samples: list[np.ndarray], scale_ratios: list | None = None,
           marker_length_m: float | None = None) -> None:
    if len(samples) < 3:
        print(f"\n[Report] Only {len(samples)} samples; need >=3 for a meaningful "
              f"spread. Reposition the marker so more poses see it, then rerun.")
        return

    positions = np.stack([T[:3, 3] for T in samples], axis=0)      # (N, 3) meters
    rotations = [T[:3, :3] for T in samples]
    centroid = positions.mean(axis=0)
    dev = np.linalg.norm(positions - centroid, axis=1)            # meters
    dev_mm = dev * 1000.0
    rms_mm = float(np.sqrt(np.mean(dev_mm ** 2)))
    max_mm = float(dev_mm.max())

    R_mean = _mean_rotation(rotations)
    ang = np.array([_geodesic_angle_deg(R, R_mean) for R in rotations])

    print("\n================ Hand-Eye Verification ================")
    print("Marker position in the base frame at each pose (should be identical):")
    print("  #   x(m)     y(m)     z(m)     pos_err(mm)  rot_err(deg)")
    for i, (p, d, a) in enumerate(zip(positions, dev_mm, ang)):
        print(f"  {i:<3d} {p[0]:+8.4f} {p[1]:+8.4f} {p[2]:+8.4f}   {d:7.2f}      {a:6.2f}")

    print("\n  Centroid (mean marker position in base):"
          f" x={centroid[0]:+.4f} y={centroid[1]:+.4f} z={centroid[2]:+.4f} m")
    print(f"  Position spread:  RMS={rms_mm:.2f} mm   MAX={max_mm:.2f} mm")
    print(f"  Rotation spread:  RMS={float(np.sqrt(np.mean(ang ** 2))):.2f} deg   "
          f"MAX={float(ang.max()):.2f} deg")
    print(f"  Samples used: {len(samples)}")
    print(f"\n  VERDICT: {_verdict(max_mm)}  "
          f"(<5mm excellent, <10mm good, <20mm marginal, else poor)")
    if max_mm >= 20.0:
        print("  -> Recalibrate: collect >=15 samples with more rotation diversity,")
        print("     confirm ArUco marker_length_m matches the printed size, and that")
        print("     the marker stays flat and fully in view.")
    elif max_mm >= 10.0:
        print("  -> Usable but not tight. A residual constant bias can be trimmed with")
        print("     calibration.hand_eye_compensation_m (x/y/z, base frame).")
    if scale_ratios:
        ratio = float(np.median(scale_ratios))
        print(f"\n  Depth cross-check: median RGB-D / PnP distance ratio = {ratio:.3f}"
              f"  ({len(scale_ratios)} poses)")
        if abs(ratio - 1.0) > 0.05 and marker_length_m:
            print("  -> solvePnP scale looks off; marker_length_m likely mismatches the print.")
            print(f"     Set calibration.aruco.marker_length_m ~= {marker_length_m * ratio:.3f} m "
                  f"(now {marker_length_m:.3f} m).")
        else:
            print("  -> solvePnP scale matches depth; marker_length_m looks correct.")
    print("=======================================================")


def make_input_thread(line_queue: "queue.Queue") -> threading.Thread:
    def _loop():
        while True:
            try:
                line_queue.put(input())
            except (EOFError, KeyboardInterrupt):
                line_queue.put(None)
                break
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify eye-in-hand calibration")
    parser.add_argument("--max-poses", type=int, default=len(CALIB_MOVES),
                        help="number of anchor-relative poses to visit")
    parser.add_argument("--web-port", type=int, default=8888,
                        help="web console port; 0 disables it")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config" / "default.yaml")

    cam_type = cfg["camera"]["type"]
    aruco_cfg = cfg["calibration"]["aruco"]

    T_hand_eye, mode = load_hand_eye(root, cam_type, cfg)
    if T_hand_eye is None:
        print(f"[Error] No hand_eye.npz for camera '{cam_type}'. Run "
              "./run.sh calib first.")
        sys.exit(1)
    if mode != "eye_in_hand":
        print(f"[Error] hand_eye.npz mode is '{mode}', expected 'eye_in_hand'.")
        sys.exit(1)

    print("\n=== Eye-in-Hand Verification ===")
    print(f"Camera: {cam_type}  |  Poses: up to {args.max_poses}")
    print("The arm will make small non-contact moves around its current pose.")
    print("Keep the ArUco marker fixed and flat in view. Press q / Ctrl+C to stop early.\n")

    cam = make_camera(cfg)
    cam.setup_aruco(
        marker_length_m=aruco_cfg["marker_length_m"],
        dict_id=aruco_cfg.get("dict_id", 0),
        target_marker_id=aruco_cfg.get("target_marker_id"),
    )

    web = None
    if args.web_port:
        web = WebDebug(args.web_port, "Eye-in-Hand Verification",
                       buttons=[("stop", "Stop and report")])
        web.start()
        print(f"[Web] Console on http://0.0.0.0:{args.web_port}/")

    samples: list[np.ndarray] = []
    scale_ratios: list[float] = []
    robot: RobotDriver | None = None
    line_queue: "queue.Queue | None" = None
    if sys.stdin.isatty():
        line_queue = queue.Queue()
        make_input_thread(line_queue)

    def stop_requested() -> bool:
        if web is not None and web.pop_command() is not None:
            print("[Stop] Requested from web console.")
            return True
        if line_queue is not None:
            try:
                if str(line_queue.get_nowait() or "").strip().lower() in {"q", "c", ""}:
                    print("[Stop] Requested from terminal.")
                    return True
            except queue.Empty:
                pass
        return False

    try:
        cam.open()
        print("Warming up camera...", end="", flush=True)
        cam.warm_up(20)
        print(" ready\n")

        robot = make_robot(cfg)
        robot.start()

        anchor_view = wait_for_marker(cam)
        T_anchor = robot.get_tcp_pose()
        pivot_m = (float(anchor_view.T_marker2cam[2, 3])
                   if anchor_view is not None else DEFAULT_PIVOT_M)
        poses = build_calib_poses(T_anchor, pivot_m)[:args.max_poses]
        print(f"[Robot] Anchored; visiting {len(poses)} poses "
              f"(pivot {pivot_m:.2f} m).\n")

        pose_source_state: dict = {}
        for i, (x, y, z, roll, pitch, yaw) in enumerate(poses):
            if stop_requested():
                break
            switch_msg = check_pose_source_switch(robot, pose_source_state)
            if switch_msg:
                print(f"[Error] {switch_msg} Stopping early; this run's spread is not meaningful.")
                break
            print(f"[Pose {i + 1}/{len(poses)}] moving...", flush=True)
            if not robot.move_to_pose(x, y, z, roll=roll, pitch=pitch, yaw=yaw,
                                      duration=AUTO_MOVE_DURATION_S):
                print("  unreachable, skipping")
                continue
            time.sleep(AUTO_SETTLE_EXTRA_S)

            pose, depth_z = sample_marker_at_pose(cam, web)
            if pose is None:
                print("  marker not visible, skipping")
                continue

            T_tcp2base = robot.get_tcp_pose()
            T_marker2base = T_tcp2base @ T_hand_eye @ pose.T_marker2cam
            samples.append(T_marker2base)
            z_pnp = float(pose.T_marker2cam[2, 3])
            if depth_z is not None and z_pnp > 1e-3:
                scale_ratios.append(depth_z / z_pnp)
            p = T_marker2base[:3, 3]
            print(f"  marker@base: x={p[0]:+.4f} y={p[1]:+.4f} z={p[2]:+.4f} m "
                  f"(sample {len(samples)})")
            if web is not None:
                web.set_status({
                    "samples": str(len(samples)),
                    "pose": f"{i + 1}/{len(poses)}",
                    "marker@base": f"{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f} m",
                })

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping.")
    except Exception as e:
        print(f"[Error] {e}")
    finally:
        try:
            cam.close()
        except Exception:
            pass
        if robot is not None:
            try:
                robot.end()
            except Exception:
                pass
        report(samples, scale_ratios, aruco_cfg["marker_length_m"])
        if web is not None:
            print("[Web] Console closing in 10s...")
            time.sleep(10.0)
            web.stop()


if __name__ == "__main__":
    main()
    os._exit(0)
