"""
Eye-in-hand calibration data collection and solving (RGB-D camera + robot arm).

Modes:
  Auto mode (default): the arm traverses 26 preset poses (CALIB_MOVES), captures
                       samples when ArUco is detected, and skips timed-out poses.
  Manual mode (--manual): hand guiding lets the user move the arm by hand.
                          Release the arm, then press Enter. PiperX only;
                          TRON2 has no hand guiding.

Setup:
  The camera is mounted on the end effector.
  The ArUco marker is fixed on the work surface.

Usage:
    ./run.sh calib           # auto mode
    ./run.sh calib --manual  # manual hand-guiding mode
"""

import os
import sys
import threading
import argparse
import queue
import time
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import numpy as np

from drivers.camera import make_camera
from drivers.robot import RobotDriver, make_robot
from calibration.hand_eye import CalibMode, HandEyeCalibrator
from utils.camera_utils import load_config
from utils.transforms import pose6d_to_mat4, rotation_matrix_to_euler_zyx
from utils.web_debug import WebDebug


# ==========================================
# Calibration poses are built at runtime as offsets from the pose measured at
# startup, because the controller exposes no offline IK and its absolute frame
# depends on the deployment. Each entry is applied in the gripper frame as
# (roll, pitch, yaw) about a pivot on the view axis, plus a base-frame shift:
#   (drx, dry, drz, dx, dy, dz)  radians and meters
# ==========================================
# The wrist orbits the marker (radius = marker distance) so the marker stays
# centred; that lets tilts reach ~0.22 rad for the rotation spread TSAI needs
# without losing the marker. pitch is kept >= -0.10 and yaw <= 0 because the
# TRON2 workspace rejects large -pitch / +yaw (seen as unreachable poses). Each
# pose also carries a base shift (dx,dy,dz ~ 0.08-0.12 m); that shift, not a
# short radius, is what moves the marker in the camera (t_B != 0) so the hand-eye
# translation is observable instead of collapsing to |t| ~= marker distance.
CALIB_MOVES = [
    ( 0.00,  0.00,  0.00,   0.00,  0.00,  0.00),   # anchor
    ( 0.20,  0.00, -0.10,   0.00,  0.10,  0.00),
    (-0.20,  0.00, -0.10,   0.00, -0.10,  0.00),
    ( 0.00,  0.20, -0.10,   0.10,  0.00,  0.00),
    ( 0.00,  0.15, -0.20,  -0.10,  0.00,  0.00),
    ( 0.20,  0.15,  0.00,   0.09,  0.07,  0.00),
    (-0.20,  0.15,  0.00,  -0.09,  0.07,  0.00),
    ( 0.20,  0.00, -0.20,   0.10,  0.00,  0.05),
    (-0.20,  0.00, -0.20,  -0.10,  0.00, -0.05),
    ( 0.15,  0.20, -0.10,   0.08,  0.08,  0.00),
    (-0.15,  0.20, -0.10,  -0.08,  0.08,  0.00),
    ( 0.00,  0.20, -0.20,   0.00,  0.10,  0.05),
    ( 0.10, -0.10,  0.00,   0.08, -0.08,  0.00),
    ( 0.15,  0.15, -0.20,   0.08,  0.07, -0.05),
    (-0.15,  0.15, -0.15,  -0.08,  0.07,  0.05),
    ( 0.00,  0.00,  0.00,   0.12,  0.00,  0.00),
    ( 0.00,  0.00,  0.00,  -0.12,  0.00,  0.00),
    ( 0.00,  0.00,  0.00,   0.00,  0.12,  0.00),
    ( 0.00,  0.00,  0.00,   0.00, -0.12,  0.00),
    ( 0.22,  0.10, -0.10,   0.08,  0.06,  0.00),
    (-0.22,  0.10, -0.10,  -0.08,  0.06,  0.00),
    ( 0.15,  0.20, -0.20,   0.07,  0.08,  0.00),
    (-0.15,  0.18, -0.15,  -0.07,  0.07,  0.05),
    ( 0.20,  0.15, -0.15,   0.08,  0.06, -0.05),
    ( 0.10,  0.10, -0.10,   0.09,  0.06,  0.00),
    ( 0.00,  0.00,  0.00,   0.00,  0.00,  0.00),   # anchor
]

# Fallback pivot distance when ArUco is not visible at the anchor pose.
DEFAULT_PIVOT_M = 0.30

# Upper cap on the wrist orbit radius. Normally the radius equals the marker
# distance, so the wrist orbits the marker and it stays centred while large tilts
# still fit in view. Translation observability (t_B) comes from the per-pose base
# shifts, not from a short radius. This cap only bounds wrist travel when the
# marker is far; keep the marker within ~0.5 m for best results.
ROT_RADIUS_CAP_M = 0.60

AUTO_MOVE_DURATION_S = 3.0
AUTO_SETTLE_EXTRA_S = 0.6
AUTO_MARKER_TIMEOUT_S = 4.0
AUTO_MARKER_STABLE_FRAMES = 4
MIN_CALIB_SAMPLES = 5


def build_calib_poses(T_anchor, pivot_m: float):
    """Turn CALIB_MOVES into absolute poses around the measured anchor.

    The camera mounting is unknown, so the tool axis pointing most downward at
    the anchor is used as the view-axis proxy. The wrist orbits a point on that
    axis at min(pivot_m, ROT_RADIUS_CAP_M) -- normally the marker distance, so
    the marker stays centred and large tilts stay in view. Translation is made
    observable by the per-pose base shifts (dx,dy,dz) that move the marker in the
    camera (t_B != 0); the radius only bounds wrist travel for a far marker.
    """
    R0, p0 = T_anchor[:3, :3], T_anchor[:3, 3]
    candidates = np.hstack([R0, -R0])
    view_world = candidates[:, int(np.argmin(candidates[2, :]))]
    view_tool = R0.T @ view_world
    radius = min(float(pivot_m), ROT_RADIUS_CAP_M)
    pivot = p0 + view_world * radius

    poses = []
    for drx, dry, drz, dx, dy, dz in CALIB_MOVES:
        R = R0 @ pose6d_to_mat4(0, 0, 0, drx, dry, drz)[:3, :3]
        p = pivot - R @ view_tool * radius + np.array([dx, dy, dz])
        roll, pitch, yaw = rotation_matrix_to_euler_zyx(R)
        poses.append((p[0], p[1], p[2], roll, pitch, yaw))
    return poses


def _depth_at_marker(depth_mm, pose, radius: int = 3):
    """Median depth (m) in a small patch at the marker centre.

    Independent of marker_length, so it cross-checks the solvePnP distance and
    exposes a wrong ArUco marker size, the usual cause of a bad marker pose.
    Returns None when depth or the marker centre is unavailable.
    """
    if depth_mm is None or pose is None or getattr(pose, "corners", None) is None:
        return None
    center = pose.center_px
    if center is None:
        return None
    h, w = depth_mm.shape[:2]
    cu, cv = int(round(center[0])), int(round(center[1]))
    if not (0 <= cu < w and 0 <= cv < h):
        return None
    x0, x1 = max(0, cu - radius), min(w, cu + radius + 1)
    y0, y1 = max(0, cv - radius), min(h, cv + radius + 1)
    patch = depth_mm[y0:y1, x0:x1].astype(np.float64)
    valid = patch[patch > 0.0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / 1000.0


class _Tee:
    """Mirror stdout into the web console so headless runs see every message."""

    def __init__(self, stream, web):
        self._stream, self._web = stream, web

    def write(self, text):
        self._stream.write(text)
        line = text.strip()
        if line:
            low = line.lower()
            level = "err" if "[error]" in low or "failed" in low else \
                    "warn" if "[warning]" in low or "unreachable" in low else \
                    "ok" if "[ok]" in low else "info"
            self._web.add_event(line, level)

    def flush(self):
        self._stream.flush()


def wait_for_marker(cam, timeout_s: float = 60.0):
    """Block until the marker is visible, so the anchor pose actually sees it."""
    deadline = time.monotonic() + timeout_s
    asked = False
    while time.monotonic() < deadline:
        bgr, _ = cam.get_frame()
        if bgr is not None:
            found = cam.detect_aruco(bgr)
            if found:
                return found
        if not asked:
            print(f"[Robot] Marker not in view. Slide the ArUco marker under the "
                  f"camera; waiting up to {timeout_s:.0f}s...")
            asked = True
        time.sleep(0.1)
    return None


def make_input_thread(line_queue: queue.Queue) -> threading.Thread:
    def _loop():
        while True:
            try:
                line_queue.put(input())
            except EOFError:
                line_queue.put(None)
                break
            except KeyboardInterrupt:
                line_queue.put(None)
                break
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ==========================================
# Main flow.
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Eye-in-hand calibration data collection")
    parser.add_argument("--manual", action="store_true",
                        help="manual mode: hand guiding; move the arm by hand and press Enter to capture")
    parser.add_argument("--web-port", type=int, default=8888,
                        help="web console port; 0 disables it")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg  = load_config(root / "config" / "default.yaml")

    cam_type   = cfg["camera"]["type"]
    calib_dir  = root / "config" / "calibration" / cam_type
    aruco_cfg  = cfg["calibration"]["aruco"]
    he_method  = cfg["calibration"].get("hand_eye_method", "TSAI")
    save_path  = calib_dir / "hand_eye.npz"

    # Camera.
    cam = make_camera(cfg)
    cam.setup_aruco(
        marker_length_m=aruco_cfg["marker_length_m"],
        dict_id=aruco_cfg.get("dict_id", 0),
        target_marker_id=aruco_cfg.get("target_marker_id"),
    )

    # Calibrator.
    calibrator = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method=he_method)

    # Robot.
    mode_str = "manual (hand guiding)" if args.manual else f"auto ({len(CALIB_MOVES)} poses around the anchor)"
    robot: RobotDriver | None = None
    calib_poses: list = []
    auto = {
        "enabled": not args.manual,
        "idx": 0,
        "pose_idx": None,
        "phase": "idle",
        "settle_until": 0.0,
        "timeout_at": 0.0,
        "stable_frames": 0,
        "status": "waiting to start",
        "finished": False,
    }
    result_saved = False

    web: WebDebug | None = None
    if args.web_port:
        web = WebDebug(args.web_port, "Eye-in-Hand Calibration", buttons=[
            ("capture", "Capture (manual)"),
            ("pos", "Print TCP pose"),
            ("finish", "Finish and solve"),
        ])
        web.start()
        sys.stdout = _Tee(sys.stdout, web)
        print(f"[Web] Console on http://0.0.0.0:{args.web_port}/")

    print("\n=== Eye-in-Hand Calibration ===")
    print(f"Camera: {cam_type}  |  Mode: {mode_str}  |  Solver: {he_method}")
    print(f"ArUco size: {aruco_cfg['marker_length_m']*100:.0f}cm  |  Output: {save_path}")
    print()

    # Open the camera first so the arm is not enabled if camera setup fails.
    try:
        cam.open()
        print("Warming up camera...", end="", flush=True)
        cam.warm_up(20)
        print(" ready\n")
    except Exception as e:
        try:
            cam.close()
        except Exception:
            pass
        print(f"[Camera] Initialization failed: {e}")
        sys.exit(1)

    # Default working distance until the anchor ArUco view refines it (auto mode);
    # also the reference for rejecting gross marker mis-detections.
    pivot_m = DEFAULT_PIVOT_M
    try:
        robot = make_robot(cfg)
        robot.start()
        if args.manual:
            robot.enter_drag_mode()
            print("[Robot] Manual mode ready. Move the arm by hand, then press Enter to capture.")
        else:
            # The controller has no offline IK, so anchor on the pose it reports
            # and keep every target a small offset away from it.
            anchor_view = wait_for_marker(cam)
            T_anchor = robot.get_tcp_pose()
            if anchor_view is None:
                pivot_m = DEFAULT_PIVOT_M
                print("[Robot] [Warning] Marker still not visible; every pose will "
                      "probably time out. Reposition the marker and restart.")
            else:
                pivot_m = float(anchor_view.T_marker2cam[2, 3])
            calib_poses = build_calib_poses(T_anchor, pivot_m)
            t = T_anchor[:3, 3]
            print(f"[Robot] Anchor pose x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f} m, "
                  f"pivot {pivot_m:.2f} m "
                  f"({'from ArUco' if anchor_view else 'default, marker not visible'})")
            print(f"[Robot] Auto mode ready. {len(calib_poses)} poses will be traversed.")
    except Exception as e:
        try:
            cam.close()
        except Exception:
            pass
        print(f"[Robot] Connection failed: {e}")
        sys.exit(1)

    if args.manual:
        print("[Controls] Enter=capture  c/q=finish and solve  pos=print current TCP pose")
    else:
        print("[Controls] Auto traversal and capture  c/q=stop and solve  pos=print current TCP pose")
    print()

    latest_pose = None
    latest_depth = None
    line_queue: queue.Queue | None = None
    if sys.stdin.isatty():
        line_queue = queue.Queue()
        make_input_thread(line_queue)
    else:
        print("[Hint] Non-interactive terminal detected; terminal commands are disabled")

    def _print_fk() -> None:
        try:
            T = robot.get_tcp_pose()
            t = T[:3, 3]
            R = T[:3, :3]
            _r, _p, _y = rotation_matrix_to_euler_zyx(R)
            print(f"  FK: x={t[0]:+.3f} y={t[1]:+.3f} z={t[2]:+.3f} m"
                  f"  rpy=[{_r:+.2f} {_p:+.2f} {_y:+.2f}] rad")
        except Exception as e:
            print(f"  [Error] {e}")

    def capture_sample(cur, source: str) -> bool:
        if cur is None:
            print("  [Skip] Marker is not visible; adjust the pose and try again")
            return False

        print(f"\n[Sample {calibrator.n_samples + 1}] {source}")
        print(f"  ArUco: x={cur.T_marker2cam[0,3]:.3f} "
              f"y={cur.T_marker2cam[1,3]:.3f} "
              f"z={cur.T_marker2cam[2,3]:.3f} m")
        z_pnp = float(cur.T_marker2cam[2, 3])
        # Reject gross mis-detections (a wrong/reflected marker far outside the working
        # distance) so they cannot corrupt the solve; pivot_m is the anchor distance.
        z_hi = max(1.8 * pivot_m, 0.80)
        if not (0.4 * pivot_m <= z_pnp <= z_hi):
            print(f"  [Reject] Marker z={z_pnp:.3f} m is implausible vs the ~{pivot_m:.2f} m "
                  f"working distance (likely a mis-detection); not recorded.")
            return False
        z_depth = _depth_at_marker(latest_depth, cur)
        if z_depth is not None and z_pnp > 1e-3:
            ratio = z_depth / z_pnp
            print(f"  Depth cross-check: RGB-D z={z_depth:.3f} m  vs  PnP z={z_pnp:.3f} m  (ratio {ratio:.2f})")
            if abs(ratio - 1.0) > 0.08:
                suggested = aruco_cfg["marker_length_m"] * ratio
                print(f"  [Scale] PnP distance is off by {ratio:.2f}x -> marker_length_m likely "
                      f"mismatches the print; set calibration.aruco.marker_length_m "
                      f"~= {suggested:.3f} m (now {aruco_cfg['marker_length_m']:.3f} m)")
        try:
            T_g2b = robot.get_tcp_pose()
            t = T_g2b[:3, 3]
            print(f"  End effector (FK): x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            calibrator.add_sample(T_g2b, cur.T_marker2cam)
            print(f"  [OK] Recorded, total samples: {calibrator.n_samples}"
                  + ("  will solve automatically on finish" if calibrator.n_samples >= 15 else ""))
            return True
        except Exception as e:
            print(f"  [Error] Failed to read TCP pose: {e}")
            return False

    def compute_and_save(reason: str) -> bool:
        nonlocal result_saved
        print(f"\n[Finish] {reason}")
        if calibrator.n_samples < MIN_CALIB_SAMPLES:
            print(f"[Result] Not enough samples ({calibrator.n_samples} < {MIN_CALIB_SAMPLES}); calibration was not solved")
            if save_path.exists():
                print("[Result] Existing hand_eye.npz was not updated")
            return False

        print(f"[Result] Solving with {calibrator.n_samples} samples...")
        try:
            result = calibrator.calibrate(min_samples=MIN_CALIB_SAMPLES)
            t = result.T_result[:3, 3]
            R = result.T_result[:3, :3]
            print(f"[Result] T_cam2gripper translation: x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m (|t|={np.linalg.norm(t):.3f} m)")
            print(f"[Result] Rotation matrix:\n{R}")
            print(f"[Result] Residuals: rot RMS={result.rot_rms_deg:.2f} deg  "
                  f"trans RMS={result.trans_rms_m*1000:.1f} mm  "
                  f"rotation spread={result.rot_spread_deg:.0f} deg")
            if result.suspect:
                print("[Result] [WARNING] Result looks WRONG and was NOT saved. Likely causes:")
                if np.linalg.norm(t) > 0.30:
                    print("  - |translation| > 30cm: physically impossible for a wrist camera (degenerate solve)")
                if result.rot_spread_deg < 25.0:
                    print("  - rotation spread < 25 deg: poses too similar; tilt the wrist more between samples")
                if result.rot_rms_deg > 3.0 or result.trans_rms_m > 0.02:
                    print("  - large AX=XB residual: bad FK/ArUco samples or too few good poses")
                print("  Reposition the marker so every pose sees it, keep poses reachable, "
                      "collect >=15 varied samples, then re-run.")
                if save_path.exists():
                    print("[Result] Existing hand_eye.npz was kept unchanged")
                return False
            HandEyeCalibrator.save(result, save_path)
            print(f"[Result] [OK] Saved to {save_path}")
            if calibrator.n_samples < 15:
                print("[Result] Tip: fewer than 15 samples; collect more samples for better accuracy")
            result_saved = True
            return True
        except Exception as e:
            print(f"[Result] [Error] Solve failed: {e}")
            return False

    def start_next_auto_pose() -> bool:
        if not auto["enabled"] or robot is None:
            return False

        total = len(calib_poses)
        while auto["idx"] < total:
            idx = auto["idx"]
            x, y, z, roll, pitch, yaw = calib_poses[idx]
            print(f"\n[Auto] Pose {idx+1}/{total}: "
                  f"pos=({x:.2f},{y:.2f},{z:.2f}) rpy=({roll:.2f},{pitch:.2f},{yaw:.2f})")
            ok = robot.move_to_pose(x, y, z, roll=roll, pitch=pitch, yaw=yaw, duration=AUTO_MOVE_DURATION_S)
            if ok:
                now = time.monotonic()
                auto["pose_idx"] = idx
                auto["phase"] = "settling"
                auto["settle_until"] = now + AUTO_SETTLE_EXTRA_S
                auto["timeout_at"] = auto["settle_until"] + AUTO_MARKER_TIMEOUT_S
                auto["stable_frames"] = 0
                auto["status"] = f"pose {idx+1}/{total} moving"
                return False

            print(f"[Auto] Pose {idx+1}/{total} is not reachable, skipping")
            auto["idx"] += 1

        auto["phase"] = "done"
        auto["finished"] = True
        auto["status"] = "all poses completed"
        print("\n[Auto] All preset poses completed")
        return True

    def tick_auto(cur) -> bool:
        if not auto["enabled"] or auto["finished"]:
            return auto["finished"]

        if auto["phase"] == "idle":
            return start_next_auto_pose()

        pose_idx = auto["pose_idx"]
        total = len(calib_poses)
        now = time.monotonic()

        if auto["phase"] == "settling":
            remain = auto["settle_until"] - now
            if remain > 0.0:
                auto["stable_frames"] = 0
                auto["status"] = f"pose {pose_idx+1}/{total} moving/settling {remain:.1f}s"
                return False
            auto["phase"] = "searching"

        if cur is not None:
            auto["stable_frames"] += 1
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = (
                f"pose {pose_idx+1}/{total} marker stable "
                f"{auto['stable_frames']}/{AUTO_MARKER_STABLE_FRAMES}  remaining {remain:.1f}s"
            )
            if auto["stable_frames"] >= AUTO_MARKER_STABLE_FRAMES:
                capture_sample(cur, f"auto pose {pose_idx+1}/{total}")
                auto["idx"] += 1
                auto["phase"] = "idle"
                auto["stable_frames"] = 0
                return start_next_auto_pose()
        else:
            auto["stable_frames"] = 0
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = f"pose {pose_idx+1}/{total} waiting for ArUco {remain:.1f}s"

        if now >= auto["timeout_at"]:
            print(f"[Auto] Pose {pose_idx+1}/{total} timed out without ArUco, skipping")
            auto["idx"] += 1
            auto["phase"] = "idle"
            auto["stable_frames"] = 0
            return start_next_auto_pose()

        return False

    def shutdown_robot() -> None:
        """Leave hand guiding if needed, then release the arm connection."""
        if robot is None:
            return
        if args.manual:
            try:
                robot.exit_drag_mode()
            except Exception as e:
                print(f"[Robot] Leaving hand-guiding failed: {e}")
        try:
            robot.end()
        except Exception as e:
            print(f"[Robot] Disconnect failed: {e}")

    def handle_line(raw: str) -> bool:
        if raw is None:
            print("\n[Interrupt] Terminal input closed; stopping and trying to solve")
            return True

        line = raw.strip().lower()

        if line in {"q", "c"}:
            return True

        if line == "pos":
            _print_fk()
            return False

        if args.manual and line == "":
            capture_sample(latest_pose, "manual capture")
            return False

        if line:
            if args.manual:
                print("  Manual commands: Enter=capture  c/q=finish and solve  pos=print current TCP pose")
            else:
                print("  Auto commands: c/q=finish and solve  pos=print current TCP pose")
        return False

    # Main loop.
    WIN = "Eye-in-Hand Calibration  (operate in terminal)"
    show_window = bool(os.environ.get("DISPLAY"))
    finish_reason = "normal finish"
    try:
        if show_window:
            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

        while True:
            bgr, depth = cam.get_frame()
            if bgr is not None:
                pose = cam.detect_aruco(bgr)
                latest_pose = pose
                latest_depth = depth
                if tick_auto(pose):
                    finish_reason = "auto traversal completed"
                vis  = cam.draw_aruco(bgr)
                n    = calibrator.n_samples

                def osd(text, y, color=(220, 220, 220)):
                    cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 1, cv2.LINE_AA)

                if pose:
                    if args.manual:
                        osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  "
                            f"samples:{n}  Enter=capture  c/q=finish",
                            28, (80, 220, 80))
                    else:
                        osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  samples:{n}",
                            28, (80, 220, 80))
                    zc = _depth_at_marker(latest_depth, pose)
                    if zc is not None:
                        zp = float(pose.T_marker2cam[2, 3])
                        off = zp > 1e-3 and abs(zc / zp - 1.0) > 0.08
                        osd(f"depth z={zc:.3f}m  pnp z={zp:.3f}m"
                            + ("  <- scale mismatch!" if off else ""),
                            110, (60, 60, 220) if off else (200, 200, 200))
                else:
                    if args.manual:
                        osd(f"No marker  samples:{n}  move arm to see marker",
                            28, (80, 80, 220))
                    else:
                        osd(f"No marker  samples:{n}",
                            28, (80, 80, 220))

                if args.manual:
                    osd("MANUAL: Enter=capture  c/q=finish  pos=print fk", 50, (180, 180, 60))
                else:
                    osd(f"AUTO: {auto['status']}", 50, (180, 180, 60))

                filled = min(n, 15) * (400 // 15)
                cv2.rectangle(vis, (10, 70), (10 + filled, 82), (0, 200, 100), -1)
                cv2.rectangle(vis, (10, 70), (410, 82), (160, 160, 160), 1)
                cv2.putText(vis, f"{n}/15", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                mode_label = "MANUAL(HandGuide)" if args.manual else "AUTO"
                cv2.putText(vis, mode_label, (vis.shape[1] - 200, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1)

                if show_window:
                    cv2.imshow(WIN, vis)
                if web is not None:
                    web.set_frame(vis)
                    tcp = ""
                    try:
                        if robot is not None:
                            _t = robot.get_tcp_pose()[:3, 3]
                            tcp = f"{_t[0]:+.3f} {_t[1]:+.3f} {_t[2]:+.3f} m"
                    except Exception:
                        tcp = "unavailable"
                    web.set_status({
                        "mode": mode_label,
                        "samples": f"{n} / 15 recommended (min {MIN_CALIB_SAMPLES})",
                        "progress": auto["status"] if not args.manual else "manual",
                        "marker": (f"id={pose.id} z={pose.T_marker2cam[2,3]:.3f} m"
                                   if pose else "not visible"),
                        "tcp": tcp,
                    })

            if show_window:
                if cv2.waitKey(30) & 0xFF in [ord('q'), ord('Q'), 27]:
                    finish_reason = "window exit"
                    break
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    finish_reason = "window closed"
                    break
            else:
                time.sleep(0.03)

            if web is not None:
                cmd = web.pop_command()
                if cmd is not None:
                    mapped = {"capture": "", "finish": "q", "pos": "pos"}.get(cmd, cmd)
                    if handle_line(mapped):
                        finish_reason = "stopped from the web console"
                        break

            try:
                if line_queue is not None and handle_line(line_queue.get_nowait()):
                    finish_reason = "user interrupted"
                    break
            except queue.Empty:
                pass

            if auto["finished"]:
                break

    except KeyboardInterrupt:
        finish_reason = "Ctrl+C interrupt"
        print("\n[Ctrl+C] Stopping and trying to solve")

    finally:
        if show_window:
            cv2.destroyAllWindows()
        cam.close()
        shutdown_robot()
        compute_and_save(finish_reason)

    print(f"\nDone, total samples: {calibrator.n_samples}.")
    if calibrator.n_samples > 0 and not result_saved:
        print("Tip: hand_eye.npz was not generated; collect more samples and try again.")

    if web is not None:
        # Give the browser a couple of poll cycles to show the result before exit.
        print("[Web] Console closing in 10s...")
        time.sleep(10.0)
        sys.stdout = sys.__stdout__
        web.stop()


if __name__ == "__main__":
    main()
    os._exit(0)
