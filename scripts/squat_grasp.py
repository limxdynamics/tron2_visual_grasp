"""Watch for a target, drive up to it, squat, grasp it with the wrist camera,
release it at the drop pose, then stand up.

The approach is a visual servo on the watch camera (mission.approach); with it
disabled the base never moves and the target must already be within reach.

Usage:
    ./run.sh grasp
    ./run.sh grasp --target-class "tennis ball" --grasp-timeout 30
    ./run.sh dry-run
    ./run.sh demo          # hardware-free: simulated TRON2 and camera (--sim)
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.camera import CameraDriver, make_camera  # noqa: E402
from drivers.robot import RobotDriver, make_robot  # noqa: E402
from utils.camera_utils import compose_cam_to_base_transform, load_config, load_hand_eye  # noqa: E402
from utils.ordinary_grasp import GraspPose, draw_grasp, estimate_grasps, select_best_grasp  # noqa: E402
from utils.targets import (  # noqa: E402
    Target,
    match_target_class,
    normalize_target_classes,
    select_target,
)
from utils.transforms import (  # noqa: E402
    canonicalize_parallel_gripper_tcp_rotation,
    class_grasp_offset_m,
    mat4_to_pose6d,
    offset_pose6d,
    pose6d_to_mat4,
    rotation_matrix_to_euler_zyx,
    transform_grasp_pose_to_base,
)
from utils.visual_servo import (  # noqa: E402
    ApproachControlConfig,
    ArrivalTracker,
    BaseTwistStreamer,
    aim_column_px,
    compute_approach_twist,
    horizontal_error,
)
from utils.web_debug import WebDebug  # noqa: E402
from utils.detection import Detection, detect_objects, load_detector  # noqa: E402

Detector = Callable[[np.ndarray], tuple[Any, list[Detection]]]


@contextlib.contextmanager
def scene_pump(robot: RobotDriver, web: Optional[WebDebug]) -> Iterator[None]:
    """Stream robot-provided debug views (the simulator's 3D and top views) to the web console.

    Each view gets its own thread: the 3D renderer must stay on the thread that created it.
    """
    views = robot.debug_views() if web is not None and hasattr(robot, "debug_views") else []
    stop = threading.Event()

    def run(name: str, render: Callable[[], np.ndarray], hz: float) -> None:
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                web.set_frame(render(), name)
            except Exception as exc:
                print(f"[Web] View {name!r} stopped: {exc}")
                return
            stop.wait(max(0.0, 1.0 / hz - (time.monotonic() - t0)))

    threads = [threading.Thread(target=run, args=(name, render, hz), name=f"view-{name}", daemon=True)
               for name, _, render, hz in views]
    for thread in threads:
        thread.start()
    try:
        yield
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)


@contextlib.contextmanager
def preview_pump(
    camera: Optional[CameraDriver],
    web: Optional[WebDebug],
    view: str,
) -> Iterator[None]:
    """Keep the web preview updating while the caller blocks on robot motion."""
    if camera is None or web is None:
        yield
        return

    stop = threading.Event()

    def run() -> None:
        while not stop.is_set():
            try:
                color_bgr, _ = camera.get_frame()
            except Exception:
                return
            if color_bgr is not None:
                web.set_frame(color_bgr, view)

    thread = threading.Thread(target=run, name="preview-pump", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch, squat, grasp, stand up")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--target-class", default=None, help="Object class to grasp")
    parser.add_argument("--dry-run", action="store_true", help="Run perception without moving the robot")
    parser.add_argument("--display", action="store_true", help="Enable local OpenCV preview windows")
    parser.add_argument("--web-port", type=int, default=None, help="Web debug port; 0 disables it")
    parser.add_argument("--grasp-timeout", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--sim", action="store_true",
                        help="Hardware-free demo: simulated TRON2 and wrist camera (drivers/sim)")
    parser.add_argument("--cycles", type=int, default=0, metavar="N",
                        help="Stop after N grasp cycles; 0 follows mission.repeat")
    return parser.parse_args()


def _camera_config(cfg: dict[str, Any], camera_cfg: dict[str, Any]) -> dict[str, Any]:
    configured = copy.deepcopy(cfg)
    base_camera = dict(configured.get("camera") or {})
    base_camera.update(camera_cfg)
    configured["camera"] = base_camera
    return configured


def _target_text(target: Optional[Target]) -> str:
    if target is None:
        return "--"
    depth = "--" if target.depth_m is None else f"{target.depth_m:.2f}m"
    return (
        f"{target.detection.class_name} conf={target.detection.conf:.2f} "
        f"depth={depth} x={target.center_px[0]}"
    )


def _draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    target: Optional[Target],
    hits: int,
) -> np.ndarray:
    display = image.copy()
    selected = target.detection if target is not None else None
    for detection in detections:
        color = (0, 255, 0) if detection is selected else (0, 185, 255)
        thickness = 3 if detection is selected else 2
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            display,
            f"{detection.class_name} {detection.conf:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    depth_text = "--" if target is None or target.depth_m is None else f"{target.depth_m:.2f}m"
    cv2.putText(
        display,
        f"depth={depth_text} hits={hits}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return display


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        draw_grasp(display, grasp)

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if best is not None:
        x_m, y_m, z_m = best.position.tolist()
        best_text = (
            f"best={best.class_name} conf={best.conf:.2f} "
            f"xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f}) jaw={best.jaw_width_m * 100:.1f}cm"
        )
        cv2.putText(
            display,
            best_text,
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 140),
            2,
        )
    return display


def wait_for_target(
    camera: CameraDriver,
    detect: Detector,
    target_classes: Sequence[str],
    priority: str,
    on_frames: int,
    window_frames: int,
    show_window: bool,
    web: Optional[WebDebug],
    stop: threading.Event,
) -> bool:
    """Hold still until a target is seen in `on_frames` of the last `window_frames` frames."""
    target_class = ", ".join(target_classes)
    on_frames = max(1, int(on_frames))
    recent: deque[bool] = deque(maxlen=max(on_frames, int(window_frames)))
    print(f"[Watch] Watching for {target_class!r}")
    if web is not None:
        web.add_event(f"Watching for {target_class!r}")
    while not stop.is_set():
        color_bgr, depth_mm = camera.get_frame()
        if color_bgr is None:
            continue
        _, detections = detect(color_bgr)
        target = select_target(
            detections, depth_mm, target_classes, priority=priority,
            frame_width=color_bgr.shape[1],
        )
        recent.append(target is not None)
        hits = sum(recent)

        if show_window or web is not None:
            display = _draw_detections(color_bgr, detections, target, hits)
            if web is not None:
                web.set_frame(display, "watch")
                web.set_status({
                    "stage": "watching",
                    "targets": target_class,
                    "selected": _target_text(target),
                    "detections": len(detections),
                    "hits": f"{hits}/{on_frames}",
                })
                if web.pop_command() in ("stop", "q"):
                    web.add_event("Mission stopped from web console", "warn")
                    stop.set()
                    return False
            if show_window:
                cv2.imshow("Watch", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    stop.set()
                    return False

        if hits >= on_frames:
            print(f"[Watch] Target in view: {_target_text(target)}")
            if web is not None:
                web.add_event(f"Target in view: {_target_text(target)}", "ok")
            if show_window:
                cv2.destroyWindow("Watch")
            return True
    return False


def run_approach(
    camera: CameraDriver,
    detect: Detector,
    robot: Optional[RobotDriver],
    target_classes: Sequence[str],
    priority: str,
    control: ApproachControlConfig,
    timeout_s: float,
    twist_hz: float,
    show_window: bool,
    web: Optional[WebDebug],
    stop: threading.Event,
) -> bool:
    """Servo the base onto the target until it sits at the stop line; robot=None is dry-run."""
    target_class = ", ".join(target_classes)
    tracker = ArrivalTracker(control)
    streamer = BaseTwistStreamer(robot, twist_hz, enabled=robot is not None)
    deadline = time.monotonic() + timeout_s
    last_report = 0.0
    held_forward = 0.0
    print(f"[Approach] Driving to {target_class!r}")
    if web is not None:
        web.add_event(f"Approaching {target_class!r}")
    streamer.start()
    try:
        while time.monotonic() < deadline and not stop.is_set():
            streamer.raise_if_failed()
            color_bgr, depth_mm = camera.get_frame()
            if color_bgr is None:
                streamer.set((0.0, 0.0, 0.0))
                continue
            _, detections = detect(color_bgr)
            target = select_target(
                detections, depth_mm, target_classes, priority=priority,
                frame_width=color_bgr.shape[1],
            )
            command, frame_arrived = compute_approach_twist(
                target, color_bgr.shape[1], control, frame_height=color_bgr.shape[0],
            )
            arrived, reason = tracker.update(target, frame_arrived)
            if target is None and 0 < tracker.lost_frames <= control.hold_lost_frames:
                command = (held_forward, 0.0, 0.0)
            else:
                held_forward = command[0] if target is not None else 0.0
            streamer.set(command)
            h_err = (
                "--" if target is None
                else f"{horizontal_error(target, color_bgr.shape[1], control):+.3f}"
            )
            twist = f"vx={command[0]:+.2f} yaw={command[2]:+.2f}"

            now = time.monotonic()
            if now - last_report >= 0.5:
                print(
                    f"[Approach] {_target_text(target)} h_err={h_err} {twist} "
                    f"stable={tracker.stable_frames} lost={tracker.lost_frames}"
                )
                last_report = now

            if show_window or web is not None:
                display = _draw_detections(color_bgr, detections, target, tracker.stable_frames)
                aim_x = int(round(aim_column_px(color_bgr.shape[1], control)))
                cv2.line(display, (aim_x, 0), (aim_x, display.shape[0]), (255, 120, 0), 1)
                if control.stop_bottom_ratio is not None:
                    stop_y = int(display.shape[0] * control.stop_bottom_ratio)
                    cv2.line(display, (0, stop_y), (display.shape[1], stop_y), (255, 120, 0), 1)
                if web is not None:
                    web.set_frame(display, "watch")
                    web.set_status({
                        "stage": "approaching",
                        "mode": "dry-run" if robot is None else "execute",
                        "targets": target_class,
                        "selected": _target_text(target),
                        "twist": twist,
                        "h_err": h_err,
                        "stable_frames": tracker.stable_frames,
                        "lost_frames": tracker.lost_frames,
                    })
                    if web.pop_command() in ("stop", "q"):
                        web.add_event("Mission stopped from web console", "warn")
                        stop.set()
                        return False
                if show_window:
                    cv2.imshow("Approach", display)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        stop.set()
                        return False

            if arrived:
                print(f"[Approach] Arrived ({reason})")
                if web is not None:
                    web.add_event(f"Approach arrived: {reason}", "ok")
                return True
        if not stop.is_set():
            print(f"[Approach] Timed out after {timeout_s:.1f}s")
            if web is not None:
                web.add_event(f"Approach timed out after {timeout_s:.1f}s", "warn")
        return False
    finally:
        streamer.close()
        if show_window:
            cv2.destroyWindow("Approach")


def _kneel(
    robot: RobotDriver,
    dwell_s: float,
    body_pitch_rad: float,
    pitch_dwell_s: float,
    dry_run: bool,
    pitch_reset_dwell_s: float = 1.0,
) -> None:
    """Kneel fully, then pitch the body; returns only after both waits have elapsed.

    The body only pitches once the kneel has finished. Pitch is relative and
    the WF stand levels the body without clearing that setpoint: with no reset
    a second-cycle 0.49 rad never moved, so reset right before pitching.
    Neither posture API reports when the body stops moving, so the dwells are
    the only guarantee that the grasp starts from a settled body.
    """
    print("[Posture] Kneel")
    if not dry_run:
        robot.kneel()
        print(f"[Posture] Wait {dwell_s:.2f}s for the kneel to finish")
        time.sleep(dwell_s)
    print(f"[Posture] Clear the pitch setpoint, then pitch {body_pitch_rad:.2f} rad")
    if not dry_run:
        robot.reset_body_pitch()
        print(f"[Posture] Wait {pitch_reset_dwell_s:.2f}s after the pitch reset")
        time.sleep(pitch_reset_dwell_s)
        robot.set_body_pitch(body_pitch_rad)
        print(f"[Posture] Wait {pitch_dwell_s:.2f}s for the body pitch to settle")
        time.sleep(pitch_dwell_s)


def _stand_up(
    robot: RobotDriver,
    stand_dwell_s: float,
) -> None:
    """Fold the arm, then request the standing posture.

    The WF stand levels the body pitch itself. The arm stays parked through the
    stand: in testing it ignored joint commands during and for seconds after it.
    """
    if not robot.move_park():
        raise RuntimeError("Arm not parked; cannot start standing or the next cycle")
    print("[Posture] Stand")
    robot.stand()
    print(f"[Posture] Wait {stand_dwell_s:.2f}s for the stand to settle")
    time.sleep(stand_dwell_s)


def _describe_grasp_failure(
    grasps: list[Any],
    matching: list[Any],
    target_class: str,
) -> str:
    """Say which of the three failure modes stopped the grasp, for the timeout log."""
    if not grasps:
        return "the wrist camera detected nothing"
    if not matching:
        seen = sorted({grasp.class_name for grasp in grasps})
        return f"detected {seen} but nothing matched {target_class!r}"
    reasons = ", ".join(
        f"{grasp.class_name}({grasp.rejected_reason or 'invalid'}, "
        f"{grasp.valid_depth_pixels}px depth)"
        for grasp in matching
    )
    return f"matched {len(matching)} but all were rejected: {reasons}"


def _dump_grasp_solve(tcp_to_base, hand_eye, camera_to_base, best) -> None:
    """Print the camera->base solve so a mis-aimed grasp can be traced to its
    source (arm TCP pose, hand-eye, or the back-projected object point).
    Set GRASP_DEBUG=0 to silence."""
    def _rpy_deg(T):
        return np.degrees(rotation_matrix_to_euler_zyx(np.asarray(T)[:3, :3]))

    p_cam = np.asarray(best.position, dtype=float).reshape(3)
    cb = np.asarray(camera_to_base, float)
    p_base = (cb @ np.array([*p_cam, 1.0]))[:3]
    cam_pos = cb[:3, 3]
    optical = cb[:3, 2]  # camera +Z (optical/depth axis) in base
    optical = optical / max(np.linalg.norm(optical), 1e-9)
    pitch_deg = float(np.degrees(np.arcsin(-optical[2])))  # +down tilt of the view ray
    print("[Solve] ---- camera->base grasp solve ----")
    print(f"[Solve] arm TCP  pos={np.round(tcp_to_base[:3, 3], 4).tolist()} "
          f"rpy_deg={np.round(_rpy_deg(tcp_to_base), 1).tolist()}")
    print(f"[Solve] hand_eye pos={np.round(np.asarray(hand_eye)[:3, 3], 4).tolist()} "
          f"rpy_deg={np.round(_rpy_deg(hand_eye), 1).tolist()}")
    print(f"[Solve] camera   pos={np.round(cam_pos, 4).tolist()} "
          f"optical_axis(+Z)_in_base={np.round(optical, 3).tolist()} downtilt={pitch_deg:+.0f}deg")
    print(f"[Solve] object   cam={np.round(p_cam, 4).tolist()} base={np.round(p_base, 4).tolist()}")
    print("[Solve] sanity: base +x=forward +y=left +z=up. A look-down wrist should give "
          "optical_axis ~[+0.6..0.8, ~0, -0.6..-0.8] (downtilt ~+40-60deg); object base +x "
          "forward, +y ~0 for a centred target, +z near the floor.")


def _find_best_grasp(
    results: Any,
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
    grasp_cfg: dict[str, Any],
    cam_to_base: Optional[np.ndarray],
    target_classes: Sequence[str],
) -> tuple[list[Any], list[Any], Any]:
    grasps = estimate_grasps(
        results,
        depth_mm,
        camera_matrix,
        depth_quantile=float(grasp_cfg.get("depth_quantile", 0.75)),
        grasp_point_table=grasp_cfg.get("class_grasp_point"),
        cam_to_base=cam_to_base,
    )
    matching = [
        grasp for grasp in grasps
        if match_target_class(grasp.class_name, target_classes) is not None
    ]
    return grasps, matching, select_best_grasp(matching)


def _solve_grasp_pose(
    best: Any,
    tcp_to_base: np.ndarray,
    hand_eye: np.ndarray,
    cfg: dict[str, Any],
    grasp_cfg: dict[str, Any],
    quiet: bool = False,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Camera-frame grasp -> base-frame (grasp6d, pregrasp6d), class offset included."""
    camera_to_base = compose_cam_to_base_transform(tcp_to_base, hand_eye, cfg)
    if not quiet and os.environ.get("GRASP_DEBUG", "1") != "0":
        _dump_grasp_solve(tcp_to_base, hand_eye, camera_to_base, best)
    min_base_z_m = float(grasp_cfg.get("min_base_z_m", 0.0))
    grasp6d, pregrasp6d = transform_grasp_pose_to_base(
        best.position,
        best.tcp_rotation,
        camera_to_base,
        float(grasp_cfg.get("pregrasp_offset_m", 0.08)),
        insertion_depth_m=float(grasp_cfg.get("insertion_depth_m", 0.0)),
        top_down=bool(grasp_cfg.get("top_down", True)),
        current_tool_rotation=tcp_to_base[:3, :3],
        min_base_z_m=min_base_z_m,
        orientation_mode=str(grasp_cfg.get("orientation_mode", "keep")).lower(),
        hand_eye_rotation=hand_eye[:3, :3],
        open_flip=bool(grasp_cfg.get("gripper_open_flip", False)),
    )
    class_offset = class_grasp_offset_m(grasp_cfg, best.class_name)
    if any(class_offset):
        grasp6d = offset_pose6d(grasp6d, class_offset, min_base_z_m)
        pregrasp6d = offset_pose6d(pregrasp6d, class_offset, min_base_z_m)
        if not quiet:
            print(
                f"[Grasp] {best.class_name} offset "
                f"dx={class_offset[0]:+.3f} dy={class_offset[1]:+.3f} "
                f"dz={class_offset[2]:+.3f} m"
            )
    return grasp6d, pregrasp6d


def _print_best_grasp(grasp: GraspPose) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print("\n[G] Best grasp:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def _execute_grasp(
    robot: RobotDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    dry_run: bool,
    outcome: Optional[dict[str, Any]] = None,
    move_duration_s: float = 1.5,
    open_first: bool = True,
) -> bool:
    """Run one grasp, then park the arm; outcome["reason"] explains a failure for the web UI."""
    def _set_reason(reason: str) -> None:
        if outcome is not None:
            outcome["reason"] = reason

    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[Grasp] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    if dry_run:
        print("[Grasp] dry run; skip motion")
        _set_reason("dry run; no motion attempted")
        return False

    if open_first:
        print("[Grasp] Open gripper")
        robot.open_gripper()

    # Pregrasp temporarily disabled: current pregrasp pose is wrong, go straight to grasp.
    # print("[Grasp] Move to pregrasp")
    # if not robot.move_to_pose(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
    #     print("[Grasp] Pregrasp unreachable")
    #     return False

    print("[Grasp] Move to grasp")
    if not robot.move_to_pose(xg, yg, zg, rxg, ryg, rzg, duration=move_duration_s):
        print("[Grasp] Grasp unreachable")
        _set_reason(
            f"grasp pose unreachable: xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f}) "
            f"rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})"
        )
        return False

    print("[Grasp] Closing")
    ok = robot.grasp()
    print("[Grasp] Holding object" if ok else "[Grasp] Empty grasp")
    _set_reason("holding object" if ok else "jaws closed empty; object missed or too thin")

    # Keep holding while the arm parks; the mission releases at the drop pose, before standing.
    print("[Grasp] Park the arm")
    robot.move_park()
    return ok


def _supports_servo(robot: RobotDriver) -> bool:
    return type(robot).servo_pose is not RobotDriver.servo_pose


def _track_and_grasp(
    cfg: dict[str, Any],
    camera: CameraDriver,
    detect: Detector,
    robot: RobotDriver,
    hand_eye: np.ndarray,
    target_classes: Sequence[str],
    first_grasp6d: tuple[float, ...],
    grasp_cfg: dict[str, Any],
    show_window: bool,
    web: Optional[WebDebug],
    stop: threading.Event,
) -> bool:
    """Servo the TCP onto the target while re-detecting it every frame, so a
    rolling object is followed instead of grasped at its first-seen position.

    The orientation is locked to the first solve: in align mode the approach is
    the camera axis, so re-solving it while the camera moves would feed back.
    Once within commit distance (or the wrist camera loses the object up close)
    the last goal is finished with a short blocking move and the jaws close.
    """
    track_cfg = grasp_cfg.get("tracking") or {}
    max_speed = float(track_cfg.get("max_speed_m_s", 0.12))
    max_rot_speed = float(track_cfg.get("max_rot_speed_rad_s", 0.8))
    smoothing = float(np.clip(float(track_cfg.get("smoothing", 0.5)), 0.0, 0.95))
    deadband = float(track_cfg.get("goal_deadband_m", 0.008))
    commit_dist = float(track_cfg.get("commit_dist_m", 0.05))
    blind_commit_dist = float(track_cfg.get("blind_commit_dist_m", 0.12))
    lost_timeout = float(track_cfg.get("lost_timeout_s", 1.0))
    timeout = float(track_cfg.get("timeout_s", 10.0))
    final_duration = float(track_cfg.get("final_duration_s", 0.5))
    camera_matrix = camera.K.astype(np.float32)
    target_class = ", ".join(target_classes)

    T_goal = pose6d_to_mat4(*first_grasp6d)
    R_goal = T_goal[:3, :3].copy()
    goal_p = T_goal[:3, 3].copy()
    sent_p: Optional[np.ndarray] = None

    print("[Track] Open gripper")
    # No settle wait: the jaws finish opening while the arm travels.
    robot.open_gripper(timeout=0.0)
    print(f"[Track] Following {target_class!r}")
    if web is not None:
        web.add_event(f"Tracking {target_class!r} during approach")

    deadline = time.monotonic() + timeout
    last_seen = time.monotonic()
    last_report = 0.0
    commit_reason = ""
    try:
        while not stop.is_set():
            now = time.monotonic()
            if now > deadline:
                print(f"[Track] Timed out after {timeout:.1f}s")
                if web is not None:
                    web.add_event(f"Tracking timed out after {timeout:.1f}s", "err")
                return False

            color_bgr, depth_mm = camera.get_frame()
            # Read the pose right after the frame so the solve matches the image.
            tcp_to_base = robot.get_tcp_pose()
            best = None
            grasps: list[Any] = []
            if color_bgr is not None and depth_mm is not None:
                results, _ = detect(color_bgr)
                cam_to_base = compose_cam_to_base_transform(tcp_to_base, hand_eye, cfg)
                grasps, _, best = _find_best_grasp(
                    results, depth_mm, camera_matrix, grasp_cfg, cam_to_base, target_classes)
            if best is not None:
                grasp6d, _ = _solve_grasp_pose(
                    best, tcp_to_base, hand_eye, cfg, grasp_cfg, quiet=True)
                goal_p = smoothing * goal_p + (1.0 - smoothing) * np.asarray(grasp6d[:3])
                last_seen = now

            dist = float(np.linalg.norm(goal_p - tcp_to_base[:3, 3]))
            lost_s = now - last_seen
            if best is not None and dist <= commit_dist:
                commit_reason = f"within {dist * 1000:.0f}mm"
            elif best is None and dist <= blind_commit_dist:
                commit_reason = f"object out of view at {dist * 1000:.0f}mm"
            elif lost_s > lost_timeout:
                print(f"[Track] Lost {target_class!r} for {lost_s:.1f}s at "
                      f"{dist * 1000:.0f}mm; abort")
                if web is not None:
                    web.add_event(f"Tracking lost {target_class!r}", "err")
                return False
            if commit_reason:
                break

            # Re-send only on a real move: detection noise retargeting every frame shakes the arm.
            if sent_p is None or float(np.linalg.norm(goal_p - sent_p)) > deadband:
                T_goal[:3, :3] = R_goal
                T_goal[:3, 3] = goal_p
                robot.servo_pose(*mat4_to_pose6d(T_goal), max_speed_m_s=max_speed,
                                 max_rot_speed_rad_s=max_rot_speed)
                sent_p = goal_p.copy()

            if now - last_report >= 0.5:
                print(f"[Track] goal=({goal_p[0]:+.3f},{goal_p[1]:+.3f},{goal_p[2]:+.3f}) "
                      f"dist={dist * 1000:.0f}mm {'seen' if best is not None else f'lost {lost_s:.1f}s'}")
                last_report = now
            if color_bgr is not None and (show_window or web is not None):
                display = _render_display(
                    color_bgr, grasps, best, f"TRACK dist={dist * 1000:.0f}mm")
                if web is not None:
                    web.set_frame(display, "wrist")
                    web.set_status({
                        "stage": "tracking approach",
                        "target": target_class,
                        "dist_mm": f"{dist * 1000:.0f}",
                        "seen": best is not None,
                    })
                    if web.pop_command() in ("stop", "q"):
                        web.add_event("Mission stopped from web console", "warn")
                        stop.set()
                        return False
                if show_window:
                    cv2.imshow("Grasp - wrist", display)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        stop.set()
                        return False
    finally:
        robot.stop_servo()

    if stop.is_set():
        return False
    print(f"[Track] Commit: {commit_reason}")
    print(f"[Timing] tracking: {time.monotonic() - (deadline - timeout):.2f}s")
    T_goal[:3, :3] = R_goal
    T_goal[:3, 3] = goal_p
    final6d = mat4_to_pose6d(T_goal)
    with preview_pump(camera, web, "wrist"):
        outcome: dict[str, Any] = {}
        executed = _execute_grasp(robot, final6d, final6d, dry_run=False, outcome=outcome,
                                  move_duration_s=final_duration, open_first=False)
    reason = outcome.get("reason", "unknown")
    if web is not None:
        web.add_event(
            f"Grasp {'succeeded' if executed else 'failed'}: {reason}",
            "ok" if executed else "err",
        )
    return executed


def run_single_grasp(
    cfg: dict[str, Any],
    camera: CameraDriver,
    detect: Detector,
    robot: Optional[RobotDriver],
    target_classes: Sequence[str],
    timeout_s: float,
    dry_run: bool,
    show_window: bool,
    web: Optional[WebDebug],
    stop: threading.Event,
) -> bool:
    target_class = ", ".join(target_classes)
    cam_type = str((cfg.get("camera") or {}).get("type", "")).lower()
    hand_eye = None
    if robot is not None:
        hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type, cfg)
        if hand_eye is None or hand_eye_mode != "eye_in_hand":
            raise RuntimeError(f"Eye-in-hand calibration is unavailable for {cam_type}")

    camera_matrix = camera.K.astype(np.float32)
    grasp_cfg = (cfg.get("grasp_pipeline") or {}).get("grasp") or {}
    deadline = time.monotonic() + timeout_s
    last_failure = ""
    last_report = 0.0

    # The arm holds the observe pose for the whole loop, so one read is enough.
    # This is what tells the grasp point which short edge faces the robot; the
    # depth image cannot, since looking straight down puts both ends at the
    # same range and the comparison decays into sensor noise.
    grasp_cam_to_base = None
    if robot is not None and hand_eye is not None:
        try:
            grasp_cam_to_base = compose_cam_to_base_transform(
                robot.get_tcp_pose(), hand_eye, cfg)
        except Exception as exc:
            print("[Grasp] Base transform unavailable; near-edge choice falls "
                  f"back to camera depth: {exc}")

    try:
        while time.monotonic() < deadline:
            color_bgr, depth_mm = camera.get_frame()
            if color_bgr is None or depth_mm is None:
                continue
            results, _ = detect(color_bgr)
            grasps, matching, best = _find_best_grasp(
                results, depth_mm, camera_matrix, grasp_cfg, grasp_cam_to_base, target_classes)
            failure = (
                "" if best is not None
                else _describe_grasp_failure(grasps, matching, target_class)
            )
            if failure:
                last_failure = failure

            display = None
            if show_window or web is not None:
                display = _render_display(
                    color_bgr, grasps, best, f"AUTO target={target_class}"
                )
            if web is not None:
                web.set_frame(display, "wrist")
                web.set_status({
                    "stage": "wrist grasp detection",
                    "mode": "dry-run" if dry_run else "execute",
                    "target": target_class,
                    "grasp_candidates": len(grasps),
                    "best": best.class_name if best is not None else "--",
                    "why_not": failure or "--",
                })
                if web.pop_command() in ("stop", "q"):
                    web.add_event("Mission stopped from web console", "warn")
                    stop.set()
                    return False
            if show_window:
                cv2.imshow("Grasp - wrist", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    stop.set()
                    return False
            if best is None:
                now = time.monotonic()
                if now - last_report >= 0.5:
                    print(f"[Grasp] {failure}")
                    last_report = now
                continue

            _print_best_grasp(best)
            if web is not None:
                # Frozen copy of the frame the grasp was computed from; the live
                # wrist view keeps moving once the arm starts.
                web.set_frame(display, "grasp_shot")
                web.add_event(
                    f"Wrist camera selected {best.class_name} at "
                    f"({best.position[0]:+.3f}, {best.position[1]:+.3f}, "
                    f"{best.position[2]:+.3f})m",
                    "ok",
                )
            if robot is None:
                print("[Grasp] Dry-run perception OK; base-frame transform skipped")
                return True
            tcp_to_base = robot.get_tcp_pose()
            grasp6d, pregrasp6d = _solve_grasp_pose(best, tcp_to_base, hand_eye, cfg, grasp_cfg)
            tracking = bool((grasp_cfg.get("tracking") or {}).get("enabled", False))
            if tracking and not dry_run and _supports_servo(robot):
                return _track_and_grasp(
                    cfg, camera, detect, robot, hand_eye, target_classes, grasp6d,
                    grasp_cfg, show_window, web, stop,
                )
            with preview_pump(camera, web, "wrist"):
                outcome: dict[str, Any] = {}
                executed = _execute_grasp(robot, grasp6d, pregrasp6d,
                                          dry_run=dry_run, outcome=outcome,
                                          move_duration_s=float(grasp_cfg.get("move_duration_s", 1.5)))
            reason = outcome.get("reason", "unknown")
            if web is not None:
                web.add_event(
                    f"Grasp {'succeeded' if executed else 'failed'}: {reason}",
                    "ok" if executed else "err",
                )
            return executed
        print(f"[Grasp] No valid {target_class!r} grasp found within {timeout_s:.1f}s")
        if last_failure:
            print(f"[Grasp] Last frame: {last_failure}")
        if web is not None:
            web.add_event(
                f"No valid {target_class!r} grasp before timeout"
                f"{f'; {last_failure}' if last_failure else ''}",
                "warn",
            )
        return False
    finally:
        if show_window:
            cv2.destroyWindow("Grasp - wrist")


def _release_model(model: Any) -> None:
    if hasattr(model, "close"):
        model.close()
    del model
    gc.collect()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(config_path)
    if args.sim:
        from drivers.sim import sim_config
        cfg = sim_config(cfg)
        print("[Mission] SIMULATION: no robot or camera is used")
    mission_cfg = cfg.get("mission") or {}

    target_classes = normalize_target_classes(args.target_class or mission_cfg.get("target_class"))
    if not target_classes:
        raise ValueError("Set mission.target_class or pass --target-class")
    priority = str(mission_cfg.get("target_priority", "nearest")).lower()
    target_label = ", ".join(target_classes)
    print(f"[Mission] Targets: {target_label} (priority={priority})")
    if str((cfg.get("robot") or {}).get("type", "")).lower() not in ("tron2", "tron2_sim"):
        raise ValueError("The squat-and-grasp mission requires robot.type=tron2")

    posture_cfg = mission_cfg.get("posture") or {}
    stand_dwell_s = float(posture_cfg.get("stand_dwell_s", 0.45))
    kneel_dwell_s = float(posture_cfg.get("kneel_dwell_s", 0.6))
    pitch_dwell_s = float(posture_cfg.get("pitch_dwell_s", 2.0))
    grasp_timeout = float(args.grasp_timeout or mission_cfg.get("grasp_timeout_s", 15.0))
    web_port = int(args.web_port if args.web_port is not None else mission_cfg.get("web_port", 8888))
    show_window = args.display and bool(os.environ.get("DISPLAY"))

    watch_camera_cfg = _camera_config(cfg, mission_cfg.get("watch_camera") or {})
    # One device serves both stages unless the watch camera really differs.
    share_camera = watch_camera_cfg.get("camera") == (cfg.get("camera") or {})
    watch_camera = make_camera(watch_camera_cfg)
    grasp_camera = watch_camera if share_camera else make_camera(cfg)
    robot = make_robot(cfg)
    stop = threading.Event()
    model = None
    web = None
    robot_started = False
    needs_stand = False

    scene = contextlib.ExitStack()

    try:
        if web_port:
            views = [
                ("watch", "Watch camera"),
                ("wrist", "Wrist camera (grasp)"),
                ("grasp_shot", "Last grasp detection"),
            ]
            if hasattr(robot, "debug_views"):
                views[0:0] = [(name, label) for name, label, _, _ in robot.debug_views()]
            web = WebDebug(
                web_port,
                "TRON2 Squat and Grasp" + (" (simulation)" if args.sim else ""),
                buttons=[("stop", "Stop mission")],
                views=views,
                frames=bool(mission_cfg.get("web_frames", True)),
            )
            web.start()
            web.set_status({
                "stage": "initializing",
                "mode": "dry-run" if args.dry_run else "execute",
                "target": target_label,
            })
            print(f"[Web] Debug console: http://0.0.0.0:{web_port}/")
            if args.sim:
                print(f"[Web] Open http://localhost:{web_port}/ in a browser to watch the demo")
            scene.enter_context(scene_pump(robot, web))

        print(f"[Mission] Open watch camera: {watch_camera_cfg['camera']['type']}")
        watch_camera.open()
        watch_camera.warm_up(10)
        model, detector_opts = load_detector(cfg)
        if web is not None:
            cameras = {"Watch / wrist" if share_camera else "Watch": watch_camera}
            if not share_camera:
                cameras["Wrist"] = grasp_camera
            web.vision.bind(model, cameras)

        def detect(color_bgr: np.ndarray) -> tuple[Any, list[Detection]]:
            return detect_objects(model, color_bgr, detector_opts)

        if not args.dry_run:
            print("[Mission] Connect TRON2")
            robot.start()
            robot_started = True
            watch_joints = mission_cfg.get("watch_joints")
            if watch_joints is not None:
                print(f"[Mission] Hold watch pose {list(watch_joints)}")
                with preview_pump(watch_camera, web, "watch"):
                    if not robot.move_joints(np.asarray(watch_joints, float)):
                        raise RuntimeError("Unable to reach the watch pose")
            robot.open_gripper()
        else:
            print("[Mission] dry-run: skip TRON2 connection")

        cycle = 0
        while not stop.is_set():
            cycle += 1
            print(f"[Mission] Cycle {cycle}")
            cycle_t0 = lap_t = time.monotonic()

            def lap(stage: str) -> None:
                nonlocal lap_t
                now = time.monotonic()
                print(f"[Timing] {stage}: {now - lap_t:.2f}s (cycle {now - cycle_t0:.2f}s)")
                lap_t = now

            if not wait_for_target(
                watch_camera, detect, target_classes, priority,
                int(mission_cfg.get("detect_on_frames", 3)),
                int(mission_cfg.get("detect_window_frames", 0)),
                show_window, web, stop,
            ):
                print("[Mission] Stopped while watching")
                return 0
            lap("watch")

            approach_cfg = mission_cfg.get("approach") or {}
            if approach_cfg.get("enabled", False):
                if not run_approach(
                    watch_camera, detect,
                    robot if robot_started else None,
                    target_classes, priority,
                    ApproachControlConfig.from_dict(approach_cfg.get("control") or {}),
                    float(approach_cfg.get("timeout_s", 30.0)),
                    float(approach_cfg.get("twist_hz", 30.0)),
                    show_window, web, stop,
                ):
                    if stop.is_set():
                        print("[Mission] Stopped while approaching")
                        return 0
                    print("[Mission] Approach failed; not squatting")
                    if web is not None:
                        web.add_event("Mission failed: approach did not arrive", "err")
                    return 3
                lap("approach")

            with preview_pump(watch_camera, web, "watch"):
                # A timeout can occur after the robot has already knelt.
                needs_stand = robot_started
                _kneel(
                    robot, kneel_dwell_s,
                    float(posture_cfg.get("body_pitch_rad", 0.5)),
                    pitch_dwell_s, args.dry_run,
                    float(posture_cfg.get("pitch_reset_dwell_s", 1.0)),
                )
                lap("kneel + pitch")
                # The arm waits for the body: moving it during the kneel and
                # pitch looked like the grasp starting mid-pitch.
                if robot_started and bool(mission_cfg.get("observe_before_grasp", True)):
                    print("[Mission] Move arm to the wrist-camera observe pose")
                    ready_cfg = (cfg.get("robot") or {}).get("ready_pose") or {}
                    if not robot.move_ready(ready_cfg):
                        raise RuntimeError("Unable to reach the observe pose")
            lap("observe pose")

            if not share_camera:
                watch_camera.close()
                time.sleep(float(mission_cfg.get("camera_switch_delay_s", 1.0)))
                grasp_camera.open()
                grasp_camera.warm_up(15)

            success = run_single_grasp(
                cfg, grasp_camera, detect,
                robot if robot_started else None,
                target_classes, grasp_timeout, args.dry_run, show_window, web, stop,
            )
            print("[Mission] Grasp succeeded" if success else "[Mission] Grasp failed")
            lap("grasp")

            if robot_started:
                drop_joints = mission_cfg.get("drop_joints")
                if success and drop_joints is not None and not stop.is_set():
                    print(f"[Mission] Move to drop pose {list(drop_joints)} before standing")
                    if not robot.move_joints(np.asarray(drop_joints, float)):
                        print("[Mission] Drop pose not reached; releasing here")
                if not stop.is_set():
                    print("[Mission] Release gripper before standing")
                    robot.release_gripper()
                lap("drop + release")
                _stand_up(robot, stand_dwell_s)
                needs_stand = False
                print("[Mission] Standing again")
                lap("stand")
            if web is not None:
                web.add_event(f"Cycle {cycle} {'complete' if success else 'failed'}",
                              "ok" if success else "err")
            if stop.is_set():
                return 0
            if args.dry_run or not bool(mission_cfg.get("repeat", False)):
                return 0 if success else 3
            if args.cycles and cycle >= args.cycles:
                return 0 if success else 3
            if not share_camera:
                grasp_camera.close()
                time.sleep(float(mission_cfg.get("camera_switch_delay_s", 1.0)))
                watch_camera.open()
                watch_camera.warm_up(10)
            if stop.wait(float(mission_cfg.get("cycle_delay_s", 0.5))):
                return 0
            watch_joints = mission_cfg.get("watch_joints")
            if watch_joints is not None:
                with preview_pump(watch_camera, web, "watch"):
                    if not robot.move_joints(np.asarray(watch_joints, float)):
                        raise RuntimeError("Unable to return to the watch pose")
            lap("return to watch")
        return 0
    finally:
        scene.close()
        try:
            watch_camera.close()
        except Exception:
            pass
        try:
            grasp_camera.close()
        except Exception:
            pass
        if model is not None:
            _release_model(model)
        if robot_started:
            # An exception mid-mission can leave the body crouched.
            if needs_stand:
                try:
                    _stand_up(robot, stand_dwell_s)
                except Exception as exc:
                    print(f"[Exit] Unable to stand up: {exc}")
            try:
                robot.end()
            except Exception as exc:
                print(f"[Exit] Unable to close robot: {exc}")
        if web is not None:
            web.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
