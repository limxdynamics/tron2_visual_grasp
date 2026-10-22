"""Coordinate transform utilities."""
from typing import Optional

import numpy as np


_ROT_X_PI = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _nearest_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"rotation matrix must be (3, 3), got {R.shape}")

    if not np.all(np.isfinite(R)):
        raise ValueError("rotation matrix contains non-finite values")

    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho.astype(np.float64)


def pose6d_to_mat4(x, y, z, rx, ry, rz, degrees=False) -> np.ndarray:
    """
    Convert a 6D pose to a 4x4 homogeneous transform.

    Args:
        x, y, z: translation in meters.
        rx, ry, rz: intrinsic ZYX Euler angles in roll, pitch, yaw order.
        degrees: True when angles are in degrees; False for radians.

    Returns:
        T: (4, 4) numpy array
    """
    if degrees:
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    # Rotation around X.
    Rx = np.array([
        [1,          0,           0],
        [0,  np.cos(rx), -np.sin(rx)],
        [0,  np.sin(rx),  np.cos(rx)],
    ])
    # Rotation around Y.
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    # Rotation around Z.
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [         0,           0, 1],
    ])

    # Intrinsic ZYX rotation: R = Rz @ Ry @ Rx.
    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def mat4_to_pose6d(T: np.ndarray) -> tuple:
    """
    Convert a 4x4 transform to (x, y, z, rx, ry, rz) in radians.
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rpy = rotation_matrix_to_euler_zyx(T[:3, :3])
    return float(x), float(y), float(z), float(rpy[0]), float(rpy[1]), float(rpy[2])


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to intrinsic ZYX Euler angles."""
    R = _nearest_rotation_matrix(R)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def canonicalize_parallel_gripper_tcp_rotation(R: np.ndarray) -> np.ndarray:
    """Pick a stable equivalent TCP rotation for a parallel gripper.

    For a symmetric parallel gripper, a 180-degree twist around the tool X axis
    is usually grasp-equivalent. Choose the branch between ``R`` and
    ``R @ Rx(pi)`` that has the smaller absolute roll, making the output RPY
    easier to inspect and debug.
    """
    R = _nearest_rotation_matrix(R)
    alt = R @ _ROT_X_PI

    roll = float(rotation_matrix_to_euler_zyx(R)[0])
    alt_roll = float(rotation_matrix_to_euler_zyx(alt)[0])
    return alt if abs(alt_roll) < abs(roll) else R


def grasp_axes_to_tcp_rotation(
    grip_axis: np.ndarray,
    open_axis: np.ndarray,
    approach_axis: np.ndarray,
) -> np.ndarray:
    """Map grasp-frame axes to the arm TCP frame.

    Vision grasp convention:
      - X = grip_axis
      - Y = open_axis
      - Z = approach_axis

    Arm TCP convention:
      - X = tool-forward / approach direction
      - Y = gripper opening direction
      - Z = right-handed completion
    """
    grip = np.asarray(grip_axis, dtype=np.float64)
    open_vec = np.asarray(open_axis, dtype=np.float64)
    approach = np.asarray(approach_axis, dtype=np.float64)

    grip /= max(np.linalg.norm(grip), 1e-8)
    open_vec /= max(np.linalg.norm(open_vec), 1e-8)
    approach /= max(np.linalg.norm(approach), 1e-8)

    # tcp_x = tool-forward = approach direction pointing INTO the object (downward in base).
    # approach_axis points from the object back toward the camera, so negate it here.
    tcp_x = -approach
    tcp_y = open_vec - float(np.dot(open_vec, tcp_x)) * tcp_x
    tcp_y /= max(np.linalg.norm(tcp_y), 1e-8)
    tcp_z = np.cross(tcp_x, tcp_y)
    tcp_z /= max(np.linalg.norm(tcp_z), 1e-8)

    # Keep tcp_z aligned with the grip axis after the approach-axis flip.
    if float(np.dot(tcp_z, grip)) < 0.0:
        tcp_y = -tcp_y
        tcp_z = -tcp_z

    R = np.column_stack([tcp_x, tcp_y, tcp_z]).astype(np.float64)
    if np.linalg.det(R) < 0.0:
        R[:, 2] *= -1.0
    return R



def _flatten_to_top_down(R: np.ndarray) -> np.ndarray:
    """Force a top-down grasp: tool-X points straight down (base -Z), keeping the
    gripper opening axis aligned with the object in the horizontal plane."""
    R = _nearest_rotation_matrix(R)
    open_dir = R[:, 1].copy()
    open_dir[2] = 0.0
    if np.linalg.norm(open_dir) < 1e-6:
        open_dir = R[:, 2].copy()
        open_dir[2] = 0.0
    if np.linalg.norm(open_dir) < 1e-6:
        open_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tcp_x = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    tcp_y = open_dir / np.linalg.norm(open_dir)
    tcp_z = np.cross(tcp_x, tcp_y)
    tcp_z /= max(np.linalg.norm(tcp_z), 1e-8)
    tcp_y = np.cross(tcp_z, tcp_x)
    return _nearest_rotation_matrix(np.column_stack([tcp_x, tcp_y, tcp_z]))


def _keep_orientation_grasp(
    position_cam: np.ndarray,
    T_cam2base: np.ndarray,
    current_tool_rotation: np.ndarray,
    pregrasp_offset_m: float,
    insertion_depth_m: float,
    min_base_z_m: Optional[float] = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Descend onto the object while holding the arm's current, already-reachable
    wrist orientation. Only the object position comes from vision (via the
    hand-eye); the orientation is taken from the measured pose, so this never
    re-solves the wrist and cannot command an unreachable twist. Pre-grasp and
    insertion move along base Z (height only), matching a plain top-down descent.

    ``insertion_depth_m`` drops the grasp this far below the detected surface so
    the jaws actually straddle the object; ``min_base_z_m`` floors the grasp Z to
    keep the gripper off the ground."""
    T_cam2base = np.asarray(T_cam2base, dtype=np.float64)
    p_cam = np.asarray(position_cam, dtype=np.float64).reshape(3)
    position_base = (T_cam2base @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]

    grasp_z = float(position_base[2]) - float(insertion_depth_m)
    if min_base_z_m is not None:
        grasp_z = max(grasp_z, float(min_base_z_m))

    T_grasp = np.eye(4, dtype=np.float64)
    T_grasp[:3, :3] = _nearest_rotation_matrix(current_tool_rotation)
    T_grasp[:3, 3] = position_base
    T_grasp[2, 3] = grasp_z

    T_pregrasp = T_grasp.copy()
    T_pregrasp[2, 3] = grasp_z + float(pregrasp_offset_m)
    return mat4_to_pose6d(T_grasp), mat4_to_pose6d(T_pregrasp)


def _auto_align_grasp(
    position_cam: np.ndarray,
    T_cam2base: np.ndarray,
    open_axis_cam: np.ndarray,
    hand_eye_rotation: np.ndarray,
    open_flip: bool,
    pregrasp_offset_m: float,
    insertion_depth_m: float,
    min_base_z_m: Optional[float] = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Auto-align a top-down grasp using the true camera/gripper mounting.

    The gripper approaches along its real tool axis -- the camera optical axis,
    which the hand-eye shows is ee_link +Y on this arm, not the +X that
    ``grasp_axes_to_tcp_rotation`` assumes. That axis is already what the
    arm holds at the observe pose (``T_cam2base[:, 2]``), so the wrist only rolls
    about it to line the jaws up with the object's short axis -- a reachable roll
    (j6) instead of the unreachable twist that re-solving ee_link +X produced.

    ``hand_eye_rotation`` is R_cam2gripper (== camera frame in ee_link), so the
    commanded ee_link pose is ``R_ee = D @ hand_eye_rotation.T`` where ``D`` is
    the desired camera frame in base (x = jaw opening, z = approach). ``open_flip``
    swaps the assumed gripper opening axis by 90 deg if the jaws come out across
    the wrong object axis.
    """
    T_cam2base = np.asarray(T_cam2base, dtype=np.float64)
    R_cb = T_cam2base[:3, :3]
    p_cam = np.asarray(position_cam, dtype=np.float64).reshape(3)
    position_base = (T_cam2base @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]

    approach = R_cb[:, 2].copy()
    approach /= max(np.linalg.norm(approach), 1e-8)

    open_base = R_cb @ np.asarray(open_axis_cam, dtype=np.float64)
    open_base = open_base - float(np.dot(open_base, approach)) * approach
    if np.linalg.norm(open_base) < 1e-6:
        open_base = R_cb[:, 0] - float(np.dot(R_cb[:, 0], approach)) * approach
    open_dir = open_base / max(np.linalg.norm(open_base), 1e-8)

    D = np.column_stack([open_dir, np.cross(approach, open_dir), approach])
    G = _nearest_rotation_matrix(hand_eye_rotation)
    if open_flip:
        G = np.column_stack([G[:, 1], -G[:, 0], G[:, 2]])
    R_ee = _nearest_rotation_matrix(D @ G.T)

    grasp_z = float(position_base[2]) - float(insertion_depth_m)
    if min_base_z_m is not None:
        grasp_z = max(grasp_z, float(min_base_z_m))

    T_grasp = np.eye(4, dtype=np.float64)
    T_grasp[:3, :3] = R_ee
    T_grasp[:3, 3] = position_base
    T_grasp[2, 3] = grasp_z

    T_pregrasp = T_grasp.copy()
    T_pregrasp[2, 3] = grasp_z + float(pregrasp_offset_m)
    return mat4_to_pose6d(T_grasp), mat4_to_pose6d(T_pregrasp)


def _make_grasp_base_transform(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
    top_down: bool = False,
) -> np.ndarray:
    T_grasp_cam = np.eye(4, dtype=np.float64)
    T_grasp_cam[:3, :3] = np.asarray(tcp_rotation_cam, dtype=np.float64)
    T_grasp_cam[:3, 3] = np.asarray(position_cam, dtype=np.float64)

    T_grasp_base = np.asarray(T_cam2base, dtype=np.float64) @ T_grasp_cam
    R_base = T_grasp_base[:3, :3]
    if top_down:
        R_base = _flatten_to_top_down(R_base)
    T_grasp_base[:3, :3] = canonicalize_parallel_gripper_tcp_rotation(R_base)
    return T_grasp_base


def _offset_along_tool_x(T: np.ndarray, offset_m: float) -> np.ndarray:
    T_offset = T.copy()
    T_offset[:3, 3] = T[:3, 3] - T[:3, 0] * float(offset_m)
    return T_offset


def class_grasp_offset_m(grasp_cfg: dict, class_name: str) -> tuple[float, float, float]:
    """Base-frame grasp offset for one detected class, falling back to "default"."""
    table = grasp_cfg.get("class_offset_m") or {}
    wanted = str(class_name).strip().lower()
    entry = next(
        (value for key, value in table.items() if str(key).strip().lower() == wanted),
        None,
    )
    if entry is None:
        entry = table.get("default")
    if not entry:
        return (0.0, 0.0, 0.0)
    return (
        float(entry.get("x", 0.0)),
        float(entry.get("y", 0.0)),
        float(entry.get("z", 0.0)),
    )


def offset_pose6d(
    pose6d: tuple[float, ...],
    offset_m: tuple[float, float, float],
    min_base_z_m: Optional[float] = None,
) -> tuple[float, ...]:
    """Shift a base-frame pose, keeping the safety floor that the transform applied."""
    x, y, z, rx, ry, rz = pose6d
    z += float(offset_m[2])
    if min_base_z_m is not None:
        z = max(z, float(min_base_z_m))
    return (x + float(offset_m[0]), y + float(offset_m[1]), z, rx, ry, rz)


def transform_grasp_pose_to_base(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    insertion_depth_m: float = 0.0,
    top_down: bool = False,
    current_tool_rotation: Optional[np.ndarray] = None,
    min_base_z_m: Optional[float] = None,
    orientation_mode: str = "keep",
    hand_eye_rotation: Optional[np.ndarray] = None,
    open_flip: bool = False,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert a camera-frame grasp pose to base-frame grasp/pregrasp poses."""
    if top_down and orientation_mode == "align" and hand_eye_rotation is not None:
        return _auto_align_grasp(
            position_cam, T_cam2base,
            np.asarray(tcp_rotation_cam, dtype=np.float64)[:, 1],
            hand_eye_rotation, open_flip,
            pregrasp_offset_m, insertion_depth_m, min_base_z_m,
        )
    if top_down and orientation_mode == "keep" and current_tool_rotation is not None:
        return _keep_orientation_grasp(
            position_cam, T_cam2base, current_tool_rotation,
            pregrasp_offset_m, insertion_depth_m, min_base_z_m,
        )
    T_grasp_base = _make_grasp_base_transform(
        position_cam, tcp_rotation_cam, T_cam2base, top_down=top_down,
    )
    T_grasp_base = _offset_along_tool_x(T_grasp_base, -insertion_depth_m)
    if min_base_z_m is not None:
        T_grasp_base[2, 3] = max(float(T_grasp_base[2, 3]), float(min_base_z_m))
    T_pregrasp_base = _offset_along_tool_x(T_grasp_base, pregrasp_offset_m)
    return mat4_to_pose6d(T_grasp_base), mat4_to_pose6d(T_pregrasp_base)

