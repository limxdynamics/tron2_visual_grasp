"""手眼标定 — 基于 OpenCV calibrateHandEye。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np


class CalibMode(Enum):
    EYE_IN_HAND = "eye_in_hand"   # 相机在末端，随末端运动


_METHOD_MAP = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class CalibResult:
    T_result: np.ndarray    # (4, 4) 手眼变换矩阵
    mode: str               # CalibMode.value
    n_samples: int
    method: str
    rot_rms_deg: float = float("nan")   # AX=XB 旋转残差 RMS（度）
    trans_rms_m: float = float("nan")   # 固定标记平移拟合残差 RMS（米），退化时为 AX=XB 残差
    rot_spread_deg: float = float("nan")  # 样本间相对旋转角的跨度，判断旋转多样性

    @property
    def suspect(self) -> bool:
        """结果是否可疑：平移过大、残差过大或旋转多样性不足。"""
        t = np.linalg.norm(self.T_result[:3, 3])
        return (
            t > 0.30                                   # 手腕相机离末端不该超过 30cm
            or (self.rot_rms_deg == self.rot_rms_deg and self.rot_rms_deg > 3.0)
            or (self.trans_rms_m == self.trans_rms_m and self.trans_rms_m > 0.02)
            or (self.rot_spread_deg == self.rot_spread_deg and self.rot_spread_deg < 25.0)
        )


@dataclass
class _Sample:
    T_gripper2base: np.ndarray   # (4, 4)
    T_marker2cam:   np.ndarray   # (4, 4)


class HandEyeCalibrator:
    """
    手眼标定器。

    Eye-in-Hand 模式：
        求解 T_cam2gripper，使得
            T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
        在所有姿态下恒成立。

    使用方法：
        calib = HandEyeCalibrator(CalibMode.EYE_IN_HAND)
        calib.add_sample(T_gripper2base, T_marker2cam)
        ...
        result = calib.calibrate()
        HandEyeCalibrator.save(result, "hand_eye.npz")
    """

    def __init__(
        self,
        mode: CalibMode = CalibMode.EYE_IN_HAND,
        method: str = "TSAI",
    ) -> None:
        if mode != CalibMode.EYE_IN_HAND:
            raise ValueError("Only eye-in-hand calibration is supported")
        self._mode = mode
        self._method = method.upper()
        self._samples: List[_Sample] = []

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def add_sample(
        self,
        T_gripper2base: np.ndarray,
        T_marker2cam: np.ndarray,
    ) -> None:
        """
        添加一个标定样本。

        Args:
            T_gripper2base: (4,4) 末端到基座的变换（robot.get_tcp_pose() 输出）
            T_marker2cam:   (4,4) 标记到相机的变换（ArUco 检测输出）
        """
        self._samples.append(_Sample(
            T_gripper2base=np.asarray(T_gripper2base, dtype=np.float64),
            T_marker2cam=np.asarray(T_marker2cam, dtype=np.float64),
        ))

    def calibrate(self, min_samples: int = 5) -> CalibResult:
        """
        计算手眼变换。

        Args:
            min_samples: 最少样本数（< 此值会抛出异常）

        Returns:
            CalibResult，T_result 即手眼变换矩阵
        """
        if self.n_samples < min_samples:
            raise ValueError(
                f"样本不足：{self.n_samples} < {min_samples}，请继续采集"
            )

        cv_method = _METHOD_MAP.get(self._method, cv2.CALIB_HAND_EYE_TSAI)

        # OpenCV 接口：R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
        R_g2b = [s.T_gripper2base[:3, :3] for s in self._samples]
        t_g2b = [s.T_gripper2base[:3,  3].reshape(3, 1) for s in self._samples]
        R_t2c = [s.T_marker2cam[:3, :3] for s in self._samples]
        t_t2c = [s.T_marker2cam[:3,  3].reshape(3, 1) for s in self._samples]

        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_g2b, t_g2b, R_t2c, t_t2c, method=cv_method
        )
        # calibrateHandEye's rotation is reliable, but its translation solves the
        # AX=XB difference (R_A - I) t = ... which goes singular when the wrist can
        # only rotate a little (the TRON2 workspace caps tilts ~25-30 deg), so |t|
        # blows up. Re-solve the translation from ABSOLUTE marker positions, which
        # stays well-conditioned at low rotation. Falls back to the OpenCV t if the
        # absolute solve is rank-deficient (poses too similar).
        t_abs, trans_rms = self._solve_translation_absolute(R_c2g)
        if t_abs is None:
            t_abs, trans_rms = t_c2g.flatten(), float("nan")

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_c2g
        T[:3,  3] = t_abs

        rot_rms, _axxb_trans_rms, rot_spread = self._residuals(T)
        if trans_rms != trans_rms:  # NaN -> absolute solve unavailable
            trans_rms = _axxb_trans_rms

        return CalibResult(
            T_result=T,
            mode=self._mode.value,
            n_samples=self.n_samples,
            method=self._method,
            rot_rms_deg=rot_rms,
            trans_rms_m=trans_rms,
            rot_spread_deg=rot_spread,
        )

    def _solve_translation_absolute(self, R_c2g: np.ndarray):
        """Solve t_cam2gripper from absolute marker positions (fixed marker).

        For a marker fixed in the base frame,
            R_g2b @ (R_c2g @ p_marker_cam + t_c2g) + t_g2b = p_marker_base  (const)
        is linear in the unknowns [t_c2g (3), p_marker_base (3)]:
            [R_g2b  -I] @ [t_c2g; p_marker_base] = -(R_g2b @ R_c2g @ p_marker_cam + t_g2b)
        Least-squares over all samples. Unlike AX=XB, the coefficient of t_c2g is a
        full rotation (never singular), so translation stays observable even when the
        wrist barely rotates. Returns (t_c2g, per-sample residual RMS in m), or
        (None, nan) if the stack is rank-deficient (poses too similar).
        """
        A_rows, b_rows = [], []
        for s in self._samples:
            R_gb = s.T_gripper2base[:3, :3]
            t_gb = s.T_gripper2base[:3, 3]
            p_cam = s.T_marker2cam[:3, 3]
            A_rows.append(np.hstack([R_gb, -np.eye(3)]))
            b_rows.append(-(R_gb @ (R_c2g @ p_cam) + t_gb))
        A = np.vstack(A_rows)
        b = np.concatenate(b_rows)
        if np.linalg.matrix_rank(A, tol=1e-6) < 6:
            return None, float("nan")
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = (A @ x - b).reshape(-1, 3)
        trans_rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
        return x[:3], trans_rms


    def _residuals(self, X: np.ndarray):
        """AX=XB 一致性残差与样本旋转多样性。

        对每一对样本 (i, j)：A = Ti_g2b^-1 @ Tj_g2b, B = Ti_m2c @ Tj_m2c^-1，
        理想情况下 A @ X == X @ B。返回旋转残差 RMS（度）、平移残差 RMS（米）
        和样本间相对旋转角的最大跨度（度，用来判断姿态是否够发散）。
        """
        def _ang(R):
            return np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))

        rot_err, trans_err, rel_angles = [], [], []
        s = self._samples
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                A = np.linalg.inv(s[i].T_gripper2base) @ s[j].T_gripper2base
                B = s[i].T_marker2cam @ np.linalg.inv(s[j].T_marker2cam)
                lhs, rhs = A @ X, X @ B
                dR = lhs[:3, :3].T @ rhs[:3, :3]
                rot_err.append(_ang(dR))
                trans_err.append(float(np.linalg.norm(lhs[:3, 3] - rhs[:3, 3])))
                rel_angles.append(_ang(A[:3, :3]))
        if not rot_err:
            return float("nan"), float("nan"), float("nan")
        rot_rms = float(np.sqrt(np.mean(np.square(rot_err))))
        trans_rms = float(np.sqrt(np.mean(np.square(trans_err))))
        rot_spread = float(np.max(rel_angles))
        return rot_rms, trans_rms, rot_spread

    @staticmethod
    def save(result: CalibResult, path: Union[str, Path]) -> None:
        """保存标定结果为 .npz 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            T_result=result.T_result,
            mode=np.array([result.mode]),
            n_samples=np.array([result.n_samples]),
            method=np.array([result.method]),
            rot_rms_deg=np.array([result.rot_rms_deg]),
            trans_rms_m=np.array([result.trans_rms_m]),
            rot_spread_deg=np.array([result.rot_spread_deg]),
        )

    @staticmethod
    def load(path: Union[str, Path]) -> CalibResult:
        """从 .npz 文件加载标定结果。"""
        data = np.load(str(path), allow_pickle=False)
        return CalibResult(
            T_result=data["T_result"],
            mode=str(data["mode"][0]),
            n_samples=int(data["n_samples"][0]),
            method=str(data["method"][0]) if "method" in data else "TSAI",
            rot_rms_deg=float(data["rot_rms_deg"][0]) if "rot_rms_deg" in data else float("nan"),
            trans_rms_m=float(data["trans_rms_m"][0]) if "trans_rms_m" in data else float("nan"),
            rot_spread_deg=float(data["rot_spread_deg"][0]) if "rot_spread_deg" in data else float("nan"),
        )
