"""Third-person 3D view of the simulated TRON2 for the demo's web console.

Renders the WFYG_TRON2A model (drivers/sim/model.py) with MuJoCo, posed from the
SimWorld state every frame; MuJoCo is used for kinematics and rendering only,
there is no physics:

  torso   free joint = SimWorld.torso_world()
  legs    hip pitch and knee solved so the wheels stay on the floor; standing
          keeps the wheels under the hips, kneeling folds the legs (the knee
          only bends backward) and rests the knees on the floor behind the hips
  arm     damped least-squares IK onto the simulated TCP (the model's arm has
          the same kinematics as the piper_x FK the simulator uses)
  gripper open / closed on the ball / closed empty
  ball    a mocap sphere at the simulated ball position

Optional: without MuJoCo, the model, or an OpenGL context, the demo runs
without this view. On a headless machine set MUJOCO_GL=egl (GPU) or
MUJOCO_GL=osmesa, or run under xvfb-run.
"""

from __future__ import annotations

import importlib.util
import math
import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .world import BALL_RADIUS_M, TOOL_OFFSET, WHEEL_RADIUS_M, SimWorld

if not os.environ.get("MUJOCO_GL") and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"          # headless: GPU EGL; override with MUJOCO_GL

_ARM = [f"arm{i}_Joint" for i in range(1, 7)]
_KNEE_ON_FLOOR_Z = 0.06                      # knee link centre when kneeling
_KNEE_STAND_Z = 0.414                        # knee link centre when standing
_STAND_SEED = (0.47, -0.94)                  # hip pitch, knee (rad): knee behind the hip
_KNEEL_SEED = (1.3, -2.5)                    # legs folded, knees down, wheels in front


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


class Robot3DView:
    """Poses and renders the TRON2 model; create and render from one thread."""

    def __init__(self, world: SimWorld, xml: Path, width: int = 640, height: int = 480) -> None:
        import mujoco

        self._mj = mujoco
        self._world = world
        self.model = self._build(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)   # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)    # noqa: E731
        self._qadr = lambda n: m.jnt_qposadr[jid(n)]                          # noqa: E731
        self._free = m.jnt_qposadr[0]
        self._arm_adr = np.array([self._qadr(n) for n in _ARM])
        self._arm_dof = np.array([m.jnt_dofadr[jid(n)] for n in _ARM])
        self._arm_lo = m.jnt_range[[jid(n) for n in _ARM], 0]
        self._arm_hi = m.jnt_range[[jid(n) for n in _ARM], 1]
        self._b_torso, self._b_arm_base, self._b_flange = bid("base_Link"), bid("arm_base_Link"), bid("arm6_Link")
        self._b_wheel, self._b_knee, self._b_hip = bid("wheel_L_Link"), bid("knee_L_Link"), bid("proximal_pitch_L_Link")
        self._ball_mocap = m.body_mocapid[bid("sim_ball")]
        self._size = (width, height)
        self._renderer: Any = None
        self._cam = mujoco.MjvCamera()
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[0] = 0               # hide collision meshes
        self._lookat: Optional[np.ndarray] = None
        self._legs = _STAND_SEED
        self._arm_q = np.zeros(6)
        self._last_hint: Any = None
        self._hint_move: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self._wheel_angle = 0.0
        self._last_xy: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    # Scene

    def _build(self, xml: Path) -> Any:
        mj = self._mj
        spec = mj.MjSpec.from_file(str(xml))
        for tex in spec.textures:
            if tex.name == "texplane":                # match the simulated camera's floor
                tex.rgb1, tex.rgb2 = [0.62, 0.62, 0.62], [0.52, 0.52, 0.52]
        for geom in spec.geoms:
            if geom.name == "ground":
                geom.group = 2
                geom.rgba = [1.0, 1.0, 1.0, 1.0]     # the model default tints it blue and half transparent
        sky = spec.add_texture()
        sky.name, sky.type, sky.builtin = "sky", mj.mjtTexture.mjTEXTURE_SKYBOX, mj.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.rgb1, sky.rgb2, sky.width, sky.height = [0.62, 0.70, 0.78], [0.20, 0.22, 0.26], 256, 256
        world = spec.worldbody
        sun = world.add_light()
        sun.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
        sun.pos, sun.dir, sun.diffuse, sun.castshadow = [1.5, -1.5, 3.0], [-0.4, 0.4, -1.0], [0.6, 0.6, 0.6], True
        ball = world.add_body(name="sim_ball", mocap=True, pos=[1.0, 0.0, BALL_RADIUS_M])
        ball.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS_M, 0, 0],
                      rgba=[0.80, 0.88, 0.16, 1.0], contype=0, conaffinity=0, group=2)
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
        spec.visual.quality.shadowsize = 2048
        return spec.compile()

    # Pose solving

    def _set_torso(self, T: np.ndarray) -> None:
        a = self._free
        self.data.qpos[a:a + 3] = T[:3, 3]
        self.data.qpos[a + 3:a + 7] = _mat_to_quat(T[:3, :3])

    def _set_legs(self, hip: float, knee: float) -> None:
        for side in "LR":
            self.data.qpos[self._qadr(f"proximal_pitch_{side}_Joint")] = hip
            self.data.qpos[self._qadr(f"knee_{side}_Joint")] = knee

    def _leg_residual(self, hip: float, knee: float, kneel: float, heading: np.ndarray) -> np.ndarray:
        self._set_legs(hip, knee)
        self._mj.mj_kinematics(self.model, self.data)
        x = self.data.xpos
        wheel_z = x[self._b_wheel][2] - WHEEL_RADIUS_M
        knee_z = (1.0 - kneel) * _KNEE_STAND_Z + kneel * _KNEE_ON_FLOOR_Z
        return np.array([wheel_z, x[self._b_knee][2] - knee_z])

    def _solve_legs(self, seed: tuple[float, float], kneel: float, heading: np.ndarray) -> tuple[float, float]:
        """Newton from the previous solution, falling back to a seed on the stand-kneel path.

        The knee height descends linearly with the kneel fraction. Blending the stand
        constraint (wheel under the hip) with the kneel one (knee on the floor) instead
        has no continuous solution midway, so the knee straightened and flipped branch.
        """
        best, best_err = seed, np.inf
        path_seed = tuple(np.add(np.multiply(_STAND_SEED, 1.0 - kneel), np.multiply(_KNEEL_SEED, kneel)))
        for start in (seed, path_seed):
            q = self._newton_legs(start, kneel, heading)
            err = float(np.max(np.abs(self._leg_residual(*q, kneel, heading))))
            if err < best_err:
                best, best_err = q, err
            if err < 1e-3:
                break
        return best

    def _newton_legs(self, seed: tuple[float, float], kneel: float, heading: np.ndarray) -> tuple[float, float]:
        q = np.array(seed, float)
        for _ in range(12):
            r = self._leg_residual(*q, kneel, heading)
            if np.max(np.abs(r)) < 1e-4:
                break
            J = np.empty((2, 2))
            for i in range(2):
                dq = np.zeros(2)
                dq[i] = 1e-4
                J[:, i] = (self._leg_residual(*(q + dq), kneel, heading) - r) / 1e-4
            q -= np.linalg.lstsq(J + 1e-6 * np.eye(2), r, rcond=None)[0]
            q[0] = float(np.clip(q[0], -3.1, 2.6))
            q[1] = float(np.clip(q[1], -2.6, 0.26))
        return float(q[0]), float(q[1])

    def _solve_arm(self, T_flange_world: np.ndarray, iters: int = 60) -> None:
        mj, m, d = self._mj, self.model, self.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q = self._arm_q.copy()
        for _ in range(iters):
            d.qpos[self._arm_adr] = q
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            p_err = T_flange_world[:3, 3] - d.xpos[self._b_flange]
            R = d.xmat[self._b_flange].reshape(3, 3)
            R_err = T_flange_world[:3, :3] @ R.T
            w_err = 0.5 * np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
            if np.linalg.norm(p_err) < 5e-4 and np.linalg.norm(w_err) < 5e-3:
                break
            mj.mj_jacBody(m, d, jacp, jacr, self._b_flange)
            J = np.vstack([jacp[:, self._arm_dof], 0.3 * jacr[:, self._arm_dof]])
            e = np.concatenate([p_err, 0.3 * w_err])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            step = float(np.max(np.abs(dq)))
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, self._arm_lo, self._arm_hi)
        self._arm_q = q
        d.qpos[self._arm_adr] = q

    def pose(self) -> None:
        """Copy the SimWorld state into the model."""
        w = self._world
        with w.lock:
            w.step()
            T_torso = w.torso_world()
            T_flange = w.base_world() @ w.tcp @ np.linalg.inv(TOOL_OFFSET)
            kneel = w.kneel_fraction()
            ball = w.ball.pos.copy()
            held, gripper_open = w.ball.held, w.gripper_open
            q_hint = getattr(w, "arm_q_hint", None)
            tcp, tcp_goal = w.tcp.copy(), w.tcp_goal.copy()
            xy = np.array(w.base[:2])
        heading = T_torso[:2, 0] / max(np.linalg.norm(T_torso[:2, 0]), 1e-9)
        self._set_torso(T_torso)
        self._legs = self._solve_legs(self._legs, kneel, heading)
        if self._last_xy is not None:
            self._wheel_angle += float((xy - self._last_xy) @ heading) / WHEEL_RADIUS_M
        self._last_xy = xy
        for side in "LR":
            self.data.qpos[self._qadr(f"wheel_{side}_Joint")] = self._wheel_angle
        if q_hint is not None and q_hint is not self._last_hint:
            # A new joint move: blend the IK seed from the current joints to the target ones
            # as the TCP advances, so the arm follows that branch without snapping to it.
            self._last_hint = q_hint
            self._hint_move = (self._arm_q.copy(), np.asarray(q_hint, float).copy(), tcp, tcp_goal)
        if self._hint_move is not None:
            q0, q1, T0, T1 = self._hint_move
            if not np.allclose(tcp_goal, T1):
                self._hint_move = None                           # another command took over
            else:
                total = np.linalg.norm(T1[:3, 3] - T0[:3, 3]) + 0.1 * np.linalg.norm(T1[:3, :3] - T0[:3, :3])
                left = np.linalg.norm(T1[:3, 3] - tcp[:3, 3]) + 0.1 * np.linalg.norm(T1[:3, :3] - tcp[:3, :3])
                frac = 1.0 if total < 1e-9 else float(np.clip(1.0 - left / total, 0.0, 1.0))
                self._arm_q = q0 + frac * (q1 - q0)
                if frac >= 1.0:
                    self._hint_move = None
        self._solve_arm(T_flange)
        open_m = 0.035 if gripper_open else (BALL_RADIUS_M - 0.002 if held else 0.0)
        self.data.qpos[self._qadr("gripper1_Joint")] = open_m
        self.data.qpos[self._qadr("gripper2_Joint")] = -open_m
        self.data.mocap_pos[self._ball_mocap] = ball
        self._mj.mj_kinematics(self.model, self.data)

    # Rendering

    def _update_camera(self) -> None:
        d = self.data
        torso = d.xpos[self._b_torso]
        ball = d.mocap_pos[self._ball_mocap]
        target = np.array([torso[0], torso[1], 0.6])
        if np.linalg.norm(ball[:2] - torso[:2]) < 2.5:
            target[:2] = 0.65 * torso[:2] + 0.35 * ball[:2]
        self._lookat = target if self._lookat is None else 0.8 * self._lookat + 0.2 * target
        yaw = math.degrees(math.atan2(d.xmat[self._b_torso][3], d.xmat[self._b_torso][0]))
        self._cam.lookat[:] = self._lookat
        self._cam.azimuth = yaw - 60.0               # from the left rear quarter, so the arm reach shows
        self._cam.elevation = -22.0
        self._cam.distance = 2.2

    def render(self) -> np.ndarray:
        """BGR image of the current scene."""
        with self._lock:
            if self._renderer is None:
                try:
                    self._renderer = self._mj.Renderer(self.model, self._size[1], self._size[0])
                except Exception as exc:
                    raise RuntimeError(
                        f"no OpenGL context ({type(exc).__name__}); on a headless machine set "
                        "MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa, or run under xvfb-run") from None
            self.pose()
            self._update_camera()
            self._renderer.update_scene(self.data, self._cam, self._opt)
            rgb = self._renderer.render()
            return np.ascontiguousarray(rgb[..., ::-1])

    def close(self) -> None:
        with self._lock:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None


def make_view3d(world: SimWorld, sim_cfg: dict[str, Any]) -> Optional[Robot3DView]:
    """Build the 3D view if MuJoCo and the model are available; None (with a note) otherwise."""
    cfg = (sim_cfg or {}).get("view3d") or {}
    if not cfg.get("enabled", True):
        return None
    if importlib.util.find_spec("mujoco") is None:
        print("[Sim3D] MuJoCo not installed; 3D robot view off (pip install mujoco to enable)")
        return None
    from .model import ensure_model
    xml = ensure_model()
    if xml is None:
        print("[Sim3D] TRON2 model unavailable; 3D robot view off")
        return None
    try:
        return Robot3DView(world, xml, int(cfg.get("width", 640)), int(cfg.get("height", 480)))
    except Exception as exc:
        print(f"[Sim3D] 3D robot view off: {exc}")
        return None
