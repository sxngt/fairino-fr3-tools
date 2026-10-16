"""FR3 kinematics: classic-DH forward kinematics, analytic inverse kinematics and a DH fit.

The FR3 has the UR structure: joints 2-4 are parallel and joint 5 is perpendicular to them, so
the inverse kinematics has a closed form with up to eight solutions (Hawkins, 2013). A solution
is labelled by its *branch* ``(shoulder, elbow, wrist)``, each 0 or 1; :func:`controller_config`
maps a branch to the controller's ``config`` number used by ``MoveJ``/``GetInverseKin``.

Frames. ``flange(q)`` is base -> flange (flange z = tool-flange normal). ``tcp(q)`` adds the tool
transform from :class:`FR3Params` (``tool_xyz`` in metres, ``tool_rpy_deg``), which is identity by
default. Controller poses are ``[x, y, z, rx, ry, rz]`` in mm and degrees with
``R = Rz(rz) · Ry(ry) · Rx(rx)`` (:func:`pose_to_matrix`, :func:`matrix_to_pose`).

Calibration. The controller applies a factory DH compensation to its own forward kinematics, so
the nominal DH set is off by up to about a millimetre. :func:`fit_dh` fits the link parameters to
logged (joints, controller pose) pairs; the result matches the controller to about 0.03 mm on the
unit this was developed on.

FR3 기구학. UR 구조라 역기구학은 최대 8해의 닫힌 해를 가진다. 해는 (어깨, 팔꿈치, 손목) 가지로
구분하며 :func:`controller_config` 가 제어기의 config 번호로 바꾼다. 제어기는 공장 DH 보정을 자체
FK 에 반영하므로 공칭 DH 는 최대 약 1 mm 어긋난다. :func:`fit_dh` 는 (관절, 제어기 포즈) 기록에
링크 파라미터를 맞춘다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np

JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")
DH_NAMES = ("d1", "a2", "a3", "d4", "d5", "d6")
# Controller soft limits (deg), as read with GetJointSoftLimitDeg. / 제어기 소프트 리밋(deg).
SOFT_LIMITS_DEG = ((-175, 175), (-265, 85), (-150, 150), (-265, 85), (-175, 175), (-175, 175))


# ----------------------------------------------------------------------------- rotations and poses
def rot_x(a: float) -> np.ndarray:
    """Rotation about x (3×3). / x 축 회전."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    """Rotation about y (3×3). / y 축 회전."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    """Rotation about z (3×3). / z 축 회전."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform(rotation=None, translation=None) -> np.ndarray:
    """4×4 homogeneous transform. / 4×4 동차 변환."""
    t = np.eye(4)
    if rotation is not None:
        t[:3, :3] = np.asarray(rotation, dtype=float)
    if translation is not None:
        t[:3, 3] = np.asarray(translation, dtype=float)
    return t


def rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """``Rz(rz) · Ry(ry) · Rx(rx)`` in radians (the controller's convention). / 제어기 RPY 규약."""
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def matrix_to_rpy(r: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rpy_to_matrix` (radians). / RPY 역변환(rad)."""
    r = np.asarray(r, dtype=float)
    ry = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if abs(math.cos(ry)) < 1e-9:
        # Gimbal lock: put the whole yaw into rz. / 짐벌 락: 회전을 rz 에 몰아 준다.
        return np.array([0.0, ry, math.atan2(-r[0, 1], r[1, 1])])
    return np.array([math.atan2(r[2, 1], r[2, 2]), ry, math.atan2(r[1, 0], r[0, 0])])


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """Controller pose ``[x, y, z, rx, ry, rz]`` (mm, deg) -> 4×4 (m). / 제어기 포즈 → 4×4."""
    p = [float(v) for v in pose]
    return transform(rpy_to_matrix(*np.radians(p[3:6])), np.array(p[:3]) / 1000.0)


def matrix_to_pose(t: np.ndarray) -> list[float]:
    """4×4 (m) -> controller pose ``[x, y, z, rx, ry, rz]`` (mm, deg). / 4×4 → 제어기 포즈."""
    t = np.asarray(t, dtype=float)
    return [*(t[:3, 3] * 1000.0).tolist(), *np.degrees(matrix_to_rpy(t[:3, :3])).tolist()]


def top_down_rotation(yaw: float = 0.0) -> np.ndarray:
    """TCP rotation pointing the tool z axis straight down, turned by ``yaw`` about world z.

    툴 z 축이 바로 아래를 향하는 TCP 회전 (세계 z 축으로 ``yaw`` 회전).
    """
    return rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])


def wrap_angle(a):
    """Wrap to [-π, π). / [-π, π) 로 감기."""
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


# ----------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class FR3Params:
    """Classic-DH link parameters, tool and joint limits. / 고전 DH 링크 파라미터, 툴, 관절 리밋.

    Defaults are the nominal FR3 values with no tool. Lengths in metres.
    기본값은 공칭 FR3 값(툴 없음). 길이 단위 m.
    """

    d1: float = 0.140
    a2: float = -0.280
    a3: float = -0.240
    d4: float = 0.102
    d5: float = 0.102
    d6: float = 0.100
    tool_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    limits_deg: tuple[tuple[float, float], ...] = field(default_factory=lambda: SOFT_LIMITS_DEG)

    @property
    def dh(self) -> np.ndarray:
        return np.array([self.d1, self.a2, self.a3, self.d4, self.d5, self.d6])

    @property
    def tool_transform(self) -> np.ndarray:
        """Flange -> TCP. / 플랜지 → TCP."""
        return transform(rpy_to_matrix(*np.radians(self.tool_rpy_deg)), self.tool_xyz)

    def with_dh(self, values: Sequence[float]) -> FR3Params:
        return replace(self, **{k: float(v) for k, v in zip(DH_NAMES, values, strict=True)})

    def to_mapping(self) -> dict:
        d = asdict(self)
        d["tool_xyz"] = list(self.tool_xyz)
        d["tool_rpy_deg"] = list(self.tool_rpy_deg)
        d["limits_deg"] = [list(p) for p in self.limits_deg]
        return d

    @classmethod
    def from_mapping(cls, m: dict) -> FR3Params:
        """Build from a mapping; unknown keys are ignored. / 사전에서 생성(모르는 키는 무시)."""
        kwargs: dict = {k: float(m[k]) for k in DH_NAMES if k in m}
        for key in ("tool_xyz", "tool_rpy_deg"):
            if key in m:
                kwargs[key] = tuple(float(v) for v in m[key])
        if "limits_deg" in m:
            kwargs["limits_deg"] = tuple(tuple(float(v) for v in pair) for pair in m["limits_deg"])
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str) -> FR3Params:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_mapping(data.get("params", data))

    def to_json(self, path: str, **extra) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"params": self.to_mapping(), **extra}, f, indent=2)


def _dh_chain(p, q) -> list[np.ndarray]:
    """Cumulative base -> joint-frame transforms, one per joint plus the flange.

    관절마다의 누적 변환(베이스 → 각 관절 프레임)과 플랜지.
    """
    d1, a2, a3, d4, d5, d6 = p
    out = []
    t = np.eye(4)
    steps = (
        (transform(translation=(0, 0, d1)) @ transform(rot_x(math.pi / 2))),
        transform(translation=(a2, 0, 0)),
        transform(translation=(a3, 0, 0)),
        (transform(translation=(0, 0, d4)) @ transform(rot_x(math.pi / 2))),
        (transform(translation=(0, 0, d5)) @ transform(rot_x(-math.pi / 2))),
        transform(translation=(0, 0, d6)),
    )
    for i in range(6):
        out.append(t.copy())  # frame whose z axis is joint i+1 / z 축이 관절 i+1 인 프레임
        t = t @ transform(rot_z(q[i])) @ steps[i]
    out.append(t)
    return out


# ----------------------------------------------------------------------------- kinematics
class FR3Kinematics:
    """Forward and analytic inverse kinematics of the FR3. / FR3 정·역기구학."""

    def __init__(self, params: FR3Params | None = None):
        self.p = params or FR3Params()
        self.limits = np.radians(np.asarray(self.p.limits_deg, dtype=float))
        self._tool = self.p.tool_transform
        self._tool_inv = np.linalg.inv(self._tool)

    # ---- forward / 정기구학
    def flange(self, q) -> np.ndarray:
        """Base -> flange. / 베이스 → 플랜지."""
        return _dh_chain(self.p.dh, np.asarray(q, dtype=float))[-1]

    def tcp(self, q) -> np.ndarray:
        """Base -> TCP (flange · tool). / 베이스 → TCP."""
        return self.flange(q) @ self._tool

    def frames(self, q) -> dict[str, np.ndarray]:
        """Origins of the joint frames, flange and TCP (world). / 관절 프레임·플랜지·TCP 원점."""
        chain = _dh_chain(self.p.dh, np.asarray(q, dtype=float))
        names = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3", "flange")
        out = {n: t[:3, 3].copy() for n, t in zip(names, chain, strict=True)}
        out["tcp"] = (chain[-1] @ self._tool)[:3, 3]
        return out

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian (6×6) at the TCP, rows ``[v; ω]`` in the base frame.

        TCP 의 기하 야코비안 (베이스 좌표, 행 = [선속도; 각속도]).
        """
        chain = _dh_chain(self.p.dh, np.asarray(q, dtype=float))
        p_tcp = (chain[-1] @ self._tool)[:3, 3]
        jac = np.zeros((6, 6))
        for i in range(6):
            z = chain[i][:3, 2]
            jac[:3, i] = np.cross(z, p_tcp - chain[i][:3, 3])
            jac[3:, i] = z
        return jac

    def manipulability(self, q) -> float:
        """Smallest singular value of the Jacobian (0 at a singularity). / 야코비안 최소 특이값."""
        return float(np.linalg.svd(self.jacobian(q), compute_uv=False)[-1])

    # ---- limits / 리밋
    def in_limits(self, q, margin_deg: float = 0.0) -> bool:
        q = np.asarray(q, dtype=float)
        m = math.radians(margin_deg)
        return bool(np.all(q >= self.limits[:, 0] + m - 1e-9) and np.all(q <= self.limits[:, 1] - m + 1e-9))

    def wrap_near(self, q, q_ref=None, margin_deg: float = 0.0) -> np.ndarray | None:
        """Shift each joint by multiples of 2π into the limits, nearest to ``q_ref``.

        Returns ``None`` if some joint has no value ``margin_deg`` inside its limits. The analytic
        solver returns angles in [-π, π); j2 and j4 of the FR3 run from -265° to 85°, so a solution
        such as j2 = 100° is only reachable as -260°.
        각 관절을 2π 배수만큼 옮겨 리밋 안으로 넣는다(``q_ref`` 에 가장 가깝게). 해석 해는 [-π, π) 이고
        FR3 의 j2·j4 는 -265°~85° 이므로 j2 = 100° 같은 해는 -260° 로만 도달한다.
        """
        q = np.asarray(q, dtype=float)
        ref = np.zeros(6) if q_ref is None else np.asarray(q_ref, dtype=float)
        lo = self.limits[:, 0] + math.radians(margin_deg) - 1e-9
        hi = self.limits[:, 1] - math.radians(margin_deg) + 1e-9
        out = np.empty(6)
        for i in range(6):
            options = [q[i] + 2.0 * math.pi * k for k in range(-2, 3)]
            options = [v for v in options if lo[i] <= v <= hi[i]]
            if not options:
                return None
            out[i] = min(options, key=lambda v, r=ref[i]: abs(v - r))
        return out

    # ---- inverse / 역기구학
    def inverse_flange_all(self, flange_pose: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        """All analytic solutions for a flange pose, wrapped to [-π, π). / 플랜지 포즈의 모든 해."""
        d1, a2, a3, d4, d5, d6 = self.p.dh
        t06 = np.asarray(flange_pose, dtype=float)
        p06 = t06[:3, 3]
        p05 = p06 - d6 * t06[:3, 2]
        r05 = math.hypot(p05[0], p05[1])
        if r05 <= abs(d4):
            return []
        t60 = np.linalg.inv(t06)
        phi = math.atan2(p05[1], p05[0])
        psi = math.acos(max(-1.0, min(1.0, d4 / r05)))
        solutions = []
        for sh, th1 in ((0, phi + psi + math.pi / 2), (1, phi - psi + math.pi / 2)):
            s1, c1 = math.sin(th1), math.cos(th1)
            c5 = (p06[0] * s1 - p06[1] * c1 - d4) / d6
            if abs(c5) > 1.0 + 1e-9:
                continue
            c5 = max(-1.0, min(1.0, c5))
            for wr, th5 in ((0, math.acos(c5)), (1, -math.acos(c5))):
                s5 = math.sin(th5)
                if abs(s5) < 1e-6:
                    th6 = 0.0  # wrist singularity: j6 free / 손목 특이점: j6 임의
                else:
                    th6 = math.atan2(
                        (-t60[1, 0] * s1 + t60[1, 1] * c1) / s5, (t60[0, 0] * s1 - t60[0, 1] * c1) / s5
                    )
                t01 = (
                    transform(rot_z(th1)) @ transform(translation=(0, 0, d1)) @ transform(rot_x(math.pi / 2))
                )
                t45 = (
                    transform(rot_z(th5)) @ transform(translation=(0, 0, d5)) @ transform(rot_x(-math.pi / 2))
                )
                t56 = transform(rot_z(th6)) @ transform(translation=(0, 0, d6))
                t14 = np.linalg.inv(t01) @ t06 @ np.linalg.inv(t45 @ t56)
                p14 = t14[:3, 3]
                r14 = math.hypot(p14[0], p14[1])
                c3 = (r14 * r14 - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
                if abs(c3) > 1.0 + 1e-9:
                    continue
                c3 = max(-1.0, min(1.0, c3))
                for el, th3 in ((0, math.acos(c3)), (1, -math.acos(c3))):
                    th2 = math.atan2(p14[1], p14[0]) - math.atan2(a3 * math.sin(th3), a2 + a3 * math.cos(th3))
                    t13 = (
                        transform(rot_z(th2))
                        @ transform(translation=(a2, 0, 0))
                        @ transform(rot_z(th3))
                        @ transform(translation=(a3, 0, 0))
                    )
                    t34 = np.linalg.inv(t13) @ t14
                    th4 = math.atan2(t34[1, 0], t34[0, 0])
                    q = wrap_angle([th1, th2, th3, th4, th5, th6])
                    if np.linalg.norm(self.flange(q)[:3, 3] - p06) > 1e-6:
                        continue
                    solutions.append((q, (sh, el, wr)))
        return solutions

    def inverse_all(self, tcp_pose: np.ndarray, q_ref=None) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        """All solutions for a TCP pose that fit the joint limits. / 리밋 안의 TCP 포즈 해 전부."""
        flange = np.asarray(tcp_pose, dtype=float) @ self._tool_inv
        out = []
        for q, branch in self.inverse_flange_all(flange):
            qw = self.wrap_near(q, q_ref)
            if qw is not None:
                out.append((qw, branch))
        return out

    def inverse(self, tcp_pose: np.ndarray, branch=None, q_ref=None) -> np.ndarray | None:
        """One solution, or ``None`` if unreachable. / 해 하나 (도달 불가면 None).

        ``branch`` keeps a fixed (shoulder, elbow, wrist) branch, which is what a continuous
        Cartesian path needs. ``q_ref`` picks the solution (and joint wrap) nearest to it.
        ``branch`` 를 주면 그 가지만(연속 직교 경로에 필요), ``q_ref`` 를 주면 가장 가까운 해.
        """
        solutions = self.inverse_all(tcp_pose, q_ref)
        if branch is not None:
            solutions = [s for s in solutions if s[1] == tuple(branch)]
        if not solutions:
            return None
        if q_ref is None:
            return solutions[0][0]
        ref = np.asarray(q_ref, dtype=float)
        return min(solutions, key=lambda s: float(np.abs(s[0] - ref).max()))[0]


# ----------------------------------------------------------------------------- controller config
def controller_config(branch: Sequence[int]) -> int:
    """Branch (shoulder, elbow, wrist) -> controller ``config`` 0..7 (MoveJ, GetInverseKin).

    Checked against 43 ``GetInverseKin`` solutions on firmware v3.8.6:
    ``config = shoulder + 2 (1 - elbow) + 4 (1 - wrist)``.
    가지 → 제어기 config 번호 (펌웨어 v3.8.6 의 GetInverseKin 43 해와 대조).
    """
    sh, el, wr = (int(v) for v in branch)
    return sh + 2 * (1 - el) + 4 * (1 - wr)


def branch_of_config(config: int) -> tuple[int, int, int]:
    """Inverse of :func:`controller_config`. / config 번호 → 가지."""
    c = int(config)
    return (c & 1, 1 - ((c >> 1) & 1), 1 - ((c >> 2) & 1))


# ----------------------------------------------------------------------------- DH fit
@dataclass
class DhFit:
    """Result of :func:`fit_dh`. / DH 피팅 결과."""

    params: FR3Params
    rms_before_mm: float
    max_before_mm: float
    rms_after_mm: float
    max_after_mm: float
    n_samples: int
    fitted: tuple[str, ...]

    def summary(self) -> str:
        values = ", ".join(f"{k}={getattr(self.params, k):.6f}" for k in DH_NAMES)
        return (
            f"{self.n_samples} samples, fitted {', '.join(self.fitted)}\n"
            f"  before: rms {self.rms_before_mm:.4f} mm, max {self.max_before_mm:.4f} mm\n"
            f"  after:  rms {self.rms_after_mm:.4f} mm, max {self.max_after_mm:.4f} mm\n"
            f"  {values}, tool_xyz={[round(v, 6) for v in self.params.tool_xyz]}"
        )


def fit_dh(
    joints_deg,
    poses,
    params: FR3Params | None = None,
    fit_tool: bool = False,
    rotation_weight_mm: float = 0.0,
    iterations: int = 30,
) -> DhFit:
    """Fit the six link parameters (and optionally the tool offset) to controller poses.

    ``joints_deg`` is ``(N, 6)``; ``poses`` is ``(N, 3)`` positions in mm or ``(N, 6)`` controller
    poses ``[x, y, z, rx, ry, rz]`` (mm, deg) of the frame the controller reported (the flange when
    its tool frame is 0, otherwise the TCP; pass the matching ``params.tool_*``). With
    ``rotation_weight_mm > 0`` the orientation error (rad) also counts, scaled to mm.
    ``fit_tool`` adds the tool x and y offsets; the tool z offset is not separable from ``d6``.

    여섯 링크 파라미터(선택: 툴 x·y 오프셋)를 제어기 포즈에 맞춘다. 포즈는 위치(mm) 또는 제어기 6요소
    포즈. 툴 z 는 d6 와 구분되지 않으므로 맞추지 않는다.
    """
    q = np.radians(np.asarray(joints_deg, dtype=float))
    poses = np.asarray(poses, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or len(poses) != len(q):
        raise ValueError("joints_deg must be (N, 6) and poses (N, 3) or (N, 6)")
    params = params or FR3Params()
    target_pos = poses[:, :3] / 1000.0
    use_rot = rotation_weight_mm > 0 and poses.shape[1] >= 6
    target_rot = [rpy_to_matrix(*np.radians(p[3:6])) for p in poses] if use_rot else None

    names = list(DH_NAMES) + (["tool_x", "tool_y"] if fit_tool else [])

    def unpack(x) -> FR3Params:
        p = params.with_dh(x[:6])
        if fit_tool:
            p = replace(p, tool_xyz=(float(x[6]), float(x[7]), p.tool_xyz[2]))
        return p

    def residual(x) -> np.ndarray:
        kin = FR3Kinematics(unpack(x))
        out = []
        for i in range(len(q)):
            t = kin.tcp(q[i])
            out.append((t[:3, 3] - target_pos[i]) * 1000.0)
            if use_rot:
                d = target_rot[i].T @ t[:3, :3]
                w = 0.5 * np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
                out.append(w * rotation_weight_mm)
        return np.concatenate(out)

    x = np.concatenate([params.dh, np.array(params.tool_xyz[:2]) if fit_tool else np.zeros(0)])
    r0 = residual(x)
    for _ in range(iterations):
        r = residual(x)
        jac = np.zeros((r.size, x.size))
        for k in range(x.size):
            dx = np.zeros(x.size)
            dx[k] = 1e-7
            jac[:, k] = (residual(x + dx) - r) / 1e-7
        step = np.linalg.lstsq(jac, -r, rcond=None)[0]
        x = x + step
        if np.abs(step).max() < 1e-10:
            break
    r1 = residual(x)

    def stats(r) -> tuple[float, float]:
        e = np.linalg.norm(r.reshape(len(q), -1)[:, :3], axis=1)
        return float(np.sqrt(np.mean(e * e))), float(e.max())

    (rms0, max0), (rms1, max1) = stats(r0), stats(r1)
    return DhFit(unpack(x), rms0, max0, rms1, max1, len(q), tuple(names))
