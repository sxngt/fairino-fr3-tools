"""Calibration poses around the work area and the default task points. / 보정 후보 자세와 작업 점.

Poses keep the TCP near a centre above the table, tilted about two horizontal axes and turned about
the vertical so that relative rotations span all three axes (hand-eye needs rotation diversity).
Every pose is checked for reach, joint-limit margin (the FR3 stops with error 29 at a soft limit)
and clearance above the table.

후보 자세: 테이블 위 중심 근처에서 두 수평축 기울기 + 수직축 회전(상대 회전이 세 축을 모두 덮도록).
도달성, 관절 리밋 여유(리밋에 닿으면 FR3 는 에러 29 로 멈춘다), 테이블 위 여유 높이를 검사한다.
"""

from __future__ import annotations

import math

import numpy as np

from fr3tools.calib.handeye import Candidate
from fr3tools.kinematics import FR3Kinematics, rot_x, rot_y, rot_z, transform


def table_grid(table_pose: np.ndarray, nx: int = 3, ny: int = 3, spacing: float = 0.1, height: float = 0.0):
    """``nx × ny`` points on the table, ``spacing`` apart, ``height`` above it (base frame).

    테이블 위 격자 점(베이스 좌표). ``table_pose`` 는 테이블 중심 포즈(z = 테이블 법선).
    """
    xs = (np.arange(nx) - (nx - 1) / 2.0) * spacing
    ys = (np.arange(ny) - (ny - 1) / 2.0) * spacing
    local = np.array([[x, y, height] for x in xs for y in ys])
    t = np.asarray(table_pose, dtype=float)
    return local @ t[:3, :3].T + t[:3, 3]


def table_pose_from(x: float, y: float, z: float, yaw_deg: float = 0.0) -> np.ndarray:
    """Level table pose at (x, y, z) m, turned by ``yaw_deg``. / 수평 테이블 포즈."""
    return transform(rot_z(math.radians(yaw_deg)), (x, y, z))


def tcp_tilt_deg(pose: np.ndarray, base_rotation: np.ndarray) -> float:
    """Angle between the TCP z axis and that of ``base_rotation`` (deg). / TCP z 축 기울기."""
    c = float(pose[:3, 2] @ base_rotation[:, 2])
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def path_clear(
    kin: FR3Kinematics,
    q0,
    q1,
    min_tcp_z: float,
    tip_radius: float = 0.0,
    min_flange_z=None,
    samples: int = 12,
) -> bool:
    """Joint-space straight path keeps the TCP (minus the tool radius when tilted) and the flange
    above their floors. / 관절공간 직선 경로가 바닥 높이 위를 지나는지.
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    floor = min_tcp_z if min_flange_z is None else min_flange_z
    for s in np.linspace(0.0, 1.0, samples):
        q = q0 + (q1 - q0) * s
        tcp = kin.tcp(q)
        drop = tip_radius * math.sqrt(max(0.0, 1.0 - tcp[2, 2] ** 2))
        if tcp[2, 3] - drop < min_tcp_z or kin.flange(q)[2, 3] < floor:
            return False
    return True


def generate(
    kin: FR3Kinematics,
    base_rotation: np.ndarray,
    centre,
    q_start,
    min_tcp_z: float,
    branch=None,
    tip_radius: float = 0.05,
    n: int = 300,
    span_xy: float = 0.07,
    span_z: tuple[float, float] = (0.0, 0.09),
    max_tilt_deg: float = 28.0,
    max_yaw_deg: float = 60.0,
    max_joint_step_deg: float = 100.0,
    limit_margin_deg: float = 10.0,
    seed: int = 0,
) -> list[Candidate]:
    """Random reachable TCP poses near ``centre`` (base frame). / ``centre`` 근처의 도달 가능 TCP 포즈.

    ``branch`` keeps one IK branch (``None``: the solution nearest ``q_start``).
    """
    rng = np.random.default_rng(seed)
    q_start = np.asarray(q_start, dtype=float)
    out: list[Candidate] = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        dx, dy = rng.uniform(-span_xy, span_xy, 2)
        dz = rng.uniform(*span_z)
        tx, ty = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, 2))
        yaw = math.radians(rng.uniform(-max_yaw_deg, max_yaw_deg))
        rot = rot_z(yaw) @ rot_x(tx) @ rot_y(ty) @ base_rotation
        pose = transform(rot, np.asarray(centre, dtype=float) + (dx, dy, dz))
        q = kin.inverse(pose, branch=branch, q_ref=q_start)
        if q is None:
            continue
        q = kin.wrap_near(q, q_start, margin_deg=limit_margin_deg)
        if q is None or np.degrees(np.abs(q - q_start)).max() > max_joint_step_deg:
            continue
        drop = tip_radius * math.sin(math.radians(tcp_tilt_deg(pose, base_rotation)))
        if pose[2, 3] - drop < min_tcp_z:
            continue
        label = (
            f"d({dx * 1e3:+.0f},{dy * 1e3:+.0f},{dz * 1e3:+.0f})mm "
            f"t({math.degrees(tx):+.0f},{math.degrees(ty):+.0f}) yaw{math.degrees(yaw):+.0f}"
        )
        out.append(Candidate(q, kin.flange(q), label))
    return out


def search_poses(
    kin: FR3Kinematics, base_rotation, centre, q_start, branch=None, step_deg: int = 30, limit_margin_deg=10.0
) -> list[Candidate]:
    """Poses for the first sighting: turn about the vertical over ``centre``. / 첫 관측용 수직축 회전.

    Sweeps one way until a joint limit, then the other way until a full turn is covered. Each pose
    is wrapped next to the previous one, so the wrist never swings across.
    한쪽으로 관절 한계까지, 반대쪽으로 한 바퀴가 덮일 때까지. 각 자세는 직전 자세 옆으로 감는다.
    """

    def at(yaw_deg, q_ref):
        pose = transform(rot_z(math.radians(yaw_deg)) @ base_rotation, centre)
        q = kin.inverse(pose, branch=branch, q_ref=q_ref)
        if q is None:
            return None
        q = q_ref + (q - q_ref + math.pi) % (2 * math.pi) - math.pi
        return q if kin.in_limits(q, limit_margin_deg) else None

    out = []
    q0 = at(0, np.asarray(q_start, dtype=float))
    if q0 is None:
        return out
    out.append(Candidate(q0, kin.flange(q0), "search yaw+0"))
    reach = {1: 0, -1: 0}
    for sign in (1, -1):
        q_prev = q0
        y = 0
        while reach[1] + reach[-1] < 360 - step_deg:
            y += step_deg
            q = at(sign * y, q_prev)
            if q is None:
                break
            out.append(Candidate(q, kin.flange(q), f"search yaw{sign * y:+d}"))
            reach[sign] = y
            q_prev = q
    return out


def branch_of(kin: FR3Kinematics, q) -> tuple[int, int, int] | None:
    """IK branch of the joint vector ``q``. / 관절 벡터의 IK 가지."""
    q = np.asarray(q, dtype=float)
    for qs, branch in kin.inverse_all(kin.tcp(q), q_ref=q):
        if np.abs(qs - q).max() < 1e-6:
            return branch
    return None
