"""Synthetic eye-to-hand rig: ground truth plus noisy detections (no robot, no camera).

A camera is placed in front of the robot looking at a table, a board is mounted on the gripper
side that faces the camera, and the "camera" returns noisy corners of what it would detect. The
robot's real flange differs from forward kinematics by a small random error. :func:`compare` runs
the pose-selection strategies against simulated robot time:

    python -m fr3tools.calib.simulate --seeds 10 --budget 60

합성 eye-to-hand 장치: 참값 카메라·보드 장착과 잡음 섞인 검출. 실제 플랜지는 정기구학과 작은
무작위 오차만큼 다르다. :func:`compare` 는 자세 선택 전략을 모의 로봇 시간 기준으로 비교한다.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

from fr3tools.calib.handeye import (
    ActiveHandEye,
    GripperBoard,
    choose_next,
    init_from_sightings,
    move_time,
    perturb,
    planar_pose,
    project,
    rotation_spread_deg,
)
from fr3tools.calib.poses import generate, search_poses, table_grid, table_pose_from
from fr3tools.kinematics import FR3Kinematics, FR3Params, top_down_rotation, transform


def look_at(position, target) -> np.ndarray:
    """Camera pose (OpenCV axes: x right, y down, z forward) at ``position`` facing ``target``.

    ``position`` 에서 ``target`` 을 보는 카메라 포즈(OpenCV 축).
    """
    position = np.asarray(position, dtype=float)
    z = np.asarray(target, dtype=float) - position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return transform(np.column_stack([x, np.cross(z, x), z]), position)


@dataclass
class Scene:
    """Default synthetic setup: an FR3 with a 0.2 m tool, a table 0.45 m in front, a 1080p camera.

    기본 합성 장면: 0.2 m 툴의 FR3, 앞 0.45 m 테이블, 1080p 카메라.
    """

    params: FR3Params = field(default_factory=lambda: FR3Params(tool_xyz=(0.0, 0.0, 0.2)))
    camera_matrix: np.ndarray = field(
        default_factory=lambda: np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    )
    distortion: np.ndarray = field(default_factory=lambda: np.array([0.03, -0.04, 0.0, 0.0, 0.0]))
    image_size: tuple[int, int] = (1920, 1080)
    table: np.ndarray = field(default_factory=lambda: table_pose_from(0.0, -0.45, 0.0))
    camera_position: tuple[float, float, float] = (0.35, -1.05, 0.55)
    height: float = 0.15  # TCP above the table at the centre pose / 중심 자세의 TCP 높이
    # Tool yaw of the top-down posture; 180 deg keeps j6 near 0 here (j6 at 167 deg would leave no
    # room for the yaw sweep). / 탑다운 자세의 툴 yaw. 여기서는 180° 여야 j6 가 0 근처.
    yaw_deg: float = 180.0
    min_tcp_z: float = 0.06

    @property
    def kin(self) -> FR3Kinematics:
        return FR3Kinematics(self.params)

    @property
    def base_rotation(self) -> np.ndarray:
        return top_down_rotation(math.radians(self.yaw_deg))

    @property
    def centre(self) -> np.ndarray:
        return self.table[:3, 3] + self.table[:3, :3] @ np.array([0.0, 0.0, self.height])

    @property
    def camera_nominal(self) -> np.ndarray:
        return look_at(self.camera_position, self.table[:3, 3] + (0.0, 0.0, 0.1))

    def points(self) -> np.ndarray:
        return table_grid(self.table, 3, 3, 0.1)

    def centre_joints(self) -> np.ndarray:
        q = self.kin.inverse(
            transform(self.base_rotation, self.centre), q_ref=np.radians([90, -70, 70, -90, -90, 0])
        )
        if q is None:
            raise ValueError("centre pose unreachable")
        return q


class Rig:
    """Ground truth and noisy detections. / 참값과 잡음 검출.

    Parameters
    ----------
    camera_offset
        (m, deg) standard deviation of the true camera around the nominal one.
    px_noise
        Corner noise (px).
    fk_noise
        (m, rad) per-pose difference between the real flange and forward kinematics.
    """

    def __init__(
        self,
        scene: Scene,
        board: GripperBoard,
        rng: np.random.Generator,
        camera_offset: tuple[float, float] = (0.04, 3.0),
        px_noise: float = 0.3,
        fk_noise: tuple[float, float] = (0.0002, 0.00035),
        facing_limit_deg: float = 72.0,
    ):
        self.scene, self.board, self.rng = scene, board, rng
        self.kin = scene.kin
        self.k, self.dist = scene.camera_matrix, scene.distortion
        self.px_noise, self.fk_noise = px_noise, fk_noise
        self.facing_limit = math.cos(math.radians(facing_limit_deg))
        self.X_true = perturb(
            scene.camera_nominal,
            np.r_[rng.normal(0, math.radians(camera_offset[1]), 3), rng.normal(0, camera_offset[0], 3)],
        )
        # Board flat on the gripper side facing the camera, 10 cm above the TCP, 45 mm out.
        # 보드: 카메라 쪽 그리퍼 옆면, TCP 위 10 cm, 바깥으로 45 mm.
        q0 = scene.centre_joints()
        tcp = self.kin.tcp(q0)[:3, 3]
        to_cam = self.X_true[:3, 3] - tcp
        horiz = np.array([to_cam[0], to_cam[1], 0.0])
        horiz /= np.linalg.norm(horiz)
        y = np.array([0.0, 0.0, 1.0])
        board_in_base = transform(
            np.column_stack([np.cross(y, horiz), y, horiz]), tcp + (0, 0, 0.10) + 0.045 * horiz
        )
        self.Y_true = np.linalg.inv(self.kin.flange(q0)) @ board_in_base

    def real_flange(self, q) -> np.ndarray:
        """The flange where it really is: FK plus a small error. / 실제 플랜지(FK + 작은 오차)."""
        tp, rot = self.fk_noise
        return perturb(self.kin.flange(q), np.r_[self.rng.normal(0, rot, 3), self.rng.normal(0, tp, 3)])

    def detect_board(self, q, flange=None):
        """``(ids, corners)`` seen at joints ``q``, or ``None``. / 관절 q 에서의 보드 검출."""
        flange = self.real_flange(q) if flange is None else flange
        t = np.linalg.inv(self.X_true) @ flange @ self.Y_true
        if t[2, 3] < 0.1:
            return None
        to_cam = -t[:3, 3] / np.linalg.norm(t[:3, 3])
        if t[:3, 2] @ to_cam < self.facing_limit:
            return None
        uv = project(self.k, self.dist, t, self.board.object_points())
        w, h = self.scene.image_size
        if np.any(uv < 0) or np.any(uv[:, 0] > w) or np.any(uv[:, 1] > h):
            return None
        return list(self.board.ids), uv + self.rng.normal(0, self.px_noise, uv.shape)

    def task_error_mm(self, x_est: np.ndarray, points_base) -> float:
        """RMS error (mm) of camera-measured task points mapped through ``x_est``. / 작업 점 오차."""
        pts = np.asarray(points_base, dtype=float)
        x_inv = np.linalg.inv(self.X_true)
        p_cam = pts @ x_inv[:3, :3].T + x_inv[:3, 3]
        got = p_cam @ x_est[:3, :3].T + x_est[:3, 3]
        return float(np.sqrt(np.mean(np.sum((got - pts) ** 2, axis=1)))) * 1000.0


def run_strategy(
    strategy: str, seed: int, budget: float = 60.0, scene: Scene | None = None, n_candidates=300
):
    """One simulated session; returns ``[(t, n_obs, error_mm, sigma_mm), ...]``. / 모의 세션 하나.

    ``strategy``: ``active`` (information per second), ``random`` (random among predicted-visible)
    or ``blind`` (random). Motion is not simulated, only its duration.
    """
    scene = scene or Scene()
    rng = np.random.default_rng(seed)
    board = GripperBoard.default()
    rig = Rig(scene, board, rng)
    kin = scene.kin
    pts = scene.points()
    q0 = scene.centre_joints()
    est = ActiveHandEye(scene.camera_matrix, scene.distortion, board, image_size=scene.image_size)
    cands = generate(kin, scene.base_rotation, scene.centre, q0, scene.min_tcp_z, n=n_candidates, seed=seed)
    q = q0.copy()
    clock = 0.0
    sightings = []
    # Bootstrap: first sighting, then nearest poses until three with rotation spread.
    # 부트스트랩: 첫 관측 뒤 가까운 자세부터 회전이 퍼진 세 관측까지.
    pool = search_poses(kin, scene.base_rotation, scene.centre, q0)
    while len(sightings) < 3 or rotation_spread_deg([s[0] for s in sightings]) < 15.0:
        if sightings and pool is not cands:
            pool = cands
        options = [c for c in pool if not c.used]
        if not options:
            return None
        c = min(options, key=lambda c: float(np.abs(c.joints - q).max())) if pool is cands else options[0]
        c.used = True
        clock += move_time(c.joints - q) + 1.4
        q = c.joints
        hit = rig.detect_board(q)
        if hit is None:
            continue
        fk = kin.flange(q)
        sightings.append((fk, planar_pose(rig.k, rig.dist, board.object_points(hit[0]), hit[1]), hit))
    x0, y0 = init_from_sightings([s[0] for s in sightings], [s[1] for s in sightings])
    est.initialise(x0, y0)
    for fk, _, (ids, uv) in sightings:
        est.add(fk, ids, uv)
    est.solve()
    curve = [(clock, len(est.observations), rig.task_error_mm(est.X, pts), float("nan"))]
    while clock < budget:
        if strategy == "active":
            c, _ = choose_next(est, cands, q, pts)
        else:
            options = [
                c
                for c in cands
                if not c.used and (strategy == "blind" or est.predict(c.flange_in_base).visible)
            ]
            c = options[rng.integers(len(options))] if options else None
        if c is None:
            break
        dt = move_time(c.joints - q) + 1.4
        if clock + dt > budget:
            break
        clock += dt
        q = c.joints
        c.used = True
        hit = rig.detect_board(q)
        if hit is None:
            c.failed += 1
            continue
        est.add(kin.flange(q), *hit)
        est.solve()
        sigma = est.task_uncertainty(est.information(), pts) * 1e3
        curve.append((clock, len(est.observations), rig.task_error_mm(est.X, pts), sigma))
    return curve


def compare(strategies=("active", "random", "blind"), seeds: int = 5, budget: float = 60.0) -> dict:
    """Median task-point error per strategy at a few time marks. / 전략별 시간대 오차 중앙값."""
    marks = [t for t in (20, 30, 45, 60, 90, 120) if t <= budget]
    out = {}
    for s in strategies:
        finals: dict = {t: [] for t in marks}
        poses = []
        for seed in range(seeds):
            curve = run_strategy(s, seed, budget)
            if not curve:
                continue
            poses.append(curve[-1][1])
            for t in marks:
                done = [c for c in curve if c[0] <= t]
                if done:
                    finals[t].append(done[-1][2])
        out[s] = {
            "poses": float(np.mean(poses)) if poses else 0.0,
            **{t: float(np.median(v)) for t, v in finals.items() if v},
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="compare hand-eye pose strategies on synthetic data")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--strategies", default="active,random,blind")
    a = ap.parse_args(argv)
    for name, row in compare(a.strategies.split(","), a.seeds, a.budget).items():
        marks = "  ".join(f"t<={t}s {v:5.2f} mm" for t, v in row.items() if t != "poses")
        print(f"{name:7s} poses {row['poses']:4.1f}  task-point error (median): {marks}", flush=True)


if __name__ == "__main__":
    main()
