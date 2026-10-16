"""``fr3-handeye``: active eye-to-hand calibration on the robot, or synthetic with ``--fake SEED``.

The robot shows the gripper board to the fixed camera from poses it picks itself:

1. it rises straight up to the working height, then moves over the centre of the work area;
2. first sighting: it turns the tool about the vertical until the board is detected;
3. bootstrap: it visits nearby poses until three sightings with at least 15° of rotation between
   them, and solves ``AX = XB`` for a starting estimate (or uses ``--camera-guess`` as a prior);
4. active loop: after every pose the estimate is re-solved and the next pose is the one with the
   largest predicted drop of the task-point uncertainty per second of motion, until the points are
   known to ``--target-mm`` or the time budget runs out;
5. it returns over the centre and writes ``camera_in_base`` (``T_BC``), ``board_in_flange``, the
   predicted uncertainty, a leave-one-out spread and the mount consistency to ``--out``.

Motion is minimum-jerk joint paths (≤ 24°/s, ≤ 36°/s²) streamed through ``fr3-driver``, whose own
limits and safety monitor stay in force. Without ``--execute`` it only checks the camera, the board
and the candidate poses.

로봇이 스스로 고른 자세로 그리퍼 보드를 고정 카메라에 보여 준다: 작업 높이로 올라가 중심 위로 → 수직축
회전으로 첫 관측 → 회전이 다른 세 관측으로 AX = XB 초기값 → 자세마다 재추정하며 "이동 1초당 작업 점
불확실성 감소"가 가장 큰 자세로 이동 → 결과 저장. ``--execute`` 없이는 점검만 한다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from fr3tools.calib.handeye import (
    ActiveHandEye,
    GripperBoard,
    choose_next,
    init_from_sightings,
    move_time,
    planar_pose,
    rotation_spread_deg,
)
from fr3tools.calib.poses import (
    branch_of,
    generate,
    path_clear,
    search_poses,
    table_grid,
    table_pose_from,
    tcp_tilt_deg,
)
from fr3tools.kinematics import FR3Kinematics, FR3Params, pose_to_matrix, top_down_rotation, transform


# ----------------------------------------------------------------------------- clocks
class WallClock:
    """Real time. / 실제 시간."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, dt: float) -> None:
        if dt > 0:
            time.sleep(dt)


class SimClock:
    """Simulated time for ``--fake``: sleeping advances it instantly. / 모의 시간."""

    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, dt)


# ----------------------------------------------------------------------------- sensors
class FakeSensor:
    """Synthetic detections from a :class:`fr3tools.calib.simulate.Rig`. / 합성 검출."""

    def __init__(self, rig, robot, clock, period: float = 0.1):
        self.rig, self.robot, self.clock, self.period = rig, robot, clock, period

    def grab(self, newer_than: float = 0.0):
        due = max(newer_than, self.clock.now()) + self.period
        self.clock.sleep(due - self.clock.now())
        return due, self.rig.detect_board(self.robot.joint_positions())


class CameraSensor:
    """Frames from :class:`fr3tools.calib.camera.CameraSource`, ArUco board detection. / 실제 카메라."""

    def __init__(self, source, detector, clock):
        self.source, self.detector, self.clock = source, detector, clock

    def grab(self, newer_than: float = 0.0):
        while True:
            frame = self.source.read()
            t = self.clock.now()
            if t > newer_than:
                break
        found = self.detector.detect(frame)
        return t, None if found is None else (found[0], found[1])


# ----------------------------------------------------------------------------- motion
class Mover:
    """Smooth joint-space motion over the servo stream. / 서보 스트림 위의 부드러운 관절 이동."""

    def __init__(self, robot, clock, rate: float = 30.0):
        self.robot = robot
        self.clock = clock
        self.dt = 1.0 / rate
        self.target = robot.joint_positions()

    def move(self, q1) -> float:
        """Minimum-jerk path to ``q1``; returns the planned duration. / 최소 저크 경로."""
        q0 = self.target.copy()
        q1 = np.asarray(q1, dtype=float)
        dur = move_time(q1 - q0)
        t0 = self.clock.now()
        while True:
            s = min(1.0, (self.clock.now() - t0) / dur)
            b = s * s * s * (10 - 15 * s + 6 * s * s)
            self.target = q0 + (q1 - q0) * b
            self.robot.stream_joints(self.target)
            if s >= 1.0:
                return dur
            self.clock.sleep(self.dt)

    def settle(self, tol_deg: float = 0.05, still_s: float = 0.25, timeout: float = 2.0) -> np.ndarray:
        """Hold until the measured joints sit on the target. / 실제 관절이 목표에 멈출 때까지 유지."""
        t0 = self.clock.now()
        since = None
        while self.clock.now() - t0 < timeout:
            self.robot.stream_joints(self.target)
            q = self.robot.joint_positions()
            if np.degrees(np.abs(q - self.target)).max() < tol_deg:
                since = self.clock.now() if since is None else since
                if self.clock.now() - since >= still_s:
                    break
            else:
                since = None
            self.clock.sleep(self.dt)
        q = self.robot.joint_positions()
        off = float(np.degrees(np.abs(q - self.target)).max())
        if off > 1.0:
            # The driver ended the stream (limit, error, safety): never keep going blind.
            # 드라이버가 스트림을 끝냈다(리밋·에러·안전): 그대로 계속하지 않는다.
            raise RuntimeError(f"robot is not following the servo stream (off by {off:.1f} deg)")
        return q

    def line(self, kin: FR3Kinematics, goal_pose: np.ndarray, speed: float = 0.03) -> bool:
        """Straight TCP line at a fixed rotation (the first approach). / 고정 회전 직선 이동."""
        start = kin.tcp(self.target)
        p0, p1 = start[:3, 3], goal_pose[:3, 3]
        dur = max(1.0, 1.875 * float(np.linalg.norm(p1 - p0)) / speed)
        t0 = self.clock.now()
        while True:
            s = min(1.0, (self.clock.now() - t0) / dur)
            b = s * s * s * (10 - 15 * s + 6 * s * s)
            q = kin.inverse(transform(goal_pose[:3, :3], p0 + (p1 - p0) * b), q_ref=self.target)
            if q is None or np.degrees(np.abs(q - self.target)).max() > 5.0:
                return False
            self.target = q
            self.robot.stream_joints(q)
            if s >= 1.0:
                return True
            self.clock.sleep(self.dt)


# ----------------------------------------------------------------------------- session
@dataclass
class Settings:
    budget: float = 60.0  # s of robot time / 로봇 시간 예산
    target_mm: float = 0.3
    min_poses: int = 6
    frames: int = 2  # frames averaged per pose / 자세당 평균 프레임
    dwell: float = 1.4
    candidates: int = 300
    min_spread_deg: float = 15.0
    seed: int = 0


def run_session(
    robot,
    sensor,
    clock,
    est: ActiveHandEye,
    kin: FR3Kinematics,
    base_rotation: np.ndarray,
    centre: np.ndarray,
    points: np.ndarray,
    min_tcp_z: float,
    settings: Settings,
    log=print,
) -> dict:
    """One calibration session on ``robot`` (``FR3Ros`` or ``FakeRobot``). / 보정 세션 하나."""
    q_now = robot.joint_positions()
    tilt = tcp_tilt_deg(kin.tcp(q_now), base_rotation)
    if tilt > 20.0:
        raise RuntimeError(
            f"start posture is {tilt:.0f} deg from top-down; move the tool over the table first"
        )
    branch = branch_of(kin, q_now)
    robot.start_streaming()
    mover = Mover(robot, clock)
    t_start = clock.now()
    history = []

    def observe(tag: str):
        """Settle, average fresh detections. / 정지 후 새 프레임 검출 평균."""
        q = mover.settle()
        fk = kin.flange(q)
        t_seen = clock.now()
        uvs, ids_ref = [], None
        for _ in range(settings.frames):
            t_seen, hit = sensor.grab(t_seen)
            if hit is None or len(hit[0]) < 2:
                continue
            order = {int(i): c for i, c in zip(hit[0], np.asarray(hit[1]).reshape(-1, 4, 2), strict=True)}
            ids_ref = ids_ref or sorted(order)
            if all(i in order for i in ids_ref):
                uvs.append(np.concatenate([order[i] for i in ids_ref]))
        if not uvs:
            return None
        return fk, ids_ref, np.mean(uvs, axis=0)

    try:
        # 1. up, then over the centre / 위로, 그다음 중심 위로
        tcp_now = kin.tcp(mover.target)
        up = transform(tcp_now[:3, :3], (*tcp_now[:2, 3], max(tcp_now[2, 3], centre[2])))
        if not mover.line(kin, up):
            raise RuntimeError("straight-up approach unreachable")
        q_c = kin.inverse(transform(base_rotation, centre), branch=branch, q_ref=mover.target)
        q_c = None if q_c is None else kin.wrap_near(q_c, mover.target, margin_deg=10.0)
        if q_c is None:
            raise RuntimeError("centre pose unreachable or too close to a joint limit")
        mover.move(q_c)
        q0 = mover.settle()
        cands = generate(
            kin,
            base_rotation,
            centre,
            q0,
            min_tcp_z,
            branch=branch,
            n=settings.candidates,
            seed=settings.seed,
        )
        log(f"[handeye] {len(cands)} candidate poses, budget {settings.budget:.0f} s")

        # 2-3. first sighting and bootstrap / 첫 관측과 부트스트랩
        sightings = []
        for c in search_poses(kin, base_rotation, centre, q0, branch=branch):
            mover.move(c.joints)
            o = observe(c.label)
            if o is not None:
                sightings.append(o)
                log(f"[handeye] board found at {c.label} ({len(o[1])} markers)")
                break
        if not sightings:
            raise RuntimeError("gripper board never seen: is it attached and facing the camera?")
        if est.X is not None:
            fk, ids, uv = sightings[0]
            est.initialise_mount(
                fk, planar_pose(est.camera_matrix, est.distortion, est.board.object_points(ids), uv)
            )
        else:
            while (
                len(sightings) < 3 or rotation_spread_deg([s[0] for s in sightings]) < settings.min_spread_deg
            ):
                left = [c for c in cands if not c.used and c.failed == 0]
                if not left or clock.now() - t_start > settings.budget:
                    raise RuntimeError("bootstrap failed: too few sightings with rotation between them")
                c = min(left, key=lambda c: float(np.abs(c.joints - mover.target).max()))
                c.used = True
                if not path_clear(kin, mover.target, c.joints, min_tcp_z):
                    c.failed += 1
                    continue
                mover.move(c.joints)
                o = observe(c.label)
                if o is None:
                    c.failed += 1
                    continue
                sightings.append(o)
            boards = [
                planar_pose(est.camera_matrix, est.distortion, est.board.object_points(ids), uv)
                for _, ids, uv in sightings
            ]
            est.initialise(*init_from_sightings([s[0] for s in sightings], boards))
            log(f"[handeye] bootstrap from {len(sightings)} sightings")
        for fk, ids, uv in sightings:
            est.add(fk, ids, uv, "bootstrap")
        est.solve()

        # 4. active loop / 능동 루프
        while clock.now() - t_start < settings.budget:
            c, info = None, {}
            for _ in range(6):
                c, info = choose_next(est, cands, mover.target, points, settings.dwell)
                if c is None or path_clear(kin, mover.target, c.joints, min_tcp_z):
                    break
                c.failed += 1
            if c is None:
                log("[handeye] no informative pose left")
                break
            if clock.now() - t_start + info["seconds"] > settings.budget + 3:
                break
            mover.move(c.joints)
            c.used = True
            o = observe(c.label)
            if o is None:
                c.failed += 1
                # Learn the detector's viewing-angle limit: tighten the prediction just above the miss.
                # 검출기 시야각 한계를 배운다: 놓친 각도 바로 위로 가시성 예측을 조인다.
                facing = est.predict(c.flange_in_base).facing_cos
                est.min_facing_cos = min(0.7, max(est.min_facing_cos, facing + 0.03))
                angle = math.degrees(math.acos(facing))
                log(f"[handeye] {c.label}: board not detected (facing {angle:.0f} deg)")
                continue
            est.add(*o, c.label)
            st = est.solve()
            sigma = est.task_uncertainty(est.information(), points) * 1e3
            row = {"t": round(clock.now() - t_start, 1), "pose": c.label, "sigma_mm": sigma, **st}
            history.append(row)
            log(
                f"[handeye] {row['t']:5.1f}s #{st['n_obs']:2d} {c.label:36s} "
                f"reproj {st['reproj_rms_px']:.2f} px chi2 {st['chi2_per_dof']:.2f} "
                f"task sigma {sigma:.2f} mm (pred {info['sigma_after_mm']:.2f})"
            )
            if sigma <= settings.target_mm and len(est.observations) >= settings.min_poses:
                break

        # 5. back over the centre / 중심 위로 복귀
        mover.move(q0)
        mover.settle()
    finally:
        robot.stop_streaming()

    if len(est.observations) < settings.min_poses:
        raise RuntimeError(f"only {len(est.observations)} poses observed; calibration not written")
    sigma = est.task_uncertainty(est.information(), points) * 1e3
    return {
        "camera_in_base": est.X.tolist(),
        "camera_pose_mm_deg": [float(v) for v in _pose(est.X)],
        "board_in_flange": est.Y.tolist(),
        "n_poses": len(est.observations),
        "seconds": round(clock.now() - t_start, 1),
        "task_sigma_mm": sigma,
        "leave_one_out_max_mm": est.leave_one_out_mm(points),
        "reproj_rms_px": history[-1]["reproj_rms_px"] if history else None,
        "mount_consistency": est.mount_consistency(),
        "task_points": np.asarray(points).tolist(),
        "log": history,
    }


def _pose(t):
    from fr3tools.kinematics import matrix_to_pose

    return matrix_to_pose(t)


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="active eye-to-hand calibration with a gripper ArUco board")
    ap.add_argument("--fake", type=int, default=-1, metavar="SEED", help="synthetic robot and camera")
    ap.add_argument("--execute", action="store_true", help="robot mode: really move the arm")
    ap.add_argument("--intrinsics", help="camera JSON from fr3-camera-calib (robot mode)")
    ap.add_argument("--device", default="0", help="camera device index or path")
    ap.add_argument("--board", help="gripper board JSON (default: 2x2 DICT_4X4_50 ids 10-13, 25 mm)")
    ap.add_argument("--write-board", metavar="PATH", help="write the default board layout and exit")
    ap.add_argument("--params", help="kinematic parameters JSON (fr3-calib-dh --out); tool = AG-95 TCP")
    ap.add_argument(
        "--table",
        type=float,
        nargs=4,
        default=(0.0, -0.45, 0.0, 0.0),
        metavar=("X", "Y", "Z", "YAW"),
        help="table centre in the base frame (m) and yaw (deg)",
    )
    ap.add_argument("--grid", type=float, nargs=3, default=(3, 3, 0.1), metavar=("NX", "NY", "SPACING"))
    ap.add_argument("--points", help="JSON list of task points [x, y, z] (m, base frame) instead of the grid")
    ap.add_argument(
        "--height", type=float, default=0.15, help="TCP height of the centre pose above the table"
    )
    ap.add_argument("--min-tcp-height", type=float, default=0.06, help="TCP floor above the table (m)")
    ap.add_argument("--tool-yaw", type=float, default=180.0, help="top-down tool yaw (deg)")
    ap.add_argument(
        "--camera-guess",
        type=float,
        nargs=6,
        metavar="V",
        help="rough camera pose x y z (mm) rx ry rz (deg) as a weak prior (optional)",
    )
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--target-mm", type=float, default=0.3)
    ap.add_argument("--min-poses", type=int, default=6)
    ap.add_argument("--out", default="calib_out/handeye.json")
    ap.add_argument("--prefix", default="/fr", help="driver prefix (robot mode)")
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    board = GripperBoard.from_json(a.board) if a.board else GripperBoard.default()
    if a.write_board:
        board.to_json(a.write_board)
        print(f"board layout -> {a.write_board} (print markers {board.ids} of {board.dictionary})")
        return
    settings = Settings(budget=a.budget, target_mm=a.target_mm, min_poses=a.min_poses, seed=max(a.fake, 0))

    if a.fake >= 0:
        from fr3tools.calib.simulate import Rig, Scene
        from fr3tools.ros.adapter import FakeRobot

        scene = Scene()
        clock = SimClock()
        rig = Rig(scene, board, np.random.default_rng(a.fake))
        start = scene.centre_joints()
        start = scene.kin.inverse(
            scene.kin.tcp(start) @ transform(translation=(0.02, -0.01, 0.03)), q_ref=start
        )
        robot = FakeRobot(start, scene.params)
        sensor = FakeSensor(rig, robot, clock)
        prior = pose_to_matrix(a.camera_guess) if a.camera_guess else None
        est = ActiveHandEye(scene.camera_matrix, scene.distortion, board, prior, image_size=scene.image_size)
        result = run_session(
            robot,
            sensor,
            clock,
            est,
            scene.kin,
            scene.base_rotation,
            scene.centre,
            scene.points(),
            float(scene.table[2, 3]) + scene.min_tcp_z,
            settings,
        )
        result["fake_truth_task_error_mm"] = rig.task_error_mm(est.X, scene.points())
        print(f"[handeye] fake truth: task-point error {result['fake_truth_task_error_mm']:.3f} mm")
        _write(a.out, result)
        return

    from fr3tools.calib.camera import ArucoBoardDetector, CameraSource, load_intrinsics

    if not a.intrinsics:
        raise SystemExit("robot mode needs --intrinsics (run fr3-camera-calib first)")
    k, dist, size = load_intrinsics(a.intrinsics)
    params = FR3Params.from_json(a.params) if a.params else FR3Params(tool_xyz=(0.0, 0.0, 0.2))
    kin = FR3Kinematics(params)
    table = table_pose_from(a.table[0], a.table[1], a.table[2], a.table[3])
    if a.points:
        with open(a.points, encoding="utf-8") as f:
            points = np.asarray(json.load(f), dtype=float)
    else:
        points = table_grid(table, int(a.grid[0]), int(a.grid[1]), a.grid[2])
    base_rotation = table[:3, :3] @ top_down_rotation(math.radians(a.tool_yaw))
    centre = table[:3, 3] + table[:3, :3] @ np.array([0.0, 0.0, a.height])
    prior = pose_to_matrix(a.camera_guess) if a.camera_guess else None
    est = ActiveHandEye(k, dist, board, prior, image_size=size)
    source = CameraSource(a.device, *size)
    detector = ArucoBoardDetector(board, k, dist)
    clock = WallClock()
    sensor = CameraSensor(source, detector, clock)
    try:
        _, hit = sensor.grab()
        print(f"[handeye] gripper board in view: {None if hit is None else hit[0]}")
        if not a.execute:
            print("[handeye] dry run: add --execute to move the arm (the driver must be running)")
            return
        import rclpy
        from rclpy.node import Node

        from fr3tools.ros.adapter import FR3Ros

        rclpy.init()
        node = Node("fr3_handeye")
        try:
            robot = FR3Ros(node, params, prefix=a.prefix)
            robot.wait_ready()
            result = run_session(
                robot,
                sensor,
                clock,
                est,
                kin,
                base_rotation,
                centre,
                points,
                float(table[2, 3]) + a.min_tcp_height,
                settings,
            )
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
    finally:
        source.close()
    _write(a.out, result)


def _write(path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    result["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(
        f"[handeye] DONE {result['n_poses']} poses in {result['seconds']} s: task sigma "
        f"{result['task_sigma_mm']:.2f} mm, leave-one-out max {result['leave_one_out_max_mm']:.2f} mm, "
        f"camera {np.round(result['camera_pose_mm_deg'], 1).tolist()} (mm, deg) -> {path}"
    )


if __name__ == "__main__":
    main()
