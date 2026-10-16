"""Hand-eye maths, pose generation, the synthetic session and camera intrinsics (no robot).

보정 수학, 후보 자세, 합성 세션, 카메라 내부 파라미터 (로봇 없음).
"""

import math

import numpy as np
import pytest

from fr3tools.calib.handeye import (
    ActiveHandEye,
    GripperBoard,
    average_poses,
    init_from_sightings,
    matrix_to_rotvec,
    perturb,
    planar_pose,
    pose_error,
    project,
    rotvec_to_matrix,
    solve_ax_xb,
    undistort_points,
)
from fr3tools.calib.poses import branch_of, generate, search_poses, table_grid, table_pose_from
from fr3tools.calib.run import Settings, SimClock, run_session
from fr3tools.calib.simulate import Rig, Scene, look_at

K = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.03, -0.04, 0.001, -0.001, 0.0])


def random_pose(rng, t_scale=0.3):
    return perturb(np.eye(4), np.r_[rng.normal(0, 1.0, 3), rng.normal(0, t_scale, 3)])


def test_rotvec_round_trip_including_pi():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.normal(0, 1.0, 3)
        assert np.allclose(
            rotvec_to_matrix(matrix_to_rotvec(rotvec_to_matrix(r))), rotvec_to_matrix(r), atol=1e-9
        )
    flip = rotvec_to_matrix([math.pi, 0, 0])
    assert np.allclose(rotvec_to_matrix(matrix_to_rotvec(flip)), flip, atol=1e-9)


def test_average_poses_of_identical_and_symmetric_poses():
    rng = np.random.default_rng(1)
    t = random_pose(rng)
    assert np.allclose(average_poses([t, t, t]), t)
    a, b = perturb(t, [0, 0, 0.1, 0.01, 0, 0]), perturb(t, [0, 0, -0.1, -0.01, 0, 0])
    assert np.abs(pose_error(t, average_poses([a, b]))).max() < 1e-3


def test_undistort_inverts_the_projection():
    pts = np.array([[0.1, -0.05, 0.8], [-0.2, 0.1, 1.0], [0.0, 0.0, 0.6]])
    uv = project(K, DIST, np.eye(4), pts)
    assert np.allclose(undistort_points(K, DIST, uv), pts[:, :2] / pts[:, 2:], atol=1e-8)


def test_planar_pose_recovers_a_board():
    board = GripperBoard.default()
    truth = look_at([0.1, -0.2, -0.7], [0.0, 0.0, 0.0])
    truth = np.linalg.inv(truth)  # board in camera / 카메라 기준 보드
    uv = project(K, DIST, truth, board.object_points())
    est = planar_pose(K, DIST, board.object_points(), uv)
    assert np.abs(pose_error(truth, est)).max() < 1e-6


def test_ax_xb_recovers_the_hidden_transform():
    rng = np.random.default_rng(2)
    x = random_pose(rng)
    a_list = [random_pose(rng) for _ in range(6)]
    b_list = [np.linalg.inv(x) @ a @ x for a in a_list]
    assert np.abs(pose_error(x, solve_ax_xb(a_list, b_list))).max() < 1e-8


def test_init_from_sightings_eye_to_hand():
    rng = np.random.default_rng(3)
    x, y = random_pose(rng, 1.0), random_pose(rng, 0.1)
    flanges = [random_pose(rng, 0.3) for _ in range(5)]
    boards = [np.linalg.inv(x) @ f @ y for f in flanges]
    x0, y0 = init_from_sightings(flanges, boards)
    assert np.abs(pose_error(x, x0)).max() < 1e-8
    assert np.abs(pose_error(y, y0)).max() < 1e-8


def test_estimator_recovers_x_and_y_from_noisy_corners():
    scene = Scene()
    board = GripperBoard.default()
    rig = Rig(scene, board, np.random.default_rng(4))
    q0 = scene.centre_joints()
    cands = generate(scene.kin, scene.base_rotation, scene.centre, q0, scene.min_tcp_z, n=60, seed=4)
    obs = [(scene.kin.flange(c.joints), rig.detect_board(c.joints)) for c in cands]
    obs = [(f, hit) for f, hit in obs if hit is not None][:15]
    assert len(obs) >= 10
    est = ActiveHandEye(scene.camera_matrix, scene.distortion, board, image_size=scene.image_size)
    # Start 2 cm / 3 deg off the truth. / 참값에서 2 cm, 3° 어긋난 초기값.
    est.initialise(
        perturb(rig.X_true, [0.05, 0, 0, 0.02, 0, 0]), perturb(rig.Y_true, [0, 0.03, 0, 0, 0.01, 0])
    )
    for flange, (ids, uv) in obs:
        est.add(flange, ids, uv)
    st = est.solve()
    assert st["reproj_rms_px"] < 1.0
    assert rig.task_error_mm(est.X, scene.points()) < 1.0
    assert np.linalg.norm(pose_error(rig.Y_true, est.Y)[3:]) < 0.002


def test_task_uncertainty_shrinks_with_information():
    scene = Scene()
    rig = Rig(scene, GripperBoard.default(), np.random.default_rng(5))
    est = ActiveHandEye(
        scene.camera_matrix, scene.distortion, GripperBoard.default(), image_size=scene.image_size
    )
    est.initialise(rig.X_true, rig.Y_true)
    q0 = scene.centre_joints()
    h = est.information()
    before = est.task_uncertainty(h, scene.points())
    after = est.task_uncertainty(h + est.pose_information(scene.kin.flange(q0)), scene.points())
    assert after < before
    assert est.predict(scene.kin.flange(q0)).visible


def test_table_grid_and_candidates_respect_limits():
    pts = table_grid(table_pose_from(0.0, -0.45, 0.0, 90.0), 3, 2, 0.1)
    assert pts.shape == (6, 3) and np.allclose(pts.mean(axis=0), [0.0, -0.45, 0.0])
    scene = Scene()
    kin = scene.kin
    q0 = scene.centre_joints()
    cands = generate(kin, scene.base_rotation, scene.centre, q0, scene.min_tcp_z, n=40, seed=1)
    assert len(cands) == 40
    for c in cands:
        assert kin.in_limits(c.joints, margin_deg=10.0)
        assert kin.tcp(c.joints)[2, 3] >= scene.min_tcp_z - 1e-9
    sweep = search_poses(kin, scene.base_rotation, scene.centre, q0)
    assert len(sweep) >= 6
    for a, b in zip(sweep, sweep[1:], strict=False):
        assert np.degrees(np.abs(a.joints - b.joints)).max() < 200.0
    assert branch_of(kin, q0) is not None


@pytest.mark.parametrize("use_prior", [False, True])
def test_fake_session_recovers_the_task_points(use_prior):
    from fr3tools.ros.adapter import FakeRobot

    scene = Scene()
    board = GripperBoard.default()
    rig = Rig(scene, board, np.random.default_rng(7))
    clock = SimClock()
    robot = FakeRobot(scene.centre_joints(), scene.params)
    from fr3tools.calib.run import FakeSensor

    prior = scene.camera_nominal if use_prior else None
    est = ActiveHandEye(scene.camera_matrix, scene.distortion, board, prior, image_size=scene.image_size)
    settings = Settings(budget=30.0, target_mm=0.0, min_poses=6, candidates=80, seed=7)
    result = run_session(
        robot,
        FakeSensor(rig, robot, clock),
        clock,
        est,
        scene.kin,
        scene.base_rotation,
        scene.centre,
        scene.points(),
        scene.min_tcp_z,
        settings,
        log=lambda s: None,
    )
    assert result["n_poses"] >= 6
    assert rig.task_error_mm(est.X, scene.points()) < 1.0
    assert not robot.streaming  # the session always ends the stream / 세션은 항상 스트림을 끝낸다


def test_session_refuses_a_tilted_start():
    from fr3tools.calib.run import FakeSensor
    from fr3tools.ros.adapter import FakeRobot

    scene = Scene()
    robot = FakeRobot(np.radians([0, -90, 90, 0, 90, 0]), scene.params)
    est = ActiveHandEye(K, DIST, GripperBoard.default())
    with pytest.raises(RuntimeError, match="top-down"):
        run_session(
            robot,
            FakeSensor(None, robot, SimClock()),
            SimClock(),
            est,
            scene.kin,
            scene.base_rotation,
            scene.centre,
            scene.points(),
            0.06,
            Settings(),
            log=lambda s: None,
        )
    assert not robot.streaming


def test_chessboard_calibration_on_synthetic_views():
    cv2 = pytest.importorskip("cv2")
    from fr3tools.calib.camera import calibrate, chessboard_points, normal_spread_deg, view_normal

    rng = np.random.default_rng(8)
    obj = chessboard_points(9, 6, 0.025)
    centre = obj.mean(axis=0)
    objs, imgs, normals = [], [], []
    for _ in range(15):
        eye = centre + np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), -rng.uniform(0.35, 0.6)])
        cam = look_at(eye, centre + rng.normal(0, 0.02, 3))
        board_in_cam = np.linalg.inv(cam)
        rvec, _ = cv2.Rodrigues(board_in_cam[:3, :3])
        uv, _ = cv2.projectPoints(obj, rvec, board_in_cam[:3, 3], K, DIST)
        uv = uv.reshape(-1, 2) + rng.normal(0, 0.1, (len(obj), 2))
        objs.append(obj)
        imgs.append(uv.astype(np.float32))
        normals.append(view_normal(obj, uv, (1920, 1080)))
    result = calibrate(objs, imgs, (1920, 1080), fix_k3=True)
    k = np.asarray(result["K"])
    assert result["rms_px"] < 0.3
    assert abs(k[0, 0] - 1400.0) < 15.0 and abs(k[0, 2] - 960.0) < 15.0
    assert normal_spread_deg(normals) > 10.0
