"""Kinematics against controller-reported poses, IK round trips and the DH fit.

The reference pairs were read from a real FR3 controller (firmware v3.8.6). The calibrated set is
the DH fit of that unit; other units differ by fractions of a millimetre.
실제 FR3 제어기(펌웨어 v3.8.6)에서 읽은 기준 쌍과 그 장비의 DH 피팅값으로 검증한다.
"""

import math
import tempfile

import numpy as np
import pytest

from fr3tools.kinematics import (
    FR3Kinematics,
    FR3Params,
    branch_of_config,
    controller_config,
    fit_dh,
    matrix_to_pose,
    matrix_to_rpy,
    pose_to_matrix,
    rpy_to_matrix,
    top_down_rotation,
)

CALIBRATED = FR3Params(d1=0.14, a2=-0.280491, a3=-0.239929, d4=0.102452, d5=0.102123, d6=0.101)

# (joints deg, controller flange position mm) from GetForwardKin. / 제어기 FK 기준 쌍.
CONTROLLER_FK = [
    ([0, 0, 0, 0, 0, 0], [-520.4194, -203.4524, 37.8768]),
    ([0, -90, 0, -90, 0, 0], [0.0, -203.4524, 762.5426]),
    ([10, -80, 30, -100, 20, 30], [-186.3998, -233.2733, 705.7388]),
    ([-30, -60, 70, -120, -40, 15], [-518.3323, 91.6178, 315.1709]),
]
# A full state-packet pose (flange_cur_pos, mm and deg). / 상태 패킷의 전체 포즈.
PACKET_Q = [
    76.2692642211914,
    -62.483489990234375,
    70.84745788574219,
    -98.35896301269531,
    -90.0,
    -104.7704086303711,
]
PACKET_POSE = [-11.81578, -479.99167, 252.85191, -179.99872, 0.00484, -88.96033]


def random_q(rng, kin, margin=5.0):
    lim = np.degrees(kin.limits) + np.array([margin, -margin])
    return np.radians(rng.uniform(lim[:, 0], lim[:, 1]))


def test_forward_matches_controller_fk():
    kin = FR3Kinematics(CALIBRATED)
    for q_deg, pos in CONTROLLER_FK:
        assert np.allclose(kin.flange(np.radians(q_deg))[:3, 3] * 1000.0, pos, atol=0.01)


def test_packet_pose_matches_including_orientation():
    pose = matrix_to_pose(FR3Kinematics(CALIBRATED).flange(np.radians(PACKET_Q)))
    assert np.allclose(pose[:3], PACKET_POSE[:3], atol=0.01)
    rot = pose_to_matrix([0, 0, 0, *pose[3:]])[:3, :3]
    ref = pose_to_matrix([0, 0, 0, *PACKET_POSE[3:]])[:3, :3]
    assert np.abs(rot - ref).max() < 1e-4


def test_nominal_parameters_are_off_by_about_a_millimetre():
    kin = FR3Kinematics()
    errors = [np.linalg.norm(kin.flange(np.radians(q))[:3, 3] * 1000.0 - p) for q, p in CONTROLLER_FK]
    assert 0.5 < max(errors) < 3.0


def test_rpy_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        rpy = rng.uniform([-math.pi, -1.4, -math.pi], [math.pi, 1.4, math.pi])
        assert np.allclose(matrix_to_rpy(rpy_to_matrix(*rpy)), rpy, atol=1e-9)
    t = pose_to_matrix([100, -200, 300, 10, 20, 30])
    assert np.allclose(matrix_to_pose(t), [100, -200, 300, 10, 20, 30], atol=1e-9)


def test_inverse_all_round_trip():
    kin = FR3Kinematics(CALIBRATED)
    rng = np.random.default_rng(0)
    found = 0
    for _ in range(200):
        q = random_q(rng, kin)
        target = kin.tcp(q)
        solutions = kin.inverse_all(target, q_ref=q)
        for qs, _ in solutions:
            assert np.allclose(kin.tcp(qs), target, atol=1e-7)
            assert kin.in_limits(qs)
        found += any(np.abs(qs - q).max() < 1e-6 for qs, _ in solutions)
    assert found == 200


def test_inverse_with_tool_and_branch():
    params = FR3Params(tool_xyz=(0.01, -0.005, 0.2), tool_rpy_deg=(0.0, 0.0, 30.0))
    kin = FR3Kinematics(params)
    pose = np.eye(4)
    pose[:3, :3] = top_down_rotation(math.radians(20.0))
    pose[:3, 3] = (0.05, -0.45, 0.15)
    solutions = kin.inverse_all(pose)
    assert len(solutions) >= 2
    for q, branch in solutions:
        assert np.allclose(kin.tcp(q), pose, atol=1e-7)
        assert np.array_equal(kin.inverse(pose, branch=branch, q_ref=q), q)


def test_inverse_prefers_the_nearest_solution():
    kin = FR3Kinematics(CALIBRATED)
    q0 = np.radians([60.0, -70.0, 65.0, -85.0, -90.0, -30.0])
    q = kin.inverse(kin.tcp(q0), q_ref=q0 + 0.01)
    assert np.allclose(q, q0, atol=1e-7)


def test_unreachable_pose_returns_none():
    kin = FR3Kinematics()
    pose = np.eye(4)
    pose[:3, :3] = top_down_rotation()
    pose[:3, 3] = (0.0, -1.5, 0.1)
    assert kin.inverse(pose) is None
    assert kin.inverse_all(pose) == []


def test_controller_config_round_trip():
    seen = set()
    for config in range(8):
        branch = branch_of_config(config)
        assert controller_config(branch) == config
        seen.add(branch)
    assert len(seen) == 8


def test_wrap_near_maps_solutions_into_the_limits():
    kin = FR3Kinematics()
    wrapped = kin.wrap_near(np.radians([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]))
    assert math.isclose(math.degrees(wrapped[1]), -260.0)
    assert kin.wrap_near(np.radians([0, 0, 0, 0, 0, 179.0])) is None  # j6 limit is ±175 / j6 리밋
    assert kin.wrap_near(np.radians([0, 0, 0, 0, 0, 170.0]), margin_deg=10.0) is None
    assert kin.wrap_near(np.radians([0, 0, 0, 0, 0, 170.0])) is not None


def test_jacobian_matches_finite_differences():
    kin = FR3Kinematics(FR3Params(tool_xyz=(0.0, 0.0, 0.2)))
    q = np.radians([20.0, -80.0, 60.0, -70.0, -90.0, 10.0])
    jac = kin.jacobian(q)
    h = 1e-7
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = h
        num = (kin.tcp(q + dq)[:3, 3] - kin.tcp(q)[:3, 3]) / h
        assert np.allclose(jac[:3, i], num, atol=1e-5)
    assert kin.manipulability(q) > 0.01
    assert kin.manipulability(np.radians([0, -90, 0, -90, 0, 0])) < 1e-6  # home is singular / 홈은 특이점


def test_fit_dh_recovers_synthetic_parameters():
    rng = np.random.default_rng(3)
    truth = FR3Kinematics(CALIBRATED)
    joints = np.array([np.degrees(random_q(rng, truth)) for _ in range(40)])
    positions = np.array([truth.flange(np.radians(q))[:3, 3] * 1000.0 for q in joints])
    positions += rng.normal(0.0, 0.005, positions.shape)
    fit = fit_dh(joints, positions)
    assert fit.max_before_mm > 0.5
    assert fit.rms_after_mm < 0.02
    assert np.allclose(fit.params.dh, CALIBRATED.dh, atol=2e-5)


def test_fit_dh_with_tool_offset_and_rotation():
    rng = np.random.default_rng(4)
    truth_params = FR3Params(**{**CALIBRATED.to_mapping(), "tool_xyz": (0.004, -0.003, 0.15)})
    truth = FR3Kinematics(FR3Params.from_mapping(truth_params.to_mapping()))
    joints = np.array([np.degrees(random_q(rng, truth)) for _ in range(30)])
    poses = np.array([matrix_to_pose(truth.tcp(np.radians(q))) for q in joints])
    start = FR3Params(tool_xyz=(0.0, 0.0, 0.15))
    fit = fit_dh(joints, poses, start, fit_tool=True, rotation_weight_mm=100.0)
    assert fit.rms_after_mm < 1e-4
    assert np.allclose(fit.params.tool_xyz, (0.004, -0.003, 0.15), atol=1e-6)
    assert "tool_x" in fit.fitted


def test_params_json_round_trip():
    params = FR3Params(tool_xyz=(0.0, 0.0, 0.2), tool_rpy_deg=(0.0, 0.0, 90.0))
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/p.json"
        params.to_json(path, note="test")
        assert FR3Params.from_json(path) == params


def test_bad_fit_input_shapes():
    with pytest.raises(ValueError):
        fit_dh(np.zeros((3, 5)), np.zeros((3, 3)))
