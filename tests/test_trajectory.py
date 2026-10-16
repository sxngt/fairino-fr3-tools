"""Trajectory validation and execution through the servo stream (no ROS). / 궤적 검증·실행."""

import math
import threading

import numpy as np
import pytest

from fr3tools.servo import (
    FakeArm,
    ServoParams,
    ServoStream,
    TrajectoryError,
    TrajectoryExecutor,
    resample,
    validate_trajectory,
)

JOINTS = ("j1", "j2", "j3", "j4", "j5", "j6")


def params():
    return ServoParams(cmd_t=0.004, max_joint_speed_deg=200.0, max_joint_accel_deg=4000.0, servo_timeout=0.5)


def ramp(deg=3.0, seconds=0.2, n=5):
    t = np.linspace(0.0, seconds, n)
    q = np.zeros((n, 6))
    q[:, 0] = np.radians(np.linspace(0.0, deg, n))
    return t, q


def test_validate_accepts_and_reorders_joints():
    t, q = ramp()
    names = ["j2", "j1", "j3", "j4", "j5", "j6"]
    swapped = q[:, [1, 0, 2, 3, 4, 5]]
    t2, q2 = validate_trajectory(names, t, swapped, np.zeros(6), JOINTS, max_joint_speed_deg=100.0)
    assert np.allclose(q2, q) and np.allclose(t2, t)


@pytest.mark.parametrize(
    "case, code, match",
    [
        ("names", "invalid_joints", "joint names"),
        ("time", "invalid_goal", "does not increase"),
        ("start", "invalid_goal", "from the current joints"),
        ("speed", "invalid_goal", "peak joint speed"),
        ("empty", "invalid_goal", "empty"),
    ],
)
def test_validate_rejects(case, code, match):
    t, q = ramp(deg=3.0, seconds=0.2)
    names, q_now, limit = list(JOINTS), np.zeros(6), 100.0
    if case == "names":
        names[5] = "joint6"
    elif case == "time":
        t[2] = t[1]
    elif case == "start":
        q_now = np.radians([5.0, 0, 0, 0, 0, 0])
    elif case == "speed":
        limit = 10.0
    elif case == "empty":
        t, q = np.zeros(0), np.zeros((0, 6))
    with pytest.raises(TrajectoryError, match=match) as exc:
        validate_trajectory(names, t, q, q_now, JOINTS, max_joint_speed_deg=limit)
    assert exc.value.code == code


def test_validate_prepends_the_current_pose():
    t, q = validate_trajectory(JOINTS, [1.0], [[0.01, 0, 0, 0, 0, 0]], np.zeros(6), JOINTS, 30.0)
    assert np.allclose(t, [0.0, 1.0]) and np.allclose(q[0], 0.0)


def test_resample_hits_the_end_point():
    ts, qs = resample([0.0, 0.1], np.array([[0.0] * 6, [1.0] * 6]), 0.016)
    assert ts[-1] == pytest.approx(0.1) and np.allclose(qs[-1], 1.0)
    assert np.all(np.diff(ts) <= 0.016 + 1e-12)


def make(arm):
    stream = ServoStream(arm, params, log=lambda s: None)

    def start():
        stream.start(arm.sample()[0])
        return True, "started"

    return stream, TrajectoryExecutor(stream, start, lambda: arm.sample()[0], stop_arm=arm.stop)


def test_executor_reaches_the_goal():
    arm = FakeArm([0.0] * 6, tau=0.005)
    stream, ex = make(arm)
    t, q = ramp(deg=3.0, seconds=0.2)
    out = ex.run(t, q, threading.Event(), goal_tol_deg=0.05)
    assert out.status == "succeeded" and out.final_dev_deg <= 0.05
    assert not stream.active
    assert math.degrees(arm.sample()[0][0]) == pytest.approx(3.0, abs=0.1)


def test_executor_cancel_stops_early():
    arm = FakeArm([0.0] * 6, tau=0.005)
    stream, ex = make(arm)
    t, q = ramp(deg=20.0, seconds=1.0, n=20)
    cancel = threading.Event()
    threading.Timer(0.15, cancel.set).start()
    out = ex.run(t, q, cancel)
    assert out.status == "canceled" and not stream.active
    assert math.degrees(arm.sample()[0][0]) < 15.0


def test_executor_goal_tolerance_when_the_arm_lags():
    arm = FakeArm([0.0] * 6, tau=5.0)  # barely follows / 거의 따라오지 않는 팔
    _, ex = make(arm)
    t, q = ramp(deg=3.0, seconds=0.1)
    out = ex.run(t, q, threading.Event(), goal_tol_deg=0.05, settle_s=0.1)
    assert out.status == "aborted" and out.code == "goal_tolerance"


def test_executor_reports_a_refused_start():
    arm = FakeArm([0.0] * 6)
    stream = ServoStream(arm, params, log=lambda s: None)
    ex = TrajectoryExecutor(stream, lambda: (False, "not ready: MANUAL"), lambda: np.zeros(6))
    out = ex.run(*ramp(), threading.Event())
    assert out.status == "aborted" and "MANUAL" in out.message


def test_executor_aborts_on_a_stream_failure():
    class Failing(FakeArm):
        def servo_command(self, q, cmd_t):
            raise RuntimeError("ServoJ error 29")

    arm = Failing([0.0] * 6)
    _, ex = make(arm)
    out = ex.run(*ramp(seconds=0.5, n=10), threading.Event())
    assert out.status == "aborted" and out.code == "path_tolerance" and "29" in out.message
