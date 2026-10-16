"""Servo limiter maths and the fake servo stream (no ROS, no SDK). / 서보 제한기와 가짜 스트림."""

import math
import threading
import time

import numpy as np
import pytest

from fr3tools.servo import FakeArm, ServoParams, ServoStream, servo_step


def run_limiter(q0, target, params, ticks):
    q = np.asarray(q0, dtype=float)
    v = np.zeros_like(q)
    history = [q.copy()]
    steps = []
    for _ in range(ticks):
        q, v_new = servo_step(q, v, target, params.max_step, params.max_dv)
        steps.append((v_new - v).copy())
        v = v_new
        history.append(q.copy())
    return np.array(history), np.array(steps)


def test_per_tick_limits_follow_the_parameters():
    p = ServoParams(cmd_t=0.016, max_joint_speed_deg=30.0, max_joint_accel_deg=60.0)
    assert p.max_step == pytest.approx(math.radians(30.0) * 0.016)
    assert p.max_dv == pytest.approx(math.radians(60.0) * 0.016**2)


def test_speed_and_acceleration_are_bounded():
    p = ServoParams()
    target = np.radians([10.0, -5.0, 0.0, 2.0, 0.0, -20.0])
    history, dv = run_limiter(np.zeros(6), target, p, 2000)
    step = np.abs(np.diff(history, axis=0))
    assert step.max() <= p.max_step + 1e-12
    assert np.abs(dv).max() <= p.max_dv + 1e-12
    # It reaches the cruise speed on the long move. / 긴 이동에서는 순항 속도에 도달.
    assert step[:, 5].max() == pytest.approx(p.max_step)


def test_converges_without_meaningful_overshoot():
    p = ServoParams()
    target = np.radians([10.0, -5.0, 0.0, 0.3, 0.0, -20.0])
    history, _ = run_limiter(np.zeros(6), target, p, 3000)
    assert np.allclose(history[-1], target, atol=1e-6)
    # Braking keeps any overshoot below one acceleration quantum. / 제동으로 과도 이동 < dv.
    over = np.maximum(0.0, (history - target) * np.sign(target))
    over[:, target == 0] = 0.0
    assert over.max() <= p.max_dv


@pytest.mark.parametrize("distance_deg", [0.3, 5.0, 20.0, 90.0])
def test_move_sizes_settle_without_overshoot(distance_deg):
    p = ServoParams()
    target = np.full(6, math.radians(distance_deg))
    history, _ = run_limiter(np.zeros(6), target, p, 6000)
    assert np.allclose(history[-1], target, atol=1e-9)
    assert (history - target).max() <= p.max_dv


def test_smooth_start_from_rest():
    p = ServoParams()
    history, _ = run_limiter(np.zeros(6), np.radians([5.0] * 6), p, 5)
    first = history[1] - history[0]
    assert np.allclose(first, p.max_dv)


def test_holds_still_at_target():
    p = ServoParams()
    q = np.radians([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    q_new, v_new = servo_step(q, np.zeros(6), q, p.max_step, p.max_dv)
    assert np.array_equal(q_new, q)
    assert np.array_equal(v_new, np.zeros(6))


def test_fake_arm_follows_command():
    arm = FakeArm([0.0] * 6, tau=0.01)
    arm.set_cmd(np.radians([1.0, 0, 0, 0, 0, 0]))
    time.sleep(0.05)
    q, _, _, fresh = arm.sample()
    assert fresh
    assert q[0] == pytest.approx(math.radians(1.0), abs=1e-6)


def fast_params(**kw):
    base = dict(cmd_t=0.004, max_joint_speed_deg=200.0, max_joint_accel_deg=4000.0, servo_timeout=0.3)
    base.update(kw)
    return ServoParams(**base)


def wait_until(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_stream_moves_fake_arm_to_target_and_stops_on_request():
    arm = FakeArm([0.0] * 6, tau=0.01)
    stream = ServoStream(arm, lambda: fast_params(), log=lambda s: None)
    stream.start(np.zeros(6))
    target = np.radians([3.0, -2.0, 0.0, 0.0, 1.0, 0.0])
    stop_feeding = threading.Event()

    def feed():
        while not stop_feeding.is_set():
            stream.set_target(target)
            time.sleep(0.02)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        assert wait_until(lambda: np.allclose(arm.sample()[0], target, atol=1e-4))
        assert stream.active
    finally:
        stop_feeding.set()
        feeder.join()
    assert stream.stop() == ""
    assert not stream.active
    assert stream.sent > 0


def test_stream_ends_on_stale_target():
    arm = FakeArm([0.0] * 6)
    stream = ServoStream(arm, lambda: fast_params(servo_timeout=0.1), log=lambda s: None)
    stream.start(np.zeros(6))
    assert wait_until(lambda: not stream.active)
    assert "stale" in stream.reason


def test_stream_ends_on_target_jump():
    arm = FakeArm([0.0] * 6)
    stream = ServoStream(arm, lambda: fast_params(servo_max_jump_deg=5.0), log=lambda s: None)
    stream.start(np.zeros(6))
    stream.set_target(np.radians([10.0, 0, 0, 0, 0, 0]))
    assert wait_until(lambda: not stream.active)
    assert "jump" in stream.reason
    # The arm did not follow the jump. / 팔은 급변 목표를 따르지 않았다.
    assert abs(arm.sample()[0][0]) < math.radians(1.0)


def test_only_one_stream_at_a_time():
    arm = FakeArm([0.0] * 6)
    stream = ServoStream(arm, lambda: fast_params(servo_timeout=1.0), log=lambda s: None)
    stream.start(np.zeros(6))
    with pytest.raises(RuntimeError):
        stream.start(np.zeros(6))
    stream.stop()


class TrippingArm(FakeArm):
    """Fake arm whose monitor trips once the stream runs. / 스트림 중 트립하는 가짜 팔."""

    def monitor(self, dev_deg, on_trip, log):
        arm = self

        class Monitor:
            tripped = None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def set_target(self, q_deg):
                if not self.tripped:
                    self.tripped = "test trip"
                    on_trip(self.tripped)

        arm.ended = None
        return Monitor()

    def servo_end(self, failed):
        self.ended = failed


def test_safety_trip_ends_the_stream_as_a_failure():
    arm = TrippingArm([0.0] * 6)
    stream = ServoStream(arm, lambda: fast_params(), log=lambda s: None)
    stream.start(np.zeros(6))
    assert wait_until(lambda: not stream.active)
    assert stream.reason == "safety trip: test trip"
    assert arm.ended is True
