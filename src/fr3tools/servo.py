"""Rate-limited joint servo stream, independent of ROS. / ROS 와 무관한 속도 제한 관절 서보 스트림.

A client sends sparse joint targets (a vision loop may run at 10 Hz or slower); the stream turns them
into a smooth ServoJ stream (every ``cmd_t``, 16 ms on the FR3). Each tick the commanded joints
move towards the latest target with three limits:

* speed: at most ``max_joint_speed_deg * cmd_t`` per tick;
* acceleration: the per-tick step changes by at most ``max_joint_accel_deg * cmd_t²`` (a step
  change of speed shakes the base);
* braking: the step never exceeds the speed from which the arm can still stop within the
  remaining distance ``d`` when it slows by ``a`` per tick, ``v (v + a) / 2a <= d``, so it
  slows down smoothly into the target. (The continuous-time bound ``sqrt(2 a d)`` lets the
  discrete loop overshoot by up to ~0.2° at the default limits; this one stays below ``a``.)

The stream ends on request, when no target arrived for ``servo_timeout`` seconds, when a target
jumps more than ``servo_max_jump_deg`` from the command, on a safety trip, or on an SDK error.

클라이언트의 드문 관절 목표(10 Hz 이하일 수 있음)를 ``cmd_t`` (16 ms) 마다의
부드러운 ServoJ 스트림으로 바꾼다.
매 틱 속도·가속도·제동(틱당 a 씩 줄여 남은 거리 안에 멈출 수 있는 속도) 한계로 목표에 다가간다.
요청, 목표 무갱신(``servo_timeout``), 목표 급변(``servo_max_jump_deg``), 안전 트립, SDK 오류에서 끝난다.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ServoParams:
    """Servo stream parameters (the driver's ROS parameters). / 서보 스트림 파라미터."""

    cmd_t: float = 0.016  # ServoJ period (s) / ServoJ 주기
    max_joint_speed_deg: float = 30.0  # deg/s
    max_joint_accel_deg: float = 60.0  # deg/s²
    servo_timeout: float = 1.5  # s without a new target / 목표 무갱신 한계
    servo_max_jump_deg: float = 15.0  # target vs command / 목표-명령 차 한계
    start_tol_deg: float = 2.0  # safety monitor tracking deviation / 안전 감시 추종 편차

    @property
    def max_step(self) -> float:
        """Largest joint step per tick (rad). / 틱당 최대 관절 이동(rad)."""
        return math.radians(self.max_joint_speed_deg) * self.cmd_t

    @property
    def max_dv(self) -> float:
        """Largest change of the per-tick step (rad/tick). / 틱당 이동량의 최대 변화."""
        return math.radians(self.max_joint_accel_deg) * self.cmd_t * self.cmd_t

    @property
    def max_jump(self) -> float:
        return math.radians(self.servo_max_jump_deg)


def servo_step(
    q_cmd: np.ndarray, v_cmd: np.ndarray, target: np.ndarray, max_step: float, max_dv: float
) -> tuple[np.ndarray, np.ndarray]:
    """One tick of the limiter: new command and new per-tick step. / 한 틱: 새 명령과 틱당 이동량.

    All quantities are per joint, in radians (``v_cmd`` in rad per tick).
    모든 값은 관절별 rad (``v_cmd`` 는 rad/틱).
    """
    q_cmd = np.asarray(q_cmd, dtype=float)
    v_cmd = np.asarray(v_cmd, dtype=float)
    gap = np.asarray(target, dtype=float) - q_cmd

    # Speed limit, then the braking speed for the remaining distance: the largest v with
    # v + (v - a) + (v - 2a) + ... <= d, i.e. v (v + a) / 2a <= d.
    # 속도 한계, 그다음 남은 거리에서 틱당 a 씩 줄여 멈출 수 있는 최대 속도.
    want = np.clip(gap, -max_step, max_step)
    brake = 0.5 * (np.sqrt(max_dv * max_dv + 8.0 * max_dv * np.abs(gap)) - max_dv)
    want = np.clip(want, -brake, brake)

    # Acceleration limit on the change of the step. / 이동량 변화의 가속도 한계.
    v_new = np.clip(want, v_cmd - max_dv, v_cmd + max_dv)
    return q_cmd + v_new, v_new


class FakeArm:
    """Arm stand-in without an SDK: first-order lag towards the command. / SDK 없는 가짜 팔.

    Measured joints follow the last command with time constant ``tau``.
    측정 관절이 시정수 ``tau`` 로 마지막 명령을 따라간다.
    """

    fake = True

    def __init__(self, q0_deg, tau: float = 0.05):
        self.ip = "fake"
        self.q = np.radians(np.asarray(q0_deg, dtype=float))
        self.cmd = self.q.copy()
        self.qd = np.zeros(6)
        self.tau = tau
        self.lock = threading.Lock()
        self._t = time.monotonic()

    def set_cmd(self, q_rad) -> None:
        with self.lock:
            self.cmd = np.asarray(q_rad, dtype=float).copy()

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        with self.lock:
            now = time.monotonic()
            dt = max(1e-3, now - self._t)
            self._t = now
            q_new = self.q + (self.cmd - self.q) * min(1.0, dt / self.tau)
            self.qd = (q_new - self.q) / dt
            self.q = q_new
            return self.q.copy(), self.qd.copy(), np.zeros(6), True

    def ready_for_motion(self) -> tuple[bool, str]:
        return True, "fake"

    def stop(self) -> int:
        with self.lock:
            self.cmd = self.q.copy()
        return 0

    def servo_begin(self) -> None:
        pass

    def servo_command(self, q_rad: np.ndarray, cmd_t: float) -> None:
        self.set_cmd(q_rad)

    def servo_end(self, failed: bool) -> None:
        pass

    def monitor(self, dev_deg: float, on_trip, log):
        return None


class ServoStream:
    """One servo session at a time over an arm backend. / 팔 백엔드 위의 서보 세션(한 번에 하나).

    The arm provides ``servo_begin()``, ``servo_command(q_rad, cmd_t)``, ``servo_end(failed)``
    and ``monitor(dev_deg, on_trip, log)`` (a context manager with ``set_target`` and
    ``tripped``, or ``None``). ``params`` is read once per session.
    """

    def __init__(
        self,
        arm,
        params: Callable[[], ServoParams],
        log: Callable[[str], None] = print,
        warn: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.arm = arm
        self.params = params
        self.log = log
        self.warn = warn or log
        self.clock = clock
        self.desired: np.ndarray | None = None
        self.reason = ""
        self.sent = 0
        self._target: np.ndarray | None = None
        self._target_t = 0.0
        self._target_lock = threading.Lock()
        self._halt = threading.Event()
        self._halt.set()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_target(self, q_rad) -> None:
        """Latest joint target (rad). / 최신 관절 목표(rad)."""
        with self._target_lock:
            self._target = np.asarray(q_rad, dtype=float).copy()
            self._target_t = self.clock()

    def start(self, q0: np.ndarray) -> None:
        """Start streaming from the measured joints ``q0`` (rad). / 측정 관절에서 스트림 시작."""
        if self.active:
            raise RuntimeError("servo stream already running")
        q0 = np.asarray(q0, dtype=float).copy()
        self.set_target(q0)
        self.reason = ""
        self.sent = 0
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(q0, self._halt), daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> str:
        """Request the end and wait for it; returns the end reason. / 종료 요청 후 사유 반환."""
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout)
        return self.reason

    def _loop(self, q_cmd: np.ndarray, halt: threading.Event) -> None:
        p = self.params()
        max_step, max_dv, max_jump = p.max_step, p.max_dv, p.max_jump
        v_cmd = np.zeros_like(q_cmd)
        reason = ""
        t0 = time.perf_counter()
        monitor = self.arm.monitor(p.start_tol_deg, lambda why: halt.set(), self.warn)
        began = False
        try:
            if monitor is not None:
                monitor.__enter__()
            self.arm.servo_begin()
            began = True
            ticks = 0
            while not halt.is_set():
                with self._target_lock:
                    target = self._target
                    target_t = self._target_t
                if self.clock() - target_t > p.servo_timeout:
                    reason = f"servo target stale > {p.servo_timeout}s"
                    break
                if float(np.abs(target - q_cmd).max()) > max_jump:
                    reason = f"servo target jump > {p.servo_max_jump_deg:.0f} deg"
                    break

                q_cmd, v_cmd = servo_step(q_cmd, v_cmd, target, max_step, max_dv)
                self.desired = q_cmd
                if monitor is not None:
                    monitor.set_target(np.degrees(q_cmd))
                self.arm.servo_command(q_cmd, p.cmd_t)
                self.sent += 1
                ticks += 1

                lag = t0 + ticks * p.cmd_t - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
            if monitor is not None and monitor.tripped:
                reason = f"safety trip: {monitor.tripped}"
        except Exception as exc:
            reason = f"servo exception {type(exc).__name__}: {exc}"
        finally:
            if began:
                self.arm.servo_end(failed=bool(reason))
            elif reason:
                self.arm.stop()
            if monitor is not None:
                monitor.__exit__(None, None, None)
            self.reason = reason
            self.desired = None
            halt.set()
            message = (
                f"servo stream end: {reason or 'ok'} (sent {self.sent}, {time.perf_counter() - t0:.1f}s)"
            )
            if reason:
                self.warn(message)
            else:
                self.log(message)


# ----------------------------------------------------------------------------- joint trajectories
class TrajectoryError(ValueError):
    """A trajectory that must not be executed. / 실행하면 안 되는 궤적.

    ``code`` is ``"invalid_joints"`` or ``"invalid_goal"`` (the FollowJointTrajectory result codes).
    """

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def validate_trajectory(
    names,
    times,
    positions,
    q_now,
    joint_names=("j1", "j2", "j3", "j4", "j5", "j6"),
    max_joint_speed_deg: float = 30.0,
    start_tol_deg: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Check a joint trajectory and return ``(t, Q)`` in ``joint_names`` order (s, rad).

    Rejected: other joint names, an empty trajectory, times that do not increase, a first point
    more than ``start_tol_deg`` from the measured joints, or a segment faster than
    ``max_joint_speed_deg``. A first point after ``t = 0`` gets the measured joints prepended at
    ``t = 0``, so the approach to it is checked and streamed too.

    관절 궤적 검증: 관절 이름, 빈 궤적, 시간 단조 증가, 시작점-현재 자세 편차(``start_tol_deg``),
    구간 최고 관절 속도(``max_joint_speed_deg``). 첫 점이 t > 0 이면 현재 자세를 t = 0 에 넣는다.
    """
    names = list(names)
    if sorted(names) != sorted(joint_names):
        raise TrajectoryError("invalid_joints", f"joint names {names} != {list(joint_names)}")
    if len(times) == 0:
        raise TrajectoryError("invalid_goal", "empty trajectory")
    order = [names.index(j) for j in joint_names]
    t = np.asarray(times, dtype=float)
    q = np.asarray(positions, dtype=float)[:, order]
    q_now = np.asarray(q_now, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(q)):
        raise TrajectoryError("invalid_goal", "non-finite time or position")
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise TrajectoryError("invalid_goal", "time_from_start does not increase")
    dev = math.degrees(float(np.abs(q[0] - q_now).max()))
    if dev > start_tol_deg:
        raise TrajectoryError(
            "invalid_goal", f"start is {dev:.2f} deg from the current joints (> {start_tol_deg})"
        )
    if t[0] > 1e-6:
        t = np.concatenate([[0.0], t])
        q = np.vstack([q_now, q])
    if len(t) > 1:
        speed = np.abs(np.diff(q, axis=0)).max(axis=1) / np.diff(t)
        peak = math.degrees(float(speed.max()))
        if peak > max_joint_speed_deg:
            raise TrajectoryError(
                "invalid_goal", f"peak joint speed {peak:.1f} > {max_joint_speed_deg} deg/s"
            )
    return t - t[0], q


def resample(t, q, cmd_t: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear resampling on a ``cmd_t`` grid that always ends at the last point. / 격자 재표본.

    마지막 웨이포인트가 격자 배수가 아니어도 종점을 반드시 포함한다.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    ts = np.arange(0.0, t[-1] + 1e-9, cmd_t)
    if ts[-1] < t[-1] - 1e-9:
        ts = np.append(ts, t[-1])
    return ts, np.stack([np.interp(ts, t, q[:, i]) for i in range(q.shape[1])], axis=1)


@dataclass
class TrajectoryResult:
    """Outcome of :meth:`TrajectoryExecutor.run`. / 궤적 실행 결과."""

    status: str  # succeeded / canceled / aborted
    code: str  # successful / path_tolerance / goal_tolerance / invalid_goal
    message: str = ""
    final_dev_deg: float = float("nan")
    samples: int = 0


class TrajectoryExecutor:
    """Run a validated trajectory through a :class:`ServoStream`. / 서보 스트림으로 궤적 실행.

    The trajectory is fed sample by sample as the stream's target, so the stream's speed and
    acceleration limits, jump check and safety monitor all stay in force. Nothing else may feed the
    stream meanwhile (the driver ignores ``/fr/servo_target`` while an action owns it).

    궤적을 표본마다 서보 스트림의 목표로 넣으므로 스트림의 속도·가속도 제한, 급변 검사, 안전 감시가
    그대로 적용된다. 그동안 다른 목표 입력은 무시된다.

    Parameters
    ----------
    stream
        The servo stream.
    start
        ``start() -> (ok, message)``: starts the stream from the measured joints after the readiness
        checks (``FR3Driver.start_stream`` in the driver).
    current_q
        Measured joints (rad) or ``None``.
    stop_arm
        Called on cancel after the stream ends (``StopMotion``).
    """

    def __init__(self, stream: ServoStream, start, current_q, stop_arm=None, sleep=time.sleep):
        self.stream = stream
        self.start = start
        self.current_q = current_q
        self.stop_arm = stop_arm
        self.sleep = sleep

    def run(
        self,
        t,
        q,
        cancel: threading.Event,
        goal_tol_deg: float = 1.0,
        settle_s: float = 1.0,
        on_sample=None,
    ) -> TrajectoryResult:
        p = self.stream.params()
        ok, message = self.start()
        if not ok:
            return TrajectoryResult("aborted", "invalid_goal", message)
        ts, qs = resample(t, q, p.cmd_t)
        t0 = time.perf_counter()
        try:
            for k, target in enumerate(qs):
                if cancel.is_set():
                    return self._cancel(k)
                if not self.stream.active:
                    return self._failed(k)
                self.stream.set_target(target)
                if on_sample is not None:
                    on_sample(target)
                lag = t0 + ts[k] + p.cmd_t - time.perf_counter()
                if lag > 0:
                    self.sleep(lag)
            # Hold the last point until the arm is there (the limiter brakes into it).
            # 마지막 점을 유지하며 도달을 기다린다(제한기가 제동하며 접근).
            dev = float("inf")
            t_end = time.perf_counter() + settle_s
            while time.perf_counter() < t_end:
                if cancel.is_set():
                    return self._cancel(len(qs))
                if not self.stream.active:
                    return self._failed(len(qs))
                self.stream.set_target(qs[-1])
                q_now = self.current_q()
                if q_now is not None:
                    dev = math.degrees(float(np.abs(np.asarray(q_now) - qs[-1]).max()))
                    if dev <= goal_tol_deg:
                        break
                self.sleep(p.cmd_t)
            reason = self.stream.stop()
            if reason:
                return TrajectoryResult("aborted", "path_tolerance", reason, dev, len(qs))
            if dev <= goal_tol_deg:
                return TrajectoryResult(
                    "succeeded", "successful", f"end deviation {dev:.3f} deg", dev, len(qs)
                )
            return TrajectoryResult(
                "aborted", "goal_tolerance", f"end deviation {dev:.2f} deg > {goal_tol_deg}", dev, len(qs)
            )
        finally:
            if self.stream.active:
                self.stream.stop()

    def _cancel(self, k: int) -> TrajectoryResult:
        self.stream.stop()
        if self.stop_arm is not None:
            self.stop_arm()
        return TrajectoryResult("canceled", "successful", "canceled", samples=k)

    def _failed(self, k: int) -> TrajectoryResult:
        reason = self.stream.reason or "servo stream ended"
        return TrajectoryResult("aborted", "path_tolerance", reason, samples=k)
