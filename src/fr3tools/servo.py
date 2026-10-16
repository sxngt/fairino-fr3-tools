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
