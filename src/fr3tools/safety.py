"""Safety monitor for streamed FR3 motion and a preflight check. / 스트리밍 모션 안전 감시와 사전 점검.

A background thread reads the controller state (20 Hz) and trips on:

* controller flags: emergency stop, collision, safety stops SI0/SI1, non-zero error code;
* tracking deviation: ``max |target - measured| > dev_deg`` while a target is set (the arm was
  pushed, blocked, or is not following);
* torque jumps: ``|tau - median(last tor_win s)| > tor_abs`` per joint for ``tor_hold`` samples
  in a row (impacts; gravity torque changes slowly and is not flagged).

A trip calls ``on_trip`` (e.g. end the servo stream), then ``stop`` (``StopMotion``). It latches:
a tripped monitor never re-arms, a person has to look first.

백그라운드 스레드가 제어기 상태를 20 Hz 로 읽고 제어기 플래그(비상정지·충돌·안전정지·에러),
추종 편차, 토크 급변 중 하나라도 걸리면 ``on_trip`` → ``stop`` 순으로 멈춘다. 한 번 걸리면 다시
켜지지 않는다(사람이 확인 후 재실행).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from fr3tools.kinematics import JOINTS

DEFAULT_TOR_ABS = (8.0, 8.0, 6.0, 3.0, 3.0, 3.0)  # Nm per joint / 관절별 허용 토크 변화


class SafetyMonitor:
    """Latching safety monitor. / 래치형 안전 감시.

    Parameters
    ----------
    read_status
        Returns a :class:`fr3tools.sdk.RobotStatus`-like object (``ok``, ``error``,
        ``jnt_ang`` deg, ``jnt_tor`` Nm, ``safety_ok``, ``has_error`` and the flag fields).
    stop
        Called on a trip after ``on_trip`` (``StopMotion``); ``None`` to skip.
    dev_deg
        Tracking deviation limit (deg).
    tor_abs
        Torque jump limits (Nm per joint); ``None`` uses the defaults.
    """

    def __init__(
        self,
        read_status: Callable[[], object],
        stop: Callable[[], object] | None = None,
        rate: float = 20.0,
        dev_deg: float = 2.0,
        tor_abs=None,
        tor_win: float = 0.25,
        tor_hold: int = 3,
        warmup: float = 1.0,
        on_trip: Callable[[str], None] | None = None,
        log: Callable[[str], None] = print,
        trip_on_read_error: bool = False,
    ):
        self.read_status = read_status
        self.stop = stop
        self.rate = rate
        self.dev_deg = dev_deg
        self.tor_abs = np.asarray(tor_abs if tor_abs is not None else DEFAULT_TOR_ABS, dtype=float)
        self.tor_win = tor_win
        self.tor_hold = tor_hold
        self.warmup = warmup
        self.on_trip = on_trip
        self.log = log
        self.trip_on_read_error = trip_on_read_error

        self.tripped: str | None = None
        self.fault: str | None = None
        self.max_dev = 0.0
        self.max_tor_delta = np.zeros(6)
        self.samples: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._target: np.ndarray | None = None
        self._target_lock = threading.Lock()
        self._over = 0
        self._t0 = time.time()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def set_target(self, q_deg) -> None:
        """Commanded joints (deg) to compare against. / 비교할 명령 관절(deg)."""
        with self._target_lock:
            self._target = np.asarray(q_deg, dtype=float)

    # ---- one sample (pure apart from bookkeeping) / 표본 하나 검사
    def check(self, status, now: float | None = None) -> str | None:
        """Check one status sample; return the trip reason or ``None``. / 표본 검사, 사유 반환."""
        now = time.time() if now is None else now
        if not status.safety_ok or status.has_error:
            return (
                f"controller flags estop={status.estop} collision={status.collision} "
                f"safety={status.safety_stop0}/{status.safety_stop1} "
                f"error={status.main_code}/{status.sub_code}"
            )
        angles = np.asarray(status.jnt_ang, dtype=float)
        torques = np.asarray(status.jnt_tor, dtype=float)
        self.samples.append((now, angles, torques))
        del self.samples[:-400]

        with self._target_lock:
            target = self._target
        if target is not None:
            err = np.abs(target - angles)
            dev = float(err.max())
            self.max_dev = max(self.max_dev, dev)
            if dev > self.dev_deg:
                return (
                    f"tracking deviation {dev:.2f} deg > {self.dev_deg} (joint {JOINTS[int(err.argmax())]})"
                )

        if now - self._t0 < self.warmup:
            return None
        recent = [tau for (t, _, tau) in self.samples[-200:] if now - self.tor_win <= t < now - 0.02]
        if len(recent) < 3:
            return None
        delta = np.abs(torques - np.median(np.array(recent), axis=0))
        self.max_tor_delta = np.maximum(self.max_tor_delta, delta)
        if not (delta > self.tor_abs).any():
            self._over = 0
            return None
        self._over += 1
        if self._over < self.tor_hold:
            return None
        return (
            f"torque jump {np.round(delta, 2).tolist()} Nm > {self.tor_abs.tolist()} "
            f"(joint {JOINTS[int(delta.argmax())]})"
        )

    def trip(self, why: str) -> None:
        """Latch, call ``on_trip``, then ``stop``. / 래치 후 on_trip, stop 호출."""
        if self.tripped:
            return
        self.tripped = why
        self.log(f"[safety] TRIP: {why}")
        if self.on_trip is not None:
            try:
                self.on_trip(why)
            except Exception as exc:
                self.log(f"[safety] on_trip failed: {exc}")
        if self.stop is not None:
            try:
                self.stop()
                self.log("[safety] StopMotion sent")
            except Exception as exc:
                self.log(f"[safety] StopMotion failed: {exc}")

    # ---- thread / 스레드
    def _run(self) -> None:
        self._t0 = time.time()
        while not self._halt.is_set():
            try:
                status = self.read_status()
                if not status.ok:
                    raise RuntimeError(status.error)
                why = self.check(status)
                if why is not None:
                    self.trip(why)
            except Exception as exc:
                self.fault = self.fault or f"read error: {exc}"
                if self.trip_on_read_error:
                    self.trip(self.fault)
                    break
            self._halt.wait(1.0 / self.rate)

    def __enter__(self) -> SafetyMonitor:
        self._halt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._halt.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def summary(self) -> str:
        text = (
            f"samples {len(self.samples)}, max deviation {self.max_dev:.2f} deg, "
            f"max torque change {np.round(self.max_tor_delta, 2).tolist()} Nm"
        )
        if self.tripped:
            text += f", TRIP {self.tripped}"
        if self.fault:
            text += f", FAULT {self.fault}"
        return text


# ----------------------------------------------------------------------------- preflight
@dataclass
class Report:
    """Preflight result: ``(level, name, detail)`` checks. / 사전 점검 결과."""

    checks: list = field(default_factory=list)  # level in PASS / WARN / FAIL / INFO

    def add(self, level: str, name: str, detail: str = "") -> None:
        self.checks.append((level, name, detail))

    @property
    def ready(self) -> bool:
        return not any(level == "FAIL" for level, _, _ in self.checks)

    def failures(self) -> list[str]:
        return [name for level, name, _ in self.checks if level == "FAIL"]

    def text(self) -> str:
        width = max((len(name) for _, name, _ in self.checks), default=10)
        lines = [f"[{level:4s}] {name:<{width}s}  {detail}" for level, name, detail in self.checks]
        lines.append(f"==> {'READY' if self.ready else 'NOT READY'}")
        return "\n".join(lines)


def preflight(status, expect_mode: str | None = "AUTO", versions: dict | None = None) -> Report:
    """Checks before streaming motion: ServoJ needs AUTO, no error, no safety stop, servo enabled.

    ``status`` is a :class:`fr3tools.sdk.RobotStatus`; ``versions`` the dict of
    :func:`fr3tools.sdk.versions` (a firmware other than the tested one is a warning).
    모션 전 점검: ServoJ 는 AUTO·에러 없음·안전정지 없음·서보 enable 이 필요하다.
    """
    from fr3tools.sdk import ROBOT_STATE, TESTED_FIRMWARE, firmware_of

    r = Report()
    if not status.ok:
        r.add(
            "FAIL",
            "controller",
            f"{status.ip}: {status.error} (power, cable, IP; driver holding the state port?)",
        )
        return r
    r.add("PASS", "controller", f"{status.ip} real-time state OK (frame {status.frame_cnt})")
    if versions is not None:
        firmware = firmware_of(versions)
        level = "PASS" if firmware == TESTED_FIRMWARE else "WARN"
        note = (
            "" if level == "PASS" else f" (tested with {TESTED_FIRMWARE}; use the SDK tag of your firmware)"
        )
        r.add(level, "versions", f"software {versions.get('software')} sdk {versions.get('sdk')}{note}")
    r.add(
        "FAIL" if status.estop else "PASS",
        "e-stop",
        "pressed: release it, then fr3-power recover" if status.estop else "released",
    )
    if status.safety_stop0 or status.safety_stop1:
        r.add("FAIL", "safety stop", f"SI0={status.safety_stop0} SI1={status.safety_stop1}")
    else:
        r.add("PASS", "safety stop", "SI0/SI1 clear")
    r.add(
        "FAIL" if status.collision else "PASS",
        "collision",
        "flag set: fr3-power reset" if status.collision else "none",
    )
    if status.has_error:
        r.add("FAIL", "error code", f"{status.error_text} (fr3-power reset)")
    else:
        r.add("PASS", "error code", "0/0")
    if status.encoders_all_zero:
        r.add("FAIL", "encoders", "all joints read 0: state packet broken")
    r.add(
        "PASS" if status.in_limits else "FAIL",
        "soft limits",
        "inside" if status.in_limits else f"outside {np.round(status.jnt_ang, 1).tolist()}",
    )
    r.add(
        "INFO",
        "mode/state",
        f"{status.mode} robot_state={ROBOT_STATE.get(status.robot_state, status.robot_state)}",
    )
    if expect_mode and status.mode != expect_mode:
        r.add("FAIL", "mode", f"{status.mode}, need {expect_mode}: fr3-power auto")
    if status.robot_state == 4:
        r.add("FAIL", "drag", "drag mode on: fr3-power drag off")
    if status.enabled == 0:
        r.add("FAIL", "servo enable", "disabled: fr3-power enable")
    elif status.enabled == 1:
        r.add("PASS", "servo enable", "enabled")
    hot = [f"{JOINTS[i]}={t:.1f}Nm" for i, t in enumerate(status.jnt_tor) if abs(t) > 20]
    if hot:
        r.add("WARN", "joint torque", ", ".join(hot))
    r.add("INFO", "joints (deg)", str(np.round(status.jnt_ang, 2).tolist()))
    if status.gripper_active:
        r.add(
            "INFO", "gripper", f"active, position {status.gripper_position} %, fault {status.gripper_fault}"
        )
    return r
