"""Teleoperation controller: pad state -> grip-point target -> joint target (pure, no I/O).

State: the grip point ``p`` (the TCP, base frame) and the wrist yaw ``ψ`` about the vertical. The
tool keeps the orientation it had at the start (top-down within ``start_tilt_tol_deg``), turned
about the vertical by ``ψ - ψ₀``; so the first target *is* the current pose and nothing moves
until a stick moves.

Each tick:

1. deadzone and response curve: ``v = curve(deadzone(a)) · v_max(preset)``; the left stick is
   treated radially (the deadzone acts on its magnitude, diagonals never exceed ``v_max``); the
   right stick gives z (vertical) and the yaw rate (horizontal);
2. a first-order filter and a Cartesian acceleration limit;
3. integration into the target pose;
4. clamping to a workspace box (task-frame axes, around the start) and an absolute z floor;
5. IK on the start branch, wrapped next to the previous solution, with a joint-limit margin and a
   per-tick jump check; on failure the target holds and the flag says why, it never jumps.

Safety: motion only while the deadman is held (releasing it holds the target); stale or
disconnected input holds, then asks the runner to stop the stream. The driver's own speed limits
and safety monitor still apply downstream.

원격 조작 제어기(입출력 없음): 패드 → 그립점 목표 → 관절 목표. 매 틱 데드존·응답 곡선 → 필터·가속도 제한
→ 적분 → 작업 상자·바닥 제한 → 시작 가지 IK(직전 해 옆, 리밋 여유, 급변 거부). 실패하면 유지하고 이유를
알린다. 데드맨을 놓으면 유지, 입력이 끊기면 유지 후 정지 요청.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fr3tools.gripper import MIN_WIDTH_M, STROKE_M
from fr3tools.kinematics import FR3Kinematics, rot_z, transform
from fr3tools.teleop.mapping import Inputs


# ----------------------------------------------------------------------------- shaping
def scaled_deadzone(a: float, deadzone: float = 0.12, expo: float = 2.0) -> float:
    """Axis value in [-1, 1] -> command in [-1, 1]: 0 at the deadzone edge, power curve after it.

    축 값 → 명령: 데드존 경계에서 0 부터 시작(점프 없음), 그 뒤 거듭제곱 곡선.
    """
    a = max(-1.0, min(1.0, float(a)))
    m = abs(a)
    if m <= deadzone:
        return 0.0
    return math.copysign(((m - deadzone) / (1.0 - deadzone)) ** expo, a)


def radial(x: float, y: float, deadzone: float = 0.12, expo: float = 2.0) -> tuple[float, float]:
    """Stick (x, y) shaped on its magnitude; the result has norm ≤ 1. / 스틱 크기에 대한 성형."""
    m = math.hypot(x, y)
    if m <= deadzone:
        return 0.0, 0.0
    s = scaled_deadzone(min(m, 1.0), deadzone, expo)
    return s * x / m, s * y / m


def trigger_width(t: float, open_width: float = STROKE_M, closed_width: float = MIN_WIDTH_M) -> float:
    """Trigger t in [0, 1] -> width: released = open, fully pressed = closed. / 트리거 → 폭."""
    t = min(1.0, max(0.0, float(t)))
    return open_width - t * (open_width - closed_width)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class SpeedPreset:
    name: str
    linear: float  # m/s
    yaw_deg: float  # deg/s


DEFAULT_PRESETS = (
    SpeedPreset("slow", 0.04, 25.0),
    SpeedPreset("normal", 0.08, 40.0),
    SpeedPreset("fast", 0.14, 60.0),
)


class GripperLimiter:
    """Rate limiter for gripper commands: the AG-95 is slow and the bus must not be flooded.

    Sends when the target moved by ``min_delta`` and at most ``max_rate_hz``; once the target has
    been still for ``settle_s`` the final value is sent even if the change is small.
    목표가 ``min_delta`` 이상 바뀌었고 ``max_rate_hz`` 이하일 때만 보낸다. 목표가 ``settle_s`` 동안
    멈추면 작은 차이라도 마지막 값을 보낸다.
    """

    def __init__(self, min_delta: float = 0.001, max_rate_hz: float = 8.0, settle_s: float = 0.25):
        self.min_delta, self.period, self.settle_s = min_delta, 1.0 / max_rate_hz, settle_s
        self.last_sent: float | None = None
        self.last_send_t = -math.inf
        self.target: float | None = None
        self.target_t = -math.inf

    def update(self, target: float | None, now: float) -> float | None:
        if target is None:
            return None
        if self.target is None or abs(target - self.target) > 1e-5:
            self.target, self.target_t = float(target), now
        diff = math.inf if self.last_sent is None else abs(self.target - self.last_sent)
        if diff < 1e-5 or now - self.last_send_t < self.period:
            return None
        if diff >= self.min_delta or now - self.target_t >= self.settle_s:
            self.last_sent, self.last_send_t = self.target, now
            return self.target
        return None


# ----------------------------------------------------------------------------- configuration
@dataclass
class TeleopConfig:
    """Tuning and safety limits; lengths in m. / 조정값과 안전 한계 (m)."""

    deadzone: float = 0.12
    expo: float = 1.5
    presets: tuple[SpeedPreset, ...] = DEFAULT_PRESETS
    preset: int = 1
    linear_accel: float = 0.4  # m/s²
    yaw_accel_deg: float = 120.0  # deg/s²
    filter_tau: float = 0.06  # s
    # Workspace box in task-frame axes around the start point, plus an absolute floor.
    # 시작점 주위의 작업 상자(작업 좌표축)와 절대 바닥.
    box_xy: float = 0.20
    box_down: float = 0.20
    box_up: float = 0.20
    box: tuple | None = None  # ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in the task frame / 작업 좌표
    table_z: float = 0.0
    min_height: float = 0.05  # grip point above the table / 테이블 위 최소 높이
    start_tilt_tol_deg: float = 10.0
    max_joint_step_deg: float = 3.0  # per tick / 틱당
    limit_margin_deg: float = 5.0
    stale_s: float = 0.3
    stop_after_s: float = 1.0
    home_gain: float = 1.5  # 1/s
    open_width: float = STROKE_M
    closed_width: float = MIN_WIDTH_M
    gripper_min_delta: float = 0.001
    # At most two gripper commands per second: each AG-95 move goes through the controller's RPC
    # and back-to-back commands starve the servo stream (seen as a tracking-deviation trip).
    # 초당 2 회 이하: 연속 그리퍼 명령이 서보 스트림을 굶겨 추종 편차 정지를 일으켰다.
    gripper_rate_hz: float = 2.0
    gripper_settle_s: float = 0.25
    trigger_rearm: float = 0.05  # trigger move that re-enables it after A/B / A·B 후 재활성화 이동량
    # RT is a two-state control with hysteresis: press past ``trigger_close`` to close once, release
    # below ``trigger_open`` to open once. A grasp is never re-sent, so the gripper holds the object at
    # its force limit instead of squeezing harder. / RT 는 히스테리시스가 있는 두 상태 입력: 깊게 누르면
    # 한 번 닫고, 떼면 한 번 연다. 잡은 뒤 다시 보내지 않아 더 조이지 않고 유지한다.
    trigger_close: float = 0.6
    trigger_open: float = 0.3
    # Off by default: the gripper is driven by the toggle button (LB). / 기본은 꺼짐: LB 토글로 조작.
    trigger_gripper: bool = False


@dataclass
class TickOutput:
    """What the runner should do this tick. / 이번 틱에 러너가 할 일."""

    q: np.ndarray | None = None  # joint target to stream (active session) / 보낼 관절 목표
    request: str | None = None  # "start" / "stop" a servo session / 세션 시작·정지 요청
    gripper_width: float | None = None  # width to send now (m) / 지금 보낼 폭
    flags: set = field(default_factory=set)


# ----------------------------------------------------------------------------- controller
class TeleopController:
    """Pure teleoperation controller. / 순수 원격 조작 제어기.

    Parameters
    ----------
    kin
        Kinematics with the tool (TCP = grip point).
    config
        :class:`TeleopConfig`.
    frame
        Task frame (4×4, level: only its yaw and origin are used) or ``"base"`` / ``"start"``.
    """

    def __init__(self, kin: FR3Kinematics, config: TeleopConfig | None = None, frame="base"):
        self.kin = kin
        self.cfg = config or TeleopConfig()
        self.frame_spec = frame
        self.preset = self.cfg.preset
        self.active = False
        self.limiter = GripperLimiter(
            self.cfg.gripper_min_delta, self.cfg.gripper_rate_hz, self.cfg.gripper_settle_s
        )
        self._prev_buttons: dict = {}
        self._prev_dpad = 0.0
        self._last_now: float | None = None
        self._stale_since: float | None = None
        self.p = np.zeros(3)
        self.psi = 0.0
        self.q_cmd: np.ndarray | None = None
        self.home: tuple[np.ndarray, float] | None = None
        self.gripper_target: float | None = None
        self.trigger_latch: float | None = None
        self.going_home = False
        self.last_flags: set = set()
        self.v = np.zeros(3)
        self.w = 0.0

    # ---- session / 세션
    def begin(self, q_measured, rt: float = 0.0) -> None:
        """Start from the measured joints: the first target is the current pose. / 현재 자세에서 시작."""
        q = np.asarray(q_measured, dtype=float).copy()
        tcp = self.kin.tcp(q)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(tcp[2, 2])))))
        if tilt > self.cfg.start_tilt_tol_deg:
            raise ValueError(f"tool is {tilt:.0f} deg from top-down; align it before teleoperation")
        from fr3tools.calib.poses import branch_of

        self.branch = branch_of(self.kin, q)
        self.q_cmd = q
        self.p = tcp[:3, 3].copy()
        self.r0 = tcp[:3, :3].copy()
        self.psi0 = math.atan2(tcp[1, 0], tcp[0, 0])
        self.psi = self.psi0
        self.v = np.zeros(3)
        self.w = 0.0
        if self.frame_spec == "start":
            self.frame = transform(rot_z(self.psi0), self.p)
        elif isinstance(self.frame_spec, str):
            self.frame = np.eye(4)
        else:
            self.frame = np.asarray(self.frame_spec, dtype=float)
        p_task = self.frame[:3, :3].T @ (self.p - self.frame[:3, 3])
        if self.cfg.box is not None:
            self.box = np.asarray(self.cfg.box, dtype=float)
        else:
            b = self.cfg
            self.box = np.array(
                [
                    [p_task[0] - b.box_xy, p_task[0] + b.box_xy],
                    [p_task[1] - b.box_xy, p_task[1] + b.box_xy],
                    [p_task[2] - b.box_down, p_task[2] + b.box_up],
                ]
            )
        self.home = self.home or (self.p.copy(), self.psi)
        self.trigger_latch = rt  # the trigger counts only after it moves / 트리거는 움직인 뒤부터
        self.gripper_target = None
        self.going_home = False
        self.active = True

    def end(self) -> None:
        self.active = False
        self.v = np.zeros(3)
        self.w = 0.0

    # ---- pose / 자세
    def pose(self, p=None, psi=None) -> np.ndarray:
        p = self.p if p is None else p
        psi = self.psi if psi is None else psi
        return transform(rot_z(psi - self.psi0) @ self.r0, p)

    def _clamp(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clamp to the box (task axes) and the floor; returns (p, clamped-axis mask in base). / 제한."""
        r, o = self.frame[:3, :3], self.frame[:3, 3]
        pt = r.T @ (p - o)
        clipped = np.clip(pt, self.box[:, 0], self.box[:, 1])
        mask_task = np.abs(clipped - pt) > 1e-12
        out = r @ clipped + o
        floor = self.cfg.table_z + self.cfg.min_height
        mask = np.abs(r @ mask_task.astype(float)) > 0.5
        if out[2] < floor:
            out[2] = floor
            mask[2] = True
        return out, mask

    def _edges(self, inputs: Inputs) -> set:
        down = {n for n, v in inputs.buttons.items() if v and not self._prev_buttons.get(n, False)}
        self._prev_buttons = dict(inputs.buttons)
        return down

    # ---- one tick / 한 틱
    def step(self, inputs: Inputs, now: float) -> TickOutput:
        out = TickOutput()
        dt = 0.0 if self._last_now is None else min(0.1, max(0.0, now - self._last_now))
        self._last_now = now
        roles = ROLES
        edges = self._edges(inputs)

        # Stale or disconnected input: hold, then ask for a stop. / 입력 끊김: 유지 후 정지 요청.
        if not inputs.connected or inputs.age > self.cfg.stale_s:
            self._stale_since = now if self._stale_since is None else self._stale_since
            out.flags.add("stale")
            self.v, self.w = np.zeros(3), 0.0
            if self.active and now - self._stale_since >= self.cfg.stop_after_s:
                out.request = "stop"
            out.q = self.q_cmd if self.active else None
            self.last_flags = out.flags
            return out
        self._stale_since = None

        if roles["start"] in edges and not self.active:
            out.request = "start"
        elif roles["stop"] in edges and self.active:
            out.request = "stop"
        dpad = inputs.dpad_y
        if dpad > 0.5 >= self._prev_dpad:
            self.preset = min(len(self.cfg.presets) - 1, self.preset + 1)
        elif dpad < -0.5 <= self._prev_dpad:
            self.preset = max(0, self.preset - 1)
        self._prev_dpad = dpad
        if roles["cycle_speed"] in edges:
            self.preset = (self.preset + 1) % len(self.cfg.presets)
        if not self.active:
            self.last_flags = out.flags
            return out

        if roles["save_home"] in edges:
            self.home = (self.p.copy(), self.psi)
            out.flags.add("home saved")
        self._gripper(inputs, edges, now, out)

        # Desired velocity. / 원하는 속도.
        preset = self.cfg.presets[self.preset]
        deadman = inputs.pressed(roles["deadman"])
        v_des, w_des = np.zeros(3), 0.0
        if not deadman:
            out.flags.add("deadman released")
            self.v, self.w = np.zeros(3), 0.0  # hold / 유지
            self.going_home = False
        else:
            sx, sy = radial(inputs.lx, inputs.ly, self.cfg.deadzone, self.cfg.expo)
            sz = scaled_deadzone(inputs.ry, self.cfg.deadzone, self.cfg.expo)
            sw = scaled_deadzone(inputs.rx, self.cfg.deadzone, self.cfg.expo)
            stick = abs(sx) + abs(sy) + abs(sz) + abs(sw) > 0.0
            if inputs.pressed(roles["home"]) and not stick:
                self.going_home = True
            if stick:
                self.going_home = False
            if self.going_home and self.home is not None:
                v_des = self.cfg.home_gain * (self.home[0] - self.p)
                n = np.linalg.norm(v_des)
                if n > preset.linear:
                    v_des *= preset.linear / n
                w_des = self.cfg.home_gain * wrap(self.home[1] - self.psi)
                w_des = max(-math.radians(preset.yaw_deg), min(math.radians(preset.yaw_deg), w_des))
                out.flags.add("going home")
            else:
                v_des = self.frame[:3, :3] @ np.array([sx, sy, sz]) * preset.linear
                # Stick right = clockwise seen from above. / 스틱 오른쪽 = 위에서 볼 때 시계 방향.
                w_des = -sw * math.radians(preset.yaw_deg)

        if deadman and dt > 0:
            # Filter, then limit the acceleration. / 필터 후 가속도 제한.
            alpha = dt / (self.cfg.filter_tau + dt)
            v_f = self.v + (v_des - self.v) * alpha
            w_f = self.w + (w_des - self.w) * alpha
            dv = v_f - self.v
            n = np.linalg.norm(dv)
            if n > self.cfg.linear_accel * dt:
                dv *= self.cfg.linear_accel * dt / n
            self.v = self.v + dv
            dw_max = math.radians(self.cfg.yaw_accel_deg) * dt
            self.w = self.w + max(-dw_max, min(dw_max, w_f - self.w))

        moving = np.linalg.norm(self.v) > 1e-9 or abs(self.w) > 1e-9
        if moving and dt > 0:
            p_new, mask = self._clamp(self.p + self.v * dt)
            if mask.any():
                out.flags.add("clamped")
                self.v[mask] = 0.0
            psi_new = self.psi + self.w * dt
            q = self.kin.inverse(self.pose(p_new, psi_new), branch=self.branch, q_ref=self.q_cmd)
            if q is not None:
                q = self.kin.wrap_near(q, self.q_cmd, margin_deg=self.cfg.limit_margin_deg)
            if q is None:
                out.flags.add("IK/limit: holding")
                self.v, self.w = np.zeros(3), 0.0
            elif np.degrees(np.abs(q - self.q_cmd)).max() > self.cfg.max_joint_step_deg:
                out.flags.add("joint jump: holding")
                self.v, self.w = np.zeros(3), 0.0
            else:
                self.p, self.psi, self.q_cmd = p_new, psi_new, q
        out.q = self.q_cmd
        self.last_flags = out.flags
        return out

    def _gripper(self, inputs: Inputs, edges: set, now: float, out: TickOutput) -> None:
        roles = ROLES
        rt = inputs.rt
        if roles["gripper_toggle"] in edges:
            # First press opens; every press after that flips open <-> closed. A close is sent once
            # and never repeated, so a grasped object is held at the force limit, not squeezed harder.
            # 첫 누름은 열기, 이후 누를 때마다 열기·닫기 전환. 닫기는 한 번만 보내 더 조이지 않는다.
            opening = self.gripper_target != self.cfg.open_width
            self.gripper_target = self.cfg.open_width if opening else self.cfg.closed_width
            self.trigger_latch = rt
        elif roles["close"] in edges:
            self.gripper_target, self.trigger_latch = self.cfg.closed_width, rt
        elif roles["open"] in edges:
            self.gripper_target, self.trigger_latch = self.cfg.open_width, rt
        elif self.trigger_latch is not None and abs(rt - self.trigger_latch) > self.cfg.trigger_rearm:
            self.trigger_latch = None
        if self.cfg.trigger_gripper and self.trigger_latch is None:
            if rt >= self.cfg.trigger_close and self.gripper_target != self.cfg.closed_width:
                self.gripper_target = self.cfg.closed_width
            elif rt <= self.cfg.trigger_open and self.gripper_target != self.cfg.open_width:
                self.gripper_target = self.cfg.open_width
        out.gripper_width = self.limiter.update(self.gripper_target, now)

    def status(self, rt: float = 0.0) -> str:
        """One-line status. / 한 줄 상태."""
        preset = self.cfg.presets[self.preset].name
        if not self.active:
            return f"IDLE  speed {preset}  (Start = begin, RB = deadman)"
        grip = "-" if self.gripper_target is None else f"{self.gripper_target * 1000:5.1f} mm"
        flags = ", ".join(sorted(self.last_flags)) or "ok"
        x, y, z = self.p * 1000.0
        return (
            f"ACTIVE x {x:7.1f} y {y:7.1f} z {z:6.1f} mm  yaw {math.degrees(self.psi):7.1f}  "
            f"speed {preset:6s} grip {grip} (t {rt:.2f})  [{flags}]"
        )


# Role names used by the controller (mapping.roles maps them to buttons). / 역할 이름.
ROLES = {
    "deadman": "deadman",
    "close": "close",
    "open": "open",
    "home": "home",
    "save_home": "save_home",
    "start": "start",
    "stop": "stop",
    "cycle_speed": "cycle_speed",
    "gripper_toggle": "gripper_toggle",
}


def role_inputs(inputs: Inputs, roles: dict, trigger: str = "rt") -> Inputs:
    """Copy of ``inputs`` whose buttons are keyed by role (via ``mapping.roles``) and whose ``rt`` is
    the gripper trigger. / 역할 기준 버튼과 그리퍼 트리거로 바꾼 사본.
    """
    out = Inputs(**{k: getattr(inputs, k) for k in ("lx", "ly", "rx", "ry", "lt", "rt", "dpad_x", "dpad_y")})
    out.rt = getattr(inputs, trigger)
    out.age, out.connected = inputs.age, inputs.connected
    out.buttons = {role: inputs.pressed(button) for role, button in roles.items()}
    return out
