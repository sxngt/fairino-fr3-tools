"""Gamepad teleoperation without a pad, ROS or robot. / 패드·ROS·로봇 없는 원격 조작 시험."""

import math
import struct

import numpy as np
import pytest

from fr3tools.kinematics import FR3Kinematics, FR3Params
from fr3tools.ros.adapter import FakeRobot
from fr3tools.teleop.controller import (
    GripperLimiter,
    TeleopConfig,
    TeleopController,
    radial,
    role_inputs,
    scaled_deadzone,
    trigger_width,
)
from fr3tools.teleop.gamepad import (
    JS_EVENT_AXIS,
    JS_EVENT_BUTTON,
    JS_EVENT_INIT,
    ScriptedPad,
    devices_from_proc,
    parse_js_events,
)
from fr3tools.teleop.mapping import Inputs, Mapping, load_profile, normalize, normalize_trigger
from fr3tools.teleop.run import DEMO_SCRIPT, start_joints

KIN = FR3Kinematics(FR3Params(tool_xyz=(0.0, 0.0, 0.2)))
DT = 0.02


# ----------------------------------------------------------------------------- device and mapping
def test_parse_js_events_from_bytes():
    data = struct.pack("<IhBB", 1000, -32767, JS_EVENT_AXIS | JS_EVENT_INIT, 5)
    data += struct.pack("<IhBB", 1016, 1, JS_EVENT_BUTTON, 5) + b"\x01\x02"  # partial tail / 잘린 꼬리
    ev = parse_js_events(data)
    assert len(ev) == 2
    assert ev[0].kind == "axis" and ev[0].is_init and ev[0].value == -32767 and ev[0].number == 5
    assert ev[1].kind == "button" and not ev[1].is_init and ev[1].value == 1 and ev[1].time_ms == 1016


def test_devices_from_proc_finds_the_pad():
    text = (
        'I: Bus=0003\nN: Name="Some Joystick"\nH: Handlers=event3 js0\n\n'
        'I: Bus=0003\nN: Name="Microsoft X-Box 360 pad"\nH: Handlers=event19 js1 \n'
    )
    assert devices_from_proc(text) == [
        ("Some Joystick", "/dev/input/js0"),
        ("Microsoft X-Box 360 pad", "/dev/input/js1"),
    ]


def test_trigger_normalisation_from_raw():
    assert normalize_trigger(-32767) == 0.0  # rest / 놓음
    assert normalize_trigger(32767) == 1.0
    assert normalize_trigger(0) == pytest.approx(0.5)
    assert normalize_trigger(-32767 + 500) == 0.0  # inside the rest deadband / 데드밴드
    inputs = normalize({0: 0, 1: -32767, 2: -32767, 5: 32767}, {5: 1}, Mapping())
    assert inputs.ly == 1.0  # stick up is negative raw / 스틱 위쪽은 원시 음수
    assert inputs.rt == 1.0 and inputs.lt == 0.0 and inputs.pressed("rb")
    assert normalize({}, {}, Mapping()).rt == 0.0  # untouched triggers read as rest / 초기값 = 놓음


def test_profile_remaps_buttons(tmp_path):
    path = tmp_path / "pad.json"
    path.write_text('{"name": "other", "buttons": {"rb": 7}, "roles": {"deadman": "lb"}}')
    m = load_profile(str(path))
    assert m.buttons["rb"] == 7 and m.buttons["a"] == 0 and m.roles["deadman"] == "lb"
    assert role_inputs(normalize({}, {4: 1}, m), m.roles).pressed("deadman")


# ----------------------------------------------------------------------------- shaping
def test_deadzone_and_curve_are_proportional_without_a_jump():
    assert scaled_deadzone(0.12) == 0.0
    assert scaled_deadzone(0.1201) < 1e-6  # continuous at the edge / 경계에서 연속
    assert scaled_deadzone(1.0) == pytest.approx(1.0) and scaled_deadzone(-1.0) == pytest.approx(-1.0)
    half, full = scaled_deadzone(0.5), scaled_deadzone(1.0)
    assert 0 < half < full and half < 0.5  # expo: fine near centre / 중앙 근처 정밀
    xs = np.linspace(0, 1, 101)
    assert np.all(np.diff([scaled_deadzone(x) for x in xs]) >= 0)


def test_radial_deadzone_caps_diagonals():
    assert radial(0.08, 0.08) == (0.0, 0.0)  # magnitude 0.113 < 0.12 / 크기 기준 데드존
    assert radial(0.1, 0.1)[0] > 0.0  # 0.141 > 0.12 although each axis is inside / 축별로는 안쪽
    vx, vy = radial(1.0, 1.0)
    assert math.hypot(vx, vy) == pytest.approx(1.0)
    vx, vy = radial(0.121, 0.0)
    assert vx < 1e-5
    assert math.hypot(*radial(0.7, 0.7)) > math.hypot(*radial(0.5, 0.0))


def test_trigger_width_endpoints_and_monotonic():
    assert trigger_width(0.0) == pytest.approx(0.095)
    assert trigger_width(1.0) == pytest.approx(0.010)
    w = [trigger_width(t) for t in np.linspace(0, 1, 50)]
    assert np.all(np.diff(w) < 0)


def test_gripper_limiter_deadband_rate_and_final_value():
    lim = GripperLimiter(min_delta=0.001, max_rate_hz=8.0, settle_s=0.25)
    assert lim.update(0.050, 0.0) == 0.050
    assert lim.update(0.0495, 0.2) is None  # below the deadband / 데드밴드 아래
    assert lim.update(0.040, 0.05) is None  # too soon / 너무 이름
    assert lim.update(0.040, 0.13) == 0.040
    sent = [lim.update(0.040 - 0.002 * k, 0.13 + 0.01 * k) for k in range(1, 30)]
    assert sum(v is not None for v in sent) <= 3  # ≤ 8 Hz over 0.29 s / 8 Hz 이하
    last = 0.040 - 0.002 * 29
    lim.update(last + 0.0004, 0.5)  # small final change / 작은 마지막 변화
    assert lim.update(last + 0.0004, 0.8) == pytest.approx(last + 0.0004)  # sent once settled / 정착 후 전송


# ----------------------------------------------------------------------------- controller
def pad(**kw) -> Inputs:
    buttons = kw.pop("buttons", {})
    return Inputs(buttons=buttons, **kw)


def started(cfg=None, frame="base"):
    ctl = TeleopController(KIN, cfg or TeleopConfig(), frame)
    q0 = start_joints(KIN)
    ctl.begin(q0)
    return ctl, q0


def run(ctl, inputs, seconds, t0=0.0):
    t = t0
    out = None
    for _ in range(int(round(seconds / DT))):
        t += DT
        out = ctl.step(inputs, t)
    return out, t


def test_first_target_is_the_current_pose_and_nothing_moves_without_input():
    ctl, q0 = started()
    out, _ = run(ctl, pad(buttons={"deadman": True}), 1.0)
    assert np.array_equal(out.q, q0)


def test_deadman_gates_motion_and_release_holds():
    ctl, q0 = started()
    out, t = run(ctl, pad(lx=1.0), 0.5)
    assert np.array_equal(out.q, q0) and "deadman released" in out.flags
    out, t = run(ctl, pad(lx=1.0, buttons={"deadman": True}), 1.0, t)
    p_moving = ctl.p.copy()
    assert p_moving[0] > 0.005
    out, t = run(ctl, pad(lx=1.0), 0.5, t)
    assert np.allclose(ctl.p, p_moving)  # held at once / 즉시 유지


def test_speed_is_proportional_and_acceleration_limited():
    speeds = {}
    for a in (0.5, 1.0):
        ctl, _ = started()
        run(ctl, pad(lx=a, buttons={"deadman": True}), 1.5)
        speeds[a] = np.linalg.norm(ctl.v)
    preset = TeleopConfig().presets[1].linear
    assert speeds[1.0] == pytest.approx(preset, rel=1e-3)
    assert 0 < speeds[0.5] < 0.5 * speeds[1.0]
    ctl, _ = started()
    run(ctl, pad(lx=1.0, buttons={"deadman": True}), DT * 3)
    assert np.linalg.norm(ctl.v) <= TeleopConfig().linear_accel * DT * 3 + 1e-12


def test_workspace_box_and_floor_clamp():
    cfg = TeleopConfig(box_xy=0.02, box_down=0.5, min_height=0.15, preset=2)
    ctl, _ = started(cfg)
    x0 = ctl.p[0]
    out, t = run(ctl, pad(lx=1.0, buttons={"deadman": True}), 2.0)
    assert ctl.p[0] == pytest.approx(x0 + 0.02) and "clamped" in out.flags
    out, _ = run(ctl, pad(ry=-1.0, buttons={"deadman": True}), 4.0, t)
    assert ctl.p[2] == pytest.approx(0.15)  # floor = table_z + min_height / 바닥


def test_task_frame_turns_the_stick_directions():
    ctl, _ = started(frame="start")  # start yaw is 180 deg / 시작 yaw 180°
    x0 = ctl.p[0]
    run(ctl, pad(lx=1.0, buttons={"deadman": True}), 1.0)
    assert ctl.p[0] < x0 - 0.005  # task +x = base -x / 작업 +x = 베이스 -x


def test_stale_input_holds_then_requests_a_stop():
    ctl, _ = started()
    out, t = run(ctl, pad(lx=1.0, buttons={"deadman": True}), 1.0)
    p = ctl.p.copy()
    out, t = run(ctl, pad(lx=1.0, buttons={"deadman": True}, age=0.5), 0.5, t)
    assert "stale" in out.flags and out.request is None and np.allclose(ctl.p, p)
    out, t = run(ctl, pad(lx=1.0, buttons={"deadman": True}, connected=False), 0.6, t)
    assert out.request == "stop"


def test_yaw_wraps_continuously_through_180_degrees():
    ctl, q0 = started()  # yaw 180 deg, j6 near 0 / yaw 180°
    prev = ctl.q_cmd.copy()
    jumps = []
    t = 0.0
    for _ in range(150):  # 3 s at -30 deg/s crosses ±180 / ±180° 통과
        t += DT
        out = ctl.step(pad(rx=1.0, buttons={"deadman": True}), t)
        jumps.append(np.degrees(np.abs(out.q - prev)).max())
        prev = out.q.copy()
    assert math.degrees(ctl.psi) < 180.0 - 30.0  # turned past 180 / 180° 를 지나 회전
    assert max(jumps) < 1.0  # continuous joints / 연속 관절
    assert abs(math.degrees(wrap_deg(ctl.psi))) <= 180.0


def wrap_deg(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def test_ik_failure_holds_instead_of_jumping():
    cfg = TeleopConfig(box_xy=5.0, box_up=5.0, preset=2)
    ctl, _ = started(cfg)
    out, t = run(ctl, pad(ly=-1.0, buttons={"deadman": True}), 12.0)  # towards and past the base / 베이스 쪽
    flags = set()
    prev = out.q.copy()
    for _ in range(100):
        t += DT
        out = ctl.step(pad(ly=-1.0, buttons={"deadman": True}), t)
        flags |= out.flags
        assert np.degrees(np.abs(out.q - prev)).max() <= cfg.max_joint_step_deg + 1e-9
        prev = out.q.copy()
    assert any("holding" in f for f in flags)


def test_lb_toggles_open_then_closed_and_never_resends():
    ctl, _ = started()
    sent = []
    t = 0.0
    for _press in range(3):  # open, close, open / 열기, 닫기, 열기
        o = ctl.step(pad(buttons={"gripper_toggle": True}), t)
        if o.gripper_width is not None:
            sent.append(round(o.gripper_width, 4))
        for k in range(30):  # keep holding the grasp / 유지
            o = ctl.step(pad(), t + (k + 1) * DT)
            if o.gripper_width is not None:
                sent.append(round(o.gripper_width, 4))
        t += 31 * DT
    assert sent == [0.095, 0.010, 0.095]


def test_rt_does_nothing_by_default():
    ctl, _ = started()
    run(ctl, pad(rt=1.0), 0.5)
    assert ctl.gripper_target is None


def test_trigger_closes_and_opens_once_with_hysteresis():
    ctl, _ = started(TeleopConfig(trigger_gripper=True))
    out, t = run(ctl, pad(rt=0.0), 0.1)
    out, t = run(ctl, pad(rt=0.4), 0.5, t)
    assert ctl.gripper_target in (None, 0.095)  # between the thresholds nothing closes / 문턱 사이: 닫지 않음
    sent = []
    for k in range(40):  # press deep and keep holding / 깊게 눌러 유지
        o = ctl.step(pad(rt=1.0), t + k * DT)
        if o.gripper_width is not None:
            sent.append(o.gripper_width)
    t += 40 * DT
    assert ctl.gripper_target == pytest.approx(0.010)
    assert sent == [pytest.approx(0.010)]  # one close command, never re-sent while held / 한 번만
    run(ctl, pad(rt=0.5), 0.5, t)  # partial release keeps the grasp / 조금 떼도 유지
    assert ctl.gripper_target == pytest.approx(0.010)
    run(ctl, pad(rt=0.1), 0.5, t + 0.5)
    assert ctl.gripper_target == pytest.approx(0.095)


def test_a_b_override_until_the_trigger_moves():
    ctl, _ = started(TeleopConfig(trigger_gripper=True))
    out, t = run(ctl, pad(rt=1.0), 0.5)
    ctl.step(pad(rt=1.0, buttons={"open": True}), t + DT)
    run(ctl, pad(rt=1.0), 0.5, t + DT)
    assert ctl.gripper_target == pytest.approx(0.095)  # B wins while the trigger stays put / B 우선
    run(ctl, pad(rt=0.0), 0.3, t + 1.0)
    run(ctl, pad(rt=1.0), 0.3, t + 1.5)
    assert ctl.gripper_target == pytest.approx(0.010)  # trigger moved: back in control / 트리거 복귀


def test_trigger_is_ignored_until_it_moves_after_start():
    ctl = TeleopController(KIN)
    ctl.begin(start_joints(KIN), rt=0.4)
    out, _ = run(ctl, pad(rt=0.4), 0.5)
    assert ctl.gripper_target is None and out.gripper_width is None


def test_save_home_and_return():
    ctl, _ = started()
    ctl.step(pad(buttons={"save_home": True}), 0.0)
    home = ctl.p.copy()
    out, t = run(ctl, pad(lx=1.0, buttons={"deadman": True}), 1.0, 0.0)
    assert np.linalg.norm(ctl.p - home) > 0.01
    run(ctl, pad(buttons={"deadman": True, "home": True}), 6.0, t)
    assert np.linalg.norm(ctl.p - home) < 0.002


def test_tilted_start_is_refused():
    ctl = TeleopController(KIN)
    with pytest.raises(ValueError, match="top-down"):
        ctl.begin(np.radians([0, -90, 90, 0, 90, 0]))


# ----------------------------------------------------------------------------- scripted session
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_scripted_session_on_the_fake_arm():
    mapping = Mapping()
    clock = Clock()
    pad_in = ScriptedPad(DEMO_SCRIPT + [{"t": 9.5, "disconnect": True}], mapping, clock)
    robot = FakeRobot(start_joints(KIN), KIN.p)
    q_start = robot.joint_positions()
    ctl = TeleopController(KIN, TeleopConfig(preset=0))
    widths, starts, stops = [], 0, 0
    while clock.t < 10.0:
        clock.t += DT
        pad_in.poll()
        st = pad_in.state
        inputs = role_inputs(
            normalize(st.axes, st.buttons, mapping, clock.t - st.last_ok, st.connected), mapping.roles
        )
        out = ctl.step(inputs, clock.t)
        if out.request == "start":
            robot.start_streaming()
            ctl.begin(robot.joint_positions(), inputs.rt)
            starts += 1
        elif out.request == "stop":
            ctl.end()
            robot.stop_streaming()
            stops += 1
        if ctl.active and out.q is not None:
            robot.stream_joints(out.q)
        if out.gripper_width is not None:
            widths.append(out.gripper_width)
    assert starts == 1 and stops == 1 and not robot.streaming
    assert robot.streamed > 300
    assert widths and min(widths) == pytest.approx(0.010) and widths[-1] == pytest.approx(0.095)
    # Y brought it back to the saved home (the start). / Y 로 시작 위치 복귀.
    assert np.linalg.norm(KIN.tcp(robot.joint_positions())[:3, 3] - KIN.tcp(q_start)[:3, 3]) < 0.003


def test_joint_speed_is_capped_near_full_reach():
    cfg = TeleopConfig(preset=2, box_xy=0.5, max_joint_speed_deg=25.0)
    ctl, _ = started(cfg)
    prev, worst = ctl.q_cmd.copy(), 0.0
    t = 0.0
    for _ in range(300):  # push away from the base towards full reach / 베이스에서 멀어지는 쪽
        out = ctl.step(pad(ly=-1.0, buttons={"deadman": True}), t)
        t += DT
        worst = max(worst, float(np.degrees(np.abs(out.q - prev)).max()) / DT)
        prev = out.q.copy()
    assert worst <= 25.0 + 1e-6


def test_behind_frame_points_forward_from_the_base_to_the_gripper():
    ctl = TeleopController(KIN, TeleopConfig(), "behind")
    ctl.begin(start_joints(KIN))
    p0 = ctl.p.copy()
    run(ctl, pad(ly=1.0, buttons={"deadman": True}), 0.5)
    step = ctl.p - p0
    radial = np.array([p0[0], p0[1]]) / np.linalg.norm(p0[:2])
    assert float(step[:2] @ radial) > 0.01  # forward moves away from the base / 앞 = 베이스에서 멀어짐
