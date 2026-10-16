"""``fr3-teleop``: drive the FR3 with an Xbox 360 (or compatible) pad, top-down, through IK.

Modes / 모드:

* default: read ``/joint_states`` from ``fr3-driver`` and print the would-be targets (dry run);
* ``--execute``: stream them through the driver's servo stream and send gripper commands;
* ``--fake``: a kinematic fake arm, no ROS at all;
* ``--fake-pad [SCRIPT]``: scripted pad input (a built-in demo when no file is given);
* ``--probe-pad``: print raw pad events to build a profile.

Controls (default profile): hold **RB** (deadman) to move; left stick = x/y, right stick vertical
= z, horizontal = wrist yaw; **RT** = gripper width (released open, pressed closed); **A**/**B**
close/open; **X** saves home, **Y** (with RB) returns to it; d-pad up/down or **LB** = speed;
**Start**/**Back** begin/end the session.

Xbox 360 계열 패드로 FR3 를 탑다운 자세로 조작한다. 기본은 드라이 런(목표만 출력), ``--execute`` 로
스트리밍, ``--fake`` 는 ROS 없는 가짜 팔. RB(데드맨)를 누르고 있어야 움직인다.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from fr3tools.gripper import FakeGripper
from fr3tools.kinematics import FR3Kinematics, FR3Params, rot_z, top_down_rotation, transform
from fr3tools.teleop.controller import TeleopConfig, TeleopController, role_inputs
from fr3tools.teleop.gamepad import EvdevReader, JoystickReader, ScriptedPad, find_device, probe
from fr3tools.teleop.mapping import load_profile, normalize

# Built-in --fake-pad script: start, move, turn, grip, save home, move away, go home, stop.
# 내장 스크립트: 시작, 이동, 회전, 파지, 홈 저장, 이동, 홈 복귀, 정지.
DEMO_SCRIPT = [
    {"t": 0.2, "buttons": {"start": 1}},
    {"t": 0.3, "buttons": {"start": 0, "x": 1}},
    {"t": 0.4, "buttons": {"x": 0, "rb": 1}, "axes": {"lx": 0.6}},
    {"t": 1.4, "axes": {"lx": 0.0, "ly": 1.0}},
    {"t": 2.4, "axes": {"ly": 0.0, "ry": -0.7}},
    {"t": 3.2, "axes": {"ry": 0.0, "rx": 0.8}},
    {"t": 4.0, "axes": {"rx": 0.0, "rt": 0.5}},
    {"t": 4.4, "axes": {"rt": 1.0}},
    {"t": 5.0, "buttons": {"b": 1}},
    {"t": 5.1, "buttons": {"b": 0, "y": 1}},
    {"t": 8.5, "buttons": {"y": 0, "rb": 0}, "axes": {"rt": 0.0}},
    {"t": 8.8, "buttons": {"back": 1}},
    {"t": 9.0, "buttons": {"back": 0}},
]


def start_joints(kin: FR3Kinematics) -> np.ndarray:
    """A top-down pose 0.45 m in front of the base for ``--fake``. / --fake 시작 자세."""
    pose = transform(top_down_rotation(math.pi), (0.0, -0.45, 0.20))
    q = kin.inverse(pose, q_ref=np.radians([90, -70, 70, -90, -90, 0]))
    if q is None:
        raise ValueError("fake start pose unreachable with these parameters")
    return q


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="gamepad teleoperation of the FR3 (dry run unless --execute)")
    ap.add_argument("--device", help="joystick device (default: find an Xbox pad by name)")
    ap.add_argument(
        "--evdev", action="store_true", help="read through python-evdev instead (/dev/input/event*)"
    )
    ap.add_argument("--profile", help="pad profile (JSON or YAML)")
    ap.add_argument("--probe-pad", action="store_true", help="print raw pad events and exit")
    ap.add_argument("--fake", action="store_true", help="kinematic fake arm, no ROS")
    ap.add_argument(
        "--fake-pad", nargs="?", const="demo", help="scripted pad (JSON file or the built-in demo)"
    )
    ap.add_argument("--execute", action="store_true", help="really stream to fr3-driver and move the gripper")
    ap.add_argument("--params", help="kinematic parameters JSON (fr3-calib-dh --out)")
    ap.add_argument(
        "--tool-length", type=float, default=0.20, help="flange -> grip point (m) without --params"
    )
    ap.add_argument("--frame", default="base", help="task frame: base | start | 'x y z yaw_deg' (base frame)")
    ap.add_argument("--table-z", type=float, default=0.0, help="table height in the base frame (m)")
    ap.add_argument("--min-height", type=float, default=0.05, help="grip point above the table (m)")
    ap.add_argument(
        "--box", type=float, nargs=6, metavar="V", help="xmin xmax ymin ymax zmin zmax (task frame, m)"
    )
    ap.add_argument("--speed", choices=["slow", "normal", "fast"], default="slow")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--prefix", default="/fr")
    ap.add_argument("--quiet", action="store_true")
    return ap


def parse_frame(text: str):
    if text in ("base", "start"):
        return text
    x, y, z, yaw = (float(v) for v in text.split())
    return transform(rot_z(math.radians(yaw)), (x, y, z))


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    mapping = load_profile(a.profile)
    if a.probe_pad:
        path = a.device or find_device()
        if path is None:
            raise SystemExit("no Xbox pad found; pass --device /dev/input/jsN")
        probe(path)
        return

    if a.fake_pad:
        reader = (
            ScriptedPad(DEMO_SCRIPT, mapping)
            if a.fake_pad == "demo"
            else ScriptedPad.from_file(a.fake_pad, mapping)
        )
    else:
        path = a.device or find_device()
        if path is None:
            raise SystemExit("no Xbox pad found; pass --device /dev/input/jsN (or use --fake-pad)")
        reader = EvdevReader(path, mapping) if a.evdev else JoystickReader(path, mapping)
    params = FR3Params.from_json(a.params) if a.params else FR3Params(tool_xyz=(0.0, 0.0, a.tool_length))
    kin = FR3Kinematics(params)
    names = [p.name for p in TeleopConfig().presets]
    cfg = TeleopConfig(
        table_z=a.table_z,
        min_height=a.min_height,
        preset=names.index(a.speed),
        box=None if a.box is None else tuple(zip(a.box[0::2], a.box[1::2], strict=True)),
    )
    ctl = TeleopController(kin, cfg, parse_frame(a.frame))

    node = None
    gripper = None
    if a.fake:
        from fr3tools.ros.adapter import FakeRobot

        robot = FakeRobot(start_joints(kin), params)
        gripper = FakeGripper()
        stream = True
    else:
        import rclpy
        from rclpy.node import Node

        from fr3tools.ros.adapter import FR3Ros, RosGripper

        rclpy.init()
        node = Node("fr3_teleop")
        robot = FR3Ros(node, params, prefix=a.prefix)
        robot.wait_ready()
        stream = a.execute
        if stream:
            gripper = RosGripper(node)
    mode = "FAKE" if a.fake else ("EXECUTE" if stream else "DRY RUN (add --execute to move)")
    print(f"[teleop] {reader.name} -> {mode}; hold RB to move, Start to begin, Back to end", flush=True)

    period = 1.0 / a.rate
    t0 = time.monotonic()
    last_print = 0.0
    try:
        while True:
            now = time.monotonic()
            reader.poll()
            st = reader.state
            raw = normalize(st.axes, st.buttons, mapping, age=now - st.last_ok, connected=st.connected)
            inputs = role_inputs(raw, mapping.roles, mapping.gripper_trigger)
            out = ctl.step(inputs, now)
            if out.request == "start":
                q = robot.joint_positions()
                try:
                    if stream:
                        robot.start_streaming()
                    ctl.begin(q, inputs.rt)
                    print("\n[teleop] session started", flush=True)
                except Exception as exc:
                    print(f"\n[teleop] cannot start: {exc}", flush=True)
            elif out.request == "stop":
                ctl.end()
                if stream:
                    robot.stop_streaming()
                why = "input lost" if "stale" in out.flags else "requested"
                print(f"\n[teleop] session ended ({why})", flush=True)
            if ctl.active and out.q is not None:
                if stream:
                    robot.stream_joints(out.q)
                elif not a.quiet and now - last_print > 0.5:
                    print(f"\n[dry] q(deg) {np.round(np.degrees(out.q), 2).tolist()}", flush=True)
            if out.gripper_width is not None:
                if gripper is not None:
                    gripper.move_width(out.gripper_width)
                elif not a.quiet:
                    print(f"\n[dry] gripper -> {out.gripper_width * 1000:.1f} mm", flush=True)
            if not a.quiet and now - last_print > 0.1:
                sys.stdout.write("\r" + ctl.status(inputs.rt)[:150].ljust(150))
                sys.stdout.flush()
                last_print = now
            if a.duration and now - t0 > a.duration:
                break
            if isinstance(reader, ScriptedPad) and reader.finished and not ctl.active:
                break
            time.sleep(max(0.0, period - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        if ctl.active and stream:
            robot.stop_streaming()
        reader.close()
        if node is not None:
            import rclpy

            node.destroy_node()
            rclpy.try_shutdown()
        print("\n[teleop] bye", flush=True)


if __name__ == "__main__":
    main()
