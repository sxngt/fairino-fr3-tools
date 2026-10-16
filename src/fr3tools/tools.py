"""Command-line tools: preflight, power and mode, error lookup, DH calibration, servo probe.

* ``fr3-check``: read-only preflight of the controller (exit 0 = READY);
* ``fr3-power``: status, enable, AUTO/MANUAL, drag, stop, error reset, speed and ``recover``
  (wait for the e-stop release, reset, enable, AUTO); state changes need ``--yes``;
* ``fr3-errors``: what a return code or a (main, sub) state code means;
* ``fr3-calib-dh``: fit the DH parameters to (joints, controller pose) pairs, offline from a CSV or
  JSON log, or collect the log with ``GetForwardKin`` (a computation; the arm does not move);
* ``fr3-probe``: through the running driver, wait for ``/joint_states``, set driver parameters and,
  with ``--execute``, check that the servo stream really moves the arm (j6 ±0.3°).

The SDK tools need the state port for status: while the driver runs they get an XML-RPC-only
handle, so commands work but status does not.

명령행 도구: 사전 점검, 전원·모드, 에러 조회, DH 보정, 서보 확인. 드라이버가 상태 포트를 잡고 있으면
SDK 도구의 명령은 되지만 상태 읽기는 안 된다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time

import numpy as np

from fr3tools import errors, sdk
from fr3tools.kinematics import FR3Params, fit_dh
from fr3tools.lock import MotionLock
from fr3tools.safety import preflight


# ----------------------------------------------------------------------------- preflight
def check_main(argv=None) -> None:
    """``fr3-check``: read-only controller preflight. / 읽기 전용 제어기 사전 점검."""
    ap = argparse.ArgumentParser(description="FR3 preflight (read only); exit 0 = READY")
    ap.add_argument("--ip", default=sdk.DEFAULT_IP)
    ap.add_argument("--any-mode", action="store_true", help="do not require AUTO mode")
    ap.add_argument("--no-versions", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--watch", type=float, default=0.0, help="repeat every N seconds")
    args = ap.parse_args(argv)
    while True:
        status = sdk.read_status(args.ip)
        versions = None if (args.no_versions or not status.ok) else sdk.versions(args.ip)
        report = preflight(status, None if args.any_mode else "AUTO", versions)
        if args.json:
            out = {"ready": report.ready, "checks": report.checks, "status": status.to_dict()}
            print(json.dumps(out, default=str))
        else:
            print(time.strftime("%H:%M:%S") + "\n" + report.text(), flush=True)
        if not args.watch:
            sys.exit(0 if report.ready else 1)
        time.sleep(args.watch)


# ----------------------------------------------------------------------------- error lookup
def errors_main(argv=None) -> None:
    """``fr3-errors``: explain return codes and state codes. / 반환값·상태 코드 설명."""
    ap = argparse.ArgumentParser(description="explain FR3 / SDK error codes")
    ap.add_argument("codes", nargs="*", type=int, help="return codes, e.g. 29 -4 -1064")
    ap.add_argument("--state", nargs=2, type=int, metavar=("MAIN", "SUB"), help="state packet main/sub code")
    ap.add_argument("--list", action="store_true", help="print every known code")
    args = ap.parse_args(argv)
    if args.list:
        for table in (errors.SDK_CODES, errors.CONTROLLER_CODES, errors.ROS_HW_CODES):
            for info in table.values():
                print(info.text())
        for (main, sub), info in errors.STATE_CODES.items():
            print(f"state {main}/{sub}: {info.meaning} -> {info.hint}")
    for code in args.codes:
        print(errors.describe(code))
    if args.state:
        print(errors.describe_state(*args.state))
    if not (args.list or args.codes or args.state):
        ap.print_help()


# ----------------------------------------------------------------------------- DH calibration
def load_samples(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(joints deg, poses) from a CSV or JSON log. / CSV·JSON 기록에서 (관절, 포즈) 읽기.

    CSV: a header with ``j1..j6`` and ``x, y, z`` (mm), optionally ``rx, ry, rz`` (deg).
    JSON: ``{"samples": [{"joints_deg": [...6], "pose": [x, y, z(, rx, ry, rz)]}, ...]}`` or the list
    alone.
    """
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"{path}: no rows")
        pose_keys = ["x", "y", "z"] + (["rx", "ry", "rz"] if "rx" in rows[0] else [])
        joints = [[float(r[f"j{i}"]) for i in range(1, 7)] for r in rows]
        poses = [[float(r[k]) for k in pose_keys] for r in rows]
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        samples = data["samples"] if isinstance(data, dict) else data
        joints = [s["joints_deg"] for s in samples]
        poses = [s["pose"] for s in samples]
    return np.asarray(joints, dtype=float), np.asarray(poses, dtype=float)


def save_samples(path: str, joints, poses, **meta) -> None:
    samples = [
        {"joints_deg": list(map(float, j)), "pose": list(map(float, p))}
        for j, p in zip(joints, poses, strict=True)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**meta, "samples": samples}, f, indent=1)


def collect_samples(
    ip: str, n: int, margin_deg: float = 5.0, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Random joints inside the soft limits and the controller's ``GetForwardKin`` for each.

    ``GetForwardKin`` is a computation on the controller; the arm does not move.
    소프트 리밋 안의 무작위 관절과 제어기 GetForwardKin 결과(계산만, 팔은 움직이지 않는다).
    """
    rpc = sdk.connect(ip)
    rng = np.random.default_rng(seed)
    lim = np.asarray(FR3Params().limits_deg, dtype=float) + np.array([margin_deg, -margin_deg])
    joints, poses = [], []
    for _ in range(n):
        q = [float(v) for v in rng.uniform(lim[:, 0], lim[:, 1])]
        pose = errors.check(rpc.GetForwardKin(q), "GetForwardKin")
        joints.append(q)
        poses.append([float(v) for v in pose[:6]])
    meta: dict = {"ip": ip}
    try:
        meta["dh_compensation"] = rpc.GetDHCompensation()
    except Exception as exc:
        meta["dh_compensation"] = f"unavailable: {exc}"
    return np.asarray(joints), np.asarray(poses), meta


def calib_dh_main(argv=None) -> None:
    """``fr3-calib-dh``: fit DH parameters to controller poses. / 제어기 포즈로 DH 피팅."""
    ap = argparse.ArgumentParser(
        description="fit FR3 DH parameters to (joints, controller pose) pairs",
        epilog="offline: fr3-calib-dh --log samples.csv ; online: fr3-calib-dh --collect 40 --save log.json",
    )
    ap.add_argument("--log", help="CSV or JSON log of samples (offline mode)")
    ap.add_argument(
        "--collect", type=int, default=0, metavar="N", help="collect N samples with GetForwardKin"
    )
    ap.add_argument("--save", help="write the collected samples here (JSON)")
    ap.add_argument("--ip", default=sdk.DEFAULT_IP)
    ap.add_argument("--params", help="starting parameters (JSON from a previous fit)")
    ap.add_argument(
        "--tool-xyz", type=float, nargs=3, metavar="M", help="tool offset of the logged frame (m)"
    )
    ap.add_argument("--fit-tool", action="store_true", help="also fit the tool x/y offset")
    ap.add_argument("--rotation-weight", type=float, default=0.0, help="mm per rad of orientation error")
    ap.add_argument("--out", help="write the fitted parameters (JSON)")
    ap.add_argument("--max-error-mm", type=float, default=0.1, help="refuse to write above this max error")
    args = ap.parse_args(argv)
    if bool(args.log) == bool(args.collect):
        ap.error("give exactly one of --log or --collect")
    if args.log:
        joints, poses = load_samples(args.log)
    else:
        joints, poses, meta = collect_samples(args.ip, args.collect)
        print(f"collected {len(joints)} samples (controller DH compensation: {meta['dh_compensation']})")
        if args.save:
            save_samples(args.save, joints, poses, **{k: str(v) for k, v in meta.items()})
            print(f"samples -> {args.save}")
    start = FR3Params.from_json(args.params) if args.params else FR3Params()
    if args.tool_xyz:
        start = FR3Params.from_mapping({**start.to_mapping(), "tool_xyz": args.tool_xyz})
    fit = fit_dh(joints, poses, start, fit_tool=args.fit_tool, rotation_weight_mm=args.rotation_weight)
    print(fit.summary())
    if args.out:
        if fit.max_after_mm > args.max_error_mm:
            print(f"max error {fit.max_after_mm:.3f} mm > {args.max_error_mm}: not written")
            sys.exit(1)
        fit.params.to_json(
            args.out, fit={"rms_mm": fit.rms_after_mm, "max_mm": fit.max_after_mm, "n": fit.n_samples}
        )
        print(f"parameters -> {args.out}")


# ----------------------------------------------------------------------------- power / mode


def _show(ip: str) -> sdk.RobotStatus:
    st = sdk.read_status(ip, timeout=1.5)
    if not st.ok:
        print(f"status unavailable: {st.error}")
        return st
    print(
        f"mode {st.mode}  state {sdk.ROBOT_STATE.get(st.robot_state, st.robot_state)}  "
        f"error {st.main_code}/{st.sub_code}  estop {st.estop}  collision {st.collision}  "
        f"safety {st.safety_stop0}/{st.safety_stop1}  motion_done {st.motion_done}\n"
        f"  q(deg) {[round(v, 2) for v in st.jnt_ang]}  flange(mm,deg) {[round(v, 1) for v in st.flange_pos]}"
    )
    return st


def _report(what: str, err: int) -> bool:
    print(f"{what} -> OK" if err == 0 else errors.describe(err, what))
    return err == 0


def power_main(argv=None) -> None:
    """``fr3-power``: controller power, mode and recovery. / 제어기 전원·모드·복구."""
    ap = argparse.ArgumentParser(description="FR3 controller power/mode helper")
    ap.add_argument(
        "cmd",
        choices=[
            "status",
            "enable",
            "disable",
            "auto",
            "manual",
            "drag",
            "stop",
            "reset",
            "speed",
            "recover",
        ],
    )
    ap.add_argument("arg", nargs="?", default="on", help="drag: on|off; speed: 0..100")
    ap.add_argument("--ip", default=sdk.DEFAULT_IP)
    ap.add_argument(
        "--wait-estop", type=float, default=120.0, help="recover: seconds to wait for the e-stop release"
    )
    ap.add_argument("--yes", action="store_true", help="confirm commands that change the controller state")
    args = ap.parse_args(argv)

    if args.cmd in {"enable", "auto", "drag", "speed", "recover", "disable", "manual"} and not args.yes:
        print(f"'{args.cmd}' changes the controller state; add --yes to send it")
        sys.exit(2)
    if args.cmd == "status":
        st = _show(args.ip)
        if st.ok:
            print("versions:", sdk.versions(args.ip))
        sys.exit(0 if st.ok else 1)

    rpc = sdk.connect(args.ip)
    ok = True
    if args.cmd == "enable":
        ok = _report("RobotEnable(1)", rpc.RobotEnable(1))
        time.sleep(1.0)
    elif args.cmd == "disable":
        ok = _report("RobotEnable(0)", rpc.RobotEnable(0))
        time.sleep(1.0)
    elif args.cmd == "auto":
        ok = _report("Mode(0)", rpc.Mode(0))
        time.sleep(0.5)
    elif args.cmd == "manual":
        ok = _report("Mode(1)", rpc.Mode(1))
        time.sleep(0.5)
    elif args.cmd == "drag":
        on = args.arg.lower() in ("on", "1", "true")
        if on:
            # Drag teaching needs MANUAL mode. / 드래그 교시는 수동 모드에서.
            _report("Mode(1)", rpc.Mode(1))
            time.sleep(0.5)
        ok = _report(f"DragTeachSwitch({int(on)})", rpc.DragTeachSwitch(1 if on else 0))
        time.sleep(1.0)
    elif args.cmd == "stop":
        ok = _report("StopMotion", sdk.stop_motion(rpc))
    elif args.cmd == "reset":
        ok = _report("ResetAllError", rpc.ResetAllError())
        time.sleep(1.0)
    elif args.cmd == "speed":
        ok = _report(f"SetSpeed({args.arg})", rpc.SetSpeed(int(float(args.arg))))
    elif args.cmd == "recover":
        ok = _recover(args.ip, rpc, args.wait_estop)
    _show(args.ip)
    sys.exit(0 if ok else 1)


def _recover(ip: str, rpc, wait_estop: float) -> bool:
    """Wait for the e-stop release, then reset, enable, AUTO. / E-stop 해제 대기 후 리셋·enable·AUTO."""
    print("[recover] 1/3 waiting for the e-stop release (turn the button)...", flush=True)
    t0 = time.time()
    while True:
        st = sdk.read_status(ip, timeout=1.5)
        if st.ok and st.safety_ok:
            break
        if not st.ok and "state" in st.error:
            print("[recover] cannot read the state port (driver running?); stop the driver first")
            return False
        if time.time() - t0 > wait_estop:
            print(f"[recover] still stopped: estop={st.estop} SI={st.safety_stop0}/{st.safety_stop1}")
            return False
        time.sleep(1.0)
    print("[recover] 2/3 ResetAllError", flush=True)
    _report("ResetAllError", rpc.ResetAllError())
    time.sleep(1.0)
    print("[recover] 3/3 RobotEnable(1) + Mode(0)", flush=True)
    ok = _report("RobotEnable(1)", rpc.RobotEnable(1))
    time.sleep(1.0)
    ok = _report("Mode(0)", rpc.Mode(0)) and ok
    time.sleep(0.5)
    return ok


# ----------------------------------------------------------------------------- servo probe


def probe_offsets(duration: float = 2.0, amplitude_deg: float = 0.3, rate: float = 50.0) -> np.ndarray:
    """j6 offsets (rad) of the probe: 0 → amplitude → 0, triangular. / 탐색 j6 오프셋(삼각형)."""
    n = max(2, int(round(duration * rate)))
    s = np.linspace(0.0, 1.0, n + 1)
    return np.radians(amplitude_deg) * (1.0 - np.abs(2.0 * s - 1.0))


def parse_param(text: str) -> tuple[str, object]:
    """``name=value`` with numbers as floats. / ``이름=값`` (숫자는 float)."""
    name, _, raw = text.partition("=")
    if not name or not raw:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    try:
        return name.strip(), float(raw)
    except ValueError:
        return name.strip(), raw.strip()


def probe_main(argv=None) -> None:
    """``fr3-probe``: driver reachability, parameters and a servo motion check.

    드라이버 응답, 파라미터 설정, 서보 동작 확인.
    """
    ap = argparse.ArgumentParser(description="probe the FR3 driver's servo stream (j6 ±0.3 deg)")
    ap.add_argument("--wait", type=float, default=6.0, help="seconds to wait for /joint_states")
    ap.add_argument("--set", type=parse_param, action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--node", default="/fr3_driver", help="driver node for --set")
    ap.add_argument("--execute", action="store_true", help="really move j6 by the amplitude and back")
    ap.add_argument("--amplitude-deg", type=float, default=0.3)
    ap.add_argument("--prefix", default="/fr")
    args = ap.parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger

    rclpy.init()
    node = Node("fr3_probe")
    latest: dict = {}

    def on_joints(msg) -> None:
        d = dict(zip(msg.name, msg.position, strict=False))
        if all(j in d for j in sdk.JOINTS):
            latest["q"] = np.array([d[j] for j in sdk.JOINTS], dtype=float)

    node.create_subscription(JointState, "/joint_states", on_joints, 10)

    def spin_for(seconds: float) -> None:
        t_end = time.time() + seconds
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.02)

    def call(name: str) -> tuple[bool, str]:
        client = node.create_client(Trigger, name)
        if not client.wait_for_service(timeout_sec=3.0):
            return False, f"{name} unavailable"
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        return (False, f"{name} timed out") if result is None else (result.success, result.message)

    code = 1
    lock = MotionLock("ros" + args.prefix.replace("/", "_"))
    try:
        t0 = time.time()
        while "q" not in latest and time.time() - t0 < args.wait:
            rclpy.spin_once(node, timeout_sec=0.1)
        if "q" not in latest:
            print(f"no /joint_states within {args.wait:.0f} s (driver not running?)")
            return
        if args.set and not _set_parameters(node, args.node, args.set):
            return
        if not args.execute:
            print("driver up (dry run: add --execute for the j6 motion check)")
            code = 0
            return

        lock.acquire()
        call(f"{args.prefix}/reset_error")
        spin_for(0.5)
        ok, message = call(f"{args.prefix}/servo_start")
        if not ok:
            print(f"servo start refused: {message}")
            return
        pub = node.create_publisher(JointState, f"{args.prefix}/servo_target", 10)
        q0 = latest["q"].copy()
        peak = 0.0
        for offset in probe_offsets(amplitude_deg=args.amplitude_deg):
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = list(sdk.JOINTS)
            q = q0.copy()
            q[5] += offset
            msg.position = [float(v) for v in q]
            pub.publish(msg)
            spin_for(0.02)
            peak = max(peak, abs(math.degrees(latest["q"][5] - q0[5])))
        spin_for(0.5)
        call(f"{args.prefix}/servo_stop")
        moved = peak > 0.1
        print(f"j6 peak excursion {peak:.3f} deg -> {'MOVED' if moved else 'NOT MOVED'}")
        code = 0 if moved else 1
    finally:
        lock.release()
        node.destroy_node()
        rclpy.try_shutdown()
        sys.exit(code)


def _set_parameters(node, target: str, params: list[tuple[str, object]]) -> bool:
    """Set parameters of another node through its ``set_parameters`` service. / 다른 노드 파라미터 설정."""
    import rclpy
    from rcl_interfaces.srv import SetParameters
    from rclpy.parameter import Parameter

    client = node.create_client(SetParameters, f"{target}/set_parameters")
    if not client.wait_for_service(timeout_sec=3.0):
        print(f"{target}/set_parameters unavailable")
        return False
    request = SetParameters.Request()
    request.parameters = [Parameter(name, value=value).to_parameter_msg() for name, value in params]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    response = future.result()
    if response is None:
        print("set_parameters timed out")
        return False
    ok = True
    for (name, value), result in zip(params, response.results, strict=False):
        if not result.successful:
            print(f"parameter {name}={value} refused: {result.reason}")
            ok = False
    if ok:
        print("parameters set: " + ", ".join(f"{n}={v}" for n, v in params))
    return ok
