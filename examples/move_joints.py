"""Move to a joint target slowly through the limited servo stream (direct SDK, no ROS).

    python examples/move_joints.py --j6 5                 # plan only
    python examples/move_joints.py --j6 5 --execute       # move j6 by +5 deg and stop

The arm moves only with ``--execute``, in AUTO mode, after a READY preflight, under the motion
lock, at ``--speed`` deg/s with the stream's acceleration limit and the safety monitor. Stop the
driver first: this script needs the state port. ``--from`` plans offline without a controller.

제한된 서보 스트림으로 관절 목표까지 천천히 이동(SDK 직결). ``--execute`` 가 있어야 움직이며, 사전 점검
READY, 모션 잠금, 속도 제한, 안전 감시 아래에서만 움직인다. 드라이버를 먼저 끈다(상태 포트 필요).
"""

import argparse
import math
import time

import numpy as np

from fr3tools.servo import ServoParams, ServoStream, servo_step


def plan_duration(q0, q1, params: ServoParams) -> float:
    """Simulate the limiter to estimate the duration. / 제한기를 모의해 소요 시간 추정."""
    q, v = np.asarray(q0, float), np.zeros(6)
    for k in range(200000):
        q, v = servo_step(q, v, q1, params.max_step, params.max_dv)
        if np.abs(q - q1).max() < 1e-6 and np.abs(v).max() < 1e-9:
            return k * params.cmd_t
    return math.inf


def main() -> None:
    ap = argparse.ArgumentParser(
        description="move joints through the servo stream (dry run unless --execute)"
    )
    for j in range(1, 7):
        ap.add_argument(f"--j{j}", type=float, default=0.0, help=f"j{j} offset (deg)")
    ap.add_argument("--to", type=float, nargs=6, metavar="DEG", help="absolute target instead of offsets")
    ap.add_argument("--from", dest="start", type=float, nargs=6, metavar="DEG", help="plan offline from here")
    ap.add_argument("--speed", type=float, default=10.0, help="joint speed limit (deg/s), max 30")
    ap.add_argument("--max-move", type=float, default=30.0, help="refuse larger joint moves (deg)")
    ap.add_argument("--ip", default=None)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    params = ServoParams(max_joint_speed_deg=min(args.speed, 30.0), max_joint_accel_deg=30.0)
    arm = None
    if args.start is not None:
        q0 = np.radians(args.start)
    else:
        from fr3tools import sdk

        arm = sdk.SdkArm(args.ip or sdk.DEFAULT_IP)
        q0 = arm.sample()[0]
    offsets = np.radians([getattr(args, f"j{j}") for j in range(1, 7)])
    q1 = np.radians(args.to) if args.to else q0 + offsets
    move = np.degrees(np.abs(q1 - q0)).max()
    print(f"from {np.degrees(q0).round(2).tolist()}\n  to {np.degrees(q1).round(2).tolist()}")
    seconds = plan_duration(q0, q1, params)
    print(f"largest joint move {move:.2f} deg, about {seconds:.1f} s at {params.max_joint_speed_deg} deg/s")
    if move > args.max_move:
        raise SystemExit(f"refused: {move:.1f} deg > --max-move {args.max_move}")
    from fr3tools.kinematics import FR3Kinematics

    if not FR3Kinematics().in_limits(q1, margin_deg=5.0):
        raise SystemExit("refused: target within 5 deg of a joint limit")
    if not args.execute or arm is None:
        print("dry run: add --execute (and no --from) to move")
        return

    from fr3tools.lock import MotionLock
    from fr3tools.safety import preflight

    report = preflight(arm.status())
    print(report.text())
    if not report.ready:
        raise SystemExit("not ready")
    with MotionLock(arm.ip):
        stream = ServoStream(arm, lambda: params)
        stream.start(q0)
        try:
            t0 = time.monotonic()
            while stream.active and time.monotonic() - t0 < 60.0:
                stream.set_target(q1)  # keep the target fresh / 목표 갱신 유지
                if np.degrees(np.abs(arm.sample()[0] - q1)).max() < 0.05:
                    break
                time.sleep(0.05)
        finally:
            reason = stream.stop()
    print(f"done ({reason or 'ok'}), joints {np.degrees(arm.sample()[0]).round(2).tolist()}")


if __name__ == "__main__":
    main()
