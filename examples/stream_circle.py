"""Draw a small horizontal TCP circle through IK, streamed to ``fr3-driver``.

    python examples/stream_circle.py --from 77 -69 57 -78 -90 -13      # plan offline
    python examples/stream_circle.py                                   # plan from the driver's joints
    python examples/stream_circle.py --execute                         # stream it

The circle starts and ends at the current TCP (it is centred ``--radius`` beside it), keeps the
current orientation, ramps its speed in and out, and is checked for IK continuity and peak joint
speed before anything moves.

현재 TCP 에서 시작·끝나는 작은 수평 원을 IK 로 계산해 드라이버로 스트리밍한다. 움직이기 전에 IK
연속성과 최고 관절 속도를 검사한다.
"""

import argparse
import math
import time

import numpy as np

from fr3tools.kinematics import FR3Kinematics, FR3Params, transform


def circle_path(kin, q0, radius: float, period: float, rate: float):
    """Joint path (N, 6) and times for one circle from the TCP at ``q0``. / 원 경로."""
    start = kin.tcp(q0)
    centre = start[:3, 3] + np.array([radius, 0.0, 0.0])
    n = int(period * rate)
    ts = np.arange(n + 1) / rate
    path, q = [], np.asarray(q0, dtype=float)
    for t in ts:
        s = t / period
        phase = (
            2 * math.pi * (s - math.sin(2 * math.pi * s) / (2 * math.pi))
        )  # smooth start and stop / 부드러운 시작·끝
        p = centre + radius * np.array([-math.cos(phase), -math.sin(phase), 0.0])
        q_next = kin.inverse(transform(start[:3, :3], p), q_ref=q)
        if q_next is None or np.degrees(np.abs(q_next - q)).max() > 2.0:
            raise ValueError(f"IK lost continuity at t = {t:.2f} s")
        path.append(q_next)
        q = q_next
    return ts, np.array(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="small TCP circle (dry run unless --execute)")
    ap.add_argument("--radius", type=float, default=0.02, help="m (max 0.05)")
    ap.add_argument("--period", type=float, default=10.0, help="s per circle")
    ap.add_argument("--tool-length", type=float, default=0.2)
    ap.add_argument("--from", dest="start", type=float, nargs=6, metavar="DEG", help="plan offline from here")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    radius = min(args.radius, 0.05)
    kin = FR3Kinematics(FR3Params(tool_xyz=(0.0, 0.0, args.tool_length)))
    rate = 50.0

    robot = node = None
    if args.start is not None:
        q0 = np.radians(args.start)
    else:
        import rclpy
        from rclpy.node import Node

        from fr3tools.ros.adapter import FR3Ros

        rclpy.init()
        node = Node("fr3_circle_example")
        robot = FR3Ros(node, kin.p)
        q0 = robot.wait_ready()
    ts, path = circle_path(kin, q0, radius, args.period, rate)
    peak = np.degrees(np.abs(np.diff(path, axis=0)).max(axis=0) * rate)
    print(f"{len(ts)} points, radius {radius * 1000:.0f} mm, {args.period:.0f} s")
    print(f"peak joint speed {peak.max():.1f} deg/s")
    if peak.max() > 25.0:
        raise SystemExit("too fast for the driver's default limit: raise --period")
    if not args.execute or robot is None:
        print("dry run: add --execute (with fr3-driver running) to stream it")
        return
    robot.start_streaming()
    try:
        t0 = time.monotonic()
        for t, q in zip(ts, path, strict=True):
            robot.stream_joints(q)
            robot.spin(max(0.0, t0 + t - time.monotonic()))
        for _ in range(int(rate)):  # hold the end point for a second / 끝점 유지
            robot.stream_joints(path[-1])
            robot.spin(1.0 / rate)
    finally:
        robot.stop_streaming()
        node.destroy_node()
        rclpy.try_shutdown()
    print("done")


if __name__ == "__main__":
    main()
