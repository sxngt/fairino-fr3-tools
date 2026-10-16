"""AG-95 through the controller: activate if needed, open, go to a width, open again.

    python examples/gripper.py                 # prints what it would do
    python examples/gripper.py --execute       # moves the fingers (clear them first)

The reported width follows the command, so it says nothing about a grasp (docs/PITFALLS.md).
With ``fr3-driver`` running, send widths through its action instead
(``fr3tools.ros.adapter.RosGripper``).

제어기 경유 AG-95: 필요하면 활성화, 열기, 폭 이동, 다시 열기. 보고 폭은 명령을 따라가므로 파지 판정에
쓰지 않는다. 드라이버가 실행 중이면 그 액션(RosGripper)을 쓴다.
"""

import argparse

from fr3tools.gripper import MIN_WIDTH_M, STROKE_M, ControllerGripper


def main() -> None:
    ap = argparse.ArgumentParser(description="AG-95 example (dry run unless --execute)")
    ap.add_argument("--mm", type=float, default=40.0, help="width to visit (mm)")
    ap.add_argument("--open-pct", type=int, default=100, choices=(0, 100))
    ap.add_argument("--ip", default=None)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    width = max(MIN_WIDTH_M, min(STROKE_M, args.mm / 1000.0))
    steps = [STROKE_M, width, STROKE_M]
    print(
        "plan: activate if inactive (fingers sweep once), then widths "
        + ", ".join(f"{w * 1000:.0f} mm" for w in steps)
    )
    if not args.execute:
        print("dry run: add --execute to move the fingers")
        return
    from fr3tools import sdk
    from fr3tools.lock import MotionLock

    arm = sdk.SdkArm(args.ip or sdk.DEFAULT_IP)
    with MotionLock(arm.ip):
        g = ControllerGripper(arm, open_pct=args.open_pct)
        for w in steps:
            print(f"-> {w * 1000:.0f} mm, reported {g.move_width(w) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
