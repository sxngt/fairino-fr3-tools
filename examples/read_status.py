"""Read and decode the controller state (read only). / 제어기 상태 읽기(읽기 전용).

    python examples/read_status.py --ip 192.168.58.2 [--watch 1.0]

Needs the vendor SDK and the state port: if ``fr3-driver`` runs, read ``/fr/status`` instead
(``ros2 topic echo /fr/status``).
제조사 SDK 와 상태 포트가 필요하다. 드라이버가 실행 중이면 /fr/status 를 읽는다.
"""

import argparse
import time

import numpy as np

from fr3tools import sdk
from fr3tools.kinematics import FR3Kinematics, matrix_to_pose
from fr3tools.safety import preflight


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ip", default=sdk.DEFAULT_IP)
    ap.add_argument("--watch", type=float, default=0.0, help="repeat every N seconds")
    args = ap.parse_args()
    kin = FR3Kinematics()
    while True:
        st = sdk.read_status(args.ip)
        if not st.ok:
            print(f"no status: {st.error}")
        else:
            print(f"mode {st.mode}, state {sdk.ROBOT_STATE.get(st.robot_state)}, {st.error_text}")
            print(f"joints (deg)       {[round(v, 2) for v in st.jnt_ang]}")
            print(f"flange controller  {[round(v, 2) for v in st.flange_pos]}")
            # The nominal model differs by ~1.5 mm; fit yours with fr3-calib-dh.
            # 공칭 모델은 약 1.5 mm 다르다: fr3-calib-dh 로 피팅.
            fk = matrix_to_pose(kin.flange(np.radians(st.jnt_ang)))
            print(f"flange nominal FK  {[round(v, 2) for v in fk]}")
            print(f"gripper active {st.gripper_active}, position {st.gripper_position} %")
            print(preflight(st).text())
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
