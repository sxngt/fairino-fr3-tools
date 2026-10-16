# fairino-fr3-tools

[English](README.md) · **한국어**

**FAIRINO FR3** 협동로봇과 **DH-Robotics AG-95** 그리퍼를 쓰는 개발자·연구자를 위한 도구 모음.
공식 SDK 에 없는 부분을 실제 로봇에서 배운 대로 채웠다.

> **비공식 프로젝트입니다. FAIRINO 와 무관하며 FAIRINO 의 승인을 받지 않았습니다.**
> (Unofficial. Not affiliated with or endorsed by FAIRINO.) FAIRINO 의 Python SDK 를 사용하지만
> 그 일부가 아니다. 산업용 팔을 움직이므로 [docs/SAFETY.md](docs/SAFETY.md) 를 먼저 읽을 것.

## 기능

| | 공식 Python SDK | fr3tools |
|---|---|---|
| 제어기 호출(XML-RPC, 상태 패킷) | 있음 | 감싼다: 잠금 핸들 하나, 상태 포트 공유, 상태 해석, 에러 코드 조회 |
| 기구학 | 제어기에서만 (`GetForwardKin`, `GetInverseKin`) | 오프라인 DH 정기구학, 8 가지 해석 IK, 야코비안, 가지 ↔ 제어기 `config` |
| DH 보정 | - | `fr3-calib-dh`: 제어기 포즈에 링크 파라미터 피팅(약 0.03 mm), 기록 파일로 오프라인 |
| 안전한 스트리밍 | `ServoJ` | 속도·가속도 제한과 제동, 급변·무갱신 검사, 래치형 안전 감시 |
| 사전 점검·복구 | - | `fr3-check`(READY / NOT READY 와 조치), `fr3-power recover` |
| ROS 2 | 다른 펌웨어용 ros2_control 패키지 | `fr3-driver`: `/joint_states`, 서보 스트림, MoveIt `FollowJointTrajectory`(검증·취소), AG-95 액션 |
| AG-95 그리퍼 | `MoveGripper` 원시 호출 | 활성화, 제한, 폭 단위, Modbus RTU 직결, 피드백의 한계 정리 |
| 두 스크립트 동시 실행 | - | 협조적 모션 잠금 |
| 핸드아이 보정 | - | `fr3-handeye`: 그리퍼 ArUco 보드로 능동 eye-to-hand, 초기 추정 불필요, 합성 모드 |
| 카메라 내부 파라미터 | - | `fr3-camera-calib`: 블러·뷰 다양성 검사가 있는 체스보드 보정 |
| 원격 조작 | - | `fr3-teleop`: Xbox 360 패드, 비례 속도 탑다운 직교 제어, 데드맨 |
| 노하우 | - | [docs/PITFALLS.md](docs/PITFALLS.md): 실기에서 시간을 잡아먹은 27 가지 |

제어기와 통신하지 않는 기능은 SDK·ROS·로봇 없이 동작하며 테스트로 검증한다.

## 설치

```bash
git clone https://github.com/sxngt/fairino-fr3-tools.git
cd fairino-fr3-tools
pip install -e ".[vision]"          # 또는 uv sync --extra dev
```

extra: `vision`(OpenCV: 카메라 보정, ArUco), `serial`(USB-RS485 AG-95), `dev`(테스트, ruff).
`ros` 는 일부러 비어 있다: `rclpy` 와 메시지 패키지는 ROS 2 설치(Humble 검증)에서 오므로
`/opt/ros/humble/setup.bash` 를 source 하고 그것이 보이는 Python 을 쓴다.

### 제조사 SDK

`fairino` 패키지는 **PyPI 에 없다**. 제어기 펌웨어에 맞는
[fairino-python-sdk](https://github.com/FAIR-INNOVATION/fairino-python-sdk) 태그를 쓴다
(검증: **SDK v2.1.6 + 펌웨어 v3.8.6**, 태그 `v2.1.6_robot3.8.6`):

```bash
git clone --branch v2.1.6_robot3.8.6 --depth 1 https://github.com/FAIR-INNOVATION/fairino-python-sdk.git
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/fairino" && touch "$SITE/fairino/__init__.py"
cp fairino-python-sdk/linux/fairino/Robot.py "$SITE/fairino/"   # 또는 미리 빌드된 linux/libfairino/*.so
python -c "from fairino import Robot; print('SDK OK')"
```

펌웨어가 검증 버전과 다르면 `fr3-check` 가 경고한다.

## 빠른 시작

```bash
fr3-check                         # 읽기 전용 사전 점검, 종료 코드 0 = READY
fr3-errors 29 --state 5 1         # 반환값·상태 코드의 의미
fr3-calib-dh --collect 40 --save dh_log.json --out fr3_params.json   # 팔은 움직이지 않는다

source /opt/ros/humble/setup.bash
fr3-driver                        # 상태 포트를 잡고, 클라이언트가 요청하기 전엔 움직이지 않는다
fr3-probe                         # 드라이버 확인(드라이 런), 이어서 fr3-probe --execute (j6 ±0.3°)

python examples/move_joints.py --j6 5          # 계획만 출력, --execute 로 실제 이동
fr3-teleop --fake --fake-pad                   # 패드·로봇 없이 게임패드 원격 조작 데모
fr3-handeye --fake 0                           # 합성 데이터로 eye-to-hand 보정
```

제어기 주소 기본값은 공장 설정 `192.168.58.2` 이며 `FR3_IP` 또는 `--ip` 로 바꾼다.

| 명령 | 내용 |
|---|---|
| `fr3-check` | 사전 점검: 비상정지, 안전 입력, 충돌, 에러 코드, 모드, enable, 리밋, 버전 |
| `fr3-power` | `status`, `stop`, `reset`; `--yes` 와 함께 `enable`, `auto`, `manual`, `drag`, `speed`, `recover` |
| `fr3-errors` | 반환값과 상태 (main, sub) 코드 설명 |
| `fr3-calib-dh` | DH 파라미터 피팅: `--log`(CSV/JSON, 오프라인) 또는 `--collect N` |
| `fr3-driver` | ROS 2 드라이버 ([docs/ROS2.md](docs/ROS2.md)) |
| `fr3-probe` | 드라이버 확인, `--execute` 면 서보 스트림이 팔을 움직이는지 확인 |
| `fr3-gripper` | AG-95: `status`, `configure`; `--execute` 와 함께 `activate`, `open`, `width`, `probe` |
| `fr3-camera-calib` | 체스보드 내부 파라미터 보정(실시간 또는 `--images`) |
| `fr3-handeye` | 능동 eye-to-hand 보정(`--fake SEED`, 또는 `--execute` 로 실제 로봇) |
| `fr3-teleop` | 게임패드 원격 조작 ([docs/TELEOP.md](docs/TELEOP.md)) |

## 라이브러리

```python
import numpy as np
from fr3tools.kinematics import FR3Kinematics, FR3Params, top_down_rotation, transform

kin = FR3Kinematics(FR3Params(tool_xyz=(0.0, 0.0, 0.2)))
pose = transform(top_down_rotation(np.pi), (0.0, -0.45, 0.15))
for q, branch in kin.inverse_all(pose):
    print(branch, np.degrees(q).round(1))
```

[examples/](examples/) 와 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 참고.

## 안전 메모

* 팔을 움직이는 모든 것은 **`--execute` 가 없으면 드라이 런**이고, 제어기 상태 변경은 `--yes` 가 필요하다.
* 모션은 제한된 서보 스트림 하나(기본 30°/s, 60°/s²)를 거치며 래치형 안전 감시(비상정지, 충돌, 안전
  입력, 에러 코드, 추종 편차, 토크 급변)를 받는다.
* 협조적 모션 잠금으로 두 스크립트가 동시에 팔을 움직이지 못한다.
* 충돌 모델은 없다. 비상정지 옆에 사람이 있어야 한다. 실제 로봇이 있는 ROS 그래프에서
  `fr3-driver --fake` 를 실행하지 말 것.

전체 목록: [docs/SAFETY.md](docs/SAFETY.md). 실기 노하우: [docs/PITFALLS.md](docs/PITFALLS.md).

## 검증 환경

FR3 제어기 펌웨어 v3.8.6, fairino-python-sdk v2.1.6, 제어기 경유 AG-95, Ubuntu 22.04 의 ROS 2 Humble,
Python 3.10-3.12. 다른 펌웨어도 맞는 SDK 태그로 동작할 가능성이 높다. 결과 보고 환영
([CONTRIBUTING.md](CONTRIBUTING.md)).

## 라이선스

MIT © sxngt. FAIRINO 와 DH-Robotics 는 각 소유자의 상표다.
