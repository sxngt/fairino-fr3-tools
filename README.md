# fairino-fr3-tools

**English** · [한국어](README.ko.md)

Tools for developers and researchers who use the **FAIRINO FR3** cobot with a **DH-Robotics AG-95**
gripper: the pieces the official SDK does not provide, learnt on a real arm.

> **Unofficial. Not affiliated with or endorsed by FAIRINO.** This project uses FAIRINO's Python
> SDK but is not part of it. It moves an industrial arm: read [docs/SAFETY.md](docs/SAFETY.md).

## Features

| | official Python SDK | fr3tools |
|---|---|---|
| Controller calls (XML-RPC, state packet) | yes | wraps it: one locked handle, state-port sharing, decoded status, error-code lookup |
| Kinematics | on the controller only (`GetForwardKin`, `GetInverseKin`) | offline DH FK, analytic IK with all 8 branches, Jacobian, branch ↔ controller `config` |
| DH calibration | - | `fr3-calib-dh`: fit the link parameters to controller poses (≈0.03 mm), offline from a log |
| Safe streaming | `ServoJ` | rate/acceleration-limited servo stream with braking, jump and stale checks, latching safety monitor |
| Preflight and recovery | - | `fr3-check` (READY / NOT READY with fixes), `fr3-power recover` |
| ROS 2 | ros2_control packages for other firmware | `fr3-driver`: `/joint_states`, servo stream, MoveIt `FollowJointTrajectory` (validated, cancelable), AG-95 action |
| AG-95 gripper | raw `MoveGripper` | activation, limits, width units, direct Modbus RTU, honest feedback notes |
| Two scripts at once | - | cooperative motion lock |
| Hand-eye calibration | - | `fr3-handeye`: active eye-to-hand with a gripper ArUco board, no initial guess, synthetic mode |
| Camera intrinsics | - | `fr3-camera-calib`: chessboard with blur and view-spread guards |
| Teleoperation | - | `fr3-teleop`: Xbox 360 pad, proportional top-down Cartesian control, deadman |
| Know-how | - | [docs/PITFALLS.md](docs/PITFALLS.md): 27 things that cost hours on the robot |

Everything that does not talk to a controller works without the SDK, ROS or a robot, and is
covered by tests.

## Install

```bash
git clone https://github.com/sxngt/fairino-fr3-tools.git
cd fairino-fr3-tools
pip install -e ".[vision]"          # or: uv sync --extra dev
```

Extras: `vision` (OpenCV: camera calibration, ArUco), `serial` (AG-95 over USB-RS485), `dev`
(tests, ruff). `ros` is empty on purpose: `rclpy` and the message packages come from your ROS 2
install (Humble tested), so source `/opt/ros/humble/setup.bash` and use a Python that sees them.

### The vendor SDK

The `fairino` package is **not on PyPI**. Use the tag of
[fairino-python-sdk](https://github.com/FAIR-INNOVATION/fairino-python-sdk) that matches your
controller firmware (tested: **SDK v2.1.6 with firmware v3.8.6**, tag `v2.1.6_robot3.8.6`):

```bash
git clone --branch v2.1.6_robot3.8.6 --depth 1 https://github.com/FAIR-INNOVATION/fairino-python-sdk.git
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/fairino" && touch "$SITE/fairino/__init__.py"
cp fairino-python-sdk/linux/fairino/Robot.py "$SITE/fairino/"   # or the prebuilt linux/libfairino/*.so
python -c "from fairino import Robot; print('SDK OK')"
```

`fr3-check` warns when the controller firmware differs from the tested one.

## Quick start

```bash
fr3-check                         # read-only preflight; exit 0 = READY
fr3-errors 29 --state 5 1         # what a return code / state code means
fr3-calib-dh --collect 40 --save dh_log.json --out fr3_params.json   # arm does not move

source /opt/ros/humble/setup.bash
fr3-driver                        # holds the state port; nothing moves until a client asks
fr3-probe                         # driver up? (dry)  then: fr3-probe --execute  (j6 ±0.3°)

python examples/move_joints.py --j6 5          # prints the plan; add --execute to move
fr3-teleop --fake --fake-pad                   # gamepad teleop demo without pad or robot
fr3-handeye --fake 0                           # eye-to-hand calibration on synthetic data
```

The controller address defaults to the factory `192.168.58.2`; set `FR3_IP` or `--ip`.

| command | what |
|---|---|
| `fr3-check` | preflight: e-stop, safety inputs, collision, error code, mode, enable, limits, versions |
| `fr3-power` | `status`, `stop`, `reset`; with `--yes`: `enable`, `auto`, `manual`, `drag`, `speed`, `recover` |
| `fr3-errors` | explain return codes and state (main, sub) codes |
| `fr3-calib-dh` | fit DH parameters; `--log` (CSV/JSON, offline) or `--collect N` |
| `fr3-driver` | ROS 2 driver ([docs/ROS2.md](docs/ROS2.md)) |
| `fr3-probe` | check the driver and, with `--execute`, that the servo stream moves the arm |
| `fr3-gripper` | AG-95: `status`, `configure`; with `--execute`: `activate`, `open`, `width`, `probe` |
| `fr3-camera-calib` | chessboard intrinsics (live or `--images`) |
| `fr3-handeye` | active eye-to-hand calibration (`--fake SEED`, or the robot with `--execute`) |
| `fr3-teleop` | gamepad teleoperation ([docs/TELEOP.md](docs/TELEOP.md)) |

## Library

```python
import numpy as np
from fr3tools.kinematics import FR3Kinematics, FR3Params, top_down_rotation, transform

kin = FR3Kinematics(FR3Params(tool_xyz=(0.0, 0.0, 0.2)))
pose = transform(top_down_rotation(np.pi), (0.0, -0.45, 0.15))
for q, branch in kin.inverse_all(pose):
    print(branch, np.degrees(q).round(1))
```

See [examples/](examples/) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Safety notes

* Everything that moves the arm is **dry run unless `--execute`**; controller state changes need
  `--yes`.
* Motion goes through one limited servo stream (30°/s, 60°/s² by default) under a latching safety
  monitor (e-stop, collision, safety inputs, error code, tracking deviation, torque jumps).
* A cooperative motion lock keeps two scripts from moving the arm at once.
* There is no collision model. Keep a person at the emergency stop. Never run `fr3-driver --fake`
  on a ROS graph shared with a real robot.

Full list: [docs/SAFETY.md](docs/SAFETY.md). Hard-won details: [docs/PITFALLS.md](docs/PITFALLS.md).

## Tested with

FR3 controller firmware v3.8.6, fairino-python-sdk v2.1.6, AG-95 through the controller, ROS 2
Humble on Ubuntu 22.04, Python 3.10-3.12. Other firmware probably works with the matching SDK tag;
reports are welcome ([CONTRIBUTING.md](CONTRIBUTING.md)).

## License

MIT © sxngt. FAIRINO and DH-Robotics are trademarks of their owners.
