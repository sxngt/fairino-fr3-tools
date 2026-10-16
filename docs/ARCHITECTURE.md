# Architecture

`fr3tools` is one Python package in layers. Lower layers never import higher ones, and only the
layers marked *SDK*, *ROS* or *OpenCV* need those dependencies (imported lazily).

`fr3tools` 는 계층형 패키지 하나다. 아래 계층은 위를 import 하지 않고, SDK·ROS·OpenCV 는 표시한
계층에서만 지연 import 한다.

```
                 fr3-teleop          fr3-handeye        examples/
                 teleop/run          calib/run
                     |                   |
   teleop/controller (pure)    calib/handeye, poses, simulate (NumPy)    calib/camera (OpenCV)
   teleop/gamepad, mapping             |
                     \                 |
                      ros/adapter: FR3Ros (ROS client) | FakeRobot (no ROS)
                                 |
              /joint_states, /fr/servo_target, /fr/servo_start|stop, actions
                                 |
                      ros/driver: FR3Driver (ROS)  <- MoveIt FollowJointTrajectory
                                 |
           servo: ServoStream, TrajectoryExecutor      safety: SafetyMonitor, preflight
                                 |
               sdk: SdkArm, connect, RobotStatus (SDK)     gripper: ControllerGripper, ModbusGripper
                                 |
                  errors, kinematics, lock (pure)
```

## Modules / 모듈

| module | role | needs |
|---|---|---|
| `kinematics` | DH forward kinematics, analytic IK (8 branches), Jacobian, controller `config`, controller pose convention, DH fit | NumPy |
| `errors` | return-code and state-code lookup, `check()`, `FR3Error` | - |
| `sdk` | connection (state-port handling), `RobotStatus`, `SdkArm` (locked XML-RPC, ServoJ session) | `fairino` |
| `safety` | `SafetyMonitor` (flags, tracking deviation, torque jumps; latching), `preflight` | - |
| `servo` | `servo_step` limiter, `ServoStream` (one session, rate/accel/brake, jump and stale checks), trajectory validation and `TrajectoryExecutor`, `FakeArm` | - |
| `gripper` | AG-95 through the controller or Modbus RTU, `FakeGripper`, `fr3-gripper` | `fairino` or `pyserial` |
| `lock` | cooperative cross-process `MotionLock` (`flock`) | POSIX |
| `tools` | `fr3-check`, `fr3-power`, `fr3-errors`, `fr3-calib-dh`, `fr3-probe` | per command |
| `ros/driver` | `fr3-driver`: state publishing, servo stream, MoveIt arm action, gripper action, services | ROS 2, `fairino` |
| `ros/adapter` | `FR3Ros` client and `RosGripper`; `FakeRobot` with the same interface | ROS 2 (not `FakeRobot`) |
| `calib/*` | intrinsics, active eye-to-hand solver, pose generation, synthetic rig, `fr3-handeye` | NumPy, OpenCV for real cameras |
| `teleop/*` | pad reader, profiles, pure controller, `fr3-teleop` | - |

## The motion path / 모션 경로

Every motion in this package goes through one `ServoStream` in the process that owns the state
port, normally `fr3-driver`:

1. A client publishes sparse joint targets (or MoveIt sends a trajectory, which the
   `TrajectoryExecutor` feeds sample by sample).
2. Each 16 ms tick the stream moves the command toward the latest target under speed,
   acceleration and braking limits, and sends `ServoJ`.
3. A `SafetyMonitor` thread reads the state packet at 20 Hz and compares it to the command. A trip
   ends the stream and sends `StopMotion`; it latches.
4. The stream ends on request, on a stale target, on a target jump, on a trip or on an SDK error;
   only one session can exist, and either the topic or the action owns it.

모든 모션은 상태 포트를 가진 프로세스(보통 드라이버)의 서보 스트림 하나를 거친다. 매 틱 속도·가속도·제동
제한 후 ServoJ, 안전 감시가 20 Hz 로 비교·정지. 세션은 하나이고 토픽 또는 액션 중 하나만 소유한다.

## Threads and locks / 스레드와 잠금

* `SdkArm.lock` serialises every XML-RPC call (servo ticks, gripper, services). `StopMotion` waits
  at most 0.5 s for it.
* `ServoStream` runs one thread per session; targets are swapped under a small lock.
* `SafetyMonitor` runs its own thread; `on_trip` ends the stream before `stop()` is sent.
* `MotionLock` (a file lock per controller or per driver prefix) keeps two scripts from commanding
  the same arm. The driver holds the controller's lock for its lifetime; ROS clients hold the
  driver's lock while they stream.

## Testing without hardware / 하드웨어 없는 시험

`FakeArm` (first-order lag) stands in for `SdkArm` under `ServoStream`; `FakeRobot` stands in for
`FR3Ros`; `ScriptedPad` for the gamepad; `calib.simulate.Rig` for the camera and the true
camera/board poses; `SimClock` runs calibration sessions in simulated time. The test suite needs
none of ROS, the SDK, a camera or a robot.
