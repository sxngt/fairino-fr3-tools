# ROS 2 driver

`fr3-driver` is an `rclpy` node on top of the vendor Python SDK; it replaces ros2_control for the
FR3 (FAIRINO's ros2_control hardware packages ship a `libfairino` for other firmware). Tested on
ROS 2 Humble with firmware v3.8.6 and SDK v2.1.6.

`fr3-driver` 는 제조사 Python SDK 위의 rclpy 노드로 ros2_control 을 대신한다(Humble 검증).

```bash
source /opt/ros/humble/setup.bash        # rclpy and the message packages
fr3-driver --ip 192.168.58.2             # holds the state port and the controller's motion lock
```

`rclpy` and the messages come from the ROS install, not pip. Install `fr3tools` into a Python that
can see them (the system Python of the ROS distribution, or a venv created with
`--system-site-packages`).

## Interface / 인터페이스

| name | type | notes |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | j1..j6 (rad, rad/s, Nm) + `ag95_left/right_finger_joint` (half gap, m), 50 Hz; held when the state packet freezes |
| `/fr/status` | `std_msgs/String` (JSON) | decoded controller status + `driver: {servo_active, owner, servo_reason}`, 2 Hz |
| `/fr/servo_target` | `sensor_msgs/JointState` | latest joint target (rad) while the topic owns the servo stream |
| `/fr/servo_start`, `/fr/servo_stop` | `std_srvs/Trigger` | start (readiness checks) / end a servo session |
| `/fr/stop` | `std_srvs/Trigger` | end any session and `StopMotion` |
| `/fr/reset_error`, `/fr/enable`, `/fr/set_mode_auto` | `std_srvs/Trigger` | `ResetAllError`, `RobotEnable(1)`, `Mode(0)` |
| `/fr3_arm_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | MoveIt arm controller (see below) |
| `/ag95_gripper_controller/follow_joint_trajectory` | `control_msgs/FollowJointTrajectory` | last point's `ag95_left_finger_joint` = half gap (m) |

`--prefix` moves the `/fr/...` names; `--arm-action`, `--gripper-action` rename the actions;
`--no-arm-action` leaves only the servo stream.

### Parameters / 파라미터

Set at start (`--ros-args -p name:=value`) or live (`ros2 param set /fr3_driver name value`);
each new session reads them once.

| parameter | default | meaning |
|---|---|---|
| `max_joint_speed_deg` | 30 | stream speed limit and the trajectory peak-speed limit (deg/s) |
| `max_joint_accel_deg` | 60 | stream acceleration limit (deg/s²) |
| `start_tol_deg` | 2 | tracking deviation that trips the monitor; also the trajectory start tolerance |
| `servo_timeout` | 1.5 | s without a new target before the stream ends |
| `servo_max_jump_deg` | 15 | largest target-to-command difference accepted |
| `cmd_t` | 0.016 | ServoJ period (s) |
| `goal_tol_deg` | 1.0 | trajectory end tolerance (deg) |

## The trajectory action / 궤적 액션

Goals are checked before acceptance and again at execution: the joint names must be j1..j6
(any order), times must increase, the first point must be within `start_tol_deg` of the measured
joints, and no segment may exceed `max_joint_speed_deg`. A first point after `t = 0` gets the
measured joints prepended.

Execution feeds the trajectory, resampled on the 16 ms grid, as the target of the same limited
servo stream, under the same safety monitor; then it holds the last point until the arm is within
`goal_tol_deg` (1 s). Results: `SUCCESSFUL`, `GOAL_TOLERANCE_VIOLATED` (did not converge),
`PATH_TOLERANCE_VIOLATED` (safety trip, ServoJ error), `INVALID_GOAL`/`INVALID_JOINTS`. Cancel ends
the stream and sends `StopMotion`. Feedback (desired/actual/error) at 10 Hz.

**Mutual exclusion.** The action and the servo topic share one stream. A goal is rejected while a
topic session runs; `/fr/servo_start` fails while a trajectory runs and `/fr/servo_target` is
ignored then.

궤적 액션과 서보 토픽은 스트림 하나를 공유하며 동시에 쓸 수 없다.

### MoveIt configuration / MoveIt 설정

`moveit_controllers.yaml` of your MoveIt config package:

```yaml
moveit_controller_manager: moveit_simple_controller_manager/MoveItSimpleControllerManager
moveit_simple_controller_manager:
  controller_names: [fr3_arm_controller, ag95_gripper_controller]
  fr3_arm_controller:
    type: FollowJointTrajectory
    action_ns: follow_joint_trajectory
    default: true
    joints: [j1, j2, j3, j4, j5, j6]
  ag95_gripper_controller:
    type: FollowJointTrajectory
    action_ns: follow_joint_trajectory
    default: true
    joints: [ag95_left_finger_joint]
trajectory_execution:
  allowed_start_tolerance: 0.01
  allowed_execution_duration_scaling: 1.5
  allowed_goal_duration_margin: 2.0
```

Keep MoveIt's velocity scaling low enough that plans stay under `max_joint_speed_deg` (about 0.1
with the URDF limits of 3.15 rad/s), or the driver rejects them. See
[examples/moveit_trajectory.md](../examples/moveit_trajectory.md).

## Next to other processes / 다른 프로세스와 함께

* The driver holds the state port: SDK tools started meanwhile get an XML-RPC-only handle
  (commands work, status does not). Read the status from `/fr/status`.
* The driver holds the controller's motion lock: direct-SDK examples refuse to run.
* Clients (`fr3-teleop`, `fr3-handeye`, `fr3-probe`) hold the driver's motion lock while they
  stream, so two clients never feed the same stream.
* `--fake` runs without the SDK on a first-order arm model but publishes the same topics: use it
  only on an isolated graph (for example `ROS_DOMAIN_ID` of its own, or `ROS_LOCALHOST_ONLY=1` on a
  machine without a real robot).
