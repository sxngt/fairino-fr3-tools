# MoveIt through fr3-driver

`fr3-driver` serves `/fr3_arm_controller/follow_joint_trajectory`, so MoveIt's
`moveit_simple_controller_manager` can execute plans on the FR3 without ros2_control.
`fr3-driver` 가 궤적 액션을 제공하므로 MoveIt 이 ros2_control 없이 FR3 에서 계획을 실행한다.

## 1. Configure the MoveIt package / MoveIt 패키지 설정

* URDF/SRDF: FAIRINO's `fairino_description` (`fairino3_v6`) has joints `j1..j6`; add the AG-95 if
  you plan with it (finger joint `ag95_left_finger_joint`, half gap in metres).
* `moveit_controllers.yaml`: see [docs/ROS2.md](../docs/ROS2.md#moveit-configuration).
* `joint_limits.yaml`: keep the default velocity scaling around 0.1 so plans stay under the
  driver's `max_joint_speed_deg` (30°/s); faster plans are rejected, not slowed down.

## 2. Run / 실행

```bash
source /opt/ros/humble/setup.bash
fr3-check                                   # READY?
fr3-driver                                  # terminal 1
ros2 launch <your_moveit_config> move_group.launch.py   # terminal 2 (plus rviz)
```

Plan and execute in rviz's MotionPlanning panel. Start with a small joint goal; watch
`/fr/status` (`driver.owner` is `trajectory` while it runs).

## 3. A goal without MoveIt / MoveIt 없이 목표 보내기

The action is a plain `control_msgs/FollowJointTrajectory`. This sends j6 +5° over 2 s and back;
the first point must be the current joints (within `start_tol_deg`):

```python
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

rclpy.init()
node = Node("fjt_example")
q = {}
node.create_subscription(JointState, "/joint_states", lambda m: q.update(zip(m.name, m.position)), 10)
while len(q) < 6:
    rclpy.spin_once(node, timeout_sec=0.1)
joints = ["j1", "j2", "j3", "j4", "j5", "j6"]
q0 = [q[j] for j in joints]
goal = FollowJointTrajectory.Goal()
goal.trajectory.joint_names = joints
for t, dj6 in ((0, 0.0), (2, 0.0873), (4, 0.0)):  # 5 deg = 0.0873 rad
    p = JointTrajectoryPoint(positions=q0[:5] + [q0[5] + dj6])
    p.time_from_start = Duration(sec=t)
    goal.trajectory.points.append(p)
client = ActionClient(node, FollowJointTrajectory, "/fr3_arm_controller/follow_joint_trajectory")
client.wait_for_server()
handle = client.send_goal_async(goal)
rclpy.spin_until_future_complete(node, handle)
result = handle.result().get_result_async()
rclpy.spin_until_future_complete(node, result)
print(result.result().result.error_code, result.result().result.error_string)
```

Peak speed here is 2.5°/s. A goal that starts away from the current joints, uses other joint
names, has non-increasing times or exceeds the speed limit is rejected before anything moves.
Cancelling the goal ends the stream and sends `StopMotion`.

## 4. While a trajectory runs / 궤적 실행 중

* `/fr/servo_start` fails and `/fr/servo_target` is ignored (mutual exclusion).
* A safety trip aborts the goal with `PATH_TOLERANCE_VIOLATED` and the reason in `error_string`.
* After a trip, look at the arm, then `ros2 service call /fr/reset_error std_srvs/srv/Trigger`.
