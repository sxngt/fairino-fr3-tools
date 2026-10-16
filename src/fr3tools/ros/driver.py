"""ROS 2 driver for the FAIRINO FR3 with an AG-95 gripper (``fr3-driver``).

Interface (names are defaults; see ``--help``):

* publishes ``/joint_states`` (``sensor_msgs/JointState``, 50 Hz): j1..j6 (rad, rad/s, Nm) plus
  the two AG-95 finger joints (half gap, m), from the controller's real-time state packet;
* publishes ``/fr/status`` (``std_msgs/String``, JSON, 2 Hz): decoded controller status;
* subscribes ``/fr/servo_target`` (``sensor_msgs/JointState``, j1..j6 rad): the latest target of
  the servo stream;
* services (``std_srvs/Trigger``): ``/fr/servo_start``, ``/fr/servo_stop``, ``/fr/reset_error``,
  ``/fr/stop``, ``/fr/enable``, ``/fr/set_mode_auto``;
* action ``/fr3_arm_controller/follow_joint_trajectory`` (``control_msgs/FollowJointTrajectory``),
  for MoveIt: goals are validated (joint names, increasing time, peak joint speed, start within
  ``start_tol_deg`` of the measured joints) and executed through the same limited servo stream and
  safety monitor; cancel ends the stream and sends ``StopMotion``. The action and the servo topic
  are mutually exclusive: whichever starts first owns the stream until it ends;
* action ``/ag95_gripper_controller/follow_joint_trajectory``
  (``control_msgs/FollowJointTrajectory``): the last point's finger value is the half gap.

Parameters (``--ros-args -p name:=value`` or ``ros2 param set``): ``start_tol_deg`` (safety
tracking deviation), ``max_joint_speed_deg``, ``max_joint_accel_deg``, ``servo_timeout``,
``servo_max_jump_deg``, ``cmd_t``; see :mod:`fr3tools.servo`.

The driver never moves the arm on its own: motion happens only inside a servo session, which starts
only in AUTO mode, servo enabled, without safety stop or error, and every SDK call goes through one
lock (the SDK's XML-RPC proxy is not thread safe). The driver holds the motion lock of its
controller, so direct-SDK scripts refuse to run next to it. ``--fake`` runs without the SDK on a
first-order arm model; note that it still publishes on the ROS graph.

FR3 + AG-95 용 ROS 2 드라이버. 드라이버는 스스로 팔을 움직이지 않는다: 서보 세션 안에서만 움직이고,
세션은 AUTO·서보 enable·안전 정상·에러 없음에서만 시작한다. 모든 SDK 호출은 하나의 잠금을 거친다.
``--fake`` 는 SDK 없이 1차 지연 모델로 돌지만 ROS 그래프에는 그대로 발행한다.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np

try:
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
except ImportError as exc:  # pragma: no cover - needs a ROS 2 install
    raise ImportError(
        "fr3-driver needs ROS 2 (rclpy, control_msgs, std_srvs); source /opt/ros/<distro>/setup.bash first"
    ) from exc

from fr3tools import sdk
from fr3tools.gripper import STROKE_M, ControllerGripper, FakeGripper
from fr3tools.kinematics import JOINTS as JOINT_TUPLE
from fr3tools.lock import MotionLock
from fr3tools.servo import (
    FakeArm,
    ServoParams,
    ServoStream,
    TrajectoryError,
    TrajectoryExecutor,
    validate_trajectory,
)

JOINTS = list(JOINT_TUPLE)
FINGERS = ["ag95_left_finger_joint", "ag95_right_finger_joint"]
GRIP_ACTION = "/ag95_gripper_controller/follow_joint_trajectory"
# A pose with the arm folded above the table, for --fake. / --fake 시작 자세.
FAKE_START_DEG = (60.0, -70.0, 65.0, -85.0, -90.0, 0.0)


class FR3Driver(Node):
    """The driver node. / 드라이버 노드."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("fr3_driver")
        defaults = ServoParams()
        self.declare_parameter("start_tol_deg", defaults.start_tol_deg)
        self.declare_parameter("max_joint_speed_deg", args.max_joint_speed_deg)
        self.declare_parameter("max_joint_accel_deg", defaults.max_joint_accel_deg)
        self.declare_parameter("servo_timeout", defaults.servo_timeout)
        self.declare_parameter("servo_max_jump_deg", defaults.servo_max_jump_deg)
        self.declare_parameter("cmd_t", defaults.cmd_t)

        self.fake = args.fake
        self.prefix = args.prefix.rstrip("/")
        self.cb = ReentrantCallbackGroup()
        log = self.get_logger()
        self.lock = None
        if args.fake:
            self.arm = FakeArm(args.start_deg)
            self.gripper = FakeGripper()
        else:
            # Hold the motion lock of this controller for the driver's lifetime.
            # 드라이버가 사는 동안 이 제어기의 모션 잠금을 잡는다.
            self.lock = MotionLock(args.ip).acquire()
            self.arm = sdk.SdkArm(args.ip)
            self.gripper = ControllerGripper(self.arm, open_pct=args.gripper_open_pct, log=log.info)
        self.stream = ServoStream(self.arm, self.servo_params, log=log.info, warn=log.warning)
        # Who feeds the stream: "topic" (/fr/servo_target) or "trajectory" (the arm action).
        # 스트림에 목표를 주는 쪽: 토픽 또는 궤적 액션. 둘은 동시에 쓸 수 없다.
        self.owner = ""
        self._q: np.ndarray | None = None
        self._stale_warned = 0.0
        self._ticks = 0

        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_status = self.create_publisher(String, f"{self.prefix}/status", 10)
        self.create_subscription(
            JointState, f"{self.prefix}/servo_target", self._on_target, 10, callback_group=self.cb
        )
        services = {
            "servo_start": self._srv_servo_start,
            "servo_stop": self._srv_servo_stop,
            "stop": self._srv_stop,
            "reset_error": self._srv_reset,
            "enable": self._srv_enable,
            "set_mode_auto": self._srv_auto,
        }
        for name, handler in services.items():
            self.create_service(Trigger, f"{self.prefix}/{name}", handler, callback_group=self.cb)
        self._grip_server = ActionServer(
            self,
            FollowJointTrajectory,
            args.gripper_action,
            execute_callback=self._grip_execute,
            goal_callback=self._grip_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=self.cb,
        )
        self._setup_arm_action(args)
        self.create_timer(1.0 / args.rate, self._tick_joints, callback_group=self.cb)
        self.create_timer(0.5, self._tick_status, callback_group=self.cb)
        log.info(
            f"{'FAKE' if args.fake else args.ip}: /joint_states {args.rate:.0f} Hz, servo stream on "
            f"{self.prefix}/servo_target, gripper action {args.gripper_action}"
        )

    def _setup_arm_action(self, args) -> None:
        """Arm FollowJointTrajectory action for MoveIt. / MoveIt 용 팔 궤적 액션."""
        import threading

        self.declare_parameter("goal_tol_deg", args.goal_tol_deg)
        self._goal_handle = None
        self._desired: np.ndarray | None = None
        self._cancel = threading.Event()
        if args.no_arm_action:
            return
        self._arm_server = ActionServer(
            self,
            FollowJointTrajectory,
            args.arm_action,
            execute_callback=self._arm_execute,
            goal_callback=self._arm_goal,
            cancel_callback=self._arm_cancel,
            callback_group=self.cb,
        )
        self.get_logger().info(f"arm trajectory action {args.arm_action}")

    def _check_goal(self, trajectory):
        """Validate against the measured joints; returns (t, Q). / 현재 관절 기준 검증."""
        q_now = self.current_q()
        if q_now is None:
            raise TrajectoryError("invalid_goal", "no joint state")
        return validate_trajectory(
            trajectory.joint_names,
            [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in trajectory.points],
            [list(p.positions) for p in trajectory.points],
            q_now,
            JOINTS,
            float(self.get_parameter("max_joint_speed_deg").value),
            float(self.get_parameter("start_tol_deg").value),
        )

    def _arm_goal(self, request):
        if self.stream.active or self._goal_handle is not None:
            self.get_logger().warning(f"trajectory rejected: servo stream busy ({self.owner})")
            return GoalResponse.REJECT
        try:
            self._check_goal(request.trajectory)
        except TrajectoryError as exc:
            self.get_logger().warning(f"trajectory rejected: {exc}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _arm_cancel(self, goal_handle):
        self.get_logger().warning("trajectory cancel requested: stopping")
        self._cancel.set()
        return CancelResponse.ACCEPT

    def _arm_execute(self, goal_handle):
        import threading

        result = FollowJointTrajectory.Result()
        codes = {
            "successful": FollowJointTrajectory.Result.SUCCESSFUL,
            "invalid_goal": FollowJointTrajectory.Result.INVALID_GOAL,
            "invalid_joints": FollowJointTrajectory.Result.INVALID_JOINTS,
            "path_tolerance": FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
            "goal_tolerance": FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
        }
        try:
            t, q = self._check_goal(goal_handle.request.trajectory)
        except TrajectoryError as exc:
            result.error_code, result.error_string = codes[exc.code], str(exc)
            goal_handle.abort()
            return result
        self._cancel = threading.Event()
        self._goal_handle = goal_handle
        executor = TrajectoryExecutor(
            self.stream,
            start=lambda: self.start_stream("trajectory"),
            current_q=lambda: self.current_q(0.2),
            stop_arm=None if self.fake else self.arm.stop,
        )

        def on_sample(target) -> None:
            self._desired = target
            if goal_handle.is_cancel_requested:
                self._cancel.set()

        self.get_logger().info(f"trajectory: {len(t)} points, {t[-1]:.2f} s")
        try:
            out = executor.run(
                t, q, self._cancel, float(self.get_parameter("goal_tol_deg").value), on_sample=on_sample
            )
        finally:
            self._goal_handle = None
            self._desired = None
            self.owner = ""
        result.error_code, result.error_string = codes[out.code], out.message
        if out.status == "succeeded":
            goal_handle.succeed()
            self.get_logger().info(f"trajectory done: {out.message}")
        elif out.status == "canceled":
            goal_handle.canceled()
            self.get_logger().warning(f"trajectory canceled after {out.samples} samples")
        else:
            goal_handle.abort()
            self.get_logger().error(f"trajectory aborted: {out.message}")
        return result

    # ---- parameters / 파라미터
    def servo_params(self) -> ServoParams:
        def value(name: str) -> float:
            return float(self.get_parameter(name).value)

        return ServoParams(
            cmd_t=value("cmd_t"),
            max_joint_speed_deg=value("max_joint_speed_deg"),
            max_joint_accel_deg=value("max_joint_accel_deg"),
            servo_timeout=value("servo_timeout"),
            servo_max_jump_deg=value("servo_max_jump_deg"),
            start_tol_deg=value("start_tol_deg"),
        )

    # ---- state / 상태
    def _tick_joints(self) -> None:
        try:
            q, qd, tau, fresh = self.arm.sample()
        except Exception as exc:
            self.get_logger().warning(f"state read failed: {exc}", throttle_duration_sec=5.0)
            return
        if not fresh:
            # Publishing a frozen pose would let a client servo against stale joints.
            # 멈춘 자세를 내보내면 클라이언트가 낡은 관절로 서보한다.
            if time.monotonic() - self._stale_warned > 5.0:
                self._stale_warned = time.monotonic()
                self.get_logger().warning("real-time state packets stopped: holding /joint_states")
            return
        self._q = q
        try:
            half_gap = float(self.gripper.half_gap())
        except Exception:
            half_gap = STROKE_M / 2.0
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS + FINGERS
        msg.position = q.tolist() + [half_gap, half_gap]
        msg.velocity = qd.tolist() + [0.0, 0.0]
        msg.effort = tau.tolist() + [0.0, 0.0]
        self.pub_js.publish(msg)
        self._ticks += 1
        self._publish_feedback(q)

    def _publish_feedback(self, q: np.ndarray) -> None:
        """Trajectory feedback at 10 Hz. / 궤적 피드백(10 Hz)."""
        handle, desired = getattr(self, "_goal_handle", None), getattr(self, "_desired", None)
        if handle is None or desired is None or self._ticks % 5:
            return
        from trajectory_msgs.msg import JointTrajectoryPoint

        with contextlib.suppress(Exception):
            fb = FollowJointTrajectory.Feedback()
            fb.joint_names = JOINTS
            fb.desired = JointTrajectoryPoint(positions=[float(v) for v in desired])
            fb.actual = JointTrajectoryPoint(positions=q.tolist())
            fb.error = JointTrajectoryPoint(positions=(np.asarray(desired) - q).tolist())
            handle.publish_feedback(fb)

    def _tick_status(self) -> None:
        if self.fake:
            q = self._q if self._q is not None else np.zeros(6)
            d = {"ok": True, "fake": True, "mode": "AUTO", "jnt_ang": np.degrees(q).round(4).tolist()}
        else:
            try:
                d = self.arm.status().to_dict()
                d["fake"] = False
            except Exception as exc:
                d = {"ok": False, "error": str(exc), "fake": False}
        d["driver"] = {
            "servo_active": self.stream.active,
            "owner": self.owner,
            "servo_reason": self.stream.reason,
        }
        self.pub_status.publish(String(data=json.dumps(d, default=str)))

    def current_q(self, wait: float = 1.0) -> np.ndarray | None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait:
            if self._q is not None:
                return self._q.copy()
            time.sleep(0.02)
        return None

    # ---- servo stream / 서보 스트림
    def _on_target(self, msg) -> None:
        if self.owner != "topic":
            return  # the trajectory action owns the stream / 궤적 액션이 스트림 소유
        try:
            d = dict(zip(msg.name, msg.position, strict=False))
            q = np.array([float(d[j]) for j in JOINTS])
        except KeyError:
            return
        self.stream.set_target(q)

    def start_stream(self, owner: str) -> tuple[bool, str]:
        """Start a servo session for ``owner`` if the arm is free and ready. / 서보 세션 시작."""
        if self.stream.active:
            return False, f"busy: servo stream owned by {self.owner or '?'}"
        q0 = self.current_q(1.0)
        if q0 is None:
            return False, "no joint state"
        ok, why = self.arm.ready_for_motion()
        if not ok:
            return False, f"not ready: {why}"
        self.owner = owner
        self.stream.start(q0)
        return True, f"servo stream started ({owner})"

    def _srv_servo_start(self, request, response):
        response.success, response.message = self.start_stream("topic")
        self.get_logger().info(response.message)
        return response

    def _srv_servo_stop(self, request, response):
        if self.owner == "trajectory":
            response.success, response.message = False, "the trajectory action owns the stream; cancel it"
            return response
        reason = self.stream.stop()
        response.success, response.message = True, f"servo stream stopped ({reason or 'requested'})"
        return response

    # ---- controller services / 제어기 서비스
    def _rpc(self, what: str, fn, response):
        if self.fake:
            response.success, response.message = True, f"fake: {what} no-op"
            return response
        try:
            err = fn()
            response.success = err == 0
            response.message = f"{what} -> {'OK' if err == 0 else f'error {err}'}"
        except Exception as exc:
            response.success, response.message = False, f"{what} failed: {exc}"
        self.get_logger().info(response.message)
        return response

    def _srv_stop(self, request, response):
        self.stream.stop(timeout=0.5)
        return self._rpc("StopMotion", self.arm.stop, response)

    def _srv_reset(self, request, response):
        return self._rpc("ResetAllError", lambda: self.arm.call("ResetAllError"), response)

    def _srv_enable(self, request, response):
        return self._rpc("RobotEnable(1)", lambda: self.arm.call("RobotEnable", 1), response)

    def _srv_auto(self, request, response):
        return self._rpc("Mode(0)", lambda: self.arm.call("Mode", 0), response)

    # ---- gripper action / 그리퍼 액션
    def _grip_goal(self, request):
        trajectory = request.trajectory
        if FINGERS[0] not in trajectory.joint_names or not trajectory.points:
            self.get_logger().warning("gripper goal rejected: joint name or empty trajectory")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _grip_execute(self, goal_handle):
        result = FollowJointTrajectory.Result()
        trajectory = goal_handle.request.trajectory
        index = list(trajectory.joint_names).index(FINGERS[0])
        half_gap = float(trajectory.points[-1].positions[index])
        try:
            reported = self.gripper.move(half_gap)
            self.get_logger().info(f"gripper -> {2000 * half_gap:.1f} mm (reported {2000 * reported:.1f} mm)")
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = f"width {2000 * reported:.1f} mm"
            goal_handle.succeed()
        except Exception as exc:
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = f"gripper error: {exc}"
            self.get_logger().error(result.error_string)
            goal_handle.abort()
        return result

    def shutdown(self) -> None:
        """Stop any stream; StopMotion if one was running. / 스트림 정지."""
        was_active = self.stream.active
        self.stream.stop(timeout=1.0)
        if was_active and not self.fake:
            with contextlib.suppress(Exception):
                self.arm.stop()
                self.get_logger().warning("shutdown during a servo stream: StopMotion sent")
        if self.lock is not None:
            self.lock.release()


def parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(description="FR3 ROS 2 driver / FR3 ROS 2 드라이버")
    ap.add_argument("--ip", default=sdk.DEFAULT_IP, help="controller IP (env FR3_IP)")
    ap.add_argument("--fake", action="store_true", help="no SDK, no robot: first-order arm model")
    ap.add_argument("--rate", type=float, default=50.0, help="/joint_states rate (Hz)")
    ap.add_argument("--prefix", default="/fr", help="namespace of the servo topic, status and services")
    ap.add_argument("--max-joint-speed-deg", type=float, default=30.0)
    ap.add_argument("--arm-action", default="/fr3_arm_controller/follow_joint_trajectory")
    ap.add_argument("--no-arm-action", action="store_true", help="servo stream only")
    ap.add_argument("--goal-tol-deg", type=float, default=1.0, help="trajectory end tolerance (deg)")
    ap.add_argument("--gripper-action", default=GRIP_ACTION)
    ap.add_argument(
        "--start-deg", type=float, nargs=6, default=list(FAKE_START_DEG), metavar="DEG", help="--fake start"
    )
    ap.add_argument(
        "--gripper-open-pct",
        type=int,
        default=100,
        choices=(0, 100),
        help="controller gripper position that means fully open",
    )
    return ap.parse_known_args(argv)


def main(argv=None) -> None:
    args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = None
    executor = None
    try:
        node = FR3Driver(args)
        executor = MultiThreadedExecutor(num_threads=6)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.shutdown()
        with contextlib.suppress(Exception):
            if executor is not None:
                executor.shutdown(timeout_sec=0.5)
            if node is not None:
                node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
