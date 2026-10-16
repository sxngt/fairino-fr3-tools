"""Client side of the driver: stream joint or TCP targets over ROS 2. / 드라이버 클라이언트.

:class:`FR3Ros` talks to ``fr3-driver``: it reads ``/joint_states`` and streams targets on
``/fr/servo_target`` between ``/fr/servo_start`` and ``/fr/servo_stop``. Rate limiting and the
safety monitor stay in the driver; this side does kinematics and message plumbing only.
:class:`FakeRobot` has the same interface without ROS (instant motion), for dry runs and tests.

All robot clients in this package (teleop, hand-eye) take a cooperative :class:`MotionLock` while
they stream, so two of them never feed the same driver.

드라이버 클라이언트: ``/joint_states`` 를 읽고 서보 스트림에 목표를 보낸다(속도 제한·안전 감시는
드라이버 몫). :class:`FakeRobot` 은 ROS 없는 같은 인터페이스. 스트리밍 중에는 협조적 모션 잠금을 잡는다.
"""

from __future__ import annotations

import json
import time

import numpy as np

from fr3tools.gripper import STROKE_M
from fr3tools.kinematics import JOINTS, FR3Kinematics, FR3Params
from fr3tools.lock import MotionLock


class _Base:
    """Shared IK helpers. / 공통 IK 도우미."""

    kin: FR3Kinematics
    branch: tuple[int, int, int] | None

    def tcp_pose(self) -> np.ndarray:
        return self.kin.tcp(self.joint_positions())

    def solve_ik(self, tcp_pose: np.ndarray, q_ref=None, max_step_deg: float = 20.0) -> np.ndarray | None:
        """IK on the fixed branch, wrapped next to ``q_ref`` (default: the current joints).

        ``None`` if unreachable or if any joint would move more than ``max_step_deg`` from ``q_ref``
        (a branch flip or a wrist turn; never stream that).
        고정 가지 IK(``q_ref`` 옆으로 감음). 도달 불가 또는 관절이 ``max_step_deg`` 넘게 튀면 None.
        """
        ref = self.joint_positions() if q_ref is None else np.asarray(q_ref, dtype=float)
        q = self.kin.inverse(tcp_pose, branch=self.branch, q_ref=ref)
        if q is None or np.degrees(np.abs(q - ref)).max() > max_step_deg:
            return None
        return q


class FR3Ros(_Base):
    """Streaming FR3 over ``fr3-driver``. / ``fr3-driver`` 위의 스트리밍 FR3.

    Parameters
    ----------
    node
        An ``rclpy`` node owned by the caller (spun by the caller, or by :meth:`spin`).
    params, branch
        Kinematic parameters and the IK branch to keep (``None``: nearest solution).
    prefix
        The driver's ``--prefix``.
    """

    def __init__(self, node, params: FR3Params | None = None, branch=None, prefix: str = "/fr"):
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self.node = node
        self.kin = FR3Kinematics(params)
        self.branch = None if branch is None else tuple(int(v) for v in branch)
        self.prefix = prefix.rstrip("/")
        self.status: dict = {}
        self._q: np.ndarray | None = None
        self._q_t = 0.0
        self._msg = JointState
        self._lock = MotionLock(f"ros{self.prefix.replace('/', '_')}")
        node.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        node.create_subscription(String, f"{self.prefix}/status", self._on_status, 10)
        self._pub = node.create_publisher(JointState, f"{self.prefix}/servo_target", 10)
        self._start = node.create_client(Trigger, f"{self.prefix}/servo_start")
        self._stop = node.create_client(Trigger, f"{self.prefix}/servo_stop")
        self._reset = node.create_client(Trigger, f"{self.prefix}/reset_error")

    def _on_joints(self, msg) -> None:
        pos = dict(zip(msg.name, msg.position, strict=False))
        if all(j in pos for j in JOINTS):
            self._q = np.array([pos[j] for j in JOINTS], dtype=float)
            self._q_t = time.monotonic()

    def _on_status(self, msg) -> None:
        try:
            self.status = json.loads(msg.data)
        except ValueError:
            pass

    def spin(self, seconds: float = 0.0) -> None:
        import rclpy

        rclpy.spin_once(self.node, timeout_sec=seconds)

    def wait_ready(self, timeout: float = 10.0) -> np.ndarray:
        t0 = time.time()
        while self._q is None and time.time() - t0 < timeout:
            self.spin(0.1)
        if self._q is None:
            raise RuntimeError("no /joint_states (is fr3-driver running?)")
        return self._q.copy()

    @property
    def state_age(self) -> float:
        """Seconds since the last joint state. / 마지막 관절 상태 이후 경과 시간."""
        return time.monotonic() - self._q_t if self._q is not None else float("inf")

    @property
    def streaming(self) -> bool:
        return bool(self.status.get("driver", {}).get("servo_active", False))

    def joint_positions(self) -> np.ndarray:
        self.spin(0.0)
        if self._q is None:
            raise RuntimeError("no joint state yet")
        return self._q.copy()

    def stream_joints(self, q) -> None:
        msg = self._msg()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(JOINTS)
        msg.position = [float(v) for v in q]
        self._pub.publish(msg)

    def _call(self, client, timeout: float = 5.0):
        import rclpy

        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"{client.srv_name} unavailable (fr3-driver running?)")
        future = client.call_async(client.srv_type.Request())
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        return future.result()

    def start_streaming(self) -> None:
        """Take the motion lock, reset errors, start the servo stream. / 잠금·리셋 후 스트림 시작.

        A stopped controller ignores ServoJ until its error state is reset, and the driver's error
        flag lags the reset by one status cycle, so this retries briefly.
        정지 상태의 제어기는 에러 리셋 전까지 ServoJ 를 무시하고, 드라이버의 에러 플래그는 한 상태
        주기 늦게 풀리므로 잠시 재시도한다.
        """
        if not self._lock.held:
            self._lock.acquire()
        result = None
        for _ in range(4):
            self._call(self._reset)
            time.sleep(0.5)
            result = self._call(self._start)
            if result is not None and result.success:
                return
        self._lock.release()
        raise RuntimeError(f"servo stream refused: {None if result is None else result.message}")

    def stop_streaming(self) -> None:
        try:
            self._call(self._stop)
        finally:
            self._lock.release()


class RosGripper:
    """Gripper through the driver's FollowJointTrajectory action. / 드라이버 액션 경유 그리퍼.

    The driver owns the only SDK handle that can drive the AG-95 (it activates the gripper after a
    controller power cycle and clamps the width). Commands are non-blocking by default so a servo
    stream keeps flowing.
    """

    def __init__(self, node, action: str = "/ag95_gripper_controller/follow_joint_trajectory"):
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient

        self.node = node
        self._msg = FollowJointTrajectory
        self._client = ActionClient(node, FollowJointTrajectory, action)
        self._last = STROKE_M
        if not self._client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("gripper action server unavailable (fr3-driver running?)")

    def move_width(self, width: float, blocking: bool = False) -> None:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = self._msg.Goal()
        goal.trajectory.joint_names = ["ag95_left_finger_joint"]
        point = JointTrajectoryPoint()
        point.positions = [float(width) / 2.0]
        point.time_from_start = Duration(sec=1)
        goal.trajectory.points = [point]
        self._last = float(width)
        future = self._client.send_goal_async(goal)
        if blocking:
            import rclpy

            rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)
            handle = future.result()
            if handle is not None and handle.accepted:
                rclpy.spin_until_future_complete(self.node, handle.get_result_async(), timeout_sec=8.0)

    def width(self) -> float:
        return self._last


class FakeRobot(_Base):
    """Kinematic stand-in for :class:`FR3Ros`: every target is reached at once. / 가짜 로봇."""

    def __init__(self, q0, params: FR3Params | None = None, branch=None):
        self.kin = FR3Kinematics(params)
        self.branch = None if branch is None else tuple(int(v) for v in branch)
        self.q = np.asarray(q0, dtype=float).copy()
        self.streamed = 0
        self.streaming = False
        self.status = {"ok": True, "fake": True}

    def wait_ready(self, timeout: float = 0.0) -> np.ndarray:
        return self.q.copy()

    def spin(self, seconds: float = 0.0) -> None:
        if seconds:
            time.sleep(seconds)

    @property
    def state_age(self) -> float:
        return 0.0

    def joint_positions(self) -> np.ndarray:
        return self.q.copy()

    def stream_joints(self, q) -> None:
        if not self.streaming:
            raise RuntimeError("stream_joints outside a servo session")
        self.q = np.asarray(q, dtype=float).copy()
        self.streamed += 1

    def start_streaming(self) -> None:
        self.streaming = True

    def stop_streaming(self) -> None:
        self.streaming = False
