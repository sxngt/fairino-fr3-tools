"""ROS-free parts of the client adapter. / 클라이언트 어댑터의 ROS 없는 부분."""

import importlib
import sys

import numpy as np
import pytest

from fr3tools.ros.adapter import FakeRobot

Q0 = np.radians([60.0, -70.0, 65.0, -85.0, -90.0, 0.0])


def test_fake_robot_streams_only_inside_a_session():
    robot = FakeRobot(Q0, branch=None)
    with pytest.raises(RuntimeError):
        robot.stream_joints(Q0)
    robot.start_streaming()
    robot.stream_joints(Q0 + 0.01)
    assert np.allclose(robot.joint_positions(), Q0 + 0.01) and robot.streamed == 1


def test_solve_ik_keeps_continuity_and_rejects_jumps():
    robot = FakeRobot(Q0)
    pose = robot.tcp_pose()
    pose[:3, 3] += (0.01, 0.0, -0.01)
    q = robot.solve_ik(pose)
    assert q is not None and np.degrees(np.abs(q - Q0)).max() < 5.0
    far = robot.tcp_pose()
    far[:3, 3] += (0.0, 0.3, 0.0)
    assert robot.solve_ik(far, max_step_deg=5.0) is None


def test_driver_import_explains_missing_ros(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", None)
    sys.modules.pop("fr3tools.ros.driver", None)
    with pytest.raises(ImportError, match="ROS 2"):
        importlib.import_module("fr3tools.ros.driver")
