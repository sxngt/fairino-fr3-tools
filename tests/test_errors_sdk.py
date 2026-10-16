"""Error lookup and state-packet decoding (no SDK). / 에러 조회와 상태 패킷 해석."""

import sys
from types import SimpleNamespace

import pytest

from fr3tools import errors, sdk


def test_lookup_known_codes_from_each_source():
    assert errors.lookup(-4).source == "sdk"
    info = errors.lookup(29)
    assert info.source == "controller" and info.observed and "joint limit" in info.meaning
    assert errors.lookup(-1064).source == "ros_hw"
    assert errors.lookup(12345) is None


def test_describe_texts():
    assert "joint limit" in errors.describe(29, "ServoJ")
    assert errors.describe(29, "ServoJ").startswith("ServoJ -> ")
    assert "unknown" in errors.describe(777)
    assert errors.describe_state(0, 0) == "no error"
    assert "e-stop" in errors.describe_state(5, 1) or "emergency" in errors.describe_state(5, 1)
    assert "unknown" in errors.describe_state(9, 9)


def test_check_raises_with_meaning_and_returns_payload():
    assert errors.check(0) is None
    assert errors.check((0, [1.0, 2.0])) == [1.0, 2.0]
    assert errors.check((0, 1, 2)) == (1, 2)
    with pytest.raises(errors.FR3Error) as exc:
        errors.check(29, "ServoJ")
    assert exc.value.code == 29 and exc.value.info is not None
    with pytest.raises(errors.FR3Error):
        errors.check((-4, None), "GetForwardKin")


def test_status_from_packet_decodes_fields():
    pkg = SimpleNamespace(
        robot_mode=0,
        robot_state=2,
        EmergencyStop=0,
        collisionState=1,
        main_code=0,
        sub_code=0,
        jt_cur_pos=[1, 2, 3, 4, 5, 6],
        jt_cur_tor=[0] * 6,
        gripper_active=1,
        gripper_position=40,
        frame_cnt=7,
    )
    st = sdk.status_from_packet("ip", pkg)
    assert st.mode == "AUTO" and st.robot_state == 2
    assert st.jnt_ang == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert not st.safety_ok and not st.has_error
    assert st.enabled == -1 and st.motion_done == 1
    assert st.to_dict()["safety_ok"] is False


def test_status_flags_and_error_text():
    st = sdk.RobotStatus(ok=True, main_code=5, sub_code=1, jnt_ang=[0.0] * 6)
    assert st.has_error and st.encoders_all_zero
    assert "main 5 sub 1" in st.error_text
    assert not sdk.RobotStatus(ok=True, jnt_ang=[0, 100, 0, 0, 0, 0]).in_limits


def test_missing_sdk_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "fairino", None)
    with pytest.raises(ImportError, match="not on PyPI"):
        sdk.import_sdk()


def test_has_state_and_silence():
    rpc = SimpleNamespace(robot_state_pkg=SimpleNamespace(frame_head=0x5A5A))
    assert sdk.has_state(rpc)
    rpc = SimpleNamespace(robot_state_pkg=type("Empty", (), {}), stop_event=SimpleNamespace(set=lambda: None))
    assert not sdk.has_state(rpc)
    sdk.silence_state_thread(rpc)
    assert rpc.connect_to_robot() is True and rpc.reconnect_flag is False
