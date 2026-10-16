"""AG-95 conversions, Modbus framing and the motion lock. / AG-95 변환, Modbus, 모션 잠금."""

import subprocess
import sys
import textwrap

import pytest

from fr3tools.gripper import (
    FakeGripper,
    ModbusGripper,
    crc16,
    half_gap_to_percent,
    modbus_frame,
    percent_to_width,
    width_to_percent,
)
from fr3tools.lock import MotionLock, MotionLockBusyError, exclusive_motion


def test_width_percent_round_trip_and_clamps():
    pos, width = width_to_percent(0.04)
    assert width == pytest.approx(0.04) and pos == round(0.04 / 0.095 * 100)
    assert percent_to_width(pos) == pytest.approx(0.04, abs=0.001)
    assert width_to_percent(0.0)[1] == pytest.approx(0.010)  # never a full close / 전폐 금지
    assert width_to_percent(1.0) == (100, pytest.approx(0.095))
    assert width_to_percent(0.095, open_pct=0)[0] == 0  # reversed mounting / 반대 장착
    assert percent_to_width(0, open_pct=0) == pytest.approx(0.095)
    assert half_gap_to_percent(0.02) == width_to_percent(0.04)


def test_fake_gripper_records_clamped_commands():
    g = FakeGripper()
    assert g.move_width(0.002) == pytest.approx(0.010)
    assert g.move(0.02) == pytest.approx(0.02)
    assert g.commands == [pytest.approx(0.010), pytest.approx(0.04)]


def test_modbus_crc_and_frames():
    # Reference frame: read 1 register at 0x0200 from slave 1. / 기준 프레임.
    frame = modbus_frame(1, 0x03, 0x0200, 1)
    assert frame[:6] == bytes([0x01, 0x03, 0x02, 0x00, 0x00, 0x01])
    assert crc16(frame) == 0  # CRC over a frame with its CRC is zero / CRC 포함 시 0
    assert crc16(b"123456789") == 0x4B37


class FakeSerial:
    """Echoes writes like an AG-95 and answers reads with a fixed value. / 가짜 시리얼."""

    def __init__(self, value):
        self.value = value
        self.reply = b""

    def reset_input_buffer(self):
        pass

    def write(self, frame):
        if frame[1] == 0x06:
            self.reply = frame
        else:
            body = bytes([frame[0], 0x03, 2]) + self.value.to_bytes(2, "big")
            self.reply = body + crc16(body).to_bytes(2, "little")

    def read(self, n):
        return self.reply[:n]


def test_modbus_gripper_reads_width_and_state():
    g = ModbusGripper(serial_port=FakeSerial(500))
    assert g.width() == pytest.approx(0.0475)
    g.write(0x0103, 400)
    g.ser.value = 2
    assert g.grip_state() == "gripped"


def test_motion_lock_is_exclusive(tmp_path):
    with MotionLock("test", lock_dir=str(tmp_path)) as lock:
        assert lock.held and lock.holder()["pid"] > 0
        with pytest.raises(MotionLockBusyError, match="another process"):
            MotionLock("test", lock_dir=str(tmp_path)).acquire()
        MotionLock("other", lock_dir=str(tmp_path)).acquire().release()  # other controller / 다른 제어기
    with MotionLock("test", lock_dir=str(tmp_path)):
        pass


def test_motion_lock_across_processes(tmp_path):
    code = textwrap.dedent(
        f"""
        import time
        from fr3tools.lock import MotionLock
        with MotionLock("x", lock_dir={str(tmp_path)!r}):
            print("held", flush=True)
            time.sleep(3)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        with pytest.raises(MotionLockBusyError, match=f"pid {proc.pid}"):
            MotionLock("x", lock_dir=str(tmp_path)).acquire()
    finally:
        proc.kill()
        proc.wait()
    MotionLock("x", lock_dir=str(tmp_path)).acquire().release()  # released on exit / 종료 시 해제


def test_exclusive_motion_decorator(tmp_path):
    @exclusive_motion("deco", lock_dir=str(tmp_path))
    def move():
        with pytest.raises(MotionLockBusyError):
            MotionLock("deco", lock_dir=str(tmp_path)).acquire()
        return 7

    assert move() == 7
