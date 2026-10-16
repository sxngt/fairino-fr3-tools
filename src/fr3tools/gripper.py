"""DH-Robotics AG-95 gripper on the FR3: through the controller or directly over Modbus RTU.

Two ways to wire it / 두 가지 배선:

* **Through the controller** (:class:`ControllerGripper`): the AG-95 hangs on the tool-flange
  RS-485 bus. Register it once in the web UI or with ``SetGripperConfig(4, 0, 0, 0)`` (company 4 =
  DH-Robotics, device 0); the PGI entry works too, the AG-95 uses the same register map. Commands
  are a percentage of the 95 mm stroke (``MoveGripper``); feedback is ``gripper_position`` (%) in
  the state packet.
* **Directly** (:class:`ModbusGripper`): a USB-RS485 adapter, Modbus RTU 115200 8N1, slave 1, the
  DH common register map below. Needs ``pyserial`` (``pip install fr3tools[serial]``).

What was learnt on the robot / 실기에서 배운 것:

* **Activation.** After every controller power cycle the gripper is inactive and ``MoveGripper``
  fails (code 73 was seen). ``ActGripper(index, 1)`` activates it and sweeps the fingers over the
  full stroke once, so clear the fingers first. If the first activation reports a fault, send
  ``ActGripper(index, 0)`` and activate again. :meth:`ControllerGripper.ensure_active` does this.
* **Direction.** On the reference setup 100 % = open. Check yours (``fr3-gripper probe``) and set
  ``open_pct=0`` if it moves the other way.
* **Force and speed.** The AG-95 grips with 45-160 N; this module caps force at 30 % and speed at
  50 % and never closes below ``min_width`` (10 mm) unless asked, so a missed object does not end
  in a full-force close on the fingertips.
* **Width feedback is not a grasp sensor.** With the controller path, ``gripper_position`` follows
  the *command*: it reported the commanded 12 mm while a 25 mm ring sat between the fingers. Use
  a camera, the Modbus grip state (register 0x0201, direct wiring only) or a lift test instead.
* **Blocking does not block.** ``MoveGripper``'s blocking flag returns at once on firmware v3.8.6;
  :meth:`ControllerGripper.move` polls the position until it settles.
* **Do not stall the servo stream.** A gripper XML-RPC call can take over a second. Send it from
  another thread, under the same lock as ServoJ (see :class:`fr3tools.sdk.SdkArm`).

FR3 의 AG-95: 제어기 경유(툴 플랜지 RS-485) 또는 USB-RS485 Modbus RTU 직결. 제어기 전원을 켤 때마다
활성화(핑거가 한 번 전체 스트로크 이동)가 필요하고, 제어기 경유 폭 피드백은 명령값을 따라가므로 파지
센서로 쓸 수 없다. 힘 30 %·속도 50 % 상한, 최소 폭 10 mm.
"""

from __future__ import annotations

import argparse
import struct
import time
from collections.abc import Callable

STROKE_M = 0.095
MIN_WIDTH_M = 0.010
MAX_FORCE_PCT = 30
MAX_VEL_PCT = 50

# DH common Modbus registers (AG-95, PGI, ...). / DH 공용 Modbus 레지스터.
REG_INIT = 0x0100  # write 1: initialise, 0xA5: full recalibration / 1 초기화, 0xA5 완전 재교정
REG_FORCE = 0x0101  # 20..100 %
REG_POSITION = 0x0103  # 0..1000 permille, 0 = closed, 1000 = open / ‰, 0 폐쇄
REG_SPEED = 0x0104  # 1..100 %
REG_INIT_STATE = 0x0200  # 0 not initialised, 1 done, 2 running / 초기화 상태
REG_GRIP_STATE = 0x0201  # 0 moving, 1 reached (no object), 2 gripped, 3 dropped / 파지 상태
REG_POSITION_NOW = 0x0202  # permille / 현재 위치 ‰
GRIP_STATE = {0: "moving", 1: "reached, no object", 2: "gripped", 3: "dropped"}


def clamp_width(width: float, stroke: float = STROKE_M, min_width: float = MIN_WIDTH_M) -> float:
    """Width limited to ``[min_width, stroke]`` (m). / 폭 제한."""
    return min(stroke, max(min_width, float(width)))


def width_to_percent(
    width: float, stroke: float = STROKE_M, open_pct: int = 100, min_width: float = MIN_WIDTH_M
) -> tuple[int, float]:
    """Controller position (%) and the clamped width (m). / 폭 → 제어기 위치 %, 제한된 폭.

    ``open_pct`` is the percentage the controller calls fully open (100 on the reference setup;
    0 if the gripper moves the other way).
    """
    width = clamp_width(width, stroke, min_width)
    pct = width / stroke * 100.0
    return int(round(pct if open_pct == 100 else 100.0 - pct)), width


def percent_to_width(pos: float, stroke: float = STROKE_M, open_pct: int = 100) -> float:
    """Width (m) for a controller position (%). / 위치 % → 폭(m)."""
    pct = max(0.0, min(100.0, float(pos)))
    if open_pct != 100:
        pct = 100.0 - pct
    return stroke * pct / 100.0


def half_gap_to_percent(half_gap: float, **kw) -> tuple[int, float]:
    """Same as :func:`width_to_percent` for a finger joint value (half gap). / 반개구 버전."""
    return width_to_percent(2.0 * float(half_gap), **kw)


class ControllerGripper:
    """AG-95 through the controller of a :class:`fr3tools.sdk.SdkArm` (or any object with
    ``status()`` and a locked ``call(name, *args)``). / 제어기 경유 AG-95.
    """

    def __init__(
        self,
        arm,
        index: int = 1,
        vel: int = 30,
        force: int = 30,
        open_pct: int = 100,
        stroke: float = STROKE_M,
        min_width: float = MIN_WIDTH_M,
        log: Callable[[str], None] = print,
    ):
        self.arm = arm
        self.index = index
        self.vel = max(1, min(int(vel), MAX_VEL_PCT))
        self.force = max(20, min(int(force), MAX_FORCE_PCT))
        self.open_pct = open_pct
        self.stroke = stroke
        self.min_width = min_width
        self.log = log
        self._last = stroke

    def configure(self) -> int:
        """Register the gripper as DH-Robotics, device 0 (once per controller). / 그리퍼 등록."""
        return self.arm.call("SetGripperConfig", 4, 0, 0, 0)

    def ensure_active(self, timeout: float = 10.0) -> None:
        """Activate after a power cycle or a fault (fingers sweep once). / 비활성·고장 시 재활성화."""
        st = self.arm.status()
        if st.gripper_active == 1 and st.gripper_fault == 0:
            return
        self.log(f"gripper inactive (active {st.gripper_active}, fault {st.gripper_fault}): activating")
        if st.gripper_fault:
            self.arm.call("ActGripper", self.index, 0)
            time.sleep(0.5)
        self.arm.call("ActGripper", self.index, 1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.5)
            st = self.arm.status()
            if st.gripper_active == 1 and st.gripper_fault == 0:
                return
        raise RuntimeError("gripper activation failed (controller gripper config DH/device 0, wiring, 24 V)")

    def move_width(self, width: float, timeout: float = 5.0) -> float:
        """Move to ``width`` (m); return the reported width (m), which follows the command.

        Waits until the position is within 2 % of the goal, stops changing, or ``timeout`` passes.
        목표 ±2 %, 위치 정지, 시간 초과 중 하나까지 기다린다.
        보고 폭은 명령을 따라가므로 파지 판정에 쓰지 않는다.
        """
        pos, clamped = width_to_percent(width, self.stroke, self.open_pct, self.min_width)
        if width < self.min_width:
            self.log(f"gripper width {width * 1000:.1f} mm clamped to {self.min_width * 1000:.0f} mm")
        self.ensure_active()
        self._last = clamped
        err = self.arm.call(
            "MoveGripper", self.index, pos, self.vel, self.force, int(timeout * 1000), 0, 0, 0.0, 0, 0
        )
        if err != 0:
            from fr3tools.errors import FR3Error

            raise FR3Error(err, "MoveGripper")
        t0 = time.time()
        last_pos = None
        last_change = t0
        while time.time() - t0 < timeout:
            time.sleep(0.05)
            now_pos = self.arm.status().gripper_position
            if abs(now_pos - pos) <= 2:
                break
            if now_pos != last_pos:
                last_pos = now_pos
                last_change = time.time()
            elif time.time() - t0 > 0.3 and time.time() - last_change > 0.5:
                break
        return percent_to_width(self.arm.status().gripper_position, self.stroke, self.open_pct)

    def move(self, half_gap: float, timeout: float = 5.0) -> float:
        """Finger-joint interface (half gap, m) used by the ROS driver. / 반개구 인터페이스."""
        return self.move_width(2.0 * float(half_gap), timeout) / 2.0

    def width(self) -> float:
        """Reported width if active, else the last command (m). / 보고 폭 또는 마지막 명령."""
        st = self.arm.status()
        if st.gripper_active:
            return percent_to_width(st.gripper_position, self.stroke, self.open_pct)
        return self._last

    def half_gap(self) -> float:
        return self.width() / 2.0


class FakeGripper:
    """Gripper stand-in that reports its last command. / 마지막 명령을 보고하는 가짜 그리퍼."""

    def __init__(self, stroke: float = STROKE_M, min_width: float = MIN_WIDTH_M, delay: float = 0.0):
        self.stroke = stroke
        self.min_width = min_width
        self.delay = delay
        self._width = stroke
        self.commands: list[float] = []

    def move_width(self, width: float, timeout: float = 5.0) -> float:
        self._width = clamp_width(width, self.stroke, self.min_width)
        self.commands.append(self._width)
        if self.delay:
            time.sleep(self.delay)
        return self._width

    def move(self, half_gap: float, timeout: float = 5.0) -> float:
        return self.move_width(2.0 * float(half_gap), timeout) / 2.0

    def width(self) -> float:
        return self._width

    def half_gap(self) -> float:
        return self._width / 2.0


# ----------------------------------------------------------------------------- Modbus RTU
def crc16(data: bytes) -> int:
    """Modbus CRC-16 (polynomial 0xA001, init 0xFFFF). / Modbus CRC-16."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def modbus_frame(slave: int, function: int, register: int, value: int) -> bytes:
    """Request frame for function 0x03 (read, ``value`` = count) or 0x06 (write). / 요청 프레임."""
    body = struct.pack(">BBHH", slave, function, register, value & 0xFFFF)
    return body + struct.pack("<H", crc16(body))


class ModbusGripper:
    """AG-95 over a USB-RS485 adapter (Modbus RTU). / USB-RS485 직결 AG-95.

    Gives the grip state (``moving / reached / gripped / dropped``) that the controller path lacks.
    제어기 경유에는 없는 파지 상태를 준다.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        slave: int = 1,
        baud: int = 115200,
        vel: int = 30,
        force: int = 30,
        min_width: float = MIN_WIDTH_M,
        serial_port=None,
    ):
        if serial_port is None:
            try:
                import serial
            except ImportError as exc:
                raise ImportError("ModbusGripper needs pyserial: pip install 'fr3tools[serial]'") from exc
            serial_port = serial.Serial(port, baudrate=baud, bytesize=8, parity="N", stopbits=1, timeout=0.3)
        self.ser = serial_port
        self.slave = slave
        self.vel = max(1, min(int(vel), MAX_VEL_PCT))
        self.force = max(20, min(int(force), MAX_FORCE_PCT))
        self.min_width = min_width

    def _transfer(self, frame: bytes, n: int) -> bytes:
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        res = self.ser.read(n)
        if len(res) < n:
            raise OSError(f"modbus timeout ({len(res)}/{n} bytes)")
        if struct.unpack("<H", res[-2:])[0] != crc16(res[:-2]):
            raise OSError("modbus CRC error")
        if res[0] != self.slave or res[1] & 0x80:
            raise OSError(f"modbus exception {res[:3].hex()}")
        return res

    def write(self, register: int, value: int) -> None:
        self._transfer(modbus_frame(self.slave, 0x06, register, value), 8)

    def read(self, register: int) -> int:
        res = self._transfer(modbus_frame(self.slave, 0x03, register, 1), 7)
        return struct.unpack(">H", res[3:5])[0]

    def initialise(self, timeout: float = 10.0) -> bool:
        """Initialise (fingers sweep once). / 초기화(핑거가 한 번 움직임)."""
        self.write(REG_INIT, 1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.read(REG_INIT_STATE) == 1:
                return True
            time.sleep(0.2)
        return False

    def move_width(self, width: float, timeout: float = 5.0) -> float:
        width = clamp_width(width, STROKE_M, self.min_width)
        self.write(REG_FORCE, self.force)
        self.write(REG_SPEED, self.vel)
        self.write(REG_POSITION, int(round(width / STROKE_M * 1000)))
        t0 = time.time()
        while time.time() - t0 < timeout and self.read(REG_GRIP_STATE) == 0:
            time.sleep(0.05)
        return self.width()

    def width(self) -> float:
        return self.read(REG_POSITION_NOW) / 1000.0 * STROKE_M

    def grip_state(self) -> str:
        state = self.read(REG_GRIP_STATE)
        return GRIP_STATE.get(state, str(state))


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    """``fr3-gripper``: AG-95 through the controller (or Modbus). Motion needs ``--execute``.

    제어기(또는 Modbus) 경유 AG-95. 핑거를 움직이는 명령은 ``--execute`` 가 있어야 보낸다.
    """
    from fr3tools import sdk

    ap = argparse.ArgumentParser(description="AG-95 gripper helper (motion commands need --execute)")
    ap.add_argument("cmd", choices=["status", "configure", "activate", "open", "width", "probe"])
    ap.add_argument("--mm", type=float, default=40.0, help="width for 'width' (mm)")
    ap.add_argument("--ip", default=sdk.DEFAULT_IP)
    ap.add_argument("--modbus", default="", metavar="PORT", help="use a USB-RS485 adapter instead")
    ap.add_argument("--open-pct", type=int, default=100, choices=(0, 100))
    ap.add_argument("--execute", action="store_true", help="really move the fingers")
    args = ap.parse_args(argv)

    moves = {"activate", "open", "width", "probe"}
    if args.cmd in moves and not args.execute:
        print(f"dry run: '{args.cmd}' would move the fingers; add --execute (clear the fingers first)")
        return
    if args.modbus:
        g = ModbusGripper(args.modbus)
        if args.cmd == "status":
            print(f"width {g.width() * 1000:.1f} mm, {g.grip_state()}")
        elif args.cmd == "activate":
            print("initialised" if g.initialise() else "initialisation timed out")
        elif args.cmd in ("open", "width"):
            mm = STROKE_M * 1000 if args.cmd == "open" else args.mm
            print(f"width -> {g.move_width(mm / 1000.0) * 1000:.1f} mm, {g.grip_state()}")
        else:
            print(f"'{args.cmd}' is for the controller path")
        return

    class _Arm:
        def __init__(self, ip):
            self.rpc = sdk.connect(ip)

        def status(self):
            return sdk.read_status(args.ip)

        def call(self, name, *a):
            return getattr(self.rpc, name)(*a)

    g = ControllerGripper(_Arm(args.ip), open_pct=args.open_pct)
    if args.cmd == "status":
        st = sdk.read_status(args.ip)
        print(
            f"active {st.gripper_active} fault {st.gripper_fault} position {st.gripper_position} % "
            f"(reported width follows the command; not a grasp sensor)"
            if st.ok
            else st.error
        )
    elif args.cmd == "configure":
        print("SetGripperConfig(4, 0) ->", g.configure())
    elif args.cmd == "activate":
        g.ensure_active()
        print("active")
    elif args.cmd == "open":
        print(f"reported {g.move_width(STROKE_M) * 1000:.1f} mm")
    elif args.cmd == "width":
        print(f"reported {g.move_width(args.mm / 1000.0) * 1000:.1f} mm")
    elif args.cmd == "probe":
        for mm in (95, 60, 95):
            print(f"{mm} mm -> reported {g.move_width(mm / 1000.0) * 1000:.1f} mm")
            time.sleep(1.0)
        print("the fingers should have gone 95 -> 60 -> 95 mm; if reversed, use --open-pct 0")
