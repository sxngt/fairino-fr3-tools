"""FAIRINO FR3 controller access through the vendor SDK. / 제조사 SDK 를 통한 FR3 제어기 접근.

The vendor SDK is the ``fairino`` Python package (``from fairino import Robot``). It is **not on
PyPI**: install it from FAIRINO's ``fairino-python-sdk`` repository at the tag that matches the
controller firmware (tested: SDK v2.1.6 with firmware v3.8.6). It is imported lazily, so the rest
of the package works without it.

The controller has two ports: XML-RPC commands (20003) and a real-time state stream (20004, one
packet every 8 ms). The state stream is served to **one client only**. The process that holds it
(normally the driver) reads joints and flags from the packet; any other process gets an
XML-RPC-only handle from :func:`connect`, which silences the SDK's reconnect loop so that commands
such as ``StopMotion`` and ``ResetAllError`` still go through.

제조사 SDK(``fairino``)는 PyPI 에 없다: 제어기 펌웨어에 맞는 태그의 ``fairino-python-sdk`` 를 설치한다
(검증: SDK v2.1.6 + 펌웨어 v3.8.6). 지연 import 한다. 실시간 상태 포트(20004)는 한 클라이언트만 받으므로
다른 프로세스는 XML-RPC 전용 핸들을 쓴다(:func:`connect` 가 SDK 재접속 루프를 끈다).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from fr3tools.errors import FR3Error, describe_state
from fr3tools.kinematics import JOINTS, SOFT_LIMITS_DEG

# Controller factory default; set FR3_IP for another address. / 제어기 공장 기본값, FR3_IP 로 변경.
DEFAULT_IP = os.environ.get("FR3_IP", "192.168.58.2")
TESTED_FIRMWARE = "v3.8.6"
TESTED_SDK = "V2.1.6"

ROBOT_STATE = {1: "Stopped", 2: "Running", 3: "Paused", 4: "Drag"}
ROBOT_MODE = {0: "AUTO", 1: "MANUAL"}
PROGRAM_STATE = {1: "stopped", 2: "running", 3: "paused"}

_HANDLES: dict[str, object] = {}
_HANDLES_LOCK = threading.Lock()


def import_sdk():
    """Import the vendor ``Robot`` module or explain where it comes from. / SDK import 또는 안내."""
    try:
        from fairino import Robot
    except ImportError as exc:
        raise ImportError(
            "the FAIRINO SDK ('fairino' package) is not installed; it is not on PyPI. Install "
            "fairino-python-sdk from FAIRINO at the tag matching the controller firmware "
            f"(tested: SDK {TESTED_SDK} with firmware {TESTED_FIRMWARE}); see README."
        ) from exc
    return Robot


def has_state(rpc) -> bool:
    """Whether this handle receives real-time state packets. / 실시간 상태 패킷 수신 여부."""
    pkg = rpc.robot_state_pkg
    return not isinstance(pkg, type) and int(pkg.frame_head) == 0x5A5A


def silence_state_thread(rpc) -> None:
    """Make a handle XML-RPC only. / 핸들을 XML-RPC 전용으로 만든다.

    Without the state port the SDK loops on reconnect (up to 1000 × 2 s), and every XML-RPC call,
    ``StopMotion`` included, blocks while that loop runs. Pretend the reconnect succeeded and end
    the state thread.
    상태 포트 없이는 SDK 가 재접속 루프를 돌며 모든 XML-RPC 호출(StopMotion 포함)을 막는다.
    재접속을 성공으로 위장하고 상태 스레드를 끝낸다.
    """
    rpc.closeRPC_state = True
    rpc.robot_realstate_exit = True
    with contextlib.suppress(AttributeError):
        rpc.stop_event.set()
    rpc.connect_to_robot = lambda: True
    rpc.reconnect_flag = False


def connect(ip: str = DEFAULT_IP, wait_state: float = 1.5, require_state: bool = False):
    """SDK handle for ``ip``, cached per process. / 프로세스당 IP 별로 캐시되는 SDK 핸들.

    Waits up to ``wait_state`` seconds for state packets; if none arrive (another process holds
    the state port) the handle becomes XML-RPC only. ``require_state`` turns that case into a
    ``ConnectionError``.
    상태 패킷을 ``wait_state`` 초 기다리고, 오지 않으면 XML-RPC 전용 핸들이 된다.
    """
    with _HANDLES_LOCK:
        rpc = _HANDLES.get(ip)
        if rpc is None:
            robot = import_sdk()
            # The SDK prints connection chatter on construction. / 생성 시 SDK 출력 숨김.
            with contextlib.redirect_stdout(io.StringIO()):
                rpc = robot.RPC(ip)
            if not robot.RPC.is_conect:
                # Reset the SDK's class-wide flag, or every later attempt fails too.
                # SDK 의 클래스 전역 플래그를 복구하지 않으면 이후 시도가 모두 실패한다.
                robot.RPC.is_conect = True
                raise ConnectionError(f"{ip}: XML-RPC (port 20003) connection failed")
            t0 = time.time()
            while time.time() - t0 < wait_state and not has_state(rpc):
                time.sleep(0.05)
            if not has_state(rpc):
                silence_state_thread(rpc)
            _HANDLES[ip] = rpc
    if require_state and not has_state(rpc):
        raise ConnectionError(f"{ip}: no real-time state (port 20004 is held by another process)")
    return rpc


def disconnect(ip: str = DEFAULT_IP) -> None:
    """Close and forget the cached handle. / 캐시된 핸들을 닫는다."""
    with _HANDLES_LOCK:
        rpc = _HANDLES.pop(ip, None)
    if rpc is None:
        return
    with contextlib.suppress(Exception), contextlib.redirect_stdout(io.StringIO()):
        rpc.CloseRPC()


@dataclass
class RobotStatus:
    """Decoded real-time state packet. / 해석된 실시간 상태 패킷."""

    ok: bool
    error: str = ""
    ip: str = ""
    mode: str = ""  # AUTO / MANUAL
    robot_state: int = 0  # 1 Stopped, 2 Running, 3 Paused, 4 Drag
    program_state: int = 0
    main_code: int = 0
    sub_code: int = 0
    estop: int = 0
    collision: int = 0
    safety_stop0: int = 0
    safety_stop1: int = 0
    motion_done: int = 1
    enabled: int = -1  # servo enable state; -1 = not in the packet / -1 = 패킷에 없음
    jnt_ang: list = field(default_factory=list)  # deg
    jnt_vel: list = field(default_factory=list)  # deg/s
    jnt_tor: list = field(default_factory=list)  # Nm
    tcp_pos: list = field(default_factory=list)  # mm, deg
    flange_pos: list = field(default_factory=list)  # mm, deg
    gripper_active: int = 0
    gripper_fault: int = 0
    gripper_position: int = 0  # % of the stroke / 스트로크 %
    gripper_motiondone: int = 0
    frame_cnt: int = 0

    @property
    def has_error(self) -> bool:
        return self.main_code != 0 or self.sub_code != 0

    @property
    def safety_ok(self) -> bool:
        return not (self.estop or self.collision or self.safety_stop0 or self.safety_stop1)

    @property
    def in_limits(self) -> bool:
        return all(
            lo - 0.5 <= angle <= hi + 0.5
            for angle, (lo, hi) in zip(self.jnt_ang, SOFT_LIMITS_DEG, strict=False)
        )

    @property
    def encoders_all_zero(self) -> bool:
        """All joints read exactly 0: a broken packet, not a pose. / 모든 관절 0 = 패킷 이상."""
        return bool(self.jnt_ang) and float(np.abs(self.jnt_ang).max()) < 1e-6

    @property
    def error_text(self) -> str:
        return describe_state(self.main_code, self.sub_code)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.update(has_error=self.has_error, safety_ok=self.safety_ok, in_limits=self.in_limits)
        return d


def status_from_packet(ip: str, pkg) -> RobotStatus:
    """Decode an SDK ``robot_state_pkg``; missing fields keep defaults. / SDK 상태 패킷 해석."""

    def num(name: str, default: int = 0) -> int:
        return int(getattr(pkg, name, default))

    def vec(name: str) -> list:
        return [float(v) for v in getattr(pkg, name, ())]

    return RobotStatus(
        ok=True,
        ip=ip,
        mode=ROBOT_MODE.get(num("robot_mode"), f"?{num('robot_mode')}"),
        robot_state=num("robot_state"),
        program_state=num("program_state"),
        main_code=num("main_code"),
        sub_code=num("sub_code"),
        estop=num("EmergencyStop"),
        collision=num("collisionState"),
        safety_stop0=num("safety_stop0_state"),
        safety_stop1=num("safety_stop1_state"),
        motion_done=num("motion_done", 1),
        enabled=num("rbtEnableState", -1),
        jnt_ang=vec("jt_cur_pos"),
        jnt_vel=vec("actual_qd"),
        jnt_tor=vec("jt_cur_tor"),
        tcp_pos=vec("tl_cur_pos"),
        flange_pos=vec("flange_cur_pos"),
        gripper_active=num("gripper_active"),
        gripper_fault=num("gripper_fault"),
        gripper_position=num("gripper_position"),
        gripper_motiondone=num("gripper_motiondone"),
        frame_cnt=num("frame_cnt"),
    )


def read_status(ip: str = DEFAULT_IP, timeout: float = 3.0) -> RobotStatus:
    """Fresh status from the state port; ``ok=False`` if this process cannot hold it.

    상태 포트에서 읽은 최신 상태. 이 프로세스가 포트를 못 잡으면 ``ok=False``.
    """
    try:
        rpc = connect(ip, wait_state=timeout)
        if not has_state(rpc):
            raise ConnectionError("no real-time state (is the driver holding port 20004?)")
        first = int(rpc.robot_state_pkg.frame_cnt)
        t0 = time.time()
        while time.time() - t0 < min(timeout, 0.5) and int(rpc.robot_state_pkg.frame_cnt) == first:
            time.sleep(0.01)
        return status_from_packet(ip, rpc.robot_state_pkg)
    except Exception as exc:
        return RobotStatus(ok=False, ip=ip, error=f"{type(exc).__name__}: {exc}")


def stop_motion(rpc, lock: threading.Lock | None = None) -> int:
    """``StopMotion`` and end any servo session. / 즉시 정지 + 서보 세션 종료.

    ``lock`` guards the SDK's ``ServerProxy``, which is not thread safe. A stop must not wait
    long for it, so after 0.5 s the call goes out anyway.
    SDK 의 ServerProxy 는 스레드 안전하지 않아 잠금을 쓰되, 정지는 0.5 s 이상 기다리지 않는다.
    """
    held = lock.acquire(timeout=0.5) if lock is not None else False
    try:
        err = rpc.StopMotion()
        with contextlib.suppress(Exception):
            rpc.ServoMoveEnd()
        return err
    finally:
        if held:
            lock.release()


def versions(ip: str = DEFAULT_IP) -> dict:
    """SDK and controller software versions. / SDK·제어기 소프트웨어 버전."""
    out: dict = {}
    try:
        rpc = connect(ip)
        ret = rpc.GetSDKVersion()
        out["sdk"] = ret[1] if isinstance(ret, tuple) and ret[0] == 0 else ret
        ret = rpc.GetSoftwareVersion()
        out["software"] = list(ret[1:]) if isinstance(ret, tuple) and ret[0] == 0 else ret
    except Exception as exc:
        out["error"] = str(exc)
    return out


def firmware_of(versions_dict: dict) -> str:
    """Controller firmware string from :func:`versions` (``"?"`` if unknown). / 펌웨어 문자열."""
    software = versions_dict.get("software")
    return software[1] if isinstance(software, list) and len(software) > 1 else "?"


class SdkArm:
    """The real arm for a servo stream: state packet in, ServoJ out. / 서보 스트림용 실제 팔.

    Owns the state port (``require_state``) and the lock that serialises XML-RPC calls between
    the servo stream, the gripper and services (a shared ``ServerProxy`` raises
    ``CannotSendRequest`` when two threads call at once).
    상태 포트를 소유하고, 서보 스트림·그리퍼·서비스의 XML-RPC 호출을 하나의 잠금으로 직렬화한다.
    """

    fake = False

    def __init__(self, ip: str = DEFAULT_IP, reset_before_servo: bool = True):
        self.ip = ip
        self.rpc = connect(ip, require_state=True)
        self.lock = threading.Lock()
        self.reset_before_servo = reset_before_servo
        self._count = -1
        self._count_t = time.monotonic()

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Joints (rad), velocities (rad/s), torques (Nm) and whether the packet is fresh.

        관절(rad)·속도(rad/s)·토크(Nm)와 패킷 신선도. ``frame_cnt`` 가 1 s 멈추면 stale.
        """
        pkg = self.rpc.robot_state_pkg
        count = int(pkg.frame_cnt)
        now = time.monotonic()
        if count != self._count:
            self._count = count
            self._count_t = now
        fresh = now - self._count_t < 1.0
        q = np.radians([float(v) for v in pkg.jt_cur_pos])
        qd = np.radians([float(v) for v in pkg.actual_qd])
        tau = np.array([float(v) for v in pkg.jt_cur_tor])
        return q, qd, tau, fresh

    def status(self) -> RobotStatus:
        return status_from_packet(self.ip, self.rpc.robot_state_pkg)

    def ready_for_motion(self) -> tuple[bool, str]:
        """ServoJ needs AUTO mode, no safety stop and no error. / AUTO·안전 정상·에러 없음 필요."""
        st = self.status()
        ok = st.mode == "AUTO" and st.safety_ok and not st.has_error and st.enabled != 0
        why = (
            f"mode {st.mode}, safety_ok {st.safety_ok}, enabled {st.enabled}, "
            f"error {st.main_code}/{st.sub_code}"
        )
        return ok, why

    def call(self, name: str, *args):
        """Locked XML-RPC call by method name. / 잠금 아래 XML-RPC 호출."""
        with self.lock:
            return getattr(self.rpc, name)(*args)

    def stop(self) -> int:
        return stop_motion(self.rpc, self.lock)

    # ---- servo session (see fr3tools.servo.ServoStream) / 서보 세션
    def servo_begin(self) -> None:
        # A stopped controller ignores ServoJ until its error state is reset (firmware v3.8.6).
        # 정지 상태의 제어기는 에러 리셋 전까지 ServoJ 를 무시한다.
        if self.reset_before_servo:
            self.call("ResetAllError")
        err = self.call("ServoMoveStart")
        if err != 0:
            raise FR3Error(err, "ServoMoveStart")

    def servo_command(self, q_rad: np.ndarray, cmd_t: float) -> None:
        q_deg = [float(v) for v in np.degrees(q_rad)]
        err = self.call("ServoJ", q_deg, [0.0, 0.0, 0.0, 0.0], 0.0, 0.0, cmd_t, 0.0, 0.0)
        if err != 0:
            raise FR3Error(err, "ServoJ")

    def servo_end(self, failed: bool) -> None:
        with contextlib.suppress(Exception):
            self.call("ServoMoveEnd")
        if failed:
            with contextlib.suppress(Exception):
                self.stop()

    def monitor(self, dev_deg: float, on_trip, log):
        """Safety monitor reading this arm's state packet. / 이 팔의 상태 패킷을 읽는 안전 감시."""
        from fr3tools.safety import SafetyMonitor

        return SafetyMonitor(self.status, stop=self.stop, dev_deg=dev_deg, on_trip=on_trip, log=log)


__all__ = [
    "DEFAULT_IP",
    "JOINTS",
    "RobotStatus",
    "SdkArm",
    "connect",
    "disconnect",
    "has_state",
    "read_status",
    "status_from_packet",
    "stop_motion",
    "versions",
]
