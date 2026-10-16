"""Error-code lookup for the FR3 controller, the Python SDK and the vendor ROS 2 hardware layer.

There is no single public table, so this module collects what is known, and says where each
entry comes from:

* ``sdk``: negative return codes of the Python SDK (``fairino.Robot.RobotError``, SDK v2.1.6);
* ``controller``: positive return codes of XML-RPC commands that were *observed* on firmware
  v3.8.6, with the situation they appeared in (the meaning is inferred, not documented);
* ``ros_hw``: codes of FAIRINO's ROS 2 hardware layer (``frcobot_ros2``), shortened from FAIRINO's
  ``ErrorCode_lnstruction.pdf`` (2024-05-30);
* ``state``: (main, sub) pairs from the real-time state packet that were observed.

Unknown codes are reported as unknown: check the teach pendant or the web UI log, which shows the
controller's own text.

공개된 단일 표가 없어서 아는 것을 출처와 함께 모았다. SDK 음수 반환값, 펌웨어 v3.8.6 에서 *관찰된*
명령 반환값(의미는 추정), FAIRINO ROS 2 하드웨어 계층 코드(제조사 PDF 요약), 관찰된 상태 패킷
(main, sub) 쌍. 모르는 코드는 모른다고 보고한다(티치 펜던트·웹 UI 로그 확인).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorInfo:
    """One known code. / 알려진 코드 하나."""

    code: int
    meaning: str
    hint: str = ""
    source: str = ""  # sdk / controller / ros_hw / state
    observed: bool = False  # True when inferred from observation, not documentation / 관찰 기반 추정

    def text(self) -> str:
        tag = f"{self.source}, observed" if self.observed else self.source
        out = f"{self.code}: {self.meaning} [{tag}]"
        return f"{out} -> {self.hint}" if self.hint else out


def _table(source: str, rows, observed: bool = False) -> dict[int, ErrorInfo]:
    return {code: ErrorInfo(code, meaning, hint, source, observed) for code, meaning, hint in rows}


# Python SDK return codes (fairino.Robot.RobotError, v2.1.6). / Python SDK 반환값.
SDK_CODES = _table(
    "sdk",
    [
        (0, "success", ""),
        (-1, "other error", "see the SDK log"),
        (-2, "socket communication failed", "check the cable and the controller IP"),
        (-4, "XML-RPC call failed", "controller unreachable or the call timed out; check port 20003"),
        (-5, "Lua file not found", ""),
        (-6, "save path not found", ""),
        (-7, "upload file / point table not found", ""),
        (
            -8,
            "reconnecting (also used for 'download failed')",
            "the SDK lost the state port and is reconnecting; see PITFALLS 'one state client'",
        ),
        (-9, "file download checksum failed", ""),
        (-10, "downloaded file could not be written", ""),
        (-11, "bad file name", ""),
        (-12, "file upload failed", ""),
        (-13, "file too large", ""),
        (-14, "file could not be opened", ""),
        (-15, "socket send failed", "check the network"),
        (-16, "socket receive failed", "check the network"),
    ],
)

# Return codes of XML-RPC commands seen on firmware v3.8.6. / 펌웨어 v3.8.6 에서 관찰된 반환값.
CONTROLLER_CODES = _table(
    "controller",
    [
        (
            29,
            "joint limit (ServoJ ended as a joint reached its soft limit)",
            "move the target away from the limit; keep a margin of about 10 deg; ResetAllError",
        ),
        (
            73,
            "MoveGripper refused (seen while the gripper was not activated)",
            "activate with ActGripper(index, 1); check the gripper config (DH, device 0)",
        ),
    ],
    observed=True,
)

# FAIRINO frcobot_ros2 hardware layer (ErrorCode_lnstruction.pdf, 2024-05-30), shortened.
# FAIRINO ROS 2 하드웨어 계층 코드(제조사 PDF 요약).
ROS_HW_CODES = _table(
    "ros_hw",
    [
        (-1001, "command/state TCP socket could not be created", "restart the program"),
        (-1002, "connection to the controller failed", "check Ethernet and IP"),
        (-1011, "kinematics computation error", "check the input"),
        (-1012, "incomplete frame", "check the communication"),
        (-1021, "wrong command name", ""),
        (-1022, "illegal command parameters", ""),
        (-1023, "illegal command format", ""),
        (-1031, "illegal JNTPoint parameters", ""),
        (-1032, "illegal JNTPoint index (max = current count + 1)", ""),
        (-1033, "illegal element in parameters", ""),
        (-1041, "illegal CARTPoint parameters", ""),
        (-1042, "illegal CARTPoint index (max = current count + 1)", ""),
        (-1043, "illegal element in parameters", ""),
        (-1051, "invalid JNT index in GET", ""),
        (-1052, "invalid CART index in GET", ""),
        (-1053, "invalid GET parameter ('JNT,num' or 'CART,num')", ""),
        (-1054, "illegal command format", ""),
        (-1061, "illegal MoveJ JNTPoint index", ""),
        (-1062, "MoveJ forward kinematics error", "check the JNTPoint"),
        (-1063, "illegal MoveJ CARTPoint index", ""),
        (-1064, "MoveJ inverse kinematics error", "check the CARTPoint"),
        (-1065, "illegal MoveJ parameter", ""),
        (-1071, "illegal MoveL JNTPoint index", ""),
        (-1072, "MoveL forward kinematics error", "check the JNTPoint"),
        (-1073, "illegal MoveL CARTPoint index", ""),
        (-1074, "MoveL inverse kinematics error", "check the CARTPoint"),
        (-1075, "illegal MoveL parameter", ""),
        (-1081, "illegal MoveC JNTPoint index", ""),
        (-1082, "MoveC forward kinematics error", "check the JNTPoint"),
        (-1083, "illegal MoveC CARTPoint index", ""),
        (-1084, "MoveC inverse kinematics error", "check the CARTPoint"),
        (-1085, "illegal MoveC parameter", ""),
        (-1091, "illegal SplinePTP JNTPoint index", ""),
        (-1092, "SplinePTP forward kinematics error", "check the JNTPoint"),
        (-1093, "illegal SplinePTP parameter", ""),
        (-1101, "illegal NewSplinePoint CARTPoint index", ""),
        (-1102, "NewSplinePoint inverse kinematics error", "check the CARTPoint"),
        (-1103, "illegal NewSplinePoint parameter", ""),
    ],
)

# (main_code, sub_code) of the state packet, as observed. / 관찰된 상태 패킷 코드.
STATE_CODES: dict[tuple[int, int], ErrorInfo] = {
    (5, 1): ErrorInfo(
        5,
        "safety stop: seen with the emergency stop pressed (estop=1, SI0=SI1=1)",
        "release the e-stop, then ResetAllError, RobotEnable(1), Mode(0) (fr3-power recover)",
        "state",
        True,
    ),
}


def lookup(code: int) -> ErrorInfo | None:
    """Known meaning of a return code, or ``None``. / 반환값의 알려진 의미."""
    code = int(code)
    for table in (SDK_CODES, CONTROLLER_CODES, ROS_HW_CODES):
        if code in table:
            return table[code]
    return None


def describe(code: int, what: str = "") -> str:
    """Readable one-liner for a return code. / 반환값을 한 줄로 설명."""
    info = lookup(code)
    prefix = f"{what} -> " if what else ""
    if info is None:
        return f"{prefix}error {code}: unknown code (check the teach pendant / web UI log)"
    return f"{prefix}error {info.text()}"


def describe_state(main_code: int, sub_code: int) -> str:
    """Readable text for the state packet's (main, sub) code. / 상태 패킷 코드 설명."""
    if (main_code, sub_code) == (0, 0):
        return "no error"
    info = STATE_CODES.get((int(main_code), int(sub_code)))
    if info is None:
        return f"main {main_code} sub {sub_code}: unknown (check the teach pendant / web UI log)"
    return f"main {main_code} sub {sub_code}: {info.meaning} -> {info.hint}"


class FR3Error(RuntimeError):
    """A non-zero SDK or controller return code. / 0 이 아닌 SDK·제어기 반환값."""

    def __init__(self, code: int, what: str = ""):
        self.code = int(code)
        self.what = what
        self.info = lookup(self.code)
        super().__init__(describe(self.code, what))


def check(ret, what: str = ""):
    """Raise :class:`FR3Error` unless ``ret`` (or ``ret[0]``) is 0; return the payload.

    SDK calls return either an int or a tuple ``(code, payload...)``.
    SDK 호출은 정수 또는 (코드, 값...) 튜플을 돌려준다. 코드가 0 이 아니면 예외.
    """
    if isinstance(ret, tuple | list):
        code = int(ret[0])
        payload = ret[1] if len(ret) == 2 else tuple(ret[1:])
    else:
        code, payload = int(ret), None
    if code != 0:
        raise FR3Error(code, what)
    return payload
