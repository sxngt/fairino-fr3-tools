"""Gamepad input: the Linux joystick API (no dependency), optional ``evdev``, and scripted input.

The joystick API delivers 8-byte ``js_event`` structs (``u32 time_ms, s16 value, u8 type,
u8 number``) from ``/dev/input/js*``. Events arrive only on change, plus one ``INIT``-flagged
event per axis and button when the device is opened, so a pad held still sends nothing. Liveness
is therefore "the device can still be read": :attr:`PadState.last_ok` is updated on every
successful non-blocking read, and a disconnect (``ENODEV``) stops it.

The device is opened read-only. :func:`find_device` picks the pad by name from
``/proc/bus/input/devices`` (or the ``JSIOCGNAME`` ioctl), so another joystick on ``js0`` is not
picked by mistake.

리눅스 조이스틱 API(의존성 없음): /dev/input/js* 에서 8 바이트 js_event 를 읽는다. 값이 바뀔 때만
이벤트가 오므로 "살아 있음"은 장치를 계속 읽을 수 있는가로 판단한다. 장치는 읽기 전용으로 열고, 이름
("X-Box 360", "Xbox")으로 찾는다.
"""

from __future__ import annotations

import errno
import fcntl
import glob
import json
import os
import re
import struct
import time
from dataclasses import dataclass, field

from fr3tools.teleop.mapping import AXIS_MAX, Mapping

JS_EVENT = struct.Struct("<IhBB")
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
PAD_NAMES = ("x-box 360", "xbox")


@dataclass(frozen=True)
class JsEvent:
    time_ms: int
    value: int
    type: int
    number: int

    @property
    def is_init(self) -> bool:
        return bool(self.type & JS_EVENT_INIT)

    @property
    def kind(self) -> str:
        t = self.type & ~JS_EVENT_INIT
        return "button" if t == JS_EVENT_BUTTON else "axis" if t == JS_EVENT_AXIS else f"type{t}"


def parse_js_events(data: bytes) -> list[JsEvent]:
    """Split a byte buffer into ``js_event`` structs (a partial tail is ignored). / 바이트 → 이벤트."""
    n = len(data) // JS_EVENT.size
    return [JsEvent(*JS_EVENT.unpack_from(data, i * JS_EVENT.size)) for i in range(n)]


@dataclass
class PadState:
    """Raw state: ``{index: value}`` for axes and buttons. / 원시 상태."""

    axes: dict = field(default_factory=dict)
    buttons: dict = field(default_factory=dict)
    last_ok: float = 0.0  # monotonic time of the last successful read / 마지막 정상 읽기
    connected: bool = True

    def apply(self, ev: JsEvent) -> None:
        if ev.kind == "axis":
            self.axes[ev.number] = ev.value
        elif ev.kind == "button":
            self.buttons[ev.number] = ev.value


def js_name(path: str) -> str:
    """Device name through the ``JSIOCGNAME`` ioctl (read-only open). / 장치 이름."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        buf = bytearray(128)
        jsiocgname = (2 << 30) | (len(buf) << 16) | (ord("j") << 8) | 0x13
        fcntl.ioctl(fd, jsiocgname, buf)
        return bytes(buf).split(b"\0", 1)[0].decode(errors="replace")
    finally:
        os.close(fd)


def devices_from_proc(text: str) -> list[tuple[str, str]]:
    """``(name, /dev/input/jsN)`` pairs from ``/proc/bus/input/devices``. / 장치 목록 해석."""
    out = []
    for block in text.split("\n\n"):
        name = re.search(r'^N: Name="(.*)"', block, re.M)
        handlers = re.search(r"^H: Handlers=(.*)$", block, re.M)
        if not name or not handlers:
            continue
        for h in handlers.group(1).split():
            if re.fullmatch(r"js\d+", h):
                out.append((name.group(1), f"/dev/input/{h}"))
    return out


def find_device(names=PAD_NAMES) -> str | None:
    """Path of the first joystick whose name matches an Xbox pad. / 이름으로 패드 찾기."""
    try:
        with open("/proc/bus/input/devices", encoding="utf-8") as f:
            for name, path in devices_from_proc(f.read()):
                if any(n in name.lower() for n in names) and os.path.exists(path):
                    return path
    except OSError:
        pass
    for path in sorted(glob.glob("/dev/input/js*")):
        try:
            if any(n in js_name(path).lower() for n in names):
                return path
        except OSError:
            continue
    return None


class JoystickReader:
    """Non-blocking reader for ``/dev/input/js*``. / js 장치 비차단 읽기."""

    def __init__(self, path: str, mapping: Mapping | None = None, clock=time.monotonic):
        self.path = path
        self.clock = clock
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.state = PadState(axes=(mapping or Mapping()).rest_axes(), last_ok=clock())
        try:
            self.name = js_name(path)
        except OSError:
            self.name = path

    def poll(self) -> list[JsEvent]:
        """Read everything pending; mark a disconnect. / 대기 중인 이벤트를 모두 읽는다."""
        if not self.state.connected:
            return []
        events: list[JsEvent] = []
        while True:
            try:
                data = os.read(self.fd, JS_EVENT.size * 64)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno in (errno.ENODEV, errno.EIO, errno.EBADF):
                    self.state.connected = False
                    return events
                raise
            if not data:
                break
            events.extend(parse_js_events(data))
        for ev in events:
            self.state.apply(ev)
        self.state.last_ok = self.clock()
        return events

    def close(self) -> None:
        os.close(self.fd)


class EvdevReader:
    """Optional reader through ``python-evdev``, normalised to the joystick API's numbers.

    evdev 를 쓰는 선택적 읽기(값과 번호를 joystick API 기준으로 맞춘다).
    """

    ABS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 16: 6, 17: 7}  # ABS_X..ABS_RZ, HAT0X/Y
    KEYS = {304: 0, 305: 1, 307: 2, 308: 3, 310: 4, 311: 5, 314: 6, 315: 7, 316: 8, 317: 9, 318: 10}

    def __init__(self, path: str, mapping: Mapping | None = None, clock=time.monotonic):
        try:
            import evdev
        except ImportError as exc:
            raise ImportError("EvdevReader needs python-evdev (pip install evdev)") from exc
        self.dev = evdev.InputDevice(path)
        self.name = self.dev.name
        self.clock = clock
        self.state = PadState(axes=(mapping or Mapping()).rest_axes(), last_ok=clock())
        self._info = {
            code: self.dev.absinfo(code)
            for code in self.ABS
            if code in dict(self.dev.capabilities().get(3, []))
        }

    def poll(self) -> list:
        if not self.state.connected:
            return []
        try:
            events = list(self.dev.read())
        except BlockingIOError:
            events = []
        except OSError:
            self.state.connected = False
            return []
        for ev in events:
            if ev.type == 3 and ev.code in self.ABS:
                info = self._info.get(ev.code)
                v = ev.value
                if info is not None and info.max != info.min:
                    v = int(round(-AXIS_MAX + (v - info.min) * 2 * AXIS_MAX / (info.max - info.min)))
                self.state.axes[self.ABS[ev.code]] = v
            elif ev.type == 1 and ev.code in self.KEYS:
                self.state.buttons[self.KEYS[ev.code]] = int(ev.value != 0)
        self.state.last_ok = self.clock()
        return events

    def close(self) -> None:
        self.dev.close()


class ScriptedPad:
    """Pad driven by keyframes, for tests and ``--fake-pad``. / 키프레임으로 움직이는 가짜 패드.

    A script is a list of ``{"t": seconds, "axes": {name: value}, "buttons": {name: 0|1}}`` with
    logical names from the mapping; stick values in [-1, 1] (up and right positive), trigger
    values in [0, 1]. Each keyframe holds until the next one; ``{"t": ..., "disconnect": true}``
    simulates unplugging.
    키프레임 목록(논리 이름, 스틱 [-1, 1], 트리거 [0, 1]). 다음 키프레임까지 유지된다.
    """

    def __init__(self, script: list[dict], mapping: Mapping | None = None, clock=time.monotonic):
        self.mapping = mapping or Mapping()
        self.script = sorted(script, key=lambda k: float(k["t"]))
        self.clock = clock
        self.t0 = clock()
        self.name = "scripted pad"
        self.state = PadState(axes=self.mapping.rest_axes(), last_ok=self.t0)
        self._next = 0

    @classmethod
    def from_file(cls, path: str, mapping: Mapping | None = None, clock=time.monotonic) -> ScriptedPad:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["script"] if isinstance(data, dict) else data, mapping, clock)

    @property
    def finished(self) -> bool:
        return self._next >= len(self.script)

    def _raw_axis(self, name: str, value: float) -> int:
        m = self.mapping
        if name in ("lt", "rt"):
            return int(round(m.trigger_rest + float(value) * (m.trigger_full - m.trigger_rest)))
        v = -float(value) if m.invert.get(name, False) else float(value)
        return int(round(max(-1.0, min(1.0, v)) * AXIS_MAX))

    def poll(self) -> list:
        if not self.state.connected:
            return []
        now = self.clock() - self.t0
        applied = []
        while self._next < len(self.script) and float(self.script[self._next]["t"]) <= now:
            key = self.script[self._next]
            self._next += 1
            if key.get("disconnect"):
                self.state.connected = False
                return applied
            for name, value in key.get("axes", {}).items():
                self.state.axes[self.mapping.axes[name]] = self._raw_axis(name, value)
            for name, value in key.get("buttons", {}).items():
                self.state.buttons[self.mapping.buttons[name]] = int(value)
            applied.append(key)
        self.state.last_ok = self.clock()
        return applied

    def close(self) -> None:
        pass


def probe(path: str, seconds: float = 0.0, out=print) -> None:
    """Print raw events to help write a profile (Ctrl-C to stop). / 원시 이벤트 출력(프로필 작성용)."""
    reader = JoystickReader(path)
    out(f"{path}: {reader.name}  (move each stick, trigger and button; Ctrl-C to stop)")
    t0 = time.monotonic()
    try:
        while not seconds or time.monotonic() - t0 < seconds:
            for ev in reader.poll():
                tag = " (init)" if ev.is_init else ""
                out(f"{ev.kind:6s} {ev.number:2d} = {ev.value:7d}{tag}")
            if not reader.state.connected:
                out("disconnected")
                return
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
