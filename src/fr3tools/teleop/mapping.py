"""Pad profiles: which raw axis and button does what, and normalisation to [-1, 1] / [0, 1].

The default profile is the Linux ``xpad`` layout of an Xbox 360 pad on ``/dev/input/js*``:

    axes     0 left x   1 left y   2 LT   3 right x   4 right y   5 RT   6 d-pad x   7 d-pad y
    buttons  0 A  1 B  2 X  3 Y  4 LB  5 RB  6 Back  7 Start  8 Guide  9 LS  10 RS

Sticks report ±32767 with "up" negative, so ``ly`` and ``ry`` are inverted; triggers rest at
-32767 and reach +32767 fully pressed. A compatible pad with another layout only needs a profile
(JSON, or YAML when PyYAML is installed); ``fr3-teleop --probe-pad`` prints the raw numbers.

패드 프로필: 원시 축·버튼 번호와 역할, 그리고 정규화. 기본값은 Xbox 360 패드의 xpad 배치. 스틱 위쪽이
음수라 ly·ry 를 뒤집고, 트리거는 -32767(놓음)~+32767(끝까지). 다른 패드는 프로필만 바꾸면 된다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

AXIS_MAX = 32767


@dataclass
class Mapping:
    """Raw indices and roles. / 원시 번호와 역할."""

    name: str = "xbox360-xpad"
    axes: dict = field(
        default_factory=lambda: {
            "lx": 0,
            "ly": 1,
            "lt": 2,
            "rx": 3,
            "ry": 4,
            "rt": 5,
            "dpad_x": 6,
            "dpad_y": 7,
        }
    )
    invert: dict = field(default_factory=lambda: {"ly": True, "ry": True, "dpad_y": True})
    buttons: dict = field(
        default_factory=lambda: {
            "a": 0, "b": 1, "x": 2, "y": 3, "lb": 4, "rb": 5,
            "back": 6, "start": 7, "guide": 8, "ls": 9, "rs": 10,
        }
    )  # fmt: skip
    # Roles -> button names. LT is deliberately unmapped. / 역할 → 버튼 (LT 는 비워 둔다).
    roles: dict = field(
        default_factory=lambda: {
            "deadman": "rb",
            "close": "a",
            "open": "b",
            "home": "y",
            "save_home": "x",
            "start": "start",
            "stop": "back",
            "cycle_speed": "lb",
        }
    )
    trigger_rest: int = -AXIS_MAX
    trigger_full: int = AXIS_MAX
    gripper_trigger: str = "rt"

    def rest_axes(self) -> dict[int, int]:
        """Raw value of every axis at rest (triggers are not at 0). / 놓았을 때의 원시 값."""
        out = {i: 0 for i in self.axes.values()}
        for name in ("lt", "rt"):
            if name in self.axes:
                out[self.axes[name]] = self.trigger_rest
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Mapping:
        base = cls()
        for key in ("axes", "invert", "buttons", "roles"):
            if key in d:
                merged = dict(getattr(base, key))
                merged.update(d[key])
                setattr(base, key, merged)
        for key in ("name", "trigger_rest", "trigger_full", "gripper_trigger"):
            if key in d:
                setattr(base, key, d[key])
        return base


def load_profile(path: str | None) -> Mapping:
    """Mapping from a JSON/YAML profile; ``None`` gives the default. / 프로필 읽기."""
    if not path:
        return Mapping()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML profiles need PyYAML; use JSON or pip install pyyaml") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return Mapping.from_dict(data or {})


@dataclass
class Inputs:
    """Normalised pad state for one tick. / 한 틱의 정규화된 패드 상태.

    Sticks in [-1, 1] with up and right positive, triggers in [0, 1], ``buttons`` by name.
    """

    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    dpad_x: float = 0.0
    dpad_y: float = 0.0
    buttons: dict = field(default_factory=dict)
    age: float = 0.0  # seconds since the device was last read successfully / 마지막 정상 읽기 이후
    connected: bool = True

    def pressed(self, name: str) -> bool:
        return bool(self.buttons.get(name, False))


def normalize_trigger(
    raw: float, rest: float = -AXIS_MAX, full: float = AXIS_MAX, deadband: float = 0.03
) -> float:
    """Trigger to t in [0, 1]: rest is 0, fully pressed is 1, a small rest deadband. / 트리거 정규화."""
    t = (float(raw) - rest) / (full - rest)
    t = min(1.0, max(0.0, t))
    return 0.0 if t < deadband else t


def normalize(
    axes: dict, buttons: dict, mapping: Mapping, age: float = 0.0, connected: bool = True
) -> Inputs:
    """Raw ``{index: value}`` dicts -> :class:`Inputs`. / 원시 값 → 정규화 입력."""
    rest = mapping.rest_axes()

    def raw(name: str) -> float:
        i = mapping.axes.get(name)
        return float(axes.get(i, rest.get(i, 0))) if i is not None else 0.0

    def stick(name: str) -> float:
        v = max(-1.0, min(1.0, raw(name) / AXIS_MAX))
        return (-v if mapping.invert.get(name, False) else v) + 0.0  # no -0.0 / -0.0 방지

    out = Inputs(age=age, connected=connected)
    for name in ("lx", "ly", "rx", "ry", "dpad_x", "dpad_y"):
        setattr(out, name, stick(name))
    for name in ("lt", "rt"):
        if name in mapping.axes:
            setattr(out, name, normalize_trigger(raw(name), mapping.trigger_rest, mapping.trigger_full))
    out.buttons = {name: bool(buttons.get(i, 0)) for name, i in mapping.buttons.items()}
    return out
