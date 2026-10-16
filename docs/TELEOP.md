# Gamepad teleoperation

`fr3-teleop` drives the arm top-down with an Xbox 360 (or compatible) pad: the left stick moves
the grip point in the table plane, the right stick moves it up and down and turns the wrist, the
right trigger sets the gripper width. Speeds are proportional to stick deflection.

Xbox 360 계열 패드로 팔을 탑다운 자세로 움직인다. 왼쪽 스틱 = 평면 이동, 오른쪽 스틱 = 상하·손목 회전,
RT = 그리퍼 폭. 속도는 스틱 기울기에 비례한다.

```bash
fr3-teleop --fake --fake-pad          # no pad, no ROS: built-in demo on a fake arm
fr3-teleop --fake                     # your pad, fake arm
fr3-teleop                            # your pad, real driver, dry run (prints targets)
fr3-teleop --execute --speed slow     # streams to fr3-driver
fr3-teleop --probe-pad                # raw events, to write a profile
```

## Mapping (default profile, xpad) / 기본 배치

```
   LT  (unmapped)                                   RT  gripper width: released open, pressed closed
   LB  cycle speed preset                           RB  DEADMAN: hold to move
  .------------------------------------------------------------------------.
  |  (L) left stick          [Back]   (Guide)   [Start]          (Y) home   |
  |      x / y               end session        begin      (X)          (B) |
  |                                                        save home   open |
  |   d-pad                         (R) right stick              (A) close  |
  |   up / down = faster / slower       up / down = z                       |
  |                                     left / right = wrist yaw            |
  '------------------------------------------------------------------------'
```

| input | raw (js) | action |
|---|---|---|
| left stick | axes 0, 1 | x / y velocity in the task frame (radial deadzone, diagonals capped) |
| right stick vertical | axis 4 | z velocity |
| right stick horizontal | axis 3 | wrist yaw rate (right = clockwise from above) |
| RT | axis 5 | gripper width: `open - t·(open - closed)` |
| RB | button 5 | deadman |
| A / B | buttons 0 / 1 | close fully / open fully (the trigger is ignored until it moves again) |
| X / Y | buttons 2 / 3 | save home / return to home (smoothly, while RB is held) |
| LB, d-pad up/down | button 4, axis 7 | speed presets slow (2 cm/s, 15°/s), normal (5 cm/s, 30°/s), fast (10 cm/s, 45°/s) |
| Start / Back | buttons 7 / 6 | begin / end the session |

The pad is found by name ("X-Box 360" or "Xbox" in `/proc/bus/input/devices`, or the `JSIOCGNAME`
ioctl), so another joystick on `js0` is not picked by mistake; `--device` overrides. The device is
opened read-only. `--evdev` reads `/dev/input/event*` through python-evdev instead.

## Control model / 제어 모델

* **State:** grip point (the TCP; set the tool with `--params` or `--tool-length`) and the wrist
  yaw about the vertical. The tool keeps its start orientation (top-down within 10°), turned by
  the yaw.
* **Task frame:** `--frame base` (default), `--frame start` (axes of the gripper at the start, so
  stick-up means "the way the fingers face"), or `--frame "x y z yaw_deg"` (a table frame).
* **Each tick (50 Hz):** scaled deadzone (0.12) and power curve (exponent 2): `v = curve(a) · v_max`
  → first-order filter and Cartesian acceleration limit (0.25 m/s², 90°/s²) → integrate → clamp
  to the workspace box and the z floor → IK on the start branch next to the previous solution, 5°
  from the joint limits, at most 3° per tick → stream the joints.
* **Gripper:** RT gives `t ∈ [0, 1]` (rest −32767 → 0); commands go out only when the width moved
  by 1 mm and at most 8 times per second, and the final value is always sent once the trigger
  settles. Fully pressed commands the closed width; the force limit stops the fingers on the
  object.

## Safety rules / 안전 규칙

1. **Dry run unless `--execute`.** Without it the targets are printed, nothing is streamed.
2. **Deadman.** Motion only while RB is held; releasing it holds the current target at once.
3. **No surprise at start.** The first target is the measured pose; nothing moves until a stick
   moves.
4. **Lost input.** If the pad cannot be read for 0.3 s (unplugged, reader stalled), the target
   holds; after 1 s more the session ends and the stream stops.
5. **Workspace.** A box of ±0.20 m around the start (task axes; up 0.20 m, down 0.20 m), and a
   floor at `--table-z + --min-height` (default 0 + 0.05 m). `--box` sets it explicitly.
6. **IK holds, never jumps.** An unreachable target, a joint-limit margin or a jump larger than 3°
   per tick holds the pose and shows the reason in the status line.
7. **The driver still guards.** The servo stream's speed, acceleration and jump limits and the
   safety monitor apply downstream; teleop does not change them.

## Profiles / 프로필

Profiles are JSON (or YAML with PyYAML). Only the keys you give replace the defaults:

```json
{
  "name": "my-pad",
  "axes": {"lx": 0, "ly": 1, "rx": 2, "ry": 3, "rt": 4, "lt": 5, "dpad_x": 6, "dpad_y": 7},
  "invert": {"ly": true, "ry": true, "dpad_y": true},
  "buttons": {"a": 0, "b": 1, "x": 3, "y": 4, "lb": 6, "rb": 7, "back": 10, "start": 11},
  "roles": {"deadman": "rb", "close": "a", "open": "b", "home": "y", "save_home": "x",
            "start": "start", "stop": "back", "cycle_speed": "lb"},
  "trigger_rest": -32767,
  "trigger_full": 32767,
  "gripper_trigger": "rt"
}
```

`fr3-teleop --probe-pad` prints every raw event (`axis 5 = -32767 (init)` ...); move each control
and note its number and range. `examples/xbox360_profile.json` is the default profile written out.

## Scripted input / 스크립트 입력

`--fake-pad FILE` replays keyframes (logical names; sticks in [-1, 1] with up and right positive,
triggers in [0, 1]); each keyframe holds until the next, and `{"t": 9.5, "disconnect": true}`
simulates unplugging:

```json
[{"t": 0.2, "buttons": {"start": 1}},
 {"t": 0.3, "buttons": {"start": 0, "rb": 1}, "axes": {"lx": 0.5}},
 {"t": 2.0, "axes": {"lx": 0.0, "rt": 1.0}},
 {"t": 3.0, "buttons": {"rb": 0, "back": 1}}]
```
