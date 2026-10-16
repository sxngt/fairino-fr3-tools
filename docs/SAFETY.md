# Safety

This software moves an industrial arm. It adds layers of protection but **does not replace the
controller's safety functions, a risk assessment, or a person at the emergency stop**. Read
[PITFALLS.md](PITFALLS.md) before the first motion.

이 소프트웨어는 산업용 팔을 움직인다. 보호 계층을 더하지만 제어기 안전 기능, 위험성 평가, 비상정지
옆의 사람을 대신하지 않는다.

## Defaults that keep you safe / 기본값

* **Dry run by default.** Examples, `fr3-teleop`, `fr3-handeye`, `fr3-probe` and the motion
  commands of `fr3-gripper` print what they would do until you add `--execute`. `fr3-power`
  commands that change the controller state need `--yes`.
* **No motion on start-up.** The driver only reads state until a client starts a servo session.
  Teleop starts with the target equal to the current pose.
* **Readiness checks.** A session starts only in AUTO mode, servo enabled, with no safety stop
  and no error.
* **Limits in the stream.** 30°/s joint speed and 60°/s² acceleration by default, braking into
  the target, a 15° jump limit and a 1.5 s stale-target timeout.
* **Latching safety monitor.** Controller flags (e-stop, collision, SI0/SI1, error code), tracking
  deviation (`start_tol_deg`) and torque jumps end the stream and send `StopMotion`. A tripped
  monitor never re-arms by itself.
* **Trajectory validation.** MoveIt goals with wrong joints, non-increasing time, a start away from
  the measured joints or a peak speed above the limit are rejected before anything moves.
* **One mover at a time.** A cooperative `MotionLock` per controller (and per driver).
* **Teleop.** Deadman button (RB), Cartesian speed and acceleration limits, a workspace box around
  the start and a z floor, IK that holds instead of jumping, and hold-then-stop on lost input.
* **Gripper.** Force ≤ 30 %, speed ≤ 50 %, width ≥ 10 mm unless asked otherwise.

## What it does not protect against / 막지 못하는 것

* Collisions with the table, fixtures or people: there is no collision model. The hand-eye
  runner and teleop check only a z floor and a box. The controller's collision detection and the
  tracking/torque monitor are the remaining layers.
* Wrong kinematic parameters or tool length: the z floor and the box are only as good as
  `--params` / `--tool-length`.
* Other software: the motion lock is cooperative; the teach pendant, the web UI and other tools
  can still move the arm.
* A driver started with `--fake`: it publishes on the ROS graph like a real one. Never run it on a
  graph where a real driver or real clients run.

## Before the first motion / 첫 모션 전

1. Clear the workspace; a person at the e-stop.
2. `fr3-check` reports READY (`fr3-power recover --yes` if not).
3. Start the driver, then `fr3-probe` (no motion), then `fr3-probe --execute` (j6 ±0.3°).
4. Run the example or tool without `--execute` and read what it would do.
5. Start slow (`--speed slow`, low `max_joint_speed_deg`) and raise limits only after watching
   the arm.
