# FR3 pitfalls

Things that cost hours on a real FR3 (controller firmware v3.8.6, Python SDK v2.1.6) and are not in
the vendor manuals. Each item says what you see, why, and what `fr3tools` does about it.

실제 FR3(펌웨어 v3.8.6, SDK v2.1.6)에서 시간을 잡아먹은 것들. 증상, 원인, `fr3tools` 의 대응 순서.

## Controller and SDK

1. **The real-time state port serves one client.** Port 20004 (8 ms state packets) goes to the
   first process that connects. A second SDK process gets nothing, and the SDK then loops on
   reconnect; while it loops, *every* XML-RPC call blocks, `StopMotion` included.
   `sdk.connect()` waits briefly for packets and otherwise turns the handle XML-RPC-only
   (`silence_state_thread`). Read the state from the driver (`/fr/status`) instead.
   상태 포트(20004)는 한 클라이언트만 받는다. 두 번째 SDK 프로세스는 재접속 루프에 빠지고 그동안 모든
   XML-RPC 호출(StopMotion 포함)이 막힌다.
2. **A failed connect poisons the SDK.** `Robot.RPC.is_conect` is a class-wide flag; after one
   failure every later `RPC()` fails too. `sdk.connect()` resets it.
3. **ServoJ is ignored after a stop until the errors are reset.** After `StopMotion` or a safety
   trip the controller accepts `ServoJ` calls and does nothing. Call `ResetAllError` before
   `ServoMoveStart` (`SdkArm.servo_begin` does). The driver's error flag clears one status cycle
   after the reset, so clients retry the start briefly (`FR3Ros.start_streaming`).
   정지 후에는 에러 리셋 전까지 ServoJ 를 무시한다.
4. **ServoJ needs AUTO mode, servo enable, no error and no safety stop.** Drag teaching needs MANUAL.
   `fr3-check` lists what is missing; `fr3-power recover --yes` waits for the e-stop release, then
   resets, enables and switches to AUTO.
5. **Error 29 is a joint limit.** A `ServoJ` stream ends with return code 29 when a joint reaches
   its soft limit (j6 swinging round during a yaw turn, for example). Keep about 10° of margin and
   wrap IK solutions next to the current joints (`FR3Kinematics.wrap_near`).
   ServoJ 반환값 29 = 관절 리밋. 약 10° 여유를 두고 해를 현재 관절 옆으로 감는다.
6. **The SDK's XML-RPC proxy is not thread safe.** Two threads calling at once raise
   `CannotSendRequest`; on the robot this ended the servo stream exactly at a gripper release.
   One lock around every call (`SdkArm.call`). A stop waits at most 0.5 s for it.
   SDK 의 ServerProxy 는 스레드 안전하지 않다: 모든 호출에 잠금 하나.
7. **A gripper call can take more than a second.** Never call it from the servo loop; send it from
   another thread (the driver's action) under the same lock.
8. **The state packet can freeze.** When `frame_cnt` stops changing for 1 s the driver stops
   publishing `/joint_states` instead of repeating a stale pose that clients would servo against.
9. **The nominal DH parameters are about 1.5 mm off.** The controller applies a factory DH
   compensation to its own kinematics. Fit your unit with `fr3-calib-dh` (it uses
   `GetForwardKin`, a computation; the arm does not move); the fit matches the controller to
   about 0.03 mm.
10. **j2 and j4 run from -265° to 85°.** Analytic IK returns angles in [-π, π); a solution such
    as j2 = 100° is reachable only as -260°. Every joint range is under 360°, so the wrap is unique.
11. **Home (0, -90, 0, -90, 0, 0) is a wrist singularity** (j5 = 0, j4 parallel to j6). Cartesian
    moves started there fail; start from a bent posture.
12. **Match the SDK tag to the firmware.** The SDK repository has one tag per firmware
    (`v2.1.6_robot3.8.6`). The ros2_control hardware packages in FAIRINO's `frcobot_ros2` bundle a
    `libfairino` built for other firmware, which is one reason the driver here uses the Python SDK.

## Streaming motion

13. **The servo target must not jump.** The stream ends when a target is more than
    `servo_max_jump_deg` (15°) from the current command. Stream continuous targets; a branch flip
    or a wrist turn shows up as a jump and is refused, not executed.
14. **ServoJ starts late; the start tolerance must allow for it.** The arm starts following about
    100 ms after the first `ServoJ`. At 150°/s the commanded joints ramp 2.4° per 16 ms tick, more
    than a 2° tracking tolerance, so the safety monitor tripped within 0.1-0.4 s of every start.
    Keep `max_joint_speed_deg` low (30-40) or raise `start_tol_deg` (8 worked with 40°/s).
    ServoJ 는 약 100 ms 늦게 움직이므로 추종 편차 허용치가 그 지연을 감당해야 한다.
15. **Limit acceleration, not just speed.** A step change of joint speed shakes the base. The
    stream limits speed and acceleration (60°/s²) and brakes so that it stops within the
    remaining distance without overshoot.
16. **16 ms is the practical ServoJ period over XML-RPC.** Shorter periods jitter; a network delay
    longer than one period stalls the arm on its last point and trips the tracking monitor.
17. **A position-servoed arm gives no stall signal on contact.** A blocked descent does not stop:
    the arm pushes on, up to the tracking or torque limits. Near a goal the arm also creeps slowly
    (about 0.3 mm per 0.1 s when the target leads by a few mm), which a "no progress" detector
    mistakes for contact. Detect contact from joint torques or the tracking error, not from
    stalled progress.
    위치 서보 팔은 접촉해도 멈추지 않는다: 접촉은 토크나 추종 오차로 판단한다.
18. **MoveIt plans faster than the driver accepts.** Trajectories with a peak joint speed above
    `max_joint_speed_deg` (30°/s) are rejected; lower MoveIt's velocity scaling or retime.

## AG-95 gripper

19. **Gripper width feedback is not a grasp sensor.** Through the controller, `gripper_position`
    follows the command: it reported the commanded 12 mm with a 25 mm part between the fingers.
    Check a grasp with a camera, a lift test, or the Modbus grip state (direct RS-485 wiring).
    제어기 경유 폭 피드백은 명령값을 따라간다: 파지 판정에 쓰지 않는다.
20. **Activate after every controller power cycle.** Until `ActGripper(1, 1)` the gripper ignores
    commands (`MoveGripper` returned 73). Activation sweeps the fingers over the full stroke: clear
    them first. A fault after the first activation needs `ActGripper(1, 0)` and then `(1, 1)`.
21. **MoveGripper's blocking flag returns immediately.** Poll the position until it settles.
22. **Check the direction.** On the reference setup 100 % = open; `fr3-gripper probe --execute`
    shows it, `--open-pct 0` flips it.
23. **Register it as DH-Robotics** (`SetGripperConfig(4, 0, 0, 0)` or the web UI) once per
    controller; the PGI entry uses the same register map.

## Calibration and cameras

24. **Intrinsics need tilted views.** Views parallel to the image plane cannot separate the focal
    length from the distance; `fr3-camera-calib` waits for a 10° spread of board normals and
    rejects blurred views. Recalibrate after changing the resolution.
25. **USB cameras buffer frames.** A frame read right after the robot stops may show it moving;
    `CameraSource` drops stale frames and the hand-eye runner waits for frames newer than the
    settle time.
26. **The ArUco detector has a viewing-angle limit** that is lower than the geometry suggests. The
    hand-eye runner learns it: a predicted-visible pose that is missed tightens the facing limit.
27. **Kinematic error, not pixel noise, is the floor of hand-eye accuracy.** Each pose's corners
    are whitened with the kinematic error model; without it the estimate is overconfident.
