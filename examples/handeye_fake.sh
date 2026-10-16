#!/usr/bin/env bash
# Eye-to-hand calibration on synthetic data: no robot, no camera, no ROS.
# 합성 데이터로 eye-to-hand 보정: 로봇·카메라·ROS 불필요.
set -euo pipefail
out="${1:-calib_out}"

# One full session (real runner, fake robot, synthetic camera, simulated time), three seeds.
# 전체 세션(실제 러너, 가짜 로봇, 합성 카메라, 모의 시간), 시드 세 개.
for seed in 0 1 2; do
  fr3-handeye --fake "$seed" --budget 60 --out "$out/handeye_fake_$seed.json"
done

# The same with a rough camera guess as a weak prior instead of the AX = XB bootstrap
# (the synthetic camera sits near 350 -1050 550 mm, -123 0 30 deg, moved by a few cm per seed).
# AX = XB 부트스트랩 대신 대략의 카메라 위치를 약한 사전값으로.
fr3-handeye --fake 0 --camera-guess 360 -1040 560 -120 2 28 --out "$out/handeye_fake_prior.json"

# Pose-selection strategies against simulated robot time. / 자세 선택 전략 비교.
python -m fr3tools.calib.simulate --seeds 3 --budget 60
