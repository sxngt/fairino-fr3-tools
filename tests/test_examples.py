"""Examples run as dry runs without a controller or ROS. / 예제의 드라이 런 (제어기·ROS 없음)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
START = ["76.9", "-68.8", "57", "-78.2", "-90", "-13.1"]


@pytest.mark.parametrize(
    "args, expect",
    [
        (["move_joints.py", "--from", *START, "--j6", "5"], "dry run"),
        (["move_joints.py", "--from", *START, "--j1", "40"], "refused"),
        (["stream_circle.py", "--from", *START], "dry run"),
        (["gripper.py"], "dry run"),
    ],
)
def test_example_dry_runs(args, expect):
    out = subprocess.run([sys.executable, str(EXAMPLES / args[0]), *args[1:]], capture_output=True, text=True)
    assert expect in out.stdout + out.stderr
