"""Offline command-line tools (no SDK, no ROS). / 오프라인 명령행 도구."""

import csv
import json

import numpy as np
import pytest

from fr3tools.kinematics import FR3Kinematics, FR3Params, matrix_to_pose
from fr3tools.tools import calib_dh_main, errors_main, load_samples, parse_param, probe_offsets, save_samples

TRUTH = FR3Params(d1=0.14, a2=-0.280491, a3=-0.239929, d4=0.102452, d5=0.102123, d6=0.101)


def synthetic(n=30, seed=5):
    rng = np.random.default_rng(seed)
    kin = FR3Kinematics(TRUTH)
    joints = rng.uniform(-150, 80, (n, 6))
    poses = np.array([matrix_to_pose(kin.flange(np.radians(q))) for q in joints])
    return joints, poses


def test_samples_csv_and_json_round_trip(tmp_path):
    joints, poses = synthetic(5)
    path = tmp_path / "s.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["j1", "j2", "j3", "j4", "j5", "j6", "x", "y", "z", "rx", "ry", "rz"])
        w.writerows(np.hstack([joints, poses]).tolist())
    j, p = load_samples(str(path))
    assert np.allclose(j, joints) and np.allclose(p, poses)
    save_samples(str(tmp_path / "s.json"), joints, poses, note="x")
    j2, p2 = load_samples(str(tmp_path / "s.json"))
    assert np.allclose(j2, joints) and np.allclose(p2, poses)


def test_calib_dh_offline_writes_parameters(tmp_path, capsys):
    joints, poses = synthetic()
    log = tmp_path / "log.json"
    save_samples(str(log), joints, poses[:, :3])
    out = tmp_path / "params.json"
    calib_dh_main(["--log", str(log), "--out", str(out)])
    fitted = FR3Params.from_json(str(out))
    assert np.allclose(fitted.dh, TRUTH.dh, atol=1e-6)
    assert json.loads(out.read_text())["fit"]["max_mm"] < 0.01
    assert "after" in capsys.readouterr().out


def test_calib_dh_refuses_a_bad_fit(tmp_path):
    joints, poses = synthetic()
    poses[0, :3] += 5.0  # one bad sample / 나쁜 표본 하나
    log = tmp_path / "log.json"
    save_samples(str(log), joints, poses)
    with pytest.raises(SystemExit):
        calib_dh_main(["--log", str(log), "--out", str(tmp_path / "p.json")])
    assert not (tmp_path / "p.json").exists()


def test_errors_cli(capsys):
    errors_main(["29", "-4", "--state", "5", "1"])
    out = capsys.readouterr().out
    assert "joint limit" in out and "XML-RPC" in out and "main 5 sub 1" in out


def test_probe_profile_and_param_parsing():
    offsets = probe_offsets(duration=2.0, amplitude_deg=0.3, rate=50.0)
    assert offsets[0] == 0.0 and offsets[-1] == pytest.approx(0.0)
    assert np.degrees(offsets.max()) == pytest.approx(0.3)
    assert parse_param("start_tol_deg=8") == ("start_tol_deg", 8.0)
    assert parse_param("name=abc") == ("name", "abc")
