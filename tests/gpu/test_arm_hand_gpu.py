"""ArmAction and HandAction on the GPU: the recorded drive reference, the limits, the hand pressing on
stops, pose commands."""

import json
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HERE = pathlib.Path(__file__).parent


def run(*args) -> tuple[dict, str]:
    out = subprocess.run(
        [sys.executable, str(HERE / "arm_hand_check.py"), *map(str, args)],
        capture_output=True, text=True, check=True, timeout=1800,
    ).stdout
    return json.loads(next(line for line in out.splitlines() if line.startswith("RESULT "))[7:]), out


def test_drives_reproduce_the_recorded_reference_in_closed_loop():
    """Without limits and with the recorded per-step references (left arm) or Hermite
    points (right arm), the simulation follows the recording to float32 precision until the
    hands start pressing (measured: hand 2.0e-6 rad).

    In the last phase (from 3 s) the thumb presses against the index until the thumb-yaw drive
    reaches its torque guard (4.28 s) and pushes with a constant 9 N·m: the thumb's position is
    then set by the contact alone, and float32-level differences of the replay (the left arm's
    per-step references instead of its points) grow to about a milliradian (measured:
    9.7e-4 rad)."""
    r, _ = run("reference")
    print(r)
    d = r["max_deviation"]
    assert d["arm_q"] < 1e-5 and d["arm_target"] < 1e-5
    assert d["hand_q"] < 1e-4 and d["hand_target"] < 1e-4
    assert d["arm_q_pressing"] < 1e-5 and d["arm_target_pressing"] < 1e-5
    assert d["hand_q_pressing"] < 5e-3 and d["hand_target_pressing"] < 5e-3

def test_400_environments_move_within_the_real_limits():
    r, out = run("limits", 400, 4.0)
    print(r, "linesearch warnings:", out.count("linesearch"))
    assert r["finite"]
    assert max(r["max_reference_speed_over_limit"]) <= 1 + 1e-6
    assert max(r["max_measured_speed_over_limit"]) < 1.05  # the servo tracks the reference
    assert r["max_wrist_outside_octagon_rad"] < 1e-3
    assert r["limit_hits_mean"]["left"] > 0  # its wrist goals lie outside the octagon


def test_palm_point_pose_commands_reach_their_goal():
    r, _ = run("pose")
    print(r)
    for side in ("left", "right"):
        assert r["settle_time_2mm_s"][side] < 1.5  # 5 cm
        assert max(r["final_error_m"][side]) < 1e-4


def test_hand_pressing_on_stops_stays_within_its_guard_and_reopens():
    """The hand's torque guard: pressing on stops for 12 s, no drive asks for more than its
    force limit less 1 N·m, and the hand opens again (on MuJoCo 4 a drive held at its limit
    oscillated there and the motor model stopped the finger for good)."""
    r, _ = run("press")
    print(r)
    assert r["max_torque_over_cap"] <= -1 + 1e-3
    assert r["open"]["error"] < 1e-3 and r["open again"]["error"] < 1e-3
    assert r["open"]["speed"] < 1e-3 and r["open again"]["speed"] < 1e-3


def test_non_finite_commands_after_a_reset_hold_home():
    """NaN keeps the previous command; after a reset that is home, for the arms as for the hands."""
    r, _ = run("hold")
    print(r)
    assert r["max_drift_from_home_rad"] < 1e-3
