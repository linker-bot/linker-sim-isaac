"""Engine checks on the GPU: `linker-lab doctor` and velocity readouts (see velocity_check.py)."""

import json
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HERE = pathlib.Path(__file__).parent


def velocity_check(hz: float) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "velocity_check.py"), str(hz)],
        capture_output=True, text=True, check=True, timeout=600,
    ).stdout
    return json.loads(next(line for line in out.splitlines() if line.startswith("RESULT "))[7:])


def test_doctor_passes():
    result = subprocess.run(
        [sys.executable, "-m", "linker_lab.cli", "doctor"], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "All checks passed." in result.stdout


def test_velocity_readouts_match_pose_changes():
    r480, r960 = velocity_check(480), velocity_check(960)
    box, pendulum = r480["box"], r480["pendulum"]
    assert box["lowest_z"] < 0.05, "the box should land on the ground"
    # Exact relations: readout == backward difference, up to float32 rounding.
    assert box["linear"] < 1e-3 and box["angular"] < 2e-3
    assert pendulum["joint"] < 1e-3 and pendulum["angular"] < 2e-3
    # A point on a rotating link: chord vs tangent, first order in dt.
    assert pendulum["linear"] < 0.05
    assert 0.4 < r960["pendulum"]["linear"] / pendulum["linear"] < 0.6
