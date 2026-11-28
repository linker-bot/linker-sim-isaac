"""Snapshots on the GPU (``tests/gpu/snapshot_check.py``): restore in the same and in a new
process continues the uninterrupted run, refusals change nothing.

Acceptance: 8 decisions after a restore, joints within 1e-4 rad and the object
within 1e-5 m of the uninterrupted run; control state, counters and simulated time exact.
Measured: about 1e-7 rad and 1e-7 m, float32 rounding.
"""

import json
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HERE = pathlib.Path(__file__).parent
MODES = ("pose", "goal", "samples")


@pytest.fixture(scope="module")
def directory(tmp_path_factory):
    return tmp_path_factory.mktemp("snapshot")


def run(*args) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "snapshot_check.py"), *map(str, args)],
        capture_output=True, text=True, timeout=1800,
    )
    line = next((x for x in out.stdout.splitlines() if x.startswith("RESULT ")), None)
    assert line is not None, out.stdout[-3000:] + out.stderr[-3000:]
    return json.loads(line[7:])


def test_record_the_script(directory):
    assert run("record", directory)["recorded"] == 368


@pytest.mark.parametrize("mode", MODES)
def test_restore_in_the_same_process(directory, mode):
    r = run("same", mode, directory)
    print(r)
    assert r["lift_at_capture_m"] > 0.03  # captured while the hand holds the box in the air
    assert r["error"]["joints_rad"] < 1e-4 and r["error"]["box_m"] < 1e-5
    assert r["control_exact"] and r["clock_exact"] and r["limit_hits_same_tensor"]
    assert r["box_lift_end_m"] > 0.03  # still held


@pytest.mark.parametrize("mode", MODES)
def test_restore_in_a_new_process(directory, mode):
    r = run("new", mode, directory)
    print(r)
    assert r["error"]["joints_rad"] < 1e-4 and r["error"]["box_m"] < 1e-5
    assert r["control_exact"] and r["clock_exact"]
    assert r["box_lift_end_m"] > 0.03


@pytest.mark.parametrize(
    "case, expected",
    [
        ("two_envs", {"capture": "num_envs is 2", "restore": "num_envs is 2"}),
        ("no_contract", {"capture": "no get_state", "restore": "no get_state"}),
        ("delay", {"capture": "with a delay", "restore": "with a delay"}),
        ("modifier", {"capture": "noise model with state", "restore": "noise model with state"}),
        ("cooldown", {"capture": "has a cooldown", "restore": "has a cooldown"}),
        ("config", {"restore": "configuration differs", "versions": "other package versions", "shape": "episode_length"}),
    ],
)
def test_refusals_change_nothing(directory, case, expected):
    r = run("refuse", case, directory)
    print(r)
    raised = {name: (message, modified) for name, message, modified in r["raised"]}
    assert set(raised) == set(expected)
    for name, text in expected.items():
        message, modified = raised[name]
        assert message is not None and text in message, (name, message)
        assert modified is False
    assert r["unchanged"]
