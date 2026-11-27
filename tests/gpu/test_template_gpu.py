"""The task template (templates/task) on the GPU: its environment steps and its terms compute."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HERE = pathlib.Path(__file__).parent
TEMPLATE = HERE.parents[1] / "templates" / "task"


def test_the_template_task():
    out = subprocess.run(
        [sys.executable, str(HERE / "template_check.py")],
        capture_output=True, text=True, check=True, timeout=1200,
    ).stdout
    r = json.loads(next(line for line in out.splitlines() if line.startswith("RESULT "))[7:])
    print(r)
    assert r["action_terms"] == ["left_arm", "left_hand"] and r["action"] == 13
    assert r["observation"] == [16, 52 + 3 + 3]  # joint states, box position, palm position
    assert min(r["start_spread_m"]) > 0.05  # the boxes are placed at random ...
    assert r["start_offset_max_m"] <= 0.08 + 1e-6  # ... within the start area
    assert r["rest_height_error_m"] < 2e-3  # each comes to rest on the table
    assert r["rest_xy_drift_m"] < 5e-3 and r["rest_max_speed_m_s"] < 1e-2
    assert r["rewards_finite"] and r["reward_mean"] > 0
    assert not r["ended_while_holding"]
    assert r["dropped_terminated"] == [True] + [False] * 15  # only the dropped box's episode ends


def test_the_template_run_script():
    env = {**os.environ, "PYTHONPATH": str(TEMPLATE / "src")}
    out = subprocess.run(
        [sys.executable, str(TEMPLATE / "run.py"), "--seconds", "1"],
        capture_output=True, text=True, check=True, timeout=1200, env=env, cwd=TEMPLATE,
    ).stdout
    print(out)
    assert "observation: (16, 58), action: 13" in out
    assert "1.0 s  reward" in out
