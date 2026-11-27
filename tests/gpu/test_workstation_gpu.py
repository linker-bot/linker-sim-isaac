"""LW-v0 on the GPU: 400 environments step, the home pose holds, an object rests on the table."""

import json
import pathlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HERE = pathlib.Path(__file__).parent


def run(script: str, *args) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / script), *map(str, args)],
        capture_output=True, text=True, check=True, timeout=1200,
    ).stdout
    return json.loads(next(line for line in out.splitlines() if line.startswith("RESULT "))[7:])


def test_400_environments_hold_home_and_an_object_rests_on_the_table():
    r = run("lw_v0_check.py", 400, 2.0)
    print(r)
    assert r["finite"]
    assert r["home_max_drift_rad"] < 1e-4
    assert r["box_rest_height_error_m"] < 1e-3  # lies on the table top
    assert r["box_xy_drift_m"] < 1e-3  # did not slide or tip over
    assert r["box_max_speed_m_s"] < 1e-3  # at rest


def test_cameras_see_past_their_housings():
    r = run("camera_check.py")
    print(r)
    # The top camera looks down at the table about 0.55 m away.
    assert r["top"]["near_3cm"] == 0 and 0.4 < r["top"]["median_depth_m"] < 0.8
    for side in ("left", "right"):
        # The depth camera sees past the hand; the RGB camera's view is partly taken by
        # the thumb, which at home is rotated in front of it, 1-3 cm away.
        assert r[f"{side}_depth"]["near_3cm"] < 0.1 and r[f"{side}_depth"]["median_depth_m"] > 0.1
        assert r[f"{side}_rgb"]["near_3cm"] < 0.6
    assert all(c["finite"] for c in r.values())
