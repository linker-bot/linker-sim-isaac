"""The ycb-grasp scene: every object comes to rest when dropped; the buttons drive a grasp
to a result, headless and in its window."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
HERE = Path(__file__).parent


def report(*args) -> dict:
    result = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    line = next((x for x in result.stdout.splitlines() if x.startswith("REPORT ")), None)
    assert line is not None, result.stdout[-3000:] + result.stderr[-3000:]
    return json.loads(line[7:])


def test_every_object_comes_to_rest_and_both_grasps_hold():
    r = report(HERE / "gallery_check.py", "005_tomato_soup_can", "013_apple")
    print(r)
    assert r["objects"] == 31 and r["not_at_rest"] == []
    assert set(r["grasps"]) == {"005_tomato_soup_can", "013_apple"}
    assert all(m.endswith(": grasped.") for m in r["grasps"].values()), r["grasps"]
    assert r["reset_state"] == "empty"


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a display for the viewer")
def test_buttons_in_the_window(tmp_path):
    r = report(HERE / "gallery_viewer_check.py", tmp_path)
    print(r)
    assert r["states"] == ["empty", "settling", "ready", "grasping", "done", "empty"]
    assert ": " in r["result"]
    assert max(r["speeds"]) > 0.3  # ~0.6-0.8x real time measured on the development machine
