"""The playground self-test: the window opens, scripted keys move the arms and hands."""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="needs a display for the viewer"),
]


def test_playground_selftest(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "linker_lab.cli", "play", "--selftest", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=1200,
    )
    report = json.loads((tmp_path / "report.json").read_text())
    print(json.dumps(report, indent=1))
    assert report["passed"], result.stdout[-4000:]
    assert result.returncode == 0
    assert report["speed_median"] > 0.5  # the measured figure is in the report
