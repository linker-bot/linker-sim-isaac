"""The user docs' code blocks that run a simulation, each in its own process (GPU)."""

import re
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from test_docs import NEEDS_SIMULATION, code_blocks  # noqa: E402

BLOCKS = [b for b in code_blocks() if NEEDS_SIMULATION in b[1]]


@pytest.mark.parametrize("where, code", BLOCKS, ids=[w for w, _ in BLOCKS])
def test_code_block(where, code, tmp_path):
    script = tmp_path / "block.py"
    script.write_text(code)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=1200)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    print(result.stdout[-2000:])
    if m := re.search(r"palm point ([\d.]+) mm from its goal", result.stdout):
        assert float(m[1]) < 5.0
