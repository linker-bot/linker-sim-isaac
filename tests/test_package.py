"""Core discovery must not pull in a simulation or research runtime."""

import subprocess
import sys


def test_import_does_not_load_optional_runtimes():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, linker_lab; "
            "assert not any(m.split('.')[0] in "
            "{'isaaclab', 'isaacsim', 'torch', 'warp', 'gymnasium'} for m in sys.modules); "
            "print(linker_lab.__version__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "2.0.0"
