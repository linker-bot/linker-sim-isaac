"""Check the installation plan without installing the simulation on the test host."""

import os
import subprocess
from pathlib import Path


def test_runtime_defaults_exclude_development_tools_on_aarch64(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [
        "uv", "sync", "--directory", str(root), "--locked", "--dry-run",
        "--python-platform", "aarch64-manylinux_2_35",
    ]
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(tmp_path / ".venv")}
    runtime = subprocess.run(
        [*command, "--no-build-package", "ruckig"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert runtime.returncode == 0, runtime.stderr
    for package in ("pytest", "ruff", "ruckig"):
        assert f" + {package}==" not in runtime.stderr

    development = subprocess.run(
        [*command, "--group", "dev"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert development.returncode == 0, development.stderr
    for package in ("pytest", "ruff", "ruckig"):
        assert f" + {package}==" in development.stderr
