from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _project() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_ruff_policy_does_not_inherit_release_defaults() -> None:
    project = _project()
    ruff = project["tool"]["ruff"]
    lint = ruff["lint"]

    assert ruff["target-version"] == "py312"
    assert lint["select"] == ["E4", "E7", "E9", "F"]
    assert "extend-select" not in lint


def test_ruff_formatter_keeps_markdown_out_of_the_python_format_gate() -> None:
    project = _project()
    formatter = project["tool"]["ruff"]["format"]

    assert formatter["exclude"] == ["*.md"]


def test_dev_environment_pins_one_ruff_release() -> None:
    project = _project()
    dev_dependencies = project["project"]["optional-dependencies"]["dev"]

    ruff_dependencies = [
        dependency
        for dependency in dev_dependencies
        if dependency.partition("==")[0] == "ruff"
    ]
    assert len(ruff_dependencies) == 1
    assert "==" in ruff_dependencies[0]
