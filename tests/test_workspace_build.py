from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

import workspace_build


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_declares_the_rejecting_workspace_build_backend() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    assert project["build-system"] == {
        "requires": [],
        "build-backend": "workspace_build",
        "backend-path": ["."],
    }
    assert project["tool"]["uv"]["package"] is False


@pytest.mark.parametrize(
    "build_hook, args",
    [
        (workspace_build.build_wheel, ("dist",)),
        (workspace_build.build_sdist, ("dist",)),
        (workspace_build.build_editable, ("dist",)),
    ],
)
def test_distribution_builds_are_rejected_explicitly(build_hook, args) -> None:
    with pytest.raises(RuntimeError, match="workspace application"):
        build_hook(*args)
