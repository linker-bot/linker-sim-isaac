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


def test_simulation_dependency_stack_targets_isaac_sim_6() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    assert project["project"]["requires-python"] == "==3.12.*"
    simulation = set(project["project"]["optional-dependencies"]["simulation"])
    assert "isaacsim[all,extscache]==6.0.1.0" in simulation
    assert "torch==2.11.0" in simulation
    assert "torchvision==0.26.0" in simulation
    assert "torchaudio==2.11.0" in simulation
    assert "warp-lang==1.13.0" in simulation
    # 显存验收直接使用 cuda.bindings.nvml，不能只依赖 cuRobo 当前恰好携带的
    # transitive dependency；精确版本同时冻结与已验证 cu12 闭包的兼容性。
    assert "cuda-bindings[all]==12.9.7" in simulation
    assert any(
        dependency.startswith("nvidia-curobo[cu12] @ git+")
        and dependency.endswith("4ea77366ca48ee453e7df139e39fa6532af49f3b")
        for dependency in simulation
    )
    assert not any(dependency.startswith("usd-core") for dependency in simulation)
    assert "override-dependencies" not in project["tool"]["uv"]


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
