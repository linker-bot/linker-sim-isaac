from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CI_INCLUDE = {
    "src/linkerbot_sim/configuration",
    "scripts/check_dependency_audit_delta.py",
    "scripts/check_markdown_links.py",
    "scripts/check_pure_coverage.py",
    "scripts/check_repository_ruleset.py",
    "scripts/update_architecture_inventory.py",
    "scripts/validate_mode_config.py",
    "workspace_build.py",
}


def _json(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_editor_config_keeps_repository_discovery_scope() -> None:
    config = _json("pyrightconfig.json")

    assert config["include"] == ["src", "scripts", "tests"]
    assert config["extraPaths"] == ["src"]


def test_ci_config_defines_a_zero_diagnostic_baseline() -> None:
    config = _json("pyrightconfig.ci.json")

    assert config["extends"] == "./pyrightconfig.json"
    assert set(config["include"]) == CI_INCLUDE
    assert config["pythonVersion"] == "3.12"
    assert config["pythonPlatform"] == "Linux"
    assert config["typeCheckingMode"] == "standard"
    assert config["venvPath"] == "."
    assert config["venv"] == ".venv-dev"
    assert "ignore" not in config
    assert not any(key.startswith("report") for key in config)


@pytest.mark.parametrize("relative_path", sorted(CI_INCLUDE))
def test_ci_type_check_path_exists(relative_path: str) -> None:
    path = ROOT / relative_path

    assert path.exists()
    assert path.is_dir() or path.suffix == ".py"


def test_dev_environment_pins_the_type_checker() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"pyright==1.1.411"' in project


def test_quality_gate_runs_the_ci_type_check() -> None:
    justfile = (ROOT / "Justfile").read_text(encoding="utf-8")

    assert (
        "type-check:\n    {{uv_dev}} pyright --project pyrightconfig.ci.json"
        in justfile
    )
    quality_line = next(
        line for line in justfile.splitlines() if line.startswith("quality:")
    )
    assert "type-check" in quality_line.split()
