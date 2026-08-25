from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

import linkerbot_sim
from linkerbot_sim.mirror.cli import build_parser
from scripts.kaleidoscope_viewer import parse_args as parse_viewer_args


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_and_project_metadata_versions_match() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    version = project["project"]["version"]
    assert isinstance(version, str)
    assert version
    assert linkerbot_sim.__version__ == version
    assert linkerbot_sim.__all__ == ["REPO_ROOT", "__version__"]


@pytest.mark.parametrize(
    "parse",
    (
        lambda argv: build_parser().parse_args(argv),
        parse_viewer_args,
    ),
)
def test_product_clis_report_the_workspace_version(parse, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == (
        f"linker-sim-isaac {linkerbot_sim.__version__}\n"
    )


@pytest.mark.parametrize(
    "template",
    (
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/question.md",
    ),
)
def test_support_templates_request_workspace_version_and_commit(template: str) -> None:
    content = (ROOT / template).read_text(encoding="utf-8")

    assert "`python scripts/mirror.py --version`" in content
    assert "`git rev-parse HEAD`" in content
