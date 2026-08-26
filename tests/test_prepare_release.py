from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import prepare_release as release


REPOSITORY = "linker-bot/linker-sim-isaac"


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _release_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, Path]:
    package = tmp_path / "src" / "linkerbot_sim"
    package.mkdir(parents=True)
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname = "linker-sim-isaac"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    package_file = package / "__init__.py"
    package_file.write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [1.2.3] - 2026-08-26\n\n- Stable release.\n",
        encoding="utf-8",
    )
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "release")
    _git(tmp_path, "tag", "-a", "v1.2.3", "-m", "release 1.2.3")
    head = _git(tmp_path, "rev-parse", "HEAD")

    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release, "PROJECT_FILE", project)
    monkeypatch.setattr(release, "PACKAGE_FILE", package_file)
    monkeypatch.setattr(release, "CHANGELOG_FILE", changelog)
    evidence = tmp_path / "simulation-run.json"
    evidence.write_text(
        json.dumps(
            {
                "conclusion": "success",
                "databaseId": 123,
                "headSha": head,
                "url": f"https://github.com/{REPOSITORY}/actions/runs/123",
                "workflowName": "Simulation",
            }
        ),
        encoding="utf-8",
    )
    return head, evidence


def test_prepare_release_creates_versioned_archive_and_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, evidence = _release_repository(tmp_path, monkeypatch)

    archive, notes, checksums = release.prepare_release(
        tag="v1.2.3",
        simulation_run_id=123,
        simulation_run_json=evidence,
        repository=REPOSITORY,
        output_dir=tmp_path / "dist",
    )

    assert notes.read_text(encoding="utf-8") == "- Stable release.\n"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert checksums.read_text(encoding="utf-8") == f"{digest}  {archive.name}\n"
    with tarfile.open(archive) as source:
        names = source.getnames()
    assert "linker-sim-isaac-1.2.3/pyproject.toml" in names
    assert "linker-sim-isaac-1.2.3/simulation-run.json" not in names


def test_release_tag_must_be_annotated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release_repository(tmp_path, monkeypatch)
    _git(tmp_path, "tag", "--delete", "v1.2.3")
    _git(tmp_path, "tag", "v1.2.3")

    with pytest.raises(release.ReleaseError, match="annotated"):
        release.validate_repository("v1.2.3")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("databaseId", 124, "run ID"),
        ("conclusion", "failure", "successful Simulation"),
        ("workflowName", "Quality", "successful Simulation"),
        ("headSha", "0" * 40, "commit"),
        ("url", "https://github.com/other/repository/actions/runs/123", "URL"),
    ),
)
def test_simulation_evidence_must_match_release(
    field: str, value: object, message: str
) -> None:
    payload: dict[str, object] = {
        "conclusion": "success",
        "databaseId": 123,
        "headSha": "a" * 40,
        "url": f"https://github.com/{REPOSITORY}/actions/runs/123",
        "workflowName": "Simulation",
    }
    payload[field] = value

    with pytest.raises(release.ReleaseError, match=message):
        release.validate_simulation_run(
            payload,
            run_id=123,
            repository=REPOSITORY,
            expected_head="a" * 40,
        )


def test_changelog_requires_a_dated_exact_version_section() -> None:
    with pytest.raises(release.ReleaseError, match="no dated"):
        release.changelog_notes("1.2.3", "## [Unreleased]\n\n- pending\n")
