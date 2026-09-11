#!/usr/bin/env python3
"""Validate release evidence and build the versioned source workspace archive."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import cast
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILE = REPO_ROOT / "pyproject.toml"
PACKAGE_FILE = REPO_ROOT / "src" / "linkerbot_sim" / "__init__.py"
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.md"
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[.a-zA-Z0-9-]+)?$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ReleaseError(RuntimeError):
    """Raised when a release input or repository invariant is invalid."""


def _run_git(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise ReleaseError(f"git {' '.join(arguments)} failed: {detail}") from exc
    return result.stdout.strip()


def project_version() -> str:
    with PROJECT_FILE.open("rb") as stream:
        value = tomllib.load(stream)["project"]["version"]
    if not isinstance(value, str) or not VERSION_RE.fullmatch(value):
        raise ReleaseError("pyproject.toml contains an invalid project version")
    return value


def runtime_version() -> str:
    tree = ast.parse(
        PACKAGE_FILE.read_text(encoding="utf-8"), filename=str(PACKAGE_FILE)
    )
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ReleaseError("linkerbot_sim.__version__ must be a string literal")


def changelog_notes(version: str, content: str) -> str:
    heading = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$",
        re.MULTILINE,
    )
    match = heading.search(content)
    if match is None:
        raise ReleaseError(f"CHANGELOG.md has no dated [{version}] release section")
    next_heading = re.search(r"^## \[", content[match.end() :], re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(content)
    notes = content[match.end() : end].strip()
    if not notes:
        raise ReleaseError(f"CHANGELOG.md [{version}] release notes are empty")
    return notes + "\n"


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReleaseError(f"{label} must be a JSON object with string keys")
    return cast(Mapping[str, object], value)


def validate_simulation_run(
    payload: object,
    *,
    run_id: int,
    repository: str,
    expected_head: str,
) -> None:
    run = _mapping(payload, label="Simulation run")
    if run.get("databaseId") != run_id:
        raise ReleaseError(
            "Simulation evidence run ID does not match the dispatch input"
        )
    if run.get("workflowName") != "Simulation" or run.get("conclusion") != "success":
        raise ReleaseError("release requires a successful Simulation workflow run")
    if run.get("headSha") != expected_head:
        raise ReleaseError(
            "Simulation run commit does not match the release tag commit"
        )

    url = run.get("url")
    if not isinstance(url, str):
        raise ReleaseError("Simulation run URL is missing")
    parsed = urlparse(url)
    expected_path = f"/{repository}/actions/runs/{run_id}"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.path != expected_path
    ):
        raise ReleaseError(
            "Simulation run URL does not belong to this repository and run"
        )


def validate_repository(tag: str) -> tuple[str, str]:
    version = project_version()
    if runtime_version() != version:
        raise ReleaseError("project and runtime versions do not match")
    if tag != f"v{version}":
        raise ReleaseError(f"release tag must be exactly v{version}")
    if _run_git("cat-file", "-t", f"refs/tags/{tag}") != "tag":
        raise ReleaseError("release tag must exist and be annotated")

    head = _run_git("rev-parse", "HEAD")
    tagged_head = _run_git("rev-parse", f"refs/tags/{tag}^{{commit}}")
    if tagged_head != head:
        raise ReleaseError("release tag does not point to the checked-out commit")
    if _run_git("status", "--porcelain", "--untracked-files=no"):
        raise ReleaseError("tracked files changed while preparing the release")
    notes = changelog_notes(version, CHANGELOG_FILE.read_text(encoding="utf-8"))
    return version, notes


def prepare_release(
    *,
    tag: str,
    simulation_run_id: int,
    simulation_run_json: Path,
    repository: str,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    if simulation_run_id <= 0:
        raise ReleaseError("Simulation run ID must be a positive integer")
    if not REPOSITORY_RE.fullmatch(repository):
        raise ReleaseError("repository must use the owner/name form")
    version, notes = validate_repository(tag)
    try:
        run_payload = json.loads(simulation_run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot read Simulation run evidence: {exc}") from exc
    validate_simulation_run(
        run_payload,
        run_id=simulation_run_id,
        repository=repository,
        expected_head=_run_git("rev-parse", "HEAD"),
    )

    resolved_output = output_dir.resolve()
    if REPO_ROOT not in resolved_output.parents:
        raise ReleaseError("output directory must be inside the repository")
    resolved_output.mkdir(parents=True, exist_ok=True)
    archive = resolved_output / f"linker-sim-isaac-{version}.tar.gz"
    notes_file = resolved_output / "RELEASE_NOTES.md"
    checksums = resolved_output / "SHA256SUMS"
    _run_git(
        "archive",
        "--format=tar.gz",
        f"--prefix=linker-sim-isaac-{version}/",
        f"--output={archive}",
        tag,
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    notes_file.write_text(notes, encoding="utf-8")
    checksums.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, notes_file, checksums


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--simulation-run-id", required=True, type=int)
    parser.add_argument("--simulation-run-json", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        archive, notes, checksums = prepare_release(
            tag=args.tag,
            simulation_run_id=args.simulation_run_id,
            simulation_run_json=args.simulation_run_json,
            repository=args.repository,
            output_dir=args.output_dir,
        )
    except ReleaseError as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"archive": str(archive), "checksums": str(checksums), "notes": str(notes)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
