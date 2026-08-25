#!/usr/bin/env python3
"""Reject dependency findings that are new relative to a trusted lock graph."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Literal, cast, overload


class AuditError(RuntimeError):
    """Raised when uv could not produce a trustworthy audit report."""


@dataclass(frozen=True)
class Vulnerability:
    package: str
    version: str
    identifier: str
    aliases: frozenset[str]
    summary: str | None
    fix_versions: tuple[str, ...]

    @property
    def identifiers(self) -> frozenset[str]:
        return self.aliases | {self.identifier}


@dataclass(frozen=True)
class AdverseStatus:
    package: str
    status: str
    reason: str | None


@dataclass(frozen=True)
class AuditReport:
    audited_packages: int
    vulnerabilities: tuple[Vulnerability, ...]
    adverse_statuses: tuple[AdverseStatus, ...]


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AuditError(f"{label} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise AuditError(f"{label} contains a non-string key")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise AuditError(f"{label} must be a JSON array")
    return value


@overload
def _text(value: object, *, label: str, optional: Literal[False] = False) -> str: ...


@overload
def _text(value: object, *, label: str, optional: Literal[True]) -> str | None: ...


def _text(value: object, *, label: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AuditError(f"{label} must be a non-empty string")
    return value


def parse_report(payload: object) -> AuditReport:
    root = _mapping(payload, label="audit report")
    summary = _mapping(root.get("summary"), label="audit report summary")

    def summary_count(name: str) -> int:
        value = summary.get(name)
        if not isinstance(value, int) or value < 0:
            raise AuditError(f"summary.{name} must be a non-negative integer")
        return value

    audited_packages = summary_count("audited_packages")
    vulnerability_count = summary_count("vulnerabilities")
    adverse_status_count = summary_count("adverse_statuses")

    vulnerabilities: list[Vulnerability] = []
    for index, raw in enumerate(
        _sequence(root.get("vulnerabilities"), label="vulnerabilities")
    ):
        finding = _mapping(raw, label=f"vulnerabilities[{index}]")
        dependency = _mapping(
            finding.get("dependency"), label=f"vulnerabilities[{index}].dependency"
        )
        identifier = _text(finding.get("id"), label=f"vulnerabilities[{index}].id")
        package = _text(
            dependency.get("name"), label=f"vulnerabilities[{index}].dependency.name"
        )
        version = _text(
            dependency.get("version"),
            label=f"vulnerabilities[{index}].dependency.version",
        )
        aliases = frozenset(
            _text(value, label=f"vulnerabilities[{index}].aliases[]")
            for value in _sequence(
                finding.get("aliases", []),
                label=f"vulnerabilities[{index}].aliases",
            )
        )
        fix_versions = tuple(
            _text(value, label=f"vulnerabilities[{index}].fix_versions[]")
            for value in _sequence(
                finding.get("fix_versions", []),
                label=f"vulnerabilities[{index}].fix_versions",
            )
        )
        vulnerabilities.append(
            Vulnerability(
                package=package,
                version=version,
                identifier=identifier,
                aliases=aliases,
                summary=_text(
                    finding.get("summary"),
                    label=f"vulnerabilities[{index}].summary",
                    optional=True,
                ),
                fix_versions=fix_versions,
            )
        )

    adverse_statuses: list[AdverseStatus] = []
    for index, raw in enumerate(
        _sequence(root.get("adverse_statuses"), label="adverse_statuses")
    ):
        finding = _mapping(raw, label=f"adverse_statuses[{index}]")
        adverse_statuses.append(
            AdverseStatus(
                package=_text(
                    finding.get("name"), label=f"adverse_statuses[{index}].name"
                ),
                status=_text(
                    finding.get("status"), label=f"adverse_statuses[{index}].status"
                ),
                reason=_text(
                    finding.get("reason"),
                    label=f"adverse_statuses[{index}].reason",
                    optional=True,
                ),
            )
        )

    if len(vulnerabilities) != vulnerability_count:
        raise AuditError(
            "summary.vulnerabilities does not match the vulnerability records"
        )
    if len(adverse_statuses) != adverse_status_count:
        raise AuditError(
            "summary.adverse_statuses does not match the adverse-status records"
        )

    return AuditReport(
        audited_packages=audited_packages,
        vulnerabilities=tuple(vulnerabilities),
        adverse_statuses=tuple(adverse_statuses),
    )


def audit_project(project: Path, *, uv_executable: str = "uv") -> AuditReport:
    if (
        not (project / "pyproject.toml").is_file()
        or not (project / "uv.lock").is_file()
    ):
        raise AuditError(f"{project} must contain pyproject.toml and uv.lock")
    command = [
        uv_executable,
        "audit",
        "--no-config",
        "--frozen",
        "--python-version",
        "3.12",
        "--python-platform",
        "x86_64-unknown-linux-gnu",
        "--output-format",
        "json",
    ]
    completed = subprocess.run(
        command,
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    # uv returns 1 when findings exist. Any other non-zero status means that no
    # complete report was produced and must fail closed.
    if completed.returncode not in {0, 1}:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise AuditError(
            f"uv audit failed for {project} with exit code {completed.returncode}: "
            f"{detail}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AuditError(
            f"uv audit returned invalid JSON for {project}: {exc}"
        ) from exc
    report = parse_report(payload)
    has_findings = bool(report.vulnerabilities or report.adverse_statuses)
    if completed.returncode != int(has_findings):
        raise AuditError(
            f"uv audit exit status and report findings disagree for {project}"
        )
    return report


def new_findings(
    base: AuditReport, head: AuditReport
) -> tuple[tuple[Vulnerability, ...], tuple[AdverseStatus, ...]]:
    base_identifiers: dict[str, set[str]] = defaultdict(set)
    for finding in base.vulnerabilities:
        base_identifiers[finding.package.casefold()].update(finding.identifiers)

    vulnerabilities: list[Vulnerability] = []
    head_identifiers: dict[str, set[str]] = defaultdict(set)
    for finding in head.vulnerabilities:
        package = finding.package.casefold()
        if not finding.identifiers.isdisjoint(base_identifiers[package]):
            continue
        # OSV can return the same advisory through GHSA and PYSEC records. Treat
        # intersecting alias sets as one finding so the gate and its output do not
        # inflate a single regression.
        if not finding.identifiers.isdisjoint(head_identifiers[package]):
            continue
        vulnerabilities.append(finding)
        head_identifiers[package].update(finding.identifiers)
    base_statuses = {
        (finding.package.casefold(), finding.status.casefold())
        for finding in base.adverse_statuses
    }
    adverse_statuses = tuple(
        finding
        for finding in head.adverse_statuses
        if (finding.package.casefold(), finding.status.casefold()) not in base_statuses
    )
    return tuple(vulnerabilities), adverse_statuses


def _print_findings(
    vulnerabilities: Sequence[Vulnerability],
    adverse_statuses: Sequence[AdverseStatus],
) -> None:
    if vulnerabilities:
        print("New vulnerability findings:")
        for finding in vulnerabilities:
            detail = finding.summary or "No summary provided"
            fixed = (
                f"; fixed in {', '.join(finding.fix_versions)}"
                if finding.fix_versions
                else "; no fixed version reported"
            )
            print(
                f"- {finding.package} {finding.version}: "
                f"{finding.identifier} - {detail}{fixed}"
            )
    if adverse_statuses:
        print("New adverse project statuses:")
        for finding in adverse_statuses:
            reason = f" - {finding.reason}" if finding.reason else ""
            print(f"- {finding.package}: {finding.status}{reason}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project", type=Path, required=True)
    parser.add_argument("--head-project", type=Path, required=True)
    parser.add_argument("--uv-executable", default="uv")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        base = audit_project(
            arguments.base_project.resolve(), uv_executable=arguments.uv_executable
        )
        head = audit_project(
            arguments.head_project.resolve(), uv_executable=arguments.uv_executable
        )
        vulnerabilities, adverse_statuses = new_findings(base, head)
    except AuditError as exc:
        print(f"dependency audit error: {exc}", file=sys.stderr)
        return 2

    print(
        "Dependency audit comparison: "
        f"base={base.audited_packages} packages/"
        f"{len(base.vulnerabilities)} vulnerabilities/"
        f"{len(base.adverse_statuses)} adverse statuses; "
        f"head={head.audited_packages} packages/"
        f"{len(head.vulnerabilities)} vulnerabilities/"
        f"{len(head.adverse_statuses)} adverse statuses."
    )
    if vulnerabilities or adverse_statuses:
        _print_findings(vulnerabilities, adverse_statuses)
        return 1
    print("No new dependency vulnerability or adverse-status findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
