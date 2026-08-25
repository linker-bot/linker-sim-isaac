from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml

from scripts import check_dependency_audit_delta as audit


DEPENDABOT_PATH = Path(".github/dependabot.yml")
WORKFLOW_PATH = Path(".github/workflows/dependency-audit.yml")
PINNED_ACTION_RE = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{label} must be a mapping"
    return value


def _yaml(path: Path) -> Mapping[str, object]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return _mapping(value, label=str(path))


def _report(
    *,
    vulnerabilities: tuple[audit.Vulnerability, ...] = (),
    adverse_statuses: tuple[audit.AdverseStatus, ...] = (),
) -> audit.AuditReport:
    return audit.AuditReport(
        audited_packages=10,
        vulnerabilities=vulnerabilities,
        adverse_statuses=adverse_statuses,
    )


def _vulnerability(
    identifier: str,
    *,
    package: str = "example",
    version: str = "1.0",
    aliases: frozenset[str] = frozenset(),
) -> audit.Vulnerability:
    return audit.Vulnerability(
        package=package,
        version=version,
        identifier=identifier,
        aliases=aliases,
        summary="example advisory",
        fix_versions=("2.0",),
    )


def test_dependabot_updates_uv_and_pinned_actions_in_reviewable_groups() -> None:
    config = _yaml(DEPENDABOT_PATH)
    updates = config.get("updates")
    assert isinstance(updates, list)
    by_ecosystem = {
        str(_mapping(value, label="update")["package-ecosystem"]): _mapping(
            value, label="update"
        )
        for value in updates
    }
    assert set(by_ecosystem) == {"uv", "github-actions"}

    for update in by_ecosystem.values():
        assert update.get("directory") == "/"
        assert _mapping(update.get("schedule"), label="schedule").get("interval") == (
            "weekly"
        )
        limit = update.get("open-pull-requests-limit")
        assert isinstance(limit, str) and 0 < int(limit) <= 6

    uv_groups = _mapping(by_ecosystem["uv"].get("groups"), label="uv groups")
    assert list(uv_groups)[-1] == "application-dependencies"
    simulation = _mapping(uv_groups.get("simulation-runtime"), label="simulation")
    patterns = simulation.get("patterns")
    assert isinstance(patterns, list)
    assert {
        "isaacsim*",
        "torch",
        "warp-lang",
        "cuda-bindings",
        "nvidia-curobo",
    } <= set(patterns)
    application = _mapping(
        uv_groups.get("application-dependencies"), label="application"
    )
    assert application.get("patterns") == ["*"]

    action_groups = _mapping(
        by_ecosystem["github-actions"].get("groups"), label="action groups"
    )
    assert _mapping(
        action_groups.get("workflow-actions"), label="workflow actions"
    ).get("patterns") == ["*"]


def test_dependency_audit_workflow_is_read_only_bounded_and_uses_trusted_checker() -> (
    None
):
    workflow = _yaml(WORKFLOW_PATH)
    assert workflow.get("permissions") == {"contents": "read"}
    triggers = _mapping(workflow.get("on"), label="on")
    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    pull_request = _mapping(triggers.get("pull_request"), label="pull_request")
    assert {"pyproject.toml", "uv.lock"} <= set(pull_request.get("paths", []))

    jobs = _mapping(workflow.get("jobs"), label="jobs")
    job = _mapping(jobs.get("audit-delta"), label="audit-delta")
    assert job.get("runs-on") == "ubuntu-24.04"
    assert job.get("timeout-minutes") == "15"
    steps = job.get("steps")
    assert isinstance(steps, list)
    named = {
        str(_mapping(step, label="step")["name"]): _mapping(step, label="step")
        for step in steps
    }
    action_steps = [step for step in named.values() if "uses" in step]
    assert action_steps
    assert all(
        isinstance(step["uses"], str) and PINNED_ACTION_RE.fullmatch(step["uses"])
        for step in action_steps
    )
    for checkout_name in {"Check out trusted base", "Check out proposed head"}:
        checkout = _mapping(named[checkout_name].get("with"), label="checkout.with")
        assert checkout.get("persist-credentials") == "false"

    command = named["Reject new dependency findings"].get("run")
    assert isinstance(command, str)
    assert "checker=audit-base/scripts/check_dependency_audit_delta.py" in command
    assert "checker=audit-head/scripts/check_dependency_audit_delta.py" in command
    assert "--base-project audit-base" in command
    assert "--head-project audit-head" in command


def test_report_parser_rejects_incomplete_data() -> None:
    with pytest.raises(audit.AuditError, match="audited_packages"):
        audit.parse_report(
            {"summary": {}, "vulnerabilities": [], "adverse_statuses": []}
        )
    with pytest.raises(audit.AuditError, match="does not match"):
        audit.parse_report(
            {
                "summary": {
                    "audited_packages": 1,
                    "vulnerabilities": 1,
                    "adverse_statuses": 0,
                },
                "vulnerabilities": [],
                "adverse_statuses": [],
            }
        )


def test_delta_treats_aliases_as_the_same_advisory_and_deduplicates_head() -> None:
    base = _report(
        vulnerabilities=(
            _vulnerability("GHSA-old", aliases=frozenset({"CVE-old", "PYSEC-old"})),
        )
    )
    head = _report(
        vulnerabilities=(
            _vulnerability("PYSEC-old", aliases=frozenset({"GHSA-old"})),
            _vulnerability("GHSA-new", aliases=frozenset({"CVE-new"})),
            _vulnerability("PYSEC-new", aliases=frozenset({"CVE-new"})),
        )
    )

    vulnerabilities, statuses = audit.new_findings(base, head)

    assert [finding.identifier for finding in vulnerabilities] == ["GHSA-new"]
    assert statuses == ()


def test_delta_keys_advisories_by_package_and_detects_adverse_statuses() -> None:
    base = _report(
        vulnerabilities=(_vulnerability("GHSA-shared", package="first"),),
        adverse_statuses=(audit.AdverseStatus("old", "archived", None),),
    )
    head = _report(
        vulnerabilities=(_vulnerability("GHSA-shared", package="second"),),
        adverse_statuses=(
            audit.AdverseStatus("old", "archived", None),
            audit.AdverseStatus("new", "deprecated", "replacement available"),
        ),
    )

    vulnerabilities, statuses = audit.new_findings(base, head)

    assert [finding.package for finding in vulnerabilities] == ["second"]
    assert statuses == (
        audit.AdverseStatus("new", "deprecated", "replacement available"),
    )


def test_audit_project_fails_closed_and_disables_repository_audit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    payload = {
        "summary": {
            "audited_packages": 1,
            "vulnerabilities": 0,
            "adverse_statuses": 0,
        },
        "vulnerabilities": [],
        "adverse_statuses": [],
    }
    calls: list[tuple[list[str], Path]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs["cwd"]))
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(audit.subprocess, "run", fake_run)

    report = audit.audit_project(tmp_path, uv_executable="checked-uv")

    assert report.audited_packages == 1
    command, cwd = calls[0]
    assert cwd == tmp_path
    assert command[0] == "checked-uv"
    assert "--no-config" in command
    assert "--frozen" in command
    assert command[command.index("--output-format") + 1] == "json"

    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="network unavailable"
        ),
    )
    with pytest.raises(audit.AuditError, match="network unavailable"):
        audit.audit_project(tmp_path)
