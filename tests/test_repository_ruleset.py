from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path
import re

import pytest
import yaml

from scripts import check_repository_ruleset as ruleset


POLICY_PATH = Path(".github/rulesets/master.json")
WORKFLOW_PATH = Path(".github/workflows/repository-policy.yml")
PINNED_ACTION_RE = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{label} must be a mapping"
    return value


def _workflow() -> Mapping[str, object]:
    value = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return _mapping(value, label="repository policy workflow")


def _active_ruleset(policy: Mapping[str, object]) -> dict[str, object]:
    actual = copy.deepcopy(dict(policy))
    actual.update({"id": 42, "name": "Protect default branch"})
    return actual


def test_maintained_policy_has_no_bypass_and_enforces_review_and_quality() -> None:
    policy = ruleset.load_policy(POLICY_PATH)

    ruleset.validate_policy(policy)


def test_policy_validator_rejects_a_weaker_quality_contract() -> None:
    policy = copy.deepcopy(dict(ruleset.load_policy(POLICY_PATH)))
    rules = policy["rules"]
    assert isinstance(rules, list)
    status = next(value for value in rules if value["type"] == "required_status_checks")
    status["parameters"]["required_status_checks"] = []

    with pytest.raises(ruleset.RulesetError, match="status check"):
        ruleset.validate_policy(policy)


def test_repository_audit_reads_active_ruleset_details_and_accepts_stricter_review() -> (
    None
):
    policy = ruleset.load_policy(POLICY_PATH)
    actual = _active_ruleset(policy)
    actual_rules = actual["rules"]
    assert isinstance(actual_rules, list)
    pull_request = next(
        value for value in actual_rules if value["type"] == "pull_request"
    )
    pull_request["parameters"]["required_approving_review_count"] = 2
    requested: list[str] = []

    def fetch(url: str) -> object:
        requested.append(url)
        if url.endswith("rulesets?includes_parents=true&per_page=100"):
            return [{"id": 42, "target": "branch", "enforcement": "active"}]
        if url.endswith("rulesets/42?includes_parents=true"):
            return actual
        raise AssertionError(f"unexpected URL: {url}")

    result = ruleset.audit_repository(
        "linker-bot/linker-sim-isaac", policy, fetch_json=fetch
    )

    assert result["id"] == 42
    assert len(requested) == 2
    assert all(url.startswith("https://api.github.com/repos/") for url in requested)


def test_repository_audit_rejects_missing_or_inactive_rulesets() -> None:
    policy = ruleset.load_policy(POLICY_PATH)

    with pytest.raises(ruleset.RulesetError, match="no active ruleset"):
        ruleset.audit_repository(
            "linker-bot/linker-sim-isaac",
            policy,
            fetch_json=lambda _url: [],
        )


def test_repository_policy_workflow_is_read_only_bounded_and_not_a_pr_gate() -> None:
    workflow = _workflow()
    assert workflow.get("permissions") == {"contents": "read"}
    triggers = _mapping(workflow.get("on"), label="on")
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert "pull_request" not in triggers
    assert "push" not in triggers

    jobs = _mapping(workflow.get("jobs"), label="jobs")
    job = _mapping(jobs.get("audit-ruleset"), label="jobs.audit-ruleset")
    assert job.get("runs-on") == "ubuntu-24.04"
    assert job.get("timeout-minutes") == "5"
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
    checkout = _mapping(named["Check out repository"].get("with"), label="checkout")
    assert checkout.get("persist-credentials") == "false"
    audit = named["Audit active default branch ruleset"]
    assert "check_repository_ruleset.py" in str(audit.get("run"))
    assert _mapping(audit.get("env"), label="audit.env").get("GITHUB_TOKEN") == (
        "${{ github.token }}"
    )
