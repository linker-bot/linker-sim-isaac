from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import yaml


WORKFLOW_PATH = Path(".github/workflows/release.yml")
PINNED_ACTION_RE = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{label} must be a mapping"
    return value


def _workflow() -> Mapping[str, object]:
    value = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return _mapping(value, label="release workflow")


def _publish_job() -> Mapping[str, object]:
    jobs = _mapping(_workflow().get("jobs"), label="jobs")
    return _mapping(jobs.get("publish"), label="jobs.publish")


def _steps_by_name() -> dict[str, Mapping[str, object]]:
    steps = _publish_job().get("steps")
    assert isinstance(steps, list)
    result: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(steps):
        step = _mapping(value, label=f"steps[{index}]")
        name = step.get("name")
        assert isinstance(name, str) and name
        assert name not in result
        result[name] = step
    return result


def test_release_workflow_is_manual_and_has_bounded_permissions() -> None:
    workflow = _workflow()
    triggers = _mapping(workflow.get("on"), label="on")
    assert set(triggers) == {"workflow_dispatch"}
    dispatch = _mapping(triggers["workflow_dispatch"], label="workflow_dispatch")
    inputs = _mapping(dispatch.get("inputs"), label="inputs")
    assert set(inputs) == {"tag", "simulation_run_id", "prerelease"}
    assert all(
        _mapping(value, label=name).get("required") == "true"
        for name, value in inputs.items()
    )
    assert workflow.get("permissions") == {"actions": "read", "contents": "write"}

    job = _publish_job()
    assert job.get("environment") == "release"
    assert job.get("runs-on") == "ubuntu-24.04"
    assert job.get("timeout-minutes") == "30"


def test_release_workflow_pins_actions_and_checks_out_the_exact_tag() -> None:
    steps = _steps_by_name()
    action_steps = [step for step in steps.values() if "uses" in step]
    assert action_steps
    assert all(
        isinstance(step["uses"], str) and PINNED_ACTION_RE.fullmatch(step["uses"])
        for step in action_steps
    )
    checkout = _mapping(
        steps["Check out annotated tag"].get("with"), label="checkout.with"
    )
    assert checkout.get("ref") == "refs/tags/${{ inputs.tag }}"
    assert checkout.get("fetch-depth") == "0"
    assert checkout.get("persist-credentials") == "false"


def test_release_requires_matching_gpu_evidence_and_existing_tag() -> None:
    steps = _steps_by_name()
    evidence = steps["Read Simulation acceptance evidence"].get("run")
    validate = steps["Validate release and create source archive"].get("run")
    publish = steps["Publish GitHub release"].get("run")

    assert isinstance(evidence, str)
    assert "gh run view" in evidence
    assert "conclusion,databaseId,headSha,url,workflowName" in evidence
    assert isinstance(validate, str)
    assert "scripts/prepare_release.py" in validate
    assert "--simulation-run-id" in validate
    assert isinstance(publish, str)
    assert "gh release create" in publish
    assert "--verify-tag" in publish
    assert "SHA256SUMS" in publish


def test_release_does_not_build_or_upload_a_python_distribution() -> None:
    content = WORKFLOW_PATH.read_text(encoding="utf-8").lower()
    forbidden = ("pypi", "twine", "python -m build", "uv build", "bdist", "sdist")
    assert not any(value in content for value in forbidden)
