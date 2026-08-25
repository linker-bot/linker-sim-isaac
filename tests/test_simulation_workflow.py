from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import tomllib

import yaml


WORKFLOW_PATH = Path(".github/workflows/simulation.yml")
PINNED_ACTION_RE = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    assert isinstance(value, Mapping), f"{label} must be a mapping"
    return value


def _workflow() -> Mapping[str, object]:
    # BaseLoader keeps GitHub's `on` key as text instead of applying YAML 1.1's
    # historical boolean conversion.
    value = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return _mapping(value, label="simulation workflow")


def _simulation_job() -> Mapping[str, object]:
    jobs = _mapping(_workflow().get("jobs"), label="jobs")
    return _mapping(jobs.get("gpu-simulation"), label="jobs.gpu-simulation")


def _steps_by_name() -> dict[str, Mapping[str, object]]:
    steps = _simulation_job().get("steps")
    assert isinstance(steps, list)
    result: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(steps):
        step = _mapping(value, label=f"steps[{index}]")
        name = step.get("name")
        assert isinstance(name, str) and name
        assert name not in result
        result[name] = step
    return result


def test_simulation_workflow_runs_only_trusted_pushes_or_manual_dispatches() -> None:
    workflow = _workflow()
    triggers = _mapping(workflow.get("on"), label="on")

    assert set(triggers) == {"push", "workflow_dispatch"}
    push = _mapping(triggers.get("push"), label="on.push")
    assert push.get("branches") == ["master"]
    paths = push.get("paths")
    assert isinstance(paths, list)
    assert {
        ".github/workflows/simulation.yml",
        "apps/**",
        "configs/**",
        "src/**",
        "tests/**",
        "uv.lock",
    } <= set(paths)


def test_simulation_job_has_a_read_only_bounded_runner_contract() -> None:
    workflow = _workflow()
    assert workflow.get("permissions") == {"contents": "read"}
    concurrency = _mapping(workflow.get("concurrency"), label="concurrency")
    assert concurrency.get("cancel-in-progress") == "false"

    job = _simulation_job()
    assert job.get("runs-on") == [
        "self-hosted",
        "linux",
        "x64",
        "nvidia-gpu",
        "isaac-sim",
    ]
    assert job.get("environment") == "simulation"
    assert job.get("timeout-minutes") == "240"
    environment = _mapping(job.get("env"), label="jobs.gpu-simulation.env")
    assert environment.get("OMNI_KIT_ACCEPT_EULA") == (
        "${{ vars.OMNI_KIT_ACCEPT_EULA }}"
    )


def test_simulation_workflow_pins_actions_and_does_not_persist_credentials() -> None:
    steps = _steps_by_name()
    action_steps = [step for step in steps.values() if "uses" in step]

    assert action_steps
    assert all(
        isinstance(step["uses"], str) and PINNED_ACTION_RE.fullmatch(step["uses"])
        for step in action_steps
    )
    checkout = steps["Check out repository"]
    checkout_with = _mapping(checkout.get("with"), label="checkout.with")
    assert checkout_with.get("persist-credentials") == "false"


def test_simulation_workflow_uses_the_maintained_entrypoint_and_explicit_extras() -> (
    None
):
    steps = _steps_by_name()
    sync = steps["Sync simulation test environment"].get("run")
    execute = steps["Run GPU and Isaac acceptance matrix"].get("run")
    preflight = steps["Verify trusted NVIDIA runner"].get("run")

    assert isinstance(sync, str)
    assert "--extra simulation" in sync
    assert "--extra visualization" in sync
    assert "--extra training" in sync
    assert "--extra test" in sync
    assert "--extra dev" not in sync
    assert execute == "bash ci/simulation.sh"
    assert isinstance(preflight, str)
    assert "nvidia-smi" in preflight
    assert "OMNI_KIT_ACCEPT_EULA" in preflight


def test_test_extra_is_compatible_with_simulation_and_version_aligned() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert optional["test"] == ["coverage==7.4.4", "pytest==9.1.1"]
    assert set(optional["test"]) <= set(optional["dev"])
