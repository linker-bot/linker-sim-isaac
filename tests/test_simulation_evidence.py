"""Acceptance evidence must identify the checkout and include actual, unskipped tests."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import simulation_evidence as evidence


@pytest.fixture
def checkout(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\nversion = "0.1"\n')
    git("add", "uv.lock", "pyproject.toml")
    git("-c", "user.name=Evidence test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "Test source")
    return tmp_path, git("rev-parse", "HEAD")


@pytest.fixture
def metadata(checkout, monkeypatch):
    monkeypatch.setattr(evidence, "gpu_metadata", lambda: ([
        {"name": "Test GPU", "driver_version": "580.0", "memory_mib": 24000}
    ], None))
    monkeypatch.setattr(evidence.importlib.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": name}, version="1.0.test")
        for name in evidence.RUNTIME_PACKAGES
    ])
    root, sha = checkout
    return evidence.collect_metadata(root, sha, "success", {"run_id": "123", "run_attempt": "1"})


def junit(path, contents):
    path.write_text(contents)
    return path


def passing_report(path):
    return junit(path, '<testsuites><testsuite tests="2" skipped="0" failures="0" errors="0">'
                 '<testcase classname="gpu" name="first"/>'
                 '<testcase classname="gpu" name="second"/>'
                 '</testsuite></testsuites>')


def test_matching_clean_checkout_and_passing_tests_are_accepted(checkout, metadata):
    root, sha = checkout
    meta = root / "metadata.json"
    meta.write_text(json.dumps(metadata))
    report = passing_report(root / "junit.xml")
    result = evidence.check_evidence(root, meta, report, sha)
    assert result == {"schema": 1, "accepted": True, "sha": sha, "tests_passed": 2}
    assert metadata["source"]["file_sha256"]["uv.lock"] == evidence._digest(root / "uv.lock")


@pytest.mark.parametrize(
    "contents, message",
    [
        ('<testsuites><testsuite tests="0"/></testsuites>', "ran no tests"),
        ('<testsuite><testcase name="skip"><skipped/></testcase></testsuite>', "skipped"),
        ('<testsuite><testcase name="fail"><failure/></testcase></testsuite>', "failure"),
        ('<testsuite><testcase name="error"><error/></testcase></testsuite>', "error"),
        ('<testsuite tests="0"><testcase name="unexpected"/></testsuite>', "does not match"),
        ('<testsuite tests="3"><testcase name="one"/></testsuite>', "does not match"),
        ('<testsuite tests="bad"><testcase name="one"/></testsuite>', "count is invalid"),
        ('<testsuite skipped="1"><testcase name="one"/></testsuite>', "summary reported"),
        ('<testsuite errors="1"><testcase name="one"/></testsuite>', "summary reported"),
        ('<testsuite failures="1"><testcase name="one"/></testsuite>', "summary reported"),
        ('<testsuite tests="-1"><testcase name="one"/></testsuite>', "count is invalid"),
        ('<testsuite><testcase/></testsuite>', "unnamed"),
        ('<report><testcase name="one"/></report>', "root"),
        ('<testsuite>', "cannot read"),
    ],
)
def test_incomplete_or_nonpassing_junit_is_rejected(tmp_path, contents, message):
    with pytest.raises(evidence.EvidenceError, match=message):
        evidence.check_junit(junit(tmp_path / "junit.xml", contents))


def test_missing_junit_is_rejected(tmp_path):
    with pytest.raises(evidence.EvidenceError, match="cannot read"):
        evidence.check_junit(tmp_path / "missing.xml")


def test_nested_suite_counts_are_not_double_counted(tmp_path):
    report = junit(tmp_path / "junit.xml",
                   '<testsuites tests="2"><testsuite tests="2">'
                   '<testsuite tests="1"><testcase name="one"/></testsuite>'
                   '<testsuite tests="1"><testcase name="two"/></testsuite>'
                   '</testsuite></testsuites>')
    assert evidence.check_junit(report) == 2


@pytest.mark.parametrize("change", ["metadata_sha", "expected_sha", "dirty", "lock_changed"])
def test_other_commits_or_source_changes_are_rejected(checkout, metadata, change):
    root, sha = checkout
    if change == "metadata_sha":
        metadata["source"]["sha"] = "0" * 40
    elif change == "expected_sha":
        sha = "0" * 40
    elif change == "dirty":
        metadata["source"]["tracked_dirty"] = True
    else:
        (root / "uv.lock").write_text("version = 2\n")
    meta = root / "metadata.json"
    meta.write_text(json.dumps(metadata))
    with pytest.raises(evidence.EvidenceError, match="commit|checkout"):
        evidence.check_evidence(root, meta, passing_report(root / "junit.xml"), sha)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("dependency_sync_outcome", "failure", "synced successfully"),
        ("dependency_sync_outcome", "skipped", "synced successfully"),
        ("gpus", [], "GPU information"),
        ("gpus", [{"name": "incomplete"}], "GPU information"),
        ("gpu_error", "query failed", "GPU information"),
        ("packages", [], "missing packages"),
        ("packages", [{"name": "torch"}], "invalid package versions"),
        ("python", {}, "incomplete python"),
        ("os", {}, "incomplete os"),
    ],
)
def test_incomplete_runtime_is_rejected(checkout, metadata, field, value, message):
    root, sha = checkout
    metadata[field] = value
    meta = root / "metadata.json"
    meta.write_text(json.dumps(metadata))
    with pytest.raises(evidence.EvidenceError, match=message):
        evidence.check_evidence(root, meta, passing_report(root / "junit.xml"), sha)


def test_metadata_does_not_dump_environment_or_extra_os_fields(checkout, metadata, monkeypatch):
    root, sha = checkout
    secret = "sentinel-secret-not-for-artifacts"
    monkeypatch.setenv("EVIDENCE_TEST_SECRET", secret)
    monkeypatch.setattr(evidence.platform, "freedesktop_os_release",
                        lambda: {"ID": "test", "VERSION_ID": "1", "PRIVATE": secret})
    result = evidence.collect_metadata(root, sha, "success", {})
    assert secret not in json.dumps(result)
    assert result["os"]["distribution"] == {"ID": "test", "VERSION_ID": "1"}
    assert all(set(item) == {"name", "version"} for item in result["packages"])


def test_gpu_query_records_model_and_driver_without_uuid(monkeypatch):
    def query(args, **kwargs):
        assert args[1] == "--query-gpu=name,driver_version,memory.total"
        return subprocess.CompletedProcess(args, 0, '"NVIDIA GPU, test", 580.0, 24000\n')

    monkeypatch.setattr(evidence.subprocess, "run", query)
    devices, error = evidence.gpu_metadata()
    assert error is None
    assert devices == [{"name": "NVIDIA GPU, test", "driver_version": "580.0", "memory_mib": 24000}]


def test_failed_gpu_query_has_no_raw_command_output(monkeypatch):
    def query(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="private diagnostic")

    monkeypatch.setattr(evidence.subprocess, "run", query)
    devices, error = evidence.gpu_metadata()
    assert devices == [] and "query failed" in error
    assert "private diagnostic" not in error


def test_cli_failure_preserves_rejection_summary(checkout, metadata, capsys):
    root, sha = checkout
    meta = root / "metadata.json"
    meta.write_text(json.dumps(metadata))
    output = root / "acceptance.json"
    status = evidence.main([
        "check", "--root", str(root), "--expected-sha", sha, "--metadata", str(meta),
        "--junit", str(root / "missing.xml"), "--output", str(output),
    ])
    assert status == 1
    assert json.loads(output.read_text())["accepted"] is False
    assert "cannot read the GPU JUnit report" in capsys.readouterr().err


def test_workflow_keeps_trusted_manual_gate_and_always_preserves_evidence():
    path = Path(__file__).resolve().parents[1] / ".github/workflows/simulation.yml"
    workflow = yaml.safe_load(path.read_text())
    assert set(workflow.get("on", workflow.get(True))) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["gpu-simulation"]
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "nvidia-gpu", "isaac-sim"]
    assert job["environment"] == "simulation"
    assert job["env"]["OMNI_KIT_ACCEPT_EULA"] == "${{ vars.OMNI_KIT_ACCEPT_EULA }}"
    steps = job["steps"]
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload["uses"] == "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9"
    assert upload["if"] == "${{ always() }}"
    assert upload["with"]["if-no-files-found"] == "error"
    check = next(step for step in steps if "simulation_evidence.py check" in step.get("run", ""))
    assert check["if"] == "${{ always() }}"
    assert '--expected-sha "$GITHUB_SHA"' in check["run"]
    run = next(step["run"] for step in steps if "python -m pytest tests/gpu" in step.get("run", ""))
    assert 'test -n "${DISPLAY:-}"' in run
    assert "--junitxml=artifacts/simulation/junit.xml" in run
    assert "--basetemp=artifacts/simulation/pytest-tmp" in run
    assert "set -euo pipefail" in run
