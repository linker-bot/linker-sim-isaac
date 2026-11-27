"""Installation diagnostics must survive broken or missing runtime dependencies."""

import pytest

import linker_lab
from linker_lab import cli, engine


@pytest.fixture
def healthy_doctor(monkeypatch):
    calls = []

    def record(name, result=None, error=None):
        def run():
            calls.append(name)
            if error:
                raise error
            return result
        return run

    monkeypatch.setattr(engine, "check_versions", record("versions", [engine.Check("newton", True, "1.8.0")]))
    monkeypatch.setattr(engine, "check_gpu", record("gpu", engine.Check("gpu", True, "test GPU")))
    monkeypatch.setattr(linker_lab, "bootstrap", record("bootstrap"))
    monkeypatch.setattr(engine, "check_patches", record("patches", engine.Check("engine patches", True, "applied")))
    monkeypatch.setattr(engine, "check_simulation", record("simulation", engine.Check("simulation", True, "passed")))
    return calls, record


def test_doctor_checks_versions_before_importing_runtime(healthy_doctor, capsys):
    calls, _ = healthy_doctor
    assert cli.main(["doctor"]) == 0
    assert calls == ["versions", "gpu", "bootstrap", "patches", "simulation"]
    assert "All checks passed." in capsys.readouterr().out


@pytest.mark.parametrize("detail", ["not installed (need ==1.8.0)", "1.6.1 (need ==1.8.0)"])
def test_doctor_reports_bad_versions_without_bootstrap(healthy_doctor, monkeypatch, capsys, detail):
    calls, record = healthy_doctor
    monkeypatch.setattr(engine, "check_versions", record("versions", [engine.Check("newton", False, detail)]))
    assert cli.doctor() == 1
    assert "bootstrap" not in calls and "simulation" not in calls
    assert f"[FAIL] newton: {detail}" in capsys.readouterr().out


@pytest.mark.parametrize("name,error,label", [
    ("bootstrap", ImportError("cannot import ClampingBase"), "engine patches"),
    ("gpu", OSError("libcuda.so unavailable"), "gpu"),
    ("versions", engine.metadata.PackageNotFoundError("linker-lab"), "package versions"),
    ("simulation", RuntimeError("could not create simulation"), "simulation"),
])
def test_doctor_turns_runtime_errors_into_failures(healthy_doctor, monkeypatch, capsys, name, error, label):
    calls, record = healthy_doctor
    owner = linker_lab if name == "bootstrap" else engine
    attr = "bootstrap" if name == "bootstrap" else f"check_{name}"
    monkeypatch.setattr(owner, attr, record(name, error=error))
    assert cli.doctor() == 1
    output = capsys.readouterr().out
    assert f"[FAIL] {label}: {type(error).__name__}:" in output
    assert str(error) in output and "Some checks failed" in output
    if name != "simulation":
        assert "simulation" not in calls
