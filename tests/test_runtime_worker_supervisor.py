from __future__ import annotations

from pathlib import Path
import signal
from types import SimpleNamespace

from scripts import runtime_worker_supervisor as supervisor


def test_supervisor_requires_markers_and_clean_worker_exit(monkeypatch, capsys) -> None:
    calls: list[tuple[list[str], dict[str, str], float]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"], kwargs["timeout"]))
        return SimpleNamespace(
            returncode=0,
            stdout='WORKER_READY {"ok": true}\nworker shutdown\n',
        )

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    result = supervisor.run_supervised_worker(
        script_path=Path("worker.py"),
        argv=("--physics-backend", "newton"),
        required_markers=("WORKER_READY",),
        success_marker="WORKER_OK",
    )

    assert result == 0
    assert calls[0][0][1:] == [
        str(Path("worker.py").resolve()),
        "--physics-backend",
        "newton",
    ]
    assert calls[0][1][supervisor.WORKER_ENV_VAR] == "1"
    assert calls[0][2] == supervisor.DEFAULT_WORKER_TIMEOUT_S
    output = capsys.readouterr().out
    assert "WORKER_READY" in output
    assert "WORKER_OK" in output
    assert '"worker_exit_code": 0' in output


def test_supervisor_converts_false_green_worker_exit_to_failure(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="worker failed before its marker\n",
        ),
    )

    result = supervisor.run_supervised_worker(
        script_path=Path("worker.py"),
        argv=(),
        required_markers=("WORKER_READY",),
        success_marker="WORKER_OK",
    )

    assert result == 1
    captured = capsys.readouterr()
    assert "WORKER_OK" not in captured.out
    assert "LINKERBOT_ISAAC_RUNTIME_SUPERVISOR_FAILED" in captured.err
    assert '"missing_markers": ["WORKER_READY"]' in captured.err


def test_supervisor_times_out_worker_without_false_success(monkeypatch, capsys) -> None:
    def run(command, **kwargs):
        raise supervisor.subprocess.TimeoutExpired(
            cmd=command,
            timeout=kwargs["timeout"],
            output=b"worker reached native startup\n",
        )

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    result = supervisor.run_supervised_worker(
        script_path=Path("worker.py"),
        argv=(),
        required_markers=("WORKER_READY",),
        success_marker="WORKER_OK",
        timeout_s=0.25,
    )

    assert result == 124
    captured = capsys.readouterr()
    assert "worker reached native startup" in captured.out
    assert "WORKER_OK" not in captured.out
    assert '"timed_out": true' in captured.err
    assert '"timeout_s": 0.25' in captured.err
    assert '"worker_exit_code": null' in captured.err


def test_supervisor_rejects_invalid_timeout() -> None:
    for value in (0.0, -1.0, float("inf"), True):
        try:
            supervisor.run_supervised_worker(
                script_path=Path("worker.py"),
                argv=(),
                required_markers=("WORKER_READY",),
                success_marker="WORKER_OK",
                timeout_s=value,
            )
        except ValueError as exc:
            assert "timeout_s" in str(exc)
        else:  # pragma: no cover - assertion helper branch
            raise AssertionError(f"timeout {value!r} was accepted")


def test_supervisor_repeats_fresh_workers_and_reports_count(
    monkeypatch, capsys
) -> None:
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='WORKER_READY {"ok": true}\n')

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    result = supervisor.run_supervised_worker(
        script_path=Path("worker.py"),
        argv=(),
        required_markers=("WORKER_READY",),
        success_marker="WORKER_OK",
        repetitions=3,
    )

    assert result == 0
    assert len(calls) == 3
    output = capsys.readouterr().out
    assert sum(line.startswith("WORKER_READY ") for line in output.splitlines()) == 3
    assert '"completed_repetitions": 3' in output


def test_supervisor_names_native_signal_and_stops_repeating(
    monkeypatch, capsys
) -> None:
    calls = 0

    def run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=-signal.SIGSEGV,
            stdout='WORKER_READY {"outputs_drained": true}\n',
        )

    monkeypatch.setattr(supervisor.subprocess, "run", run)

    result = supervisor.run_supervised_worker(
        script_path=Path("worker.py"),
        argv=(),
        required_markers=("WORKER_READY",),
        success_marker="WORKER_OK",
        repetitions=5,
    )

    assert result == 128 + signal.SIGSEGV
    assert calls == 1
    captured = capsys.readouterr()
    assert "WORKER_OK" not in captured.out
    assert '"repetition": 1' in captured.err
    assert '"repetitions": 5' in captured.err
    assert '"worker_signal": "SIGSEGV"' in captured.err


def test_supervisor_rejects_invalid_repetitions() -> None:
    for value in (0, -1, 1.5, True):
        try:
            supervisor.run_supervised_worker(
                script_path=Path("worker.py"),
                argv=(),
                required_markers=("WORKER_READY",),
                success_marker="WORKER_OK",
                repetitions=value,
            )
        except ValueError as exc:
            assert "repetitions" in str(exc)
        else:  # pragma: no cover - assertion helper branch
            raise AssertionError(f"repetitions {value!r} was accepted")
