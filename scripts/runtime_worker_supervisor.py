"""Run an Isaac worker in a child process and make its exit status auditable."""

from __future__ import annotations

from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


WORKER_ENV_VAR = "LINKERBOT_ISAAC_RUNTIME_WORKER"
DEFAULT_WORKER_TIMEOUT_S = 120.0


def in_runtime_worker() -> bool:
    """Return whether the current process is the supervised Isaac child."""

    return os.environ.get(WORKER_ENV_VAR) == "1"


def run_supervised_worker(
    *,
    script_path: Path,
    argv: Sequence[str],
    required_markers: Sequence[str],
    success_marker: str,
    timeout_s: float = DEFAULT_WORKER_TIMEOUT_S,
    repetitions: int = 1,
) -> int:
    """重复运行受监督子进程，要求每次运行证据齐全且退出码为零。"""

    markers = tuple(str(marker) for marker in required_markers)
    if not markers or any(not marker for marker in markers):
        raise ValueError("required_markers must contain non-empty marker names")
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be a positive finite number")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    environment = os.environ.copy()
    environment[WORKER_ENV_VAR] = "1"
    command = [sys.executable, str(Path(script_path).resolve()), *map(str, argv)]
    for repetition in range(1, repetitions + 1):
        try:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                check=False,
                timeout=float(timeout_s),
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run 已终止并回收超时 worker。TimeoutExpired 在不同 Python 版本中
            # 可能携带 str 或 bytes，因此先规范化再回放最后一段诊断输出。
            partial = exc.stdout or ""
            output = (
                partial.decode("utf-8", errors="replace")
                if isinstance(partial, bytes)
                else str(partial)
            )
            sys.stdout.write(output)
            sys.stdout.flush()
            print(
                "LINKERBOT_ISAAC_RUNTIME_SUPERVISOR_FAILED "
                + json.dumps(
                    {
                        "event": "supervised_isaac_runtime_failed",
                        "missing_markers": list(markers),
                        "repetition": repetition,
                        "repetitions": repetitions,
                        "timed_out": True,
                        "timeout_s": float(timeout_s),
                        "worker_exit_code": None,
                        "worker_signal": None,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            return 124
        output = completed.stdout or ""
        sys.stdout.write(output)
        sys.stdout.flush()
        observed = {
            marker
            for line in output.splitlines()
            for marker in markers
            if line.startswith(marker + " ")
        }
        missing = [marker for marker in markers if marker not in observed]
        if completed.returncode == 0 and not missing:
            continue

        worker_signal = _signal_name(completed.returncode)
        print(
            "LINKERBOT_ISAAC_RUNTIME_SUPERVISOR_FAILED "
            + json.dumps(
                {
                    "event": "supervised_isaac_runtime_failed",
                    "missing_markers": missing,
                    "repetition": repetition,
                    "repetitions": repetitions,
                    "worker_exit_code": int(completed.returncode),
                    "worker_signal": worker_signal,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        if completed.returncode > 0:
            return int(completed.returncode)
        if completed.returncode < 0:
            return 128 + abs(int(completed.returncode))
        return 1

    print(
        success_marker
        + " "
        + json.dumps(
            {
                "completed_repetitions": repetitions,
                "event": "supervised_isaac_runtime_complete",
                "validated_markers": list(markers),
                "worker_exit_code": 0,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _signal_name(returncode: int) -> str | None:
    """Decode subprocess' negative POSIX return code without guessing on Windows."""

    if int(returncode) >= 0:
        return None
    try:
        return signal.Signals(abs(int(returncode))).name
    except ValueError:
        return f"SIGNAL_{abs(int(returncode))}"


__all__ = [
    "DEFAULT_WORKER_TIMEOUT_S",
    "WORKER_ENV_VAR",
    "in_runtime_worker",
    "run_supervised_worker",
]
