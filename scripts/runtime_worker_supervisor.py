"""Run an Isaac worker in a child process and make its exit status auditable."""

from __future__ import annotations

from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
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
) -> int:
    """限时捕获子进程输出，同时要求运行证据齐全且退出码为零。"""

    markers = tuple(str(marker) for marker in required_markers)
    if not markers or any(not marker for marker in markers):
        raise ValueError("required_markers must contain non-empty marker names")
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be a positive finite number")
    environment = os.environ.copy()
    environment[WORKER_ENV_VAR] = "1"
    command = [sys.executable, str(Path(script_path).resolve()), *map(str, argv)]
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
                    "timed_out": True,
                    "timeout_s": float(timeout_s),
                    "worker_exit_code": None,
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
        print(
            success_marker
            + " "
            + json.dumps(
                {
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

    print(
        "LINKERBOT_ISAAC_RUNTIME_SUPERVISOR_FAILED "
        + json.dumps(
            {
                "event": "supervised_isaac_runtime_failed",
                "missing_markers": missing,
                "worker_exit_code": int(completed.returncode),
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


__all__ = [
    "DEFAULT_WORKER_TIMEOUT_S",
    "WORKER_ENV_VAR",
    "in_runtime_worker",
    "run_supervised_worker",
]
