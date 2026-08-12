#!/usr/bin/env python3
"""用正式 Kaleidoscope PhysX composition 验收配置中的 GPU 显存预算。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.configuration import load_kaleidoscope_config  # noqa: E402
from linkerbot_sim.isaac.gpu_memory_audit import (  # noqa: E402
    CudaNvmlMemoryProbe,
    GpuMemoryAuditor,
    GpuMemoryLimits,
    GpuMemoryProbe,
)
from linkerbot_sim.kaleidoscope import make_torch_env  # noqa: E402


SUCCESS_MARKER = "LINKERBOT_PHYSX_GPU_MEMORY_BUDGET_OK"
DEFAULT_PROFILE = "physx_cuda"
DEFAULT_NUM_ENVS = 2
DEFAULT_WARMUP_STEPS = 8
DEFAULT_STEADY_STEPS = 16


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--num-envs", type=_positive_int, default=DEFAULT_NUM_ENVS)
    parser.add_argument(
        "--warmup-steps",
        type=_positive_int,
        default=DEFAULT_WARMUP_STEPS,
    )
    parser.add_argument(
        "--steady-steps",
        type=_positive_int,
        default=DEFAULT_STEADY_STEPS,
    )
    return parser.parse_args(argv)


def _resolve_physx_config(profile: str) -> tuple[object, GpuMemoryLimits]:
    config = load_kaleidoscope_config(profile)
    physics = getattr(config, "physics", None)
    selection = (
        str(getattr(physics, "engine", "")),
        str(getattr(physics, "execution", "")),
    )
    if selection != ("physx", "cuda"):
        raise RuntimeError(
            "GPU memory budget smoke requires a Kaleidoscope PhysX CUDA profile; "
            f"profile={profile!r} resolved selection={selection!r}"
        )
    return config, GpuMemoryLimits.from_budget(getattr(physics, "memory", None))


def _require_selected_cuda_device(env: object, *, cuda_device: int) -> None:
    actual = str(getattr(env, "device", ""))
    expected = f"cuda:{cuda_device}"
    if actual != expected:
        raise RuntimeError(
            "PhysX environment uses the wrong CUDA device: "
            f"expected={expected!r}, actual={actual!r}"
        )


def _run_zero_action_steps(env: object, *, steps: int) -> None:
    """用训练同款 SAME_STEP 协议跨过 episode 边界并保持固定形状。"""

    import torch

    actions = torch.zeros(
        (int(env.num_envs), int(env.action_dim)),
        device=env.device,
        dtype=torch.float32,
    )
    for _ in range(steps):
        # 不能在这里对 done mask 调用 nonzero/reset_idx：变长索引会同步主机，且不能
        # 代表 Kaleidoscope 的正式训练路径。SAME_STEP 使用持久化 N 行 CUDA mask，
        # 在保留 terminal transition 后完成选择性重置。
        token = env.begin_same_step()
        result = env.step_same_step(token, actions)
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("Kaleidoscope step must return the five-field contract")
        env.complete_same_step(token)


def _synchronize_cuda(*, cuda_device: int) -> None:
    """在冷边界等待前序 allocator/physics 操作，避免读取未完成的异步状态。"""

    import torch

    torch.cuda.synchronize(f"cuda:{cuda_device}")


def run_smoke(
    *,
    profile: str,
    num_envs: int,
    warmup_steps: int,
    steady_steps: int,
    probe: GpuMemoryProbe | None = None,
    pid: int | None = None,
) -> dict[str, object]:
    """执行 prelaunch → warmup → steady 两端的完整显存门禁。"""

    config, limits = _resolve_physx_config(profile)
    cuda_device = int(getattr(config, "cuda_device"))
    selected_probe = CudaNvmlMemoryProbe() if probe is None else probe
    auditor = GpuMemoryAuditor(
        cuda_device=cuda_device,
        limits=limits,
        probe=selected_probe,
        pid=os.getpid() if pid is None else pid,
    )

    # 此采样必须发生在 make_torch_env/SimulationApp 之前；失败时不会启动 Kit。
    auditor.capture_prelaunch()
    env = None
    failed = True
    try:
        env = make_torch_env(config=config, num_envs=num_envs)
        _require_selected_cuda_device(env, cuda_device=cuda_device)
        env.reset(seed=123)

        _run_zero_action_steps(env, steps=warmup_steps)
        _synchronize_cuda(cuda_device=cuda_device)
        auditor.capture_post_warmup()
        auditor.capture_steady_baseline()

        _run_zero_action_steps(env, steps=steady_steps)
        _synchronize_cuda(cuda_device=cuda_device)
        auditor.capture_steady_final()

        result = {
            "profile": profile,
            "physics_engine": "physx",
            "physics_execution": "cuda",
            "cuda_device": cuda_device,
            "pid": auditor.pid,
            "num_envs": int(env.num_envs),
            "warmup_steps": warmup_steps,
            "steady_steps": steady_steps,
            "audits": [report.as_dict() for report in auditor.reports],
        }
        # fast_shutdown 可能在 native close 内结束解释器，因此 marker 必须先 flush。
        print(
            SUCCESS_MARKER
            + " "
            + json.dumps(result, ensure_ascii=True, sort_keys=True),
            flush=True,
        )
        failed = False
        return result
    except BaseException as exc:
        # SimulationApp 的 fast shutdown 会直接结束解释器；必须在 close 之前输出
        # 原始异常，否则真实 GPU 门禁失败只剩退出码，无法区分预算越界和运行故障。
        print(
            "LINKERBOT_PHYSX_GPU_MEMORY_BUDGET_FAILED "
            + json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "profile": profile,
                    "num_envs": num_envs,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        if env is not None:
            env.close(exit_code=1 if failed else 0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_smoke(
        profile=str(args.profile),
        num_envs=int(args.num_envs),
        warmup_steps=int(args.warmup_steps),
        steady_steps=int(args.steady_steps),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
