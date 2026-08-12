#!/usr/bin/env python3
"""按正式 session profile 启动七个 Kit，并验证依赖来源与 physics owner。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.isaac.provenance import (  # noqa: E402
    collect_runtime_provenance,
    format_runtime_provenance,
    validate_target_runtime,
)
from linkerbot_sim.isaac.session import create_isaac_session_from_spec  # noqa: E402
from linkerbot_sim.isaac.spec import (  # noqa: E402
    IsaacComputeSpec,
    IsaacNewtonCpuSpec,
    IsaacNewtonCudaSpec,
    IsaacPhysxCpuSpec,
    IsaacPhysxCudaSpec,
    IsaacRenderSpec,
    IsaacSessionSpec,
)
from scripts.runtime_worker_supervisor import (  # noqa: E402
    in_runtime_worker,
    run_supervised_worker,
)


PHYSICS_RUNTIME_MARKER = "LINKERBOT_PHYSICS_SCENE_RUNTIME_VALID"
DEPENDENCY_RUNTIME_MARKER = "LINKERBOT_DEPENDENCY_RUNTIME_VALID"
SUCCESS_MARKER = "LINKERBOT_RUNTIME_VALID"
_PROFILES = (
    "mirror-physx-cpu",
    "mirror-newton-cpu",
    "mirror-newton-cpu-render",
    "mirror-newton-cuda",
    "mirror-newton-cuda-render",
    "kaleidoscope-physx-cuda",
    "kaleidoscope-newton-cuda",
    "kaleidoscope-physx-cuda-viewport",
    "kaleidoscope-newton-cuda-viewport",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析一个严格 session 检查 profile；CPU/CUDA Newton 共用同一 Kit。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=_PROFILES, default="mirror-physx-cpu")
    parser.add_argument("--cuda-device", type=int, default=0)
    result = parser.parse_args(argv)
    if result.cuda_device < 0:
        parser.error("--cuda-device must be non-negative")
    return result


def _session_spec(profile: str, *, cuda_device: int) -> IsaacSessionSpec:
    """把 CLI selector 映射到不可变 session 规格，不提供 backend fallback。"""

    common = {
        "compute": IsaacComputeSpec(cuda_device=cuda_device),
        "physics_dt": 1.0 / 120.0,
        "rendering_dt": 1.0 / 60.0,
        "gravity_z": -9.81,
        "add_ground": False,
    }
    if profile == "mirror-physx-cpu":
        return IsaacSessionSpec(
            experience_family="mirror",
            physics=IsaacPhysxCpuSpec(),
            **common,
        )
    if profile in {"mirror-newton-cpu", "mirror-newton-cpu-render"}:
        return IsaacSessionSpec(
            experience_family="mirror",
            physics=IsaacNewtonCpuSpec(),
            render=IsaacRenderSpec(enabled=profile == "mirror-newton-cpu-render"),
            **common,
        )
    if profile in {"mirror-newton-cuda", "mirror-newton-cuda-render"}:
        return IsaacSessionSpec(
            experience_family="mirror",
            physics=IsaacNewtonCudaSpec(),
            render=IsaacRenderSpec(enabled=profile == "mirror-newton-cuda-render"),
            **common,
        )
    if profile in {
        "kaleidoscope-physx-cuda",
        "kaleidoscope-physx-cuda-viewport",
    }:
        viewport = profile.endswith("-viewport")
        return IsaacSessionSpec(
            experience_family="kaleidoscope",
            compute=IsaacComputeSpec(cuda_device=cuda_device),
            physics=IsaacPhysxCudaSpec(),
            render=IsaacRenderSpec(
                enabled=viewport,
                visible_world_indices=(0,) if viewport else None,
            ),
            rendering_dt=common["physics_dt"],
            physics_dt=common["physics_dt"],
            gravity_z=common["gravity_z"],
            add_ground=False,
        )
    if profile in {
        "kaleidoscope-newton-cuda",
        "kaleidoscope-newton-cuda-viewport",
    }:
        viewport = profile.endswith("-viewport")
        return IsaacSessionSpec(
            experience_family="kaleidoscope",
            compute=IsaacComputeSpec(cuda_device=cuda_device),
            physics=IsaacNewtonCudaSpec(
                world_count=2,
            ),
            render=IsaacRenderSpec(
                enabled=viewport,
                visible_world_indices=(0,) if viewport else None,
            ),
            rendering_dt=common["physics_dt"],
            physics_dt=common["physics_dt"],
            gravity_z=common["gravity_z"],
            add_ground=False,
        )
    raise ValueError(f"unsupported runtime check profile {profile!r}")


def _physics_owner_probe(session: object, spec: IsaacSessionSpec) -> dict[str, object]:
    """验收 concrete owner；只对已经完整构造的 PhysX World 推进一步。"""

    runtime = getattr(session, "physics_runtime", None)
    if runtime is None:
        raise RuntimeError("IsaacSession did not publish its physics runtime")
    if getattr(runtime, "kind", None) != spec.physics.kind:
        raise RuntimeError(
            "physics runtime kind mismatch: "
            f"actual={getattr(runtime, 'kind', None)!r}, expected={spec.physics.kind!r}"
        )
    if spec.physics.kind.startswith("physx_"):
        runtime.reset()
        runtime.step(render=False)
    elif hasattr(runtime, "world"):
        raise RuntimeError("Newton runtime must not own an Isaac World")
    return {
        "backend": runtime.backend,
        "compute_device": spec.compute_device,
        "physics_device": spec.physics_device,
        "execution": runtime.execution,
        "kind": runtime.kind,
        "physics_dt": runtime.get_physics_dt(),
        "rendering_dt": runtime.get_rendering_dt(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """在受监督 Kit worker 内执行 owner 与依赖闭包检查。"""

    arguments = list(sys.argv[1:] if argv is None else argv)
    # --help/非法参数属于纯 CLI 路径，必须在创建受监督 Isaac worker 之前结束；否则
    # argparse 的正常退出会被 supervisor 误判为缺少运行态 marker。
    args = parse_args(arguments)
    if not in_runtime_worker():
        return run_supervised_worker(
            script_path=Path(__file__),
            argv=arguments,
            required_markers=(PHYSICS_RUNTIME_MARKER, DEPENDENCY_RUNTIME_MARKER),
            success_marker=SUCCESS_MARKER,
        )
    spec = _session_spec(args.profile, cuda_device=args.cuda_device)
    os.environ["LINKERBOT_RUNTIME_PROVENANCE"] = "0"
    session = create_isaac_session_from_spec(spec=spec)
    physics_execution = spec.physics_execution
    try:
        owner_report = _physics_owner_probe(session, spec)
        provenance = collect_runtime_provenance(
            cuda_device=spec.compute.cuda_device,
            include_curobo=True,
            physics_execution=physics_execution,
        )
        validate_target_runtime(
            provenance,
            require_curobo=True,
            expected_physics_backend=owner_report["backend"],
            physics_execution=physics_execution,
            experience_family=spec.experience_family,
            rendering_required=spec.render.enabled,
        )
        print(
            PHYSICS_RUNTIME_MARKER
            + " "
            + json.dumps(owner_report, ensure_ascii=True, sort_keys=True),
            flush=True,
        )
        print(
            DEPENDENCY_RUNTIME_MARKER + " " + format_runtime_provenance(provenance),
            flush=True,
        )
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
