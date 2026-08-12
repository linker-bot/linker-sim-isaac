#!/usr/bin/env python3
"""在真实 CUDA 设备上验证项目固定的 cuRobo 运行时闭包。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.backends.curobo import (  # noqa: E402
    CuroboConfig,
    CuroboContext,
    curobo_config_from_profiles,
)
from linkerbot_sim.backends.curobo.warp_compat import (  # noqa: E402
    ensure_warp_torch_namespace_compatible,
)
from linkerbot_sim.configuration import load_mirror_config  # noqa: E402
from linkerbot_sim.planning.collision_objects import CollisionObject  # noqa: E402
from linkerbot_sim.planning.requests import IKRequest, MotionRequest  # noqa: E402


ensure_warp_torch_namespace_compatible()

import torch  # noqa: E402
import warp as wp  # noqa: E402


@wp.kernel
def _warp_add_one(
    source: wp.array(dtype=wp.float32),
    destination: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    destination[index] = source[index] + 1.0


def _timed(callable_, *, cuda_device: int):
    started = perf_counter()
    value = callable_()
    torch.cuda.synchronize(cuda_device)
    return value, perf_counter() - started


def _gpu_probe(*, cuda_device: int) -> dict[str, object]:
    from curobo._src.curobolib.backends import get_backend_name

    device = f"cuda:{cuda_device}"
    torch.cuda.set_device(cuda_device)
    wp.init()
    source = torch.arange(8, device=device, dtype=torch.float32)
    torch_result = source.square() + 1.0
    torch.cuda.synchronize(cuda_device)

    destination = torch.empty_like(source)
    wp.launch(
        _warp_add_one,
        dim=source.numel(),
        inputs=[wp.from_torch(source), wp.from_torch(destination)],
        device=device,
    )
    wp.synchronize_device(device)
    expected = torch.arange(1, 9, device=device, dtype=torch.float32)
    if not torch.equal(destination, expected):
        raise RuntimeError("Warp CUDA kernel returned an unexpected result")

    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_device,
        "torch_device": device,
        "device_name": torch.cuda.get_device_name(cuda_device),
        "device_capability": list(torch.cuda.get_device_capability(cuda_device)),
        "torch_result": torch_result.cpu().tolist(),
        "warp_version": wp.__version__,
        "warp_torch_device": str(wp.torch.device_from_torch(torch.device(device))),
        "warp_result": destination.cpu().tolist(),
        "curobo_kernel_backend": str(get_backend_name()),
    }


def _load_config(*, cuda_device_override: int | None) -> tuple[CuroboConfig, int]:
    """从正式 Mirror 配置图组合 cuRobo 与机器人资源。"""

    mode_config = load_mirror_config("physx_cpu")
    cuda_device = (
        mode_config.cuda_device
        if cuda_device_override is None
        else cuda_device_override
    )
    robot = mode_config.scene.robots[0]
    if robot.resolved_profile is None:
        raise RuntimeError("Mirror config lost its resolved robot profile")
    config = curobo_config_from_profiles(
        robot.resolved_profile,
        curobo_settings=mode_config.curobo,
        cuda_device=cuda_device,
    )
    expected_device = f"cuda:{cuda_device}"
    if config.device.device != expected_device:
        raise RuntimeError(
            "cuRobo composition produced an inconsistent device: "
            f"expected={expected_device!r}, actual={config.device.device!r}"
        )
    return config, cuda_device


def _load_context(
    config: CuroboConfig, *, cuda_device: int
) -> tuple[CuroboContext, float]:
    """在根设备投影出的后端配置上创建 context，并同步同一张 GPU。"""

    started = perf_counter()
    context = CuroboContext(
        config,
        cache_root=REPO_ROOT / ".cache" / "curobo-runtime-smoke",
    )
    torch.cuda.synchronize(cuda_device)
    return context, perf_counter() - started


def _far_cuboid() -> CollisionObject:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [5.0, 5.0, 5.0]
    return CollisionObject(
        name="runtime_smoke_far_cuboid",
        shape="cuboid",
        pose=pose,
        size=(0.1, 0.1, 0.1),
    )


def _far_mesh_scene(context: CuroboContext):
    scene = context.scene_module
    mesh = scene.Cuboid(
        name="runtime_smoke_far_mesh_source",
        pose=[5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0],
        dims=[1.0, 1.0, 1.0],
    ).get_mesh()
    mesh.name = "runtime_smoke_far_mesh"
    return scene.Scene(mesh=[mesh])


def _curobo_probe(
    config: CuroboConfig,
    *,
    cuda_device: int,
    include_planning: bool,
) -> dict[str, object]:
    context = None
    result: dict[str, object] = {}
    try:
        context, context_seconds = _load_context(config, cuda_device=cuda_device)
        result.update(
            {
                "context_seconds": context_seconds,
                "kernel_backend": context.kernel_backend,
                "joint_names": context.joint_names(),
                "frame_names": context.frame_names(),
                "robot_collision_spheres": context.robot_sphere_count(),
            }
        )

        zero = np.zeros(len(context.joint_names()), dtype=float)
        goal = np.asarray([0.05, -0.05, 0.04, -0.04, 0.03, -0.03, 0.02])
        fk = context.make_forward_kinematics()
        zero_pose, cold_fk_seconds = _timed(
            lambda: fk.compute_pose(zero, context.default_tcp_frame),
            cuda_device=cuda_device,
        )
        goal_pose, warm_fk_seconds = _timed(
            lambda: fk.compute_pose(goal, context.default_tcp_frame),
            cuda_device=cuda_device,
        )
        result["fk"] = {
            "cold_seconds": cold_fk_seconds,
            "warm_seconds": warm_fk_seconds,
            "zero_position": zero_pose.position.tolist(),
            "goal_position": goal_pose.position.tolist(),
            "goal_orientation_wxyz": goal_pose.orientation.tolist(),
        }

        ik = context.make_inverse_kinematics()
        ik_request = IKRequest(
            target_position=goal_pose.position,
            target_orientation=goal_pose.orientation,
            warm_start_ik_cspace_seed=zero,
            # 正式 Mirror profile 的 direct IK 是低开销、无碰撞能力；场景碰撞由下面的
            # MotionPlanner 验证。诊断必须服从生产配置，不能为了 smoke 临时扩展 capability。
            avoid_collisions=False,
        )
        cold_ik, cold_ik_seconds = _timed(
            lambda: ik.solve(ik_request), cuda_device=cuda_device
        )
        warm_ik, warm_ik_seconds = _timed(
            lambda: ik.solve(ik_request), cuda_device=cuda_device
        )
        if not cold_ik.success or not warm_ik.success:
            raise RuntimeError(
                "cuRobo IK smoke failed: "
                f"cold={cold_ik.status!r}, warm={warm_ik.status!r}"
            )
        result["ik"] = {
            "cold_seconds": cold_ik_seconds,
            "warm_seconds": warm_ik_seconds,
            "position_error": warm_ik.position_error,
            "orientation_error": warm_ik.orientation_error,
            "solution": warm_ik.joint_positions.tolist(),
        }
        if include_planning:
            # 先同步 canonical world；随后 lazy 创建 planner 时，context 会把同一份 world
            # 投影到 planner。IK solver 不会收到它不支持的碰撞请求。
            collision_world = context.sync_collision_world((_far_cuboid(),))
            planner = context.make_motion_planner()
            request = MotionRequest(
                current_q=zero,
                goal_q=goal,
                avoid_collisions=True,
            )
            cold_plan, cold_plan_seconds = _timed(
                lambda: planner.plan(request), cuda_device=cuda_device
            )
            warm_plan, warm_plan_seconds = _timed(
                lambda: planner.plan(request), cuda_device=cuda_device
            )
            if not cold_plan.success or not warm_plan.success:
                raise RuntimeError(
                    "cuRobo planning smoke failed: "
                    f"cold={cold_plan.diagnostics.message!r}, "
                    f"warm={warm_plan.diagnostics.message!r}"
                )
            result["planning"] = {
                "cold_seconds": cold_plan_seconds,
                "warm_seconds": warm_plan_seconds,
                "waypoints": int(warm_plan.path.shape[0]),
                "path_length": warm_plan.diagnostics.metrics.get("path_length"),
            }
            result["cuboid_collision"] = {
                "canonical_obstacles": collision_world.num_canonical_obstacles,
                "materialized_counts": collision_world.materialized_counts,
                "capability_available": context.collision_capability(
                    consumer="planner"
                ).available,
            }

            raw_planner = context.motion_planner
            mesh_scene = _far_mesh_scene(context)
            context.validate_collision_cache_capacity(
                {"cuboid": 0, "mesh": 1}, consumer="planner"
            )
            mesh_update_seconds = 0.0
            mesh_plan_seconds = 0.0
            restore_seconds = 0.0
            try:
                _, mesh_update_seconds = _timed(
                    lambda: raw_planner.update_world(mesh_scene),
                    cuda_device=cuda_device,
                )
                start_state = context.joint_state_from_positions(zero.reshape(1, -1))
                goal_state = context.joint_state_from_positions(goal.reshape(1, -1))
                mesh_plan, mesh_plan_seconds = _timed(
                    lambda: raw_planner.plan_cspace(goal_state, start_state),
                    cuda_device=cuda_device,
                )
                mesh_success = bool(torch.all(mesh_plan.success).item())
                if not mesh_success:
                    raise RuntimeError(
                        "cuRobo raw mesh planning smoke failed: "
                        f"status={mesh_plan.status!r}"
                    )
            finally:
                _, restore_seconds = _timed(
                    collision_world.update_solvers,
                    cuda_device=cuda_device,
                )
            result["mesh_collision"] = {
                "mesh_count": len(mesh_scene.mesh or ()),
                "update_seconds": mesh_update_seconds,
                "planning_seconds": mesh_plan_seconds,
                "planning_success": mesh_success,
                "canonical_restore_seconds": restore_seconds,
                "checker_available": (
                    getattr(raw_planner, "scene_collision_checker", None) is not None
                ),
            }

        result["cuda_memory_before_close"] = {
            "cuda_device": cuda_device,
            "allocated": torch.cuda.memory_allocated(cuda_device),
            "reserved": torch.cuda.memory_reserved(cuda_device),
        }
        context.close()
        context.close()
        result["close_idempotent"] = True
        result["remaining_solvers"] = len(context.existing_solvers())
        return result
    finally:
        if context is not None and context.existing_solvers():
            context.close()


def _cuda_device_argument(value: str) -> int:
    """argparse 类型：拒绝负设备编号，避免在 Torch 初始化后才失败。"""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析诊断选项；未覆盖时设备来自正式 Mirror mode root。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-planning",
        action="store_true",
        help="只验证 GPU、FK 和无碰撞 IK，不创建 MotionPlanner 或碰撞世界",
    )
    parser.add_argument(
        "--cuda-device",
        type=_cuda_device_argument,
        default=None,
        help="显式覆盖 Mirror mode root 的 compute.cuda_device",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, cuda_device = _load_config(cuda_device_override=args.cuda_device)
    report = {
        "gpu": _gpu_probe(cuda_device=cuda_device),
        "curobo": _curobo_probe(
            config,
            cuda_device=cuda_device,
            include_planning=not args.skip_planning,
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
