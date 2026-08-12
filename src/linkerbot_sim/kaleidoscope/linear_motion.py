"""CUDA 上的同步 TCP 直线运动 primitive；它不是轨迹规划器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from linkerbot_sim.kaleidoscope.geometry import (
    normalize_quaternion_wxyz,
    quaternion_slerp_wxyz,
)
from linkerbot_sim.kaleidoscope.ik import DeviceBatchIKSolver
from linkerbot_sim.kaleidoscope.tensors import (
    assert_finite_async,
    require_common_cuda_device,
    require_cuda_tensor,
)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class BatchLinearMotionTensorResult:
    """固定 tick 的 joint targets 与逐环境失败诊断。"""

    joint_positions: "torch.Tensor"
    success: "torch.Tensor"
    first_failure_step: "torch.Tensor"
    position_error: "torch.Tensor"
    orientation_error: "torch.Tensor | None" = None


def path_progress(
    *, steps: int, mode: str, device: "torch.device", dtype: "torch.dtype"
) -> "torch.Tensor":
    """生成不含起点、包含终点的固定 path progress。"""

    import torch

    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive int")
    if mode not in {"linear", "smoothstep"}:
        raise ValueError("progress mode must be linear or smoothstep")
    alpha = torch.arange(1, steps + 1, device=device, dtype=dtype) / float(steps)
    return alpha if mode == "linear" else alpha * alpha * (3.0 - 2.0 * alpha)


def solve_linear_motion_batch(
    solver: DeviceBatchIKSolver,
    *,
    start_positions: "torch.Tensor",
    target_positions: "torch.Tensor",
    start_orientations_wxyz: "torch.Tensor",
    target_orientations_wxyz: "torch.Tensor | None",
    seeds: "torch.Tensor",
    waypoint_count: int,
    physics_ticks_per_action: int,
    orientation_mode: str,
    progress_mode: str,
    active_mask: "torch.Tensor | None" = None,
) -> BatchLinearMotionTensorResult:
    """在 TCP 空间建直线路径、批量 IK，并重采样为固定 physics ticks。

    env 维并行、时间维由 solver 以 warm-start 顺序求解。失败环境从首个失败 waypoint 起
    保持最后成功关节目标；其它环境仍执行相同 tick 数，避免一个 World 内各 clone 时间分叉。
    """

    import torch

    start_p = require_cuda_tensor(
        start_positions, name="linear start positions", ndim=2
    )
    target_p = require_cuda_tensor(
        target_positions, name="linear target positions", ndim=2
    )
    start_q = require_cuda_tensor(
        start_orientations_wxyz, name="linear start orientations", ndim=2
    )
    seed = require_cuda_tensor(seeds, name="linear seeds", ndim=2)
    if start_p.shape != target_p.shape or start_p.shape[1:] != (3,):
        raise ValueError("linear positions must have matching shape (N,3)")
    if start_q.shape != (start_p.shape[0], 4):
        raise ValueError("linear start orientation must have shape (N,4)")
    if seed.shape != (start_p.shape[0], int(solver.command_dim)):
        raise ValueError("linear seeds have wrong shape")
    if type(physics_ticks_per_action) is not int or physics_ticks_per_action < 1:
        raise ValueError("physics_ticks_per_action must be a positive int")
    if orientation_mode not in {"free", "current", "target"}:
        raise ValueError("orientation_mode must be free/current/target")
    if progress_mode not in {"linear", "smoothstep"}:
        raise ValueError("progress mode must be linear or smoothstep")
    tensors = [start_p, target_p, start_q, seed]
    target_q = target_orientations_wxyz
    if orientation_mode == "target":
        if target_q is None:
            raise ValueError("target orientation mode requires target quaternion")
        target_q = require_cuda_tensor(
            target_q, name="linear target orientations", ndim=2
        )
        if target_q.shape != start_q.shape:
            raise ValueError("linear target orientation must have shape (N,4)")
        tensors.append(target_q)
    elif target_q is not None:
        raise ValueError(
            "target quaternion is only valid for orientation_mode='target'"
        )
    if active_mask is not None:
        active_mask = require_cuda_tensor(
            active_mask, name="linear active mask", ndim=1, dtype=torch.bool
        )
        if active_mask.shape != (start_p.shape[0],):
            raise ValueError("linear active mask must have shape (N,)")
        tensors.append(active_mask)
    device = require_common_cuda_device(tensors, label="linear motion inputs")
    if device != solver.device:
        raise ValueError(f"linear solver uses {solver.device}, inputs use {device}")
    for label, tensor in (
        ("start positions", start_p),
        ("target positions", target_p),
        ("seeds", seed),
    ):
        assert_finite_async(tensor, name=f"linear {label}")

    # Waypoint 是 IK 的几何离散，不承担执行时间参数化。始终等距采样可确保
    # smoothstep 只在下方 physics tick 重采样时应用一次，也避免改变 IK 路径分辨率。
    alpha = path_progress(
        steps=waypoint_count, mode="linear", device=device, dtype=start_p.dtype
    )
    positions = (
        start_p[None, :, :] + alpha[:, None, None] * (target_p - start_p)[None, :, :]
    )
    if orientation_mode == "free":
        orientations = None
    elif orientation_mode == "current":
        orientations = normalize_quaternion_wxyz(start_q)[None, :, :].expand(
            waypoint_count, -1, -1
        )
    else:
        assert target_q is not None
        orientations = quaternion_slerp_wxyz(start_q, target_q, alpha)

    result = solver.solve_waypoints(
        target_positions=positions,
        target_orientations_wxyz=orientations,
        seeds=seed,
        active_mask=active_mask,
    )
    expected = (waypoint_count, start_p.shape[0], seed.shape[1])
    if result.joint_positions.shape != expected:
        raise ValueError(
            f"waypoint IK result must have shape {expected}, got {tuple(result.joint_positions.shape)}"
        )
    ticks = _resample_waypoints(
        seed,
        result.joint_positions,
        output_steps=physics_ticks_per_action,
        mode=progress_mode,
    )
    return BatchLinearMotionTensorResult(
        joint_positions=ticks,
        success=result.success,
        first_failure_step=result.first_failure_step,
        position_error=result.position_error,
        orientation_error=result.orientation_error,
    )


def _resample_waypoints(
    start: "torch.Tensor",
    waypoints: "torch.Tensor",
    *,
    output_steps: int,
    mode: str,
) -> "torch.Tensor":
    """仅用 Torch gather/lerp 把 waypoint 重采样到固定 tick 数。"""

    import torch

    anchors = torch.cat((start[None, :, :], waypoints), dim=0)
    target = path_progress(
        steps=output_steps, mode=mode, device=start.device, dtype=start.dtype
    ) * float(waypoints.shape[0])
    left = torch.floor(target).to(dtype=torch.int64).clamp(max=waypoints.shape[0] - 1)
    right = (left + 1).clamp(max=waypoints.shape[0])
    weight = (target - left.to(dtype=target.dtype))[:, None, None]
    return torch.lerp(
        anchors.index_select(0, left),
        anchors.index_select(0, right),
        weight,
    )


__all__ = [
    "BatchLinearMotionTensorResult",
    "path_progress",
    "solve_linear_motion_batch",
]
