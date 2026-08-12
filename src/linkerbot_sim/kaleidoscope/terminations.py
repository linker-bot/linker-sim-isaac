"""T-block v1 success/failure/horizon 判定。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class TerminationResult:
    success: "torch.Tensor"
    task_failure: "torch.Tensor"
    next_success_streak: "torch.Tensor"
    terminated: "torch.Tensor"
    truncated: "torch.Tensor"


def evaluate_tblock_termination(
    *,
    block_position: "torch.Tensor",
    distance: "torch.Tensor",
    heading_error: "torch.Tensor",
    planar_speed: "torch.Tensor",
    success_streak: "torch.Tensor",
    next_episode_length: "torch.Tensor",
    numeric_failure: "torch.Tensor",
    external_safety_stop: "torch.Tensor",
    horizon: int,
    success_distance_m: float = 0.02,
    success_heading_rad: float = 0.10,
    success_planar_speed_m_s: float = 0.03,
    required_success_streak: int = 5,
    failure_aabb_min: tuple[float, float, float] = (-0.05, -0.20, -0.48),
    failure_aabb_max: tuple[float, float, float] = (0.35, 0.20, -0.28),
) -> TerminationResult:
    """按冻结顺序计算连续成功、工作区失败与 truncation。"""

    import torch

    success_now = (
        (~numeric_failure)
        & (distance <= success_distance_m)
        & (heading_error <= success_heading_rad)
        & (planar_speed <= success_planar_speed_m_s)
    )
    next_streak = torch.where(
        success_now, success_streak + 1, torch.zeros_like(success_streak)
    )
    success = next_streak >= required_success_streak
    minimum = failure_aabb_min
    maximum = failure_aabb_max
    within = (
        (block_position[:, 0] >= minimum[0])
        & (block_position[:, 0] <= maximum[0])
        & (block_position[:, 1] >= minimum[1])
        & (block_position[:, 1] <= maximum[1])
        & (block_position[:, 2] >= minimum[2])
        & (block_position[:, 2] <= maximum[2])
    )
    task_failure = (~numeric_failure) & (~within)
    terminated = (~numeric_failure) & (success | task_failure)
    truncated = (
        (next_episode_length >= int(horizon)) | numeric_failure | external_safety_stop
    )
    return TerminationResult(
        success=success,
        task_failure=task_failure,
        next_success_streak=next_streak,
        terminated=terminated,
        truncated=truncated,
    )


__all__ = ["TerminationResult", "evaluate_tblock_termination"]
