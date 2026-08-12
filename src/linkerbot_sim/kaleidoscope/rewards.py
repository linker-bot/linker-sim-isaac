"""T-block v1 reward 的设备实现。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def tblock_reward(
    *,
    distance: "torch.Tensor",
    previous_distance: "torch.Tensor",
    heading_error: "torch.Tensor",
    previous_heading_error: "torch.Tensor",
    hand_distance: "torch.Tensor",
    previous_hand_distance: "torch.Tensor",
    action: "torch.Tensor",
    previous_action: "torch.Tensor",
    success: "torch.Tensor",
    task_failure: "torch.Tensor",
    numeric_failure: "torch.Tensor",
    distance_progress_weight: float = 8.0,
    heading_progress_weight: float = 0.5,
    hand_progress_weight: float = 0.25,
    action_l2_weight: float = -0.002,
    action_rate_l2_weight: float = -0.010,
    success_reward: float = 10.0,
    task_failure_reward: float = -5.0,
) -> "torch.Tensor":
    """逐项实现冻结公式；numeric failure 行强制返回有限的 0。"""

    import torch

    reward = (
        distance_progress_weight * (previous_distance - distance)
        + heading_progress_weight * (previous_heading_error - heading_error)
        + hand_progress_weight * (previous_hand_distance - hand_distance)
        + action_l2_weight * torch.mean(action * action, dim=1)
        + action_rate_l2_weight * torch.mean((action - previous_action) ** 2, dim=1)
        + success_reward * success.to(dtype=action.dtype)
        + task_failure_reward * task_failure.to(dtype=action.dtype)
    )
    return torch.where(numeric_failure, torch.zeros_like(reward), reward)


__all__ = ["tblock_reward"]
