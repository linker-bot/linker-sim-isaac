"""T-block 任务的预分配 CUDA buffer 集合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class TaskBuffers:
    """影响下一拍的完整任务状态；这些字段全部进入 snapshot/clone。"""

    episode_length: "torch.Tensor"
    episode_physics_steps: "torch.Tensor"
    episode_return: "torch.Tensor"
    previous_action: "torch.Tensor"
    previous_distance: "torch.Tensor"
    previous_heading_error: "torch.Tensor"
    previous_hand_distance: "torch.Tensor"
    success_streak: "torch.Tensor"
    goal_position: "torch.Tensor"
    goal_yaw: "torch.Tensor"
    reward: "torch.Tensor"
    terminated: "torch.Tensor"
    truncated: "torch.Tensor"
    needs_reset: "torch.Tensor"
    numeric_failure: "torch.Tensor"
    last_finite_observation: "torch.Tensor"
    rng_key: "torch.Tensor"
    rng_counter: "torch.Tensor"

    @classmethod
    def allocate(
        cls,
        *,
        num_envs: int,
        action_dim: int,
        observation_dim: int,
        device: "torch.device",
        dtype: "torch.dtype",
        base_seed: int,
    ) -> "TaskBuffers":
        """一次分配所有持久 buffer，step/reset 不再创建 N-scale host 状态。"""

        import torch

        if num_envs < 1 or action_dim < 1 or observation_dim < 1:
            raise ValueError("task buffer dimensions must be positive")

        def zeros_f(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, device=device, dtype=dtype)

        def zeros_i(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, device=device, dtype=torch.int64)

        def zeros_b(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, device=device, dtype=torch.bool)

        # 每个 env 的 logical key 不依赖物理 grid 位置；clone_state 默认连同 counter 一起复制。
        rng_key = torch.arange(num_envs, device=device, dtype=torch.int64)
        rng_key.mul_(6364136223846793005).add_(int(base_seed))
        return cls(
            episode_length=zeros_i(num_envs),
            episode_physics_steps=zeros_i(num_envs),
            episode_return=zeros_f(num_envs),
            previous_action=zeros_f(num_envs, action_dim),
            previous_distance=zeros_f(num_envs),
            previous_heading_error=zeros_f(num_envs),
            previous_hand_distance=zeros_f(num_envs),
            success_streak=zeros_i(num_envs),
            goal_position=zeros_f(num_envs, 3),
            goal_yaw=zeros_f(num_envs),
            reward=zeros_f(num_envs),
            terminated=zeros_b(num_envs),
            truncated=zeros_b(num_envs),
            needs_reset=zeros_b(num_envs),
            numeric_failure=zeros_b(num_envs),
            last_finite_observation=zeros_f(num_envs, observation_dim),
            rng_key=rng_key,
            rng_counter=zeros_i(num_envs),
        )

    def state_fields(self) -> dict[str, "torch.Tensor"]:
        """返回 snapshot/state API 使用的稳定字段名。"""

        return {
            "task.episode_length": self.episode_length,
            "task.episode_physics_steps": self.episode_physics_steps,
            "task.episode_return": self.episode_return,
            "task.previous_action": self.previous_action,
            "task.previous_distance": self.previous_distance,
            "task.previous_heading_error": self.previous_heading_error,
            "task.previous_hand_distance": self.previous_hand_distance,
            "task.success_streak": self.success_streak,
            "task.goal_position": self.goal_position,
            "task.goal_yaw": self.goal_yaw,
            "task.reward": self.reward,
            "task.terminated": self.terminated,
            "task.truncated": self.truncated,
            "task.needs_reset": self.needs_reset,
            "task.numeric_failure": self.numeric_failure,
            "task.last_finite_observation": self.last_finite_observation,
            "rng.key": self.rng_key,
            "rng.counter": self.rng_counter,
        }


__all__ = ["TaskBuffers"]
