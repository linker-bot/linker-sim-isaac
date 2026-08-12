"""Kaleidoscope VectorTask 的最小设备原生合同。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class TaskStepResult:
    """一次 decision 的 CUDA 输出；所有字段第一维均为 ``num_envs``。"""

    observations: "torch.Tensor"
    rewards: "torch.Tensor"
    terminated: "torch.Tensor"
    truncated: "torch.Tensor"
    info: Mapping[str, "torch.Tensor"]


@runtime_checkable
class VectorTask(Protocol):
    """Runtime 可替换的任务边界；Task 不持有 Isaac handle。"""

    num_envs: int
    device: "torch.device"
    observation_dim: int
    action_dim: int

    def reset_command(self, env_ids: "torch.Tensor") -> object: ...

    def masked_reset_command(
        self, reset_mask: "torch.Tensor", state: object
    ) -> object: ...

    def initialize_after_reset(
        self, env_ids: "torch.Tensor", state: object
    ) -> None: ...

    def initialize_after_masked_reset(
        self, reset_mask: "torch.Tensor", state: object
    ) -> None: ...

    def step(self, state: object, actions: "torch.Tensor") -> TaskStepResult: ...

    def close(self) -> None: ...


__all__ = ["TaskStepResult", "VectorTask"]
