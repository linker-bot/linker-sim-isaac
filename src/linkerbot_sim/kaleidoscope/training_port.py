"""Kaleidoscope 向训练框架公开的最小 CUDA SAME_STEP port。

训练集成只能依赖这个结构化合同，不能穿透到 ``TorchKaleidoscopeEnv``、task buffer 或
tensor helper 的实现模块。token 被刻意声明为 ``object``：它只是一拍事务的不可解释凭据，
skrl 不应读取 generation，更不能自行构造或复用 token。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class KaleidoscopeTrainingPort(Protocol):
    """训练 adapter 所需的稳定、设备原生环境表面。

    返回的 tensor 均位于 ``device``，且是 env 拥有的借用 buffer；consumer 若要跨下一次
    ``step_same_step``/``complete_same_step`` 保存 terminal 数据，必须先复制到自己的 CUDA
    buffer。这个约束避免为每个并行环境逐拍分配新 tensor。
    """

    num_envs: int
    action_dim: int
    action_low: float
    action_high: float
    observation_dim: int
    device: "torch.device"

    def reset(
        self,
    ) -> tuple["torch.Tensor", Mapping[str, "torch.Tensor"]]: ...

    def begin_same_step(self) -> object:
        """占有下一拍 SAME_STEP 事务并返回不透明 token。"""

        ...

    def step_same_step(
        self, token: object, actions: "torch.Tensor"
    ) -> tuple[
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        Mapping[str, "torch.Tensor"],
    ]:
        """推进一拍，但暂不 reset 已结束的行。"""

        ...

    def complete_same_step(self, token: object) -> "torch.Tensor":
        """reset done 行并返回完整的 post-reset observation buffer。"""

        ...

    def close(self) -> None: ...


__all__ = ["KaleidoscopeTrainingPort"]
