"""GPU 常驻的 Kaleidoscope episode snapshot。

它与长期持久化的 ``SceneSnapshot`` 有意分型：这里的每个字段都是 owned CUDA tensor，适合
课程学习、评估分支和同进程回滚；只有显式调用 checkpoint 冷边界时才允许下载到 CPU。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from linkerbot_sim.controllers.control_mode import (
    require_control_mode,
    require_expected_generation,
)
from linkerbot_sim.controllers.types import ControlMode
from linkerbot_sim.kaleidoscope.tensors import (
    require_common_cuda_device,
    require_cuda_tensor,
)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class KaleidoscopeEpisodeSnapshot:
    """一次批量环境状态的 GPU-owned 快照。"""

    env_ids: "torch.Tensor"
    fields: Mapping[str, "torch.Tensor"]
    compatibility_fingerprint: str = "unbound"
    control_mode: ControlMode | None = "position"
    control_generation: int = 0
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError(
                f"unsupported KaleidoscopeEpisodeSnapshot schema {self.schema_version}"
            )
        if self.schema_version == 2:
            if self.control_mode is None:
                raise ValueError("schema 2 snapshot requires control_mode")
            object.__setattr__(
                self,
                "control_mode",
                require_control_mode(self.control_mode, label="snapshot.control_mode"),
            )
            require_expected_generation(self.control_generation)
        else:
            if self.control_mode not in {None, "position"}:
                raise ValueError("schema 1 snapshot can only represent position mode")
            if self.control_generation != 0:
                raise ValueError("schema 1 snapshot control_generation must be zero")
        if (
            not isinstance(self.compatibility_fingerprint, str)
            or not self.compatibility_fingerprint.strip()
        ):
            raise ValueError("snapshot compatibility_fingerprint cannot be empty")
        ids = require_cuda_tensor(
            self.env_ids, name="snapshot.env_ids", ndim=1, dtype=_torch_int64()
        )
        if not self.fields:
            raise ValueError("snapshot.fields cannot be empty")
        normalized: dict[str, torch.Tensor] = {}
        for name, tensor in self.fields.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("snapshot field names must be non-empty strings")
            value = require_cuda_tensor(
                tensor,
                name=f"snapshot.fields[{name!r}]",
                leading_dim=ids.numel(),
            )
            normalized[name] = value
        device = require_common_cuda_device(
            (ids, *normalized.values()), label="snapshot tensors"
        )
        if ids.device != device:
            raise AssertionError("snapshot device validation is inconsistent")

        # 冻结 mapping 本身，阻止调用者在构造后替换字段；字段 tensor 已由 capture 路径 clone，
        # 因而 snapshot 与 live state 不共享 storage。
        object.__setattr__(self, "fields", MappingProxyType(normalized))

    @property
    def device(self) -> "torch.device":
        return self.env_ids.device

    @property
    def count(self) -> int:
        return self.env_ids.numel()

    def clone(self) -> "KaleidoscopeEpisodeSnapshot":
        """在同一 GPU 上复制整份快照，返回完全独立的 storage。"""

        return KaleidoscopeEpisodeSnapshot(
            env_ids=self.env_ids.clone(),
            fields={name: value.clone() for name, value in self.fields.items()},
            compatibility_fingerprint=self.compatibility_fingerprint,
            control_mode=self.control_mode,
            control_generation=self.control_generation,
            schema_version=self.schema_version,
        )


def _torch_int64() -> object:
    import torch

    return torch.int64


__all__ = ["KaleidoscopeEpisodeSnapshot"]
