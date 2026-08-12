"""Kaleidoscope 的 CUDA tensor 边界校验工具。

本模块只检查 tensor 的元数据并在 GPU 上提交数值断言。这样状态、快照和克隆 API 不需要
为了验证 selector 或有限性而把数据下载到 CPU。``torch._assert_async`` 的错误会在同一 CUDA
stream 后续同步点暴露；调用方仍可获得明确错误，同时训练热路径不会插入 ``item()``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from linkerbot_sim.utils.tensors import (
    assert_finite_async,
    require_common_cuda_device,
    require_cuda_tensor,
)

if TYPE_CHECKING:
    import torch


def normalize_env_ids(
    env_ids: "torch.Tensor | None",
    *,
    num_envs: int,
    device: "torch.device",
    allow_empty: bool = False,
) -> "torch.Tensor":
    """规范 env selector，并在设备端检查范围和重复项。

    ``None`` 生成设备端 ``arange``。已有 selector 必须已经位于目标 GPU，避免一个看似方便的
    ``torch.as_tensor`` 悄悄引入 CPU 到 GPU copy。重复和越界检查使用异步设备断言。
    """

    import torch

    if env_ids is None:
        return torch.arange(num_envs, dtype=torch.int64, device=device)
    ids = require_cuda_tensor(env_ids, name="env_ids", ndim=1, dtype=torch.int64)
    if ids.device != device:
        raise ValueError(f"env_ids must live on {device}, got {ids.device}")
    if not allow_empty and ids.numel() == 0:
        raise ValueError("env_ids cannot be empty")
    if ids.numel() == 0:
        return ids

    torch._assert_async(torch.all(ids >= 0), "env_ids contains a negative index")
    torch._assert_async(
        torch.all(ids < int(num_envs)), "env_ids contains an out-of-range index"
    )
    if ids.numel() > 1:
        sorted_ids = torch.sort(ids).values
        torch._assert_async(
            torch.all(sorted_ids[1:] != sorted_ids[:-1]),
            "env_ids cannot contain duplicates",
        )
    return ids


__all__ = [
    "assert_finite_async",
    "normalize_env_ids",
    "require_common_cuda_device",
    "require_cuda_tensor",
]
