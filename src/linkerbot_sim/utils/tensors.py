"""跨产品复用的 tensor 转换与 CUDA 元数据校验。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def require_cuda_tensor(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
    leading_dim: int | None = None,
    dtype: object | None = None,
) -> "torch.Tensor":
    """只校验已有 CUDA tensor 元数据，不做隐式搬运或 dtype 转换。"""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must live on CUDA, got {value.device}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {value.ndim}")
    if leading_dim is not None and (value.ndim == 0 or value.shape[0] != leading_dim):
        raise ValueError(
            f"{name} leading dimension must be {leading_dim}, got {tuple(value.shape)}"
        )
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use dtype={dtype}, got {value.dtype}")
    return value


def require_common_cuda_device(
    tensors: Iterable["torch.Tensor"], *, label: str
) -> "torch.device":
    """验证一组 tensor 全部位于同一个 CUDA device。"""

    iterator = iter(tensors)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError(f"{label} cannot be empty") from exc
    require_cuda_tensor(first, name=f"{label}[0]")
    device = first.device
    for index, tensor in enumerate(iterator, start=1):
        require_cuda_tensor(tensor, name=f"{label}[{index}]")
        if tensor.device != device:
            raise ValueError(
                f"{label} must share one CUDA device; expected {device}, "
                f"got {tensor.device} at index {index}"
            )
    return device


def assert_finite_async(value: "torch.Tensor", *, name: str) -> None:
    """在当前 CUDA stream 提交有限性断言，不同步回主机。"""

    import torch

    if value.is_floating_point() or value.is_complex():
        torch._assert_async(torch.all(torch.isfinite(value)), f"{name} is not finite")


def tensor_like_to_numpy(
    value: object,
    *,
    dtype: object | None = None,
) -> np.ndarray:
    """Move a tensor-like value to CPU and expose it as a NumPy array."""

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return np.asarray(candidate, dtype=dtype)


__all__ = [
    "assert_finite_async",
    "require_common_cuda_device",
    "require_cuda_tensor",
    "tensor_like_to_numpy",
]
