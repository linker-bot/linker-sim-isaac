"""Kaleidoscope 共用的 CUDA 角度与 wxyz 四元数运算。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from linkerbot_sim.kaleidoscope.tensors import (
    assert_finite_async,
    require_common_cuda_device,
    require_cuda_tensor,
)

if TYPE_CHECKING:
    import torch


def normalize_quaternion_wxyz(value: "torch.Tensor") -> "torch.Tensor":
    """归一化批量四元数；零/非有限输入通过设备断言失败。"""

    import torch

    quaternion = require_cuda_tensor(value, name="quaternion", ndim=2)
    if quaternion.shape[1:] != (4,):
        raise ValueError("quaternion must have shape (N,4)")
    assert_finite_async(quaternion, name="quaternion")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    torch._assert_async(torch.all(norm > 1.0e-8), "quaternion norm is zero")
    return quaternion / norm


def quaternion_multiply_wxyz(
    left: "torch.Tensor",
    right: "torch.Tensor",
    *,
    normalize_result: bool = False,
) -> "torch.Tensor":
    """按最后一维计算 Hamilton 积，可选归一化结果且不发生主机同步。"""

    import torch

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    result = torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )
    if not normalize_result:
        return result
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(
        1.0e-8
    )


def quaternion_rotate_wxyz(
    quaternion: "torch.Tensor", vector: "torch.Tensor"
) -> "torch.Tensor":
    """用单位 wxyz 四元数批量旋转最后一维为 3 的向量。"""

    import torch

    scalar = quaternion[..., :1]
    imaginary = quaternion[..., 1:]
    return vector + 2.0 * torch.cross(
        imaginary,
        torch.cross(imaginary, vector, dim=-1) + scalar * vector,
        dim=-1,
    )


def quaternion_slerp_wxyz(
    start: "torch.Tensor",
    target: "torch.Tensor",
    alpha: "torch.Tensor",
) -> "torch.Tensor":
    """执行最短弧 SLERP；输入 shape 为 ``(N,4),(N,4),(T,)``。"""

    import torch

    q0 = normalize_quaternion_wxyz(start)
    q1 = normalize_quaternion_wxyz(target)
    progress = require_cuda_tensor(alpha, name="slerp alpha", ndim=1)
    require_common_cuda_device((q0, q1, progress), label="slerp inputs")
    if q0.shape != q1.shape:
        raise ValueError("slerp start and target shapes must match")
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(max=1.0)

    # 近重合时 lerp 避免 sin(theta) 的数值放大；where 保持固定设备控制流。
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    progress = progress[:, None, None]
    safe_denominator = torch.where(
        sin_theta.abs() > 1.0e-6, sin_theta, torch.ones_like(sin_theta)
    )
    spherical = (
        torch.sin((1.0 - progress) * theta[None, :, :])
        / safe_denominator[None, :, :]
        * q0[None, :, :]
        + torch.sin(progress * theta[None, :, :])
        / safe_denominator[None, :, :]
        * q1[None, :, :]
    )
    linear = (1.0 - progress) * q0[None, :, :] + progress * q1[None, :, :]
    use_linear = (sin_theta.abs() <= 1.0e-6)[None, :, :]
    return normalize_quaternion_wxyz(
        torch.where(use_linear, linear, spherical).reshape(-1, 4)
    ).reshape(alpha.numel(), q0.shape[0], 4)


def wrap_to_pi(value: "torch.Tensor") -> "torch.Tensor":
    """把任意弧度角映射到 ``[-pi, pi]``，保持 tensor 的设备和 shape。"""

    import torch

    return torch.atan2(torch.sin(value), torch.cos(value))


__all__ = [
    "normalize_quaternion_wxyz",
    "quaternion_multiply_wxyz",
    "quaternion_rotate_wxyz",
    "quaternion_slerp_wxyz",
    "wrap_to_pi",
]
