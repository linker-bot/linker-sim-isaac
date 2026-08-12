"""cuRobo kinematics capability 的设备原生结果与 CUDA 校验原语。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from linkerbot_sim.utils.tensors import (
    assert_finite_async,
    require_common_cuda_device,
    require_cuda_tensor,
)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class BatchIKTensorResult:
    """一次设备原生 IK 的固定 shape 结果。"""

    joint_positions: "torch.Tensor"
    success: "torch.Tensor"
    position_error: "torch.Tensor"
    orientation_error: "torch.Tensor | None" = None

    def __post_init__(self) -> None:
        import torch

        q = require_cuda_tensor(self.joint_positions, name="IK joint positions", ndim=2)
        success = require_cuda_tensor(
            self.success, name="IK success", ndim=1, dtype=torch.bool
        )
        position_error = require_cuda_tensor(
            self.position_error, name="IK position error", ndim=1
        )
        tensors = [q, success, position_error]
        if self.orientation_error is not None:
            tensors.append(
                require_cuda_tensor(
                    self.orientation_error, name="IK orientation error", ndim=1
                )
            )
        require_common_cuda_device(tensors, label="IK result")
        if any(value.shape[0] != q.shape[0] for value in tensors[1:]):
            raise ValueError("IK result leading dimensions must match")
        assert_finite_async(q, name="IK joint positions")
        assert_finite_async(position_error, name="IK position error")
        if self.orientation_error is not None:
            assert_finite_async(self.orientation_error, name="IK orientation error")


@dataclass(frozen=True, slots=True)
class BatchIKWaypointTensorResult:
    """同步直线 primitive 使用的 ``(T,N,C)`` waypoint IK 结果。"""

    joint_positions: "torch.Tensor"
    success: "torch.Tensor"
    first_failure_step: "torch.Tensor"
    position_error: "torch.Tensor"
    orientation_error: "torch.Tensor | None" = None

    def __post_init__(self) -> None:
        import torch

        q = require_cuda_tensor(
            self.joint_positions, name="waypoint IK joint positions", ndim=3
        )
        success = require_cuda_tensor(
            self.success, name="waypoint IK success", ndim=1, dtype=torch.bool
        )
        first_failure = require_cuda_tensor(
            self.first_failure_step,
            name="waypoint IK first failure step",
            ndim=1,
            dtype=torch.int64,
        )
        position_error = require_cuda_tensor(
            self.position_error, name="waypoint IK position error", ndim=1
        )
        tensors = [q, success, first_failure, position_error]
        if self.orientation_error is not None:
            tensors.append(
                require_cuda_tensor(
                    self.orientation_error,
                    name="waypoint IK orientation error",
                    ndim=1,
                )
            )
        require_common_cuda_device(tensors, label="waypoint IK result")
        rows = q.shape[1]
        if any(value.shape[0] != rows for value in tensors[1:]):
            raise ValueError("waypoint IK result env dimensions must match")
        if q.shape[0] < 1 or q.shape[2] < 1:
            raise ValueError(
                "waypoint IK result must have non-empty T and C dimensions"
            )
        assert_finite_async(q, name="waypoint IK joint positions")
        assert_finite_async(position_error, name="waypoint IK position error")
        if self.orientation_error is not None:
            assert_finite_async(
                self.orientation_error, name="waypoint IK orientation error"
            )


__all__ = ["BatchIKTensorResult", "BatchIKWaypointTensorResult"]
