"""不带碰撞模型的 CUDA batch IK 合同。

Kaleidoscope 只依赖 kinematics capability；该 import closure 不允许出现 MotionPlanner、collision
world/cache 或规划 worker。cuRobo adapter 应实现这里的 Protocol，并保持输入输出在同一 GPU。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from linkerbot_sim.backends.curobo.kinematics.types import (
    BatchIKTensorResult,
    BatchIKWaypointTensorResult,
)
from linkerbot_sim.kaleidoscope.geometry import (
    quaternion_multiply_wxyz,
    quaternion_rotate_wxyz,
)
from linkerbot_sim.kaleidoscope.tensors import (
    assert_finite_async,
    require_common_cuda_device,
    require_cuda_tensor,
)

if TYPE_CHECKING:
    import torch


@runtime_checkable
class DeviceBatchIKSolver(Protocol):
    """cuRobo kinematics adapter 必须实现的设备合同。"""

    device: "torch.device"
    command_dim: int

    def solve(
        self,
        *,
        target_positions: "torch.Tensor",
        target_orientations_wxyz: "torch.Tensor | None",
        seeds: "torch.Tensor",
        active_mask: "torch.Tensor | None" = None,
    ) -> BatchIKTensorResult: ...

    def solve_waypoints(
        self,
        *,
        target_positions: "torch.Tensor",
        target_orientations_wxyz: "torch.Tensor | None",
        seeds: "torch.Tensor",
        active_mask: "torch.Tensor | None" = None,
    ) -> BatchIKWaypointTensorResult: ...

    def close(self) -> None: ...


class EnvLocalDeviceBatchIKSolver:
    """把 env-local TCP goal 转成 cuRobo robot-base frame 的 CUDA wrapper。

    PhysX state 先减去 env origin，因此 action term 使用 env-local pose；cuRobo 模型则以
    robot base 为原点。root pose 是构造期固定事实，本 wrapper 在 GPU 上执行逆刚体变换，
    不让 scene assembly 或 task 混入 frame 数学。
    """

    def __init__(
        self,
        solver: DeviceBatchIKSolver,
        *,
        robot_root_position_local: "torch.Tensor",
        robot_root_orientation_wxyz: "torch.Tensor",
    ) -> None:
        position = require_cuda_tensor(
            robot_root_position_local,
            name="robot root position",
            ndim=1,
        )
        orientation = require_cuda_tensor(
            robot_root_orientation_wxyz,
            name="robot root orientation",
            ndim=1,
        )
        if position.shape != (3,) or orientation.shape != (4,):
            raise ValueError("robot root transform must have shapes (3,) and (4,)")
        if position.device != solver.device or orientation.device != solver.device:
            raise ValueError("robot root transform must live on solver.device")
        self.solver = solver
        self.device = solver.device
        self.command_dim = solver.command_dim
        self.root_position = position.clone()
        norm = orientation.square().sum().sqrt().clamp_min(1.0e-8)
        normalized = orientation / norm
        self.inverse_root_orientation = normalized.clone()
        self.inverse_root_orientation[1:].neg_()
        self._closed = False

    def solve(
        self,
        *,
        target_positions: "torch.Tensor",
        target_orientations_wxyz: "torch.Tensor | None",
        seeds: "torch.Tensor",
        active_mask: "torch.Tensor | None" = None,
    ) -> BatchIKTensorResult:
        self._require_open()
        positions, orientations = self._to_base_frame(
            target_positions,
            target_orientations_wxyz,
        )
        return self.solver.solve(
            target_positions=positions,
            target_orientations_wxyz=orientations,
            seeds=seeds,
            active_mask=active_mask,
        )

    def solve_waypoints(
        self,
        *,
        target_positions: "torch.Tensor",
        target_orientations_wxyz: "torch.Tensor | None",
        seeds: "torch.Tensor",
        active_mask: "torch.Tensor | None" = None,
    ) -> BatchIKWaypointTensorResult:
        self._require_open()
        positions = require_cuda_tensor(
            target_positions,
            name="env-local waypoint positions",
            ndim=3,
        )
        steps, rows, _ = positions.shape
        flat_orientation = (
            None
            if target_orientations_wxyz is None
            else require_cuda_tensor(
                target_orientations_wxyz,
                name="env-local waypoint orientations",
                ndim=3,
            ).reshape(steps * rows, 4)
        )
        base_position, base_orientation = self._to_base_frame(
            positions.reshape(steps * rows, 3),
            flat_orientation,
        )
        return self.solver.solve_waypoints(
            target_positions=base_position.reshape(steps, rows, 3),
            target_orientations_wxyz=(
                None
                if base_orientation is None
                else base_orientation.reshape(steps, rows, 4)
            ),
            seeds=seeds,
            active_mask=active_mask,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.solver.close()
        self._closed = True

    def _to_base_frame(
        self,
        positions: "torch.Tensor",
        orientations: "torch.Tensor | None",
    ) -> tuple["torch.Tensor", "torch.Tensor | None"]:
        value = require_cuda_tensor(
            positions,
            name="env-local IK positions",
            ndim=2,
        )
        inverse = self.inverse_root_orientation[None, :].expand(value.shape[0], -1)
        translated = value - self.root_position[None, :]
        base_position = quaternion_rotate_wxyz(inverse, translated)
        if orientations is None:
            return base_position, None
        quaternion = require_cuda_tensor(
            orientations,
            name="env-local IK orientations",
            ndim=2,
        )
        if quaternion.shape != (value.shape[0], 4):
            raise ValueError("IK orientations must have shape (N,4)")
        return base_position, quaternion_multiply_wxyz(
            inverse,
            quaternion,
            normalize_result=True,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("env-local IK solver is closed")


def solve_ik_batch(
    solver: DeviceBatchIKSolver,
    *,
    target_positions: "torch.Tensor",
    target_orientations_wxyz: "torch.Tensor | None",
    seeds: "torch.Tensor",
    active_mask: "torch.Tensor | None" = None,
) -> BatchIKTensorResult:
    """校验设备边界、调用 IK，并让失败行保持 seed。"""

    import torch

    positions = require_cuda_tensor(
        target_positions, name="IK target positions", ndim=2
    )
    if positions.shape[1:] != (3,):
        raise ValueError("IK target_positions must have shape (N,3)")
    seed = require_cuda_tensor(seeds, name="IK seeds", ndim=2)
    if seed.shape[0] != positions.shape[0]:
        raise ValueError("IK seeds and targets must share batch size")
    if seed.shape[1] != int(solver.command_dim):
        raise ValueError("IK seed width does not match solver.command_dim")
    tensors = [positions, seed]
    orientations = target_orientations_wxyz
    if orientations is not None:
        orientations = require_cuda_tensor(
            orientations, name="IK target orientations", ndim=2
        )
        if orientations.shape != (positions.shape[0], 4):
            raise ValueError("IK target_orientations_wxyz must have shape (N,4)")
        tensors.append(orientations)
    mask = active_mask
    if mask is not None:
        mask = require_cuda_tensor(
            mask, name="IK active mask", ndim=1, dtype=torch.bool
        )
        if mask.shape[0] != positions.shape[0]:
            raise ValueError("IK active_mask must have shape (N,)")
        tensors.append(mask)
    device = require_common_cuda_device(tensors, label="IK inputs")
    if device != solver.device:
        raise ValueError(f"IK solver uses {solver.device}, inputs use {device}")
    assert_finite_async(positions, name="IK target positions")
    assert_finite_async(seed, name="IK seeds")
    if orientations is not None:
        assert_finite_async(orientations, name="IK target orientations")

    raw = solver.solve(
        target_positions=positions,
        target_orientations_wxyz=orientations,
        seeds=seed,
        active_mask=mask,
    )
    if raw.joint_positions.shape != seed.shape:
        raise ValueError("IK result joint_positions must match seed shape")
    effective_success = raw.success if mask is None else raw.success & mask
    held = torch.where(effective_success[:, None], raw.joint_positions, seed)
    return BatchIKTensorResult(
        joint_positions=held,
        success=effective_success,
        position_error=raw.position_error,
        orientation_error=raw.orientation_error,
    )


__all__ = [
    "BatchIKTensorResult",
    "BatchIKWaypointTensorResult",
    "DeviceBatchIKSolver",
    "EnvLocalDeviceBatchIKSolver",
    "solve_ik_batch",
]
