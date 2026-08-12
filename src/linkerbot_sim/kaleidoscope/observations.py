"""T-block 状态读取后的 observation 与几何量计算。"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from linkerbot_sim.kaleidoscope.geometry import quaternion_rotate_wxyz, wrap_to_pi
from linkerbot_sim.kaleidoscope.tensors import (
    require_common_cuda_device,
    require_cuda_tensor,
)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True, init=False)
class TBlockState:
    """Runtime 从物理后端 borrowed view 复制出的 owned CUDA 状态。"""

    joint_positions: "torch.Tensor"
    joint_velocities: "torch.Tensor"
    position_references: "torch.Tensor"
    tcp_positions_local: "torch.Tensor"
    tcp_orientations_wxyz: "torch.Tensor"
    block_position_local: "torch.Tensor"
    block_orientation_wxyz: "torch.Tensor"
    block_com_velocity: "torch.Tensor"
    external_safety_stop: "torch.Tensor"

    def __init__(
        self,
        *,
        joint_positions: "torch.Tensor",
        joint_velocities: "torch.Tensor",
        position_references: "torch.Tensor | None" = None,
        command_targets: "torch.Tensor | None" = None,
        tcp_positions_local: "torch.Tensor",
        tcp_orientations_wxyz: "torch.Tensor",
        block_position_local: "torch.Tensor",
        block_orientation_wxyz: "torch.Tensor",
        block_com_velocity: "torch.Tensor",
        external_safety_stop: "torch.Tensor",
    ) -> None:
        if (position_references is None) == (command_targets is None):
            raise ValueError(
                "exactly one of position_references or deprecated command_targets "
                "must be provided"
            )
        reference = (
            position_references if position_references is not None else command_targets
        )
        assert reference is not None
        for name, value in (
            ("joint_positions", joint_positions),
            ("joint_velocities", joint_velocities),
            ("position_references", reference),
            ("tcp_positions_local", tcp_positions_local),
            ("tcp_orientations_wxyz", tcp_orientations_wxyz),
            ("block_position_local", block_position_local),
            ("block_orientation_wxyz", block_orientation_wxyz),
            ("block_com_velocity", block_com_velocity),
            ("external_safety_stop", external_safety_stop),
        ):
            object.__setattr__(self, name, value)

    @property
    def command_targets(self) -> "torch.Tensor":
        """Deprecated read-only alias for position-reference consumers."""

        return self.position_references


@dataclass(frozen=True, slots=True)
class ObservationMetrics:
    observations: "torch.Tensor"
    finite: "torch.Tensor"
    distance: "torch.Tensor"
    heading_error: "torch.Tensor"
    hand_distance: "torch.Tensor"
    planar_speed: "torch.Tensor"


def observation_dimension(
    *, command_dim: int, robot_count: int, action_dim: int
) -> int:
    """返回 v1 flatten observation 的稳定列数。"""

    return (
        command_dim * 3
        + robot_count * 3
        + robot_count * 4
        + 3
        + 4
        + 6
        + 3
        + 2
        + action_dim
        + 1
    )


def build_tblock_observation(
    state: TBlockState,
    *,
    goal_position: "torch.Tensor",
    goal_yaw: "torch.Tensor",
    heading_axis: "torch.Tensor",
    nominal_heading: "torch.Tensor",
    previous_action: "torch.Tensor",
    episode_length: "torch.Tensor",
    horizon: int,
) -> ObservationMetrics:
    """构造 observation，并返回 reward/termination 需要的几何量。"""

    import torch

    tensors = _validate_state(state)
    goal = require_cuda_tensor(goal_position, name="goal position", ndim=2)
    yaw_goal = require_cuda_tensor(goal_yaw, name="goal yaw", ndim=1)
    axis = require_cuda_tensor(heading_axis, name="heading axis", ndim=1)
    reference_heading = require_cuda_tensor(
        nominal_heading, name="nominal heading", ndim=0
    )
    prev_action = require_cuda_tensor(previous_action, name="previous action", ndim=2)
    length = require_cuda_tensor(
        episode_length, name="episode length", ndim=1, dtype=torch.int64
    )
    require_common_cuda_device(
        (*tensors, goal, yaw_goal, axis, reference_heading, prev_action, length),
        label="observation inputs",
    )
    num_envs = state.joint_positions.shape[0]
    if goal.shape != (num_envs, 3) or yaw_goal.shape != (num_envs,):
        raise ValueError("goal tensors have the wrong shape")
    if state.tcp_positions_local.ndim != 3 or state.tcp_positions_local.shape[2:] != (
        3,
    ):
        raise ValueError("tcp positions must have shape (N,R,3)")
    if state.tcp_orientations_wxyz.shape != (*state.tcp_positions_local.shape[:2], 4):
        raise ValueError("tcp orientations must have shape (N,R,4)")

    heading, heading_finite = tblock_heading(
        state.block_orientation_wxyz,
        heading_axis=axis,
        nominal_heading=reference_heading,
    )
    heading_error = wrap_to_pi(heading - yaw_goal).abs()
    goal_delta = goal - state.block_position_local
    distance = torch.linalg.vector_norm(goal_delta[:, :2], dim=1)
    hand_distance = torch.linalg.vector_norm(
        state.tcp_positions_local - state.block_position_local[:, None, :], dim=2
    ).mean(dim=1)
    planar_speed = torch.linalg.vector_norm(state.block_com_velocity[:, :2], dim=1)
    finite = heading_finite
    for tensor in (*tensors, goal, yaw_goal, prev_action):
        if tensor.is_floating_point():
            finite = finite & torch.all(
                torch.isfinite(tensor).reshape(num_envs, prod(tensor.shape[1:])),
                dim=1,
            )

    target_error = state.position_references - state.joint_positions
    yaw_signed = wrap_to_pi(heading - yaw_goal)
    normalized_time = length.to(dtype=state.joint_positions.dtype) / float(horizon)
    observation = torch.cat(
        (
            state.joint_positions,
            state.joint_velocities,
            target_error,
            state.tcp_positions_local.flatten(1),
            state.tcp_orientations_wxyz.flatten(1),
            state.block_position_local,
            state.block_orientation_wxyz,
            state.block_com_velocity,
            goal_delta,
            torch.stack((torch.sin(yaw_signed), torch.cos(yaw_signed)), dim=1),
            prev_action,
            normalized_time[:, None],
        ),
        dim=1,
    )
    return ObservationMetrics(
        observations=observation,
        finite=finite,
        distance=distance,
        heading_error=heading_error,
        hand_distance=hand_distance,
        planar_speed=planar_speed,
    )


def tblock_heading(
    quaternion_wxyz: "torch.Tensor",
    *,
    heading_axis: "torch.Tensor",
    nominal_heading: "torch.Tensor | None" = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """投影配置的本体轴，返回绝对或相对 nominal 的平面 heading。

    ``nominal_heading`` 为空时用于构造期计算 reference；任务热路径始终传入该 reference，
    因此 reset goal、observation、reward 和 termination 共用同一个相对角度坐标系。
    """

    import torch

    q = require_cuda_tensor(quaternion_wxyz, name="block quaternion", ndim=2)
    if q.shape[1:] != (4,):
        raise ValueError("block quaternion must have shape (N,4)")
    axis = require_cuda_tensor(heading_axis, name="heading axis", ndim=1)
    if axis.shape != (3,):
        raise ValueError("heading axis must have shape (3,)")
    inputs = [q, axis]
    reference = None
    if nominal_heading is not None:
        reference = require_cuda_tensor(nominal_heading, name="nominal heading", ndim=0)
        inputs.append(reference)
    require_common_cuda_device(inputs, label="T-block heading inputs")

    norm = torch.linalg.vector_norm(q, dim=1)
    axis_norm = torch.linalg.vector_norm(axis)
    axis_finite = torch.all(torch.isfinite(axis)) & (axis_norm > 1.0e-6)
    finite = torch.all(torch.isfinite(q), dim=1) & (norm > 1.0e-6) & axis_finite
    safe_norm = torch.where(norm > 1.0e-6, norm, torch.ones_like(norm))
    normalized = q / safe_norm[:, None]
    safe_axis_norm = torch.where(axis_norm > 1.0e-6, axis_norm, axis_norm.new_ones(()))
    body_axis = axis / safe_axis_norm
    rotated_axis = quaternion_rotate_wxyz(
        normalized,
        body_axis[None, :].expand(normalized.shape[0], -1),
    )
    projection_norm = torch.linalg.vector_norm(rotated_axis[:, :2], dim=1)
    finite = finite & (projection_norm > 1.0e-6)
    heading = torch.atan2(rotated_axis[:, 1], rotated_axis[:, 0])
    if reference is not None:
        finite = finite & torch.isfinite(reference)
        heading = wrap_to_pi(heading - reference)
    return heading, finite


def _validate_state(state: TBlockState) -> tuple["torch.Tensor", ...]:
    import torch

    names = (
        "joint_positions",
        "joint_velocities",
        "position_references",
        "tcp_positions_local",
        "tcp_orientations_wxyz",
        "block_position_local",
        "block_orientation_wxyz",
        "block_com_velocity",
        "external_safety_stop",
    )
    tensors = tuple(
        require_cuda_tensor(getattr(state, name), name=name) for name in names
    )
    num_envs = tensors[0].shape[0]
    if any(tensor.ndim == 0 or tensor.shape[0] != num_envs for tensor in tensors):
        raise ValueError("all TBlockState fields must share leading num_envs")
    if (
        state.external_safety_stop.dtype != torch.bool
        or state.external_safety_stop.ndim != 1
    ):
        raise TypeError("external_safety_stop must be CUDA bool (N,)")
    return tensors


__all__ = [
    "ObservationMetrics",
    "TBlockState",
    "build_tblock_observation",
    "observation_dimension",
    "tblock_heading",
]
