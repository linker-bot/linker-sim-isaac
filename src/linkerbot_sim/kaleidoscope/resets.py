"""设备端 counter RNG 与 T-block reset 随机化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from linkerbot_sim.kaleidoscope.geometry import (
    normalize_quaternion_wxyz,
    quaternion_multiply_wxyz,
    wrap_to_pi,
)
from linkerbot_sim.kaleidoscope.observations import tblock_heading
from linkerbot_sim.kaleidoscope.tensors import require_cuda_tensor

if TYPE_CHECKING:
    import torch


class DeviceCounterRNG:
    """按 env key/counter 生成确定性 CUDA 随机数。

    这里不使用一个全局 ``torch.Generator``，因为全局 generator state 无法按 env 克隆。每个环境
    拥有独立 key/counter，snapshot/clone 后的后续随机序列因而可精确复现。
    """

    def __init__(self, key: "torch.Tensor", counter: "torch.Tensor") -> None:
        import torch

        self.key = require_cuda_tensor(key, name="rng key", ndim=1, dtype=torch.int64)
        self.counter = require_cuda_tensor(
            counter, name="rng counter", ndim=1, dtype=torch.int64
        )
        if (
            self.key.shape != self.counter.shape
            or self.key.device != self.counter.device
        ):
            raise ValueError("rng key/counter must share shape and CUDA device")

    def uniform(
        self,
        env_ids: "torch.Tensor",
        *,
        columns: int,
        advance_mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """返回固定形状随机数，并只推进 mask 选中的 logical counter。"""

        import torch

        ids = require_cuda_tensor(
            env_ids, name="rng env ids", ndim=1, dtype=torch.int64
        )
        if ids.device != self.key.device:
            raise ValueError("rng selector must share RNG device")
        if type(columns) is not int or columns < 1:
            raise ValueError("rng columns must be a positive int")
        increments = torch.full_like(ids, columns, dtype=torch.int64)
        if advance_mask is not None:
            mask = require_cuda_tensor(
                advance_mask,
                name="rng advance mask",
                ndim=1,
                leading_dim=ids.numel(),
                dtype=torch.bool,
            )
            if mask.device != ids.device:
                raise ValueError("rng advance mask must share selector device")
            increments.mul_(mask.to(dtype=torch.int64))
        key = self.key.index_select(0, ids)[:, None]
        counter = self.counter.index_select(0, ids)[:, None]
        lane = torch.arange(columns, device=ids.device, dtype=torch.int64)[None, :]
        # 31-bit LCG 保证各 CUDA 后端都支持一致的整数位运算；lane 使用不同 Weyl increment。
        state = key + (counter + lane * 747796405) * 2891336453
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        self.counter.index_add_(
            0,
            ids,
            increments,
        )
        return state.to(dtype=torch.float32) * (1.0 / 2147483648.0)


@dataclass(frozen=True, slots=True)
class TBlockResetCommand:
    """Runtime 写入设备物理后端前使用的完整 K 行 reset command。

    普通 ``reset_idx`` 的 K 行全部有效，``device_reset_mask`` 为 ``None``。SAME_STEP
    为避免 ``nonzero`` 主机同步而提交固定 N 行，只有 device mask 为 True 的行属于新
    episode；后端必须用该 mask 清理 per-world solver persistent state。
    """

    env_ids: "torch.Tensor"
    joint_positions: "torch.Tensor"
    joint_velocities: "torch.Tensor"
    joint_targets: "torch.Tensor"
    block_position: "torch.Tensor"
    block_orientation_wxyz: "torch.Tensor"
    block_velocity: "torch.Tensor"
    goal_position: "torch.Tensor"
    goal_yaw: "torch.Tensor"
    # SAME_STEP 保持固定 N 行以避免 ``nonzero`` 主机同步；该 mask 由支持设备选择的后端
    # 消费。普通显式 reset 使用 ``None``，其 env_ids 本身就是精确选择。
    device_reset_mask: "torch.Tensor | None" = None


def build_tblock_reset_command(
    *,
    env_ids: "torch.Tensor",
    rng: DeviceCounterRNG,
    nominal_joint_positions: "torch.Tensor",
    nominal_block_position: "torch.Tensor",
    nominal_block_orientation_wxyz: "torch.Tensor",
    heading_axis: "torch.Tensor",
    nominal_heading: "torch.Tensor",
    joint_delta_range: tuple[float, float] = (-0.03, 0.03),
    block_x_range: tuple[float, float] = (-0.02, 0.02),
    block_y_range: tuple[float, float] = (-0.02, 0.02),
    block_yaw_range: tuple[float, float] = (-0.15, 0.15),
    goal_x_range: tuple[float, float] = (0.06, 0.14),
    goal_y_range: tuple[float, float] = (-0.06, 0.06),
    goal_yaw_range: tuple[float, float] = (-0.25, 0.25),
    rng_advance_mask: "torch.Tensor | None" = None,
) -> TBlockResetCommand:
    """实现 v1 冻结的 joint/block/goal 随机化范围。"""

    import torch

    ids = require_cuda_tensor(env_ids, name="reset env ids", ndim=1, dtype=torch.int64)
    nominal_q = require_cuda_tensor(
        nominal_joint_positions, name="nominal joint positions", ndim=1
    )
    nominal_p = require_cuda_tensor(
        nominal_block_position, name="nominal block position", ndim=1
    )
    nominal_quat = require_cuda_tensor(
        nominal_block_orientation_wxyz,
        name="nominal block orientation",
        ndim=1,
    )
    axis = require_cuda_tensor(heading_axis, name="heading axis", ndim=1)
    reference_heading = require_cuda_tensor(
        nominal_heading, name="nominal heading", ndim=0
    )
    if nominal_p.shape != (3,) or nominal_quat.shape != (4,):
        raise ValueError("nominal block pose must have shapes (3,) and (4,)")
    if axis.shape != (3,):
        raise ValueError("heading axis must have shape (3,)")
    if not (
        ids.device
        == nominal_q.device
        == nominal_p.device
        == nominal_quat.device
        == axis.device
        == reference_heading.device
    ):
        raise ValueError("reset inputs must share one CUDA device")
    count = ids.numel()
    random = rng.uniform(
        ids,
        columns=nominal_q.numel() + 5,
        advance_mask=rng_advance_mask,
    )
    q_noise = _uniform_range(random[:, : nominal_q.numel()], joint_delta_range)
    joint_positions = nominal_q[None, :] + q_noise.to(dtype=nominal_q.dtype)
    joint_velocities = torch.zeros_like(joint_positions)

    offset = nominal_q.numel()
    block_position = nominal_p[None, :].expand(count, -1).clone()
    block_position[:, 0].add_(_uniform_range(random[:, offset], block_x_range))
    block_position[:, 1].add_(_uniform_range(random[:, offset + 1], block_y_range))
    reset_yaw = _uniform_range(random[:, offset + 2], block_yaw_range)
    qz = torch.stack(
        (
            torch.cos(0.5 * reset_yaw),
            torch.zeros_like(reset_yaw),
            torch.zeros_like(reset_yaw),
            torch.sin(0.5 * reset_yaw),
        ),
        dim=1,
    ).to(dtype=nominal_quat.dtype)
    nominal_batch = nominal_quat[None, :].expand(count, -1)
    block_orientation = quaternion_multiply_wxyz(qz, nominal_batch)
    block_orientation = normalize_quaternion_wxyz(block_orientation)
    block_velocity = torch.zeros((count, 6), device=ids.device, dtype=nominal_p.dtype)

    goal_position = block_position.clone()
    goal_position[:, 0].add_(_uniform_range(random[:, offset + 3], goal_x_range))
    goal_position[:, 1].add_(_uniform_range(random[:, offset + 4], goal_y_range))
    goal_delta_yaw = _uniform_range(
        rng.uniform(ids, columns=1, advance_mask=rng_advance_mask)[:, 0],
        goal_yaw_range,
    )
    reset_heading, heading_finite = tblock_heading(
        block_orientation,
        heading_axis=axis,
        nominal_heading=reference_heading,
    )
    torch._assert_async(
        torch.all(heading_finite),
        "reset orientation cannot produce a finite T-block heading",
    )
    goal_yaw = wrap_to_pi(reset_heading + goal_delta_yaw)
    return TBlockResetCommand(
        env_ids=ids,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        joint_targets=joint_positions.clone(),
        block_position=block_position,
        block_orientation_wxyz=block_orientation,
        block_velocity=block_velocity,
        goal_position=goal_position,
        goal_yaw=goal_yaw,
    )


def _uniform_range(unit: "torch.Tensor", bounds: tuple[float, float]) -> "torch.Tensor":
    low, high = (float(bounds[0]), float(bounds[1]))
    if low > high:
        raise ValueError("randomization range minimum must not exceed maximum")
    return low + unit * (high - low)


__all__ = ["DeviceCounterRNG", "TBlockResetCommand", "build_tblock_reset_command"]
