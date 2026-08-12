"""双机器人 T-block 推动任务 v1 的完整 CUDA 状态机。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from linkerbot_sim.kaleidoscope.observations import (
    TBlockState,
    build_tblock_observation,
    observation_dimension,
    tblock_heading,
)
from linkerbot_sim.kaleidoscope.resets import (
    DeviceCounterRNG,
    TBlockResetCommand,
    build_tblock_reset_command,
)
from linkerbot_sim.kaleidoscope.rewards import tblock_reward
from linkerbot_sim.kaleidoscope.task import TaskStepResult
from linkerbot_sim.kaleidoscope.task_buffers import TaskBuffers
from linkerbot_sim.kaleidoscope.tensors import (
    normalize_env_ids,
    require_common_cuda_device,
    require_cuda_tensor,
)
from linkerbot_sim.kaleidoscope.terminations import evaluate_tblock_termination

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class TBlockPushV1Settings:
    """任务数学合同中允许配置的构造期参数。"""

    horizon: int = 500
    physics_ticks_per_action: int = 2
    base_seed: int = 0
    ik_failure_penalty: float = -1.0
    distance_progress_weight: float = 8.0
    heading_progress_weight: float = 0.5
    hand_progress_weight: float = 0.25
    action_l2_weight: float = -0.002
    action_rate_l2_weight: float = -0.010
    success_reward: float = 10.0
    task_failure_reward: float = -5.0
    success_distance_m: float = 0.02
    success_heading_rad: float = 0.10
    success_planar_speed_m_s: float = 0.03
    success_streak: int = 5
    failure_aabb_min: tuple[float, float, float] = (-0.05, -0.20, -0.48)
    failure_aabb_max: tuple[float, float, float] = (0.35, 0.20, -0.28)
    robot_joint_delta_rad: tuple[float, float] = (-0.03, 0.03)
    block_x_delta_m: tuple[float, float] = (-0.02, 0.02)
    block_y_delta_m: tuple[float, float] = (-0.02, 0.02)
    block_yaw_delta_rad: tuple[float, float] = (-0.15, 0.15)
    goal_x_delta_m: tuple[float, float] = (0.06, 0.14)
    goal_y_delta_m: tuple[float, float] = (-0.06, 0.06)
    goal_yaw_delta_rad: tuple[float, float] = (-0.25, 0.25)
    heading_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive int")
        if (
            type(self.physics_ticks_per_action) is not int
            or self.physics_ticks_per_action < 1
        ):
            raise ValueError("physics_ticks_per_action must be a positive int")
        if self.success_streak < 1:
            raise ValueError("success_streak must be positive")
        if any(
            low >= high
            for low, high in zip(
                self.failure_aabb_min, self.failure_aabb_max, strict=True
            )
        ):
            raise ValueError("failure AABB minimum must be below maximum")
        if (
            min(
                self.success_distance_m,
                self.success_heading_rad,
                self.success_planar_speed_m_s,
            )
            <= 0
        ):
            raise ValueError("success thresholds must be positive")
        if (
            len(self.heading_axis) != 3
            or not all(math.isfinite(component) for component in self.heading_axis)
            or abs(sum(component * component for component in self.heading_axis) - 1.0)
            > 1.0e-6
        ):
            raise ValueError("heading_axis must be a finite unit 3-vector")
        for name in (
            "failure_aabb_min",
            "failure_aabb_max",
            "robot_joint_delta_rad",
            "block_x_delta_m",
            "block_y_delta_m",
            "block_yaw_delta_rad",
            "goal_x_delta_m",
            "goal_y_delta_m",
            "goal_yaw_delta_rad",
        ):
            values = getattr(self, name)
            if name.startswith("failure_"):
                continue
            if values[0] > values[1]:
                raise ValueError(f"{name} minimum must not exceed maximum")

    @classmethod
    def from_configuration(
        cls, task: object, *, base_seed: int = 0
    ) -> "TBlockPushV1Settings":
        """从 pure ``KaleidoscopeTaskSettings`` 复制全部数值事实。"""

        action = getattr(task, "action")
        reward = getattr(task, "reward")
        termination = getattr(task, "termination")
        randomization = getattr(task, "randomization")
        return cls(
            horizon=termination.horizon_decisions,
            physics_ticks_per_action=action.physics_ticks_per_action,
            base_seed=base_seed,
            ik_failure_penalty=reward.motion_failure,
            distance_progress_weight=reward.distance_progress,
            heading_progress_weight=reward.heading_progress,
            hand_progress_weight=reward.hand_proximity_progress,
            action_l2_weight=reward.action_l2,
            action_rate_l2_weight=reward.action_rate_l2,
            success_reward=reward.success,
            task_failure_reward=reward.task_failure,
            success_distance_m=termination.success_distance_m,
            success_heading_rad=termination.success_heading_rad,
            success_planar_speed_m_s=termination.success_planar_speed_m_s,
            success_streak=termination.success_streak,
            failure_aabb_min=termination.failure_aabb_min,
            failure_aabb_max=termination.failure_aabb_max,
            robot_joint_delta_rad=randomization.robot_joint_delta_rad,
            block_x_delta_m=randomization.block_x_delta_m,
            block_y_delta_m=randomization.block_y_delta_m,
            block_yaw_delta_rad=randomization.block_yaw_delta_rad,
            goal_x_delta_m=randomization.goal_x_delta_m,
            goal_y_delta_m=randomization.goal_y_delta_m,
            goal_yaw_delta_rad=randomization.goal_yaw_delta_rad,
            heading_axis=task.heading_axis,
        )


class TBlockPushV1:
    """不持有 Isaac handle 的设备原生 VectorTask。

    Runtime 负责动作写入、推进设备物理后端和刷新 state；本类只管理 reward/done/reset 随机化及所有
    episode 历史。任务 buffer 可直接注册到 ``KaleidoscopeStateAPI``，因此 snapshot/clone 会包含
    success streak、previous metrics、command history 和 logical RNG 状态。
    """

    def __init__(
        self,
        *,
        num_envs: int,
        command_dim: int,
        action_dim: int,
        robot_count: int,
        device: "torch.device",
        dtype: "torch.dtype",
        nominal_joint_positions: "torch.Tensor",
        nominal_block_position: "torch.Tensor",
        nominal_block_orientation_wxyz: "torch.Tensor",
        settings: TBlockPushV1Settings | None = None,
    ) -> None:
        self.settings = settings or TBlockPushV1Settings()
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.command_dim = int(command_dim)
        self.robot_count = int(robot_count)
        if min(self.num_envs, self.action_dim, self.command_dim, self.robot_count) < 1:
            raise ValueError("task dimensions must be positive")
        self.device = device
        self.dtype = dtype
        self.observation_dim = observation_dimension(
            command_dim=self.command_dim,
            robot_count=self.robot_count,
            action_dim=self.action_dim,
        )
        self._nominal_joint_positions = require_cuda_tensor(
            nominal_joint_positions,
            name="nominal joint positions",
            ndim=1,
            dtype=dtype,
        ).clone()
        self._nominal_block_position = require_cuda_tensor(
            nominal_block_position,
            name="nominal block position",
            ndim=1,
            dtype=dtype,
        ).clone()
        self._nominal_block_orientation = require_cuda_tensor(
            nominal_block_orientation_wxyz,
            name="nominal block orientation",
            ndim=1,
            dtype=dtype,
        ).clone()
        require_common_cuda_device(
            (
                self._nominal_joint_positions,
                self._nominal_block_position,
                self._nominal_block_orientation,
            ),
            label="task nominal tensors",
        )
        if self._nominal_joint_positions.shape != (self.command_dim,):
            raise ValueError("nominal joint position width does not match command_dim")
        import torch

        self._heading_axis = torch.tensor(
            self.settings.heading_axis,
            device=device,
            dtype=dtype,
        )
        nominal_heading, nominal_heading_finite = tblock_heading(
            self._nominal_block_orientation[None, :],
            heading_axis=self._heading_axis,
        )
        torch._assert_async(
            torch.all(nominal_heading_finite),
            "nominal T-block orientation has no finite planar heading",
        )
        self._nominal_heading = nominal_heading[0].clone()
        self.buffers = TaskBuffers.allocate(
            num_envs=self.num_envs,
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            device=device,
            dtype=dtype,
            base_seed=self.settings.base_seed,
        )
        self._rng = DeviceCounterRNG(self.buffers.rng_key, self.buffers.rng_counter)
        self._all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.int64
        )

    def reset_command(self, env_ids: "torch.Tensor") -> TBlockResetCommand:
        """生成 K 行 reset command，并登记对应 goal。"""

        ids = normalize_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self.device,
            allow_empty=True,
        )
        command = build_tblock_reset_command(
            env_ids=ids,
            rng=self._rng,
            nominal_joint_positions=self._nominal_joint_positions,
            nominal_block_position=self._nominal_block_position,
            nominal_block_orientation_wxyz=self._nominal_block_orientation,
            heading_axis=self._heading_axis,
            nominal_heading=self._nominal_heading,
            joint_delta_range=self.settings.robot_joint_delta_rad,
            block_x_range=self.settings.block_x_delta_m,
            block_y_range=self.settings.block_y_delta_m,
            block_yaw_range=self.settings.block_yaw_delta_rad,
            goal_x_range=self.settings.goal_x_delta_m,
            goal_y_range=self.settings.goal_y_delta_m,
            goal_yaw_range=self.settings.goal_yaw_delta_rad,
        )
        self.buffers.goal_position.index_copy_(0, ids, command.goal_position)
        self.buffers.goal_yaw.index_copy_(0, ids, command.goal_yaw)
        return command

    def masked_reset_command(
        self, reset_mask: "torch.Tensor", state: TBlockState
    ) -> TBlockResetCommand:
        """为 SAME_STEP 构造固定 N 行命令，非 reset 行逐字段保持原值。

        这里不能先用 ``nonzero`` 提取 done ids：变长 CUDA 输出会同步主机。全 N candidate
        只让 mask 行推进 RNG，再用设备端 ``where`` 与 canonical state 混合。
        """

        import torch

        mask = require_cuda_tensor(
            reset_mask,
            name="SAME_STEP reset mask",
            ndim=1,
            leading_dim=self.num_envs,
            dtype=torch.bool,
        )
        if mask.device != self.device:
            raise ValueError("SAME_STEP reset mask must live on task.device")
        if state.joint_positions.shape[0] != self.num_envs:
            raise ValueError("masked reset state must contain all environments")
        ids = self._all_env_ids
        candidate = build_tblock_reset_command(
            env_ids=ids,
            rng=self._rng,
            nominal_joint_positions=self._nominal_joint_positions,
            nominal_block_position=self._nominal_block_position,
            nominal_block_orientation_wxyz=self._nominal_block_orientation,
            heading_axis=self._heading_axis,
            nominal_heading=self._nominal_heading,
            joint_delta_range=self.settings.robot_joint_delta_rad,
            block_x_range=self.settings.block_x_delta_m,
            block_y_range=self.settings.block_y_delta_m,
            block_yaw_range=self.settings.block_yaw_delta_rad,
            goal_x_range=self.settings.goal_x_delta_m,
            goal_y_range=self.settings.goal_y_delta_m,
            goal_yaw_range=self.settings.goal_yaw_delta_rad,
            rng_advance_mask=mask,
        )

        def blend(fresh: "torch.Tensor", current: "torch.Tensor") -> "torch.Tensor":
            selector = mask.reshape((self.num_envs,) + (1,) * (fresh.ndim - 1))
            return torch.where(selector, fresh, current)

        command = TBlockResetCommand(
            env_ids=ids,
            joint_positions=blend(candidate.joint_positions, state.joint_positions),
            joint_velocities=blend(candidate.joint_velocities, state.joint_velocities),
            joint_targets=blend(candidate.joint_targets, state.command_targets),
            block_position=blend(candidate.block_position, state.block_position_local),
            block_orientation_wxyz=blend(
                candidate.block_orientation_wxyz,
                state.block_orientation_wxyz,
            ),
            block_velocity=blend(candidate.block_velocity, state.block_com_velocity),
            goal_position=blend(candidate.goal_position, self.buffers.goal_position),
            goal_yaw=blend(candidate.goal_yaw, self.buffers.goal_yaw),
            device_reset_mask=mask,
        )
        self.buffers.goal_position.copy_(command.goal_position)
        self.buffers.goal_yaw.copy_(command.goal_yaw)
        return command

    def initialize_after_reset(
        self, env_ids: "torch.Tensor", state: TBlockState
    ) -> None:
        """物理 reset 写入并 refresh 后，原子初始化全部 task history。"""

        import torch

        ids = normalize_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self.device,
            allow_empty=True,
        )
        if state.joint_positions.shape[0] != ids.numel():
            raise ValueError("reset state must contain exactly the selected rows")
        zeros_action = torch.zeros(
            (ids.numel(), self.action_dim), device=self.device, dtype=self.dtype
        )
        zeros_length = torch.zeros(ids.numel(), device=self.device, dtype=torch.int64)
        metrics = build_tblock_observation(
            state,
            goal_position=self.buffers.goal_position.index_select(0, ids),
            goal_yaw=self.buffers.goal_yaw.index_select(0, ids),
            heading_axis=self._heading_axis,
            nominal_heading=self._nominal_heading,
            previous_action=zeros_action,
            episode_length=zeros_length,
            horizon=self.settings.horizon,
        )
        torch._assert_async(
            torch.all(metrics.finite),
            "reset state cannot produce a finite T-block observation",
        )
        zero_float = torch.zeros(ids.numel(), device=self.device, dtype=self.dtype)
        zero_int = torch.zeros(ids.numel(), device=self.device, dtype=torch.int64)
        zero_bool = torch.zeros(ids.numel(), device=self.device, dtype=torch.bool)
        for target, value in (
            (self.buffers.episode_length, zero_int),
            (self.buffers.episode_physics_steps, zero_int),
            (self.buffers.episode_return, zero_float),
            (self.buffers.previous_distance, metrics.distance),
            (self.buffers.previous_heading_error, metrics.heading_error),
            (self.buffers.previous_hand_distance, metrics.hand_distance),
            (self.buffers.success_streak, zero_int),
            (self.buffers.reward, zero_float),
            (self.buffers.terminated, zero_bool),
            (self.buffers.truncated, zero_bool),
            (self.buffers.needs_reset, zero_bool),
            (self.buffers.numeric_failure, zero_bool),
        ):
            target.index_copy_(0, ids, value)
        self.buffers.previous_action.index_copy_(0, ids, zeros_action)
        self.buffers.last_finite_observation.index_copy_(0, ids, metrics.observations)

    def initialize_after_masked_reset(
        self, reset_mask: "torch.Tensor", state: TBlockState
    ) -> None:
        """只初始化 mask 行，所有 task/episode buffer 在非 mask 行逐位保留。"""

        import torch

        mask = require_cuda_tensor(
            reset_mask,
            name="SAME_STEP reset mask",
            ndim=1,
            leading_dim=self.num_envs,
            dtype=torch.bool,
        )
        if mask.device != self.device:
            raise ValueError("SAME_STEP reset mask must live on task.device")
        if state.joint_positions.shape[0] != self.num_envs:
            raise ValueError("masked reset state must contain all environments")
        zeros_action = torch.zeros_like(self.buffers.previous_action)
        zeros_length = torch.zeros_like(self.buffers.episode_length)
        metrics = build_tblock_observation(
            state,
            goal_position=self.buffers.goal_position,
            goal_yaw=self.buffers.goal_yaw,
            heading_axis=self._heading_axis,
            nominal_heading=self._nominal_heading,
            previous_action=zeros_action,
            episode_length=zeros_length,
            horizon=self.settings.horizon,
        )
        torch._assert_async(
            torch.all(metrics.finite | ~mask),
            "reset rows cannot produce a finite T-block observation",
        )

        def masked_copy(target: "torch.Tensor", value: "torch.Tensor") -> None:
            selector = mask.reshape((self.num_envs,) + (1,) * (target.ndim - 1))
            target.copy_(torch.where(selector, value, target))

        for target, value in (
            (self.buffers.episode_length, zeros_length),
            (self.buffers.episode_physics_steps, zeros_length),
            (
                self.buffers.episode_return,
                torch.zeros_like(self.buffers.episode_return),
            ),
            (self.buffers.previous_distance, metrics.distance),
            (self.buffers.previous_heading_error, metrics.heading_error),
            (self.buffers.previous_hand_distance, metrics.hand_distance),
            (self.buffers.success_streak, zeros_length),
            (self.buffers.reward, torch.zeros_like(self.buffers.reward)),
            (self.buffers.terminated, torch.zeros_like(self.buffers.terminated)),
            (self.buffers.truncated, torch.zeros_like(self.buffers.truncated)),
            (self.buffers.needs_reset, torch.zeros_like(self.buffers.needs_reset)),
            (
                self.buffers.numeric_failure,
                torch.zeros_like(self.buffers.numeric_failure),
            ),
        ):
            masked_copy(target, value)
        masked_copy(self.buffers.previous_action, zeros_action)
        masked_copy(self.buffers.last_finite_observation, metrics.observations)

    def step(self, state: TBlockState, actions: "torch.Tensor") -> TaskStepResult:
        """消费 physics 后状态，严格按 v1 顺序更新 reward、done 和历史。"""

        import torch

        action = require_cuda_tensor(
            actions,
            name="task actions",
            ndim=2,
            leading_dim=self.num_envs,
            dtype=self.dtype,
        )
        if action.shape[1] != self.action_dim or action.device != self.device:
            raise ValueError("task actions have the wrong shape/device")
        previous_action = self.buffers.previous_action
        metrics = build_tblock_observation(
            state,
            goal_position=self.buffers.goal_position,
            goal_yaw=self.buffers.goal_yaw,
            heading_axis=self._heading_axis,
            nominal_heading=self._nominal_heading,
            # step 返回的是执行当前 action 后的 state，因此 observation 中的
            # previous_action 就是刚执行的 action，与本拍结束后的持久 buffer 一致。
            previous_action=action,
            episode_length=self.buffers.episode_length + 1,
            horizon=self.settings.horizon,
        )
        numeric_failure = ~metrics.finite
        # 非有限行沿用上一有限 metric，防止 NaN 污染 task history。
        distance = torch.where(
            numeric_failure, self.buffers.previous_distance, metrics.distance
        )
        heading_error = torch.where(
            numeric_failure,
            self.buffers.previous_heading_error,
            metrics.heading_error,
        )
        hand_distance = torch.where(
            numeric_failure,
            self.buffers.previous_hand_distance,
            metrics.hand_distance,
        )
        next_length = self.buffers.episode_length + 1
        termination = evaluate_tblock_termination(
            block_position=state.block_position_local,
            distance=distance,
            heading_error=heading_error,
            planar_speed=metrics.planar_speed,
            success_streak=self.buffers.success_streak,
            next_episode_length=next_length,
            numeric_failure=numeric_failure,
            external_safety_stop=state.external_safety_stop,
            horizon=self.settings.horizon,
            success_distance_m=self.settings.success_distance_m,
            success_heading_rad=self.settings.success_heading_rad,
            success_planar_speed_m_s=self.settings.success_planar_speed_m_s,
            required_success_streak=self.settings.success_streak,
            failure_aabb_min=self.settings.failure_aabb_min,
            failure_aabb_max=self.settings.failure_aabb_max,
        )
        reward = tblock_reward(
            distance=distance,
            previous_distance=self.buffers.previous_distance,
            heading_error=heading_error,
            previous_heading_error=self.buffers.previous_heading_error,
            hand_distance=hand_distance,
            previous_hand_distance=self.buffers.previous_hand_distance,
            action=action,
            previous_action=previous_action,
            success=termination.success,
            task_failure=termination.task_failure,
            numeric_failure=numeric_failure,
            distance_progress_weight=self.settings.distance_progress_weight,
            heading_progress_weight=self.settings.heading_progress_weight,
            hand_progress_weight=self.settings.hand_progress_weight,
            action_l2_weight=self.settings.action_l2_weight,
            action_rate_l2_weight=self.settings.action_rate_l2_weight,
            success_reward=self.settings.success_reward,
            task_failure_reward=self.settings.task_failure_reward,
        )
        observation = torch.where(
            numeric_failure[:, None],
            self.buffers.last_finite_observation,
            metrics.observations,
        )

        self.buffers.episode_length.copy_(next_length)
        self.buffers.episode_physics_steps.add_(self.settings.physics_ticks_per_action)
        self.buffers.episode_return.add_(reward)
        self.buffers.success_streak.copy_(termination.next_success_streak)
        self.buffers.previous_distance.copy_(distance)
        self.buffers.previous_heading_error.copy_(heading_error)
        self.buffers.previous_hand_distance.copy_(hand_distance)
        self.buffers.previous_action.copy_(
            torch.where(numeric_failure[:, None], previous_action, action)
        )
        self.buffers.reward.copy_(reward)
        self.buffers.terminated.copy_(termination.terminated)
        self.buffers.truncated.copy_(termination.truncated)
        self.buffers.needs_reset.copy_(termination.terminated | termination.truncated)
        self.buffers.numeric_failure.copy_(numeric_failure)
        self.buffers.last_finite_observation.copy_(observation)
        return TaskStepResult(
            observations=observation,
            rewards=reward,
            terminated=termination.terminated,
            truncated=termination.truncated,
            info={
                "success": termination.success,
                "task_failure": termination.task_failure,
                "numeric_failure": numeric_failure,
                "episode_return": self.buffers.episode_return,
                "episode_length": self.buffers.episode_length,
            },
        )

    def state_fields(self) -> dict[str, "torch.Tensor"]:
        return self.buffers.state_fields()

    def reseed(self, seed: int) -> None:
        """在显式 reset 冷边界重建 per-env logical RNG key。"""

        import torch

        if type(seed) is not int:
            raise TypeError("task seed must be an int")
        keys = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
        keys.mul_(6364136223846793005).add_(seed)
        self.buffers.rng_key.copy_(keys)
        self.buffers.rng_counter.zero_()

    def close(self) -> None:
        """Task 没有外部资源；方法用于统一 Runtime 关闭协议。"""


__all__ = ["TBlockPushV1", "TBlockPushV1Settings"]
