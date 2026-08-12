"""Kaleidoscope 固定形状动作合同与 GPU joint target 累加器。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import TYPE_CHECKING

from linkerbot_sim.controllers.control_mode import ControlModeIncompatibleError
from linkerbot_sim.controllers.types import ControlMode
from linkerbot_sim.kaleidoscope.control_commands import (
    EffortControlTrajectory,
    PositionControlTrajectory,
    VelocityControlTrajectory,
)
from linkerbot_sim.kaleidoscope.geometry import (
    normalize_quaternion_wxyz,
    quaternion_multiply_wxyz,
)
from linkerbot_sim.kaleidoscope.tensors import (
    assert_finite_async,
    normalize_env_ids,
    require_common_cuda_device,
    require_cuda_tensor,
)

if TYPE_CHECKING:
    import torch

    from linkerbot_sim.kaleidoscope.ik import DeviceBatchIKSolver


class ActionMode(StrEnum):
    """每个环境 profile 构造时冻结的动作模式。"""

    JOINT_CONTROL = "joint_control"
    JOINT_DELTA = "joint_delta"
    EE_DELTA_POSITION = "ee_delta_position"
    EE_DELTA_POSE = "ee_delta_pose"
    EE_POSE_POSITION = "ee_pose_position"
    EE_POSE_FULL = "ee_pose_full"
    EE_LINEAR_PATH_POSITION = "ee_linear_path_position"
    EE_LINEAR_PATH_FULL = "ee_linear_path_full"


def action_spec_from_configuration(
    action: object,
    *,
    robot_labels: tuple[str, ...],
    command_dims: tuple[int, ...],
) -> "ActionSpec":
    """把 pure config action union 转成 runtime 固定合同。"""

    mode = ActionMode(str(getattr(action, "mode")))
    failure = str(getattr(action, "failure_policy", "hold_penalty_truncate"))
    return ActionSpec(
        mode=mode,
        robot_labels=robot_labels,
        command_dims=command_dims,
        physics_ticks_per_action=int(action.physics_ticks_per_action),
        scale=float(
            getattr(
                action,
                "position_delta_scale_rad",
                getattr(action, "scale_rad", 1.0),
            )
        ),
        clip=(
            float(action.clip)
            if mode in {ActionMode.JOINT_CONTROL, ActionMode.JOINT_DELTA}
            else None
        ),
        velocity_scale=float(getattr(action, "velocity_scale_rad_s", 1.0)),
        effort_limit_fraction=float(getattr(action, "effort_limit_fraction", 1.0)),
        reference_velocity_limit=float(
            getattr(action, "reference_velocity_limit_rad_s", 1.0)
        ),
        waypoint_count=getattr(action, "waypoint_count", None),
        orientation_mode=(
            "target"
            if mode is ActionMode.EE_LINEAR_PATH_FULL
            else getattr(action, "orientation_mode", None)
        ),
        progress_mode=str(getattr(action, "progress_mode", "smoothstep")),
        failure_policy=(
            "hold" if failure == "hold_from_first_failure" else "hold_and_truncate"
        ),
    )


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """动作空间的构造期事实；不会作为逐 step JSON discriminator。"""

    mode: ActionMode
    robot_labels: tuple[str, ...]
    command_dims: tuple[int, ...]
    physics_ticks_per_action: int
    scale: float = 1.0
    clip: float | None = None
    velocity_scale: float = 1.0
    effort_limit_fraction: float = 1.0
    reference_velocity_limit: float = 1.0
    waypoint_count: int | None = None
    orientation_mode: str | None = None
    progress_mode: str = "smoothstep"
    failure_policy: str = "hold_and_truncate"

    def __post_init__(self) -> None:
        if not self.robot_labels or len(self.robot_labels) != len(self.command_dims):
            raise ValueError("robot_labels and command_dims must be non-empty/equal")
        if len(set(self.robot_labels)) != len(self.robot_labels):
            raise ValueError("robot_labels cannot contain duplicates")
        if any(not label.strip() for label in self.robot_labels):
            raise ValueError("robot labels must be non-empty")
        if any(type(width) is not int or width < 1 for width in self.command_dims):
            raise ValueError("command_dims must contain positive ints")
        if (
            type(self.physics_ticks_per_action) is not int
            or self.physics_ticks_per_action < 1
        ):
            raise ValueError("physics_ticks_per_action must be a positive int")
        if float(self.scale) <= 0.0:
            raise ValueError("action scale must be positive")
        if self.mode in {ActionMode.JOINT_CONTROL, ActionMode.JOINT_DELTA}:
            if (
                self.clip is None
                or not math.isfinite(float(self.clip))
                or float(self.clip) <= 0.0
            ):
                raise ValueError("joint action clip must be positive")
        elif self.clip is not None:
            raise ValueError("action clip is only valid for joint control modes")
        if not math.isfinite(self.velocity_scale) or self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be finite and positive")
        if (
            not math.isfinite(self.effort_limit_fraction)
            or not 0.0 < self.effort_limit_fraction <= 1.0
        ):
            raise ValueError("effort_limit_fraction must be in (0, 1]")
        if (
            not math.isfinite(self.reference_velocity_limit)
            or self.reference_velocity_limit <= 0.0
        ):
            raise ValueError("reference_velocity_limit must be finite and positive")
        if self.progress_mode not in {"linear", "smoothstep"}:
            raise ValueError("progress_mode must be linear or smoothstep")
        if self.failure_policy not in {"hold", "hold_and_truncate", "reject_batch"}:
            raise ValueError("unsupported IK failure policy")
        is_linear = self.mode in {
            ActionMode.EE_LINEAR_PATH_POSITION,
            ActionMode.EE_LINEAR_PATH_FULL,
        }
        if is_linear:
            if type(self.waypoint_count) is not int or self.waypoint_count < 1:
                raise ValueError("linear action requires positive waypoint_count")
            if self.orientation_mode not in {"free", "current", "target"}:
                raise ValueError(
                    "linear action orientation_mode must be free/current/target"
                )
        elif self.waypoint_count is not None:
            raise ValueError("waypoint_count is only valid for linear action modes")

    @property
    def action_dim(self) -> int:
        per_robot = {
            ActionMode.JOINT_CONTROL: None,
            ActionMode.JOINT_DELTA: None,
            ActionMode.EE_DELTA_POSITION: 3,
            ActionMode.EE_DELTA_POSE: 6,
            ActionMode.EE_POSE_POSITION: 3,
            ActionMode.EE_POSE_FULL: 7,
            ActionMode.EE_LINEAR_PATH_POSITION: 3,
            ActionMode.EE_LINEAR_PATH_FULL: 7,
        }[self.mode]
        return (
            sum(self.command_dims)
            if per_robot is None
            else len(self.robot_labels) * per_robot
        )

    def robot_slices(self) -> dict[str, slice]:
        """返回稳定 label 到动作列的映射。"""

        if self.mode in {ActionMode.JOINT_CONTROL, ActionMode.JOINT_DELTA}:
            widths = self.command_dims
        else:
            width = self.action_dim // len(self.robot_labels)
            widths = (width,) * len(self.robot_labels)
        offset = 0
        result: dict[str, slice] = {}
        for label, width in zip(self.robot_labels, widths, strict=True):
            result[label] = slice(offset, offset + width)
            offset += width
        return result


class JointDeltaActionTerm:
    """实现 target-accumulating joint-delta，不把实际 q 当作每拍锚点。"""

    def __init__(
        self,
        *,
        lower: "torch.Tensor",
        upper: "torch.Tensor",
        scale: "torch.Tensor",
        clip: float,
        num_envs: int,
        target: "torch.Tensor",
    ) -> None:
        import torch

        lower = require_cuda_tensor(lower, name="joint lower", ndim=1)
        upper = require_cuda_tensor(upper, name="joint upper", ndim=1)
        scale = require_cuda_tensor(scale, name="joint scale", ndim=1)
        self.device = require_common_cuda_device(
            (lower, upper, scale), label="joint action tensors"
        )
        if not (lower.shape == upper.shape == scale.shape):
            raise ValueError("joint lower/upper/scale shapes must match")
        torch._assert_async(torch.all(lower < upper), "joint lower must be below upper")
        torch._assert_async(torch.all(scale > 0), "joint scale must be positive")
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.command_dim = lower.numel()
        self.clip = float(clip)
        if not math.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("joint action clip must be finite and positive")
        self.lower = lower.clone()
        self.upper = upper.clone()
        self.scale = scale.clone()
        canonical_target = require_cuda_tensor(
            target,
            name="joint target owner",
            ndim=2,
            leading_dim=self.num_envs,
            dtype=lower.dtype,
        )
        if canonical_target.device != self.device or canonical_target.shape[1:] != (
            self.command_dim,
        ):
            raise ValueError(
                "joint target owner must match controller shape and CUDA device"
            )
        # 累加器和 state API 必须引用同一块 canonical storage；否则 clone/restore 只会
        # 更新物理 target，下一拍仍从旧 controller target 累加并立即分叉。
        self.target = canonical_target

    def reset_targets(
        self,
        joint_positions: "torch.Tensor",
        env_ids: "torch.Tensor | None" = None,
    ) -> None:
        """把 reset 行的 command target 锚到 reset command joint。"""

        ids = normalize_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self.device,
            allow_empty=True,
        )
        values = require_cuda_tensor(
            joint_positions,
            name="reset joint positions",
            leading_dim=ids.numel(),
            dtype=self.target.dtype,
        )
        if values.shape[1:] != (self.command_dim,):
            raise ValueError("reset joint positions have the wrong command dimension")
        assert_finite_async(values, name="reset joint positions")
        self.target.index_copy_(0, ids, values)

    def apply(self, actions: "torch.Tensor") -> "torch.Tensor":
        """按 strict config 的 clip 计算并保存累加 joint target。"""

        import torch

        action = require_cuda_tensor(
            actions,
            name="joint delta actions",
            ndim=2,
            leading_dim=self.num_envs,
            dtype=self.target.dtype,
        )
        if action.shape[1] != self.command_dim:
            raise ValueError(
                f"joint delta action width must be {self.command_dim}, got {action.shape[1]}"
            )
        assert_finite_async(action, name="joint delta actions")
        self.target.copy_(
            torch.clamp(
                self.target + torch.clamp(action, -self.clip, self.clip) * self.scale,
                min=self.lower,
                max=self.upper,
            )
        )
        return self.target


class JointDeltaRuntimeAction:
    """把 joint-delta 累加器适配为 Runtime 的固定 tick ``ActionTerm``。"""

    def __init__(
        self,
        controller: JointDeltaActionTerm,
        *,
        physics_ticks_per_action: int,
        physics_dt: float = 1.0,
        reference_velocity_limit_rad_s: float = 1.0,
    ) -> None:
        import torch

        if type(physics_ticks_per_action) is not int or physics_ticks_per_action < 1:
            raise ValueError("physics_ticks_per_action must be a positive int")
        self.controller = controller
        self.physics_ticks_per_action = physics_ticks_per_action
        self.action_dim = controller.command_dim
        self.action_low = -controller.clip
        self.action_high = controller.clip
        self.supported_control_modes: tuple[ControlMode, ...] = (
            "position",
            "velocity",
        )
        self.physics_dt = _positive_finite(physics_dt, label="physics_dt")
        self.reference_velocity_limit_rad_s = _positive_finite(
            reference_velocity_limit_rad_s,
            label="reference_velocity_limit_rad_s",
        )
        self._failure = torch.zeros(
            controller.num_envs, device=controller.device, dtype=torch.bool
        )

    def apply(
        self,
        actions: "torch.Tensor",
        state: object,
        active_mode: ControlMode = "position",
    ):
        from linkerbot_sim.kaleidoscope.runtime import ActionExecution

        targets = self.controller.apply(actions)
        trajectory = targets[None, :, :].expand(self.physics_ticks_per_action, -1, -1)
        control, info = _position_reference_control(
            trajectory,
            state=state,
            active_mode=active_mode,
            physics_dt=self.physics_dt,
            velocity_limit=self.reference_velocity_limit_rad_s,
            failure_mask=self._failure,
            info={},
        )
        return ActionExecution(
            control=control,
            position_reference=targets,
            failure_mask=self._failure,
            info=info,
        )

    def reset(self, env_ids: "torch.Tensor", command: object) -> None:
        self.controller.reset_targets(command.joint_targets, env_ids)

    def close(self) -> None:
        """joint-only term 不拥有外部资源，保留统一幂等生命周期。"""

        return None


class JointControlRuntimeAction:
    """Direct three-mode joint action with one persistent position reference."""

    def __init__(
        self,
        controller: JointDeltaActionTerm,
        *,
        physics_ticks_per_action: int,
        velocity_scale_rad_s: float,
        effort_limits: "torch.Tensor",
        effort_limit_fraction: float,
        physics_dt: float,
    ) -> None:
        import torch

        if type(physics_ticks_per_action) is not int or physics_ticks_per_action < 1:
            raise ValueError("physics_ticks_per_action must be a positive int")
        limits = require_cuda_tensor(
            effort_limits,
            name="joint-control effort limits",
            ndim=1,
            dtype=controller.target.dtype,
        )
        if limits.device != controller.device or limits.shape != (
            controller.command_dim,
        ):
            raise ValueError("joint-control effort limits have the wrong shape/device")
        torch._assert_async(
            torch.all(torch.isfinite(limits) & (limits > 0.0)),
            "joint-control effort limits must be finite and positive",
        )
        self.controller = controller
        self.physics_ticks_per_action = physics_ticks_per_action
        self.physics_dt = _positive_finite(physics_dt, label="physics_dt")
        self.velocity_scale_rad_s = _positive_finite(
            velocity_scale_rad_s,
            label="velocity_scale_rad_s",
        )
        self.effort_limit_fraction = _positive_finite(
            effort_limit_fraction,
            label="effort_limit_fraction",
        )
        if self.effort_limit_fraction > 1.0:
            raise ValueError("effort_limit_fraction must be in (0, 1]")
        self.effort_limits = limits.clone()
        self.action_dim = controller.command_dim
        self.action_low = -controller.clip
        self.action_high = controller.clip
        self.supported_control_modes: tuple[ControlMode, ...] = (
            "position",
            "velocity",
            "effort",
        )
        self._failure = torch.zeros(
            controller.num_envs, device=controller.device, dtype=torch.bool
        )

    def apply(
        self,
        actions: "torch.Tensor",
        state: object,
        active_mode: ControlMode = "position",
    ):
        import torch

        from linkerbot_sim.kaleidoscope.runtime import ActionExecution

        action = require_cuda_tensor(
            actions,
            name="joint control actions",
            ndim=2,
            leading_dim=self.controller.num_envs,
            dtype=self.controller.target.dtype,
        )
        if action.device != self.controller.device or action.shape[1:] != (
            self.controller.command_dim,
        ):
            raise ValueError("joint control actions have the wrong shape/device")
        assert_finite_async(action, name="joint control actions")
        clipped = torch.clamp(action, -self.controller.clip, self.controller.clip)
        normalized = clipped / self.controller.clip
        if active_mode == "position":
            reference = self.controller.apply(action)
            positions = reference[None, :, :].expand(
                self.physics_ticks_per_action, -1, -1
            )
            control = PositionControlTrajectory(
                positions=positions,
                velocities=torch.zeros_like(positions),
            )
        elif active_mode == "velocity":
            target = normalized * self.velocity_scale_rad_s
            velocities = target[None, :, :].expand(
                self.physics_ticks_per_action, -1, -1
            )
            reference = torch.clamp(
                self.controller.target
                + target * (self.physics_dt * self.physics_ticks_per_action),
                min=self.controller.lower,
                max=self.controller.upper,
            )
            self.controller.target.copy_(reference)
            control = VelocityControlTrajectory(velocities=velocities)
        elif active_mode == "effort":
            efforts = normalized * self.effort_limits * self.effort_limit_fraction
            expanded = efforts[None, :, :].expand(self.physics_ticks_per_action, -1, -1)
            reference = state.joint_positions.clone()
            self.controller.target.copy_(reference)
            control = EffortControlTrajectory(efforts=expanded)
        else:
            raise ControlModeIncompatibleError(
                f"joint_control does not support active mode {active_mode!r}",
                active_mode=active_mode,
                operation="joint_control",
            )
        return ActionExecution(
            control=control,
            position_reference=reference,
            failure_mask=self._failure,
            info={},
        )

    def reset(self, env_ids: "torch.Tensor", command: object) -> None:
        self.controller.reset_targets(command.joint_targets, env_ids)

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class KinematicsRobotBinding:
    """一个机器人在 action/state 列与 device IK solver 之间的冻结映射。"""

    label: str
    action_slice: slice
    command_slice: slice
    tcp_index: int
    solver: "DeviceBatchIKSolver"
    fixed_orientation_wxyz: "torch.Tensor | None" = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("kinematics binding label cannot be empty")
        for name, value in (
            ("action_slice", self.action_slice),
            ("command_slice", self.command_slice),
        ):
            if (
                value.step not in {None, 1}
                or type(value.start) is not int
                or type(value.stop) is not int
                or value.start < 0
                or value.stop <= value.start
            ):
                raise ValueError(f"{name} must be a positive contiguous slice")
        if type(self.tcp_index) is not int or self.tcp_index < 0:
            raise ValueError("tcp_index must be a non-negative int")
        if self.fixed_orientation_wxyz is not None:
            fixed = require_cuda_tensor(
                self.fixed_orientation_wxyz,
                name=f"{self.label} fixed orientation",
                ndim=1,
            )
            if fixed.shape != (4,) or fixed.device != self.solver.device:
                raise ValueError("fixed orientation must be CUDA (4,) on solver.device")


class IKRuntimeAction:
    """EE position/pose action 到非避障 device batch IK 的固定 adapter。"""

    def __init__(
        self,
        *,
        spec: ActionSpec,
        bindings: tuple[KinematicsRobotBinding, ...],
        command_dim: int,
        physics_dt: float = 1.0,
    ) -> None:
        allowed = {
            ActionMode.EE_DELTA_POSITION,
            ActionMode.EE_DELTA_POSE,
            ActionMode.EE_POSE_POSITION,
            ActionMode.EE_POSE_FULL,
        }
        if spec.mode not in allowed:
            raise ValueError("IKRuntimeAction requires a non-linear EE action mode")
        _validate_kinematics_bindings(spec, bindings, command_dim=command_dim)
        self.spec = spec
        self.bindings = bindings
        self.command_dim = command_dim
        self.action_dim = spec.action_dim
        # EE 动作直接使用米、旋转向量或 wxyz 数值，不伪装成归一化区间。
        self.action_low = -math.inf
        self.action_high = math.inf
        self.physics_ticks_per_action = spec.physics_ticks_per_action
        self.physics_dt = _positive_finite(physics_dt, label="physics_dt")
        self.supported_control_modes: tuple[ControlMode, ...] = (
            "position",
            "velocity",
        )
        self._closed_solver_ids: set[int] = set()

    def apply(
        self,
        actions: "torch.Tensor",
        state: object,
        active_mode: ControlMode = "position",
    ):
        import torch

        from linkerbot_sim.kaleidoscope.ik import solve_ik_batch
        from linkerbot_sim.kaleidoscope.runtime import ActionExecution

        action = _action_tensor(actions, self.spec, state)
        joint_target = state.position_references.clone()
        failures = torch.zeros(
            state.joint_positions.shape[0], device=action.device, dtype=torch.bool
        )
        info: dict[str, torch.Tensor] = {}
        for binding in self.bindings:
            values = action[:, binding.action_slice]
            current_position = state.tcp_positions_local[:, binding.tcp_index]
            current_orientation = state.tcp_orientations_wxyz[:, binding.tcp_index]
            target_position, target_orientation = _ee_target(
                mode=self.spec.mode,
                values=values,
                current_position=current_position,
                current_orientation=current_orientation,
                orientation_mode=self.spec.orientation_mode,
                fixed_orientation=binding.fixed_orientation_wxyz,
            )
            result = solve_ik_batch(
                binding.solver,
                target_positions=target_position,
                target_orientations_wxyz=target_orientation,
                seeds=state.joint_positions[:, binding.command_slice],
            )
            joint_target[:, binding.command_slice] = result.joint_positions
            failures |= ~result.success
            info[f"ik_success.{binding.label}"] = result.success
            info[f"ik_position_error.{binding.label}"] = result.position_error
            if result.orientation_error is not None:
                info[f"ik_orientation_error.{binding.label}"] = result.orientation_error
        positions = joint_target[None, :, :].expand(
            self.physics_ticks_per_action, -1, -1
        )
        control, info = _position_reference_control(
            positions,
            state=state,
            active_mode=active_mode,
            physics_dt=self.physics_dt,
            velocity_limit=self.spec.reference_velocity_limit,
            failure_mask=failures,
            info=info,
        )
        return ActionExecution(
            control=control,
            position_reference=_held_position_references(
                positions,
                state=state,
                failure_mask=failures,
            )[-1],
            failure_mask=failures,
            info=info,
        )

    def reset(self, _env_ids: "torch.Tensor", _command: object) -> None:
        return None

    def close(self) -> None:
        """关闭各 binding 独占的 solver/context；失败项下次调用继续重试。"""

        _close_binding_solvers(self.bindings, closed_ids=self._closed_solver_ids)


class LinearRuntimeAction:
    """固定 waypoint/tick 的同步 TCP 直线 action term。"""

    def __init__(
        self,
        *,
        spec: ActionSpec,
        bindings: tuple[KinematicsRobotBinding, ...],
        command_dim: int,
        physics_dt: float = 1.0,
    ) -> None:
        if spec.mode not in {
            ActionMode.EE_LINEAR_PATH_POSITION,
            ActionMode.EE_LINEAR_PATH_FULL,
        }:
            raise ValueError("LinearRuntimeAction requires a linear EE action mode")
        _validate_kinematics_bindings(spec, bindings, command_dim=command_dim)
        self.spec = spec
        self.bindings = bindings
        self.command_dim = command_dim
        self.action_dim = spec.action_dim
        # 直线终点直接使用 env-local 位置和可选 wxyz，不做隐藏归一化。
        self.action_low = -math.inf
        self.action_high = math.inf
        self.physics_ticks_per_action = spec.physics_ticks_per_action
        self.physics_dt = _positive_finite(physics_dt, label="physics_dt")
        self.supported_control_modes: tuple[ControlMode, ...] = (
            "position",
            "velocity",
        )
        self._closed_solver_ids: set[int] = set()

    def apply(
        self,
        actions: "torch.Tensor",
        state: object,
        active_mode: ControlMode = "position",
    ):
        import torch

        from linkerbot_sim.kaleidoscope.linear_motion import (
            solve_linear_motion_batch,
        )
        from linkerbot_sim.kaleidoscope.runtime import ActionExecution

        action = _action_tensor(actions, self.spec, state)
        trajectory = (
            state.position_references[None, :, :]
            .expand(self.physics_ticks_per_action, -1, -1)
            .clone()
        )
        failures = torch.zeros(
            state.joint_positions.shape[0], device=action.device, dtype=torch.bool
        )
        info: dict[str, torch.Tensor] = {}
        assert self.spec.waypoint_count is not None
        for binding in self.bindings:
            values = action[:, binding.action_slice]
            current_position = state.tcp_positions_local[:, binding.tcp_index]
            current_orientation = state.tcp_orientations_wxyz[:, binding.tcp_index]
            target_position = values[:, :3]
            target_orientation = (
                values[:, 3:7]
                if self.spec.mode is ActionMode.EE_LINEAR_PATH_FULL
                else (
                    binding.fixed_orientation_wxyz
                    if self.spec.orientation_mode == "target"
                    else None
                )
            )
            if target_orientation is not None and target_orientation.ndim == 1:
                target_orientation = target_orientation[None, :].expand(
                    state.joint_positions.shape[0], -1
                )
            result = solve_linear_motion_batch(
                binding.solver,
                start_positions=current_position,
                target_positions=target_position,
                start_orientations_wxyz=current_orientation,
                target_orientations_wxyz=target_orientation,
                seeds=state.joint_positions[:, binding.command_slice],
                waypoint_count=self.spec.waypoint_count,
                physics_ticks_per_action=self.physics_ticks_per_action,
                orientation_mode=(
                    "target"
                    if self.spec.mode is ActionMode.EE_LINEAR_PATH_FULL
                    else str(self.spec.orientation_mode)
                ),
                progress_mode=self.spec.progress_mode,
            )
            trajectory[:, :, binding.command_slice] = result.joint_positions
            failures |= ~result.success
            info[f"linear_success.{binding.label}"] = result.success
            info[f"linear_first_failure_step.{binding.label}"] = (
                result.first_failure_step
            )
            info[f"linear_position_error.{binding.label}"] = result.position_error
            if result.orientation_error is not None:
                info[f"linear_orientation_error.{binding.label}"] = (
                    result.orientation_error
                )
        held = _held_position_references(
            trajectory,
            state=state,
            failure_mask=failures,
        )
        control, info = _position_reference_control(
            held,
            state=state,
            active_mode=active_mode,
            physics_dt=self.physics_dt,
            velocity_limit=self.spec.reference_velocity_limit,
            failure_mask=failures,
            info=info,
            positions_are_held=True,
        )
        return ActionExecution(
            control=control,
            position_reference=held[-1],
            failure_mask=failures,
            info=info,
        )

    def reset(self, _env_ids: "torch.Tensor", _command: object) -> None:
        return None

    def close(self) -> None:
        """关闭各 binding 独占的 solver/context；失败项下次调用继续重试。"""

        _close_binding_solvers(self.bindings, closed_ids=self._closed_solver_ids)


def _validate_kinematics_bindings(
    spec: ActionSpec,
    bindings: tuple[KinematicsRobotBinding, ...],
    *,
    command_dim: int,
) -> None:
    if tuple(binding.label for binding in bindings) != spec.robot_labels:
        raise ValueError("kinematics bindings must follow ActionSpec.robot_labels")
    if not bindings:
        raise ValueError("kinematics bindings cannot be empty")
    device = bindings[0].solver.device
    action_columns: set[int] = set()
    command_columns: set[int] = set()
    for binding in bindings:
        if binding.solver.device != device:
            raise ValueError("all kinematics solvers must share one CUDA device")
        expected_command = binding.command_slice.stop - binding.command_slice.start
        if expected_command != binding.solver.command_dim:
            raise ValueError("command slice width must match solver.command_dim")
        action_columns.update(
            range(binding.action_slice.start, binding.action_slice.stop)
        )
        command_columns.update(
            range(binding.command_slice.start, binding.command_slice.stop)
        )
    if action_columns != set(range(spec.action_dim)):
        raise ValueError("kinematics action slices must exactly cover action columns")
    if command_columns != set(range(command_dim)):
        raise ValueError("kinematics command slices must exactly cover command columns")


def _close_binding_solvers(
    bindings: tuple[KinematicsRobotBinding, ...],
    *,
    closed_ids: set[int],
) -> None:
    """尽力关闭所有唯一 solver，并只提交已成功项的关闭进度。"""

    first_error: BaseException | None = None
    seen: set[int] = set()
    for binding in bindings:
        solver = binding.solver
        identity = id(solver)
        if identity in seen or identity in closed_ids:
            continue
        seen.add(identity)
        close = getattr(solver, "close", None)
        try:
            if not callable(close):
                raise TypeError(
                    f"kinematics solver {binding.label!r} must implement close()"
                )
            close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(
                    f"solver {binding.label!r} close also failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            closed_ids.add(identity)
    if first_error is not None:
        raise first_error


def _action_tensor(actions: object, spec: ActionSpec, state: object) -> "torch.Tensor":
    value = require_cuda_tensor(
        actions,
        name=f"{spec.mode} actions",
        ndim=2,
        leading_dim=state.joint_positions.shape[0],
        dtype=state.joint_positions.dtype,
    )
    if (
        value.device != state.joint_positions.device
        or value.shape[1] != spec.action_dim
    ):
        raise ValueError("EE actions have the wrong CUDA device/shape")
    assert_finite_async(value, name=f"{spec.mode} actions")
    return value


def _held_position_references(
    positions: "torch.Tensor",
    *,
    state: object,
    failure_mask: "torch.Tensor",
) -> "torch.Tensor":
    """Turn every failed environment into a full-decision reference hold."""

    import torch

    previous = state.position_references[None, :, :].expand_as(positions)
    return torch.where(failure_mask[None, :, None], previous, positions)


def _position_reference_control(
    positions: "torch.Tensor",
    *,
    state: object,
    active_mode: ControlMode,
    physics_dt: float,
    velocity_limit: float,
    failure_mask: "torch.Tensor",
    info: dict[str, "torch.Tensor"],
    positions_are_held: bool = False,
):
    """Convert rad references to one typed position/velocity trajectory."""

    import torch

    held = (
        positions
        if positions_are_held
        else _held_position_references(
            positions,
            state=state,
            failure_mask=failure_mask,
        )
    )
    if active_mode == "position":
        return (
            PositionControlTrajectory(
                positions=held,
                velocities=torch.zeros_like(held),
            ),
            info,
        )
    if active_mode != "velocity":
        raise ControlModeIncompatibleError(
            f"position-reference action does not support active mode {active_mode!r}",
            active_mode=active_mode,
            operation="action.apply",
        )
    previous = torch.cat((state.joint_positions[None, :, :], held[:-1]), dim=0)
    raw_velocity = (held - previous) / physics_dt
    velocities = torch.clamp(raw_velocity, min=-velocity_limit, max=velocity_limit)
    # A failed IK/waypoint row is a logical hold even if its previous position reference
    # differs from the current measured q. It must never inherit a catch-up velocity.
    velocities = torch.where(
        failure_mask[None, :, None],
        torch.zeros_like(velocities),
        velocities,
    )
    saturated = (
        torch.any(
            torch.abs(raw_velocity) > velocity_limit,
            dim=(0, 2),
        )
        & ~failure_mask
    )
    result_info = dict(info)
    result_info["control_velocity_saturated"] = saturated
    return VelocityControlTrajectory(velocities=velocities), result_info


def _positive_finite(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _ee_target(
    *,
    mode: ActionMode,
    values: "torch.Tensor",
    current_position: "torch.Tensor",
    current_orientation: "torch.Tensor",
    orientation_mode: str | None,
    fixed_orientation: "torch.Tensor | None",
) -> tuple["torch.Tensor", "torch.Tensor | None"]:
    if mode is ActionMode.EE_DELTA_POSITION:
        position = current_position + values
        orientation = _position_orientation(
            orientation_mode, current_orientation, fixed_orientation
        )
    elif mode is ActionMode.EE_POSE_POSITION:
        position = values
        orientation = _position_orientation(
            orientation_mode, current_orientation, fixed_orientation
        )
    elif mode is ActionMode.EE_DELTA_POSE:
        position = current_position + values[:, :3]
        orientation = quaternion_multiply_wxyz(
            _rotvec_quaternion(values[:, 3:6]), current_orientation
        )
    elif mode is ActionMode.EE_POSE_FULL:
        position = values[:, :3]
        orientation = normalize_quaternion_wxyz(values[:, 3:7])
    else:
        raise AssertionError(f"unexpected IK action mode {mode}")
    return position, orientation


def _position_orientation(
    mode: str | None,
    current: "torch.Tensor",
    fixed: "torch.Tensor | None",
) -> "torch.Tensor | None":
    if mode == "free":
        return None
    if mode == "current":
        return current
    if mode == "target" and fixed is not None:
        return fixed[None, :].expand(current.shape[0], -1)
    raise ValueError("orientation_mode='target' requires fixed_orientation_wxyz")


def _rotvec_quaternion(rotvec: "torch.Tensor") -> "torch.Tensor":
    import torch

    angle = torch.linalg.vector_norm(rotvec, dim=1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-7,
        torch.sin(half) / torch.where(angle > 0, angle, torch.ones_like(angle)),
        0.5 - angle * angle / 48.0,
    )
    return torch.cat((torch.cos(half), rotvec * scale), dim=1)


__all__ = [
    "ActionMode",
    "ActionSpec",
    "JointDeltaActionTerm",
    "JointControlRuntimeAction",
    "JointDeltaRuntimeAction",
    "IKRuntimeAction",
    "KinematicsRobotBinding",
    "LinearRuntimeAction",
    "action_spec_from_configuration",
]
