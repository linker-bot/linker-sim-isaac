"""Kaleidoscope 任务、动作、观测、奖励与 episode 规则配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..common import (
    ConfigurationError,
    as_bool,
    as_float,
    as_float_tuple,
    as_int,
    as_string,
    require_keys,
    strict_mapping,
)


@dataclass(frozen=True)
class JointControlActionSettings:
    """可在 position/velocity/effort 间切换的固定维度关节动作。"""

    mode: Literal["joint_control"]
    physics_ticks_per_action: int
    clip: float
    position_delta_scale_rad: float
    velocity_scale_rad_s: float
    effort_limit_fraction: float


@dataclass(frozen=True)
class JointDeltaActionSettings:
    """不创建 IK context 的固定 tick 关节增量动作。"""

    mode: Literal["joint_delta"]
    physics_ticks_per_action: int
    scale_rad: float
    clip: float
    reference_velocity_limit_rad_s: float


@dataclass(frozen=True)
class EePositionActionSettings:
    """仅用无碰撞 batch IK 求解 position goal 的固定形状动作。"""

    mode: Literal["ee_delta_position", "ee_pose_position"]
    physics_ticks_per_action: int
    orientation_mode: Literal["free", "current", "target"]
    failure_policy: Literal["hold_penalty_truncate"]
    reference_velocity_limit_rad_s: float


@dataclass(frozen=True)
class EeFullPoseActionSettings:
    """用无碰撞 batch IK 求解 position+quaternion goal 的固定形状动作。"""

    mode: Literal["ee_delta_pose", "ee_pose_full"]
    physics_ticks_per_action: int
    failure_policy: Literal["hold_penalty_truncate"]
    reference_velocity_limit_rad_s: float


@dataclass(frozen=True)
class EeLinearPositionActionSettings:
    """固定 waypoint/tick 的同步直线 position 动作，不是轨迹规划器。"""

    mode: Literal["ee_linear_path_position"]
    waypoint_count: int
    physics_ticks_per_action: int
    orientation_mode: Literal["free", "current", "target"]
    progress_mode: Literal["linear", "smoothstep"]
    failure_policy: Literal["hold_from_first_failure"]
    reference_velocity_limit_rad_s: float


@dataclass(frozen=True)
class EeLinearFullActionSettings:
    """带姿态插值的同步直线动作；不拥有 path search、retiming 或避障资源。"""

    mode: Literal["ee_linear_path_full"]
    waypoint_count: int
    physics_ticks_per_action: int
    progress_mode: Literal["linear", "smoothstep"]
    failure_policy: Literal["hold_from_first_failure"]
    reference_velocity_limit_rad_s: float


ActionSettings: TypeAlias = (
    JointControlActionSettings
    | JointDeltaActionSettings
    | EePositionActionSettings
    | EeFullPoseActionSettings
    | EeLinearPositionActionSettings
    | EeLinearFullActionSettings
)


def _action_from_mapping(value: object, *, label: str) -> ActionSettings:
    """按固定 action mode 拒绝不属于该 variant 的列或算法字段。"""

    mapping = strict_mapping(value, label=label)
    mode = as_string(mapping.get("mode"), label=f"{label}.mode")
    if mode == "joint_control":
        required = {
            "mode",
            "physics_ticks_per_action",
            "clip",
            "position_delta_scale_rad",
            "velocity_scale_rad_s",
            "effort_limit_fraction",
        }
        require_keys(mapping, required=required, label=label)
        fraction = as_float(
            mapping["effort_limit_fraction"],
            label=f"{label}.effort_limit_fraction",
            strictly_positive=True,
        )
        if fraction > 1.0:
            raise ConfigurationError(
                f"{label}.effort_limit_fraction must be within the range (0, 1]"
            )
        return JointControlActionSettings(
            mode="joint_control",
            physics_ticks_per_action=as_int(
                mapping["physics_ticks_per_action"],
                label=f"{label}.physics_ticks_per_action",
                minimum=1,
            ),
            clip=as_float(
                mapping["clip"], label=f"{label}.clip", strictly_positive=True
            ),
            position_delta_scale_rad=as_float(
                mapping["position_delta_scale_rad"],
                label=f"{label}.position_delta_scale_rad",
                strictly_positive=True,
            ),
            velocity_scale_rad_s=as_float(
                mapping["velocity_scale_rad_s"],
                label=f"{label}.velocity_scale_rad_s",
                strictly_positive=True,
            ),
            effort_limit_fraction=fraction,
        )
    if mode == "joint_delta":
        required = {
            "mode",
            "physics_ticks_per_action",
            "scale_rad",
            "clip",
            "reference_velocity_limit_rad_s",
        }
        require_keys(mapping, required=required, label=label)
        return JointDeltaActionSettings(
            mode="joint_delta",
            physics_ticks_per_action=as_int(
                mapping["physics_ticks_per_action"],
                label=f"{label}.physics_ticks_per_action",
                minimum=1,
            ),
            scale_rad=as_float(
                mapping["scale_rad"], label=f"{label}.scale_rad", strictly_positive=True
            ),
            clip=as_float(
                mapping["clip"], label=f"{label}.clip", strictly_positive=True
            ),
            reference_velocity_limit_rad_s=as_float(
                mapping["reference_velocity_limit_rad_s"],
                label=f"{label}.reference_velocity_limit_rad_s",
                strictly_positive=True,
            ),
        )

    common_required = {
        "mode",
        "physics_ticks_per_action",
        "failure_policy",
        "reference_velocity_limit_rad_s",
    }
    if mode in {"ee_delta_position", "ee_pose_position"}:
        required = common_required | {"orientation_mode"}
        require_keys(mapping, required=required, label=label)
        return EePositionActionSettings(
            mode=mode,  # type: ignore[arg-type]
            physics_ticks_per_action=as_int(
                mapping["physics_ticks_per_action"],
                label=f"{label}.physics_ticks_per_action",
                minimum=1,
            ),
            orientation_mode=as_string(
                mapping["orientation_mode"],
                label=f"{label}.orientation_mode",
                choices={"free", "current", "target"},
            ),  # type: ignore[arg-type]
            failure_policy=as_string(
                mapping["failure_policy"],
                label=f"{label}.failure_policy",
                choices={"hold_penalty_truncate"},
            ),  # type: ignore[arg-type]
            reference_velocity_limit_rad_s=as_float(
                mapping["reference_velocity_limit_rad_s"],
                label=f"{label}.reference_velocity_limit_rad_s",
                strictly_positive=True,
            ),
        )
    if mode in {"ee_delta_pose", "ee_pose_full"}:
        require_keys(mapping, required=common_required, label=label)
        return EeFullPoseActionSettings(
            mode=mode,  # type: ignore[arg-type]
            physics_ticks_per_action=as_int(
                mapping["physics_ticks_per_action"],
                label=f"{label}.physics_ticks_per_action",
                minimum=1,
            ),
            failure_policy=as_string(
                mapping["failure_policy"],
                label=f"{label}.failure_policy",
                choices={"hold_penalty_truncate"},
            ),  # type: ignore[arg-type]
            reference_velocity_limit_rad_s=as_float(
                mapping["reference_velocity_limit_rad_s"],
                label=f"{label}.reference_velocity_limit_rad_s",
                strictly_positive=True,
            ),
        )
    linear_common = common_required | {"waypoint_count", "progress_mode"}
    if mode == "ee_linear_path_position":
        required = linear_common | {"orientation_mode"}
        require_keys(mapping, required=required, label=label)
        return EeLinearPositionActionSettings(
            mode="ee_linear_path_position",
            waypoint_count=as_int(
                mapping["waypoint_count"], label=f"{label}.waypoint_count", minimum=2
            ),
            physics_ticks_per_action=as_int(
                mapping["physics_ticks_per_action"],
                label=f"{label}.physics_ticks_per_action",
                minimum=1,
            ),
            orientation_mode=as_string(
                mapping["orientation_mode"],
                label=f"{label}.orientation_mode",
                choices={"free", "current", "target"},
            ),  # type: ignore[arg-type]
            progress_mode=as_string(
                mapping["progress_mode"],
                label=f"{label}.progress_mode",
                choices={"linear", "smoothstep"},
            ),  # type: ignore[arg-type]
            failure_policy=as_string(
                mapping["failure_policy"],
                label=f"{label}.failure_policy",
                choices={"hold_from_first_failure"},
            ),  # type: ignore[arg-type]
            reference_velocity_limit_rad_s=as_float(
                mapping["reference_velocity_limit_rad_s"],
                label=f"{label}.reference_velocity_limit_rad_s",
                strictly_positive=True,
            ),
        )
    if mode == "ee_linear_path_full":
        require_keys(mapping, required=linear_common, label=label)
        return EeLinearFullActionSettings(
            mode="ee_linear_path_full",
            waypoint_count=as_int(
                mapping["waypoint_count"], label=f"{label}.waypoint_count", minimum=2
            ),
            physics_ticks_per_action=as_int(
                mapping["physics_ticks_per_action"],
                label=f"{label}.physics_ticks_per_action",
                minimum=1,
            ),
            progress_mode=as_string(
                mapping["progress_mode"],
                label=f"{label}.progress_mode",
                choices={"linear", "smoothstep"},
            ),  # type: ignore[arg-type]
            failure_policy=as_string(
                mapping["failure_policy"],
                label=f"{label}.failure_policy",
                choices={"hold_from_first_failure"},
            ),  # type: ignore[arg-type]
            reference_velocity_limit_rad_s=as_float(
                mapping["reference_velocity_limit_rad_s"],
                label=f"{label}.reference_velocity_limit_rad_s",
                strictly_positive=True,
            ),
        )
    allowed = (
        "joint_control, joint_delta, ee_delta_position, ee_delta_pose, "
        "ee_pose_position, ee_pose_full, "
        "ee_linear_path_position, ee_linear_path_full"
    )
    raise ConfigurationError(f"{label}.mode must be one of the fixed action variants: {allowed}")


@dataclass(frozen=True)
class ObservationSettings:
    command_target_error: bool
    tcp_pose: bool
    dynamic_object_state: bool
    goal: bool
    previous_action: bool
    normalized_time: bool

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "ObservationSettings":
        mapping = strict_mapping(value, label=label)
        names = {
            "command_target_error",
            "tcp_pose",
            "dynamic_object_state",
            "goal",
            "previous_action",
            "normalized_time",
        }
        require_keys(mapping, required=names, label=label)
        result = cls(
            **{name: as_bool(mapping[name], label=f"{label}.{name}") for name in names}
        )
        for mandatory in (
            "command_target_error",
            "tcp_pose",
            "dynamic_object_state",
            "goal",
            "previous_action",
        ):
            if not getattr(result, mandatory):
                raise ConfigurationError(
                    f"{label}.{mandatory} is a required observation for the tblock task"
                )
        return result


@dataclass(frozen=True)
class RewardSettings:
    distance_progress: float
    heading_progress: float
    hand_proximity_progress: float
    action_l2: float
    action_rate_l2: float
    success: float
    task_failure: float
    motion_failure: float

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "RewardSettings":
        mapping = strict_mapping(value, label=label)
        names = {
            "distance_progress",
            "heading_progress",
            "hand_proximity_progress",
            "action_l2",
            "action_rate_l2",
            "success",
            "task_failure",
            "motion_failure",
        }
        require_keys(mapping, required=names, label=label)
        return cls(
            **{name: as_float(mapping[name], label=f"{label}.{name}") for name in names}
        )


@dataclass(frozen=True)
class TerminationSettings:
    horizon_decisions: int
    success_distance_m: float
    success_heading_rad: float
    success_planar_speed_m_s: float
    success_streak: int
    failure_aabb_min: tuple[float, float, float]
    failure_aabb_max: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "TerminationSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "horizon_decisions",
            "success_distance_m",
            "success_heading_rad",
            "success_planar_speed_m_s",
            "success_streak",
            "failure_aabb_min",
            "failure_aabb_max",
        }
        require_keys(mapping, required=required, label=label)
        minimum = as_float_tuple(
            mapping["failure_aabb_min"], label=f"{label}.failure_aabb_min", length=3
        )
        maximum = as_float_tuple(
            mapping["failure_aabb_max"], label=f"{label}.failure_aabb_max", length=3
        )
        if any(low >= high for low, high in zip(minimum, maximum, strict=True)):
            raise ConfigurationError(f"{label} failure AABB requires each min to be less than max")
        return cls(
            horizon_decisions=as_int(
                mapping["horizon_decisions"],
                label=f"{label}.horizon_decisions",
                minimum=1,
            ),
            success_distance_m=as_float(
                mapping["success_distance_m"],
                label=f"{label}.success_distance_m",
                strictly_positive=True,
            ),
            success_heading_rad=as_float(
                mapping["success_heading_rad"],
                label=f"{label}.success_heading_rad",
                strictly_positive=True,
            ),
            success_planar_speed_m_s=as_float(
                mapping["success_planar_speed_m_s"],
                label=f"{label}.success_planar_speed_m_s",
                strictly_positive=True,
            ),
            success_streak=as_int(
                mapping["success_streak"], label=f"{label}.success_streak", minimum=1
            ),
            failure_aabb_min=minimum,  # type: ignore[arg-type]
            failure_aabb_max=maximum,  # type: ignore[arg-type]
        )


def _range2(value: object, *, label: str) -> tuple[float, float]:
    result = as_float_tuple(value, label=label, length=2)
    if result[0] > result[1]:
        raise ConfigurationError(f"{label}[0] must be <= {label}[1]")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class RandomizationSettings:
    robot_joint_delta_rad: tuple[float, float]
    block_x_delta_m: tuple[float, float]
    block_y_delta_m: tuple[float, float]
    block_yaw_delta_rad: tuple[float, float]
    goal_x_delta_m: tuple[float, float]
    goal_y_delta_m: tuple[float, float]
    goal_yaw_delta_rad: tuple[float, float]

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "RandomizationSettings":
        mapping = strict_mapping(value, label=label)
        names = {
            "robot_joint_delta_rad",
            "block_x_delta_m",
            "block_y_delta_m",
            "block_yaw_delta_rad",
            "goal_x_delta_m",
            "goal_y_delta_m",
            "goal_yaw_delta_rad",
        }
        require_keys(mapping, required=names, label=label)
        return cls(
            **{name: _range2(mapping[name], label=f"{label}.{name}") for name in names}
        )


@dataclass(frozen=True)
class KaleidoscopeTaskSettings:
    task_id: str
    dynamic_object: str
    heading_axis: tuple[float, float, float]
    action: ActionSettings
    observation: ObservationSettings
    reward: RewardSettings
    termination: TerminationSettings
    randomization: RandomizationSettings

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "task"
    ) -> "KaleidoscopeTaskSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "id",
            "dynamic_object",
            "heading_axis",
            "action",
            "observation",
            "reward",
            "termination",
            "randomization",
        }
        require_keys(mapping, required=required, label=label)
        heading_axis = as_float_tuple(
            mapping["heading_axis"], label=f"{label}.heading_axis", length=3
        )
        norm_squared = sum(component * component for component in heading_axis)
        if abs(norm_squared - 1.0) > 1e-6:
            raise ConfigurationError(f"{label}.heading_axis must be a unit vector")
        return cls(
            task_id=as_string(mapping["id"], label=f"{label}.id"),
            dynamic_object=as_string(
                mapping["dynamic_object"], label=f"{label}.dynamic_object"
            ),
            heading_axis=heading_axis,  # type: ignore[arg-type]
            action=_action_from_mapping(mapping["action"], label=f"{label}.action"),
            observation=ObservationSettings.from_mapping(
                mapping["observation"], label=f"{label}.observation"
            ),
            reward=RewardSettings.from_mapping(
                mapping["reward"], label=f"{label}.reward"
            ),
            termination=TerminationSettings.from_mapping(
                mapping["termination"], label=f"{label}.termination"
            ),
            randomization=RandomizationSettings.from_mapping(
                mapping["randomization"], label=f"{label}.randomization"
            ),
        )


__all__ = [
    "ActionSettings",
    "EeFullPoseActionSettings",
    "EeLinearFullActionSettings",
    "EeLinearPositionActionSettings",
    "EePositionActionSettings",
    "JointControlActionSettings",
    "JointDeltaActionSettings",
    "KaleidoscopeTaskSettings",
    "ObservationSettings",
    "RandomizationSettings",
    "RewardSettings",
    "TerminationSettings",
]
