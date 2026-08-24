"""Mirror 单场景交互控制与接口资源配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .common import (
    ConfigurationError,
    as_bool,
    as_float,
    as_float_tuple,
    as_int,
    as_string,
    as_string_tuple,
    require_keys,
    strict_mapping,
)


@dataclass(frozen=True)
class MirrorInterfaceSettings:
    """Mirror owner queue 与本地 ingress 的全部资源边界。

    这些字段决定常驻内存、单连接输入大小以及后台线程的启动/关闭 deadline，必须由
    strict profile 明确给出。TCP/WebSocket 是否监听及其地址仍是进程级 CLI 选择；一旦
    选择端点，端点只能使用这里冻结的容量和 timeout，避免 CLI 再维护第二套默认值。
    """

    admission_capacity: int
    terminal_history_capacity: int
    stdin_enabled: bool
    stdin_eof_policy: Literal["exit", "keep_alive"]
    response_timeout_s: float
    queue_poll_timeout_s: float
    max_message_bytes: int
    max_connections: int
    startup_timeout_s: float
    shutdown_timeout_s: float

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "control.interface"
    ) -> "MirrorInterfaceSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "admission_capacity",
            "terminal_history_capacity",
            "stdin_enabled",
            "stdin_eof_policy",
            "response_timeout_s",
            "queue_poll_timeout_s",
            "max_message_bytes",
            "max_connections",
            "startup_timeout_s",
            "shutdown_timeout_s",
        }
        require_keys(mapping, required=required, label=label)
        admission_capacity = as_int(
            mapping["admission_capacity"],
            label=f"{label}.admission_capacity",
            minimum=1,
        )
        terminal_history_capacity = as_int(
            mapping["terminal_history_capacity"],
            label=f"{label}.terminal_history_capacity",
            minimum=1,
        )
        if terminal_history_capacity < admission_capacity:
            raise ConfigurationError(
                f"{label}.terminal_history_capacity must be >= admission_capacity"
            )
        return cls(
            admission_capacity=admission_capacity,
            terminal_history_capacity=terminal_history_capacity,
            stdin_enabled=as_bool(
                mapping["stdin_enabled"], label=f"{label}.stdin_enabled"
            ),
            stdin_eof_policy=as_string(
                mapping["stdin_eof_policy"],
                label=f"{label}.stdin_eof_policy",
                choices={"exit", "keep_alive"},
            ),  # type: ignore[arg-type]
            response_timeout_s=as_float(
                mapping["response_timeout_s"],
                label=f"{label}.response_timeout_s",
                strictly_positive=True,
            ),
            queue_poll_timeout_s=as_float(
                mapping["queue_poll_timeout_s"],
                label=f"{label}.queue_poll_timeout_s",
                strictly_positive=True,
            ),
            max_message_bytes=as_int(
                mapping["max_message_bytes"],
                label=f"{label}.max_message_bytes",
                minimum=1,
            ),
            max_connections=as_int(
                mapping["max_connections"],
                label=f"{label}.max_connections",
                minimum=1,
            ),
            startup_timeout_s=as_float(
                mapping["startup_timeout_s"],
                label=f"{label}.startup_timeout_s",
                strictly_positive=True,
            ),
            shutdown_timeout_s=as_float(
                mapping["shutdown_timeout_s"],
                label=f"{label}.shutdown_timeout_s",
                strictly_positive=True,
            ),
        )


def _non_negative_six(value: object, *, label: str) -> tuple[float, ...]:
    result = as_float_tuple(value, label=label, length=6)
    if any(item < 0.0 for item in result):
        raise ConfigurationError(f"{label} entries must each be >= 0")
    return result


def _positive_six(value: object, *, label: str) -> tuple[float, ...]:
    result = as_float_tuple(value, label=label, length=6)
    if any(item <= 0.0 for item in result):
        raise ConfigurationError(f"{label} entries must each be > 0")
    return result


def _bool_six(value: object, *, label: str) -> tuple[bool, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"{label} must be a boolean sequence of length 6")
    if len(value) != 6:
        raise ConfigurationError(f"{label} must contain exactly 6 items")
    return tuple(
        as_bool(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class HybridMotionSettings:
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "HybridMotionSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(mapping, required={"stiffness", "damping"}, label=label)
        return cls(
            stiffness=_non_negative_six(
                mapping["stiffness"], label=f"{label}.stiffness"
            ),
            damping=_non_negative_six(mapping["damping"], label=f"{label}.damping"),
        )


@dataclass(frozen=True)
class HybridForceSettings:
    proportional: tuple[float, ...]
    integral: tuple[float, ...]
    integral_abs_limit: tuple[float, ...]
    wrench_lpf_cutoff_hz: float

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "HybridForceSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "proportional",
            "integral",
            "integral_abs_limit",
            "wrench_lpf_cutoff_hz",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            proportional=_non_negative_six(
                mapping["proportional"], label=f"{label}.proportional"
            ),
            integral=_non_negative_six(mapping["integral"], label=f"{label}.integral"),
            integral_abs_limit=_non_negative_six(
                mapping["integral_abs_limit"],
                label=f"{label}.integral_abs_limit",
            ),
            wrench_lpf_cutoff_hz=as_float(
                mapping["wrench_lpf_cutoff_hz"],
                label=f"{label}.wrench_lpf_cutoff_hz",
                strictly_positive=True,
            ),
        )


@dataclass(frozen=True)
class HybridPostureSettings:
    enabled: bool
    stiffness: float
    damping: float
    characteristic_length_m: float
    singularity_damping: float
    minimum_singular_value: float
    maximum_condition_number: float

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "HybridPostureSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "enabled",
            "stiffness",
            "damping",
            "characteristic_length_m",
            "singularity_damping",
            "minimum_singular_value",
            "maximum_condition_number",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            enabled=as_bool(mapping["enabled"], label=f"{label}.enabled"),
            stiffness=as_float(
                mapping["stiffness"], label=f"{label}.stiffness", minimum=0.0
            ),
            damping=as_float(mapping["damping"], label=f"{label}.damping", minimum=0.0),
            characteristic_length_m=as_float(
                mapping["characteristic_length_m"],
                label=f"{label}.characteristic_length_m",
                strictly_positive=True,
            ),
            singularity_damping=as_float(
                mapping["singularity_damping"],
                label=f"{label}.singularity_damping",
                strictly_positive=True,
            ),
            minimum_singular_value=as_float(
                mapping["minimum_singular_value"],
                label=f"{label}.minimum_singular_value",
                strictly_positive=True,
            ),
            maximum_condition_number=as_float(
                mapping["maximum_condition_number"],
                label=f"{label}.maximum_condition_number",
                minimum=1.0,
            ),
        )


@dataclass(frozen=True)
class HybridTareSettings:
    warmup_ticks: int
    sample_count: int
    maximum_joint_speed: float
    maximum_std_wrench: tuple[float, ...]

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "HybridTareSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "warmup_ticks",
            "sample_count",
            "maximum_joint_speed",
            "maximum_std_wrench",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            warmup_ticks=as_int(
                mapping["warmup_ticks"], label=f"{label}.warmup_ticks", minimum=0
            ),
            sample_count=as_int(
                mapping["sample_count"], label=f"{label}.sample_count", minimum=1
            ),
            maximum_joint_speed=as_float(
                mapping["maximum_joint_speed"],
                label=f"{label}.maximum_joint_speed",
                minimum=0.0,
            ),
            maximum_std_wrench=_positive_six(
                mapping["maximum_std_wrench"],
                label=f"{label}.maximum_std_wrench",
            ),
        )


@dataclass(frozen=True)
class HybridContactSettings:
    enter_abs_wrench: tuple[float, ...]
    exit_abs_wrench: tuple[float, ...]
    enter_ticks: int
    exit_ticks: int
    max_free_space_displacement_m: float

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "HybridContactSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "enter_abs_wrench",
            "exit_abs_wrench",
            "enter_ticks",
            "exit_ticks",
            "max_free_space_displacement_m",
        }
        require_keys(mapping, required=required, label=label)
        enter = _positive_six(
            mapping["enter_abs_wrench"], label=f"{label}.enter_abs_wrench"
        )
        exit_values = _non_negative_six(
            mapping["exit_abs_wrench"], label=f"{label}.exit_abs_wrench"
        )
        if any(
            exit_value >= enter_value
            for exit_value, enter_value in zip(exit_values, enter, strict=True)
        ):
            raise ConfigurationError(
                f"{label}.exit_abs_wrench entries must each be less than enter_abs_wrench"
            )
        return cls(
            enter_abs_wrench=enter,
            exit_abs_wrench=exit_values,
            enter_ticks=as_int(
                mapping["enter_ticks"], label=f"{label}.enter_ticks", minimum=1
            ),
            exit_ticks=as_int(
                mapping["exit_ticks"], label=f"{label}.exit_ticks", minimum=1
            ),
            max_free_space_displacement_m=as_float(
                mapping["max_free_space_displacement_m"],
                label=f"{label}.max_free_space_displacement_m",
                strictly_positive=True,
            ),
        )


@dataclass(frozen=True)
class HybridLimitSettings:
    max_abs_wrench: tuple[float, ...]
    max_abs_pose_error: tuple[float, ...]
    max_abs_joint_effort: float
    max_joint_effort_rate: float
    max_joint_speed: float
    sensor_stale_ticks: int
    ramp_down_ticks: int

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "HybridLimitSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "max_abs_wrench",
            "max_abs_pose_error",
            "max_abs_joint_effort",
            "max_joint_effort_rate",
            "max_joint_speed",
            "sensor_stale_ticks",
            "ramp_down_ticks",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            max_abs_wrench=_positive_six(
                mapping["max_abs_wrench"], label=f"{label}.max_abs_wrench"
            ),
            max_abs_pose_error=_positive_six(
                mapping["max_abs_pose_error"],
                label=f"{label}.max_abs_pose_error",
            ),
            max_abs_joint_effort=as_float(
                mapping["max_abs_joint_effort"],
                label=f"{label}.max_abs_joint_effort",
                strictly_positive=True,
            ),
            max_joint_effort_rate=as_float(
                mapping["max_joint_effort_rate"],
                label=f"{label}.max_joint_effort_rate",
                strictly_positive=True,
            ),
            max_joint_speed=as_float(
                mapping["max_joint_speed"],
                label=f"{label}.max_joint_speed",
                strictly_positive=True,
            ),
            sensor_stale_ticks=as_int(
                mapping["sensor_stale_ticks"],
                label=f"{label}.sensor_stale_ticks",
                minimum=1,
            ),
            ramp_down_ticks=as_int(
                mapping["ramp_down_ticks"],
                label=f"{label}.ramp_down_ticks",
                minimum=1,
            ),
        )


@dataclass(frozen=True)
class HybridTuningLimits:
    max_motion_stiffness: tuple[float, ...]
    max_motion_damping: tuple[float, ...]
    max_force_proportional: tuple[float, ...]
    max_force_integral: tuple[float, ...]
    max_posture_stiffness: float
    max_posture_damping: float

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "HybridTuningLimits":
        mapping = strict_mapping(value, label=label)
        required = {
            "max_motion_stiffness",
            "max_motion_damping",
            "max_force_proportional",
            "max_force_integral",
            "max_posture_stiffness",
            "max_posture_damping",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            max_motion_stiffness=_non_negative_six(
                mapping["max_motion_stiffness"],
                label=f"{label}.max_motion_stiffness",
            ),
            max_motion_damping=_non_negative_six(
                mapping["max_motion_damping"],
                label=f"{label}.max_motion_damping",
            ),
            max_force_proportional=_non_negative_six(
                mapping["max_force_proportional"],
                label=f"{label}.max_force_proportional",
            ),
            max_force_integral=_non_negative_six(
                mapping["max_force_integral"],
                label=f"{label}.max_force_integral",
            ),
            max_posture_stiffness=as_float(
                mapping["max_posture_stiffness"],
                label=f"{label}.max_posture_stiffness",
                minimum=0.0,
            ),
            max_posture_damping=as_float(
                mapping["max_posture_damping"],
                label=f"{label}.max_posture_damping",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True)
class HybridForcePositionSettings:
    minimum_physics_frequency_hz: float
    max_duration_s: float
    supported_reference_frames: tuple[str, ...]
    allowed_force_axes: tuple[bool, ...]
    motion: HybridMotionSettings
    force: HybridForceSettings
    posture: HybridPostureSettings
    tare: HybridTareSettings
    contact: HybridContactSettings
    limits: HybridLimitSettings
    tuning: HybridTuningLimits

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        label: str = "hybrid_force_position",
    ) -> "HybridForcePositionSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "minimum_physics_frequency_hz",
            "max_duration_s",
            "supported_reference_frames",
            "allowed_force_axes",
            "motion",
            "force",
            "posture",
            "tare",
            "contact",
            "limits",
            "tuning",
        }
        require_keys(mapping, required=required, label=label)
        frames = as_string_tuple(
            mapping["supported_reference_frames"],
            label=f"{label}.supported_reference_frames",
        )
        if frames != ("world",):
            raise ConfigurationError(
                f"{label}.supported_reference_frames must be exactly [world] in the first phase"
            )
        axes = _bool_six(
            mapping["allowed_force_axes"], label=f"{label}.allowed_force_axes"
        )
        if not any(axes):
            raise ConfigurationError(f"{label}.allowed_force_axes must enable at least one axis")
        motion = HybridMotionSettings.from_mapping(
            mapping["motion"], label=f"{label}.motion"
        )
        force = HybridForceSettings.from_mapping(
            mapping["force"], label=f"{label}.force"
        )
        posture = HybridPostureSettings.from_mapping(
            mapping["posture"], label=f"{label}.posture"
        )
        tuning = HybridTuningLimits.from_mapping(
            mapping["tuning"], label=f"{label}.tuning"
        )
        _require_not_above(
            motion.stiffness,
            tuning.max_motion_stiffness,
            label=f"{label}.motion.stiffness",
        )
        _require_not_above(
            motion.damping,
            tuning.max_motion_damping,
            label=f"{label}.motion.damping",
        )
        _require_not_above(
            force.proportional,
            tuning.max_force_proportional,
            label=f"{label}.force.proportional",
        )
        _require_not_above(
            force.integral,
            tuning.max_force_integral,
            label=f"{label}.force.integral",
        )
        if posture.stiffness > tuning.max_posture_stiffness:
            raise ConfigurationError(f"{label}.posture.stiffness exceeds the tuning limit")
        if posture.damping > tuning.max_posture_damping:
            raise ConfigurationError(f"{label}.posture.damping exceeds the tuning limit")
        return cls(
            minimum_physics_frequency_hz=as_float(
                mapping["minimum_physics_frequency_hz"],
                label=f"{label}.minimum_physics_frequency_hz",
                strictly_positive=True,
            ),
            max_duration_s=as_float(
                mapping["max_duration_s"],
                label=f"{label}.max_duration_s",
                strictly_positive=True,
            ),
            supported_reference_frames=frames,
            allowed_force_axes=axes,
            motion=motion,
            force=force,
            posture=posture,
            tare=HybridTareSettings.from_mapping(
                mapping["tare"], label=f"{label}.tare"
            ),
            contact=HybridContactSettings.from_mapping(
                mapping["contact"], label=f"{label}.contact"
            ),
            limits=HybridLimitSettings.from_mapping(
                mapping["limits"], label=f"{label}.limits"
            ),
            tuning=tuning,
        )


def _require_not_above(
    values: tuple[float, ...],
    maximum: tuple[float, ...],
    *,
    label: str,
) -> None:
    if any(value > limit for value, limit in zip(values, maximum, strict=True)):
        raise ConfigurationError(f"{label} exceeds the tuning limit")


@dataclass(frozen=True)
class MirrorControlSettings:
    """单环境交互控制与空闲步进语义。

    后端专用 drive bundle 由根配置根据 ``physics.engine`` 派生，不在本 profile
    重复声明；scene/robot 的显式实例覆盖仍由配置图 catalog 解析。
    """

    mode: Literal["position", "velocity", "effort"]
    idle_physics_policy: Literal["pause", "hold_step"]
    idle_step_duration_s: float
    sync_simulation_to_wall_clock: bool
    joint_interpolation: Literal["linear", "smoothstep"]
    pose_frame: Literal["env", "world"]
    orientation_mode: Literal["free", "current", "target"]
    interface: MirrorInterfaceSettings

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "control"
    ) -> "MirrorControlSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "mode",
            "idle_physics_policy",
            "idle_step_duration_s",
            "sync_simulation_to_wall_clock",
            "joint_interpolation",
            "pose_frame",
            "orientation_mode",
            "interface",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            mode=as_string(
                mapping["mode"],
                label=f"{label}.mode",
                choices={"position", "velocity", "effort"},
            ),  # type: ignore[arg-type]
            idle_physics_policy=as_string(
                mapping["idle_physics_policy"],
                label=f"{label}.idle_physics_policy",
                choices={"pause", "hold_step"},
            ),  # type: ignore[arg-type]
            idle_step_duration_s=as_float(
                mapping["idle_step_duration_s"],
                label=f"{label}.idle_step_duration_s",
                strictly_positive=True,
            ),
            sync_simulation_to_wall_clock=as_bool(
                mapping["sync_simulation_to_wall_clock"],
                label=f"{label}.sync_simulation_to_wall_clock",
            ),
            joint_interpolation=as_string(
                mapping["joint_interpolation"],
                label=f"{label}.joint_interpolation",
                choices={"linear", "smoothstep"},
            ),  # type: ignore[arg-type]
            pose_frame=as_string(
                mapping["pose_frame"],
                label=f"{label}.pose_frame",
                choices={"env", "world"},
            ),  # type: ignore[arg-type]
            orientation_mode=as_string(
                mapping["orientation_mode"],
                label=f"{label}.orientation_mode",
                choices={"free", "current", "target"},
            ),  # type: ignore[arg-type]
            interface=MirrorInterfaceSettings.from_mapping(
                mapping["interface"], label=f"{label}.interface"
            ),
        )


__all__ = [
    "HybridContactSettings",
    "HybridForcePositionSettings",
    "HybridForceSettings",
    "HybridLimitSettings",
    "HybridMotionSettings",
    "HybridPostureSettings",
    "HybridTareSettings",
    "HybridTuningLimits",
    "MirrorControlSettings",
    "MirrorInterfaceSettings",
]
