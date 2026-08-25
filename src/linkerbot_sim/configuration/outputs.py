"""Mirror 的渲染、相机、日志与遥测输出配置。"""

from __future__ import annotations

from dataclasses import dataclass

from .common import (
    ConfigurationError,
    as_bool,
    as_float,
    as_int,
    as_string,
    require_keys,
    strict_mapping,
)


@dataclass(frozen=True)
class RenderOutputSettings:
    """Mirror render cadence；该类型不会被 Kaleidoscope 根配置引用。"""

    enabled: bool
    gui: bool
    renderer: str
    width: int
    height: int
    samples_per_pixel_per_frame: int

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "RenderOutputSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "enabled",
            "gui",
            "renderer",
            "width",
            "height",
            "samples_per_pixel_per_frame",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            enabled=as_bool(mapping["enabled"], label=f"{label}.enabled"),
            gui=as_bool(mapping["gui"], label=f"{label}.gui"),
            renderer=as_string(mapping["renderer"], label=f"{label}.renderer"),
            width=as_int(mapping["width"], label=f"{label}.width", minimum=1),
            height=as_int(mapping["height"], label=f"{label}.height", minimum=1),
            samples_per_pixel_per_frame=as_int(
                mapping["samples_per_pixel_per_frame"],
                label=f"{label}.samples_per_pixel_per_frame",
                minimum=1,
            ),
        )


@dataclass(frozen=True)
class CameraOutputSettings:
    """相机输出队列与 sink；相机几何本身属于 scene profile。

    ``save_root`` 会由 Mirror 装配层按 camera ID 派生唯一子目录；live endpoint 与
    MCAP path 则由所有 camera 共享一个 sink，避免每台相机重复绑定端口或打开文件。
    """

    enabled: bool
    save_root: str | None
    foxglove_live_host: str
    foxglove_live_port: int | None
    foxglove_mcap_path: str | None
    queue_size: int
    overflow_policy: str
    worker_poll_interval_s: float
    existing_data_policy: str
    shutdown_policy: str
    rgb_format: str
    depth_format: str
    metadata_flush_interval_frames: int
    max_bytes_per_camera: int
    shutdown_timeout_s: float

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "CameraOutputSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "enabled",
            "save_root",
            "foxglove_live_host",
            "foxglove_live_port",
            "foxglove_mcap_path",
            "queue_size",
            "overflow_policy",
            "worker_poll_interval_s",
            "existing_data_policy",
            "shutdown_policy",
            "rgb_format",
            "depth_format",
            "metadata_flush_interval_frames",
            "max_bytes_per_camera",
            "shutdown_timeout_s",
        }
        require_keys(mapping, required=required, label=label)
        enabled = as_bool(mapping["enabled"], label=f"{label}.enabled")
        save_root = _optional_string(mapping["save_root"], label=f"{label}.save_root")
        live_port = _optional_port(
            mapping["foxglove_live_port"],
            label=f"{label}.foxglove_live_port",
        )
        mcap_path = _optional_string(
            mapping["foxglove_mcap_path"],
            label=f"{label}.foxglove_mcap_path",
        )
        has_consumer = (
            save_root is not None or live_port is not None or mcap_path is not None
        )
        if enabled and not has_consumer:
            raise ConfigurationError(
                f"{label} must configure save_root, live_port or mcap_path when enabled=true"
            )
        return cls(
            enabled=enabled,
            save_root=save_root,
            foxglove_live_host=as_string(
                mapping["foxglove_live_host"],
                label=f"{label}.foxglove_live_host",
            ),
            foxglove_live_port=live_port,
            foxglove_mcap_path=mcap_path,
            queue_size=as_int(
                mapping["queue_size"], label=f"{label}.queue_size", minimum=1
            ),
            overflow_policy=as_string(
                mapping["overflow_policy"],
                label=f"{label}.overflow_policy",
                choices={"block", "drop_oldest", "drop_newest", "error"},
            ),
            worker_poll_interval_s=as_float(
                mapping["worker_poll_interval_s"],
                label=f"{label}.worker_poll_interval_s",
                strictly_positive=True,
            ),
            existing_data_policy=as_string(
                mapping["existing_data_policy"],
                label=f"{label}.existing_data_policy",
                choices={"error", "truncate", "resume", "timestamped_dir"},
            ),
            shutdown_policy=as_string(
                mapping["shutdown_policy"],
                label=f"{label}.shutdown_policy",
                choices={"drain", "discard"},
            ),
            rgb_format=as_string(
                mapping["rgb_format"],
                label=f"{label}.rgb_format",
                choices={"ppm", "png", "npy"},
            ),
            depth_format=as_string(
                mapping["depth_format"],
                label=f"{label}.depth_format",
                choices={"npy", "npz"},
            ),
            metadata_flush_interval_frames=as_int(
                mapping["metadata_flush_interval_frames"],
                label=f"{label}.metadata_flush_interval_frames",
                minimum=1,
            ),
            max_bytes_per_camera=as_int(
                mapping["max_bytes_per_camera"],
                label=f"{label}.max_bytes_per_camera",
                minimum=1,
            ),
            shutdown_timeout_s=as_float(
                mapping["shutdown_timeout_s"],
                label=f"{label}.shutdown_timeout_s",
                strictly_positive=True,
            ),
        )


@dataclass(frozen=True)
class LoggingOutputSettings:
    """Mirror 关节 CSV 的唯一不可变设置对象。"""

    enabled: bool
    existing_data_policy: str
    joint_tracking_path: str | None
    flush_interval_s: float
    interval_steps: int
    log_actual_position: bool
    log_actual_velocity: bool
    log_command_position: bool
    log_command_velocity: bool
    log_command_effort: bool
    log_action_effort: bool
    log_measured_effort: bool
    log_applied_effort: bool
    hybrid_control_path: str | None = None
    log_hybrid_control: bool = False

    def should_write_step(self, step: int) -> bool:
        """返回当前 physics step 是否落在启用的日志采样周期上。"""

        return self.enabled and int(step) % max(1, int(self.interval_steps)) == 0

    def flush_interval_steps(self, physics_dt: float) -> int:
        """把 flush 时间间隔投影为至少一个 physics step。"""

        if physics_dt <= 0:
            return 1
        return max(1, int(round(float(self.flush_interval_s) / float(physics_dt))))

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "LoggingOutputSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "enabled",
            "existing_data_policy",
            "joint_tracking_path",
            "hybrid_control_path",
            "log_hybrid_control",
            "flush_interval_s",
            "interval_steps",
            "log_actual_position",
            "log_actual_velocity",
            "log_command_position",
            "log_command_velocity",
            "log_command_effort",
            "log_action_effort",
            "log_measured_effort",
            "log_applied_effort",
        }
        require_keys(mapping, required=required, label=label)
        enabled = as_bool(mapping["enabled"], label=f"{label}.enabled")
        path = _optional_string(
            mapping["joint_tracking_path"],
            label=f"{label}.joint_tracking_path",
        )
        if enabled and path is None:
            raise ConfigurationError(
                f"{label}.joint_tracking_path must be present when enabled=true"
            )
        hybrid_path = _optional_string(
            mapping["hybrid_control_path"],
            label=f"{label}.hybrid_control_path",
        )
        log_hybrid = as_bool(
            mapping["log_hybrid_control"],
            label=f"{label}.log_hybrid_control",
        )
        if log_hybrid and not enabled:
            raise ConfigurationError(
                f"{label}.log_hybrid_control=true requires enabled=true"
            )
        if log_hybrid and hybrid_path is None:
            raise ConfigurationError(
                f"{label}.hybrid_control_path must be present when log_hybrid_control=true"
            )
        return cls(
            enabled=enabled,
            existing_data_policy=as_string(
                mapping["existing_data_policy"],
                label=f"{label}.existing_data_policy",
                choices={"error", "truncate", "resume", "timestamped_dir"},
            ),
            joint_tracking_path=path,
            flush_interval_s=as_float(
                mapping["flush_interval_s"],
                label=f"{label}.flush_interval_s",
                strictly_positive=True,
            ),
            interval_steps=as_int(
                mapping["interval_steps"],
                label=f"{label}.interval_steps",
                minimum=1,
            ),
            hybrid_control_path=hybrid_path,
            log_hybrid_control=log_hybrid,
            **{
                name: as_bool(mapping[name], label=f"{label}.{name}")
                for name in (
                    "log_actual_position",
                    "log_actual_velocity",
                    "log_command_position",
                    "log_command_velocity",
                    "log_command_effort",
                    "log_action_effort",
                    "log_measured_effort",
                    "log_applied_effort",
                )
            },
        )


@dataclass(frozen=True)
class TelemetryOutputSettings:
    enabled: bool
    rate_hz: float
    buffer_size: int
    drop_policy: str
    on_error: str
    include_joint_states: bool
    include_state_json: bool
    include_scene_markers: bool
    include_efforts: bool
    include_objects: bool
    joint_effort_field: str
    foxglove_live_host: str
    foxglove_live_port: int | None
    mcap_path: str | None
    existing_data_policy: str
    topics: "TelemetryTopicSettings"
    shutdown_timeout_s: float
    include_hybrid_control: bool = False

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "TelemetryOutputSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(
            mapping,
            required={
                "enabled",
                "rate_hz",
                "buffer_size",
                "drop_policy",
                "on_error",
                "include_joint_states",
                "include_state_json",
                "include_scene_markers",
                "include_efforts",
                "include_objects",
                "include_hybrid_control",
                "joint_effort_field",
                "foxglove_live_host",
                "foxglove_live_port",
                "mcap_path",
                "existing_data_policy",
                "topics",
                "shutdown_timeout_s",
            },
            label=label,
        )
        enabled = as_bool(mapping["enabled"], label=f"{label}.enabled")
        rate_hz = as_float(mapping["rate_hz"], label=f"{label}.rate_hz", minimum=0.0)
        if enabled and rate_hz <= 0.0:
            raise ConfigurationError(f"{label}.rate_hz must be > 0 when enabled=true")
        live_port = _optional_port(
            mapping["foxglove_live_port"],
            label=f"{label}.foxglove_live_port",
        )
        mcap_path = _optional_string(mapping["mcap_path"], label=f"{label}.mcap_path")
        if enabled and live_port is None and mcap_path is None:
            raise ConfigurationError(
                f"{label} must configure live_port or mcap_path when enabled=true"
            )
        include_joint_states = as_bool(
            mapping["include_joint_states"],
            label=f"{label}.include_joint_states",
        )
        include_state_json = as_bool(
            mapping["include_state_json"], label=f"{label}.include_state_json"
        )
        include_scene_markers = as_bool(
            mapping["include_scene_markers"],
            label=f"{label}.include_scene_markers",
        )
        include_hybrid_control = as_bool(
            mapping["include_hybrid_control"],
            label=f"{label}.include_hybrid_control",
        )
        if enabled and not (
            include_joint_states
            or include_state_json
            or include_scene_markers
            or include_hybrid_control
        ):
            raise ConfigurationError(
                f"{label} must select at least one telemetry modality when enabled"
            )
        include_efforts = as_bool(
            mapping["include_efforts"], label=f"{label}.include_efforts"
        )
        effort_field = as_string(
            mapping["joint_effort_field"],
            label=f"{label}.joint_effort_field",
            choices={"none", "commanded", "measured", "applied"},
        )
        if include_efforts and effort_field == "none":
            raise ConfigurationError(
                f"{label}.joint_effort_field must not be none when include_efforts=true"
            )
        return cls(
            enabled=enabled,
            rate_hz=rate_hz,
            buffer_size=as_int(
                mapping["buffer_size"], label=f"{label}.buffer_size", minimum=1
            ),
            drop_policy=as_string(
                mapping["drop_policy"],
                label=f"{label}.drop_policy",
                choices={"latest", "drop_oldest", "drop_newest"},
            ),
            on_error=as_string(
                mapping["on_error"],
                label=f"{label}.on_error",
                choices={"stop", "continue"},
            ),
            include_joint_states=include_joint_states,
            include_state_json=include_state_json,
            include_scene_markers=include_scene_markers,
            include_efforts=include_efforts,
            include_objects=as_bool(
                mapping["include_objects"], label=f"{label}.include_objects"
            ),
            include_hybrid_control=include_hybrid_control,
            joint_effort_field=effort_field,
            foxglove_live_host=as_string(
                mapping["foxglove_live_host"],
                label=f"{label}.foxglove_live_host",
            ),
            foxglove_live_port=live_port,
            mcap_path=mcap_path,
            existing_data_policy=as_string(
                mapping["existing_data_policy"],
                label=f"{label}.existing_data_policy",
                choices={"error", "truncate", "timestamped_dir"},
            ),
            topics=TelemetryTopicSettings.from_mapping(
                mapping["topics"], label=f"{label}.topics"
            ),
            shutdown_timeout_s=as_float(
                mapping["shutdown_timeout_s"],
                label=f"{label}.shutdown_timeout_s",
                strictly_positive=True,
            ),
        )


@dataclass(frozen=True)
class TelemetryTopicSettings:
    """Mirror telemetry 的三个互不冲突的绝对 topic。"""

    joint_states: str
    scene: str
    state: str
    hybrid_control: str = "/linkerbot/mirror/hybrid_control"

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "TelemetryTopicSettings":
        mapping = strict_mapping(value, label=label)
        names = {"joint_states", "scene", "state", "hybrid_control"}
        require_keys(mapping, required=names, label=label)
        values = {
            name: as_string(mapping[name], label=f"{label}.{name}") for name in names
        }
        if any(not value.startswith("/") or ".." in value for value in values.values()):
            raise ConfigurationError(
                f"{label} topics must be absolute paths without '..'"
            )
        if len(set(values.values())) != len(values):
            raise ConfigurationError(f"{label} topics must be distinct")
        return cls(**values)


@dataclass(frozen=True)
class MirrorOutputsSettings:
    render: RenderOutputSettings
    camera: CameraOutputSettings
    logging: LoggingOutputSettings
    telemetry: TelemetryOutputSettings

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "outputs"
    ) -> "MirrorOutputsSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(
            mapping,
            required={"render", "camera", "logging", "telemetry"},
            label=label,
        )
        return cls(
            render=RenderOutputSettings.from_mapping(
                mapping["render"], label=f"{label}.render"
            ),
            camera=CameraOutputSettings.from_mapping(
                mapping["camera"], label=f"{label}.camera"
            ),
            logging=LoggingOutputSettings.from_mapping(
                mapping["logging"], label=f"{label}.logging"
            ),
            telemetry=TelemetryOutputSettings.from_mapping(
                mapping["telemetry"], label=f"{label}.telemetry"
            ),
        )


__all__ = [
    "CameraOutputSettings",
    "LoggingOutputSettings",
    "MirrorOutputsSettings",
    "RenderOutputSettings",
    "TelemetryOutputSettings",
    "TelemetryTopicSettings",
]


def _optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return as_string(value, label=label)


def _optional_port(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return as_int(value, label=label, minimum=1, maximum=65_535)
