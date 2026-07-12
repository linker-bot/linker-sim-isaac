"""Tiled telemetry 的纯配置模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from linkerbot_sim.telemetry.foxglove import FoxgloveTopicConfig
from linkerbot_sim.utils.output_paths import validate_existing_data_policy


@dataclass(frozen=True)
class TiledTelemetryConfig:
    """选择 telemetry env、降采样频率、topic 与输出模态。"""

    selected_env_ids: tuple[int, ...]
    primary_env_id: int
    publish_decimation: int = 1
    include_full_batch_json: bool = True
    include_standard_joint_states: bool = True
    include_scene_markers: bool = True
    include_efforts: bool = False
    include_objects: bool = True
    topics: FoxgloveTopicConfig = field(default_factory=FoxgloveTopicConfig)
    buffer_size: int = 1
    drop_policy: str = "latest"
    on_error: str = "stop"
    shutdown_timeout_s: float = 2.0
    mcap_existing_file_policy: str = "error"

    def __post_init__(self) -> None:
        if not self.selected_env_ids:
            raise ValueError("selected_env_ids cannot be empty")
        if any(
            isinstance(env_id, bool) or not isinstance(env_id, int)
            for env_id in self.selected_env_ids
        ):
            raise ValueError("selected_env_ids must contain integers")
        if len(set(self.selected_env_ids)) != len(self.selected_env_ids):
            raise ValueError("selected_env_ids cannot contain duplicates")
        if any(env_id < 0 for env_id in self.selected_env_ids):
            raise ValueError("selected_env_ids cannot contain negative values")
        if isinstance(self.primary_env_id, bool) or not isinstance(
            self.primary_env_id, int
        ):
            raise ValueError("primary_env_id must be an integer")
        if self.primary_env_id not in self.selected_env_ids:
            raise ValueError("primary_env_id must be included in selected_env_ids")
        if (
            isinstance(self.publish_decimation, bool)
            or not isinstance(self.publish_decimation, int)
            or self.publish_decimation < 1
        ):
            raise ValueError("publish_decimation must be a positive integer")
        for label, value in (
            ("include_full_batch_json", self.include_full_batch_json),
            ("include_standard_joint_states", self.include_standard_joint_states),
            ("include_scene_markers", self.include_scene_markers),
            ("include_efforts", self.include_efforts),
            ("include_objects", self.include_objects),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be a boolean")
        if not isinstance(self.topics, FoxgloveTopicConfig):
            raise ValueError("topics must be FoxgloveTopicConfig")
        if (
            isinstance(self.buffer_size, bool)
            or not isinstance(self.buffer_size, int)
            or self.buffer_size < 1
        ):
            raise ValueError("buffer_size must be a positive integer")
        if self.drop_policy not in {"latest", "drop_oldest", "drop_newest"}:
            raise ValueError(
                "drop_policy must be one of: latest, drop_oldest, drop_newest"
            )
        if self.on_error not in {"stop", "continue"}:
            raise ValueError("on_error must be one of: stop, continue")
        if (
            isinstance(self.shutdown_timeout_s, bool)
            or not isinstance(self.shutdown_timeout_s, (int, float))
            or not isfinite(float(self.shutdown_timeout_s))
            or float(self.shutdown_timeout_s) < 0.0
        ):
            raise ValueError("shutdown_timeout_s must be a non-negative finite number")
        validate_existing_data_policy(
            self.mcap_existing_file_policy,
            label="mcap_existing_file_policy",
        )


def parse_env_ids(value: str) -> tuple[int, ...]:
    """解析 CLI 逗号分隔 env IDs；空 token 跳过，整体为空则拒绝。"""

    if not value.strip():
        raise ValueError("env id list cannot be empty")
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


__all__ = ["TiledTelemetryConfig", "parse_env_ids"]
