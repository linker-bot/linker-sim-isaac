"""从 env profile 解析场景级传感器配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from linkerbot_sim.sensors.camera.config import SensorCameraSettings


@dataclass(frozen=True)
class SceneSensorSettings:
    """env profile 中的 sensor 配置集合。"""

    cameras: tuple[SensorCameraSettings, ...] = ()

    @classmethod
    def from_env_config(cls, config: Mapping[str, object]) -> "SceneSensorSettings":
        """从完整 env profile 解析 sensors 顶层分组。"""

        if "sensors" not in config:
            return cls()
        sensors = config["sensors"]
        if sensors is None:
            raise ValueError("sensors must be a mapping")
        if not isinstance(sensors, Mapping):
            raise ValueError("sensors must be a mapping")
        _reject_keys(sensors, {"cameras"}, "sensors")
        if "cameras" not in sensors:
            cameras: object = {}
        else:
            cameras = sensors["cameras"]
        if not isinstance(cameras, Mapping):
            raise ValueError("sensors.cameras must be a mapping")
        return cls(
            cameras=tuple(
                SensorCameraSettings.from_mapping(
                    name, camera_data, label="sensors.cameras"
                )
                for name, camera_data in cameras.items()
            )
        )

    @property
    def enabled_cameras(self) -> tuple[SensorCameraSettings, ...]:
        """返回 enabled=true 的摄像机配置。"""

        return tuple(camera for camera in self.cameras if camera.enabled)

    @property
    def has_output_consumers(self) -> bool:
        """返回是否有已启用摄像机会创建文件或 Foxglove 输出端。"""

        return any(camera.output.has_consumer for camera in self.enabled_cameras)

    def validate_single_scene_camera_scope(self) -> None:
        """在构建 SingleSceneRuntime 前拒绝仅 TiledSceneRuntime 支持的 camera selector。

        非 tiled 场景没有 env 维度，接受 ``env_ids`` 会产生看似生效但实际被忽略的配置，
        因此在创建任何相机资源前直接报完整 YAML 路径。
        """

        for camera in self.cameras:
            if camera.env_ids is not None:
                raise ValueError(
                    f"sensors.cameras.{camera.name}.env_ids is only valid for "
                    "TiledSceneRuntime"
                )


def _reject_keys(data: Mapping[str, object], allowed: set[str], label: str) -> None:
    """拒绝未知 sensor 字段，并报告其完整 YAML 路径。"""

    unsupported = sorted(str(key) for key in data if key not in allowed)
    if unsupported:
        raise ValueError(f"{label}.{unsupported[0]} is not supported")
