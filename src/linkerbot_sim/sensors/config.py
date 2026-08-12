"""Mirror resolved scene 投影出的场景级 typed 传感器配置。"""

from __future__ import annotations

from dataclasses import dataclass

from linkerbot_sim.sensors.camera.config import SensorCameraSettings


@dataclass(frozen=True)
class SceneSensorSettings:
    """Mirror assembly 直接构造的 sensor 配置集合。"""

    cameras: tuple[SensorCameraSettings, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cameras, tuple) or not all(
            isinstance(camera, SensorCameraSettings) for camera in self.cameras
        ):
            raise TypeError("cameras must be a tuple of SensorCameraSettings")
        names = tuple(camera.name for camera in self.cameras)
        if len(set(names)) != len(names):
            raise ValueError("camera names must be unique")

    @property
    def enabled_cameras(self) -> tuple[SensorCameraSettings, ...]:
        """返回 enabled=true 的摄像机配置。"""

        return tuple(camera for camera in self.cameras if camera.enabled)

    @property
    def has_output_consumers(self) -> bool:
        """返回是否有已启用摄像机会创建文件或 Foxglove 输出端。"""

        return any(camera.output.has_consumer for camera in self.enabled_cameras)

    def validate_mirror_camera_scope(self) -> None:
        """在构建 Mirror 场景前拒绝复制环境专用的 camera selector。

        Mirror 没有复制环境维度，接受 ``env_ids`` 会产生看似生效但实际被忽略的配置，
        因此在创建任何相机资源前直接报完整 YAML 路径。
        """

        for camera in self.cameras:
            if camera.env_ids is not None:
                raise ValueError(
                    f"sensors.cameras.{camera.name}.env_ids is only valid for "
                    "replicated environments"
                )
