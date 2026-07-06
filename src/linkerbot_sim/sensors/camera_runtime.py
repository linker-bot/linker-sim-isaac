"""Runtime helpers for Isaac sensor cameras.

Isaac/Omni imports are delayed until a camera is actually created, so config parsing and
ordinary unit tests do not require Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from linkerbot_sim.sensors.camera_config import (
    SceneSensorSettings,
    SensorCameraSettings,
)


ArrayFactory = Callable[[tuple[float, ...]], object]
QuatFactory = Callable[[tuple[float, float, float]], object]

_MODALITY_ATTACH_METHODS = {
    "rgb": "add_rgb_to_frame",
    "depth": "add_distance_to_image_plane_to_frame",
    "semantic_segmentation": "add_semantic_segmentation_to_frame",
    "instance_segmentation": "add_instance_segmentation_to_frame",
}


@dataclass(frozen=True)
class SensorCameraRuntime:
    """已创建的仿真传感器摄像机。"""

    settings: SensorCameraSettings
    camera: object

    @property
    def name(self) -> str:
        """配置中的摄像机名称。"""

        return self.settings.name

    @property
    def prim_path(self) -> str:
        """摄像机 USD prim path。"""

        return self.settings.prim_path

    def get_current_frame(self, *, clone: bool = False) -> object:
        """读取 Isaac Camera 当前帧。"""

        return self.camera.get_current_frame(clone=clone)

    def get_rgb(self, *, device: str | None = None) -> object:
        """读取 RGB 图像。"""

        self._require_modality("rgb")
        return self.camera.get_rgb(device=device)

    def get_depth(self, *, device: str | None = None) -> object:
        """读取 depth 图像；语义为 distance to image plane。"""

        self._require_modality("depth")
        return self.camera.get_depth(device=device)

    def get_intrinsics_matrix(self, *, device: str | None = None) -> object:
        """读取相机内参矩阵。"""

        return self.camera.get_intrinsics_matrix(device=device)

    def _require_modality(self, modality: str) -> None:
        if modality not in self.settings.modalities:
            raise ValueError(
                f"camera {self.name!r} was not configured with modality {modality!r}"
            )


def create_sensor_camera_runtimes(
    *,
    stage: object,
    sensors: SceneSensorSettings,
) -> tuple[SensorCameraRuntime, ...]:
    """为 enabled=true 的 camera 配置创建 Isaac Camera wrapper。"""

    if not sensors.enabled_cameras:
        return ()
    return tuple(
        create_sensor_camera_runtime(stage=stage, settings=settings)
        for settings in sensors.enabled_cameras
    )


def create_sensor_camera_runtime(
    *,
    stage: object,
    settings: SensorCameraSettings,
    camera_type: type | None = None,
    array_factory: ArrayFactory | None = None,
    quat_from_rpy: QuatFactory | None = None,
) -> SensorCameraRuntime:
    """创建单个 sensor camera runtime。

    ``camera_type``、``array_factory`` 和 ``quat_from_rpy`` 主要用于轻量测试注入；
    正常运行时会延迟导入 Isaac Camera、numpy 和旋转工具。
    """

    _validate_parent_prim(stage=stage, settings=settings)
    if camera_type is None or array_factory is None or quat_from_rpy is None:
        default_camera_type, default_array_factory, default_quat_from_rpy = (
            _load_camera_dependencies()
        )
        camera_type = camera_type or default_camera_type
        array_factory = array_factory or default_array_factory
        quat_from_rpy = quat_from_rpy or default_quat_from_rpy

    camera = camera_type(
        prim_path=settings.prim_path,
        name=settings.name,
        dt=1.0 / settings.frequency,
        resolution=settings.resolution,
        translation=array_factory(settings.pose_xyz),
        orientation=quat_from_rpy(settings.pose_rpy),
    )
    return SensorCameraRuntime(settings=settings, camera=camera)


def initialize_sensor_camera_runtimes(
    cameras: tuple[SensorCameraRuntime, ...],
) -> None:
    """在 ``world.reset()`` 后初始化 camera wrapper 并挂载 annotator。"""

    for runtime in cameras:
        camera = runtime.camera
        camera.initialize()
        near, far = runtime.settings.clipping_range
        camera.set_clipping_range(near_distance=near, far_distance=far)
        for modality in runtime.settings.modalities:
            attach_method_name = _MODALITY_ATTACH_METHODS[modality]
            getattr(camera, attach_method_name)()


def _load_camera_dependencies() -> tuple[type, ArrayFactory, QuatFactory]:
    """延迟加载 Isaac camera 依赖。"""

    import numpy as np
    from isaacsim.core.utils.numpy import rotations as rot_utils
    from isaacsim.sensors.camera import Camera

    def array_factory(values: tuple[float, ...]) -> object:
        return np.array(values, dtype=float)

    def quat_from_rpy(rpy: tuple[float, float, float]) -> object:
        return rot_utils.euler_angles_to_quats(np.array(rpy, dtype=float), degrees=False)

    return Camera, array_factory, quat_from_rpy


def _validate_parent_prim(*, stage: object, settings: SensorCameraSettings) -> None:
    """校验 parent_prim_path 指向当前 stage 中已有 prim。"""

    if settings.parent_prim_path is None:
        return
    prim = stage.GetPrimAtPath(settings.parent_prim_path)
    if prim is None or not _prim_is_valid(prim):
        raise ValueError(
            f"sensors.cameras.{settings.name}.parent_prim_path does not exist: "
            f"{settings.parent_prim_path}"
        )


def _prim_is_valid(prim: Any) -> bool:
    """兼容 USD Prim 和测试 fake prim 的有效性检查。"""

    is_valid = getattr(prim, "IsValid", None)
    if is_valid is None:
        return False
    return bool(is_valid())
