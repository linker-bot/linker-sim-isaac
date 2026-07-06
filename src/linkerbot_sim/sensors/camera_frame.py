"""Camera frame containers and sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from linkerbot_sim.sensors.camera_runtime import SensorCameraRuntime


class CameraFrameNotReady(RuntimeError):
    """Camera annotator has not produced a usable frame yet."""


@dataclass(frozen=True)
class CameraFrame:
    """一帧已经从 Isaac 主线程采样出的 camera 数据。"""

    camera_name: str
    modality: str
    frame_index: int
    simulation_step: int
    time_s: float
    data: np.ndarray
    intrinsics: np.ndarray | None = None
    camera_position_world: tuple[float, float, float] | None = None
    camera_orientation_world: tuple[float, float, float, float] | None = None

    def metadata(self, *, relative_path: str | None = None) -> dict[str, object]:
        """返回不含图像 payload 的 JSON metadata。"""

        result: dict[str, object] = {
            "camera_name": self.camera_name,
            "modality": self.modality,
            "frame_index": self.frame_index,
            "simulation_step": self.simulation_step,
            "time_s": self.time_s,
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
        }
        if relative_path is not None:
            result["relative_path"] = relative_path
        if self.intrinsics is not None:
            result["intrinsics"] = self.intrinsics.tolist()
        if self.camera_position_world is not None:
            result["camera_position_world"] = list(self.camera_position_world)
        if self.camera_orientation_world is not None:
            result["camera_orientation_world"] = list(self.camera_orientation_world)
        return result


def sample_camera_frames(
    camera_runtime: SensorCameraRuntime,
    *,
    frame_indices: dict[tuple[str, str], int],
    simulation_step: int,
    time_s: float,
) -> tuple[CameraFrame, ...]:
    """从一个 sensor camera runtime 采样当前配置的 modalities。"""

    intrinsics = _optional_array(camera_runtime.get_intrinsics_matrix)
    position, orientation = _optional_world_pose(camera_runtime.camera)
    frames: list[CameraFrame] = []
    for modality in camera_runtime.settings.modalities:
        key = (camera_runtime.name, modality)
        try:
            data = _sample_modality(camera_runtime, modality)
        except CameraFrameNotReady:
            continue
        frame_index = frame_indices.get(key, 0)
        frame_indices[key] = frame_index + 1
        frames.append(
            CameraFrame(
                camera_name=camera_runtime.name,
                modality=modality,
                frame_index=frame_index,
                simulation_step=simulation_step,
                time_s=time_s,
                data=data,
                intrinsics=intrinsics,
                camera_position_world=position,
                camera_orientation_world=orientation,
            )
        )
    return tuple(frames)


def _sample_modality(camera_runtime: SensorCameraRuntime, modality: str) -> np.ndarray:
    """读取单个 modality 并规范化为可序列化数组。"""

    if modality == "rgb":
        return _rgb_uint8(camera_runtime.get_rgb(device="cpu"))
    if modality == "depth":
        return _depth_float32(camera_runtime.get_depth(device="cpu"))
    frame = camera_runtime.get_current_frame(clone=True)
    if isinstance(frame, dict) and modality in frame:
        array = np.asarray(frame[modality])
        if array.ndim == 0 or array.size == 0:
            raise CameraFrameNotReady(f"{modality} frame is not ready")
        return array
    raise CameraFrameNotReady(f"{modality} frame is not ready")


def _rgb_uint8(data: object) -> np.ndarray:
    """把 Isaac RGB 输出规范化为 HxWx3 uint8。"""

    array = np.asarray(data)
    if array.ndim == 0 or array.size == 0:
        raise CameraFrameNotReady("rgb frame is not ready")
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"rgb frame must have shape HxWx3/4, got {array.shape}")
    array = array[:, :, :3]
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    values = np.asarray(array, dtype=np.float32)
    if values.size and float(np.nanmax(values)) <= 1.0:
        values = values * 255.0
    return np.ascontiguousarray(np.clip(values, 0.0, 255.0).astype(np.uint8))


def _depth_float32(data: object) -> np.ndarray:
    """把 Isaac depth 输出规范化为 HxW float32。"""

    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 0 or array.size == 0:
        raise CameraFrameNotReady("depth frame is not ready")
    if array.ndim != 2:
        raise ValueError(f"depth frame must have shape HxW, got {array.shape}")
    return np.ascontiguousarray(array)


def _optional_array(callable_obj) -> np.ndarray | None:
    """调用可选数据读取方法；不可用时返回 None。"""

    try:
        return np.asarray(callable_obj(device="cpu"), dtype=float)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _optional_world_pose(camera: Any) -> tuple[
    tuple[float, float, float] | None,
    tuple[float, float, float, float] | None,
]:
    """读取 camera world pose；不可用时返回空 pose。"""

    get_world_pose = getattr(camera, "get_world_pose", None)
    if get_world_pose is None:
        return None, None
    try:
        position, orientation = get_world_pose()
    except (TypeError, ValueError, RuntimeError):
        return None, None
    return _tuple_or_none(position, 3), _tuple_or_none(orientation, 4)


def _tuple_or_none(value: object, expected_size: int) -> tuple[float, ...] | None:
    """把可选 pose 数组转为 tuple。"""

    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != expected_size:
        return None
    return tuple(float(item) for item in array)
