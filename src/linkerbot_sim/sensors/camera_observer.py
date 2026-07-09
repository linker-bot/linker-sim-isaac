"""Execution-layer camera frame observer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from linkerbot_sim.sensors.camera_frame import sample_camera_frames
from linkerbot_sim.sensors.camera_recorder import (
    CameraFramePublisher,
    CameraFrameSink,
    CompositeCameraFrameSink,
    OfflineCameraFrameSink,
)
from linkerbot_sim.sensors.camera_foxglove import FoxgloveCameraFrameSink
from linkerbot_sim.sensors.camera_runtime import SensorCameraRuntime


PathResolver = Callable[[str], Path]


class CameraFrameObserver:
    """按 camera frequency 在 world step 后采样。"""

    def __init__(
        self,
        *,
        cameras: Sequence[SensorCameraRuntime],
        publisher: CameraFramePublisher,
    ) -> None:
        """记录需要观察的 camera，并初始化每个 camera 的采样时间游标。"""

        self.cameras = tuple(cameras)
        self.publisher = publisher
        self._next_sample_time: dict[str, float] = {}
        self._frame_indices: dict[tuple[str, str], int] = {}

    def observe(self, world, *, step: int, phase: str | None = None) -> None:
        """在 physics step 后由执行层调用；phase 预留给后续 metadata 扩展。"""

        del phase
        time_s = (step + 1) * float(world.get_physics_dt())
        for camera in self.cameras:
            if not self._should_sample(camera, time_s):
                continue
            for frame in sample_camera_frames(
                camera,
                frame_indices=self._frame_indices,
                simulation_step=step,
                time_s=time_s,
            ):
                self.publisher.publish(frame)

    def _should_sample(self, camera: SensorCameraRuntime, time_s: float) -> bool:
        """按 camera.frequency 判断当前仿真时间是否应该采样。"""

        next_time = self._next_sample_time.get(camera.name)
        if next_time is not None and time_s + 1.0e-9 < next_time:
            return False
        self._next_sample_time[camera.name] = time_s + 1.0 / camera.settings.frequency
        return True

    def reset(self) -> None:
        """清理 reset 前的采样节奏和 frame index。"""

        self._next_sample_time.clear()
        self._frame_indices.clear()


@dataclass
class CameraOutputHandle:
    """Camera 输出运行时句柄。"""

    observer: CameraFrameObserver
    publisher: CameraFramePublisher

    def close(self) -> None:
        """关闭后台 publisher 和其持有的所有输出 sink。"""

        self.publisher.close()


def start_offline_camera_output(
    cameras: Sequence[SensorCameraRuntime],
    *,
    path_resolver: PathResolver | None = None,
) -> CameraOutputHandle | None:
    """根据 camera output.save_dir 启动离线保存。"""

    return _start_camera_output(
        cameras, path_resolver=path_resolver, include_foxglove=False
    )


def start_camera_output(
    cameras: Sequence[SensorCameraRuntime],
    *,
    path_resolver: PathResolver | None = None,
) -> CameraOutputHandle | None:
    """根据 camera output 配置启动离线和 Foxglove 输出。"""

    return _start_camera_output(
        cameras, path_resolver=path_resolver, include_foxglove=True
    )


def _start_camera_output(
    cameras: Sequence[SensorCameraRuntime],
    *,
    path_resolver: PathResolver | None,
    include_foxglove: bool,
) -> CameraOutputHandle | None:
    """启动 camera 输出。"""

    sinks: list[CameraFrameSink] = []
    observed_cameras: list[SensorCameraRuntime] = []
    live_groups: dict[tuple[str, int], dict[str, str]] = {}
    mcap_groups: dict[Path, dict[str, str]] = {}
    for camera in cameras:
        output = camera.settings.output
        save_dir = output.save_dir
        if save_dir is None:
            pass
        else:
            output_dir = (
                Path(save_dir) if path_resolver is None else path_resolver(save_dir)
            )
            sinks.append(
                OfflineCameraFrameSink(camera_name=camera.name, save_dir=output_dir)
            )
        topic_prefix = output.foxglove_topic_prefix or f"/cameras/{camera.name}"
        if include_foxglove and output.foxglove_live_port is not None:
            key = (output.foxglove_live_host, output.foxglove_live_port)
            live_groups.setdefault(key, {})[camera.name] = topic_prefix
        if include_foxglove and output.foxglove_mcap_path is not None:
            mcap_path = (
                Path(output.foxglove_mcap_path)
                if path_resolver is None
                else path_resolver(output.foxglove_mcap_path)
            )
            mcap_groups.setdefault(mcap_path, {})[camera.name] = topic_prefix
        if (
            save_dir is not None
            or (include_foxglove and output.foxglove_live_port is not None)
            or (include_foxglove and output.foxglove_mcap_path is not None)
        ):
            observed_cameras.append(camera)
    for (host, port), prefixes in live_groups.items():
        sinks.append(
            FoxgloveCameraFrameSink.open_live(
                host=host,
                port=port,
                topic_prefix_by_camera=prefixes,
            )
        )
    for mcap_path, prefixes in mcap_groups.items():
        sinks.append(
            FoxgloveCameraFrameSink.open_mcap(
                mcap_path,
                topic_prefix_by_camera=prefixes,
            )
        )
    if not sinks:
        return None
    sink: CameraFrameSink = (
        sinks[0] if len(sinks) == 1 else CompositeCameraFrameSink(sinks)
    )
    publisher = CameraFramePublisher(sink=sink, name="camera-offline-writer")
    observer = CameraFrameObserver(cameras=observed_cameras, publisher=publisher)
    publisher.start()
    return CameraOutputHandle(observer=observer, publisher=publisher)
