"""Foxglove camera frame sinks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from linkerbot_sim.sensors.camera_frame import CameraFrame
from linkerbot_sim.telemetry.foxglove import (
    _load_foxglove,
    _load_foxglove_channels,
    _ns_time,
    _timestamp,
)


class FoxgloveCameraFrameSink:
    """把 camera frames 写到 Foxglove live server 或 MCAP。"""

    def __init__(self, sink, *, topic_prefix_by_camera: Mapping[str, str]) -> None:
        self.foxglove, self.messages = _load_foxglove()
        self.channels = _load_foxglove_channels()
        self.sink = sink
        self.topic_prefix_by_camera = dict(topic_prefix_by_camera)
        self.image_channels = {}
        self.info_channels = {}

    @classmethod
    def open_live(
        cls,
        *,
        host: str,
        port: int,
        topic_prefix_by_camera: Mapping[str, str],
        name: str = "linkerbot-sim-camera",
    ) -> "FoxgloveCameraFrameSink":
        """打开 Foxglove live server camera 输出。"""

        foxglove, _messages = _load_foxglove()
        return cls(
            foxglove.start_server(name=name, host=host, port=int(port)),
            topic_prefix_by_camera=topic_prefix_by_camera,
        )

    @classmethod
    def open_mcap(
        cls,
        path: str | Path,
        *,
        topic_prefix_by_camera: Mapping[str, str],
        allow_overwrite: bool = True,
    ) -> "FoxgloveCameraFrameSink":
        """打开 Foxglove MCAP camera 输出。"""

        foxglove, _messages = _load_foxglove()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            foxglove.open_mcap(output_path, allow_overwrite=allow_overwrite),
            topic_prefix_by_camera=topic_prefix_by_camera,
        )

    def publish(self, frame: CameraFrame) -> None:
        prefix = self.topic_prefix_by_camera.get(frame.camera_name)
        if prefix is None:
            return
        if frame.modality in {"rgb", "depth"}:
            self._image_channel(prefix, frame.modality).log(
                _raw_image(frame, self.messages),
                log_time=_ns_time(frame.time_s),
            )
        self._info_channel(prefix).log(
            frame.metadata(), log_time=_ns_time(frame.time_s)
        )

    def close(self) -> None:
        close = getattr(self.sink, "close", None)
        if close is not None:
            close()

    def _image_channel(self, prefix: str, modality: str):
        key = (prefix, modality)
        channel = self.image_channels.get(key)
        if channel is None:
            channel = self.channels.RawImageChannel(_topic(prefix, modality))
            self.image_channels[key] = channel
        return channel

    def _info_channel(self, prefix: str):
        channel = self.info_channels.get(prefix)
        if channel is None:
            channel = self.foxglove.Channel(
                _topic(prefix, "info"),
                message_encoding="json",
            )
            self.info_channels[prefix] = channel
        return channel


def _raw_image(frame: CameraFrame, messages):
    """把 CameraFrame 转成 Foxglove RawImage。"""

    data, encoding, step, width, height = _raw_image_payload(frame)
    return messages.RawImage(
        timestamp=_timestamp(frame.time_s, messages),
        frame_id=frame.camera_name,
        width=width,
        height=height,
        encoding=encoding,
        step=step,
        data=data,
    )


def _raw_image_payload(frame: CameraFrame) -> tuple[bytes, str, int, int, int]:
    """返回 RawImage payload、encoding、row step、width 和 height。"""

    if frame.modality == "rgb":
        array = np.asarray(frame.data, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"rgb RawImage must have shape HxWx3, got {array.shape}")
        height, width, _channels = array.shape
        return np.ascontiguousarray(array).tobytes(), "rgb8", width * 3, width, height
    if frame.modality == "depth":
        array = np.asarray(frame.data, dtype="<f4")
        if array.ndim != 2:
            raise ValueError(f"depth RawImage must have shape HxW, got {array.shape}")
        height, width = array.shape
        return np.ascontiguousarray(array).tobytes(), "32FC1", width * 4, width, height
    raise ValueError(f"unsupported Foxglove camera modality {frame.modality!r}")


def _topic(prefix: str, suffix: str) -> str:
    """拼接 Foxglove topic。"""

    return prefix.rstrip("/") + "/" + suffix.lstrip("/")

