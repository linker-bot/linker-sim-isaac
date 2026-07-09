"""Offline camera frame recording."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import Protocol

import numpy as np

from linkerbot_sim.sensors.camera_frame import CameraFrame


class CameraFrameSink(Protocol):
    """Camera frame 输出端协议。"""

    def publish(self, frame: CameraFrame) -> None:
        """发布一帧 camera 数据。"""

    def close(self) -> None:
        """关闭输出端。"""


class CompositeCameraFrameSink:
    """把同一帧 camera 数据发布到多个 sink。"""

    def __init__(self, sinks: Sequence[CameraFrameSink]) -> None:
        """保存输出端快照；后续 publish/close 按固定顺序广播。"""

        self.sinks = tuple(sinks)

    def publish(self, frame: CameraFrame) -> None:
        """把同一帧依次发布到所有子 sink。"""

        for sink in self.sinks:
            sink.publish(frame)

    def close(self) -> None:
        """依次关闭所有子 sink。"""

        for sink in self.sinks:
            sink.close()


class OfflineCameraFrameSink:
    """把 camera frame 写成离线文件序列。"""

    def __init__(self, *, camera_name: str, save_dir: str | Path) -> None:
        """创建单 camera 离线输出目录和 metadata.jsonl 文件。"""

        self.camera_name = camera_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.save_dir / "metadata.jsonl"
        self.metadata_file = self.metadata_path.open("a", encoding="utf-8")

    def publish(self, frame: CameraFrame) -> None:
        """写入匹配 camera 的一帧 payload，并追加一行 metadata。"""

        if frame.camera_name != self.camera_name:
            return
        relative_path = _write_frame_payload(self.save_dir, frame)
        self.metadata_file.write(
            json.dumps(
                frame.metadata(relative_path=relative_path),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        self.metadata_file.flush()

    def close(self) -> None:
        """关闭 metadata 文件句柄。"""

        self.metadata_file.close()


class CameraFramePublisher:
    """后台线程：从 bounded queue 消费 camera frame 并写入 sink。"""

    def __init__(
        self,
        *,
        sink: CameraFrameSink,
        name: str = "camera-frame-publisher",
        max_queue_size: int = 128,
    ) -> None:
        """创建 bounded queue publisher。

        queue 满时会丢弃旧帧保留新帧，避免磁盘或网络输出慢时阻塞仿真主线程。
        """

        self.sink = sink
        self.queue: Queue[CameraFrame | None] = Queue(maxsize=max_queue_size)
        self.name = name
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.last_error: Exception | None = None
        self.dropped_frames = 0

    def start(self) -> None:
        """启动后台发布线程；重复调用保持幂等。"""

        if self.thread is not None:
            return
        self.thread = Thread(target=self._run, name=self.name, daemon=True)
        self.thread.start()

    def publish(self, frame: CameraFrame) -> None:
        """从仿真线程提交一帧；队列满时尽量丢旧帧保新帧。"""

        if self.stop_event.is_set():
            return
        try:
            self.queue.put_nowait(frame)
        except Full:
            self.dropped_frames += 1
            try:
                self.queue.get_nowait()
            except Exception:
                pass
            try:
                self.queue.put_nowait(frame)
            except Full:
                self.dropped_frames += 1

    def close(self) -> None:
        """请求后台线程退出，等待短时间后关闭底层 sink。"""

        self.stop_event.set()
        try:
            self.queue.put(None, timeout=2.0)
        except Full:
            pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        self.sink.close()

    def _run(self) -> None:
        """后台线程主循环：消费队列并把异常记录到 ``last_error``。"""

        while True:
            frame = self.queue.get()
            if frame is None:
                break
            try:
                self.sink.publish(frame)
            except Exception as exc:
                self.last_error = exc
                self.stop_event.set()
                print(f"CAMERA_FRAME_PUBLISHER_FAILED {type(exc).__name__}: {exc}", flush=True)


def _write_frame_payload(save_dir: Path, frame: CameraFrame) -> str:
    """写单帧 payload 并返回相对路径。"""

    if frame.modality == "rgb":
        relative_path = Path("rgb") / f"{frame.frame_index:06d}.ppm"
        _write_ppm(save_dir / relative_path, frame.data)
        return relative_path.as_posix()
    if frame.modality == "depth":
        relative_path = Path("depth") / f"{frame.frame_index:06d}.npy"
        output_path = save_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, np.asarray(frame.data, dtype=np.float32))
        return relative_path.as_posix()
    relative_path = Path(frame.modality) / f"{frame.frame_index:06d}.npy"
    output_path = save_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(frame.data))
    return relative_path.as_posix()


def _write_ppm(path: Path, rgb: np.ndarray) -> None:
    """写 binary PPM，避免为 RGB 帧引入额外图像依赖。"""

    data = np.asarray(rgb, dtype=np.uint8)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"rgb PPM data must have shape HxWx3, got {data.shape}")
    height, width, _channels = data.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        stream.write(np.ascontiguousarray(data).tobytes())
