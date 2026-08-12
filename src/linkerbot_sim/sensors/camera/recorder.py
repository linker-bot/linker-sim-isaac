"""相机帧的离线文件记录、metadata 索引和有界后台发布。

每个离线相机独占一个输出目录：图像/深度 payload 放在 modality 子目录，
``metadata.jsonl`` 是已提交帧的索引。单帧采用“exclusive 创建 payload -> 校验大小与
配额 -> 追加完整 metadata 行 -> 推进内存计数”的提交顺序。metadata 追加失败时会截断
回原 offset 并删除未索引 payload；进程在两步之间异常退出留下的孤立 payload 则由
``resume`` 扫描保留其 index，避免后续覆盖。

目录配额按实际常规文件字节统计，包含 payload 与 metadata。后台 publisher 使用有界
queue 将渲染/physics step 与磁盘或网络 I/O 解耦，队列饱和和关闭行为必须由配置显式
选择，避免离线数据任务在无提示情况下丢帧。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace
import math
from pathlib import Path
from queue import Empty, Full, Queue
import struct
from threading import Event, Thread
import time
from typing import BinaryIO, Protocol
import zlib

import numpy as np

from linkerbot_sim.utils.output_paths import (
    OutputPathPlan,
    apply_output_path_plans,
    plan_output_directory,
)
from linkerbot_sim.utils.json import strict_json_dumps, strict_json_loads

from .frame import CameraFrame
from .limits import DEFAULT_MAX_BYTES_PER_CAMERA


RGB_FORMATS = frozenset({"ppm", "png", "npy"})
DEPTH_FORMATS = frozenset({"npy", "npz"})


class CameraFrameSink(Protocol):
    """Camera frame 输出端协议。"""

    def publish(self, frame: CameraFrame) -> None:
        """发布一帧 camera 数据。"""

    def close(self) -> None:
        """关闭输出端。"""


class CompositeCameraFrameSink:
    """把同一帧相机数据按固定顺序发布到多个 sink。

    ``publish`` 不是跨 sink 原子事务：较早 sink 成功而后续 sink 失败时不会反向撤销。
    ``close`` 则会继续尝试所有未关闭 sink，并在完成清理后重抛首个异常。
    """

    def __init__(self, sinks: Sequence[CameraFrameSink]) -> None:
        """保存输出端快照；后续 publish/close 按固定顺序广播。"""

        self.sinks = tuple(sinks)
        self._closed_sink_indices: set[int] = set()

    def publish(self, frame: CameraFrame) -> None:
        """把同一帧依次发布到所有子 sink。"""

        for sink in self.sinks:
            sink.publish(frame)

    def close(self) -> None:
        """尽力关闭所有子 sink，并在最后重抛第一个关闭异常。"""

        first_error: BaseException | None = None
        for index, sink in enumerate(self.sinks):
            if index in self._closed_sink_indices:
                continue
            try:
                sink.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._closed_sink_indices.add(index)
        if first_error is not None:
            raise first_error

    def status(self) -> dict[str, object]:
        """汇总实现了 ``status`` 的子 sink 诊断信息。"""

        children = []
        for sink in self.sinks:
            callback = getattr(sink, "status", None)
            if callable(callback):
                children.append(dict(callback()))
        return {"type": "composite", "children": children}


@dataclass(frozen=True)
class OfflineCameraFrameSinkPlan:
    """已校验但尚未打开文件的单相机离线输出计划。

    plan 固化路径策略、恢复后的各 modality 下一个 index、格式和字节预算。把预检结果与
    打开文件分离，允许上层先验证全部相机/MCAP 路径，再统一执行会修改文件系统的步骤。
    """

    camera_name: str
    path_plan: OutputPathPlan
    next_frame_indices: Mapping[str, int]
    rgb_format: str
    depth_format: str
    metadata_flush_interval_frames: int
    max_bytes_per_camera: int
    used_bytes: int


class OfflineCameraFrameSink:
    """把单个相机的 frame 写成 payload 文件序列和 JSONL 索引。"""

    def __init__(
        self,
        *,
        camera_name: str,
        save_dir: str | Path,
        existing_data_policy: str = "error",
        timestamped_run_name: str | None = None,
        rgb_format: str = "ppm",
        depth_format: str = "npy",
        metadata_flush_interval_frames: int = 1,
        max_bytes_per_camera: int = DEFAULT_MAX_BYTES_PER_CAMERA,
    ) -> None:
        """预检、应用路径策略并打开单相机离线输出 namespace。

        多输出启动应优先使用 ``prepare``/``open_prepared`` 两阶段 API；该构造器用于只有
        一个目录的调用方，会立即执行 truncate/create 等路径变更。
        """

        plan = self.prepare(
            camera_name=camera_name,
            save_dir=save_dir,
            existing_data_policy=existing_data_policy,
            timestamped_run_name=timestamped_run_name,
            rgb_format=rgb_format,
            depth_format=depth_format,
            metadata_flush_interval_frames=metadata_flush_interval_frames,
            max_bytes_per_camera=max_bytes_per_camera,
        )
        apply_output_path_plans((plan.path_plan,))
        self._initialize_from_plan(plan)

    @classmethod
    def prepare(
        cls,
        *,
        camera_name: str,
        save_dir: str | Path,
        existing_data_policy: str,
        timestamped_run_name: str | None,
        rgb_format: str,
        depth_format: str,
        metadata_flush_interval_frames: int,
        max_bytes_per_camera: int,
    ) -> OfflineCameraFrameSinkPlan:
        """只读预检一个输出 namespace，不创建目录或打开文件。

        ``resume`` 会校验 metadata、文件格式和最新 payload，并扫描目录当前总字节数；
        其它策略只规划目标。返回成功意味着输入在预检时刻可用，但真正打开前仍需由路径
        plan 的批量应用阶段重新校验，缩小 TOCTOU 窗口。
        """

        del cls
        rgb = _output_format(rgb_format, RGB_FORMATS, label="rgb_format")
        depth = _output_format(depth_format, DEPTH_FORMATS, label="depth_format")
        flush_interval = _positive_int(
            metadata_flush_interval_frames,
            label="metadata_flush_interval_frames",
        )
        byte_limit = _positive_int(
            max_bytes_per_camera,
            label="max_bytes_per_camera",
        )
        path_plan = plan_output_directory(
            save_dir,
            policy=existing_data_policy,
            run_name=timestamped_run_name,
        )
        next_indices: dict[str, int] = {}
        used_bytes = 0
        if path_plan.policy == "resume" and path_plan.existed_at_preflight:
            used_bytes = _regular_file_bytes(path_plan.resolved_path)
            if used_bytes > byte_limit:
                raise CameraOutputQuotaExceededError(
                    camera_name=camera_name,
                    used_bytes=used_bytes,
                    max_bytes=byte_limit,
                )
            metadata_path = path_plan.resolved_path / "metadata.jsonl"
            next_indices = _resume_frame_indices(
                path_plan.resolved_path,
                metadata_path,
                camera_name=camera_name,
                rgb_format=rgb,
                depth_format=depth,
            )
        return OfflineCameraFrameSinkPlan(
            camera_name=camera_name,
            path_plan=path_plan,
            next_frame_indices=next_indices,
            rgb_format=rgb,
            depth_format=depth,
            metadata_flush_interval_frames=flush_interval,
            max_bytes_per_camera=byte_limit,
            used_bytes=used_bytes,
        )

    @classmethod
    def open_prepared(
        cls,
        plan: OfflineCameraFrameSinkPlan,
    ) -> OfflineCameraFrameSink:
        """在完整启动批次的路径计划已应用后打开一个预检 plan。"""

        sink = cls.__new__(cls)
        sink._initialize_from_plan(plan)
        return sink

    def _initialize_from_plan(self, plan: OfflineCameraFrameSinkPlan) -> None:
        """从不可变 plan 初始化计数器，并以 append/exclusive 模式打开索引。"""

        self.camera_name = plan.camera_name
        self.save_dir = plan.path_plan.resolved_path
        self.metadata_path = self.save_dir / "metadata.jsonl"
        self.next_frame_indices = dict(plan.next_frame_indices)
        self.rgb_format = plan.rgb_format
        self.depth_format = plan.depth_format
        self.metadata_flush_interval_frames = plan.metadata_flush_interval_frames
        self.max_bytes_per_camera = plan.max_bytes_per_camera
        self.used_bytes = plan.used_bytes
        self._metadata_rows_since_flush = 0
        _reject_symlink(self.save_dir, label="camera output directory")
        _reject_symlink(self.metadata_path, label="camera metadata path")
        mode = "ab" if self.metadata_path.exists() else "xb"
        self.metadata_file = self.metadata_path.open(mode)

    def publish(self, frame: CameraFrame) -> None:
        """提交匹配相机的一帧 payload，并追加一行 metadata。

        不匹配 ``camera_name`` 的帧直接忽略，便于同一 composite sink 接收共享 frame 流。
        同一相机内，payload 只有在 metadata 完整追加后才视为已提交；index、配额计数和
        flush 行计数也只在该提交点之后推进。
        """

        if frame.camera_name != self.camera_name:
            return
        if self.used_bytes >= self.max_bytes_per_camera:
            raise CameraOutputQuotaExceededError(
                camera_name=self.camera_name,
                used_bytes=self.used_bytes,
                max_bytes=self.max_bytes_per_camera,
            )
        modality = _safe_modality(frame.modality)
        frame_index = max(
            _safe_frame_index(frame.frame_index, label="camera frame_index"),
            self.next_frame_indices.get(modality, 0),
        )
        # 使用 ``xb`` exclusive 创建解决 resume 后或外部并发新增文件造成的 index 冲突；
        # 冲突时只递增 index，绝不覆盖已有 payload。
        while True:
            stored_frame = replace(frame, modality=modality, frame_index=frame_index)
            relative_path = _payload_relative_path(
                modality,
                frame_index,
                rgb_format=self.rgb_format,
                depth_format=self.depth_format,
            ).as_posix()
            metadata_record = (
                strict_json_dumps(
                    stored_frame.metadata(relative_path=relative_path),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            payload_path = self.save_dir / relative_path
            try:
                written_path = _write_frame_payload(
                    self.save_dir,
                    stored_frame,
                    rgb_format=self.rgb_format,
                    depth_format=self.depth_format,
                )
            except FileExistsError:
                frame_index += 1
                continue
            except BaseException as exc:
                _remove_unindexed_payload(
                    payload_path,
                    cause=exc,
                    context="camera payload write failed",
                )
                raise
            try:
                if written_path != relative_path:
                    raise RuntimeError("camera payload path planning mismatch")
                payload_bytes = payload_path.stat().st_size
            except BaseException as exc:
                _remove_unindexed_payload(
                    payload_path,
                    cause=exc,
                    context="camera payload validation failed",
                )
                raise
            break
        # 配额同时计入 payload 与其索引行。先写 payload 才能获得编码后的真实大小；若预算
        # 不足会立即删除这个尚未被 metadata 引用的文件。
        metadata_bytes = len(metadata_record)
        projected_bytes = self.used_bytes + payload_bytes + metadata_bytes
        if projected_bytes > self.max_bytes_per_camera:
            quota_error = CameraOutputQuotaExceededError(
                camera_name=self.camera_name,
                used_bytes=self.used_bytes,
                max_bytes=self.max_bytes_per_camera,
                attempted_bytes=payload_bytes + metadata_bytes,
            )
            _remove_unindexed_payload(
                payload_path,
                cause=quota_error,
                context="camera output quota was exceeded",
            )
            raise quota_error
        # metadata append 是帧提交点。失败时回退到精确 offset，并删除对应 payload，保持
        # “每一条索引都有文件、每个正常返回的 publish 都有索引”的进程内一致性。
        metadata_offset = self.metadata_file.tell()
        flush_metadata = (
            self._metadata_rows_since_flush + 1 >= self.metadata_flush_interval_frames
        )
        try:
            written = self.metadata_file.write(metadata_record)
            if written != metadata_bytes:
                raise OSError(
                    f"camera metadata short write: {written}/{metadata_bytes} bytes"
                )
            if flush_metadata:
                self.metadata_file.flush()
        except BaseException as exc:
            _rollback_uncommitted_frame(
                metadata_file=self.metadata_file,
                metadata_offset=metadata_offset,
                payload_path=payload_path,
                cause=exc,
            )
            raise
        self.next_frame_indices[modality] = frame_index + 1
        self.used_bytes = projected_bytes
        if flush_metadata:
            self._metadata_rows_since_flush = 0
        else:
            self._metadata_rows_since_flush += 1

    def close(self) -> None:
        """关闭 metadata 文件句柄。"""

        self.metadata_file.close()

    def status(self) -> dict[str, object]:
        """返回离线目录的已用、上限和剩余字节，供运行监控。"""

        return {
            "type": "offline_camera",
            "camera_name": self.camera_name,
            "save_dir": str(self.save_dir),
            "used_bytes": self.used_bytes,
            "max_bytes": self.max_bytes_per_camera,
            "remaining_bytes": self.max_bytes_per_camera - self.used_bytes,
        }


class CameraFramePublisher:
    """从有界 queue 消费相机帧并写入 sink 的后台线程。

    producer 运行在仿真采样路径，consumer 承担磁盘/网络 I/O。``overflow_policy`` 决定
    背压、丢帧或立即报错；离线输出由上层限制为 ``block``/``error``，以维持数据完整性。
    worker 首次失败会保存 ``last_error``、停止接收并清空待处理帧。
    """

    def __init__(
        self,
        *,
        sink: CameraFrameSink,
        name: str = "camera-frame-publisher",
        max_queue_size: int = 128,
        overflow_policy: str = "block",
        worker_poll_interval_s: float = 0.1,
        shutdown_policy: str = "drain",
        shutdown_timeout_s: float = 2.0,
    ) -> None:
        """创建有界 queue publisher，但不启动线程。

        queue 饱和策略和 shutdown 行为是显式 runtime policy；所有数值会在创建线程或打开
        sink 之前校验，避免配置错误留下部分启动的输出资源。
        """

        timeout = validate_camera_frame_publisher_settings(
            max_queue_size=max_queue_size,
            overflow_policy=overflow_policy,
            worker_poll_interval_s=worker_poll_interval_s,
            shutdown_policy=shutdown_policy,
            shutdown_timeout_s=shutdown_timeout_s,
        )
        poll_interval = float(worker_poll_interval_s)
        self.sink = sink
        self.queue: Queue[CameraFrame | None] = Queue(maxsize=int(max_queue_size))
        self.name = name
        self.overflow_policy = overflow_policy
        self.worker_poll_interval_s = poll_interval
        self.shutdown_policy = shutdown_policy
        self.shutdown_timeout_s = timeout
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.last_error: Exception | None = None
        self.dropped_frames = 0
        self.aborted_frames = 0
        self.overflow_errors = 0
        self.published_frames = 0
        self.shutdown_timed_out = False
        self._sink_closed = False

    def start(self) -> None:
        """启动后台发布线程；重复调用保持幂等。"""

        if self.thread is not None:
            return
        thread = Thread(target=self._run, name=self.name, daemon=True)
        thread.start()
        self.thread = thread

    def publish(self, frame: CameraFrame) -> None:
        """按照配置的饱和策略提交一帧。

        ``block`` 以短 timeout 轮询，确保等待 queue 空位时仍能及时观察 worker 失败或关闭
        请求；丢帧策略更新诊断计数；``error`` 用专用异常让调用方立即感知数据未入队。
        """

        if self.stop_event.is_set():
            self._raise_if_failed()
            raise RuntimeError("camera frame publisher is closing")
        if self.overflow_policy == "block":
            while not self.stop_event.is_set():
                try:
                    self.queue.put(frame, timeout=self.worker_poll_interval_s)
                    return
                except Full:
                    self._raise_if_failed()
            self._raise_if_failed()
            raise RuntimeError("camera frame publisher stopped while enqueueing")
        try:
            self.queue.put_nowait(frame)
            return
        except Full:
            pass
        if self.overflow_policy == "drop_newest":
            self.dropped_frames += 1
            return
        if self.overflow_policy == "error":
            self.overflow_errors += 1
            raise CameraFrameQueueFullError(
                f"camera output queue is full (capacity={self.queue.maxsize})"
            )
        discarded = False
        try:
            self.queue.get_nowait()
            discarded = True
        except Empty:
            pass
        if discarded:
            self.dropped_frames += 1
        try:
            self.queue.put_nowait(frame)
        except Full:
            self.dropped_frames += 1

    def _raise_if_failed(self) -> None:
        """在 producer 线程重抛 worker 故障，并保留原异常链。"""

        if self.last_error is not None:
            raise RuntimeError(
                "camera frame publisher worker failed"
            ) from self.last_error

    def _discard_queued_frames(self) -> int:
        """清空尚未消费的 queue，并把数量计入主动中止帧。"""

        discarded = 0
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
            discarded += 1
        self.aborted_frames += discarded
        return discarded

    def close(self, *, timeout_s: float | None = None) -> bool:
        """在有界时间内停止后台线程并关闭 sink。

        ``drain`` 会先消费完已接收帧，``abort`` 会立即丢弃队列。超时返回 ``False``，同时
        保留仍存活的线程和未关闭 sink，避免主线程与 worker 并发关闭同一资源；调用方可
        通过 ``status`` 诊断或稍后再次关闭。
        """

        timeout = (
            self.shutdown_timeout_s
            if timeout_s is None
            else _nonnegative_finite_float(timeout_s, label="timeout_s")
        )
        self.shutdown_timed_out = False
        self.stop_event.set()
        if self.shutdown_policy == "discard":
            self._discard_queued_frames()
        deadline = time.monotonic() + timeout
        thread = self.thread
        if (
            thread is None
            and self.shutdown_policy == "drain"
            and not self.queue.empty()
        ):
            self.start()
            thread = self.thread
        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                self.shutdown_timed_out = True
                return False
            self.thread = None
        if not self._sink_closed:
            self.sink.close()
            self._sink_closed = True
        return True

    def status(self) -> dict[str, object]:
        """返回 queue、丢帧、异常和 shutdown 状态。"""

        thread = self.thread
        result = {
            "name": self.name,
            "queue_depth": self.queue.qsize(),
            "queue_capacity": self.queue.maxsize,
            "overflow_policy": self.overflow_policy,
            "worker_poll_interval_s": self.worker_poll_interval_s,
            "shutdown_policy": self.shutdown_policy,
            "dropped_frames": self.dropped_frames,
            "aborted_frames": self.aborted_frames,
            "overflow_errors": self.overflow_errors,
            "published_frames": self.published_frames,
            "thread_alive": thread is not None and thread.is_alive(),
            "shutdown_requested": self.stop_event.is_set(),
            "shutdown_timed_out": self.shutdown_timed_out,
            "sink_closed": self._sink_closed,
            "last_error": (
                None
                if self.last_error is None
                else f"{type(self.last_error).__name__}: {self.last_error}"
            ),
        }
        sink_status = getattr(self.sink, "status", None)
        if callable(sink_status):
            result["sink"] = dict(sink_status())
        return result

    def _run(self) -> None:
        """后台线程主循环：消费队列并把异常记录到 ``last_error``。"""

        while not (
            self.stop_event.is_set()
            and (self.shutdown_policy == "discard" or self.queue.empty())
        ):
            try:
                frame = self.queue.get(timeout=self.worker_poll_interval_s)
            except Empty:
                continue
            if frame is None:
                continue
            try:
                self.sink.publish(frame)
                self.published_frames += 1
            except Exception as exc:
                self.last_error = exc
                self.stop_event.set()
                self._discard_queued_frames()
                print(
                    f"CAMERA_FRAME_PUBLISHER_FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )
                break


class CameraFrameQueueFullError(RuntimeError):
    """fail-fast 相机输出无法接纳新帧时抛出。"""


class CameraOutputQuotaExceededError(RuntimeError):
    """离线相机 namespace 即将超过字节配额时、提交前抛出。"""

    def __init__(
        self,
        *,
        camera_name: str,
        used_bytes: int,
        max_bytes: int,
        attempted_bytes: int = 0,
    ) -> None:
        detail = (
            f"camera {camera_name!r} output quota exceeded: "
            f"used={used_bytes} attempted={attempted_bytes} max={max_bytes} bytes"
        )
        super().__init__(detail)


def _remove_unindexed_payload(
    payload_path: Path,
    *,
    cause: BaseException,
    context: str,
) -> None:
    """删除尚未获得已提交 metadata 记录的 payload。

    若清理也失败，会以清理异常包装原始 ``cause``，因为此时目录中已存在需要人工处理的
    孤立文件，不能只报告最初的写入/配额错误。
    """

    try:
        payload_path.unlink(missing_ok=True)
    except OSError as cleanup_error:
        raise RuntimeError(
            f"{context}; unindexed payload cleanup failed for {payload_path}: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        ) from cause


def _rollback_uncommitted_frame(
    *,
    metadata_file: BinaryIO,
    metadata_offset: int,
    payload_path: Path,
    cause: BaseException,
) -> None:
    """把 metadata 截断回 append 起点，并删除对应的未索引 payload。

    两个补偿动作都会尝试；任一失败都汇总为“不完整回滚”，避免调用方误认为该帧完全没有
    落盘痕迹。
    """

    rollback_errors: list[BaseException] = []
    try:
        metadata_file.seek(metadata_offset)
        metadata_file.truncate()
        metadata_file.flush()
    except BaseException as exc:
        rollback_errors.append(exc)
    try:
        payload_path.unlink(missing_ok=True)
    except BaseException as exc:
        rollback_errors.append(exc)
    if rollback_errors:
        details = "; ".join(
            f"{type(error).__name__}: {error}" for error in rollback_errors
        )
        raise RuntimeError(
            "camera metadata commit failed and frame rollback was incomplete: "
            f"{details}"
        ) from cause


def _regular_file_bytes(root: Path) -> int:
    """统计 resume 目录下常规文件总字节数，不跟随任何符号链接。

    metadata、已索引 payload 和崩溃遗留的孤立 payload 都计入预算；这样 resume 不会把
    既有磁盘占用当成零而突破每相机上限。
    """

    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        _reject_symlink(directory, label="camera output directory")
        for entry in directory.iterdir():
            _reject_symlink(entry, label="camera output entry")
            if entry.is_dir():
                pending.append(entry)
            elif entry.is_file():
                total += entry.stat().st_size
    return total


def validate_camera_frame_publisher_settings(
    *,
    max_queue_size: object,
    overflow_policy: object,
    worker_poll_interval_s: object,
    shutdown_policy: object,
    shutdown_timeout_s: object,
) -> float:
    """在任何输出路径被修改前校验 publisher 策略与数值边界。"""

    if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int):
        raise ValueError("max_queue_size must be a positive integer")
    if max_queue_size < 1:
        raise ValueError("max_queue_size must be a positive integer")
    if not isinstance(overflow_policy, str) or overflow_policy not in {
        "drop_oldest",
        "drop_newest",
        "block",
        "error",
    }:
        raise ValueError(
            "overflow_policy must be drop_oldest, drop_newest, block, or error"
        )
    if not isinstance(shutdown_policy, str) or shutdown_policy not in {
        "drain",
        "discard",
    }:
        raise ValueError("shutdown_policy must be drain or discard")
    _positive_finite_float(
        worker_poll_interval_s,
        label="worker_poll_interval_s",
    )
    return _nonnegative_finite_float(
        shutdown_timeout_s,
        label="shutdown_timeout_s",
    )


def _write_frame_payload(
    save_dir: Path,
    frame: CameraFrame,
    *,
    rgb_format: str,
    depth_format: str,
) -> str:
    """写单帧 payload 并返回相对路径。"""

    relative_path = _payload_relative_path(
        frame.modality,
        frame.frame_index,
        rgb_format=rgb_format,
        depth_format=depth_format,
    )
    _prepare_payload_parent(save_dir / relative_path)
    if frame.modality == "rgb" and rgb_format == "ppm":
        _write_ppm(save_dir / relative_path, frame.data)
        return relative_path.as_posix()
    if frame.modality == "rgb" and rgb_format == "png":
        _write_png(save_dir / relative_path, frame.data)
        return relative_path.as_posix()
    output_path = save_dir / relative_path
    data = (
        np.asarray(frame.data, dtype=np.float32)
        if frame.modality == "depth"
        else np.asarray(frame.data, dtype=np.uint8)
        if frame.modality == "rgb"
        else np.asarray(frame.data)
    )
    with output_path.open("xb") as stream:
        if frame.modality == "depth" and depth_format == "npz":
            np.savez_compressed(stream, data=data)
        else:
            np.save(stream, data)
    return relative_path.as_posix()


def _write_ppm(path: Path, rgb: np.ndarray) -> None:
    """写 binary PPM，避免为 RGB 帧引入额外图像依赖。"""

    data = np.asarray(rgb, dtype=np.uint8)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"rgb PPM data must have shape HxWx3, got {data.shape}")
    height, width, _channels = data.shape
    with path.open("xb") as stream:
        stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        stream.write(np.ascontiguousarray(data).tobytes())


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """仅使用 Python 标准库写入 RGB8、无交错 PNG。"""

    data = np.asarray(rgb, dtype=np.uint8)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"rgb PNG data must have shape HxWx3, got {data.shape}")
    height, width, _channels = data.shape
    rows = b"".join(
        b"\x00" + np.ascontiguousarray(data[row]).tobytes() for row in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with path.open("xb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        stream.write(_png_chunk(b"IHDR", header))
        stream.write(_png_chunk(b"IDAT", zlib.compress(rows)))
        stream.write(_png_chunk(b"IEND", b""))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    """编码带长度和 CRC32 的单个 PNG chunk。"""

    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def _resume_frame_indices(
    save_dir: Path,
    metadata_path: Path,
    *,
    camera_name: str,
    rgb_format: str,
    depth_format: str,
) -> dict[str, int]:
    """校验已有离线输出，并返回各 modality 的下一个安全 frame index。

    metadata 行必须严格 JSON、属于当前相机、键值唯一且指向按 modality/index 推导出的
    实际文件。扫描 payload 目录时也会把未索引文件纳入最大 index，防止覆盖崩溃窗口留下
    的文件；每个 modality 的最高 index payload 还会做完整格式校验后才允许 append。
    """

    next_indices: dict[str, int] = {}
    seen_indices: set[tuple[str, int]] = set()
    seen_paths: set[str] = set()
    last_payloads: dict[str, tuple[int, Path]] = {}
    _reject_symlink(metadata_path, label="camera metadata path")
    if metadata_path.exists():
        _require_terminated_metadata_record(metadata_path)
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line_number, line in enumerate(metadata_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = strict_json_loads(line)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid camera metadata JSON at line {line_number}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"camera metadata line {line_number} must be a JSON object"
                    )
                if row.get("camera_name") != camera_name:
                    raise ValueError(
                        f"camera metadata line {line_number} belongs to a different camera"
                    )
                modality = _safe_modality(row.get("modality"))
                frame_index = _safe_frame_index(
                    row.get("frame_index"),
                    label=f"camera metadata line {line_number} frame_index",
                )
                relative_path = _validated_metadata_payload_path(
                    save_dir,
                    row.get("relative_path"),
                    modality=modality,
                    frame_index=frame_index,
                    line_number=line_number,
                    rgb_format=rgb_format,
                    depth_format=depth_format,
                )
                key = (modality, frame_index)
                if key in seen_indices or relative_path in seen_paths:
                    raise ValueError(
                        f"camera metadata line {line_number} duplicates an existing payload"
                    )
                seen_indices.add(key)
                seen_paths.add(relative_path)
                next_indices[modality] = max(
                    next_indices.get(modality, 0), frame_index + 1
                )
                _record_last_payload(
                    last_payloads,
                    modality=modality,
                    frame_index=frame_index,
                    path=save_dir / relative_path,
                )

    # 进程可能在 exclusive payload 创建后、metadata 追加前退出；孤立 payload 的 index
    # 也必须在 resume 时保留，后续写入不能覆盖它。
    for modality_dir in save_dir.iterdir():
        _reject_symlink(modality_dir, label="camera output entry")
        if not modality_dir.is_dir():
            continue
        try:
            modality = _safe_modality(modality_dir.name)
        except ValueError:
            continue
        suffix = _payload_suffix(
            modality,
            rgb_format=rgb_format,
            depth_format=depth_format,
        )
        for payload_path in modality_dir.iterdir():
            _reject_symlink(payload_path, label="camera payload path")
            if (
                payload_path.is_file()
                and payload_path.stem.isdigit()
                and payload_path.suffix in {".ppm", ".png", ".npy", ".npz"}
                and payload_path.suffix != suffix
            ):
                raise ValueError(
                    "camera resume payload format does not match configured format: "
                    f"{payload_path}"
                )
            if (
                not payload_path.is_file()
                or payload_path.suffix != suffix
                or not payload_path.stem.isdigit()
            ):
                continue
            frame_index = int(payload_path.stem)
            next_indices[modality] = max(next_indices.get(modality, 0), frame_index + 1)
            _record_last_payload(
                last_payloads,
                modality=modality,
                frame_index=frame_index,
                path=payload_path,
            )
    for modality, (_frame_index, payload_path) in last_payloads.items():
        _validate_complete_payload(
            payload_path,
            modality=modality,
            rgb_format=rgb_format,
            depth_format=depth_format,
        )
    return next_indices


def _require_terminated_metadata_record(path: Path) -> None:
    """拒绝未以换行结束的 JSON 尾记录，避免 append 后拼接成损坏行。"""

    size = path.stat().st_size
    if size == 0:
        return
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        if stream.read(1) not in {b"\n", b"\r"}:
            raise ValueError(
                f"camera metadata has an unterminated final JSON record: {path}"
            )


def _reject_symlink(path: Path, *, label: str) -> None:
    """拒绝输出树中的符号链接，避免读写逃逸到配置目录之外。"""

    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")


def _prepare_payload_parent(path: Path) -> None:
    """创建 modality 目录，并在创建前后都拒绝符号链接。

    双重检查用于覆盖“路径原先不存在、mkdir 后再次观察”的边界；它不是跨进程锁，但能
    防止正常配置直接把 recorder 指向任意链接目标。
    """

    parent = path.parent
    _reject_symlink(parent, label="camera modality directory")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink(parent, label="camera modality directory")


def _record_last_payload(
    targets: dict[str, tuple[int, Path]],
    *,
    modality: str,
    frame_index: int,
    path: Path,
) -> None:
    """记录每个 modality 当前最高 index 的 payload，供 resume 完整性检查。"""

    current = targets.get(modality)
    if current is None or frame_index > current[0]:
        targets[modality] = (frame_index, path)


def _validate_complete_payload(
    path: Path,
    *,
    modality: str,
    rgb_format: str,
    depth_format: str,
) -> None:
    """在 resume 追加前校验最高 index payload 可完整读取且类型匹配。"""

    try:
        suffix = _payload_suffix(
            modality,
            rgb_format=rgb_format,
            depth_format=depth_format,
        )
        if suffix == ".ppm":
            _validate_ppm(path)
            return
        if suffix == ".png":
            _validate_png(path)
            return
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if "data" not in archive:
                    raise ValueError("NPZ payload is missing the data array")
                _validate_payload_array(archive["data"], modality=modality)
            return
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        _validate_payload_array(array, modality=modality)
    except Exception as exc:
        raise ValueError(
            f"camera resume payload is incomplete or unreadable: {path}"
        ) from exc


def _validate_payload_array(array: np.ndarray, *, modality: str) -> None:
    """校验 numpy payload 的 canonical RGB/depth shape 与 dtype。"""

    value = np.asarray(array)
    if modality == "rgb" and (
        value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8
    ):
        raise ValueError("RGB array payload must be HxWx3 uint8")
    if modality == "depth" and (value.ndim != 2 or value.dtype != np.dtype(np.float32)):
        raise ValueError("depth array payload must be HxW float32")


def _validate_ppm(path: Path) -> None:
    """校验 PPM header 与文件字节数一致，拒绝截断图像。"""

    with path.open("rb") as stream:
        if stream.readline() != b"P6\n":
            raise ValueError("invalid PPM magic")
        dimensions = stream.readline().strip().split()
        if len(dimensions) != 2:
            raise ValueError("invalid PPM dimensions")
        width, height = (int(value) for value in dimensions)
        if width < 1 or height < 1 or stream.readline() != b"255\n":
            raise ValueError("invalid PPM header")
        expected_size = stream.tell() + width * height * 3
    if path.stat().st_size != expected_size:
        raise ValueError("truncated PPM payload")


def _validate_png(path: Path) -> None:
    """校验 PNG chunk、CRC、RGB8 header 及解压后的完整扫描行长度。"""

    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("invalid PNG signature")
        width = height = None
        compressed = bytearray()
        saw_end = False
        while not saw_end:
            length_bytes = stream.read(4)
            if len(length_bytes) != 4:
                raise ValueError("truncated PNG chunk length")
            length = struct.unpack(">I", length_bytes)[0]
            kind = stream.read(4)
            payload = stream.read(length)
            checksum = stream.read(4)
            if len(kind) != 4 or len(payload) != length or len(checksum) != 4:
                raise ValueError("truncated PNG chunk")
            expected_crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
            if struct.unpack(">I", checksum)[0] != expected_crc:
                raise ValueError("invalid PNG chunk CRC")
            if kind == b"IHDR":
                width, height, depth, color, compression, filtering, interlace = (
                    struct.unpack(">IIBBBBB", payload)
                )
                if (
                    width < 1
                    or height < 1
                    or (depth, color, compression, filtering, interlace)
                    != (8, 2, 0, 0, 0)
                ):
                    raise ValueError("unsupported PNG header")
            elif kind == b"IDAT":
                compressed.extend(payload)
            elif kind == b"IEND":
                saw_end = True
        if stream.read(1):
            raise ValueError("PNG has trailing bytes")
    if width is None or height is None or not compressed:
        raise ValueError("PNG is missing IHDR or IDAT")
    decoded = zlib.decompress(bytes(compressed))
    if len(decoded) != height * (1 + width * 3):
        raise ValueError("truncated PNG image data")


def _validated_metadata_payload_path(
    save_dir: Path,
    value: object,
    *,
    modality: str,
    frame_index: int,
    line_number: int,
    rgb_format: str,
    depth_format: str,
) -> str:
    """校验 metadata path 与 modality/index 一致且 payload 已存在。"""

    if not isinstance(value, str) or not value:
        raise ValueError(
            f"camera metadata line {line_number} relative_path must be a string"
        )
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"camera metadata line {line_number} relative_path is unsafe")
    expected = _payload_relative_path(
        modality,
        frame_index,
        rgb_format=rgb_format,
        depth_format=depth_format,
    ).as_posix()
    if relative_path.as_posix() != expected:
        raise ValueError(
            f"camera metadata line {line_number} path does not match modality/frame_index"
        )
    if not (save_dir / relative_path).is_file():
        raise ValueError(
            f"camera metadata line {line_number} references a missing payload"
        )
    return expected


def _payload_relative_path(
    modality: str,
    frame_index: int,
    *,
    rgb_format: str,
    depth_format: str,
) -> Path:
    """构造经过约束的 modality/index payload 相对路径。"""

    modality = _safe_modality(modality)
    index = _safe_frame_index(frame_index, label="camera frame_index")
    return Path(modality) / (
        f"{index:06d}"
        f"{_payload_suffix(modality, rgb_format=rgb_format, depth_format=depth_format)}"
    )


def _payload_suffix(
    modality: str,
    *,
    rgb_format: str,
    depth_format: str,
) -> str:
    """返回 modality 的离线 payload 扩展名。"""

    if modality == "rgb":
        return f".{rgb_format}"
    if modality == "depth":
        return f".{depth_format}"
    return ".npy"


def _output_format(value: object, allowed: frozenset[str], *, label: str) -> str:
    """把输出格式约束到调用方提供的 canonical allowlist。"""

    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{label} must be one of {'|'.join(sorted(allowed))}")
    return value


def _positive_int(value: object, *, label: str) -> int:
    """读取严格正整数；显式拒绝 Python 中属于 ``int`` 子类的 bool。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_finite_float(value: object, *, label: str) -> float:
    """读取严格正且有限的浮点配置。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return parsed


def _nonnegative_finite_float(value: object, *, label: str) -> float:
    """读取非负且有限的浮点配置。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a non-negative finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return parsed


def _safe_modality(value: object) -> str:
    """拒绝会逃逸输出目录的 modality 名称。"""

    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("camera modality must be a safe non-empty path segment")
    return value


def _safe_frame_index(value: object, *, label: str) -> int:
    """校验非负整数 frame index。"""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a non-negative integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed
