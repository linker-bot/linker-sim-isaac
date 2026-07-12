"""把 realtime state snapshots 序列化到 Foxglove live/MCAP 的 sink。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Event, Thread
from typing import Literal, Protocol, cast

import numpy as np

from linkerbot_sim.telemetry.foxglove import FoxgloveLogger, FoxgloveTopicConfig
from linkerbot_sim.telemetry.state_snapshot import StateSnapshot, StateStream


JointEffortField = Literal["none", "commanded", "measured", "applied"]

_RUNTIME_OBJECT_MARKER_RADIUS_M = 0.025
_RUNTIME_OBJECT_MARKER_COLOR_RGBA = (0.9, 0.2, 0.15, 1.0)


class StateSnapshotSink(Protocol):
    """状态快照输出端协议。"""

    def publish(self, snapshot: StateSnapshot) -> None:
        """发布一帧状态快照。"""

    def close(self) -> None:
        """关闭输出端。"""


class CompositeStateSink:
    """把同一帧状态发布到多个 sink。"""

    def __init__(self, sinks: Sequence[StateSnapshotSink]) -> None:
        """保存多个状态输出端。"""

        self.sinks = tuple(sinks)
        self._closed_sink_indices: set[int] = set()

    def publish(self, snapshot: StateSnapshot) -> None:
        """向所有 sink 发布同一帧状态。"""

        for sink in self.sinks:
            sink.publish(snapshot)

    def close(self) -> None:
        """尽力关闭所有 sink，并在最后重抛第一个关闭异常。"""

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


class FoxgloveStateSink:
    """把 `StateSnapshot` 写入 Foxglove live server 或 MCAP。"""

    def __init__(
        self,
        logger: FoxgloveLogger,
        *,
        joint_effort_field: JointEffortField = "none",
        publish_joint_states: bool = True,
        publish_state_json: bool = True,
        publish_scene_markers: bool = True,
    ) -> None:
        """创建 Foxglove 状态 sink。"""

        self.logger = logger
        self.joint_effort_field: JointEffortField = _validate_effort_field(
            joint_effort_field
        )
        self.publish_joint_states = bool(publish_joint_states)
        self.publish_state_json = bool(publish_state_json)
        self.publish_scene_markers = bool(publish_scene_markers)

    @classmethod
    def open_live(
        cls,
        *,
        host: str = "127.0.0.1",
        port: int,
        name: str = "linkerbot-sim-state",
        joint_effort_field: JointEffortField = "none",
        publish_joint_states: bool = True,
        publish_state_json: bool = True,
        publish_scene_markers: bool = True,
        topics: FoxgloveTopicConfig | None = None,
    ) -> "FoxgloveStateSink":
        """打开 Foxglove live server 输出。"""

        return cls(
            FoxgloveLogger.open_live_server(
                host=host,
                port=port,
                name=name,
                topics=topics,
            ),
            joint_effort_field=joint_effort_field,
            publish_joint_states=publish_joint_states,
            publish_state_json=publish_state_json,
            publish_scene_markers=publish_scene_markers,
        )

    @classmethod
    def open_mcap(
        cls,
        path: str | Path,
        *,
        joint_effort_field: JointEffortField = "none",
        publish_joint_states: bool = True,
        publish_state_json: bool = True,
        publish_scene_markers: bool = True,
        topics: FoxgloveTopicConfig | None = None,
    ) -> "FoxgloveStateSink":
        """打开 Foxglove MCAP 输出。"""

        return cls(
            FoxgloveLogger.open_mcap(
                path,
                topics=topics,
            ),
            joint_effort_field=joint_effort_field,
            publish_joint_states=publish_joint_states,
            publish_state_json=publish_state_json,
            publish_scene_markers=publish_scene_markers,
        )

    def publish(self, snapshot: StateSnapshot) -> None:
        """发布一帧状态到 Foxglove topics。"""

        joint_names, positions, velocities, efforts = _joint_state_arrays(
            snapshot, self.joint_effort_field
        )
        if self.publish_joint_states and joint_names:
            self.logger.log_joint_state(
                joint_names=joint_names,
                positions=positions,
                velocities=velocities,
                efforts=efforts,
                time_s=snapshot.time_s,
            )
        if self.publish_state_json:
            self.logger.log_state_json(snapshot.as_dict(), time_s=snapshot.time_s)
        if self.publish_scene_markers and snapshot.objects:
            self.logger.log_scene_spheres(
                entity_id="runtime_objects",
                positions=np.asarray(
                    [obj.position_m for obj in snapshot.objects], dtype=float
                ),
                radius=_RUNTIME_OBJECT_MARKER_RADIUS_M,
                color=_RUNTIME_OBJECT_MARKER_COLOR_RGBA,
                time_s=snapshot.time_s,
            )

    def close(self) -> None:
        """关闭底层 Foxglove logger。"""

        self.logger.close()


class StatePublisher:
    """后台线程：从 `StateStream` 取最新快照并发布到 sink。"""

    def __init__(
        self,
        *,
        stream: StateStream,
        sink: StateSnapshotSink,
        name: str = "state-publisher",
        shutdown_timeout_s: float = 2.0,
        worker_poll_interval_s: float = 0.1,
        on_error: str = "stop",
    ) -> None:
        """创建后台 publisher；调用 `start()` 后才会启动线程。"""

        self.stream = stream
        self.sink = sink
        self.name = name
        self.shutdown_timeout_s = _non_negative_timeout(
            shutdown_timeout_s, label="shutdown_timeout_s"
        )
        self.worker_poll_interval_s = _positive_timeout(
            worker_poll_interval_s, label="worker_poll_interval_s"
        )
        if on_error not in {"stop", "continue"}:
            raise ValueError("on_error must be one of: stop, continue")
        self.on_error = on_error
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.last_error: BaseException | None = None
        self.error_count = 0
        self.last_published_sequence: int | None = None
        self.shutdown_timed_out = False
        self._sink_closed = False

    def start(self) -> None:
        """启动后台发布线程。"""

        if self.thread is not None:
            return
        thread = Thread(target=self._run, name=self.name, daemon=True)
        self.thread = thread
        try:
            thread.start()
        except BaseException:
            self.thread = None
            raise

    def stop(self, *, timeout_s: float | None = None) -> bool:
        """请求停止并有界等待；超时时保留仍存活的线程句柄。"""

        self.shutdown_timed_out = False
        self.stop_event.set()
        self.stream.close()
        timeout = (
            self.shutdown_timeout_s
            if timeout_s is None
            else _non_negative_timeout(timeout_s, label="timeout_s")
        )
        thread = self.thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                self.shutdown_timed_out = True
                return False
            self.thread = None
        return True

    def close(self, *, timeout_s: float | None = None) -> bool:
        """停止线程并关闭 sink；线程超时时避免与 sink.close 并发。"""

        stopped = self.stop(timeout_s=timeout_s)
        if stopped and not self._sink_closed:
            try:
                self.sink.close()
            except BaseException as exc:
                self.last_error = exc
                self.error_count += 1
                return False
            self._sink_closed = True
        return stopped

    def status(self) -> dict[str, object]:
        """返回 publisher 生命周期和失败诊断。"""

        thread = self.thread
        return {
            "name": self.name,
            "thread_alive": thread is not None and thread.is_alive(),
            "shutdown_requested": self.stop_event.is_set(),
            "shutdown_timed_out": self.shutdown_timed_out,
            "sink_closed": self._sink_closed,
            "on_error": self.on_error,
            "error_count": self.error_count,
            "last_published_sequence": self.last_published_sequence,
            "last_error": (
                None
                if self.last_error is None
                else f"{type(self.last_error).__name__}: {self.last_error}"
            ),
            **self.stream.status(),
        }

    def _run(self) -> None:
        """后台发布循环，只消费快照，不访问 Isaac runtime。"""

        sequence = 0
        while True:
            item = self.stream.wait_next(
                sequence, timeout_s=self.worker_poll_interval_s
            )
            if item is None:
                if self.stream.is_closed():
                    return
                continue
            sequence, snapshot = item
            try:
                self.sink.publish(snapshot)
            except Exception as exc:
                self.last_error = exc
                self.error_count += 1
                print(
                    f"STATE_PUBLISHER_FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if self.on_error == "stop":
                    self.stop_event.set()
                    self.stream.close(discard_pending=True)
                    return
            else:
                self.last_published_sequence = sequence


def _joint_state_arrays(
    snapshot: StateSnapshot, effort_field: JointEffortField
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None]:
    """把完整快照展平成 Foxglove JointStates 数组。"""

    field = _validate_effort_field(effort_field)
    names: list[str] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    efforts: list[np.ndarray] = []
    include_efforts = field != "none"
    for robot in snapshot.robots:
        names.extend(f"{robot.label}/{name}" for name in robot.joint_names)
        positions.append(np.asarray(robot.positions_rad, dtype=float).reshape(-1))
        velocities.append(np.asarray(robot.velocities_rad_s, dtype=float).reshape(-1))
        if include_efforts:
            effort_values = robot.effort_values(field)
            if effort_values is None:
                effort_values = np.full(len(robot.joint_names), np.nan, dtype=float)
            efforts.append(np.asarray(effort_values, dtype=float).reshape(-1))
    if not names:
        return [], np.asarray([], dtype=float), np.asarray([], dtype=float), None
    return (
        names,
        np.concatenate(positions) if positions else np.asarray([], dtype=float),
        np.concatenate(velocities) if velocities else np.asarray([], dtype=float),
        np.concatenate(efforts) if include_efforts and efforts else None,
    )


def _validate_effort_field(value: str) -> JointEffortField:
    """校验 Foxglove JointStates.effort 选择。"""

    normalized = str(value).lower()
    if normalized not in {"none", "commanded", "measured", "applied"}:
        raise ValueError(
            "joint_effort_field must be one of: none, commanded, measured, applied"
        )
    return cast(JointEffortField, normalized)


def _non_negative_timeout(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a non-negative finite number")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return parsed


def _positive_timeout(value: object, *, label: str) -> float:
    parsed = _non_negative_timeout(value, label=label)
    if parsed == 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed
