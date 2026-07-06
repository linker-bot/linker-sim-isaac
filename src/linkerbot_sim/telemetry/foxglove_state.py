"""Foxglove sink for realtime state snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Event, Thread
from typing import Literal, Protocol, cast

import numpy as np

from linkerbot_sim.telemetry.foxglove import FoxgloveLogger
from linkerbot_sim.telemetry.state_snapshot import StateSnapshot, StateStream


JointEffortField = Literal["none", "commanded", "measured", "applied"]


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

    def publish(self, snapshot: StateSnapshot) -> None:
        """向所有 sink 发布同一帧状态。"""

        for sink in self.sinks:
            sink.publish(snapshot)

    def close(self) -> None:
        """关闭所有 sink。"""

        for sink in self.sinks:
            sink.close()


class FoxgloveStateSink:
    """把 `StateSnapshot` 写入 Foxglove live server 或 MCAP。"""

    def __init__(
        self,
        logger: FoxgloveLogger,
        *,
        joint_effort_field: JointEffortField = "none",
        publish_scene_objects: bool = True,
    ) -> None:
        """创建 Foxglove 状态 sink。"""

        self.logger = logger
        self.joint_effort_field = _validate_effort_field(joint_effort_field)
        self.publish_scene_objects = bool(publish_scene_objects)

    @classmethod
    def open_live(
        cls,
        *,
        host: str = "127.0.0.1",
        port: int,
        name: str = "linkerbot-sim-state",
        joint_effort_field: JointEffortField = "none",
        publish_scene_objects: bool = True,
    ) -> "FoxgloveStateSink":
        """打开 Foxglove live server 输出。"""

        return cls(
            FoxgloveLogger.open_live_server(host=host, port=port, name=name),
            joint_effort_field=joint_effort_field,
            publish_scene_objects=publish_scene_objects,
        )

    @classmethod
    def open_mcap(
        cls,
        path: str | Path,
        *,
        joint_effort_field: JointEffortField = "none",
        publish_scene_objects: bool = True,
    ) -> "FoxgloveStateSink":
        """打开 Foxglove MCAP 输出。"""

        return cls(
            FoxgloveLogger.open_mcap(path),
            joint_effort_field=joint_effort_field,
            publish_scene_objects=publish_scene_objects,
        )

    def publish(self, snapshot: StateSnapshot) -> None:
        """发布一帧状态到 Foxglove topics。"""

        joint_names, positions, velocities, efforts = _joint_state_arrays(
            snapshot, self.joint_effort_field
        )
        if joint_names:
            self.logger.log_joint_state(
                joint_names=joint_names,
                positions=positions,
                velocities=velocities,
                efforts=efforts,
                time_s=snapshot.time_s,
            )
        self.logger.log_state_json(snapshot.as_dict(), time_s=snapshot.time_s)
        if self.publish_scene_objects and snapshot.objects:
            self.logger.log_scene_spheres(
                entity_id="runtime_objects",
                positions=np.asarray(
                    [obj.position_m for obj in snapshot.objects], dtype=float
                ),
                radius=0.025,
                color=(0.9, 0.2, 0.15, 1.0),
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
    ) -> None:
        """创建后台 publisher；调用 `start()` 后才会启动线程。"""

        self.stream = stream
        self.sink = sink
        self.name = name
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.last_error: Exception | None = None

    def start(self) -> None:
        """启动后台发布线程。"""

        if self.thread is not None:
            return
        self.thread = Thread(target=self._run, name=self.name, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """请求停止并等待线程退出。"""

        self.stop_event.set()
        self.stream.close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def close(self) -> None:
        """停止线程并关闭 sink。"""

        self.stop()
        self.sink.close()

    def _run(self) -> None:
        """后台发布循环，只消费快照，不访问 Isaac runtime。"""

        sequence = 0
        while not self.stop_event.is_set():
            item = self.stream.wait_next(sequence, timeout_s=0.1)
            if item is None:
                continue
            sequence, snapshot = item
            try:
                self.sink.publish(snapshot)
            except Exception as exc:
                self.last_error = exc
                self.stop_event.set()
                print(
                    f"STATE_PUBLISHER_FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )


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
        names.extend(f"{robot.side}/{name}" for name in robot.joint_names)
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
