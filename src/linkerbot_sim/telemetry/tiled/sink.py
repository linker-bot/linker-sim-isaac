"""Tiled interactive Foxglove/MCAP sink 生命周期、背压与发布。"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread

from linkerbot_sim.telemetry.foxglove import (
    FoxgloveLogger,
    FoxgloveTopicConfig,
    prepare_mcap_output,
)
from linkerbot_sim.telemetry.tiled.config import TiledTelemetryConfig
from linkerbot_sim.telemetry.tiled.payloads import (
    SceneMarkers,
    build_json_payload,
    selected_joint_efforts,
    selected_joint_state_arrays,
    selected_scene_markers,
)
from linkerbot_sim.utils.output_paths import (
    OutputPathPlan,
    apply_output_path_plans,
)


_OBJECT_MARKER_RADIUS_M = 0.025
_OBJECT_MARKER_COLOR_RGBA = (0.95, 0.72, 0.20, 1.0)
_TCP_MARKER_RADIUS_M = 0.018
_TCP_MARKER_COLOR_RGBA = (0.1, 0.72, 0.95, 1.0)


@dataclass(frozen=True)
class _TiledTelemetryFrame:
    sequence: int
    step: int
    time_s: float
    payload: Mapping[str, object]
    joint_state: tuple[list[str], object, object] | None
    joint_efforts: object | None
    scene_markers: SceneMarkers


class TiledInteractiveTelemetrySink:
    """把主线程冻结的 tiled state 写入有界 Foxglove/MCAP publisher。"""

    def __init__(
        self,
        loggers: Sequence[FoxgloveLogger],
        *,
        config: TiledTelemetryConfig,
        asynchronous: bool = False,
    ) -> None:
        if not loggers:
            raise ValueError("at least one FoxgloveLogger is required")
        self.loggers = tuple(loggers)
        self.config = config
        self.last_published_step: int | None = None
        self.last_published_sequence: int | None = None
        self.last_error: BaseException | None = None
        self.error_count = 0
        self.dropped_snapshots = 0
        self._next_sequence = 0
        self._last_admitted_step: int | None = None
        self._condition = Condition()
        self._queue: deque[_TiledTelemetryFrame] = deque()
        self._closing = False
        self._stopped_on_error = False
        self.shutdown_timed_out = False
        self._closed_logger_indices: set[int] = set()
        self._loggers_closed = False
        self._thread: Thread | None = None
        if asynchronous:
            thread = Thread(
                target=self._run,
                name="tiled-telemetry-publisher",
                daemon=True,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                try:
                    self.close()
                except BaseException:
                    pass
                raise

    @classmethod
    def open(
        cls,
        *,
        config: TiledTelemetryConfig,
        live_host: str = "127.0.0.1",
        live_port: int | None = None,
        mcap_path: str | Path | None = None,
        mcap_output_plan: OutputPathPlan | None = None,
        output_paths_applied: bool = False,
    ) -> "TiledInteractiveTelemetrySink | None":
        """预检全部文件目标后创建 live/MCAP 输出和后台 publisher。"""

        if live_port is None and mcap_path is None:
            return None
        mcap_plan = mcap_output_plan
        if mcap_plan is None:
            mcap_plan = prepare_mcap_output(
                mcap_path,
                existing_file_policy=config.mcap_existing_file_policy,
            )
        elif (
            mcap_path is None
            or mcap_plan.requested_path != Path(mcap_path).expanduser()
            or mcap_plan.policy != config.mcap_existing_file_policy
        ):
            raise ValueError(
                "prepared MCAP output does not match telemetry configuration"
            )
        if mcap_plan is not None and not output_paths_applied:
            apply_output_path_plans((mcap_plan,))

        topics = _topics_for_config(config)
        loggers: list[FoxgloveLogger] = []
        try:
            if live_port is not None:
                loggers.append(
                    FoxgloveLogger.open_live_server(
                        host=live_host,
                        port=int(live_port),
                        name="linkerbot-tiled-interactive",
                        topics=topics,
                    )
                )
            if mcap_plan is not None:
                loggers.append(
                    FoxgloveLogger.open_mcap(
                        mcap_plan.resolved_path,
                        topics=topics,
                    )
                )
        except BaseException:
            for logger in reversed(loggers):
                try:
                    logger.close()
                except BaseException:
                    pass
            raise
        return cls(loggers, config=config, asynchronous=True)

    def close(self) -> bool:
        """排空有界队列并关闭 sinks；超时时保留 live thread 和 logger。"""

        with self._condition:
            self.shutdown_timed_out = False
            self._closing = True
            self._condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=float(self.config.shutdown_timeout_s))
            if thread.is_alive():
                with self._condition:
                    self.shutdown_timed_out = True
                return False
            self._thread = None
        first_error: BaseException | None = None
        for index, logger in enumerate(self.loggers):
            if index in self._closed_logger_indices:
                continue
            try:
                logger.close()
            except BaseException as exc:
                with self._condition:
                    self.last_error = exc
                    self.error_count += 1
                if first_error is None:
                    first_error = exc
            else:
                self._closed_logger_indices.add(index)
        with self._condition:
            self._loggers_closed = len(self._closed_logger_indices) == len(self.loggers)
        if first_error is not None:
            return False
        return True

    def status(self) -> dict[str, object]:
        """返回 queue、丢帧、错误和最后成功发布序号。"""

        with self._condition:
            thread = self._thread
            return {
                "buffer_depth": len(self._queue),
                "buffer_capacity": int(self.config.buffer_size),
                "drop_policy": self.config.drop_policy,
                "on_error": self.config.on_error,
                "dropped_snapshots": self.dropped_snapshots,
                "error_count": self.error_count,
                "last_published_sequence": self.last_published_sequence,
                "last_published_step": self.last_published_step,
                "last_error": (
                    None
                    if self.last_error is None
                    else f"{type(self.last_error).__name__}: {self.last_error}"
                ),
                "thread_alive": thread is not None and thread.is_alive(),
                "stopped_on_error": self._stopped_on_error,
                "shutdown_timed_out": self.shutdown_timed_out,
                "sink_closed": self._loggers_closed,
            }

    def record_error(self, exc: Exception) -> None:
        """按 publisher 相同策略记录主线程 telemetry 采样错误。

        sampling 在 Isaac 主线程发生，但错误计数、停止策略和最后异常必须与后台 sink 写入
        失败共享同一状态机，避免 status 对失败来源给出不一致结论。
        """

        self._record_error(exc)

    def publish_interactive_state(
        self,
        state_response: Mapping[str, object],
        *,
        event: str,
        trigger_response: Mapping[str, object] | None = None,
    ) -> bool:
        """冻结并提交一帧；返回 False 表示被 decimation 或背压丢弃。"""

        step = int(state_response.get("step", 0))
        if not self._should_publish(step=step, event=event):
            return False
        with self._condition:
            if self._closing or self._stopped_on_error:
                return False
            self._next_sequence += 1
            sequence = self._next_sequence
            self._last_admitted_step = step
        frame = self._build_frame(
            state_response,
            sequence=sequence,
            step=step,
            event=event,
            trigger_response=trigger_response,
        )
        if self._thread is None:
            return self._publish_direct(frame)
        return self._enqueue(frame)

    def _build_frame(
        self,
        state_response: Mapping[str, object],
        *,
        sequence: int,
        step: int,
        event: str,
        trigger_response: Mapping[str, object] | None,
    ) -> _TiledTelemetryFrame:
        """在提交线程中提取一帧 publisher 所需的各模态数据。

        JSON 是否保留 objects、是否采集 effort 均由配置决定；标准关节状态和 scene
        marker 只选择 ``primary_env_id``。后台线程仅消费构造完成的 frame，不再访问
        Isaac runtime。
        """

        selected_env_id = self.config.primary_env_id
        payload_source = _state_response_without_objects(state_response)
        if self.config.include_objects:
            payload_source = state_response
        return _TiledTelemetryFrame(
            sequence=sequence,
            step=step,
            time_s=float(state_response.get("time_s", 0.0)),
            payload=build_json_payload(
                payload_source,
                event=event,
                trigger_response=trigger_response,
            ),
            joint_state=selected_joint_state_arrays(
                state_response,
                selected_env_id=selected_env_id,
            ),
            joint_efforts=(
                selected_joint_efforts(
                    state_response,
                    selected_env_id=selected_env_id,
                )
                if self.config.include_efforts
                else None
            ),
            scene_markers=selected_scene_markers(
                state_response,
                selected_env_id=selected_env_id,
            ),
        )

    def _enqueue(self, frame: _TiledTelemetryFrame) -> bool:
        """在条件变量保护下按背压策略把 frame 放入有界队列。

        ``latest`` 会清空所有待发帧，``drop_newest`` 在队列满时拒绝当前帧，其余策略
        淘汰最早帧。sink 已关闭、因错误停止或当前帧被拒绝时返回 ``False``。
        """

        with self._condition:
            if self._closing or self._stopped_on_error:
                return False
            if self.config.drop_policy == "latest":
                self.dropped_snapshots += len(self._queue)
                self._queue.clear()
            elif len(self._queue) >= int(self.config.buffer_size):
                self.dropped_snapshots += 1
                if self.config.drop_policy == "drop_newest":
                    return False
                self._queue.popleft()
            self._queue.append(frame)
            self._condition.notify()
            return True

    def _publish_direct(self, frame: _TiledTelemetryFrame) -> bool:
        """同步发布一帧，并把成功或异常写入与异步模式共用的状态机。"""

        try:
            self._publish_frame(frame)
        except Exception as exc:
            self._record_error(exc)
            return False
        self._record_success(frame)
        return True

    def _run(self) -> None:
        """消费后台队列，关闭时排空已接纳帧后退出。

        发布发生在条件变量锁外，避免阻塞生产线程和状态查询。单帧失败时按
        ``on_error`` 决定继续消费还是停止；停止策略会由 ``_record_error`` 清空余帧。
        """

        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closing)
                if not self._queue:
                    return
                frame = self._queue.popleft()
            try:
                self._publish_frame(frame)
            except Exception as exc:
                self._record_error(exc)
                if self.config.on_error == "stop":
                    return
            else:
                self._record_success(frame)

    def _publish_frame(self, frame: _TiledTelemetryFrame) -> None:
        """按配置将一帧的各模态依次写入全部 logger。

        logger 按创建顺序串行写入；任一写入异常立即向调用方传播，本方法不尝试跨
        logger 回滚，错误计数与继续/停止策略由上层统一处理。
        """

        selected_env_id = self.config.primary_env_id
        for logger in self.loggers:
            if self.config.include_full_batch_json:
                logger.log_state_json(frame.payload, time_s=frame.time_s)
            if (
                self.config.include_standard_joint_states
                and frame.joint_state is not None
            ):
                names, positions, velocities = frame.joint_state
                logger.log_joint_state(
                    joint_names=names,
                    positions=positions,
                    velocities=velocities,
                    efforts=frame.joint_efforts,
                    time_s=frame.time_s,
                )
            if self.config.include_scene_markers:
                _publish_scene_markers(
                    logger,
                    frame.scene_markers,
                    selected_env_id=selected_env_id,
                    include_objects=self.config.include_objects,
                    time_s=frame.time_s,
                )

    def _record_success(self, frame: _TiledTelemetryFrame) -> None:
        """原子更新最后成功发布的 step 与单调递增 sequence。"""

        with self._condition:
            self.last_published_step = frame.step
            self.last_published_sequence = frame.sequence

    def _record_error(self, exc: Exception) -> None:
        """原子记录发布错误，并在停止策略下关闭接纳入口、丢弃所有待发帧。

        清空队列时同步累计丢帧数并唤醒消费者，使后台线程可以立即观察关闭状态。
        """

        with self._condition:
            self.last_error = exc
            self.error_count += 1
            if self.config.on_error == "stop":
                self._stopped_on_error = True
                self._closing = True
                self.dropped_snapshots += len(self._queue)
                self._queue.clear()
                self._condition.notify_all()

    def _should_publish(self, *, step: int, event: str) -> bool:
        """应用 decimation/去重规则；reset 与 set_state 始终立即发布。"""

        if event in {"reset", "set_state"}:
            return True
        if self._last_admitted_step == step and event != "state":
            return False
        return step % int(self.config.publish_decimation) == 0


def _topics_for_config(config: TiledTelemetryConfig) -> FoxgloveTopicConfig:
    """返回 runtime profile 已解析的 exact topics。"""

    return config.topics


def _state_response_without_objects(
    state_response: Mapping[str, object],
) -> dict[str, object]:
    """浅拷贝 tiled payload 并移除可选 object modality。"""

    result = dict(state_response)
    state = state_response.get("state")
    if isinstance(state, Mapping):
        filtered_state = dict(state)
        filtered_state.pop("objects", None)
        result["state"] = filtered_state
    return result


def _publish_scene_markers(
    logger: FoxgloveLogger,
    markers: SceneMarkers,
    *,
    selected_env_id: int,
    include_objects: bool,
    time_s: float,
) -> None:
    """发布 primary env object/TCP markers。"""

    log_scene_spheres = getattr(logger, "log_scene_spheres", None)
    if not callable(log_scene_spheres):
        return
    env_prefix = f"env_{int(selected_env_id):03d}"
    if include_objects:
        for object_name, point in markers.get("objects", []):
            log_scene_spheres(
                entity_id=f"{env_prefix}/object/{object_name}",
                positions=[point],
                frame_id="world",
                radius=_OBJECT_MARKER_RADIUS_M,
                color=_OBJECT_MARKER_COLOR_RGBA,
                time_s=time_s,
            )
    for robot_name, point in markers.get("tcps", []):
        log_scene_spheres(
            entity_id=f"{env_prefix}/tcp/{robot_name}",
            positions=[point],
            frame_id="world",
            radius=_TCP_MARKER_RADIUS_M,
            color=_TCP_MARKER_COLOR_RGBA,
            time_s=time_s,
        )


__all__ = ["TiledInteractiveTelemetrySink"]
