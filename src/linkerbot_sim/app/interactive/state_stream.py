"""Interactive runtime wiring for Foxglove realtime state streaming."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from linkerbot_sim.app.runtime.dual_robot import DualRobotAppRuntime
from linkerbot_sim.telemetry.foxglove_state import (
    CompositeStateSink,
    FoxgloveStateSink,
    JointEffortField,
    StatePublisher,
    StateSnapshotSink,
)
from linkerbot_sim.telemetry.state_snapshot import (
    DualRobotStateObserver,
    DualRobotStateSampler,
    SingleRobotStateObserver,
    SingleRobotStateSampler,
    StateStream,
)


@dataclass(frozen=True)
class InteractiveStateStreamConfig:
    """交互实时状态流配置。第一阶段只支持 Foxglove 输出。"""

    rate_hz: float = 60.0
    include_efforts: bool = False
    include_objects: bool = False
    foxglove_live_host: str = "127.0.0.1"
    foxglove_live_port: int | None = None
    foxglove_mcap_path: str | Path | None = None
    foxglove_joint_effort_field: JointEffortField = "none"

    def enabled(self) -> bool:
        """返回是否需要启动状态流。"""

        return self.rate_hz > 0.0 and (
            self.foxglove_live_port is not None or self.foxglove_mcap_path is not None
        )


@dataclass
class InteractiveStateStreamHandle:
    """交互状态流运行时句柄，用于关闭线程和恢复 execution runtime。"""

    runtime: object
    previous_execution: object
    publisher: StatePublisher
    stream: StateStream

    def close(self) -> None:
        """停止后台 publisher，并移除 execution 上的 observer。"""

        try:
            self.publisher.close()
        finally:
            self.runtime.execution = self.previous_execution


def start_interactive_state_stream(
    runtime: DualRobotAppRuntime | object,
    *,
    config: InteractiveStateStreamConfig | None,
    status_prefix: str | None = None,
) -> InteractiveStateStreamHandle | None:
    """按配置启动 Foxglove 状态流。"""

    if config is None or not config.enabled():
        return None

    sinks = _foxglove_sinks(config)
    if not sinks:
        return None
    sink: StateSnapshotSink = (
        sinks[0] if len(sinks) == 1 else CompositeStateSink(tuple(sinks))
    )
    stream = StateStream()
    observer = _state_observer_for_runtime(
        runtime,
        stream=stream,
        config=config,
    )
    publisher = StatePublisher(
        stream=stream, sink=sink, name="interactive-foxglove-state"
    )
    previous_execution = runtime.execution
    runtime.execution = replace(previous_execution, state_observer=observer)
    publisher.start()
    _print_status(
        status_prefix,
        "STATE_STREAM "
        f"rate_hz={config.rate_hz:g} include_efforts={config.include_efforts} "
        f"include_objects={config.include_objects} "
        f"joint_effort_field={config.foxglove_joint_effort_field}",
    )
    return InteractiveStateStreamHandle(
        runtime=runtime,
        previous_execution=previous_execution,
        publisher=publisher,
        stream=stream,
    )


def _state_observer_for_runtime(
    runtime: object,
    *,
    stream: StateStream,
    config: InteractiveStateStreamConfig,
) -> DualRobotStateObserver | SingleRobotStateObserver:
    """按 app runtime 类型创建主线程状态 observer。"""

    execution = getattr(runtime, "execution")
    sampler_kwargs = {
        "stage": runtime.session.stage,
        "object_handles": runtime.object_handles,
        "rate_hz": config.rate_hz,
        "include_efforts": config.include_efforts,
        "include_objects": config.include_objects,
    }
    if hasattr(execution, "left") and hasattr(execution, "right"):
        return DualRobotStateObserver(
            sampler=DualRobotStateSampler(**sampler_kwargs),
            stream=stream,
        )
    return SingleRobotStateObserver(
        runtime=runtime,
        sampler=SingleRobotStateSampler(**sampler_kwargs),
        stream=stream,
    )


def _foxglove_sinks(
    config: InteractiveStateStreamConfig,
) -> list[FoxgloveStateSink]:
    """按配置创建一个或多个 Foxglove sink。"""

    sinks: list[FoxgloveStateSink] = []
    if config.foxglove_live_port is not None:
        sinks.append(
            FoxgloveStateSink.open_live(
                host=config.foxglove_live_host,
                port=int(config.foxglove_live_port),
                joint_effort_field=config.foxglove_joint_effort_field,
                publish_scene_objects=config.include_objects,
            )
        )
    if config.foxglove_mcap_path is not None:
        sinks.append(
            FoxgloveStateSink.open_mcap(
                config.foxglove_mcap_path,
                joint_effort_field=config.foxglove_joint_effort_field,
                publish_scene_objects=config.include_objects,
            )
        )
    return sinks


def _print_status(status_prefix: str | None, message: str) -> None:
    """输出状态流启动信息。"""

    if status_prefix is None:
        return
    print(f"{status_prefix}_{message}", flush=True)
