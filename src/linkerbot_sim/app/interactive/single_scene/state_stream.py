"""把 SingleSceneRuntime 主线程采样接入 Foxglove realtime state stream。

observer 只在 simulation step 边界采集 Isaac 状态，后台 publisher 只消费冻结后的
snapshot 并写 Foxglove/MCAP，从而避免 telemetry 线程触碰 articulation 或 stage。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from linkerbot_sim.telemetry.foxglove import (
    FoxgloveTopicConfig,
    prepare_mcap_output,
)
from linkerbot_sim.telemetry.foxglove_state import (
    CompositeStateSink,
    FoxgloveStateSink,
    JointEffortField,
    StatePublisher,
    StateSnapshotSink,
)
from linkerbot_sim.telemetry.state_snapshot import (
    SceneRobotStateObserver,
    SceneRobotStateSampler,
    StateStream,
)
from linkerbot_sim.utils.output_paths import (
    OutputPathPlan,
    apply_output_path_plans,
)


@dataclass(frozen=True)
class InteractiveStateStreamConfig:
    """交互实时状态流的采样、模态、背压和 Foxglove 输出配置。"""

    rate_hz: float = 60.0
    buffer_size: int = 1
    drop_policy: str = "latest"
    on_error: str = "stop"
    include_joint_states: bool = True
    include_state_json: bool = True
    include_scene_markers: bool = True
    include_efforts: bool = False
    include_objects: bool = False
    topics: FoxgloveTopicConfig = field(default_factory=FoxgloveTopicConfig)
    foxglove_live_host: str = "127.0.0.1"
    foxglove_live_port: int | None = None
    foxglove_mcap_path: str | Path | None = None
    mcap_existing_file_policy: str = "error"
    mcap_output_plan: OutputPathPlan | None = None
    output_paths_applied: bool = False
    foxglove_joint_effort_field: JointEffortField = "none"
    shutdown_timeout_s: float = 2.0

    def enabled(self) -> bool:
        """返回是否需要启动状态流。"""

        has_modality = (
            self.include_joint_states
            or self.include_state_json
            or self.include_scene_markers
        )
        has_output = (
            self.foxglove_live_port is not None or self.foxglove_mcap_path is not None
        )
        return self.rate_hz > 0.0 and has_modality and has_output


@dataclass
class InteractiveStateStreamHandle:
    """交互状态流运行时句柄，用于关闭线程和恢复原 observer。"""

    runtime: object
    previous_observer: object | None
    previous_status_provider: object | None
    publisher: StatePublisher
    stream: StateStream

    def close(self) -> bool:
        """停止后台 publisher，并移除 execution 上的 observer。"""

        stopped = False
        try:
            stopped = self.publisher.close()
            return stopped
        finally:
            self.runtime.state_observer = self.previous_observer
            self.runtime.telemetry_status_provider = (
                self.previous_status_provider if stopped else self.publisher.status
            )


def start_interactive_state_stream(
    runtime: object,
    *,
    config: InteractiveStateStreamConfig | None,
    status_prefix: str | None = None,
) -> InteractiveStateStreamHandle | None:
    """按配置启动 Foxglove 状态流。"""

    if config is None or not config.enabled():
        return None
    if not _looks_like_single_scene_runtime(runtime):
        raise ValueError("interactive state stream requires SingleSceneRuntime")

    sinks = _foxglove_sinks(config)
    if not sinks:
        return None
    sink: StateSnapshotSink = (
        sinks[0] if len(sinks) == 1 else CompositeStateSink(tuple(sinks))
    )
    previous_observer = getattr(runtime, "state_observer", None)
    previous_status_provider = getattr(runtime, "telemetry_status_provider", None)
    publisher: StatePublisher | None = None
    observer_installed = False
    try:
        stream = StateStream(
            capacity=config.buffer_size,
            drop_policy=config.drop_policy,
        )
        observer = _state_observer_for_runtime(
            runtime,
            stream=stream,
            config=config,
        )
        publisher = StatePublisher(
            stream=stream,
            sink=sink,
            name="interactive-foxglove-state",
            shutdown_timeout_s=config.shutdown_timeout_s,
            on_error=config.on_error,
        )
        runtime.state_observer = observer
        runtime.telemetry_status_provider = publisher.status
        observer_installed = True
        publisher.start()
    except BaseException:
        if observer_installed:
            runtime.state_observer = previous_observer
            runtime.telemetry_status_provider = previous_status_provider
        try:
            if publisher is None:
                sink.close()
            else:
                publisher.close()
        except BaseException:
            pass
        raise
    _print_status(
        status_prefix,
        "STATE_STREAM "
        f"rate_hz={config.rate_hz:g} include_efforts={config.include_efforts} "
        f"include_objects={config.include_objects} "
        f"buffer_size={config.buffer_size} drop_policy={config.drop_policy} "
        f"joint_effort_field={config.foxglove_joint_effort_field}",
    )
    return InteractiveStateStreamHandle(
        runtime=runtime,
        previous_observer=previous_observer,
        previous_status_provider=previous_status_provider,
        publisher=publisher,
        stream=stream,
    )


def _state_observer_for_runtime(
    runtime: object,
    *,
    stream: StateStream,
    config: InteractiveStateStreamConfig,
) -> SceneRobotStateObserver:
    """为 canonical SingleSceneRuntime 创建主线程状态 observer。"""

    sampler_kwargs = {
        "stage": runtime.session.stage,
        "object_handles": runtime.object_handles,
        "rate_hz": config.rate_hz,
        "include_efforts": config.include_efforts,
        "include_objects": config.include_objects,
    }
    return SceneRobotStateObserver(
        sampler=SceneRobotStateSampler(**sampler_kwargs),
        stream=stream,
    )


def _looks_like_single_scene_runtime(runtime: object) -> bool:
    """用最小结构检查区分 SingleSceneRuntime，避免 telemetry 层反向导入具体 runtime 类型。"""

    return hasattr(runtime, "robots_by_id") and hasattr(runtime, "robot_registry")


def _foxglove_sinks(
    config: InteractiveStateStreamConfig,
) -> list[FoxgloveStateSink]:
    """按配置创建一个或多个 Foxglove sink。"""

    mcap_plan = config.mcap_output_plan
    if mcap_plan is None:
        mcap_plan = prepare_mcap_output(
            config.foxglove_mcap_path,
            existing_file_policy=config.mcap_existing_file_policy,
        )
    elif (
        config.foxglove_mcap_path is None
        or mcap_plan.requested_path != Path(config.foxglove_mcap_path).expanduser()
        or mcap_plan.policy != config.mcap_existing_file_policy
    ):
        raise ValueError("prepared MCAP output does not match telemetry configuration")
    if mcap_plan is not None and not config.output_paths_applied:
        apply_output_path_plans((mcap_plan,))

    sinks: list[FoxgloveStateSink] = []
    try:
        if config.foxglove_live_port is not None:
            sinks.append(
                FoxgloveStateSink.open_live(
                    host=config.foxglove_live_host,
                    port=int(config.foxglove_live_port),
                    joint_effort_field=config.foxglove_joint_effort_field,
                    publish_joint_states=config.include_joint_states,
                    publish_state_json=config.include_state_json,
                    publish_scene_markers=config.include_scene_markers,
                    topics=config.topics,
                )
            )
        if mcap_plan is not None:
            sinks.append(
                FoxgloveStateSink.open_mcap(
                    mcap_plan.resolved_path,
                    joint_effort_field=config.foxglove_joint_effort_field,
                    publish_joint_states=config.include_joint_states,
                    publish_state_json=config.include_state_json,
                    publish_scene_markers=config.include_scene_markers,
                    topics=config.topics,
                )
            )
    except BaseException:
        try:
            CompositeStateSink(tuple(reversed(sinks))).close()
        except BaseException:
            pass
        raise
    return sinks


def _print_status(status_prefix: str | None, message: str) -> None:
    """输出状态流启动信息。"""

    if status_prefix is None:
        return
    print(f"{status_prefix}_{message}", flush=True)
