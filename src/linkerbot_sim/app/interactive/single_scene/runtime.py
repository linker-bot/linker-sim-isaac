"""在仿真主线程执行 canonical robot-ID timeline 的交互 runtime。

transport、状态流和命令队列只负责跨线程传递纯数据；本模块串行处理 snapshot/reset，
编译 timeline，并在每个 physics tick 写 articulation target。这个顺序确保 USD、PhysX
和 cuRobo context 的读写不会从后台网络线程发生。
"""

from __future__ import annotations

from linkerbot_sim.app.interactive.policies import (
    InteractiveRuntimePolicy,
    resolve_interactive_runtime_policy,
)
from linkerbot_sim.app.interactive.single_scene.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.single_scene.state_stream import (
    InteractiveStateStreamConfig,
    start_interactive_state_stream,
)
from linkerbot_sim.configs.runtime import (
    InteractiveRuntimeSettings,
    RuntimeExecutionSettings,
    RuntimePlannerSettings,
    ShutdownSettings,
)
from linkerbot_sim.app.interactive.single_scene.transports import (
    start_interactive_transports,
)
from linkerbot_sim.app.motion.timeline.executor import (
    TimelineExecutionInterrupted,
    TimelinePostStepError,
    execute_robot_timeline,
)
from linkerbot_sim.app.motion.timeline.compiler import TimelinePlanningSession
from linkerbot_sim.app.motion.timeline.requests import (
    JointGroupTrackRequest,
    RobotMotionUnitRequest,
    RobotTimelineRequest,
    RobotTrackRequest,
    TimelineSegmentRequest,
)
from linkerbot_sim.app.runtime.single_scene_reset import (
    SingleSceneResetOptions,
    reset_single_scene_runtime,
)
from linkerbot_sim.app.runtime.single_scene_runtime import SingleSceneRuntime
from linkerbot_sim.snapshots import get_single_scene_snapshot, set_single_scene_snapshot


def run_single_scene_interactive_motion(
    runtime: SingleSceneRuntime,
    *,
    stdin_enabled: bool = True,
    tcp_jsonl_host: str | None = None,
    tcp_jsonl_port: int | None = None,
    websocket_host: str | None = None,
    websocket_port: int | None = None,
    state_stream_config: InteractiveStateStreamConfig | None = None,
    start_step: int = 0,
    planner_backend: str = "curobo",
    policy: InteractiveRuntimePolicy | None = None,
    interactive_settings: InteractiveRuntimeSettings | None = None,
    execution_settings: RuntimeExecutionSettings | None = None,
    planner_settings: RuntimePlannerSettings | None = None,
    shutdown_settings: ShutdownSettings | None = None,
) -> int:
    """在仿真主线程运行 canonical JSON timeline 循环。

    stdin EOF 和空闲 physics 由两个独立策略控制。snapshot/reset 在命令边界优先执行；
    运动命令先完整编译，再进入逐 tick executor。返回值是退出时的 global simulation step。
    """

    if policy is None:
        policy = resolve_interactive_runtime_policy(
            default_stdin_eof_policy="exit",
            default_idle_physics_policy="hold_step",
        )
    interactive_settings = interactive_settings or InteractiveRuntimeSettings()
    execution_settings = execution_settings or RuntimeExecutionSettings()
    planner_settings = planner_settings or RuntimePlannerSettings()
    shutdown_settings = shutdown_settings or ShutdownSettings()

    transport_settings = interactive_settings.transport
    queue = InteractiveMotionQueue(
        request_capacity=transport_settings.request_queue_capacity,
        terminal_history_capacity=interactive_settings.command_history_capacity,
        snapshot_request_capacity=interactive_settings.snapshot_request_capacity,
        snapshot_timeout_s=interactive_settings.snapshot_timeout_s,
        planner_request_defaults=planner_settings.request_defaults,
        command_defaults=execution_settings.command_defaults,
    )
    queue.set_status_provider(runtime.status)
    transports = None
    state_stream = None
    step = int(start_step)
    planner = TimelinePlanningSession(runtime, planner_backend=planner_backend)
    try:
        state_stream = start_interactive_state_stream(
            runtime,
            config=state_stream_config,
            status_prefix=runtime.status_prefix,
        )
        transports = start_interactive_transports(
            queue=queue,
            stdin_enabled=stdin_enabled,
            tcp_jsonl_host=tcp_jsonl_host,
            tcp_jsonl_port=tcp_jsonl_port,
            websocket_host=websocket_host,
            websocket_port=websocket_port,
            stdin_eof_policy=policy.stdin_eof_policy,
            keepalive_consumer_active=(
                state_stream is not None
                or getattr(runtime, "camera_output", None) is not None
            ),
            max_message_bytes=transport_settings.max_message_bytes,
            max_connections=transport_settings.max_connections,
            event_queue_capacity=transport_settings.event_queue_capacity,
            startup_timeout_s=transport_settings.startup_timeout_s,
            server_poll_interval_s=transport_settings.server_poll_interval_s,
            response_poll_interval_s=(transport_settings.response_poll_interval_s),
            shutdown_timeout_s=shutdown_settings.transport_timeout_s,
        )
        print("SINGLE_SCENE_INTERACTIVE_READY", flush=True)
        while not queue.quit_requested():
            if not runtime.session.app.is_running():
                break
            snapshot_request = queue.consume_snapshot_request()
            if snapshot_request is not None:
                if not queue.begin_snapshot_request(snapshot_request):
                    continue
                try:
                    queue.mark_snapshot_done(
                        snapshot_request,
                        _handle_snapshot_request(runtime, snapshot_request),
                    )
                except Exception as exc:
                    queue.mark_snapshot_failed(snapshot_request, str(exc))
                continue
            reset_request = queue.consume_reset_request()
            if reset_request is not None:
                try:
                    result = reset_single_scene_runtime(
                        runtime,
                        options=SingleSceneResetOptions(
                            hold_after_reset=reset_request.hold_after_reset
                        ),
                    )
                    step = result.step
                    if reset_request.hold_after_reset:
                        step = _hold_all(
                            runtime,
                            planner=planner,
                            step=step,
                            duration_s=execution_settings.idle_step_duration_s,
                            should_stop=queue.should_stop_current,
                        )
                    queue.mark_reset_done(reset_request.reset_id, step=step)
                except TimelinePostStepError as exc:
                    step = exc.step
                    queue.mark_reset_failed(reset_request.reset_id, str(exc))
                except Exception as exc:
                    queue.mark_reset_failed(reset_request.reset_id, str(exc))
                continue

            queued = queue.next_pending(
                timeout_s=interactive_settings.queue_poll_timeout_s
            )
            if queued is None:
                if queue.estop_requested():
                    break
                if policy.steps_while_idle:
                    step = _hold_all(
                        runtime,
                        planner=planner,
                        step=step,
                        duration_s=execution_settings.idle_step_duration_s,
                        should_stop=queue.should_stop_current,
                    )
                continue
            try:
                timeline = planner.compile(queued.timeline)
                step = execute_robot_timeline(
                    runtime,
                    timeline,
                    start_step=step,
                    should_stop=queue.should_stop_current,
                )
            except TimelineExecutionInterrupted as exc:
                step = exc.step
                queue.mark_cancelled(
                    queued.command_id,
                    error="interrupted",
                    steps=step,
                )
                if queue.estop_requested():
                    break
                continue
            except TimelinePostStepError as exc:
                step = exc.step
                queue.mark_failed(queued.command_id, str(exc))
                continue
            except Exception as exc:
                queue.mark_failed(queued.command_id, str(exc))
                continue
            queue.mark_done(queued.command_id, steps=step)
    finally:
        queue.request_quit()
        if transports is not None:
            shutdown_report = transports.stop(
                timeout_s=shutdown_settings.transport_timeout_s
            )
            if not shutdown_report.stopped:
                _retain_shutdown_resource(runtime, "interactive_transports", transports)
                print(
                    "SINGLE_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT "
                    f"live_resources={list(shutdown_report.live_resources)}",
                    flush=True,
                )
        if state_stream is not None:
            if not state_stream.close():
                _retain_shutdown_resource(runtime, "state_stream", state_stream)
                print(
                    "SINGLE_SCENE_INTERACTIVE_STATE_SHUTDOWN_TIMEOUT "
                    f"status={state_stream.publisher.status()}",
                    flush=True,
                )
        print("SINGLE_SCENE_INTERACTIVE_EXIT", flush=True)
    return step


def _retain_shutdown_resource(
    runtime: SingleSceneRuntime,
    name: str,
    resource: object,
) -> None:
    """若 runtime 支持托管关闭资源，则把超时未退出的异步句柄移交给它。

    移交后由 runtime 保持强引用并在后续关闭阶段重试；不支持该能力的轻量测试
    runtime 会被安全忽略。
    """

    retain = getattr(runtime, "retain_shutdown_resource", None)
    if callable(retain):
        retain(name, resource)


def _hold_all(
    runtime: SingleSceneRuntime,
    *,
    planner: TimelinePlanningSession,
    step: int,
    duration_s: float,
    should_stop,
) -> int:
    """为每个 robot 的现有 arm/hand group 编译短 hold，并共同推进 idle ticks。"""

    tracks = []
    for robot in runtime.robots_by_id.values():
        groups = []
        for group in ("arm", "hand"):
            if robot.joint_groups.names(group):
                groups.append(
                    JointGroupTrackRequest(
                        group=group,
                        segments=(
                            TimelineSegmentRequest(
                                kind="hold",
                                duration_s=duration_s,
                            ),
                        ),
                    )
                )
        tracks.append(
            RobotTrackRequest(
                robot_id=robot.robot_id,
                robot_label=robot.label,
                units=(RobotMotionUnitRequest(tuple(groups)),),
            )
        )
    request = RobotTimelineRequest(tracks=tuple(tracks))
    return execute_robot_timeline(
        runtime,
        planner.compile(request),
        start_step=step,
        should_stop=should_stop,
    )


def _handle_snapshot_request(runtime: SingleSceneRuntime, request) -> dict[str, object]:
    """在主线程执行 Single Scene snapshot get/set，并生成 transport response payload。"""

    if request.kind == "get_snapshot":
        return {
            "event": "snapshot",
            "accepted": True,
            "backend": "isaac",
            "snapshot": get_single_scene_snapshot(runtime).as_dict(),
        }
    if request.kind == "set_snapshot":
        if request.snapshot is None:
            raise ValueError("set_snapshot requires snapshot")
        result = set_single_scene_snapshot(
            runtime,
            request.snapshot,
            label_map=request.label_map,
            strict=bool(request.strict),
        )
        return {**result.as_dict(), "backend": "isaac"}
    raise ValueError(f"unsupported snapshot request kind: {request.kind!r}")


__all__ = ["run_single_scene_interactive_motion"]
