"""Interactive single-arm motion runtime."""

from __future__ import annotations

from linkerbot_sim.app.interactive.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.state_stream import (
    InteractiveStateStreamConfig,
    start_interactive_state_stream,
)
from linkerbot_sim.app.interactive.transports import start_interactive_transports
from linkerbot_sim.app.motion.single_arm import (
    DEFAULT_HOLD_REFRESH_DURATION_S,
    SingleArmCuMotionExecutionSession,
    current_command,
)
from linkerbot_sim.app.runtime.reset import (
    RuntimeResetOptions,
    reset_single_robot_runtime,
)
from linkerbot_sim.app.runtime.single_robot import SingleRobotRuntime
from linkerbot_sim.backends.cumotion.context import default_tcp_frame_name
from linkerbot_sim.execution.steps import (
    CommandExecutionInterrupted,
    HoldCommandPositionTargetStep,
)
from linkerbot_sim.snapshots import (
    get_single_robot_snapshot,
    set_single_robot_snapshot,
)


def run_interactive_single_arm_motion(
    runtime: SingleRobotRuntime,
    *,
    cumotion_profile: str = "default",
    stdin_enabled: bool = True,
    tcp_jsonl_host: str | None = None,
    tcp_jsonl_port: int | None = None,
    websocket_host: str | None = None,
    websocket_port: int | None = None,
    state_stream_config: InteractiveStateStreamConfig | None = None,
    start_step: int = 0,
) -> int:
    """Run a long-lived interactive single-arm motion loop."""

    del cumotion_profile  # SingleRobotRuntime already bakes this profile at creation.
    queue = InteractiveMotionQueue()
    default_tcp = default_tcp_frame_name(runtime.robot_cumotion)
    if default_tcp is None:
        raise ValueError("single-arm runtime has no default TCP/frame")
    default_tcp_by_side = {
        "left": default_tcp,
        "right": default_tcp,
    }
    transports = None
    state_stream = None
    step = int(start_step)
    try:
        state_stream = start_interactive_state_stream(
            runtime,
            config=state_stream_config,
            status_prefix=runtime.status_prefix,
        )
        with SingleArmCuMotionExecutionSession(runtime) as session:
            session.step = step
            transports = start_interactive_transports(
                queue=queue,
                default_tcp_by_side=default_tcp_by_side,
                default_side="left",
                stdin_enabled=stdin_enabled,
                tcp_jsonl_host=tcp_jsonl_host,
                tcp_jsonl_port=tcp_jsonl_port,
                websocket_host=websocket_host,
                websocket_port=websocket_port,
            )
            print("SINGLE_ARM_INTERACTIVE_READY", flush=True)
            while not queue.quit_requested():
                if _simulation_app_stopped(runtime):
                    break
                snapshot_request = queue.consume_snapshot_request()
                if snapshot_request is not None:
                    # snapshot 会直接读写 articulation/prim 状态，必须在仿真主循环中处理；
                    # 放在 motion queue 之前消费，能让外部调试端快速读取或恢复当前状态。
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
                        queue.emit(
                            {
                                "event": "reset_started",
                                **reset_request.snapshot(),
                            }
                        )
                        print(
                            "SINGLE_ARM_INTERACTIVE_RESET_STARTED "
                            f"id={reset_request.reset_id}",
                            flush=True,
                        )
                        result = reset_single_robot_runtime(
                            runtime,
                            options=RuntimeResetOptions(
                                hold_after_reset=reset_request.hold_after_reset
                            ),
                        )
                        step = int(result.step)
                        if reset_request.hold_after_reset:
                            step = _hold_for_duration(
                                runtime,
                                step=step,
                                duration_s=DEFAULT_HOLD_REFRESH_DURATION_S,
                                should_stop=queue.should_stop_current,
                            )
                        session.step = step
                        queue.mark_reset_done(reset_request.reset_id, step=step)
                        print(
                            "SINGLE_ARM_INTERACTIVE_RESET_DONE "
                            f"id={reset_request.reset_id} step={step}",
                            flush=True,
                        )
                        continue
                    except Exception as exc:
                        queue.mark_reset_failed(reset_request.reset_id, str(exc))
                        print(
                            "SINGLE_ARM_INTERACTIVE_RESET_FAILED "
                            f"id={reset_request.reset_id} "
                            f"error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue
                queued = queue.next_pending(timeout_s=0.05)
                if queued is None:
                    if queue.estop_requested():
                        break
                    try:
                        step = _refresh_hold_once(
                            runtime,
                            step=step,
                            should_stop=queue.should_stop_current,
                        )
                    except CommandExecutionInterrupted as exc:
                        if exc.step is not None:
                            step = int(exc.step)
                        if queue.estop_requested():
                            break
                    session.step = step
                    continue
                print(
                    f"SINGLE_ARM_INTERACTIVE_RUNNING id={queued.command_id}", flush=True
                )
                try:
                    if queued.moves:
                        step = session.execute_moves(
                            queued.moves,
                            start_step=step,
                            should_stop=queue.should_stop_current,
                        )
                    else:
                        step = _hold_for_duration(
                            runtime,
                            step=step,
                            duration_s=queued.duration_s
                            or DEFAULT_HOLD_REFRESH_DURATION_S,
                            should_stop=queue.should_stop_current,
                        )
                    session.step = step
                except CommandExecutionInterrupted as exc:
                    if exc.step is not None:
                        step = int(exc.step)
                        session.step = step
                    queue.mark_cancelled(
                        queued.command_id,
                        error="interrupted",
                        steps=step,
                    )
                    print(
                        f"SINGLE_ARM_INTERACTIVE_CANCELLED id={queued.command_id}",
                        flush=True,
                    )
                    if queue.estop_requested():
                        print(
                            f"SINGLE_ARM_INTERACTIVE_ESTOP id={queued.command_id}",
                            flush=True,
                        )
                        break
                    continue
                except Exception as exc:
                    queue.mark_failed(queued.command_id, str(exc))
                    print(
                        "SINGLE_ARM_INTERACTIVE_FAILED "
                        f"id={queued.command_id} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                queue.mark_done(queued.command_id, steps=step)
                print(
                    f"SINGLE_ARM_INTERACTIVE_DONE id={queued.command_id} steps={step}",
                    flush=True,
                )
    finally:
        if state_stream is not None:
            state_stream.close()
        if transports is not None:
            transports.stop()
        print("SINGLE_ARM_INTERACTIVE_EXIT", flush=True)
    return step


def _refresh_hold_once(
    runtime: SingleRobotRuntime,
    *,
    step: int,
    should_stop,
) -> int:
    """空闲时刷新一次保持命令，让 GUI/Foxglove 状态继续向前推进。"""

    if runtime.execution.simulation_app is not None:
        return _hold_for_duration(
            runtime,
            step=step,
            duration_s=DEFAULT_HOLD_REFRESH_DURATION_S,
            should_stop=should_stop,
        )
    return step


def _hold_for_duration(
    runtime: SingleRobotRuntime,
    *,
    step: int,
    duration_s: float,
    should_stop,
) -> int:
    """在指定时间内保持当前 command position，不生成新的规划目标。"""

    current = current_command(runtime.execution)
    return HoldCommandPositionTargetStep(
        target_command=current,
        duration=duration_s,
        phase="single_interactive_hold",
        should_stop=should_stop,
    ).run(runtime.execution, step)


def _simulation_app_stopped(runtime: SingleRobotRuntime) -> bool:
    """判断 Isaac SimulationApp 是否已经被用户关闭。"""

    app = runtime.execution.simulation_app
    return app is not None and not app.is_running()


def _handle_snapshot_request(
    runtime: SingleRobotRuntime,
    request,
) -> dict[str, object]:
    """在仿真主线程处理 single-arm snapshot 请求。

    ``get_snapshot`` 读取当前 single scene；``set_snapshot`` 会先做兼容性检查，再按
    ``robot_map``/``strict`` 写回 command joints 和 objects。
    """

    if request.kind == "get_snapshot":
        return {
            "event": "snapshot",
            "accepted": True,
            "backend": "single",
            "snapshot": get_single_robot_snapshot(runtime).as_dict(),
        }
    if request.kind == "set_snapshot":
        if request.snapshot is None:
            raise ValueError("set_snapshot requires snapshot")
        result = set_single_robot_snapshot(
            runtime,
            request.snapshot,
            robot_map=request.robot_map,
            strict=bool(request.strict),
        )
        return {**result.as_dict(), "backend": "single"}
    raise ValueError(f"unsupported snapshot request kind: {request.kind!r}")
