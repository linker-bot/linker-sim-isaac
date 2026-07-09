"""Interactive dual-arm motion runtime."""

from __future__ import annotations

from linkerbot_sim.app.motion.dual_arm import (
    DEFAULT_HOLD_REFRESH_DURATION_S,
    DualArmCuMotionExecutionSession,
    current_command,
)
from linkerbot_sim.app.motion.dual_arm_semantics import (
    dual_arm_semantics_from_robot_configs,
)
from linkerbot_sim.app.runtime.dual_robot import DualRobotAppRuntime
from linkerbot_sim.app.runtime.reset import (
    RuntimeResetOptions,
    reset_dual_robot_runtime,
)
from linkerbot_sim.app.interactive.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.state_stream import (
    InteractiveStateStreamConfig,
    start_interactive_state_stream,
)
from linkerbot_sim.app.interactive.transports import start_interactive_transports
from linkerbot_sim.execution.dual_steps import (
    DualCommandExecutionInterrupted,
    DualCommandPositionTargetStep,
)
from linkerbot_sim.snapshots import (
    get_dual_robot_snapshot,
    set_dual_robot_snapshot,
)


def run_interactive_dual_arm_motion(
    runtime: DualRobotAppRuntime,
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
    """Run a long-lived interactive motion loop."""

    queue = InteractiveMotionQueue()
    semantics = dual_arm_semantics_from_robot_configs(runtime.side_robot_configs)
    default_tcp_by_side = {
        "left": semantics.left_default_tcp_frame,
        "right": semantics.right_default_tcp_frame,
    }
    transports = start_interactive_transports(
        queue=queue,
        default_tcp_by_side=default_tcp_by_side,
        stdin_enabled=stdin_enabled,
        tcp_jsonl_host=tcp_jsonl_host,
        tcp_jsonl_port=tcp_jsonl_port,
        websocket_host=websocket_host,
        websocket_port=websocket_port,
    )
    state_stream = None
    step = int(start_step)
    try:
        state_stream = start_interactive_state_stream(
            runtime,
            config=state_stream_config,
            status_prefix=runtime.status_prefix,
        )
        print("DUAL_ARM_INTERACTIVE_READY", flush=True)
        with DualArmCuMotionExecutionSession(
            runtime,
            cumotion_profile=cumotion_profile,
        ) as session:
            session.step = step
            while not queue.quit_requested():
                if _simulation_app_stopped(runtime):
                    break
                snapshot_request = queue.consume_snapshot_request()
                if snapshot_request is not None:
                    # snapshot 读写会触碰左右 execution 的 articulation/prim 状态；放到主循环
                    # 中处理，避免 transport 线程和 PhysX step 交错。
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
                            "DUAL_ARM_INTERACTIVE_RESET_STARTED "
                            f"id={reset_request.reset_id}",
                            flush=True,
                        )
                        result = reset_dual_robot_runtime(
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
                            "DUAL_ARM_INTERACTIVE_RESET_DONE "
                            f"id={reset_request.reset_id} step={step}",
                            flush=True,
                        )
                        continue
                    except Exception as exc:
                        queue.mark_reset_failed(reset_request.reset_id, str(exc))
                        print(
                            "DUAL_ARM_INTERACTIVE_RESET_FAILED "
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
                    except DualCommandExecutionInterrupted as exc:
                        if exc.step is not None:
                            step = int(exc.step)
                        if queue.estop_requested():
                            break
                    session.step = step
                    continue
                print(f"DUAL_ARM_INTERACTIVE_RUNNING id={queued.command_id}", flush=True)
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
                except DualCommandExecutionInterrupted as exc:
                    if exc.step is not None:
                        step = int(exc.step)
                        session.step = step
                    queue.mark_cancelled(
                        queued.command_id,
                        error="interrupted",
                        steps=step,
                    )
                    print(
                        f"DUAL_ARM_INTERACTIVE_CANCELLED id={queued.command_id}",
                        flush=True,
                    )
                    if queue.estop_requested():
                        print(
                            f"DUAL_ARM_INTERACTIVE_ESTOP id={queued.command_id}",
                            flush=True,
                        )
                        break
                    continue
                except Exception as exc:
                    queue.mark_failed(queued.command_id, str(exc))
                    print(
                        "DUAL_ARM_INTERACTIVE_FAILED "
                        f"id={queued.command_id} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                queue.mark_done(queued.command_id, steps=step)
                print(
                    f"DUAL_ARM_INTERACTIVE_DONE id={queued.command_id} steps={step}",
                    flush=True,
                )
    finally:
        if state_stream is not None:
            state_stream.close()
        transports.stop()
        print("DUAL_ARM_INTERACTIVE_EXIT", flush=True)
    return step


def _refresh_hold_once(
    runtime: DualRobotAppRuntime,
    *,
    step: int,
    should_stop,
) -> int:
    """空闲时刷新一次保持命令，让 GUI 模式下仿真持续向前推进。"""

    if runtime.execution.simulation_app is not None:
        return _hold_for_duration(
            runtime,
            step=step,
            duration_s=DEFAULT_HOLD_REFRESH_DURATION_S,
            should_stop=should_stop,
        )
    return step


def _hold_for_duration(
    runtime: DualRobotAppRuntime,
    *,
    step: int,
    duration_s: float,
    should_stop,
) -> int:
    """在指定时间内保持左右臂当前 command position，不生成新的规划目标。"""

    current_left = current_command(runtime.execution.left)
    current_right = current_command(runtime.execution.right)
    return DualCommandPositionTargetStep(
        left_start_command=current_left,
        right_start_command=current_right,
        left_target_command=current_left,
        right_target_command=current_right,
        duration=duration_s,
        phase="dual_interactive_hold",
        should_stop=should_stop,
    ).run(runtime.execution, step)


def _simulation_app_stopped(runtime: DualRobotAppRuntime) -> bool:
    """判断 Isaac SimulationApp 是否已经被用户关闭。"""

    app = runtime.execution.simulation_app
    return app is not None and not app.is_running()


def _handle_snapshot_request(
    runtime: DualRobotAppRuntime,
    request,
) -> dict[str, object]:
    """在仿真主线程处理 dual-arm snapshot 请求。

    dual snapshot 默认按 ``left``/``right`` role 恢复；如果来源是 single 或只想恢复某一
    侧，则由请求里的 ``robot_map`` 指定 source role 到目标侧的映射。
    """

    if request.kind == "get_snapshot":
        return {
            "event": "snapshot",
            "accepted": True,
            "backend": "dual",
            "snapshot": get_dual_robot_snapshot(runtime).as_dict(),
        }
    if request.kind == "set_snapshot":
        if request.snapshot is None:
            raise ValueError("set_snapshot requires snapshot")
        result = set_dual_robot_snapshot(
            runtime,
            request.snapshot,
            robot_map=request.robot_map,
            strict=bool(request.strict),
        )
        return {**result.as_dict(), "backend": "dual"}
    raise ValueError(f"unsupported snapshot request kind: {request.kind!r}")
