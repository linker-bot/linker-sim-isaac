"""Interactive dual-arm motion runtime."""

from __future__ import annotations

from linkerbot_sim.app.motion.specs import DualArmTcpSpec
from linkerbot_sim.app.motion.dual_arm import (
    DEFAULT_HOLD_REFRESH_DURATION_S,
    DualArmCuMotionExecutionSession,
    current_command,
)
from linkerbot_sim.app.runtime.dual_robot import DualRobotAppRuntime
from linkerbot_sim.app.interactive.queue import InteractiveMotionQueue
from linkerbot_sim.app.interactive.transports import start_interactive_transports
from linkerbot_sim.execution.dual_steps import (
    DualCommandExecutionInterrupted,
    DualCommandPositionTargetStep,
)


def run_interactive_dual_arm_motion(
    runtime: DualRobotAppRuntime,
    *,
    tcp: DualArmTcpSpec,
    cumotion_profile: str = "default",
    dual_arm_profile: str = "ar5v2_l6v1_dual",
    stdin_enabled: bool = True,
    tcp_jsonl_host: str | None = None,
    tcp_jsonl_port: int | None = None,
    websocket_host: str | None = None,
    websocket_port: int | None = None,
    start_step: int = 0,
) -> int:
    """Run a long-lived interactive motion loop."""

    queue = InteractiveMotionQueue()
    default_tcp_by_side = {
        "left": tcp.left.frame_name,
        "right": tcp.right.frame_name,
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
    step = int(start_step)
    print("DUAL_ARM_INTERACTIVE_READY", flush=True)
    try:
        with DualArmCuMotionExecutionSession(
            runtime,
            tcp=tcp,
            cumotion_profile=cumotion_profile,
            dual_arm_profile=dual_arm_profile,
        ) as session:
            session.step = step
            while not queue.quit_requested():
                if _simulation_app_stopped(runtime):
                    break
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
