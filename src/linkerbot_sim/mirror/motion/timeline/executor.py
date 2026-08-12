"""Mirror 多机器人 timeline 的主线程控制写入与 world execution loop。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.mirror.motion.timeline.model import RobotTimeline
from linkerbot_sim.controllers.types import ControlTargets
from linkerbot_sim.snapshots.transactions import require_runtime_mutable
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


class TimelineExecutionInterrupted(RuntimeError):
    """携带已完成全局 step 的协作式中断。"""

    def __init__(self, message: str, *, step: int) -> None:
        super().__init__(message)
        self.step = int(step)


class TimelinePostStepError(RuntimeError):
    """表示物理步已提交，但后处理失败；``step`` 是真实累计步数。"""

    def __init__(self, message: str, *, step: int) -> None:
        super().__init__(message)
        self.step = int(step)


def execute_robot_timeline(
    runtime: object,
    timeline: RobotTimeline,
    *,
    start_step: int = 0,
    should_stop: Callable[[], bool] | None = None,
    render_frame: Callable[[], object] | None = None,
    before_step: Callable[[float], None] | None = None,
) -> int:
    """每个 tick 先写完所有 robot targets，再共同推进一次 world。

    ``render_frame`` 由产品根绑定到唯一 ``RenderCoordinator``。Timeline 不直接要求
    physics adapter 在 ``step`` 内渲染，否则 Newton 会跳过单快照、多 camera
    render budget 与 viewport 轮转合同。
    """

    require_runtime_mutable(runtime, operation="execute_robot_timeline")
    world = _mirror_physics(runtime)
    runtime_dt = float(world.get_physics_dt())
    if not np.isclose(runtime_dt, timeline.physics_dt, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"timeline physics_dt={timeline.physics_dt} "
            f"does not match world dt={runtime_dt}"
        )
    states = {
        track.robot_id: _ExecutionState.from_robot(
            _mirror_robot(runtime, track.robot_id, track.robot_label)
        )
        for track in timeline.tracks
    }
    step = int(start_step)
    try:
        for tick in range(int(timeline.duration_ticks)):
            if should_stop is not None and should_stop():
                raise TimelineExecutionInterrupted(
                    "timeline execution interrupted", step=step
                )
            if not _simulation_running(runtime):
                break
            tick_phases = []
            pending: list[tuple[_ExecutionState, ControlTargets]] = []
            for track in timeline.tracks:
                state = states[track.robot_id]
                unit = track.active_unit(tick)
                velocities = np.zeros(len(state.command_names), dtype=float)
                efforts = np.zeros(len(state.command_names), dtype=float)
                if unit is not None:
                    sampled = unit.sample(tick, state.command_by_name())
                    for name, (position, velocity, effort, phase) in sampled.items():
                        index = state.index_by_name.get(name)
                        if index is None:
                            raise ValueError(
                                f"timeline joint {name!r} is not a command joint "
                                f"of robot {track.robot_id}"
                            )
                        state.command[index] = position
                        velocities[index] = velocity
                        efforts[index] = effort
                        tick_phases.append(f"{track.robot_id}:{phase}")
                targets = state.controller.build_control_targets(
                    command_positions=state.command,
                    command_velocities=velocities,
                    command_efforts=efforts,
                    base_positions=state.base_positions,
                )
                pending.append((state, targets))

            # 先完成所有 articulation 写入，再推进共享 world，确保多机器人同 tick。
            for state, targets in pending:
                state.controller.apply_targets(state.action_type, targets)
                state.base_positions = targets.positions.copy()
            # 物理始终只推进一次且不在 concrete runtime 内隐式渲染。若产品启用 renderer，
            # 下面在已完成 step 边界调用同一个 RenderCoordinator；idle 与 timeline 因而
            # 共享 Newton 的一次 pre_render + N 次纯 render_update 事务。
            if before_step is not None:
                before_step(runtime_dt)
            world.step(render=False)
            claim_completed_step = getattr(runtime, "claim_completed_step", None)
            sample_step = (
                int(claim_completed_step()) if callable(claim_completed_step) else step
            )
            step += 1
            collision_registry = getattr(runtime, "collision_registry", None)
            mark_dirty = getattr(collision_registry, "mark_dirty", None)
            phase = ",".join(tick_phases) if tick_phases else "timeline_hold"
            try:
                if callable(mark_dirty):
                    mark_dirty()
                if render_frame is not None:
                    render_frame()
                for state, targets in pending:
                    _write_log(
                        state,
                        targets,
                        step=sample_step,
                        phase=phase,
                        physics_dt=runtime_dt,
                    )
                _observe_once(
                    runtime,
                    states.values(),
                    step=sample_step,
                    phase=phase,
                )
            except Exception as exc:
                raise TimelinePostStepError(
                    f"timeline post-step processing failed: {exc}",
                    step=step,
                ) from exc
    finally:
        # 正常、取消和异常退出均通过当前逻辑模式下发 neutral target。直接改实际 qd
        # 既不能清除 drive velocity target，也不能清除 direct effort target。
        for state in states.values():
            actual_positions = tensor_like_to_numpy(
                state.articulation.get_joint_positions(), dtype=float
            ).reshape(-1)
            zeros = np.zeros(len(state.command_names), dtype=float)
            neutral = state.controller.build_control_targets(
                command_positions=actual_positions[
                    np.asarray(state.controller.command_indices, dtype=int)
                ],
                command_velocities=zeros,
                command_efforts=zeros,
                base_positions=actual_positions,
            )
            state.controller.apply_targets(state.action_type, neutral)
    return step


@dataclass
class _ExecutionState:
    """执行循环持有的单机器人 command-space 可变状态。"""

    robot_id: int
    articulation: object
    controller: object
    action_type: object
    execution: object
    command_names: tuple[str, ...]
    index_by_name: dict[str, int]
    command: np.ndarray
    base_positions: np.ndarray
    num_dof: int

    @classmethod
    def from_robot(cls, robot: object) -> "_ExecutionState":
        """从 RobotRuntime/ExecutionRuntime 冻结 executor 所需的 command-space 索引。"""

        execution = getattr(robot, "execution", robot)
        articulation = execution.articulation
        controller = execution.joint_controller
        names = tuple(str(name) for name in controller.command_joint_names)
        full = tensor_like_to_numpy(
            articulation.get_joint_positions(), dtype=float
        ).reshape(-1)
        indices = np.asarray(controller.command_indices, dtype=int)
        if full.size != int(getattr(articulation, "num_dof", full.size)):
            raise ValueError("articulation joint position size mismatch")
        return cls(
            robot_id=int(getattr(robot, "robot_id", -1)),
            articulation=articulation,
            controller=controller,
            action_type=execution.articulation_action_type,
            execution=execution,
            command_names=names,
            index_by_name={name: index for index, name in enumerate(names)},
            command=full[indices].copy(),
            base_positions=full.copy(),
            num_dof=full.size,
        )

    def command_by_name(self) -> dict[str, float]:
        """把当前可变 command vector 映射回 joint-name keyed baseline。"""

        return dict(zip(self.command_names, self.command, strict=True))


def _mirror_physics(runtime: object):
    """解析 MirrorSceneResources 或测试 double 的唯一 physics adapter。"""

    physics = getattr(runtime, "physics", None)
    if physics is None:
        physics = getattr(runtime, "simulation_world", None)
    if physics is None:
        raise RuntimeError("timeline runtime 缺少 physics adapter")
    return physics


def _mirror_robot(runtime: object, robot_id: int, label: str | None):
    """按 robot ID 解析执行对象，并可校验 timeline 冻结的 label assertion。"""

    resolver = getattr(runtime, "robot", None)
    robot = resolver(robot_id) if callable(resolver) else runtime.robots_by_id[robot_id]
    actual_label = getattr(robot, "label", None)
    if label is not None and actual_label != label:
        raise ValueError(
            f"robot_id {robot_id} is label {actual_label!r}, not {label!r}"
        )
    return robot


def _simulation_running(runtime: object) -> bool:
    """查询 SimulationApp 生命周期；测试 runtime 没有 app 时视为继续运行。"""

    app = getattr(getattr(runtime, "session", None), "app", None)
    if app is None:
        return True
    method = getattr(app, "is_running", None)
    return True if not callable(method) else bool(method())


def _observe_once(runtime: object, states, *, step: int, phase: str) -> None:
    """在全部 articulation 写入并 step 后，各调用一次 state/camera observer。"""

    canonical_observer = getattr(runtime, "observe_after_step", None)
    if callable(canonical_observer):
        canonical_observer(step=step, phase=phase, write_idle_logs=False)
        return

    state_observer = getattr(runtime, "state_observer", None)
    camera_observer = getattr(runtime, "camera_observer", None)
    states = tuple(states)
    if state_observer is None and states:
        state_observer = getattr(states[0].execution, "state_observer", None)
    if camera_observer is None and states:
        camera_observer = getattr(states[0].execution, "camera_observer", None)
    observe = getattr(state_observer, "observe", None)
    if callable(observe):
        observe(runtime, step=step, phase=phase)
    observe_camera = getattr(camera_observer, "observe", None)
    if callable(observe_camera):
        observe_camera(_mirror_physics(runtime), step=step, phase=phase)


def _write_log(
    state: _ExecutionState,
    targets: ControlTargets,
    *,
    step: int,
    phase: str,
    physics_dt: float,
) -> None:
    """按 logger decimation 采集实际/目标状态，并写入当前 timeline phase。"""

    logger = getattr(state.execution, "drive_logger", None)
    if logger is None or not logger.should_write(step):
        return
    indices = np.asarray(state.controller.driven_indices, dtype=int)
    values = logger.collect_step_values(
        state.articulation, state.controller, targets, indices
    )
    logger.write(
        step=step,
        time_s=(step + 1) * physics_dt,
        phase=phase,
        drive_update=True,
        **values,
    )


__all__ = [
    "TimelineExecutionInterrupted",
    "TimelinePostStepError",
    "execute_robot_timeline",
]
