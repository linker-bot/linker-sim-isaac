"""cuMotion app 入口之间共享的运行期 helper。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.app.cumotion_motion_specs import (
    CumotionMoveSpec,
    MoveSpec,
    default_move_phase,
)
from linkerbot_sim.backends.cumotion.trajectory_sampler import (
    joint_trajectory_from_cumotion,
)
from linkerbot_sim.planning.requests import IKRequest
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.trajectories.types import JointTrajectory


def current_command_from_runtime(runtime) -> np.ndarray:
    """读取 articulation 当前关节位置，并投影到 controller command-space。"""

    positions = np.asarray(runtime.articulation.get_joint_positions(), dtype=float).reshape(-1)
    return positions[np.asarray(runtime.joint_controller.command_indices, dtype=int)]


def cspace_vector_from_command(
    *,
    joint_names: Sequence[str],
    command_joint_names: Sequence[str],
    command: np.ndarray,
    label: str = "command",
) -> np.ndarray:
    """按 cuMotion C-space 关节名从 command-space 向量取值。"""

    values_by_name = command_values_by_name(command_joint_names, command, label=label)
    missing = [str(name) for name in joint_names if str(name) not in values_by_name]
    if missing:
        raise ValueError(
            f"cuMotion C-space joints are missing from {label}-space: {missing}"
        )
    return np.asarray([values_by_name[str(name)] for name in joint_names], dtype=float)


def cspace_goal_to_command_vector(
    *,
    command_joint_names: Sequence[str],
    base_command: np.ndarray,
    joint_names: Sequence[str],
    goal_q: np.ndarray,
) -> np.ndarray:
    """把 C-space goal 中匹配 command-space 的关节写回 command 向量。"""

    target = np.asarray(base_command, dtype=float).reshape(-1).copy()
    command_names = tuple(str(name) for name in command_joint_names)
    command_index_by_name = {name: index for index, name in enumerate(command_names)}
    cspace_index_by_name = {str(name): index for index, name in enumerate(joint_names)}
    goal = np.asarray(goal_q, dtype=float).reshape(-1)
    if goal.size != len(tuple(joint_names)):
        raise ValueError(
            f"goal_q expected {len(tuple(joint_names))} values, got {goal.size}"
        )
    for name in command_names:
        if name in cspace_index_by_name:
            target[command_index_by_name[name]] = goal[cspace_index_by_name[name]]
    return target


def command_indices_for_cspace_joints(
    *,
    command_joint_names: Sequence[str],
    cspace_joint_names: Sequence[str],
    label: str = "arm",
) -> np.ndarray:
    """返回 C-space 轨迹列在 command-space 中的列索引。"""

    command_index_by_name = {
        str(name): index for index, name in enumerate(command_joint_names)
    }
    missing = [
        str(name) for name in cspace_joint_names if str(name) not in command_index_by_name
    ]
    if missing:
        raise ValueError(
            f"{label} joints are missing from command-space joints: {missing}"
        )
    return np.asarray(
        [command_index_by_name[str(name)] for name in cspace_joint_names],
        dtype=int,
    )


def cspace_linear_trajectory(
    *,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    phase: str,
) -> JointTrajectory:
    """为 IK 动作构造一条 C-space 线性插值轨迹。"""

    start = np.asarray(start_q, dtype=float).reshape(-1)
    goal = np.asarray(goal_q, dtype=float).reshape(-1)
    if start.size != goal.size:
        raise ValueError(
            f"C-space start/goal shape mismatch: {start.size} vs {goal.size}"
        )
    times = trajectory_sample_times(duration_s=duration_s, sample_dt=sample_dt)
    alpha = (times / float(duration_s)).reshape(-1, 1) if duration_s > 0 else 1.0
    positions = start.reshape(1, -1) + alpha * (goal - start).reshape(1, -1)
    positions[-1] = goal
    return joint_trajectory_from_positions(
        times=times,
        positions=positions,
        joint_names=tuple(joint_names),
        phase=phase,
    )


def cspace_trajectory_from_motion_result(
    result,
    *,
    joint_names: Sequence[str],
    duration_s: float,
    sample_dt: float,
    phase: str,
) -> JointTrajectory:
    """把 cuMotion MotionResult 转成项目 C-space 轨迹。"""

    if result.trajectory is not None:
        return joint_trajectory_from_cumotion(
            result.trajectory,
            joint_names=tuple(joint_names),
            sample_dt=sample_dt,
            phase=phase,
        )
    if result.path is None:
        raise RuntimeError(
            f"cuMotion planner returned no executable trajectory: status={result.status}"
        )
    path = np.asarray(result.path, dtype=float)
    if path.ndim != 2 or path.shape[0] == 0:
        raise RuntimeError(
            f"cuMotion planner returned an empty path: status={result.status}"
        )
    if path.shape[0] == 1:
        times = np.asarray([max(float(duration_s), float(sample_dt))], dtype=float)
    else:
        times = np.linspace(0.0, float(duration_s), path.shape[0], dtype=float)[1:]
        path = path[1:]
        if path.shape[0] == 0:
            path = np.asarray(result.path, dtype=float)[-1:].copy()
            times = np.asarray([float(duration_s)], dtype=float)
    return joint_trajectory_from_positions(
        times=times,
        positions=path,
        joint_names=tuple(joint_names),
        phase=phase,
    )


def solve_ik_request(context, request: IKRequest, *, tcp_frame_name: str, label: str):
    """执行 IK request，并把失败错误归一化。"""

    request.validate_structure()
    solver = context.make_inverse_kinematics(tcp_frame_name=tcp_frame_name)
    result = cumotion_boundary("IK", solver.solve, request)
    if not result.success:
        raise RuntimeError(
            f"cuMotion {label} IK failed: "
            f"tcp={tcp_frame_name} status={result.status} message={result.message}"
        )
    return result


def duration_for_move(move: MoveSpec) -> float:
    """读取 move 的执行时长。"""

    duration = getattr(move, "duration_s", None)
    if duration is None and isinstance(move, CumotionMoveSpec):
        duration = getattr(move.request, "duration_s", None)
    if duration is None:
        raise ValueError(f"{type(move).__name__} requires duration_s")
    return float(duration)


def explicit_tcp_frame_name(move: MoveSpec) -> str | None:
    """读取 move 或高级 request 中显式给出的 TCP 名称。"""

    value = getattr(move, "tcp_frame_name", None)
    if value is not None and str(value):
        return str(value)
    if isinstance(move, CumotionMoveSpec):
        request_tcp = getattr(move.request, "tcp_frame_name", None)
        if request_tcp is not None and str(request_tcp):
            return str(request_tcp)
    return None


def phase_for_move(
    move: MoveSpec,
    *,
    side: str | None = None,
    dual_cspace: bool = False,
) -> str:
    """读取或生成 move phase。"""

    phase = getattr(move, "phase", None)
    if phase is not None and str(phase):
        return str(phase)
    return default_move_phase(move, side=side, dual_cspace=dual_cspace)


def cumotion_boundary(label: str, func, *args, **kwargs):
    """把 cuMotion/Kit 边界的静默 SystemExit 转成测试失败。"""

    try:
        return func(*args, **kwargs)
    except SystemExit as exc:
        raise RuntimeError(
            f"cuMotion {label} requested process exit: code={exc.code!r}"
        ) from exc


def command_values_by_name(
    command_joint_names: Sequence[str],
    command: np.ndarray,
    *,
    label: str,
) -> dict[str, float]:
    """把 command-space 向量转成按关节名索引的字典。"""

    names = tuple(str(name) for name in command_joint_names)
    values = np.asarray(command, dtype=float).reshape(-1)
    if len(names) != values.size:
        raise ValueError(
            f"{label} command shape mismatch: {len(names)} names, {values.size} values"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{label} command_joint_names contains duplicates")
    return {name: float(value) for name, value in zip(names, values)}


def trajectory_sample_times(*, duration_s: float, sample_dt: float) -> np.ndarray:
    duration = max(float(duration_s), float(sample_dt))
    dt = max(float(sample_dt), 1.0e-6)
    steps = max(1, int(np.ceil(duration / dt)))
    return np.asarray(
        [min(duration, (index + 1) * dt) for index in range(steps)],
        dtype=float,
    )
