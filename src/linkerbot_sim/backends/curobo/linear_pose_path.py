"""cuRobo linear-pose-path 适配。

本模块只保留项目当前需要的“TCP 位置 + 姿态线性运动”：先把 task-space 线性路径离散成一批
TCP pose，再按路径顺序求解 IK，并把上一 waypoint 的 C-space 解作为下一次 seed，最后构造项目侧
``JointTrajectory``。当前接口只接受线性 pose path，不提供圆弧、composite transition 或其他
指定路径语义，避免生成几何含义不同的近似结果。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from linkerbot_sim.backends.curobo.collision_capability import (
    collision_capability_message,
    context_supports_collision_queries,
)
from linkerbot_sim.backends.curobo.tensor_adapter import (
    seed_config_from_state_or_seed,
    tensor_like_to_numpy,
)
from linkerbot_sim.backends.curobo.tool_pose import (
    goal_tool_pose_from_single_tcp_target,
    update_active_tool_pose_criteria,
)
from linkerbot_sim.planning.requests import (
    LinearPosePathRequest,
    TaskSpacePath,
    TcpLineSegment,
    TcpPoseSequenceSegment,
)
from linkerbot_sim.planning.results import MotionResult, PlanningDiagnostics
from linkerbot_sim.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from linkerbot_sim.utils.rotations import normalize_quat_wxyz_or_identity


def plan_linear_pose_path(
    context,
    request: LinearPosePathRequest,
    *,
    tcp_frame_name: str,
) -> MotionResult:
    """用 cuRobo sequential IK 求解项目侧 ``LinearPosePathRequest``。

    输入的 ``request.path`` 只支持 ``TaskSpacePath``，且 segment 只能是
    ``TcpLineSegment`` 和 ``TcpPoseSequenceSegment``。它们会被离散成 TCP pose 后按顺序 IK。
    """

    try:
        request.validate_structure()
    except ValueError as exc:
        return _unsupported(str(exc))
    if request.avoid_collisions and not context_supports_collision_queries(
        context,
        consumer="ik",
    ):
        return _failed(
            "COLLISION_UNSUPPORTED",
            "cuRobo collision-aware linear pose path cannot satisfy "
            "avoid_collisions=True: "
            + collision_capability_message(context, consumer="ik"),
        )
    duration_s = _request_duration_s(request)
    if not isinstance(request.path, TaskSpacePath):
        return _unsupported(
            f"cuRobo linear-pose path does not support {type(request.path).__name__}"
        )

    frame_name = str(request.tcp_frame_name or tcp_frame_name)
    current_q = np.asarray(request.current_q, dtype=float).reshape(-1)
    try:
        current_position, current_orientation = _current_tcp_pose(
            context,
            current_q,
            tcp_frame_name=frame_name,
        )
        samples = _sample_task_space_linear_path(
            request.path,
            current_position=current_position,
            current_orientation=current_orientation,
            duration_s=duration_s,
            sample_dt_s=_request_sample_dt_s(request),
        )
    except ValueError as exc:
        return _unsupported(str(exc))

    if samples.positions.shape[0] <= 1:
        return _failed("INVALID_PATH", "linear pose path produced no target samples")
    ik_result = _solve_linear_pose_samples(
        context,
        positions=samples.positions[1:],
        orientations_wxyz=samples.orientations_wxyz[1:],
        orientation_free=samples.orientation_free[1:],
        seed=current_q,
        tcp_frame_name=frame_name,
    )
    if isinstance(ik_result, MotionResult):
        return ik_result

    positions = np.vstack([current_q.reshape(1, -1), ik_result])
    trajectory = joint_trajectory_from_positions(
        times=samples.times,
        positions=positions,
        joint_names=tuple(context.joint_names()),
        phase="curobo_linear_pose_path",
    )
    return MotionResult(
        path=positions,
        trajectory=trajectory,
        success=True,
        status="SUCCESS",
        diagnostics=PlanningDiagnostics(
            status="SUCCESS",
            message=(
                "pipeline=curobo linear_pose_path family=task_space "
                f"samples={positions.shape[0]} frame={frame_name}"
            ),
            metrics={
                "samples": float(positions.shape[0]),
                "duration_s": float(duration_s),
            },
        ),
    )


@dataclass(frozen=True)
class _TaskSpaceSamples:
    """离散后的 TCP pose 采样。"""

    times: np.ndarray
    positions: np.ndarray
    orientations_wxyz: np.ndarray
    orientation_free: np.ndarray


def _sample_task_space_linear_path(
    path: TaskSpacePath,
    *,
    current_position: np.ndarray,
    current_orientation: np.ndarray,
    duration_s: float,
    sample_dt_s: float,
) -> _TaskSpaceSamples:
    """把支持的 task-space segment 离散成连续 TCP pose 样本。"""

    positions: list[np.ndarray] = [np.asarray(current_position, dtype=float).reshape(3)]
    orientations: list[np.ndarray] = [
        normalize_quat_wxyz_or_identity(current_orientation)
    ]
    orientation_free: list[bool] = [False]
    segment_count = max(1, len(path.segments))
    segment_duration_s = duration_s / float(segment_count)
    for index, segment in enumerate(path.segments):
        start_position = positions[-1]
        start_orientation = orientations[-1]
        if isinstance(segment, TcpLineSegment):
            _validate_line_start_position(
                start_position,
                segment,
                segment_index=index,
            )
            target_position = _line_target_position(start_position, segment)
            line_orientation_free = _line_orientation_is_free(segment)
            target_orientation = _line_target_orientation(start_orientation, segment)
            _append_linear_pose_segment(
                positions,
                orientations,
                orientation_free,
                start_position=start_position,
                start_orientation=start_orientation,
                target_position=target_position,
                target_orientation=target_orientation,
                target_orientation_free=line_orientation_free,
                duration_s=segment_duration_s,
                sample_dt_s=sample_dt_s,
            )
            continue
        if isinstance(segment, TcpPoseSequenceSegment):
            _append_pose_sequence(
                positions,
                orientations,
                orientation_free,
                segment=segment,
                segment_duration_s=segment_duration_s,
                sample_dt_s=sample_dt_s,
            )
            continue
        raise ValueError(
            "cuRobo linear-pose path does not support "
            f"{type(segment).__name__} at segment {index}"
        )
    times = np.linspace(0.0, duration_s, len(positions))
    return _TaskSpaceSamples(
        times=times,
        positions=np.vstack(positions),
        orientations_wxyz=np.vstack(orientations),
        orientation_free=np.asarray(orientation_free, dtype=bool),
    )


def _append_pose_sequence(
    positions: list[np.ndarray],
    orientations: list[np.ndarray],
    orientation_free: list[bool],
    *,
    segment: TcpPoseSequenceSegment,
    segment_duration_s: float,
    sample_dt_s: float,
) -> None:
    """把 pose sequence 解释为多段线性位姿插值。"""

    pose_count = max(1, len(segment.poses))
    per_pose_duration = segment_duration_s / float(pose_count)
    for pose in segment.poses:
        _append_linear_pose_segment(
            positions,
            orientations,
            orientation_free,
            start_position=positions[-1],
            start_orientation=orientations[-1],
            target_position=np.asarray(pose.position, dtype=float).reshape(3),
            target_orientation=np.asarray(pose.orientation, dtype=float).reshape(4),
            target_orientation_free=False,
            duration_s=per_pose_duration,
            sample_dt_s=sample_dt_s,
        )


def _append_linear_pose_segment(
    positions: list[np.ndarray],
    orientations: list[np.ndarray],
    orientation_free: list[bool],
    *,
    start_position: np.ndarray,
    start_orientation: np.ndarray,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
    target_orientation_free: bool,
    duration_s: float,
    sample_dt_s: float,
) -> None:
    """追加一段位置线性、姿态 Slerp 的 TCP pose 采样。"""

    steps = max(1, int(np.ceil(float(duration_s) / float(sample_dt_s))))
    alpha = np.linspace(0.0, 1.0, steps + 1, dtype=float)[1:]
    start = np.asarray(start_position, dtype=float).reshape(3)
    target = np.asarray(target_position, dtype=float).reshape(3)
    segment_positions = start.reshape(1, 3) + (target - start).reshape(
        1, 3
    ) * alpha.reshape(-1, 1)
    segment_orientations = _slerp_quat_wxyz(
        start_orientation,
        target_orientation,
        alpha,
    )
    positions.extend(np.asarray(row, dtype=float) for row in segment_positions)
    orientations.extend(np.asarray(row, dtype=float) for row in segment_orientations)
    orientation_free.extend([bool(target_orientation_free)] * int(steps))


def _solve_linear_pose_samples(
    context,
    *,
    positions: np.ndarray,
    orientations_wxyz: np.ndarray,
    orientation_free: np.ndarray,
    seed: np.ndarray,
    tcp_frame_name: str,
) -> np.ndarray | MotionResult:
    """顺序 IK 求解离散 TCP pose 样本，并用上一点解 warm-start 下一点。"""

    target_positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    target_orientations = np.asarray(orientations_wxyz, dtype=float).reshape(-1, 4)
    target_orientation_free = np.asarray(orientation_free, dtype=bool).reshape(-1)
    if target_positions.shape[0] != target_orientations.shape[0]:
        return _failed(
            "INVALID_PATH",
            "linear pose path positions/orientations sample count mismatch",
        )
    if target_positions.shape[0] != target_orientation_free.shape[0]:
        return _failed(
            "INVALID_PATH",
            "linear pose path orientation mode sample count mismatch",
        )
    current_seed = np.asarray(seed, dtype=float).reshape(1, -1)
    solutions: list[np.ndarray] = []
    for sample_index, (position, orientation, is_orientation_free) in enumerate(
        zip(target_positions, target_orientations, target_orientation_free, strict=True)
    ):
        seed_config = current_seed.copy()
        target_orientation = None if bool(is_orientation_free) else orientation
        goal = goal_tool_pose_from_single_tcp_target(
            context,
            tcp_frame_name=tcp_frame_name,
            target_position=position,
            target_orientation=target_orientation,
            seed=seed_config,
        )
        update_active_tool_pose_criteria(
            context,
            context.ik_solver,
            active_tool_frame=tcp_frame_name,
            orientation_free=bool(is_orientation_free),
        )
        current_state = context.joint_state_from_positions(seed_config)
        result = context.ik_solver.solve_pose(
            goal,
            current_state=current_state,
            seed_config=seed_config_from_state_or_seed(current_state, seed_config),
        )
        success = _result_success_vector(result, rows=1)
        if not bool(success[0]):
            return _failed(
                "IK_FAILED",
                f"cuRobo linear-pose path IK failed at sample index {sample_index}",
                metrics={"failed_samples": 1.0},
            )
        solution = _result_positions(result, expected_shape=seed_config.shape)
        if solution is None:
            # IK 报成功但没有给出形状正确的解，说明后端结果异常。此时必须显式失败，
            # 不能静默回退成 seed（否则会得到「原地不动」的假成功轨迹）。
            return _failed(
                "INVALID_IK_RESULT",
                "cuRobo linear-pose path IK returned no valid solution at sample "
                f"index {sample_index}",
            )
        current_seed = np.asarray(solution, dtype=float).reshape(1, -1)
        solutions.append(current_seed.reshape(-1).copy())
    if not solutions:
        return np.empty((0, current_seed.shape[1]), dtype=float)
    return np.vstack(solutions)


def _current_tcp_pose(
    context,
    current_q: np.ndarray,
    *,
    tcp_frame_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """读取当前 C-space 下的 TCP position / orientation。"""

    positions, orientations = context.compute_tcp_poses(
        np.asarray(current_q, dtype=float).reshape(1, -1),
        tcp_frame_name=tcp_frame_name,
    )
    return (
        np.asarray(positions, dtype=float).reshape(-1, 3)[0],
        normalize_quat_wxyz_or_identity(
            np.asarray(orientations, dtype=float).reshape(-1, 4)[0]
        ),
    )


def _line_target_position(
    current_position: np.ndarray, segment: TcpLineSegment
) -> np.ndarray:
    """解析直线段目标位置。"""

    if segment.target_position is not None:
        return np.asarray(segment.target_position, dtype=float).reshape(3)
    assert segment.target_offset is not None
    return np.asarray(current_position, dtype=float).reshape(3) + np.asarray(
        segment.target_offset, dtype=float
    ).reshape(3)


def _validate_line_start_position(
    current_position: np.ndarray,
    segment: TcpLineSegment,
    *,
    segment_index: int,
) -> None:
    """校验显式 line 起点与当前 FK/上一段终点连续。"""

    if segment.start_position is None:
        return
    expected = np.asarray(current_position, dtype=float).reshape(3)
    declared = np.asarray(segment.start_position, dtype=float).reshape(3)
    if not np.allclose(declared, expected, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"path.segments[{segment_index}].start_position does not match "
            "the current TCP position or previous segment endpoint"
        )


def _line_target_orientation(
    current_orientation: np.ndarray, segment: TcpLineSegment
) -> np.ndarray:
    """解析直线段目标姿态。

    ``free`` 会在求解阶段切换成 cuRobo position-only criteria；这里仍用起点姿态填充
    合法 quaternion，避免给 ``GoalToolPose`` 传入无效姿态。
    """

    mode = str(segment.orientation_mode)
    if mode in {"free", "current"}:
        return normalize_quat_wxyz_or_identity(current_orientation)
    if mode == "target":
        assert segment.target_orientation is not None
        return normalize_quat_wxyz_or_identity(segment.target_orientation)
    raise ValueError("orientation_mode must be one of: free, current, target")


def _line_orientation_is_free(segment: TcpLineSegment) -> bool:
    """返回 line segment 是否应忽略 TCP orientation constraint。"""

    return str(segment.orientation_mode) == "free"


def _slerp_quat_wxyz(start, target, alpha: np.ndarray) -> np.ndarray:
    """用 SciPy Slerp 插值 wxyz 四元数。"""

    start_wxyz = normalize_quat_wxyz_or_identity(start)
    target_wxyz = normalize_quat_wxyz_or_identity(target)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if alpha.size == 0:
        return np.empty((0, 4), dtype=float)
    if np.allclose(start_wxyz, target_wxyz, atol=1.0e-9):
        return np.repeat(start_wxyz.reshape(1, 4), alpha.size, axis=0)
    rotations = Rotation.from_quat(
        np.asarray(
            [
                [start_wxyz[1], start_wxyz[2], start_wxyz[3], start_wxyz[0]],
                [target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]],
            ],
            dtype=float,
        )
    )
    xyzw = Slerp([0.0, 1.0], rotations)(alpha).as_quat()
    return np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])


def _result_positions(result, *, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    """从 cuRobo/fake IK result 中读取 batch 解；找不到形状匹配的解时返回 ``None``。"""

    for name in ("solution", "joint_positions", "position"):
        value = getattr(result, name, None)
        if value is None:
            continue
        array = tensor_like_to_numpy(value)
        if array.ndim == 3:
            array = array[:, 0, :]
        if array.shape == tuple(expected_shape):
            return np.asarray(array, dtype=float)
    return None


def _result_success_vector(result, *, rows: int) -> np.ndarray:
    """从 cuRobo/fake result 中读取 per-sample success。"""

    value = getattr(result, "success", None)
    if value is None:
        return np.zeros(rows, dtype=bool)
    success = np.asarray(tensor_like_to_numpy(value), dtype=bool)
    if success.ndim > 1:
        success = success.any(axis=tuple(range(1, success.ndim)))
    success = success.reshape(-1)
    if success.size != rows:
        raise ValueError("cuRobo IK success mask has wrong length")
    return success


def _request_duration_s(request: LinearPosePathRequest) -> float:
    """返回指定路径持续时间，缺省 1s。"""

    return 1.0 if request.duration_s is None else float(request.duration_s)


def _request_sample_dt_s(request: LinearPosePathRequest) -> float:
    """返回调用方解析后的采样周期；runtime 必须从 physics dt 注入缺省值。"""

    if request.sample_dt_s is None:
        raise ValueError(
            "linear pose path sample_dt_s is required; inject the runtime physics dt"
        )
    return float(request.sample_dt_s)


def _unsupported(message: str) -> MotionResult:
    """构造 unsupported MotionResult。"""

    return MotionResult(
        path=None,
        trajectory=None,
        success=False,
        status="UNSUPPORTED",
        diagnostics=PlanningDiagnostics(status="UNSUPPORTED", message=message),
    )


def _failed(
    status: str,
    message: str,
    *,
    metrics: dict[str, float] | None = None,
) -> MotionResult:
    """构造失败 MotionResult。"""

    return MotionResult(
        path=None,
        trajectory=None,
        success=False,
        status=str(status),
        diagnostics=PlanningDiagnostics(
            status=str(status),
            message=str(message),
            metrics={} if metrics is None else dict(metrics),
        ),
    )


__all__ = ["plan_linear_pose_path"]
