"""cuRobo GoalToolPose 构造、多 TCP 填充与 active tool criteria。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.backends.curobo.runtime_imports import (
    import_curobo_public,
    import_torch_module,
)
from linkerbot_sim.utils.rotations import normalize_quat_wxyz_or_identity


def goal_tool_pose_from_arrays(
    *,
    positions: np.ndarray,
    orientations_wxyz: np.ndarray | None,
    tool_frames: Sequence[str],
    device=None,
    dtype=None,
):
    """把 ``(B,L,3/4)`` 数组转换为 cuRobo ``GoalToolPose``。"""

    torch = import_torch_module()
    types = import_curobo_public("types")
    tensor_dtype = dtype if dtype is not None else torch.float32
    frame_names = tuple(str(name) for name in tool_frames)
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 2:
        pos = pos[:, None, :]
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError("positions must have shape (B, 3) or (B, L, 3)")
    if pos.shape[1] != len(frame_names):
        raise ValueError("positions link dimension must match tool_frames")
    if orientations_wxyz is None:
        quat = np.zeros((pos.shape[0], pos.shape[1], 4), dtype=float)
        quat[..., 0] = 1.0
    else:
        quat = np.asarray(orientations_wxyz, dtype=float)
        if quat.ndim == 2:
            quat = quat[:, None, :]
        if quat.ndim != 3 or quat.shape[2] != 4:
            raise ValueError("orientations_wxyz must have shape (B, 4) or (B, L, 4)")
        if quat.shape[:2] != pos.shape[:2]:
            raise ValueError("orientation batch/link dimensions must match positions")
        flat = quat.reshape(-1, 4)
        quat = np.vstack(
            [normalize_quat_wxyz_or_identity(row) for row in flat]
        ).reshape(quat.shape)
    return types.GoalToolPose(
        tool_frames=list(frame_names),
        position=torch.as_tensor(
            pos[:, None, :, None, :], device=device, dtype=tensor_dtype
        ),
        quaternion=torch.as_tensor(
            quat[:, None, :, None, :], device=device, dtype=tensor_dtype
        ),
    )


def goal_tool_pose_from_single_tcp_target(
    context,
    *,
    tcp_frame_name: str,
    target_position,
    target_orientation,
    seed: np.ndarray | None,
):
    """构造单 active TCP goal，并用 seed FK 填充其它 TCP 当前姿态。"""

    frame_name = str(tcp_frame_name)
    frame_names = tuple(str(name) for name in context.frame_names())
    if frame_name not in set(frame_names):
        raise ValueError(f"cuRobo frame {frame_name!r} not found")
    if len(frame_names) <= 1:
        return context.goal_tool_pose_from_arrays(
            positions=np.asarray(target_position, dtype=float).reshape(1, 3),
            orientations_wxyz=_active_orientation_array(
                context,
                frame_name=frame_name,
                target_orientation=target_orientation,
                seed=seed,
            ),
            tool_frames=(frame_name,),
        )
    if seed is None:
        raise ValueError(
            "multi-link cuRobo IK requires warm_start_ik_cspace_seed "
            "to preserve inactive TCP poses"
        )
    seed_matrix = np.asarray(seed, dtype=float).reshape(1, -1)
    positions, orientations = _current_tool_pose_arrays(
        context, seed_matrix, frame_names=frame_names
    )
    active_index = frame_names.index(frame_name)
    positions[0, active_index] = np.asarray(target_position, dtype=float).reshape(3)
    if target_orientation is not None:
        orientations[0, active_index] = np.asarray(
            target_orientation, dtype=float
        ).reshape(4)
    return context.goal_tool_pose_from_arrays(
        positions=positions,
        orientations_wxyz=orientations,
        tool_frames=frame_names,
    )


def update_active_tool_pose_criteria(
    context,
    solver,
    *,
    active_tool_frame: str,
    orientation_free: bool,
    tool_frames: Sequence[str] | None = None,
) -> bool:
    """为 active TCP 配置 position-only/full-pose tracking criteria。"""

    update = getattr(solver, "update_tool_pose_criteria", None)
    if not callable(update):
        return False
    criteria_type = _tool_pose_criteria_type(context)
    if criteria_type is None:
        return False
    active = str(active_tool_frame)
    frame_names = (
        tuple(str(name) for name in tool_frames)
        if tool_frames is not None
        else _context_frame_names(context)
    )
    if active not in frame_names:
        frame_names = (active,)
    criteria = {
        frame_name: _criteria_for_frame(
            criteria_type,
            position_only=bool(orientation_free and frame_name == active),
        )
        for frame_name in frame_names
    }
    update(criteria)
    return True


def _active_orientation_array(
    context,
    *,
    frame_name: str,
    target_orientation,
    seed: np.ndarray | None,
) -> np.ndarray | None:
    """优先使用显式目标姿态，否则尝试从 seed 计算 active TCP 当前姿态。"""

    if target_orientation is not None:
        return np.asarray(target_orientation, dtype=float).reshape(1, 4)
    if seed is None:
        return None
    try:
        _position, orientation = context.compute_tcp_poses(
            np.asarray(seed, dtype=float).reshape(1, -1),
            tcp_frame_name=frame_name,
        )
    except (AttributeError, ValueError):
        return None
    return np.asarray(orientation, dtype=float).reshape(-1, 4)[:1]


def _current_tool_pose_arrays(
    context,
    seed: np.ndarray,
    *,
    frame_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """逐 tool frame 计算当前 pose，并堆叠成 cuRobo Goal 要求的 batch shape。"""

    current_positions = []
    current_orientations = []
    for frame_name in frame_names:
        position, orientation = context.compute_tcp_poses(
            seed, tcp_frame_name=frame_name
        )
        current_positions.append(np.asarray(position, dtype=float).reshape(-1, 3)[0])
        current_orientations.append(
            np.asarray(orientation, dtype=float).reshape(-1, 4)[0]
        )
    return (
        np.asarray(current_positions, dtype=float).reshape(1, len(frame_names), 3),
        np.asarray(current_orientations, dtype=float).reshape(1, len(frame_names), 4),
    )


def _tool_pose_criteria_type(context):
    """从 context optional type namespace 读取 ``ToolPoseCriteria``。"""

    types = getattr(context, "types", None)
    return None if types is None else getattr(types, "ToolPoseCriteria", None)


def _context_frame_names(context) -> tuple[str, ...]:
    """读取 context 已 materialize 的 frame names；不支持时返回空 tuple。"""

    frame_names = getattr(context, "frame_names", None)
    if not callable(frame_names):
        return ()
    return tuple(str(name) for name in frame_names())


def _criteria_for_frame(criteria_type, *, position_only: bool):
    """为单个 tool frame 选择 position-only 或 position+orientation criteria。"""

    if position_only:
        return criteria_type.track_position()
    return criteria_type.track_position_and_orientation()


__all__ = [
    "goal_tool_pose_from_arrays",
    "goal_tool_pose_from_single_tcp_target",
    "update_active_tool_pose_criteria",
]
