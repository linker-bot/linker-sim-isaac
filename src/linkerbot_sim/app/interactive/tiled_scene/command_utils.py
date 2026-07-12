"""tiled interactive runtime 共享的 action shape、插值和 selector 辅助函数。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from linkerbot_sim.tiled.control.types import TiledCommandAction


def _action_decimation(
    action: TiledCommandAction,
    *,
    default_decimation: int,
    physics_dt: float | None = None,
) -> int:
    """返回 action 实际执行的 physics tick 数。"""

    if action.duration_s is not None:
        dt = 0.0 if physics_dt is None else float(physics_dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("duration_s requires a positive physics_dt")
        requested_ticks = float(action.duration_s) / dt
        return max(1, int(np.ceil(requested_ticks - 1.0e-12)))

    return _positive_decimation(
        action.decimation,
        default_decimation=default_decimation,
    )


def _linear_path_ik_steps(
    action: TiledCommandAction,
    *,
    execution_ticks: int,
    physics_dt: float,
) -> int:
    """按 logical duration/sample dt 返回顺序 batched IK 次数。"""

    if action.kind != "ee_linear_path":
        raise ValueError("IK sample steps are only defined for ee_linear_path")
    dt = float(physics_dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("ee_linear_path requires a positive physics_dt")
    duration = (
        float(action.duration_s)
        if action.duration_s is not None
        else int(execution_ticks) * dt
    )
    sample_dt = dt if action.sample_dt_s is None else float(action.sample_dt_s)
    return max(1, int(np.ceil(duration / sample_dt)))


def _positive_decimation(
    value: int | None,
    *,
    default_decimation: int,
) -> int:
    """解析 command/trajectory step 的正整数 decimation。"""

    ticks = int(default_decimation if value is None else value)
    if ticks < 1:
        raise ValueError("decimation must be >= 1")
    return ticks


def _action_width(action: TiledCommandAction, *, default_width: int) -> int:
    """根据 action values 推导要写入的 command joint 前缀宽度。"""

    if action.kind == "hold":
        width = int(default_width)
    else:
        if action.values is None:
            raise ValueError(f"{action.kind}.values is required")
        values = np.asarray(action.values, dtype=float)
        width = int(values.shape[1]) if values.ndim == 2 else int(values.size)
    if width < 1:
        raise ValueError("action width must be >= 1")
    if width > int(default_width):
        raise ValueError(
            f"action width {width} exceeds selected robot command width {default_width}"
        )
    return width


def _apply_joint_targets(
    view: object,
    targets: np.ndarray,
    *,
    joint_indices: np.ndarray,
    velocities: np.ndarray | None = None,
) -> None:
    """向 Isaac batched Articulation view 一次性写入整批关节目标。"""

    from isaacsim.core.utils.types import ArticulationActions

    position_targets = _finite_array(targets, "joint position targets")
    velocity_targets = (
        None
        if velocities is None
        else _finite_array(velocities, "joint velocity targets")
    )
    view.apply_action(
        ArticulationActions(
            joint_positions=position_targets,
            joint_velocities=velocity_targets,
            joint_indices=np.asarray(joint_indices, dtype=int),
        )
    )


def _apply_runtime_mimic_targets(articulation: object) -> None:
    """根据各环境中 master 关节的实际状态写入 MJCF follower 目标。

    follower 的位置由 mimic 多项式计算，速度使用该多项式的一阶导数计算；读取实际
    状态而非上一拍目标，避免物理跟踪误差在 follower 链路中被忽略。
    """

    controls = tuple(getattr(articulation, "runtime_mimic_controls", ()))
    if not controls:
        return
    view = articulation.view
    actual_positions = np.asarray(view.get_joint_positions(), dtype=float)
    actual_velocities = np.asarray(view.get_joint_velocities(), dtype=float)
    if actual_positions.ndim == 1:
        actual_positions = actual_positions.reshape(1, -1)
    if actual_velocities.ndim == 1:
        actual_velocities = actual_velocities.reshape(1, -1)
    if actual_positions.shape != actual_velocities.shape:
        raise RuntimeError(
            "Tiled articulation position/velocity shapes differ while applying mimic"
        )

    follower_positions = np.empty(
        (actual_positions.shape[0], len(controls)), dtype=float
    )
    follower_velocities = np.empty_like(follower_positions)
    follower_indices = np.empty(len(controls), dtype=int)
    for column, control in enumerate(controls):
        master_position = actual_positions[:, control.master_index]
        master_velocity = actual_velocities[:, control.master_index]
        follower_positions[:, column] = sum(
            coefficient * master_position**power
            for power, coefficient in enumerate(control.polycoef)
        )
        follower_velocities[:, column] = (
            sum(
                power * coefficient * master_position ** (power - 1)
                for power, coefficient in enumerate(control.polycoef[1:], start=1)
            )
            * master_velocity
        )
        follower_indices[column] = control.dependent_index

    _apply_joint_targets(
        view,
        follower_positions,
        velocities=follower_velocities,
        joint_indices=follower_indices,
    )


def _filter_isaac_state_fields(
    payload: Mapping[str, object],
    fields: tuple[str, ...],
) -> dict[str, object]:
    """按 get_state.fields 裁剪 Isaac state payload。"""

    result: dict[str, object] = {}
    robots = payload.get("robots", {})
    objects = payload.get("objects", {})
    for field in fields:
        parts = field.split(".")
        if len(parts) == 1:
            if field in payload:
                result[field] = payload[field]
            continue
        if parts[0] == "robots":
            nested = robots
        elif parts[0] == "objects":
            nested = objects
        else:
            continue
        if not isinstance(nested, Mapping):
            continue
        item_name = parts[1]
        item_state = nested.get(item_name)
        if not isinstance(item_state, Mapping):
            continue
        result_group = result.setdefault(parts[0], {})
        if not isinstance(result_group, dict):
            continue
        if len(parts) == 2:
            result_group[item_name] = item_state
            continue
        value_key = parts[2]
        if value_key not in item_state:
            continue
        item_result = result_group.setdefault(item_name, {})
        if isinstance(item_result, dict):
            item_result[value_key] = item_state[value_key]
    return result


def _normalize_env_ids(env_ids: np.ndarray | None, num_envs: int) -> np.ndarray:
    """把 env_ids 规范化为一维唯一 int 数组。"""

    if env_ids is None:
        return np.arange(int(num_envs), dtype=int)
    array = np.asarray(env_ids, dtype=int)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError("env_ids must be a 1D array")
    if array.size == 0:
        raise ValueError("env_ids cannot be empty")
    if np.unique(array).size != array.size:
        raise ValueError("env_ids cannot contain duplicates")
    if np.any(array < 0) or np.any(array >= int(num_envs)):
        raise ValueError("env_ids contains out-of-range env id")
    return array.astype(int, copy=True)


def _action_for_selected_envs(
    *,
    action: TiledCommandAction,
    env_ids: np.ndarray,
    current_positions: np.ndarray,
    current_tcp_positions: np.ndarray,
    current_tcp_orientations_wxyz: np.ndarray,
    env_origins: np.ndarray,
) -> TiledCommandAction:
    """把 selected-env action 扩展为 full batch action。"""

    num_envs = int(current_positions.shape[0])
    selected = np.asarray(env_ids, dtype=int)
    if selected.size == num_envs and np.array_equal(selected, np.arange(num_envs)):
        return action
    values = action.values
    if action.kind == "hold":
        return action
    if action.kind == "joint_position_target":
        selected_values = _selected_action_rows(
            values, selected.size, current_positions.shape[1], action.kind
        )
        full = current_positions.copy()
        full[selected, :] = selected_values
        return _replace_action_values(action, full)
    if action.kind == "joint_delta_pos":
        selected_values = _selected_action_rows(
            values, selected.size, current_positions.shape[1], action.kind
        )
        full = np.zeros_like(current_positions)
        full[selected, :] = selected_values
        return _replace_action_values(action, full)
    if action.kind == "ee_delta_pos":
        selected_values = _selected_action_rows(values, selected.size, 3, action.kind)
        full = np.zeros((num_envs, 3), dtype=float)
        full[selected, :] = selected_values
        return _replace_action_values(action, full)
    if action.kind == "ee_linear_path":
        if action.target_position is not None:
            selected_values = _selected_rows(
                action.target_position,
                selected.size,
                3,
                "ee_linear_path.target_position",
            )
            full_target = current_tcp_positions.copy()
            if action.pose_reference_frame == "env":
                full_target -= env_origins
            full_target[selected, :] = selected_values
            result = replace(action, target_position=full_target)
        else:
            source = (
                action.target_offset if action.target_offset is not None else values
            )
            label = (
                "ee_linear_path.target_offset"
                if action.target_offset is not None
                else "ee_linear_path.values"
            )
            selected_values = _selected_rows(source, selected.size, 3, label)
            full_target = np.zeros((num_envs, 3), dtype=float)
            full_target[selected, :] = selected_values
            result = (
                replace(action, target_offset=full_target)
                if action.target_offset is not None
                else replace(action, values=full_target)
            )
        if action.target_orientation_wxyz is not None:
            selected_orientations = _selected_rows(
                action.target_orientation_wxyz,
                selected.size,
                4,
                "ee_linear_path.target_orientation_quat_wxyz",
            )
            full_orientations = current_tcp_orientations_wxyz.copy()
            full_orientations[selected, :] = selected_orientations
            result = replace(result, target_orientation_wxyz=full_orientations)
        return result
    if action.kind == "ee_pose_target":
        selected_values = _selected_action_rows(values, selected.size, 7, action.kind)
        full = np.zeros((num_envs, 7), dtype=float)
        if action.pose_reference_frame == "env":
            full[:, :3] = current_tcp_positions - env_origins
        else:
            full[:, :3] = current_tcp_positions
        full[:, 3:7] = current_tcp_orientations_wxyz
        full[selected, :] = selected_values
        return _replace_action_values(action, full)
    if action.kind == "ee_delta_pose":
        value_array = np.asarray(values, dtype=float)
        width = (
            int(value_array.shape[1])
            if value_array.ndim == 2
            else int(value_array.size)
        )
        if width not in (6, 7):
            raise ValueError("ee_delta_pose.values must have width 6 or 7")
        selected_values = _selected_action_rows(
            values, selected.size, width, action.kind
        )
        full = np.zeros((num_envs, width), dtype=float)
        if width == 7:
            full[:, 3:7] = current_tcp_orientations_wxyz
        full[selected, :] = selected_values
        return _replace_action_values(action, full)
    return action


def _replace_action_values(
    action: TiledCommandAction, values: np.ndarray
) -> TiledCommandAction:
    """复制 action 元数据并替换 values。"""

    return replace(action, values=values)


def _selected_action_rows(
    values: np.ndarray | None,
    selected_count: int,
    width: int,
    label: str,
) -> np.ndarray:
    """把 action values 规范化为 selected env 行数。"""

    return _selected_rows(values, selected_count, width, f"{label}.values")


def _selected_rows(
    values: object,
    selected_count: int,
    width: int,
    label: str,
) -> np.ndarray:
    """把输入规范化为 ``(selected_count, width)``，允许单行广播。"""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != int(width):
        raise ValueError(f"{label} must have shape (N, {width})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    if array.shape[0] == 1 and int(selected_count) != 1:
        return np.repeat(array, int(selected_count), axis=0)
    if array.shape[0] != int(selected_count):
        raise ValueError(f"{label} first dimension must be 1 or len(env_ids)")
    return array.astype(float, copy=True)


def _selected_int_rows(values: object, selected_count: int, label: str) -> np.ndarray:
    """把整数状态字段规范化为 selected env 长度。"""

    array = np.asarray(values, dtype=int)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D integer array")
    if array.size == 1 and int(selected_count) != 1:
        return np.repeat(array, int(selected_count))
    if array.size != int(selected_count):
        raise ValueError(f"{label} length must be 1 or len(env_ids)")
    return array.astype(int, copy=True)


def _batched_values(
    values: np.ndarray | None,
    num_envs: int,
    width: int,
    label: str,
) -> np.ndarray:
    """把 action values 规范化为 ``(num_envs, width)``。"""

    if values is None:
        raise ValueError(f"{label} is required")
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} must have shape (N, {width})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    if array.shape[0] == 1 and num_envs != 1:
        return np.repeat(array, num_envs, axis=0)
    if array.shape[0] != num_envs:
        raise ValueError(f"{label} first dimension must be 1 or num_envs")
    return array.astype(float, copy=True)


def _repeat_or_validate_rows(
    values: np.ndarray,
    row_count: int,
    trailing_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    """把单行 root frame 数据广播到 batch，或校验已有 batch 行数。"""

    array = np.asarray(values, dtype=float)
    expected_single = (1, *trailing_shape)
    expected_batch = (int(row_count), *trailing_shape)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    if array.shape == expected_single and int(row_count) != 1:
        return np.repeat(array, int(row_count), axis=0)
    if array.shape != expected_batch:
        raise ValueError(
            f"{label} must have shape {expected_single} or {expected_batch}"
        )
    return array


def _normalize_quaternions(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """归一化 wxyz 四元数。"""

    quats = np.asarray(quaternions_wxyz, dtype=float)
    if not np.all(np.isfinite(quats)):
        raise ValueError("quaternion values must be finite")
    norms = np.linalg.norm(quats, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("quaternion values must be non-zero")
    return quats / norms[:, None]


def _finite_array(values: object, label: str) -> np.ndarray:
    """返回 float 数组，并在 Isaac action 边界拒绝非有限目标。"""

    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values")
    return array


def _quat_multiply_rows(left_wxyz: np.ndarray, right_wxyz: np.ndarray) -> np.ndarray:
    """逐行相乘两个 wxyz 四元数数组，返回归一化 wxyz。"""

    left = _normalize_quaternions(np.asarray(left_wxyz, dtype=float).reshape(-1, 4))
    right = _normalize_quaternions(np.asarray(right_wxyz, dtype=float).reshape(-1, 4))
    if left.shape[0] == 1 and right.shape[0] != 1:
        left = np.repeat(left, right.shape[0], axis=0)
    if right.shape[0] == 1 and left.shape[0] != 1:
        right = np.repeat(right, left.shape[0], axis=0)
    if left.shape[0] != right.shape[0]:
        raise ValueError("quaternion row counts must match")
    composed = Rotation.from_quat(left[:, [1, 2, 3, 0]]) * Rotation.from_quat(
        right[:, [1, 2, 3, 0]]
    )
    return composed.as_quat()[:, [3, 0, 1, 2]]


def _quat_inverse_rows(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """逐行求 wxyz 单位四元数逆。"""

    quats = _normalize_quaternions(
        np.asarray(quaternions_wxyz, dtype=float).reshape(-1, 4)
    )
    inverse = quats.copy()
    inverse[:, 1:] *= -1.0
    return inverse


def _jsonable_mapping(data: Mapping[str, object]) -> dict[str, object]:
    """把 info mapping 中的 numpy 值转成 JSON-compatible 值。"""

    return {str(key): _jsonable(value) for key, value in data.items()}


def _jsonable(value: object) -> object:
    """递归转换 numpy 值，并把诊断中的缺失浮点值表示为 null。"""

    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
