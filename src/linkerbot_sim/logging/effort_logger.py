"""关节 effort 采样辅助函数。

Isaac 的 position/velocity drive 最终也会在 PhysX 中产生关节力/力矩，因此 effort 日志不只
服务 direct effort 控制。这里区分三种值：

* ``commanded_effort``：项目控制器在 Python 侧显式下发的 effort。implicit drive 没有这个值。
* ``measured_effort``：PhysX 求解器沿 DOF 方向计算/测得的关节 effort。
* ``applied_effort``：Isaac runtime 当前记录的关节 actuation effort。

不同 Isaac wrapper 暴露的读取方法略有差异。本模块优先使用 ``SingleArticulation`` 的
``get_measured_joint_efforts`` / ``get_applied_joint_efforts``，再回退到底层
``_articulation_view``；读取失败时返回 ``nan``，避免日志功能打断仿真主循环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from linkerbot_sim.utils.tensors import tensor_like_to_numpy


@dataclass(frozen=True)
class JointEffortSample:
    """一次关节 effort 采样。

    输入字段:
        measured: PhysX 测得/计算的关节 effort，单位由关节类型决定。
        applied: Isaac runtime 当前 actuation effort。
    输出:
        两个数组顺序均与调用 ``read_joint_efforts`` 时传入的 ``joint_indices`` 一致。
    """

    measured: np.ndarray
    applied: np.ndarray


def _expected_size(robot, joint_indices: np.ndarray | None) -> int:
    """计算本次 effort 采样应返回的数组长度。

    优先使用显式 ``joint_indices`` 长度；读取全部 DOF 时再回退到 Isaac articulation 的
    ``num_dof`` 或 ``dof_names``。
    """

    if joint_indices is not None:
        return int(joint_indices.size)
    if hasattr(robot, "num_dof"):
        return int(robot.num_dof)
    return len(getattr(robot, "dof_names", ()))


def _nan_vector(size: int) -> np.ndarray:
    """返回固定长度的 ``nan`` 数组，用于表示未采样或读取失败。"""

    return np.full(int(size), np.nan, dtype=float)


def _to_numpy_vector(values: Any, expected_size: int) -> np.ndarray | None:
    """把 Isaac/torch/warp/numpy 返回值压平成一维 numpy 数组。"""

    if values is None:
        return None
    try:
        array = tensor_like_to_numpy(values, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.size != expected_size:
        return None
    return array


def _read_effort_method(
    source, method_name: str, joint_indices: np.ndarray | None, expected_size: int
) -> np.ndarray | None:
    """调用一个 Isaac effort 读取方法，并把返回值规范化。"""

    method = getattr(source, method_name, None)
    if method is None:
        return None
    try:
        if joint_indices is None:
            values = method()
        else:
            try:
                values = method(joint_indices=joint_indices)
            except TypeError:
                values = method(joint_indices)
    except Exception:
        return None
    return _to_numpy_vector(values, expected_size)


def _read_effort(
    robot, method_name: str, joint_indices: np.ndarray | None, expected_size: int
) -> np.ndarray:
    """从 robot 或其 articulation view 读取一种 effort。

    Isaac 版本之间 API 暴露位置不同；这里先试高层 articulation，再试底层 view。两处都失败
    时返回 ``nan``，让日志缺测保持显式但不打断仿真。
    """

    values = _read_effort_method(robot, method_name, joint_indices, expected_size)
    if values is not None:
        return values
    view = getattr(robot, "_articulation_view", None)
    if view is not None:
        values = _read_effort_method(view, method_name, joint_indices, expected_size)
        if values is not None:
            return values
    return _nan_vector(expected_size)


def read_joint_efforts(
    robot,
    joint_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
    *,
    measured: bool = True,
    applied: bool = True,
) -> JointEffortSample:
    """读取一组关节的 measured/applied effort。

    参数:
        robot: Isaac articulation 对象。
        joint_indices: 可选 DOF index 列表；为空时读取全部 DOF。
        measured: 是否读取 PhysX measured effort。
        applied: 是否读取 Isaac applied effort。
    返回:
        ``JointEffortSample``。未启用或读取失败的数组填 ``nan``。
    """

    indices = (
        None
        if joint_indices is None
        else np.asarray(joint_indices, dtype=int).reshape(-1)
    )
    expected_size = _expected_size(robot, indices)
    return JointEffortSample(
        measured=(
            _read_effort(robot, "get_measured_joint_efforts", indices, expected_size)
            if measured
            else _nan_vector(expected_size)
        ),
        applied=(
            _read_effort(robot, "get_applied_joint_efforts", indices, expected_size)
            if applied
            else _nan_vector(expected_size)
        ),
    )


def commanded_efforts_from_controller(
    controller, joint_indices: np.ndarray | list[int] | tuple[int, ...]
) -> np.ndarray:
    """从 ``JointController.last_commanded_efforts`` 中取出日志关节切片。

    参数:
        controller: 项目 ``JointController`` 或兼容对象。
        joint_indices: 需要记录的完整 DOF index。
    返回:
        与 ``joint_indices`` 等长的 commanded effort；缺失时填 ``nan``。
    """

    indices = np.asarray(joint_indices, dtype=int).reshape(-1)
    values = getattr(controller, "last_commanded_efforts", None)
    if values is None:
        return _nan_vector(indices.size)
    try:
        efforts = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return _nan_vector(indices.size)
    if indices.size and (np.min(indices) < 0 or np.max(indices) >= efforts.size):
        return _nan_vector(indices.size)
    return efforts[indices]


__all__ = [
    "JointEffortSample",
    "commanded_efforts_from_controller",
    "read_joint_efforts",
]
