"""tiled plan/trajectory/hand 消息共享的 canonical payload 与数组校验。

这里是 JSON 边界而不是便利转换层：字符串数字、truthy 值和标量不会被隐式转换成协议
声明的 number、boolean 或 array。通过校验后才构造 NumPy 数组，避免 ``np.asarray``
悄悄接受协议之外的 Python/JSON 值。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def reject_unknown_fields(
    payload: Mapping[str, object],
    allowed: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    """拒绝命令当前公开 schema 之外的全部字段，避免拼写错误被静默忽略。"""

    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown}")


def command_rows_for_selected(
    values: object,
    *,
    current_positions: np.ndarray,
    command_joint_names: tuple[str, ...],
    joint_names: tuple[str, ...] | None,
    base: str,
    label: str,
) -> np.ndarray:
    """把 selected-env 关节行按名称补齐到完整 command-space。

    未出现在 payload 中的关节由 ``base`` 决定保留当前位置或补零；返回值始终是独立数组，
    调用方后续写入不会反向修改 runtime 提供的 ``current_positions``。
    """

    current = np.asarray(current_positions, dtype=float)
    if current.ndim != 2:
        raise ValueError("current_positions must have shape (E,D)")
    rows = selected_variable_width_rows(
        json_numeric_array(values, label=label),
        selected_count=current.shape[0],
        label=label,
    )
    if rows.shape[1] > current.shape[1]:
        raise ValueError(
            f"{label} width {rows.shape[1]} exceeds command width {current.shape[1]}"
        )
    if base == "current":
        full = current.copy()
    elif base == "zero":
        full = np.zeros_like(current)
    else:
        raise ValueError("base must be 'current' or 'zero'")
    if joint_names is None:
        full[:, : rows.shape[1]] = rows
        return full
    if len(joint_names) != rows.shape[1]:
        raise ValueError(f"joint_names expected {rows.shape[1]} names")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names cannot contain duplicates")
    index_by_name = {name: index for index, name in enumerate(command_joint_names)}
    unknown = [name for name in joint_names if name not in index_by_name]
    if unknown:
        if rows.shape[1] == current.shape[1] and are_generated_command_names(
            command_joint_names
        ):
            return rows
        raise ValueError(f"unknown plan joint_names: {unknown}")
    for source_index, name in enumerate(joint_names):
        full[:, index_by_name[name]] = rows[:, source_index]
    return full


def selected_variable_width_rows(
    values: object,
    *,
    selected_count: int,
    label: str,
) -> np.ndarray:
    """把 ``(D,)``/``(E,D)`` 数组规范化为 selected env 行数。

    单行命令会显式广播到所有选中环境，其余行数必须精确匹配 ``env_ids``；复制结果使
    每个环境拥有可独立修改的一行，而不是共享广播视图。
    """

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{label} must have shape (N,D)")
    if array.shape[0] == 1 and int(selected_count) != 1:
        array = np.repeat(array, int(selected_count), axis=0)
    if array.shape[0] != int(selected_count):
        raise ValueError(f"{label} first dimension must be 1 or len(env_ids)")
    return array.astype(float, copy=True)


def optional_str_tuple(
    payload: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> tuple[str, ...] | None:
    """读取可选的非空 JSON 字符串数组，不执行元素或容器类型转换。"""

    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must contain non-empty strings")
    names = tuple(value)
    if not names:
        raise ValueError(f"{label} cannot be empty")
    return names


def are_generated_command_names(names: tuple[str, ...]) -> bool:
    """判断是否为 debug runtime 生成的 joint_0/joint_1/... 名称。"""

    return all(name == f"joint_{index}" for index, name in enumerate(names))


def optional_json_string(
    payload: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> str | None:
    """读取可选非空 JSON 字符串；字段缺失与显式 ``null`` 具有不同语义。"""

    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def optional_json_integer(
    payload: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> int | None:
    """读取可选 JSON 整数，不接受 bool、浮点数或数字字符串。"""

    if key not in payload:
        return None
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"{label} must be a JSON integer")
    return value


def json_number(value: object, *, label: str) -> float:
    """读取一个有限 JSON number，不接受 bool 或数字字符串。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def json_numeric_array(value: object, *, label: str) -> np.ndarray:
    """递归校验 JSON 数值数组，并拒绝 NumPy 本可转换的非 number 叶节点。

    错误路径保留每一层数组索引，便于客户端定位具体坏值；完成整棵树校验后才交给
    NumPy，因此转换不会成为额外的协议入口。
    """

    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")

    def validate(item: object, path: str) -> None:
        """深度优先检查叶节点，并保留其完整 JSON 索引路径。"""

        if isinstance(item, list):
            for index, child in enumerate(item):
                validate(child, f"{path}[{index}]")
            return
        json_number(item, label=path)

    validate(value, label)
    return np.asarray(value, dtype=float)


def strict_optional_bool(
    payload: Mapping[str, object],
    key: str,
    *,
    default: bool,
    label: str,
) -> bool:
    """读取可选 JSON boolean，不接受 truthy 字符串或数字。"""

    if key not in payload:
        return bool(default)
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be a boolean")
    return value


__all__ = [
    "are_generated_command_names",
    "command_rows_for_selected",
    "json_number",
    "json_numeric_array",
    "optional_json_integer",
    "optional_json_string",
    "optional_str_tuple",
    "reject_unknown_fields",
    "selected_variable_width_rows",
    "strict_optional_bool",
]
