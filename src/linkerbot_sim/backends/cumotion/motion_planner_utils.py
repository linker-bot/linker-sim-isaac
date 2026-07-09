"""cuMotion motion-planner pipeline 共享工具函数。

这些 helper 都是“边界适配”性质：把项目 mapping 写入 pybind config、兼容属性/方法两种结果
暴露形式、统一 path/状态字符串等。放在单独模块里可以避免 graph/optimizer/specified path
各自复制一份细节处理。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def apply_config_params(
    target,
    params: Mapping[str, Any],
    param_value_type,
    *,
    setter_name: str = "set_param",
) -> None:
    """把项目参数 mapping 写入 cuMotion config/generator。

    cuMotion 的 ``set_param`` / ``set_solver_param`` 通常返回 bool；返回 ``False`` 说明参数名或
    类型未被后端接受。这里立即抛出 ``ValueError``，让错误停在配置应用阶段。
    """

    setter = getattr(target, setter_name)
    for name, value in params.items():
        ok = setter(str(name), param_value_type(value))
        if ok is False:
            raise ValueError(f"cuMotion config rejected parameter {name!r}")


def attr(obj, name: str, *, default=None):
    """读取可能是属性也可能是零参方法的 pybind/fake 字段。

    真实 cuMotion pybind 对象和单元测试 fake 有时暴露形态不同；主逻辑只关心值本身。
    """

    value = getattr(obj, name, default)
    return value() if callable(value) else value


def result_path_samples(results, *, prefer_interpolated: bool) -> list[np.ndarray]:
    """从 graph ``MotionPlanner.Results`` 读取本次应消费的 path。"""

    names = ("interpolated_path", "path") if prefer_interpolated else ("path",)
    for name in names:
        samples = attr(results, name, default=())
        if samples is None:
            continue
        if len(samples) > 0:
            return [np.asarray(sample, dtype=float).reshape(-1) for sample in samples]
    return []


def stack_path(path_samples: Sequence[np.ndarray]) -> np.ndarray | None:
    """把 C-space sample 列表堆叠成 ``(N, dof)`` 矩阵。"""

    if not path_samples:
        return None
    return np.vstack(
        [np.asarray(sample, dtype=float).reshape(1, -1) for sample in path_samples]
    )


def path_length(joint_path: np.ndarray | None) -> float:
    """计算诊断用 C-space 几何路径长度。"""

    if joint_path is None or joint_path.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(joint_path, axis=0), axis=1).sum())


def validate_cspace_width(context, values: np.ndarray, label: str) -> None:
    """确保请求向量宽度等于当前 cuMotion C-space 自由度数量。"""

    if values.size != context.expected_cspace_width:
        raise ValueError(
            f"{label} expected {context.expected_cspace_width} values, got {values.size}"
        )


def collision_world_for_pipeline(context, *, use_environment_obstacles: bool):
    """按 pipeline 碰撞语义返回当前环境 world 或 context 缓存的空 world。"""

    if use_environment_obstacles:
        return context.collision_world()
    return context.empty_collision_world()


def status_name(value) -> str:
    """把 pybind enum/status 归一化成稳定可打印字符串。"""

    if value is None:
        return ""
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text
