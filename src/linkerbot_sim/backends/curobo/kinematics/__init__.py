"""不含 motion planner/collision world 的 cuRobo kinematics lazy facade。

包初始化不能急切导入 ``device_batch_ik``：类型模块和纯配置工具会经过本 facade，
而设备 solver 才需要 Torch。具体 capability 在首次访问对应 public symbol 时导入。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BatchIKTensorResult": (
        "linkerbot_sim.backends.curobo.kinematics.types",
        "BatchIKTensorResult",
    ),
    "BatchIKWaypointTensorResult": (
        "linkerbot_sim.backends.curobo.kinematics.types",
        "BatchIKWaypointTensorResult",
    ),
    "CuroboDeviceBatchIKSolver": (
        "linkerbot_sim.backends.curobo.kinematics.device_batch_ik",
        "CuroboDeviceBatchIKSolver",
    ),
    "CuroboKinematicsContext": (
        "linkerbot_sim.backends.curobo.kinematics.context",
        "CuroboKinematicsContext",
    ),
    "create_kinematics_context": (
        "linkerbot_sim.backends.curobo.kinematics.context",
        "create_kinematics_context",
    ),
    "kinematics_config_from_robot_profile": (
        "linkerbot_sim.backends.curobo.kinematics.context",
        "kinematics_config_from_robot_profile",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
