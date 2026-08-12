"""cuRobo 的能力分型 lazy public facade。

Mirror 使用完整 ``CuroboContext`` 做规划；Kaleidoscope 只能使用不加载 planner/collision
资源的 kinematics context 与 device batch IK adapter。根包导入本身不加载 Torch/CUDA，
具体 capability 只在用户首次访问对应符号时导入。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CuroboConfig": ("linkerbot_sim.backends.curobo.config", "CuroboConfig"),
    "CuroboContext": ("linkerbot_sim.backends.curobo.context", "CuroboContext"),
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
    "curobo_config_from_profiles": (
        "linkerbot_sim.backends.curobo.profile_merge",
        "curobo_config_from_profiles",
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
