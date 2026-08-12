"""Kaleidoscope：面向大规模并行强化学习的 GPU 原生仿真模式。

Facade 使用 PEP 562 lazy import；普通配置工具导入根包时不会加载 Torch、Gymnasium、Isaac 或
cuRobo，更不会初始化 CUDA/Kit。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ControlModeChange": (
        "linkerbot_sim.controllers.control_mode",
        "ControlModeChange",
    ),
    "ControlModeGenerationConflict": (
        "linkerbot_sim.controllers.control_mode",
        "ControlModeGenerationConflict",
    ),
    "ControlModeIncompatibleError": (
        "linkerbot_sim.controllers.control_mode",
        "ControlModeIncompatibleError",
    ),
    "ControlModeLockedError": (
        "linkerbot_sim.controllers.control_mode",
        "ControlModeLockedError",
    ),
    "ControlModeRollbackError": (
        "linkerbot_sim.controllers.control_mode",
        "ControlModeRollbackError",
    ),
    "ControlModeState": (
        "linkerbot_sim.controllers.control_mode",
        "ControlModeState",
    ),
    "ControlModeSwitchError": (
        "linkerbot_sim.controllers.control_mode",
        "ControlModeSwitchError",
    ),
    "GymnasiumKaleidoscopeAdapter": (
        "linkerbot_sim.kaleidoscope.adapters.gymnasium",
        "GymnasiumKaleidoscopeAdapter",
    ),
    "KaleidoscopeConfig": (
        "linkerbot_sim.configuration.modes.kaleidoscope",
        "KaleidoscopeConfig",
    ),
    "KaleidoscopeTrainingPort": (
        "linkerbot_sim.kaleidoscope.training_port",
        "KaleidoscopeTrainingPort",
    ),
    "KaleidoscopeEpisodeSnapshot": (
        "linkerbot_sim.kaleidoscope.snapshot",
        "KaleidoscopeEpisodeSnapshot",
    ),
    "TorchKaleidoscopeEnv": (
        "linkerbot_sim.kaleidoscope.env",
        "TorchKaleidoscopeEnv",
    ),
    "make_gymnasium_env": (
        "linkerbot_sim.kaleidoscope.bootstrap",
        "make_gymnasium_env",
    ),
    "make_torch_env": (
        "linkerbot_sim.kaleidoscope.bootstrap",
        "make_torch_env",
    ),
    "make_viewport_env": (
        "linkerbot_sim.kaleidoscope.bootstrap",
        "make_viewport_env",
    ),
    "register_gymnasium_envs": (
        "linkerbot_sim.kaleidoscope.registration",
        "register_gymnasium_envs",
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
