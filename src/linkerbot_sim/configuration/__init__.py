"""Mirror/Kaleidoscope 的 pure typed configuration facade。

导入此 facade 不会启动 Kit、导入 Torch/CUDA 或读取配置文件；只有显式调用
``load_*_config`` 才执行 YAML I/O。facade 保持惰性，避免仅导入 schema 时连带创建
catalog reader 或触发任何文件访问。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ComputeSettings": (
        "linkerbot_sim.configuration.modes.common",
        "ComputeSettings",
    ),
    "KaleidoscopeConfig": (
        "linkerbot_sim.configuration.modes.kaleidoscope",
        "KaleidoscopeConfig",
    ),
    "KaleidoscopeEnvironmentSettings": (
        "linkerbot_sim.configuration.modes.kaleidoscope",
        "KaleidoscopeEnvironmentSettings",
    ),
    "KaleidoscopeViewportSettings": (
        "linkerbot_sim.configuration.visualization.kaleidoscope",
        "KaleidoscopeViewportSettings",
    ),
    "MirrorConfig": (
        "linkerbot_sim.configuration.modes.mirror",
        "MirrorConfig",
    ),
    "SkrlTrainingSettings": (
        "linkerbot_sim.configuration.training.skrl",
        "SkrlTrainingSettings",
    ),
    "NewtonCpuSettings": (
        "linkerbot_sim.configuration.physics",
        "NewtonCpuSettings",
    ),
    "NewtonCudaSettings": (
        "linkerbot_sim.configuration.physics",
        "NewtonCudaSettings",
    ),
    "PhysicsSettings": (
        "linkerbot_sim.configuration.physics",
        "PhysicsSettings",
    ),
    "PhysicsEngine": (
        "linkerbot_sim.configuration.physics",
        "PhysicsEngine",
    ),
    "PhysicsExecution": (
        "linkerbot_sim.configuration.physics",
        "PhysicsExecution",
    ),
    "PhysxCpuSettings": (
        "linkerbot_sim.configuration.physics",
        "PhysxCpuSettings",
    ),
    "PhysxCudaSettings": (
        "linkerbot_sim.configuration.physics",
        "PhysxCudaSettings",
    ),
    "semantic_config_fingerprint": (
        "linkerbot_sim.configuration.fingerprint",
        "semantic_config_fingerprint",
    ),
    "semantic_config_payload": (
        "linkerbot_sim.configuration.fingerprint",
        "semantic_config_payload",
    ),
    "load_kaleidoscope_config": (
        "linkerbot_sim.configuration.catalog",
        "load_kaleidoscope_config",
    ),
    "load_kaleidoscope_viewport_config": (
        "linkerbot_sim.configuration.catalog",
        "load_kaleidoscope_viewport_config",
    ),
    "load_mirror_config": (
        "linkerbot_sim.configuration.catalog",
        "load_mirror_config",
    ),
    "load_skrl_training_settings": (
        "linkerbot_sim.configuration.catalog",
        "load_skrl_training_settings",
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
