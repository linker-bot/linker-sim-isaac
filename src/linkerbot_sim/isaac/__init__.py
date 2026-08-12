"""Isaac Sim 基础设施边界。

该 package 只承载 Kit/Stage/物理运行时等基础设施，不包含 Mirror 或 Kaleidoscope 的
产品行为。重型 Isaac/Omni import 必须继续放在显式启动边界之后。
"""

from __future__ import annotations


_LAZY_EXPORTS = {
    "IsaacSession": ("linkerbot_sim.isaac.session", "IsaacSession"),
    "create_isaac_session_from_spec": (
        "linkerbot_sim.isaac.session",
        "create_isaac_session_from_spec",
    ),
    "IsaacAppSpec": ("linkerbot_sim.isaac.spec", "IsaacAppSpec"),
    "IsaacComputeSpec": ("linkerbot_sim.isaac.spec", "IsaacComputeSpec"),
    "IsaacNewtonCpuSpec": (
        "linkerbot_sim.isaac.spec",
        "IsaacNewtonCpuSpec",
    ),
    "IsaacNewtonCudaSpec": (
        "linkerbot_sim.isaac.spec",
        "IsaacNewtonCudaSpec",
    ),
    "IsaacPhysxCpuSpec": ("linkerbot_sim.isaac.spec", "IsaacPhysxCpuSpec"),
    "IsaacPhysxCudaSpec": ("linkerbot_sim.isaac.spec", "IsaacPhysxCudaSpec"),
    "IsaacRenderSpec": ("linkerbot_sim.isaac.spec", "IsaacRenderSpec"),
    "IsaacSessionSpec": ("linkerbot_sim.isaac.spec", "IsaacSessionSpec"),
}


def __getattr__(name: str) -> object:
    """延迟导出基础设施类型，避免 package facade 启动 Isaac/Omni。"""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_LAZY_EXPORTS)
