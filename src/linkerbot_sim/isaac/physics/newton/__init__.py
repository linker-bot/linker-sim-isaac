"""项目自有 Newton 运行时。

该子包与 Isaac Newton extension 严格互斥；multi-world engine 基础设施完整保留，但当前
产品组合是否允许多 world 由上层 bootstrap 单独决定。
"""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """延迟暴露重型实现，pure import 不应加载 Newton/Warp。"""

    if name == "NewtonRuntime":
        from linkerbot_sim.isaac.physics.newton.manager import (
            NewtonRuntime,
        )

        return NewtonRuntime
    raise AttributeError(name)


__all__ = ["NewtonRuntime"]
