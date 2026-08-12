"""不同物理引擎共同遵守的最小运行时合同。

合同故意不包含 ``world``：PhysX runtime 拥有 Isaac ``World``，项目自有 Newton
runtime 则直接拥有 Model/State/Control/Solver。把 World 放进公共协议会迫使后者制造一个
假的 Isaac owner，并重新引入双步进和交错销毁风险。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable


# 在 Isaac 层本地声明窄字面量，避免 runtime 协议反向依赖具体 manager/factory。
PhysicsBackend: TypeAlias = Literal["physx", "newton"]
PhysicsExecution: TypeAlias = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class PhysicsCapabilities:
    """具体 runtime 已验证的能力事实。"""

    supports_multiple_worlds: bool = False
    rendering: bool = True
    dynamic_chain: bool = True
    selected_reset: bool = True
    cuda_graph: bool = False


@runtime_checkable
class PhysicsRuntime(Protocol):
    """session 唯一拥有的物理时间和状态 owner。"""

    backend: PhysicsBackend
    kind: str
    execution: PhysicsExecution
    capabilities: PhysicsCapabilities
    scene: object

    def reset(self) -> None: ...

    def forward(self) -> None: ...

    def step(self, *, render: bool = False) -> None: ...

    def render(self) -> None: ...

    def pre_render(self) -> None: ...

    def close(self) -> None: ...

    def get_physics_dt(self) -> float: ...

    def get_rendering_dt(self) -> float: ...


class PhysicsRuntimeFactory(Protocol):
    """可注入的 runtime factory 形状，供 bootstrap 和纯测试替换。"""

    def __call__(self, **kwargs: object) -> PhysicsRuntime: ...


__all__ = [
    "PhysicsCapabilities",
    "PhysicsExecution",
    "PhysicsRuntime",
    "PhysicsRuntimeFactory",
]
