"""Single Scene / Tiled Scene 交互入口共享的进程存活与空闲物理策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast


StdinEofPolicy = Literal["exit", "keep_alive"]
IdlePhysicsPolicy = Literal["pause", "hold_step"]


@dataclass(frozen=True)
class InteractiveRuntimePolicy:
    """把 stdin 生命周期与空闲 physics 推进拆成两个独立策略。"""

    stdin_eof_policy: StdinEofPolicy
    idle_physics_policy: IdlePhysicsPolicy

    def __post_init__(self) -> None:
        if self.stdin_eof_policy not in {"exit", "keep_alive"}:
            raise ValueError("stdin_eof_policy must be one of: exit, keep_alive")
        if self.idle_physics_policy not in {"pause", "hold_step"}:
            raise ValueError("idle_physics_policy must be one of: pause, hold_step")

    @property
    def keeps_alive_on_stdin_eof(self) -> bool:
        """返回 stdin EOF 是否应保持进程存活。"""

        return self.stdin_eof_policy == "keep_alive"

    @property
    def steps_while_idle(self) -> bool:
        """返回空闲期是否应保持 target 并推进 physics。"""

        return self.idle_physics_policy == "hold_step"


def resolve_interactive_runtime_policy(
    *,
    stdin_eof_policy: str | None = None,
    idle_physics_policy: str | None = None,
    default_stdin_eof_policy: StdinEofPolicy = "exit",
    default_idle_physics_policy: IdlePhysicsPolicy = "pause",
) -> InteractiveRuntimePolicy:
    """解析 stdin 生命周期与空闲 physics 的独立显式策略。"""

    return InteractiveRuntimePolicy(
        stdin_eof_policy=(
            default_stdin_eof_policy
            if stdin_eof_policy is None
            else cast(StdinEofPolicy, stdin_eof_policy)
        ),
        idle_physics_policy=(
            default_idle_physics_policy
            if idle_physics_policy is None
            else cast(IdlePhysicsPolicy, idle_physics_policy)
        ),
    )


__all__ = [
    "IdlePhysicsPolicy",
    "InteractiveRuntimePolicy",
    "StdinEofPolicy",
    "resolve_interactive_runtime_policy",
]
