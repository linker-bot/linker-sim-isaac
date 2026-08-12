"""标量与批量调用方共享的 motion planner backend contract 和名称校验。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

from linkerbot_sim.planning.requests import LinearPosePathRequest, MotionRequest
from linkerbot_sim.planning.results import MotionResult


PlannerBackendName = Literal["curobo", "linear"]
PlanningRequest: TypeAlias = MotionRequest | LinearPosePathRequest


@runtime_checkable
class PlannerBackend(Protocol):
    """不同 runtime 共同依赖的 canonical scalar planner 接口。"""

    def joint_names(self) -> Sequence[str]:
        """返回 backend 输入和输出采用的 joint order。"""

    def plan(self, request: PlanningRequest) -> MotionResult:
        """规划一条 canonical request，并返回 backend-neutral result。"""


def normalize_planner_backend(value: object) -> PlannerBackendName:
    """规范化并校验 public planner backend 名称。"""

    backend = str(value or "curobo").strip().lower()
    if backend not in {"curobo", "linear"}:
        raise ValueError("planner backend must be one of curobo or linear")
    return cast(PlannerBackendName, backend)


__all__ = [
    "PlannerBackend",
    "PlannerBackendName",
    "PlanningRequest",
    "normalize_planner_backend",
]
