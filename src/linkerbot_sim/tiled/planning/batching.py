"""tiled request 的连续分组键、problem 计数与 batch 行布局。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from linkerbot_sim.tiled.planning.types import TiledPlanningRequest


def normalize_joint_batch_mode(value: object) -> str:
    """校验 cuRobo tiled joint batch 调度模式。"""

    mode = str(value or "auto").strip().lower()
    if mode not in {"auto", "per_env", "batch_only"}:
        raise ValueError("joint_batch_mode must be one of auto, per_env, batch_only")
    return mode


@dataclass(frozen=True)
class PlanningBatchLayout:
    """保持 FIFO request 顺序及其在合并数组中的连续行区间。

    该对象只描述 tiled DTO 与数组行号的关系，不生成替代 ``env_ids``，也不关心后端如何
    padding 或解释求解结果。组合后端按 ``row_slices`` 切分结果时即可恢复每条请求原有的
    request identity、真实 env IDs 和回放元数据。
    """

    requests: tuple[TiledPlanningRequest, ...]
    row_slices: tuple[slice, ...]
    problem_count: int


def planning_batch_layout(
    requests: Sequence[TiledPlanningRequest],
) -> PlanningBatchLayout | None:
    """为同构 joint-space requests 创建连续 row layout。

    返回 ``None`` 表示请求为空、包含 task-space segment，或规划语义不一致，调用方应保持
    request 粒度逐条执行。输入顺序直接决定 row 顺序，因此 manager 的 FIFO 语义不会被后端
    重排。
    """

    batch = tuple(requests)
    if not batch:
        return None
    key = planning_batch_key(batch[0])
    if key is None or any(planning_batch_key(request) != key for request in batch[1:]):
        return None
    row_slices: list[slice] = []
    start = 0
    for request in batch:
        stop = start + request_problem_count(request)
        row_slices.append(slice(start, stop))
        start = stop
    return PlanningBatchLayout(
        requests=batch,
        row_slices=tuple(row_slices),
        problem_count=start,
    )


def request_problem_count(request: TiledPlanningRequest) -> int:
    """返回 request 中按 env 行排列的独立问题数。"""

    return int(len(request.env_ids))


def planning_batch_key(request: TiledPlanningRequest) -> tuple[object, ...] | None:
    """返回 manager 侧保守 batch key；task-space path 暂不跨 request 合并。"""

    segment_key = _joint_space_segment_key(request)
    if segment_key is None:
        return None
    return (
        request.robot_name,
        request.joint_names,
        float(request.duration_s),
        float(request.sample_dt_s),
        bool(request.avoid_collisions),
        segment_key,
    )


def _joint_space_segment_key(
    request: TiledPlanningRequest,
) -> tuple[tuple[object, ...], ...] | None:
    """为可合批 joint segments 生成结构 key；task-space 或缺失 goal 时返回 None。"""

    if not request.segments:
        if request.goal_positions is None:
            return None
        return (
            (
                "joint_position_target",
                float(request.duration_s),
                float(request.sample_dt_s),
            ),
        )
    parts: list[tuple[object, ...]] = []
    for segment in request.segments:
        if segment.goal_positions is None:
            return None
        parts.append(
            (
                segment.kind,
                float(
                    request.duration_s
                    if segment.duration_s is None
                    else segment.duration_s
                ),
                float(
                    request.sample_dt_s
                    if segment.sample_dt_s is None
                    else segment.sample_dt_s
                ),
                segment.tcp_frame_name,
            )
        )
    return tuple(parts)


__all__ = [
    "PlanningBatchLayout",
    "normalize_joint_batch_mode",
    "planning_batch_key",
    "planning_batch_layout",
    "request_problem_count",
]
