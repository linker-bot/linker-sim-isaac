"""cuRobo context 的碰撞查询能力 gate 与稳定诊断。"""

from __future__ import annotations


def context_supports_collision_queries(context, *, consumer: str | None = None) -> bool:
    """查询 context 的碰撞能力，同时支持单元测试 fake。"""

    ensure = getattr(context, "ensure_collision_checker", None)
    if consumer is not None and callable(ensure):
        capability = ensure(consumer)
        available = getattr(capability, "available", None)
        if available is not None:
            return bool(available)
    checker = getattr(context, "collision_queries_enabled", None)
    if callable(checker):
        return bool(checker())
    return True


def collision_capability_message(
    context,
    *,
    consumer: str | None = None,
) -> str:
    """返回指定 consumer 的稳定 missing requirements 诊断。"""

    getter = getattr(context, "collision_capability", None)
    if not callable(getter):
        return "collision capability is unavailable"
    capability = getter() if consumer is None else getter(consumer=consumer)
    missing = tuple(getattr(capability, "missing_requirements", ()) or ())
    if not missing:
        return "collision capability is unavailable"
    return "missing " + ", ".join(str(item) for item in missing)


__all__ = ["collision_capability_message", "context_supports_collision_queries"]
