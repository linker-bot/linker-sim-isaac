"""Mirror 资源关闭返回值的统一判定语义。"""

from __future__ import annotations

from collections.abc import Mapping


_MISSING = object()


def close_result_stopped(result: object) -> bool:
    """判断一次同步 ``close()`` 是否已经完成资源停止。

    没有返回状态的同步 closer 默认在返回时已完成；显式 bool、带 ``stopped``
    属性的报告以及遗留状态 mapping 则保留其可重试语义。
    """

    if result is None:
        return True
    if isinstance(result, bool):
        return result
    stopped = getattr(result, "stopped", _MISSING)
    if stopped is not _MISSING and stopped is not None:
        return bool(stopped)
    if isinstance(result, Mapping):
        if "stopped" in result:
            return bool(result["stopped"])
        return not bool(result.get("shutdown_timed_out", False))
    return True


__all__ = ["close_result_stopped"]
