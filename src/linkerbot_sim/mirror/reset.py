"""Mirror 单场景事务 reset 用例。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass
class MirrorResetService:
    resetter: Callable[..., object]

    def reset(self, *, hold_after_reset: bool = True) -> object:
        result = self.resetter(hold_after_reset=bool(hold_after_reset))
        serializer = getattr(result, "as_dict", None)
        if callable(serializer):
            return serializer()
        if isinstance(result, Mapping):
            return dict(result)
        return result


__all__ = ["MirrorResetService"]
