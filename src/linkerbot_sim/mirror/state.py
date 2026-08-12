"""Mirror 低频状态读写用例。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class MirrorStateService:
    """把物理状态 adapter 包装成不泄漏内部可写引用的产品服务。

    setter 的事务/回滚由 engine-aware adapter 实现；本层负责只接受 mapping，并在边界
    深复制输入输出，避免 transport 调用方在校验后并发修改 payload。
    """

    getter: Callable[[], Mapping[str, object]]
    setter: Callable[..., object]

    def get_state(self) -> dict[str, object]:
        value = self.getter()
        if not isinstance(value, Mapping):
            raise RuntimeError("Mirror state adapter 必须返回 mapping")
        return deepcopy(dict(value))

    def set_state(
        self,
        state: Mapping[str, object],
        *,
        strict: bool = True,
    ) -> object:
        if not isinstance(state, Mapping):
            raise ValueError("state 必须是 mapping")
        return self.setter(deepcopy(dict(state)), strict=bool(strict))


__all__ = ["MirrorStateService"]
