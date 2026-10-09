"""cuRobo 0.8 的 Warp torch namespace 适配。"""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def ensure_warp_torch_namespace_compatible() -> None:
    """为新版 Warp 补回 cuRobo 0.8 使用的 ``wp.torch`` 入口。"""

    try:
        warp_module = importlib.import_module("warp")
    except ModuleNotFoundError:
        return
    if getattr(warp_module, "torch", None) is not None:
        return
    device_from_torch = getattr(warp_module, "device_from_torch", None)
    if callable(device_from_torch):
        warp_module.torch = SimpleNamespace(device_from_torch=device_from_torch)


__all__ = ["ensure_warp_torch_namespace_compatible"]
