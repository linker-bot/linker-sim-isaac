"""Mirror：现实场景映像的稳定 lazy public facade。

导入本包不会加载 Isaac、cuRobo、transport 或启动线程；具体对象只在首次访问对应
public symbol 时按需导入。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "MirrorConfig": (
        "linkerbot_sim.configuration.modes.mirror",
        "MirrorConfig",
    ),
    "MirrorController": ("linkerbot_sim.mirror.controller", "MirrorController"),
    "MirrorRuntime": ("linkerbot_sim.mirror.runtime", "MirrorRuntime"),
    "create_mirror_runtime": (
        "linkerbot_sim.mirror.bootstrap",
        "create_mirror_runtime",
    ),
    "run_mirror": ("linkerbot_sim.mirror.app", "run_mirror"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
