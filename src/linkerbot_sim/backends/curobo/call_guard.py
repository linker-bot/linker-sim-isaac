"""阻止第三方 cuRobo API 通过 SystemExit 终止仿真宿主。"""

from __future__ import annotations


def call_curobo(label: str, func, *args, **kwargs):
    """调用 cuRobo API，并把 ``SystemExit`` 转成普通 ``RuntimeError``。"""

    try:
        return func(*args, **kwargs)
    except SystemExit as exc:
        raise RuntimeError(
            f"cuRobo {label} requested process exit: code={exc.code!r}"
        ) from exc


__all__ = ["call_curobo"]
