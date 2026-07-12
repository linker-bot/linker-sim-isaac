"""Isaac bundled Warp 与 cuRobo 使用方式之间的第三方版本适配。"""

from __future__ import annotations

import functools
import importlib
import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


def ensure_warp_func_module_keyword_compatible() -> None:
    """为旧 Warp 的 ``wp.func`` 补齐新版 ``module`` keyword。"""

    try:
        warp_module = importlib.import_module("warp")
    except ModuleNotFoundError:
        return
    func = getattr(warp_module, "func", None)
    if not callable(func) or getattr(func, "_linkerbot_accepts_module_keyword", False):
        return
    if _callable_accepts_keyword(func, "module"):
        return
    context = getattr(warp_module, "context", None)
    codegen = getattr(warp_module, "codegen", None)
    if context is None:
        context = importlib.import_module("warp.context")
    if codegen is None:
        codegen = importlib.import_module("warp.codegen")
    warp_module.func = _warp_func_with_module_keyword(
        original_func=func,
        function_type=getattr(context, "Function"),
        module_type=getattr(context, "Module"),
        get_module=getattr(context, "get_module"),
        make_full_qualified_name=getattr(codegen, "make_full_qualified_name"),
    )


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


def _callable_accepts_keyword(func: Callable[..., object], keyword: str) -> bool:
    """用 inspect 判断 callable 是否声明指定 keyword 或 ``**kwargs``。"""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in signature.parameters.values()
    )


def _warp_func_with_module_keyword(
    *,
    original_func: Callable[..., object],
    function_type: type,
    module_type: type,
    get_module: Callable[[str], object],
    make_full_qualified_name: Callable[[Callable[..., object]], str],
) -> Callable[..., object]:
    """包装旧版 ``wp.func``，补充新版 cuRobo 需要的 ``module`` keyword。"""

    @functools.wraps(original_func)
    def func_with_module(
        f: Callable[..., object] | None = None,
        *,
        name: str | None = None,
        module: object | None = None,
    ):
        """兼容 decorator 直接调用和带参数调用两种 ``wp.func`` 形式。"""

        frame = inspect.currentframe()
        scope_locals = (
            {} if frame is None or frame.f_back is None else frame.f_back.f_locals
        )

        def wrapper(target: Callable[..., object], *args: Any, **kwargs: Any):
            """按目标 Python module 注册 Warp Function，并保留原 callable metadata。"""

            del args, kwargs
            key = make_full_qualified_name(target) if name is None else name
            warp_target_module = _resolve_warp_module_for_func(
                target=target,
                module=module,
                module_type=module_type,
                get_module=get_module,
            )
            function_type(
                func=target,
                key=key,
                namespace="",
                module=warp_target_module,
                value_func=None,
                scope_locals=scope_locals,
                doc=(getattr(target, "__doc__", "") or "").strip(),
            )
            registered = warp_target_module.functions[key]
            return functools.update_wrapper(registered, target)

        return wrapper if f is None else wrapper(f)

    func_with_module._linkerbot_accepts_module_keyword = True  # type: ignore[attr-defined]
    return func_with_module


def _resolve_warp_module_for_func(
    *,
    target: Callable[..., object],
    module: object | None,
    module_type: type,
    get_module: Callable[[str], object],
) -> object:
    """把 None、``unique``、module name 或 Module object 归一化为 Warp Module。"""

    if module is None:
        return get_module(target.__module__)
    if module == "unique":
        return module_type(target.__name__, None)
    if isinstance(module, str):
        return get_module(module)
    return module


__all__ = [
    "ensure_warp_func_module_keyword_compatible",
    "ensure_warp_torch_namespace_compatible",
]
