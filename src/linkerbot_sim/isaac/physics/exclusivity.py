"""项目 Newton 与 Isaac 托管物理 owner 的排他性证明。"""

from __future__ import annotations

from collections.abc import Iterable

from linkerbot_sim.isaac.extensions import enumerate_enabled_kit_extensions


_NEWTON_FORBIDDEN_EXACT = frozenset(
    {
        "isaacsim.core.api",
        "isaacsim.core.cloner",
        "isaacsim.core.experimental.prims",
        "isaacsim.core.simulation_manager",
        "isaacsim.core.utils",
        "isaacsim.pip.newton",
    }
)
_NEWTON_FORBIDDEN_PREFIXES = (
    "isaacsim.physics.",
    "isaacsim.sensors.physx",
    "omni.physics.",
    "omni.physx",
)


def newton_forbidden_extensions(names: Iterable[str]) -> tuple[str, ...]:
    """返回闭包中会创建、携带或注入第二物理 owner 的扩展。"""

    return tuple(
        sorted(
            {
                str(name)
                for name in names
                if str(name) in _NEWTON_FORBIDDEN_EXACT
                or any(
                    str(name).startswith(prefix)
                    for prefix in _NEWTON_FORBIDDEN_PREFIXES
                )
            }
        )
    )


def validate_newton_exclusivity(
    *,
    extension_manager: object | None = None,
    stage: object | None = None,
    phase: str = "startup",
) -> None:
    """证明项目 Newton 进程没有 Isaac physics owner。

    该函数在 App 启动后执行一次，并在业务资产全部导入、Newton model finalize 前再次
    执行。第二次检查能捕获 importer 或其它业务扩展在启动验证之后偷偷带入的 owner。
    """

    extensions = enumerate_enabled_kit_extensions(extension_manager)
    forbidden = newton_forbidden_extensions(item.name for item in extensions)
    if forbidden:
        raise RuntimeError(
            "Newton runtime requires Isaac physics-owner extensions to be "
            f"disabled during {phase}: {list(forbidden)!r}"
        )

    active_stage = stage if stage is not None else _current_stage()
    if active_stage is None:
        return
    try:
        from pxr import UsdPhysics
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "UsdPhysics schema is unavailable while auditing Newton stage"
        ) from exc
    scenes = sorted(
        str(prim.GetPath())
        for prim in active_stage.Traverse()
        if prim.IsA(UsdPhysics.Scene)
    )
    if scenes:
        raise RuntimeError(
            "Newton runtime requires a stage without an Isaac physics scene "
            f"during {phase}: {scenes!r}"
        )


def _current_stage() -> object | None:
    """读取当前 stage；没有活动 context 时由调用方按“空 stage”处理。"""

    try:
        import omni.usd
    except (ImportError, ModuleNotFoundError):
        return None
    try:
        return omni.usd.get_context().get_stage()
    except (AttributeError, RuntimeError):
        return None


__all__ = [
    "newton_forbidden_extensions",
    "validate_newton_exclusivity",
]
