"""``SimulationApp`` 与精确 physics owner 的关闭收口工具。"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from threading import RLock


_binding_lock = RLock()
# App 未必支持 weakref 或任意属性，因此使用 identity-checked 强引用记录。成功关闭后立即
# 删除；构造/关闭失败时保留记录，避免下一次清理误认或丢失仍需诊断的 owner。


@dataclass
class _AppPhysicsBinding:
    """记录可重试 teardown 已完成的阶段。"""

    app: object
    runtime: object
    runtime_closed: bool = False
    backend_cleared: bool = False
    imports_released: bool = False
    app_closed: bool = False


_app_physics_bindings: dict[int, _AppPhysicsBinding] = {}


def register_simulation_app_physics_runtime(app: object, runtime: object) -> None:
    """把 App 与它唯一拥有的 physics runtime 绑定。

    同一 App 重复登记同一 runtime 是幂等的；任何不同组合都视为所有权错误。映射用于
    记录可重试 teardown 的阶段，即使 native App 已关闭而 runtime 首次关闭失败，也不会
    在第二次 ``IsaacSession.close`` 时重复释放 importer 或原生 App。
    """

    key = id(app)
    with _binding_lock:
        existing = _app_physics_bindings.get(key)
        if existing is not None and (
            existing.app is not app or existing.runtime is not runtime
        ):
            raise RuntimeError("SimulationApp already has a different physics runtime")
        if existing is None:
            _app_physics_bindings[key] = _AppPhysicsBinding(
                app=app,
                runtime=runtime,
            )


def bound_simulation_app_physics_runtime(app: object) -> object | None:
    """返回与该精确 App identity 绑定的 runtime。"""

    with _binding_lock:
        binding = _app_physics_bindings.get(id(app))
        if binding is None or binding.app is not app:
            return None
        return binding.runtime


def close_simulation_app(
    app: object,
    *,
    exit_code: int = 0,
    physics_runtime: object | None = None,
) -> None:
    """按 runtime/importer/App 顺序关闭，且绝不猜测其它 session 的 owner。

    早期实现无条件释放进程全局 active manager；若第二个 session 在构造中失败，会把第一
    个 session 的 Newton owner 误关。现在只关闭显式传入或与该 App identity 绑定的
    runtime。任一 Python 资源报错后仍会尝试 native App shutdown，并在最后重抛首个异常。
    """

    with _binding_lock:
        binding = _app_physics_bindings.get(id(app))
        if binding is not None and binding.app is not app:
            binding = None
    bound_runtime = None if binding is None else binding.runtime
    if (
        physics_runtime is not None
        and bound_runtime is not None
        and physics_runtime is not bound_runtime
    ):
        raise RuntimeError(
            "refusing to close SimulationApp with a different physics runtime"
        )
    runtime = physics_runtime if physics_runtime is not None else bound_runtime
    if runtime is not None and binding is None:
        # 构造失败可能发生在正式 session 返回之前；仍为显式 runtime 建立 teardown 记录，
        # 使 close 报错后的重试不会再次关闭已经退出的 native App。
        register_simulation_app_physics_runtime(app, runtime)
        with _binding_lock:
            binding = _app_physics_bindings[id(app)]
    first_error: BaseException | None = None

    def attempt(label: str, callback) -> bool:
        nonlocal first_error
        try:
            callback()
        except SystemExit as exc:
            if exc.code in (None, 0):
                return True
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(f"{label} also failed: SystemExit({exc.code})")
            return False
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(
                    f"{label} also failed: {type(exc).__name__}: {exc}"
                )
            return False
        return True

    if runtime is not None and not bool(binding and binding.runtime_closed):
        runtime_closed = attempt(
            "physics runtime close",
            lambda: _close_exact_physics_runtime(runtime),
        )
        if runtime_closed and binding is not None:
            binding.runtime_closed = True
    else:
        runtime_closed = True
    if (
        runtime_closed
        and getattr(runtime, "backend", None) == "newton"
        and not bool(binding and binding.backend_cleared)
    ):
        from linkerbot_sim.isaac.physics.backend import (
            clear_runtime_physics_backend,
        )

        backend_cleared = attempt(
            "runtime backend clear",
            lambda: clear_runtime_physics_backend(
                backend=getattr(runtime, "backend", None)
            ),
        )
        if backend_cleared and binding is not None:
            binding.backend_cleared = True

    # file-backed importer 的临时分层必须在 Kit native shutdown 开始前显式释放。
    from linkerbot_sim.assets.robot_import import release_imported_asset_files

    if not bool(binding and binding.imports_released):
        imports_released = attempt(
            "imported asset release",
            release_imported_asset_files,
        )
        if imports_released and binding is not None:
            binding.imports_released = True

    def close_native_app() -> None:
        close = app.close
        try:
            supports_exit_code = "exit_code" in inspect.signature(close).parameters
        except (TypeError, ValueError):
            supports_exit_code = False
        if supports_exit_code:
            close(exit_code=int(exit_code))
        else:
            close()

    if not bool(binding and binding.app_closed):
        app_closed = attempt("SimulationApp.close", close_native_app)
        if app_closed and binding is not None:
            binding.app_closed = True
    else:
        app_closed = True
    if runtime_closed and app_closed and first_error is None:
        _remove_simulation_app_binding(app=app, runtime=runtime)
    if first_error is not None:
        raise first_error


def _close_exact_physics_runtime(runtime: object) -> None:
    """关闭指定 runtime；Newton registry 只接受同一 identity。"""

    from linkerbot_sim.isaac.physics.manager import (
        active_physics_manager,
        release_physics_manager,
    )

    active = active_physics_manager(required=False)
    if active is runtime:
        release_physics_manager(runtime, close=True)
        return
    if getattr(runtime, "backend", None) == "newton" and active is not None:
        raise RuntimeError(
            "refusing to close a Newton runtime while a different manager is active"
        )
    close = getattr(runtime, "close", None)
    if not callable(close):
        raise TypeError("physics runtime must implement close()")
    close()


def _remove_simulation_app_binding(*, app: object, runtime: object | None) -> None:
    with _binding_lock:
        binding = _app_physics_bindings.get(id(app))
        if binding is None or binding.app is not app:
            return
        if runtime is not None and binding.runtime is not runtime:
            raise RuntimeError("refusing to remove a different App/runtime binding")
        _app_physics_bindings.pop(id(app), None)


__all__ = [
    "bound_simulation_app_physics_runtime",
    "close_simulation_app",
    "register_simulation_app_physics_runtime",
]
