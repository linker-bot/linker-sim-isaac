"""集中识别 Isaac 物理后端，并让专用功能在错误后端上显式失败。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypeAlias, cast
import warnings


PhysicsBackend: TypeAlias = Literal["physx", "newton"]
PhysicsExecution: TypeAlias = Literal["cpu", "cuda"]
_SUPPORTED_BACKENDS = frozenset({"physx", "newton"})
_SUPPORTED_EXECUTIONS = frozenset({"cpu", "cuda"})
_RUNTIME_OVERRIDE: PhysicsBackend | None = None
_RUNTIME_EXECUTION: PhysicsExecution | None = None
_RUNTIME_REGISTRATION_COUNT = 0


def normalize_physics_backend(value: object) -> PhysicsBackend:
    """规范后端名，并拒绝未知引擎或空值。"""

    name = str(value).strip().lower()
    if name not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"unsupported physics backend {value!r}; "
            f"expected one of {sorted(_SUPPORTED_BACKENDS)}"
        )
    return cast(PhysicsBackend, name)


def active_physics_backend(*, fallback: PhysicsBackend = "physx") -> PhysicsBackend:
    """读取 Isaac 6 active engine；旧 Isaac 或纯 Python 环境回退到 PhysX。"""

    if _RUNTIME_OVERRIDE is not None:
        return _RUNTIME_OVERRIDE

    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except (ImportError, ModuleNotFoundError):
        return normalize_physics_backend(fallback)
    getter = getattr(SimulationManager, "get_active_physics_engine", None)
    if not callable(getter):
        return normalize_physics_backend(fallback)
    # SimulationManager 存在时，getter 异常或未知 engine 都表示已启动运行时不可信。
    # 此处不能伪装成 fallback，否则 Newton/PhysX 能力门控可能在错误后端继续执行。
    return normalize_physics_backend(getter())


def set_runtime_physics_backend(
    backend: object,
    *,
    execution: object,
) -> PhysicsBackend:
    """登记绕过 ``SimulationManager`` 的项目 Newton owner。"""

    normalized = normalize_physics_backend(backend)
    if normalized != "newton":
        raise ValueError("runtime backend override is reserved for Newton")
    normalized_execution = normalize_physics_execution(execution)
    global _RUNTIME_OVERRIDE, _RUNTIME_EXECUTION, _RUNTIME_REGISTRATION_COUNT
    if _RUNTIME_OVERRIDE is not None and (
        normalized != _RUNTIME_OVERRIDE or normalized_execution != _RUNTIME_EXECUTION
    ):
        raise RuntimeError(
            "a different Newton physics backend override is already active"
        )
    _RUNTIME_OVERRIDE = normalized
    _RUNTIME_EXECUTION = normalized_execution
    _RUNTIME_REGISTRATION_COUNT += 1
    return normalized


def clear_runtime_physics_backend(*, backend: object | None = None) -> None:
    """释放一次 Newton App/backend 登记，最后一个登记释放时清除声明。"""

    global _RUNTIME_OVERRIDE, _RUNTIME_EXECUTION, _RUNTIME_REGISTRATION_COUNT
    if _RUNTIME_OVERRIDE is None:
        if _RUNTIME_REGISTRATION_COUNT != 0:
            raise RuntimeError("runtime backend registration count is inconsistent")
        return
    if backend is not None and _RUNTIME_OVERRIDE is not None:
        expected = normalize_physics_backend(backend)
        if expected != _RUNTIME_OVERRIDE:
            raise RuntimeError(
                "refusing to clear a different runtime physics backend override"
            )
    if _RUNTIME_REGISTRATION_COUNT <= 0:
        raise RuntimeError("runtime backend registration count is inconsistent")
    _RUNTIME_REGISTRATION_COUNT -= 1
    if _RUNTIME_REGISTRATION_COUNT == 0:
        _RUNTIME_OVERRIDE = None
        _RUNTIME_EXECUTION = None


def normalize_physics_execution(value: object) -> PhysicsExecution:
    """规范物理执行设备类别。"""

    execution = str(value).strip().lower()
    if execution not in _SUPPORTED_EXECUTIONS:
        raise ValueError(
            f"unsupported physics execution {value!r}; "
            f"expected one of {sorted(_SUPPORTED_EXECUTIONS)}"
        )
    return cast(PhysicsExecution, execution)


def active_physics_execution(*, fallback: PhysicsExecution = "cpu") -> PhysicsExecution:
    """返回当前 owner 的 CPU/CUDA 执行类别。"""

    if _RUNTIME_EXECUTION is not None:
        return _RUNTIME_EXECUTION
    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except (ImportError, ModuleNotFoundError):
        return normalize_physics_execution(fallback)
    getter = getattr(SimulationManager, "get_physics_sim_device", None)
    if not callable(getter):
        return normalize_physics_execution(fallback)
    device = str(getter()).strip().lower()
    if device == "cpu":
        return "cpu"
    if device == "cuda" or device.startswith("cuda:"):
        return "cuda"
    raise RuntimeError(f"unsupported active physics device {device!r}")


def require_physics_backend(
    supported: PhysicsBackend | Iterable[PhysicsBackend],
    *,
    feature: str,
    backend: object | None = None,
) -> PhysicsBackend:
    """要求功能运行在声明支持的后端，禁止静默忽略专用配置。"""

    allowed = (
        frozenset({normalize_physics_backend(supported)})
        if isinstance(supported, str)
        else frozenset(normalize_physics_backend(item) for item in supported)
    )
    active = (
        active_physics_backend()
        if backend is None
        else normalize_physics_backend(backend)
    )
    if active not in allowed:
        raise RuntimeError(
            f"{feature} requires physics backend in {sorted(allowed)}, "
            f"but active backend is {active!r}"
        )
    return active


class PhysicsCompatibilityWarning(RuntimeWarning):
    """携带后端、功能和被跳过字段的运行时兼容性告警。"""

    def __init__(
        self,
        *,
        backend: object,
        feature: str,
        skipped_fields: Iterable[str],
        reason: str,
    ) -> None:
        self.backend: PhysicsBackend = normalize_physics_backend(backend)
        self.feature = str(feature)
        self.skipped_fields = tuple(sorted(set(str(item) for item in skipped_fields)))
        self.reason = str(reason)
        fields = ", ".join(self.skipped_fields)
        super().__init__(
            "physics compatibility degradation: "
            f"backend={self.backend!r}; feature={self.feature!r}; "
            f"skipped_fields=[{fields}]; reason={self.reason}"
        )


def warn_unsupported_physics_fields(
    *,
    backend: object,
    feature: str,
    fields: Iterable[str],
    reason: str,
    stacklevel: int = 2,
) -> int:
    """对无等价后端语义的配置发出结构化告警，并返回唯一字段数。"""

    skipped_fields = tuple(sorted(set(str(item) for item in fields)))
    if not skipped_fields:
        return 0
    warnings.warn(
        PhysicsCompatibilityWarning(
            backend=backend,
            feature=feature,
            skipped_fields=skipped_fields,
            reason=reason,
        ),
        stacklevel=stacklevel,
    )
    return len(skipped_fields)


__all__ = [
    "PhysicsCompatibilityWarning",
    "PhysicsBackend",
    "PhysicsExecution",
    "active_physics_backend",
    "active_physics_execution",
    "clear_runtime_physics_backend",
    "normalize_physics_backend",
    "normalize_physics_execution",
    "require_physics_backend",
    "set_runtime_physics_backend",
    "warn_unsupported_physics_fields",
]
