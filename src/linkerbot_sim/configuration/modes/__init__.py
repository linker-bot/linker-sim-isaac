"""Mirror 与 Kaleidoscope 产品模式的根配置。"""

from .common import ComputeSettings
from .kaleidoscope import (
    KaleidoscopeConfig,
    KaleidoscopeEnvironmentSettings,
    KaleidoscopePhysicsSettings,
    KaleidoscopeProfileReferences,
    kaleidoscope_mode_from_mapping,
    validate_kaleidoscope_closure,
)
from .mirror import (
    MirrorConfig,
    MirrorPhysicsSettings,
    MirrorProfileReferences,
    mirror_mode_from_mapping,
)

__all__ = [
    "ComputeSettings",
    "KaleidoscopeConfig",
    "KaleidoscopeEnvironmentSettings",
    "KaleidoscopePhysicsSettings",
    "KaleidoscopeProfileReferences",
    "MirrorConfig",
    "MirrorPhysicsSettings",
    "MirrorProfileReferences",
    "kaleidoscope_mode_from_mapping",
    "mirror_mode_from_mapping",
    "validate_kaleidoscope_closure",
]
