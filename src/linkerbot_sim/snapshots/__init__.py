"""版本化 SceneSnapshot schema 与显式冷存储 facade。"""

from .persistence import (
    load_scene_snapshot,
    save_scene_snapshot,
    validate_scene_snapshot,
)
from .schema import SceneSnapshot

__all__ = [
    "SceneSnapshot",
    "load_scene_snapshot",
    "save_scene_snapshot",
    "validate_scene_snapshot",
]
