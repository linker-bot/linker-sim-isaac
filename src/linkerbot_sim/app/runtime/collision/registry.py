"""规划碰撞 provider registry、不可变快照与场景指纹。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from time import perf_counter
from typing import Protocol

import numpy as np

from linkerbot_sim.app.runtime.collision.object_provider import (
    collision_objects_from_runtime_objects,
)
from linkerbot_sim.planning.collision_objects import CollisionObject


class CollisionGeometryProvider(Protocol):
    """仅在请求场景快照时采样的 CPU 侧碰撞几何来源。"""

    def collision_objects(self) -> Sequence[CollisionObject]:
        """采样 provider 当前 CPU 几何；调用方随后会复制并冻结结果。"""

        ...


@dataclass(frozen=True)
class SceneCollisionGeometry:
    """冻结后的一个碰撞几何及其过滤所有权。"""

    collision: CollisionObject
    source: str
    owner_robot_id: int | None = None


@dataclass(frozen=True)
class PlanningSceneSnapshot:
    """一次 planning transaction 内所有规划共享的不可变场景。"""

    version: int
    geometries: tuple[SceneCollisionGeometry, ...]
    fingerprint: str
    sampled_at_s: float

    def collision_objects_for(
        self,
        target_robot_id: int,
        *,
        include_other_robots: bool = True,
    ) -> tuple[CollisionObject, ...]:
        """排除目标机器人，并按 coordination policy 决定是否保留其它机器人。"""

        result = []
        for geometry in self.geometries:
            owner = geometry.owner_robot_id
            if owner == int(target_robot_id):
                continue
            if owner is not None and not include_other_robots:
                continue
            result.append(geometry.collision)
        return tuple(result)

    def view_fingerprint(
        self,
        target_robot_id: int,
        *,
        include_other_robots: bool,
        shape_policy: str = "curobo-v0.8-cuboid-mesh",
        model_fingerprint: str = "",
    ) -> str:
        """生成目标机器人可见碰撞视图的稳定缓存键。"""

        payload = {
            "snapshot": self.fingerprint,
            "target_robot_id": int(target_robot_id),
            "include_other_robots": bool(include_other_robots),
            "shape_policy": str(shape_policy),
            "model_fingerprint": str(model_fingerprint),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass
class _ProviderEntry:
    """provider callable 及其过滤所有权、来源标签。"""

    name: str
    provider: CollisionGeometryProvider | Callable[[], Sequence[CollisionObject]]
    owner_robot_id: int | None
    source: str


class SceneCollisionRegistry:
    """一个 scene runtime 的权威 CPU 碰撞描述。"""

    def __init__(self) -> None:
        self._providers: dict[str, _ProviderEntry] = {}
        self._lock = RLock()
        self._version = 0
        self._dirty = True
        self._snapshot: PlanningSceneSnapshot | None = None
        self._last_snapshot_duration_s = 0.0

    @property
    def version(self) -> int:
        """返回随动态状态失效递增的 scene version。"""

        with self._lock:
            return self._version

    @property
    def dirty(self) -> bool:
        """返回上次 snapshot 后 scene 是否又发生动态变化。"""

        with self._lock:
            return self._dirty

    def register_provider(
        self,
        name: str,
        provider: CollisionGeometryProvider | Callable[[], Sequence[CollisionObject]],
        *,
        owner_robot_id: int | None = None,
        source: str = "scene",
    ) -> None:
        """注册唯一命名的 provider，并使现有快照失效。"""

        normalized = str(name).strip()
        if not normalized:
            raise ValueError("collision provider name cannot be empty")
        with self._lock:
            if normalized in self._providers:
                raise ValueError(
                    f"collision provider already registered: {normalized!r}"
                )
            self._providers[normalized] = _ProviderEntry(
                normalized,
                provider,
                None if owner_robot_id is None else int(owner_robot_id),
                str(source),
            )
            self._mark_dirty_locked()

    def unregister_provider(self, name: str) -> None:
        """移除命名 provider；存在时递增 version 并使 snapshot 失效。"""

        with self._lock:
            if self._providers.pop(str(name), None) is not None:
                self._mark_dirty_locked()

    def register_runtime_objects(
        self,
        object_handles: Sequence[object],
        *,
        stage: object | None,
        name: str = "runtime_objects",
    ) -> None:
        """注册一组随 stage 当前 pose 采样的 runtime objects。"""

        handles = tuple(object_handles)
        self.register_provider(
            name,
            lambda: collision_objects_from_runtime_objects(handles, stage=stage),
            source="object",
        )

    def mark_dirty(self) -> int:
        """记录动态状态变化，但不直接触碰任何 GPU collision checker。"""

        with self._lock:
            self._mark_dirty_locked()
            return self._version

    def _mark_dirty_locked(self) -> None:
        """在持有 registry lock 时推进 version 并标记 snapshot stale。"""

        self._version += 1
        self._dirty = True

    def snapshot(self, *, force: bool = False) -> PlanningSceneSnapshot:
        """每个 provider 采样一次，返回冻结后的 canonical 场景。"""

        with self._lock:
            if not force and not self._dirty and self._snapshot is not None:
                return self._snapshot
            started = perf_counter()
            geometries: list[SceneCollisionGeometry] = []
            for entry in self._providers.values():
                provider = entry.provider
                values = (
                    provider.collision_objects()
                    if hasattr(provider, "collision_objects")
                    else provider()
                )
                for collision in tuple(values):
                    geometries.append(
                        SceneCollisionGeometry(
                            collision=_freeze_collision_object(collision),
                            source=entry.source,
                            owner_robot_id=entry.owner_robot_id,
                        )
                    )
            frozen = tuple(geometries)
            snapshot = PlanningSceneSnapshot(
                version=self._version,
                geometries=frozen,
                fingerprint=_geometry_fingerprint(frozen),
                sampled_at_s=perf_counter(),
            )
            self._snapshot = snapshot
            self._dirty = False
            self._last_snapshot_duration_s = perf_counter() - started
            return snapshot

    def metrics(self) -> dict[str, object]:
        """返回 scene version、provider/obstacle 数量和最近采样耗时。"""

        with self._lock:
            return {
                "scene_version": self._version,
                "dirty": self._dirty,
                "provider_count": len(self._providers),
                "obstacle_count": (
                    0 if self._snapshot is None else len(self._snapshot.geometries)
                ),
                "snapshot_duration_s": self._last_snapshot_duration_s,
            }


def _freeze_collision_object(value: CollisionObject) -> CollisionObject:
    """深拷贝 CollisionObject pose 并设为只读，隔离 provider 后续修改。"""

    pose = np.asarray(value.pose, dtype=float).reshape(4, 4).copy()
    pose.setflags(write=False)
    return CollisionObject(
        name=str(value.name),
        shape=str(value.shape).lower(),
        pose=pose,
        size=tuple(float(item) for item in value.size),
        enabled=bool(value.enabled),
        padding=float(value.padding),
    )


def _geometry_fingerprint(values: Sequence[SceneCollisionGeometry]) -> str:
    """对排序稳定的 canonical geometry payload 计算 SHA-256 指纹。"""

    payload = []
    for item in values:
        collision = item.collision
        payload.append(
            {
                "name": collision.name,
                "shape": collision.shape,
                "pose": np.asarray(collision.pose).round(12).tolist(),
                "size": list(collision.size),
                "enabled": collision.enabled,
                "padding": collision.padding,
                "owner_robot_id": item.owner_robot_id,
                "source": item.source,
            }
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "CollisionGeometryProvider",
    "PlanningSceneSnapshot",
    "SceneCollisionGeometry",
    "SceneCollisionRegistry",
]
