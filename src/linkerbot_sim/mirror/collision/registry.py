"""Mirror 规划碰撞 provider registry、不可变快照与场景指纹。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from threading import RLock
from time import perf_counter
from typing import Protocol

import numpy as np

from linkerbot_sim.mirror.collision.object_provider import (
    collision_objects_from_runtime_objects,
)
from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.planning.mounted_geometry import (
    MountedCollisionModel,
    cover_box_with_spheres,
)
from linkerbot_sim.planning.collision_validation import AllowedPlanningContact


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
    mounted_models: tuple[MountedCollisionModel, ...] = ()
    allowed_contacts: tuple[AllowedPlanningContact, ...] = ()

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
            "allowed_contacts": [
                (item.robot_id, item.link_name, item.geometry_name)
                for item in self.allowed_contacts
            ],
            "mounted_models": [
                model.fingerprint
                for model in self.mounted_models
                if model.robot_id == target_robot_id
            ],
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

    def __init__(
        self, *, allowed_contacts: tuple[AllowedPlanningContact, ...] = ()
    ) -> None:
        self._providers: dict[str, _ProviderEntry] = {}
        self._lock = RLock()
        self._version = 0
        self._dirty = True
        self._snapshot: PlanningSceneSnapshot | None = None
        self._last_snapshot_duration_s = 0.0
        self._allowed_contacts = allowed_contacts
        self._attachments: tuple[tuple[str, int, tuple[str, ...]], ...] = ()

    @contextmanager
    def contact_scope(self, contacts: Sequence[AllowedPlanningContact]):
        """Add exact planning allowances for one owner-thread task phase only."""
        additions = tuple(contacts)
        if any(
            not isinstance(item, AllowedPlanningContact)
            or not item.link_name
            or not item.geometry_name
            for item in additions
        ):
            raise ValueError(
                "planning contacts require explicit link and geometry names"
            )
        with self._lock:
            previous = self._allowed_contacts
            self._allowed_contacts = tuple(dict.fromkeys((*previous, *additions)))
            self._mark_dirty_locked()
        try:
            yield
        finally:
            with self._lock:
                self._allowed_contacts = previous
                self._mark_dirty_locked()

    def attach_object(
        self, object_name: str, robot_id: int, *, touch_links: Sequence[str] = ()
    ) -> None:
        """Declare a carried planning object; this does not create a physical weld."""
        with self._lock:
            if any(name == object_name for name, _, _ in self._attachments):
                raise ValueError(f"object is already attached: {object_name!r}")
            provider = self._mounted_provider(robot_id)
            model = provider.mounted_collision_model()
            links = tuple(str(link) for link in touch_links)
            if not set(links).issubset(model.source_links):
                raise ValueError("touch_links must name links in the mounted assembly")
            snapshot = self.snapshot(force=True)
            if not any(
                item.source == "object"
                and _belongs_to_object(item.collision.name, object_name)
                for item in snapshot.geometries
            ):
                raise ValueError(
                    f"object has no registered planning geometry: {object_name!r}"
                )
            self._attachments += ((str(object_name), int(robot_id), links),)
            self._mark_dirty_locked()

    def detach_object(self, object_name: str) -> None:
        """Return the object to the live world view on the next snapshot."""
        with self._lock:
            if not any(name == object_name for name, _, _ in self._attachments):
                raise ValueError(f"object is not attached: {object_name!r}")
            self._attachments = tuple(
                item for item in self._attachments if item[0] != object_name
            )
            self._mark_dirty_locked()

    def _mounted_provider(self, robot_id: int):
        providers = [
            entry.provider
            for entry in self._providers.values()
            if entry.owner_robot_id == robot_id
            and callable(getattr(entry.provider, "flange_world_pose", None))
        ]
        if len(providers) != 1:
            raise ValueError(f"robot {robot_id} requires one mounted geometry provider")
        return providers[0]

    def capture_attachments(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "object_name": name,
                    "robot_label": self._mounted_provider(robot_id).label,
                    "touch_links": list(links),
                }
                for name, robot_id, links in self._attachments
            ]

    def prepare_attachments(
        self, payload, *, label_map: Mapping[str, str] | None = None
    ):
        """Validate restore metadata before any physics mutation; keep labels stable."""
        if not isinstance(payload, (list, tuple)):
            raise ValueError("planning attachments must be a sequence")
        labels = {
            entry.provider.label: entry.owner_robot_id
            for entry in self._providers.values()
            if callable(getattr(entry.provider, "flange_world_pose", None))
        }
        objects = {
            item.collision.name.split("/", 1)[0]
            for item in self.snapshot(force=True).geometries
            if item.source in {"object", "payload"}
        }
        result = []
        for value in payload:
            if not isinstance(value, Mapping) or set(value) != {
                "object_name",
                "robot_label",
                "touch_links",
            }:
                raise ValueError("invalid planning attachment metadata")
            label = str(value["robot_label"])
            label = label if label_map is None else label_map.get(label, label)
            if label not in labels or value["object_name"] not in objects:
                raise ValueError(
                    "planning attachment references an unknown robot or object"
                )
            links = value["touch_links"]
            if not isinstance(links, (list, tuple)) or not all(
                isinstance(link, str) for link in links
            ):
                raise ValueError(
                    "planning attachment touch_links must be a sequence of names"
                )
            robot_id = labels[label]
            model = self._mounted_provider(robot_id).mounted_collision_model()
            if not set(links).issubset(model.source_links):
                raise ValueError("planning attachment references an unknown touch link")
            result.append((str(value["object_name"]), robot_id, tuple(links)))
        if len({item[0] for item in result}) != len(result):
            raise ValueError("an object cannot be attached more than once")
        return tuple(result)

    def restore_attachments(self, prepared) -> None:
        with self._lock:
            self._attachments = tuple(prepared)
            self._mark_dirty_locked()

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
        state_views: Mapping[str, object] | None = None,
        name: str = "runtime_objects",
    ) -> None:
        """注册一组随 stage 当前 pose 采样的 runtime objects。"""

        handles = tuple(object_handles)
        views = {} if state_views is None else dict(state_views)
        self.register_provider(
            name,
            lambda: collision_objects_from_runtime_objects(
                handles,
                stage=stage,
                state_views=views,
            ),
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
            models: list[MountedCollisionModel] = []
            for entry in self._providers.values():
                provider = entry.provider
                mounted = getattr(provider, "mounted_collision_model", None)
                model = mounted() if callable(mounted) else None
                if model is not None:
                    models.append(model)
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
            for object_name, robot_id, touch_links in self._attachments:
                provider = self._mounted_provider(robot_id)
                inverse = np.linalg.inv(provider.flange_world_pose())
                index = next(
                    i for i, model in enumerate(models) if model.robot_id == robot_id
                )
                model = models[index]
                payload_link = f"payload:{object_name}"
                spheres = list(model.spheres)
                sphere_links = list(model.sphere_links)
                for i, geometry in enumerate(geometries):
                    if geometry.source != "object" or not _belongs_to_object(
                        geometry.collision.name, object_name
                    ):
                        continue
                    obj = geometry.collision
                    size = obj.padded_size()
                    if obj.shape == "sphere":
                        cells = np.array([[0, 0, 0, size[0]]])
                    else:
                        dimensions = (
                            size
                            if obj.shape == "cuboid"
                            else (2 * size[0], 2 * size[0], size[1] + 2 * size[0])
                        )
                        cells = cover_box_with_spheres(dimensions)
                    relative = inverse @ obj.pose
                    cells[:, :3] = cells[:, :3] @ relative[:3, :3].T + relative[:3, 3]
                    spheres.extend(
                        tuple(float(v) for v in row) for row in cells.round(9)
                    )
                    sphere_links.extend([payload_link] * len(cells))
                    geometries[i] = replace(
                        geometry, source="payload", owner_robot_id=robot_id
                    )
                models[index] = replace(
                    model,
                    spheres=tuple(spheres),
                    sphere_links=tuple(sphere_links),
                    source_links=(*model.source_links, payload_link),
                    payload_contacts=(
                        *model.payload_contacts,
                        *((payload_link, link) for link in touch_links),
                    ),
                )
            frozen = tuple(geometries)
            snapshot = PlanningSceneSnapshot(
                version=self._version,
                geometries=frozen,
                fingerprint=_geometry_fingerprint(frozen),
                sampled_at_s=perf_counter(),
                mounted_models=tuple(models),
                allowed_contacts=self._allowed_contacts,
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
                "attachments": self.capture_attachments(),
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


def _belongs_to_object(geometry_name: str, object_name: str) -> bool:
    return geometry_name == object_name or geometry_name.startswith(object_name + "/")
