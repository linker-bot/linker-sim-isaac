"""Multi-robot ownership and dispatch for internal collision query services."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    SceneCollisionTargetCatalog,
)
from linkerbot_sim.kaleidoscope.collision_hits import AggregatedCollisionQueryHits
from linkerbot_sim.kaleidoscope.collision_query import CollisionQueryTarget


@runtime_checkable
class ArticulatedCollisionServicePort(Protocol):
    robot_label: str
    command_joint_names: tuple[str, ...]

    def query(
        self,
        *,
        joint_positions: np.ndarray,
        env_ids: Sequence[int],
        target_catalog: SceneCollisionTargetCatalog,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> AggregatedCollisionQueryHits: ...

    def close(self) -> None: ...


CollisionServiceFactory = Callable[[str], ArticulatedCollisionServicePort]


class ArticulatedCollisionServiceRegistry:
    """Own one internal collision service for every scene robot label."""

    def __init__(
        self,
        *,
        robot_labels: tuple[str, ...],
        services: tuple[ArticulatedCollisionServicePort, ...],
    ) -> None:
        if (
            not isinstance(robot_labels, tuple)
            or not robot_labels
            or len(set(robot_labels)) != len(robot_labels)
            or any(not isinstance(label, str) or not label for label in robot_labels)
        ):
            raise ValueError("robot_labels must be a non-empty unique string tuple")
        if (
            not isinstance(services, tuple)
            or len(services) != len(robot_labels)
            or any(
                not isinstance(service, ArticulatedCollisionServicePort)
                for service in services
            )
        ):
            raise TypeError("services must contain one collision service per robot")
        actual_labels = tuple(service.robot_label for service in services)
        if actual_labels != robot_labels:
            raise ValueError("service labels must exactly match robot label order")
        self._robot_labels = robot_labels
        self._services = services
        self._by_label = dict(zip(robot_labels, services, strict=True))
        self._closed = False

    @property
    def robot_labels(self) -> tuple[str, ...]:
        return self._robot_labels

    @property
    def closed(self) -> bool:
        return self._closed

    def command_joint_names_for(self, robot_label: str) -> tuple[str, ...]:
        """Return the exact public input order for one robot service."""

        self._require_open()
        try:
            return self._by_label[robot_label].command_joint_names
        except KeyError as exc:
            raise KeyError(f"unknown robot label: {robot_label!r}") from exc

    def query(
        self,
        *,
        robot_label: str,
        joint_positions: np.ndarray,
        env_ids: Sequence[int],
        target_catalog: SceneCollisionTargetCatalog,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> AggregatedCollisionQueryHits:
        """Dispatch one request without changing its env or sample row order."""

        self._require_open()
        try:
            service = self._by_label[robot_label]
        except KeyError as exc:
            raise KeyError(f"unknown robot label: {robot_label!r}") from exc
        return service.query(
            joint_positions=joint_positions,
            env_ids=env_ids,
            target_catalog=target_catalog,
            targets=targets,
        )

    def close(self) -> None:
        """Close robot services in reverse construction order."""

        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for service in reversed(self._services):
            try:
                service.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note(
                        f"additional collision service close failure: {exc!r}"
                    )
        self._services = ()
        self._by_label.clear()
        if first_error is not None:
            raise first_error

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("articulated collision service registry is closed")


def build_articulated_collision_service_registry(
    *,
    robot_catalog: RobotCollisionCatalog,
    service_factory: CollisionServiceFactory,
) -> ArticulatedCollisionServiceRegistry:
    """Build all robot services with reverse rollback after partial failure."""

    if not isinstance(robot_catalog, RobotCollisionCatalog):
        raise TypeError("robot_catalog must be RobotCollisionCatalog")
    if not callable(service_factory):
        raise TypeError("service_factory must be callable")
    created: list[object] = []
    validated: list[ArticulatedCollisionServicePort] = []
    try:
        for robot_label in robot_catalog.robot_labels:
            service = service_factory(robot_label)
            created.append(service)
            if not isinstance(service, ArticulatedCollisionServicePort):
                raise TypeError(
                    f"factory returned an invalid service for {robot_label!r}"
                )
            if service.robot_label != robot_label:
                raise ValueError(
                    "factory service label does not match requested robot: "
                    f"{service.robot_label!r} != {robot_label!r}"
                )
            validated.append(service)
        return ArticulatedCollisionServiceRegistry(
            robot_labels=robot_catalog.robot_labels,
            services=tuple(validated),
        )
    except BaseException as exc:
        _close_partial_services(created, original=exc)
        raise


def _close_partial_services(
    services: Sequence[object],
    *,
    original: BaseException,
) -> None:
    for service in reversed(services):
        close = getattr(service, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as cleanup_exc:
            original.add_note(
                f"partial collision service cleanup failed: {cleanup_exc!r}"
            )


__all__ = [
    "ArticulatedCollisionServicePort",
    "ArticulatedCollisionServiceRegistry",
    "CollisionServiceFactory",
    "build_articulated_collision_service_registry",
]
