from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.collision_shapes import (
    CONVEX_DECOMPOSITION,
    PhysxCollisionShapeMetadata,
)
from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    RobotCollisionShape,
)
from linkerbot_sim.kaleidoscope.collision_hits import AggregatedCollisionQueryHits
from linkerbot_sim.kaleidoscope.collision_query import (
    CollisionHitCategory,
    CollisionQueryTarget,
)
from linkerbot_sim.kaleidoscope.collision_registry import (
    ArticulatedCollisionServiceRegistry,
    build_articulated_collision_service_registry,
)


IDENTITY = tuple(np.eye(4, dtype=np.float64).reshape(-1))


class _Service:
    def __init__(
        self,
        robot_label: str,
        events: list[tuple[str, str]],
        *,
        fail_close: bool = False,
    ) -> None:
        self.robot_label = robot_label
        self.command_joint_names = ("joint_0",)
        self.events = events
        self.fail_close = fail_close
        self.calls: list[dict[str, object]] = []

    def query(self, **kwargs: object) -> AggregatedCollisionQueryHits:
        self.events.append(("query", self.robot_label))
        self.calls.append(kwargs)
        positions = kwargs["joint_positions"]
        assert isinstance(positions, np.ndarray)
        return AggregatedCollisionQueryHits(
            collision_mask=np.zeros(positions.shape[:2], dtype=np.bool_),
            first_hit_sample=np.full((positions.shape[0],), -1, dtype=np.int64),
            first_hit_category=np.full(
                (positions.shape[0],),
                CollisionHitCategory.NONE,
                dtype=np.int64,
            ),
        )

    def close(self) -> None:
        self.events.append(("close", self.robot_label))
        if self.fail_close:
            raise RuntimeError(f"injected {self.robot_label} close failure")


def _robot_catalog() -> RobotCollisionCatalog:
    shapes = []
    for robot_id, label in enumerate(("left", "right")):
        imported_root = f"/World/envs/env_0/{label}"
        articulation = f"{imported_root}/Robot"
        rigid_body = f"{articulation}/finger"
        collision = f"{rigid_body}/collision"
        shapes.append(
            RobotCollisionShape(
                env_id=0,
                robot_id=robot_id,
                robot_label=label,
                env_root_path="/World/envs/env_0",
                imported_root_path=imported_root,
                articulation_path=articulation,
                collision_suffix="/finger/collision",
                rigid_body_suffix="/finger",
                metadata=PhysxCollisionShapeMetadata(
                    collision_path=collision,
                    rigid_body_path=rigid_body,
                    geometry_type="Mesh",
                    mesh_approximation=CONVEX_DECOMPOSITION,
                    shape_to_rigid_body_matrix_row_major=IDENTITY,
                ),
            )
        )
    return RobotCollisionCatalog(
        num_envs=1,
        robot_labels=("left", "right"),
        shapes=tuple(shapes),
    )


def test_registry_builds_in_catalog_order_and_dispatches_exact_label() -> None:
    events: list[tuple[str, str]] = []
    services: dict[str, _Service] = {}

    def factory(label: str) -> _Service:
        events.append(("create", label))
        service = _Service(label, events)
        services[label] = service
        return service

    registry = build_articulated_collision_service_registry(
        robot_catalog=_robot_catalog(),
        service_factory=factory,
    )
    positions = np.zeros((1, 2, 1), dtype=np.float64)
    result = registry.query(
        robot_label="right",
        joint_positions=positions,
        env_ids=(0,),
        target_catalog=object(),  # type: ignore[arg-type]
        targets=(CollisionQueryTarget.TABLE,),
    )

    assert registry.robot_labels == ("left", "right")
    assert events == [("create", "left"), ("create", "right"), ("query", "right")]
    assert result.collision_mask.shape == (1, 2)
    assert services["left"].calls == []
    assert services["right"].calls[0]["env_ids"] == (0,)

    with pytest.raises(KeyError, match="unknown robot label"):
        registry.query(
            robot_label="missing",
            joint_positions=positions,
            env_ids=(0,),
            target_catalog=object(),  # type: ignore[arg-type]
            targets=(CollisionQueryTarget.TABLE,),
        )


def test_registry_close_is_reverse_idempotent_and_continues_after_error() -> None:
    events: list[tuple[str, str]] = []
    left = _Service("left", events)
    right = _Service("right", events, fail_close=True)
    registry = ArticulatedCollisionServiceRegistry(
        robot_labels=("left", "right"),
        services=(left, right),
    )

    with pytest.raises(RuntimeError, match="right close failure"):
        registry.close()

    assert events == [("close", "right"), ("close", "left")]
    assert registry.closed is True
    registry.close()
    with pytest.raises(RuntimeError, match="registry is closed"):
        registry.query(
            robot_label="left",
            joint_positions=np.zeros((1, 1, 1), dtype=np.float64),
            env_ids=(0,),
            target_catalog=object(),  # type: ignore[arg-type]
            targets=(CollisionQueryTarget.TABLE,),
        )


def test_registry_factory_rolls_back_previous_services_in_reverse() -> None:
    events: list[tuple[str, str]] = []

    def factory(label: str) -> _Service:
        events.append(("create", label))
        if label == "right":
            raise RuntimeError("injected right construction failure")
        return _Service(label, events)

    with pytest.raises(RuntimeError, match="right construction failure"):
        build_articulated_collision_service_registry(
            robot_catalog=_robot_catalog(),
            service_factory=factory,
        )

    assert events == [
        ("create", "left"),
        ("create", "right"),
        ("close", "left"),
    ]


def test_registry_factory_rejects_wrong_service_identity_and_cleans_it() -> None:
    events: list[tuple[str, str]] = []

    def factory(label: str) -> _Service:
        events.append(("create", label))
        return _Service("left", events)

    with pytest.raises(ValueError, match="does not match requested robot"):
        build_articulated_collision_service_registry(
            robot_catalog=_robot_catalog(),
            service_factory=factory,
        )

    assert events[-2:] == [("close", "left"), ("close", "left")]
