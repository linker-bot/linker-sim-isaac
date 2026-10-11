from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.collision_shapes import (
    CONVEX_DECOMPOSITION,
    PhysxCollisionShapeMetadata,
)
from linkerbot_sim.isaac.physics.query_shadow import PhysxQueryShadowOwner
from linkerbot_sim.isaac.physics.scene_query import PhysxSceneQueryAdapter
from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    RobotCollisionShape,
    SceneCollisionTargetCatalog,
    SceneCollisionTargetShape,
)
from linkerbot_sim.kaleidoscope.collision_kinematics import (
    CandidateColliderTransformBatch,
)
from linkerbot_sim.kaleidoscope.collision_overlap import (
    PhysxCandidateOverlapOwner,
)
from linkerbot_sim.kaleidoscope.collision_query import (
    CollisionHitCategory,
    CollisionQueryTarget,
)


IDENTITY = tuple(np.eye(4, dtype=np.float64).reshape(-1))


class _ShadowBackend:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        fail_create: bool = False,
        piece_count: int = 1,
    ) -> None:
        self.events = events
        self.fail_create = fail_create
        self.piece_count = piece_count

    def create_root(self, root_path: str) -> None:
        self.events.append(("create_root", root_path))

    def create_shadow(
        self,
        source_path: str,
        shadow_path: str,
    ) -> tuple[str, ...]:
        self.events.append(("create_shadow", source_path, shadow_path))
        if self.fail_create:
            raise RuntimeError("injected shadow create failure")
        if self.piece_count == 1:
            return (shadow_path,)
        return tuple(
            f"{shadow_path}/piece_{index:04d}" for index in range(self.piece_count)
        )

    def set_world_transform(
        self,
        shadow_path: str,
        matrix_row_major: tuple[float, ...],
    ) -> None:
        self.events.append(("set", shadow_path, matrix_row_major))

    def remove_prim(self, prim_path: str) -> None:
        self.events.append(("remove", prim_path))


class _SceneQueryInterface:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        scheduled_paths: tuple[tuple[str, ...], ...],
    ) -> None:
        self.events = events
        self.scheduled_paths = list(scheduled_paths)

    def overlap_shape(
        self,
        encoded_0: int,
        encoded_1: int,
        callback: object,
        any_hit: bool,
    ) -> None:
        self.events.append(("overlap", encoded_0, encoded_1, any_hit))
        paths = self.scheduled_paths.pop(0)
        for path in paths:
            callback(
                SimpleNamespace(
                    collision=path,
                    rigid_body=path.rsplit("/", maxsplit=1)[0],
                    protoIndex=0,
                )
            )


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


def _target(
    category: CollisionQueryTarget,
    source_name: str,
    *,
    owner_robot_label: str | None = None,
) -> SceneCollisionTargetShape:
    root = f"/World/envs/env_0/{source_name}"
    if owner_robot_label is None:
        collision = f"{root}/collision"
        rigid_body = None
    else:
        collision = f"{root}/Robot/finger/collision"
        rigid_body = f"{root}/Robot/finger"
    return SceneCollisionTargetShape(
        env_id=0,
        category=category,
        source_name=source_name,
        owner_robot_label=owner_robot_label,
        root_path=root,
        collision_path=collision,
        rigid_body_path=rigid_body,
    )


def _target_catalog() -> SceneCollisionTargetCatalog:
    return SceneCollisionTargetCatalog(
        num_envs=1,
        robot_labels=("left", "right"),
        shapes=(
            _target(CollisionQueryTarget.TABLE, "workstation"),
            _target(CollisionQueryTarget.DYNAMIC_OBJECT, "Tblock"),
            _target(
                CollisionQueryTarget.OTHER_ROBOT,
                "left",
                owner_robot_label="left",
            ),
            _target(
                CollisionQueryTarget.OTHER_ROBOT,
                "right",
                owner_robot_label="right",
            ),
        ),
    )


def _candidates(
    *,
    suffix: str = "/finger/collision",
) -> CandidateColliderTransformBatch:
    matrices = np.broadcast_to(np.eye(4), (1, 2, 1, 4, 4)).copy()
    matrices[0, 0, 0, 3, 0] = 10.0
    matrices[0, 1, 0, 3, 0] = 20.0
    return CandidateColliderTransformBatch(
        env_ids=(0,),
        collision_suffixes=(suffix,),
        collision_paths=(("/World/envs/env_0/left/Robot/finger/collision",),),
        matrices=matrices,
    )


def _owners(
    events: list[tuple[object, ...]],
    scheduled_paths: tuple[tuple[str, ...], ...],
    *,
    fail_create: bool = False,
    piece_count: int = 1,
) -> tuple[PhysxQueryShadowOwner, PhysxSceneQueryAdapter]:
    shadow = PhysxQueryShadowOwner(
        object(),
        backend=_ShadowBackend(
            events,
            fail_create=fail_create,
            piece_count=piece_count,
        ),
    )
    interface = _SceneQueryInterface(events, scheduled_paths)
    query = PhysxSceneQueryAdapter(
        interface,
        path_encoder=lambda _path: (101, 202),
    )
    return shadow, query


def test_overlap_owner_streams_transforms_hits_and_aggregated_result() -> None:
    events: list[tuple[object, ...]] = []
    current_robot = "/World/envs/env_0/left/Robot/finger/collision"
    tblock = "/World/envs/env_0/Tblock/collision"
    other_robot = "/World/envs/env_0/right/Robot/finger/collision"
    shadow, query = _owners(
        events,
        ((current_robot,), (other_robot, tblock)),
    )
    owner = PhysxCandidateOverlapOwner(
        robot_catalog=_robot_catalog(),
        robot_label="left",
        shadow_owner=shadow,
        scene_query=query,
    )

    result = owner.execute(
        candidates=_candidates(),
        target_catalog=_target_catalog(),
        targets=tuple(CollisionQueryTarget),
    )

    np.testing.assert_array_equal(result.collision_mask, ((False, True),))
    np.testing.assert_array_equal(result.first_hit_sample, (1,))
    np.testing.assert_array_equal(
        result.first_hit_category,
        (CollisionHitCategory.DYNAMIC_OBJECT,),
    )
    operations = [event[0] for event in events]
    assert operations == [
        "create_root",
        "create_shadow",
        "set",
        "overlap",
        "set",
        "overlap",
    ]
    set_events = [event for event in events if event[0] == "set"]
    assert set_events[0][2][12] == 10.0
    assert set_events[1][2][12] == 20.0
    assert all(event[3] is False for event in events if event[0] == "overlap")

    owner.close()
    assert owner.closed is True
    assert query.closed is True
    assert events[-1] == ("remove", "/World/__linkerbot_collision_query__")
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.execute(
            candidates=_candidates(),
            target_catalog=_target_catalog(),
            targets=(CollisionQueryTarget.TABLE,),
        )


def test_overlap_owner_rejects_candidate_topology_before_moving_shadow() -> None:
    events: list[tuple[object, ...]] = []
    shadow, query = _owners(events, ())
    owner = PhysxCandidateOverlapOwner(
        robot_catalog=_robot_catalog(),
        robot_label="left",
        shadow_owner=shadow,
        scene_query=query,
    )

    with pytest.raises(ValueError, match="suffixes do not match"):
        owner.execute(
            candidates=_candidates(suffix="/wrong/collision"),
            target_catalog=_target_catalog(),
            targets=(CollisionQueryTarget.TABLE,),
        )

    assert not any(event[0] == "set" for event in events)
    owner.close()


def test_overlap_owner_merges_hits_from_every_convex_piece() -> None:
    events: list[tuple[object, ...]] = []
    tblock = "/World/envs/env_0/Tblock/collision"
    shadow, query = _owners(
        events,
        ((), (), (tblock,), ()),
        piece_count=2,
    )
    owner = PhysxCandidateOverlapOwner(
        robot_catalog=_robot_catalog(),
        robot_label="left",
        shadow_owner=shadow,
        scene_query=query,
    )

    result = owner.execute(
        candidates=_candidates(),
        target_catalog=_target_catalog(),
        targets=(CollisionQueryTarget.DYNAMIC_OBJECT,),
    )

    np.testing.assert_array_equal(result.collision_mask, ((False, True),))
    assert len([event for event in events if event[0] == "overlap"]) == 4
    owner.close()


def test_overlap_owner_cleans_both_dependencies_on_construction_failure() -> None:
    events: list[tuple[object, ...]] = []
    shadow, query = _owners(events, (), fail_create=True)

    with pytest.raises(RuntimeError, match="injected shadow create failure"):
        PhysxCandidateOverlapOwner(
            robot_catalog=_robot_catalog(),
            robot_label="left",
            shadow_owner=shadow,
            scene_query=query,
        )

    assert query.closed is True
    assert shadow.closed is True
    assert events[-1] == ("remove", "/World/__linkerbot_collision_query__")
