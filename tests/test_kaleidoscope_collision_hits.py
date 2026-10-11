from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.scene_query import PhysxOverlapHit
from linkerbot_sim.kaleidoscope.collision_catalog import (
    SceneCollisionTargetCatalog,
    SceneCollisionTargetShape,
)
from linkerbot_sim.kaleidoscope.collision_hits import (
    AggregatedCollisionQueryHits,
    CollisionQueryHitAccumulator,
    aggregate_collision_query_hits,
)
from linkerbot_sim.kaleidoscope.collision_kinematics import (
    CandidateColliderTransformBatch,
)
from linkerbot_sim.kaleidoscope.collision_query import (
    CollisionHitCategory,
    CollisionQueryTarget,
)


def _target(
    env_id: int,
    category: CollisionQueryTarget,
    source_name: str,
    *,
    owner_robot_label: str | None = None,
) -> SceneCollisionTargetShape:
    root = f"/World/envs/env_{env_id}/{source_name}"
    collision = f"{root}/collision"
    return SceneCollisionTargetShape(
        env_id=env_id,
        category=category,
        source_name=source_name,
        owner_robot_label=owner_robot_label,
        root_path=root,
        collision_path=collision,
        rigid_body_path=None if owner_robot_label is None else root,
    )


def _catalog() -> SceneCollisionTargetCatalog:
    shapes = []
    for env_id in range(2):
        shapes.extend(
            (
                _target(env_id, CollisionQueryTarget.TABLE, "workstation"),
                _target(env_id, CollisionQueryTarget.DYNAMIC_OBJECT, "Tblock"),
                _target(
                    env_id,
                    CollisionQueryTarget.OTHER_ROBOT,
                    "left",
                    owner_robot_label="left",
                ),
                _target(
                    env_id,
                    CollisionQueryTarget.OTHER_ROBOT,
                    "right",
                    owner_robot_label="right",
                ),
            )
        )
    return SceneCollisionTargetCatalog(
        num_envs=2,
        robot_labels=("left", "right"),
        shapes=tuple(shapes),
    )


def _candidates() -> CandidateColliderTransformBatch:
    matrices = np.broadcast_to(np.eye(4), (2, 3, 2, 4, 4)).copy()
    return CandidateColliderTransformBatch(
        env_ids=(0, 1),
        collision_suffixes=("/palm/collision", "/finger/collision"),
        collision_paths=(
            (
                "/World/envs/env_0/left/Robot/palm/collision",
                "/World/envs/env_0/left/Robot/finger/collision",
            ),
            (
                "/World/envs/env_1/left/Robot/palm/collision",
                "/World/envs/env_1/left/Robot/finger/collision",
            ),
        ),
        matrices=matrices,
    )


def _hit(path: str) -> PhysxOverlapHit:
    return PhysxOverlapHit(
        collision_path=path,
        rigid_body_path=path.rsplit("/", maxsplit=1)[0],
        prototype_index=0,
    )


def _empty_hits() -> list[list[list[tuple[PhysxOverlapHit, ...]]]]:
    return [[[(), ()] for _sample in range(3)] for _env in range(2)]


def test_aggregation_filters_identity_and_uses_stable_first_category() -> None:
    hits = _empty_hits()
    hits[0][0][0] = (
        _hit("/World/envs/env_0/left/collision"),
        _hit("/World/envs/env_1/workstation/collision"),
        _hit("/World/envs/env_0/workstation/collision_suffix"),
        _hit("/World/__linkerbot_collision_query__/shadow"),
    )
    hits[0][1][0] = (
        _hit("/World/envs/env_0/Tblock/collision"),
        _hit("/World/envs/env_0/workstation/collision"),
    )
    hits[0][1][1] = (_hit("/World/envs/env_0/right/collision"),)
    hits[0][2][0] = (_hit("/World/envs/env_0/Tblock/collision"),)

    result = aggregate_collision_query_hits(
        candidates=_candidates(),
        target_catalog=_catalog(),
        robot_label="left",
        targets=tuple(CollisionQueryTarget),
        overlap_hits=hits,
    )

    np.testing.assert_array_equal(
        result.collision_mask,
        ((False, True, True), (False, False, False)),
    )
    np.testing.assert_array_equal(result.first_hit_sample, (1, -1))
    np.testing.assert_array_equal(
        result.first_hit_category,
        (CollisionHitCategory.TABLE, CollisionHitCategory.NONE),
    )
    assert result.collision_mask.flags.writeable is False
    assert result.first_hit_sample.flags.writeable is False
    assert result.first_hit_category.flags.writeable is False


def test_aggregation_honors_requested_target_subset() -> None:
    hits = _empty_hits()
    hits[0][0][0] = (_hit("/World/envs/env_0/workstation/collision"),)
    hits[0][2][0] = (_hit("/World/envs/env_0/Tblock/collision"),)

    result = aggregate_collision_query_hits(
        candidates=_candidates(),
        target_catalog=_catalog(),
        robot_label="left",
        targets=(CollisionQueryTarget.DYNAMIC_OBJECT,),
        overlap_hits=hits,
    )

    np.testing.assert_array_equal(result.collision_mask[0], (False, False, True))
    assert result.first_hit_sample[0] == 2
    assert result.first_hit_category[0] == CollisionHitCategory.DYNAMIC_OBJECT


def test_streaming_accumulator_requires_each_cell_once_and_keeps_no_hit_batch() -> None:
    accumulator = CollisionQueryHitAccumulator(
        candidates=_candidates(),
        target_catalog=_catalog(),
        robot_label="left",
        targets=tuple(CollisionQueryTarget),
    )
    accumulator.record(
        env_row=0,
        sample_index=0,
        collider_index=0,
        hits=(),
    )
    with pytest.raises(RuntimeError, match="already recorded"):
        accumulator.record(
            env_row=0,
            sample_index=0,
            collider_index=0,
            hits=(),
        )
    with pytest.raises(RuntimeError, match="unrecorded cells"):
        accumulator.finish()

    for env_row in range(2):
        for sample_index in range(3):
            for collider_index in range(2):
                if (env_row, sample_index, collider_index) == (0, 0, 0):
                    continue
                hits: tuple[PhysxOverlapHit, ...] = ()
                if (env_row, sample_index, collider_index) == (0, 1, 0):
                    hits = (_hit("/World/envs/env_0/Tblock/collision"),)
                if (env_row, sample_index, collider_index) == (0, 1, 1):
                    hits = (_hit("/World/envs/env_0/workstation/collision"),)
                accumulator.record(
                    env_row=env_row,
                    sample_index=sample_index,
                    collider_index=collider_index,
                    hits=hits,
                )

    result = accumulator.finish()
    assert accumulator.finish() is result
    assert result.first_hit_sample[0] == 1
    assert result.first_hit_category[0] == CollisionHitCategory.TABLE
    with pytest.raises(RuntimeError, match="already finished"):
        accumulator.record(
            env_row=0,
            sample_index=2,
            collider_index=0,
            hits=(),
        )


@pytest.mark.parametrize(
    ("hits", "message"),
    [
        ([], "environment axis"),
        (
            [
                [[(), ()]],
                [[(), ()]],
            ],
            "sample axis",
        ),
        (
            [
                [[()] for _sample in range(3)],
                [[()] for _sample in range(3)],
            ],
            "collider axis",
        ),
        (
            [
                [[(object(),), ()] for _sample in range(3)],
                [[(), ()] for _sample in range(3)],
            ],
            "invalid hit value",
        ),
    ],
)
def test_aggregation_rejects_misaligned_hit_batches(
    hits: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        aggregate_collision_query_hits(
            candidates=_candidates(),
            target_catalog=_catalog(),
            robot_label="left",
            targets=(CollisionQueryTarget.TABLE,),
            overlap_hits=hits,  # type: ignore[arg-type]
        )


def test_aggregated_result_rejects_inconsistent_first_hit_contract() -> None:
    with pytest.raises(ValueError, match="first_hit_sample"):
        AggregatedCollisionQueryHits(
            collision_mask=np.asarray([[False, True]], dtype=np.bool_),
            first_hit_sample=np.asarray([0], dtype=np.int64),
            first_hit_category=np.asarray(
                [CollisionHitCategory.TABLE],
                dtype=np.int64,
            ),
        )
    with pytest.raises(ValueError, match="first_hit_category"):
        AggregatedCollisionQueryHits(
            collision_mask=np.asarray([[False, False]], dtype=np.bool_),
            first_hit_sample=np.asarray([-1], dtype=np.int64),
            first_hit_category=np.asarray(
                [CollisionHitCategory.TABLE],
                dtype=np.int64,
            ),
        )
