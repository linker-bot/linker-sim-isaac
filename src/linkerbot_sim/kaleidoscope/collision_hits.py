"""Pure exact-identity filtering and discrete-path collision aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.isaac.physics.scene_query import PhysxOverlapHit
from linkerbot_sim.kaleidoscope.collision_catalog import SceneCollisionTargetCatalog
from linkerbot_sim.kaleidoscope.collision_kinematics import (
    CandidateColliderTransformBatch,
)
from linkerbot_sim.kaleidoscope.collision_query import (
    CollisionHitCategory,
    CollisionQueryTarget,
)


_HIT_CATEGORY = {
    CollisionQueryTarget.TABLE: CollisionHitCategory.TABLE,
    CollisionQueryTarget.DYNAMIC_OBJECT: CollisionHitCategory.DYNAMIC_OBJECT,
    CollisionQueryTarget.OTHER_ROBOT: CollisionHitCategory.OTHER_ROBOT,
}


@dataclass(frozen=True, slots=True)
class AggregatedCollisionQueryHits:
    """Read-only NumPy result before the public Torch boundary."""

    collision_mask: np.ndarray
    first_hit_sample: np.ndarray
    first_hit_category: np.ndarray

    def __post_init__(self) -> None:
        mask = self.collision_mask
        first = self.first_hit_sample
        category = self.first_hit_category
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or min(mask.shape) < 1:
            raise ValueError("collision_mask must have shape [envs, samples]")
        if mask.dtype != np.dtype(np.bool_):
            raise TypeError("collision_mask must use bool dtype")
        expected_rows = (mask.shape[0],)
        if not isinstance(first, np.ndarray) or first.shape != expected_rows:
            raise ValueError("first_hit_sample must have shape [envs]")
        if first.dtype != np.dtype(np.int64):
            raise TypeError("first_hit_sample must use int64 dtype")
        if not isinstance(category, np.ndarray) or category.shape != expected_rows:
            raise ValueError("first_hit_category must have shape [envs]")
        if category.dtype != np.dtype(np.int64):
            raise TypeError("first_hit_category must use int64 dtype")

        valid_categories = {int(value) for value in CollisionHitCategory}
        if any(int(value) not in valid_categories for value in category):
            raise ValueError("first_hit_category contains an unknown category code")
        for row in range(mask.shape[0]):
            hit_samples = np.flatnonzero(mask[row])
            expected_first = -1 if hit_samples.size == 0 else int(hit_samples[0])
            if int(first[row]) != expected_first:
                raise ValueError("first_hit_sample does not match collision_mask")
            has_category = int(category[row]) != int(CollisionHitCategory.NONE)
            if has_category != (expected_first >= 0):
                raise ValueError("first_hit_category does not match collision_mask")
        mask.setflags(write=False)
        first.setflags(write=False)
        category.setflags(write=False)


class CollisionQueryHitAccumulator:
    """Incrementally aggregate native callbacks without retaining every hit."""

    def __init__(
        self,
        *,
        candidates: CandidateColliderTransformBatch,
        target_catalog: SceneCollisionTargetCatalog,
        robot_label: str,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> None:
        _validate_context(
            candidates=candidates,
            target_catalog=target_catalog,
            robot_label=robot_label,
            targets=targets,
        )
        shape = candidates.matrices.shape[:3]
        self._candidates = candidates
        self._target_catalog = target_catalog
        self._robot_label = robot_label
        self._targets = targets
        self._mask = np.zeros(shape[:2], dtype=np.bool_)
        self._sample_category = np.zeros(shape[:2], dtype=np.int64)
        self._visited = np.zeros(shape, dtype=np.bool_)
        self._result: AggregatedCollisionQueryHits | None = None

    def record(
        self,
        *,
        env_row: int,
        sample_index: int,
        collider_index: int,
        hits: Sequence[PhysxOverlapHit],
    ) -> None:
        """Consume one shadow-overlap callback result exactly once."""

        if self._result is not None:
            raise RuntimeError("collision hit accumulator is already finished")
        indices = (env_row, sample_index, collider_index)
        limits = self._visited.shape
        if any(type(value) is not int for value in indices) or any(
            value < 0 or value >= limit
            for value, limit in zip(indices, limits, strict=True)
        ):
            raise ValueError("collision hit accumulator index is outside batch shape")
        if self._visited[indices]:
            raise RuntimeError("collision hit accumulator cell was already recorded")
        hit_values = tuple(hits)
        if any(not isinstance(hit, PhysxOverlapHit) for hit in hit_values):
            raise TypeError("hits contains an invalid PhysxOverlapHit value")

        env_id = self._candidates.env_ids[env_row]
        for hit in hit_values:
            selected = self._target_catalog.category_for_collision_path(
                hit.collision_path,
                env_id=env_id,
                robot_label=self._robot_label,
                targets=self._targets,
            )
            if selected is None:
                continue
            code = int(_HIT_CATEGORY[selected])
            self._mask[env_row, sample_index] = True
            current = int(self._sample_category[env_row, sample_index])
            if current == int(CollisionHitCategory.NONE) or code < current:
                self._sample_category[env_row, sample_index] = code
        self._visited[indices] = True

    def finish(self) -> AggregatedCollisionQueryHits:
        """Return the immutable result after every candidate cell was recorded."""

        if self._result is not None:
            return self._result
        if not bool(np.all(self._visited)):
            raise RuntimeError("collision hit accumulator has unrecorded cells")
        first = np.full((self._mask.shape[0],), -1, dtype=np.int64)
        category = np.full(
            (self._mask.shape[0],),
            int(CollisionHitCategory.NONE),
            dtype=np.int64,
        )
        for env_row in range(self._mask.shape[0]):
            hit_samples = np.flatnonzero(self._mask[env_row])
            if hit_samples.size == 0:
                continue
            sample_index = int(hit_samples[0])
            first[env_row] = sample_index
            category[env_row] = self._sample_category[env_row, sample_index]
        self._result = AggregatedCollisionQueryHits(
            collision_mask=self._mask,
            first_hit_sample=first,
            first_hit_category=category,
        )
        return self._result


def aggregate_collision_query_hits(
    *,
    candidates: CandidateColliderTransformBatch,
    target_catalog: SceneCollisionTargetCatalog,
    robot_label: str,
    targets: tuple[CollisionQueryTarget, ...],
    overlap_hits: Sequence[Sequence[Sequence[Sequence[PhysxOverlapHit]]]],
) -> AggregatedCollisionQueryHits:
    """Filter exact target paths and aggregate ``[env, sample, collider]`` hits."""

    accumulator = CollisionQueryHitAccumulator(
        candidates=candidates,
        target_catalog=target_catalog,
        robot_label=robot_label,
        targets=targets,
    )
    env_count, sample_count, collider_count = candidates.matrices.shape[:3]
    hit_rows = _validated_hit_rows(
        overlap_hits,
        env_count=env_count,
        sample_count=sample_count,
        collider_count=collider_count,
    )
    for env_row in range(env_count):
        for sample_index in range(sample_count):
            for collider_index in range(collider_count):
                accumulator.record(
                    env_row=env_row,
                    sample_index=sample_index,
                    collider_index=collider_index,
                    hits=hit_rows[env_row][sample_index][collider_index],
                )
    return accumulator.finish()


def _validate_context(
    *,
    candidates: CandidateColliderTransformBatch,
    target_catalog: SceneCollisionTargetCatalog,
    robot_label: str,
    targets: tuple[CollisionQueryTarget, ...],
) -> None:
    if not isinstance(candidates, CandidateColliderTransformBatch):
        raise TypeError("candidates must be CandidateColliderTransformBatch")
    if not isinstance(target_catalog, SceneCollisionTargetCatalog):
        raise TypeError("target_catalog must be SceneCollisionTargetCatalog")
    if any(
        type(env_id) is not int or not 0 <= env_id < target_catalog.num_envs
        for env_id in candidates.env_ids
    ):
        raise ValueError("candidate env_id is outside target catalog range")
    if robot_label not in target_catalog.robot_labels:
        raise KeyError(f"unknown robot label: {robot_label!r}")
    for env_id in candidates.env_ids:
        # Validate the request even when every native callback is empty.
        target_catalog.targets_for(
            env_id=env_id,
            robot_label=robot_label,
            targets=targets,
        )


def _validated_hit_rows(
    values: Sequence[Sequence[Sequence[Sequence[PhysxOverlapHit]]]],
    *,
    env_count: int,
    sample_count: int,
    collider_count: int,
) -> tuple[tuple[tuple[tuple[PhysxOverlapHit, ...], ...], ...], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("overlap_hits must be a nested sequence")
    env_rows = tuple(values)
    if len(env_rows) != env_count:
        raise ValueError("overlap_hits environment axis does not match candidates")
    result = []
    for samples in env_rows:
        sample_rows = tuple(samples)
        if len(sample_rows) != sample_count:
            raise ValueError("overlap_hits sample axis does not match candidates")
        result_samples = []
        for colliders in sample_rows:
            collider_rows = tuple(colliders)
            if len(collider_rows) != collider_count:
                raise ValueError("overlap_hits collider axis does not match candidates")
            result_colliders = []
            for hits in collider_rows:
                hit_values = tuple(hits)
                if any(not isinstance(hit, PhysxOverlapHit) for hit in hit_values):
                    raise TypeError("overlap_hits contains an invalid hit value")
                result_colliders.append(hit_values)
            result_samples.append(tuple(result_colliders))
        result.append(tuple(result_samples))
    return tuple(result)


__all__ = [
    "AggregatedCollisionQueryHits",
    "CollisionQueryHitAccumulator",
    "aggregate_collision_query_hits",
]
