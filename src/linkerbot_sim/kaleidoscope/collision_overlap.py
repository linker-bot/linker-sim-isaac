"""Isaac-thread execution of candidate colliders through query-only shadows."""

from __future__ import annotations

from linkerbot_sim.isaac.physics.query_shadow import (
    PhysxQueryShadow,
    PhysxQueryShadowOwner,
)
from linkerbot_sim.isaac.physics.scene_query import PhysxSceneQueryAdapter
from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    SceneCollisionTargetCatalog,
)
from linkerbot_sim.kaleidoscope.collision_hits import (
    AggregatedCollisionQueryHits,
    CollisionQueryHitAccumulator,
)
from linkerbot_sim.kaleidoscope.collision_kinematics import (
    CandidateColliderTransformBatch,
)
from linkerbot_sim.kaleidoscope.collision_query import CollisionQueryTarget


class PhysxCandidateOverlapOwner:
    """Own fixed query shadows and execute one complete candidate batch.

    Construction takes ownership of ``shadow_owner`` and ``scene_query``.
    The owner must be created, queried, and closed on the Isaac owner thread.
    """

    def __init__(
        self,
        *,
        robot_catalog: RobotCollisionCatalog,
        robot_label: str,
        shadow_owner: PhysxQueryShadowOwner,
        scene_query: PhysxSceneQueryAdapter,
    ) -> None:
        if not isinstance(robot_catalog, RobotCollisionCatalog):
            raise TypeError("robot_catalog must be RobotCollisionCatalog")
        if robot_label not in robot_catalog.robot_labels:
            raise KeyError(f"unknown robot label: {robot_label!r}")
        if not isinstance(shadow_owner, PhysxQueryShadowOwner):
            raise TypeError("shadow_owner must be PhysxQueryShadowOwner")
        if not isinstance(scene_query, PhysxSceneQueryAdapter):
            raise TypeError("scene_query must be PhysxSceneQueryAdapter")

        self._robot_catalog = robot_catalog
        self._robot_label = robot_label
        self._shadow_owner = shadow_owner
        self._scene_query = scene_query
        self._collision_suffixes: tuple[str, ...] = ()
        self._shadows: tuple[PhysxQueryShadow, ...] = ()
        self._closed = False
        try:
            canonical_shapes = robot_catalog.shapes_for(
                env_id=0,
                robot_label=robot_label,
            )
            self._collision_suffixes = tuple(
                shape.collision_suffix for shape in canonical_shapes
            )
            created: list[PhysxQueryShadow] = []
            for index, shape in enumerate(canonical_shapes):
                created.append(
                    shadow_owner.create_shadow(
                        name=f"candidate_{index:04d}",
                        source_collision_path=shape.collision_path,
                    )
                )
            self._shadows = tuple(created)
        except BaseException as exc:
            self._close_dependencies_after_failure(exc)
            raise

    @property
    def robot_label(self) -> str:
        return self._robot_label

    @property
    def shadows(self) -> tuple[PhysxQueryShadow, ...]:
        return self._shadows

    @property
    def closed(self) -> bool:
        return self._closed

    def execute(
        self,
        *,
        candidates: CandidateColliderTransformBatch,
        target_catalog: SceneCollisionTargetCatalog,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> AggregatedCollisionQueryHits:
        """Execute all ``[env, sample, collider]`` overlaps synchronously."""

        self._require_open()
        self._validate_candidates(candidates)
        if not isinstance(target_catalog, SceneCollisionTargetCatalog):
            raise TypeError("target_catalog must be SceneCollisionTargetCatalog")
        if target_catalog.num_envs != self._robot_catalog.num_envs:
            raise ValueError("target and robot catalog num_envs do not match")
        if target_catalog.robot_labels != self._robot_catalog.robot_labels:
            raise ValueError("target and robot catalog labels do not match")
        accumulator = CollisionQueryHitAccumulator(
            candidates=candidates,
            target_catalog=target_catalog,
            robot_label=self._robot_label,
            targets=targets,
        )

        env_count, sample_count, collider_count = candidates.matrices.shape[:3]
        for env_row in range(env_count):
            for sample_index in range(sample_count):
                for collider_index in range(collider_count):
                    shadow = self._shadows[collider_index]
                    matrix = candidates.matrices[
                        env_row,
                        sample_index,
                        collider_index,
                    ]
                    self._shadow_owner.set_world_transform(
                        shadow,
                        matrix_row_major=tuple(float(value) for value in matrix.flat),
                    )
                    hits = tuple(
                        hit
                        for query_path in shadow.query_prim_paths
                        for hit in self._scene_query.overlap_shape(
                            query_path,
                            stop_after_first=False,
                        )
                    )
                    accumulator.record(
                        env_row=env_row,
                        sample_index=sample_index,
                        collider_index=collider_index,
                        hits=hits,
                    )
        return accumulator.finish()

    def close(self) -> None:
        """Idempotently close query entry point, then remove all shadows."""

        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        try:
            self._scene_query.close()
        except BaseException as exc:
            first_error = exc
        try:
            self._shadow_owner.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(f"query-shadow close also failed: {exc!r}")
        self._shadows = ()
        if first_error is not None:
            raise first_error

    def _validate_candidates(
        self,
        candidates: CandidateColliderTransformBatch,
    ) -> None:
        if not isinstance(candidates, CandidateColliderTransformBatch):
            raise TypeError("candidates must be CandidateColliderTransformBatch")
        if candidates.collision_suffixes != self._collision_suffixes:
            raise ValueError("candidate collider suffixes do not match shadow topology")
        for row, env_id in enumerate(candidates.env_ids):
            expected_paths = tuple(
                shape.collision_path
                for shape in self._robot_catalog.shapes_for(
                    env_id=env_id,
                    robot_label=self._robot_label,
                )
            )
            if candidates.collision_paths[row] != expected_paths:
                raise ValueError(
                    f"candidate collision paths do not match robot catalog env {env_id}"
                )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PhysX candidate overlap owner is closed")

    def _close_dependencies_after_failure(self, original: BaseException) -> None:
        for label, dependency in (
            ("scene-query", self._scene_query),
            ("query-shadow", self._shadow_owner),
        ):
            try:
                dependency.close()
            except BaseException as cleanup_exc:
                original.add_note(
                    f"{label} construction cleanup failed: {cleanup_exc!r}"
                )
        self._closed = True
        self._shadows = ()


__all__ = ["PhysxCandidateOverlapOwner"]
