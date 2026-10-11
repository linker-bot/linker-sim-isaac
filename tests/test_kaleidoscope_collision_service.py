from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.collision_shapes import (
    CONVEX_DECOMPOSITION,
    PhysxCollisionShapeMetadata,
)
from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    RobotCollisionShape,
    SceneCollisionTargetCatalog,
    SceneCollisionTargetShape,
)
from linkerbot_sim.kaleidoscope.collision_hits import (
    AggregatedCollisionQueryHits,
)
from linkerbot_sim.kaleidoscope.collision_kinematics import (
    CandidateColliderTransformBatch,
    RobotRootWorldTransformBatch,
    build_full_articulated_kinematics_spec,
)
from linkerbot_sim.kaleidoscope.collision_query import (
    CollisionHitCategory,
    CollisionQueryTarget,
)
from linkerbot_sim.kaleidoscope.collision_service import (
    ReferenceArticulatedCollisionQueryService,
)


IDENTITY = tuple(np.eye(4, dtype=np.float64).reshape(-1))


class _OverlapPort:
    def __init__(self, robot_label: str = "left") -> None:
        self.robot_label = robot_label
        self.calls: list[dict[str, object]] = []
        self.close_count = 0

    def execute(self, **kwargs: object) -> AggregatedCollisionQueryHits:
        self.calls.append(kwargs)
        candidates = kwargs["candidates"]
        assert isinstance(candidates, CandidateColliderTransformBatch)
        env_count, sample_count = candidates.matrices.shape[:2]
        return AggregatedCollisionQueryHits(
            collision_mask=np.zeros((env_count, sample_count), dtype=np.bool_),
            first_hit_sample=np.full((env_count,), -1, dtype=np.int64),
            first_hit_category=np.full(
                (env_count,),
                CollisionHitCategory.NONE,
                dtype=np.int64,
            ),
        )

    def close(self) -> None:
        self.close_count += 1


def _kinematics_spec(tmp_path: Path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
  <link name="world"/>
  <link name="finger"/>
  <joint name="finger_joint" type="revolute">
    <parent link="world"/><child link="finger"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    profile = type("Profile", (), {"asset_type": "urdf", "asset_path": str(urdf)})()
    return build_full_articulated_kinematics_spec(
        robot_profile=profile,
        command_joint_names=("finger_joint",),
        required_link_names=("finger",),
    )


def _robot_catalog() -> RobotCollisionCatalog:
    shapes = []
    for env_id in range(2):
        for robot_id, label in enumerate(("left", "right")):
            env_root = f"/World/envs/env_{env_id}"
            imported_root = f"{env_root}/{label}"
            articulation = f"{imported_root}/Robot"
            rigid_body = f"{articulation}/finger"
            collision = f"{rigid_body}/collision"
            shapes.append(
                RobotCollisionShape(
                    env_id=env_id,
                    robot_id=robot_id,
                    robot_label=label,
                    env_root_path=env_root,
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
        num_envs=2,
        robot_labels=("left", "right"),
        shapes=tuple(shapes),
    )


def _target(
    env_id: int,
    category: CollisionQueryTarget,
    source_name: str,
    *,
    owner_robot_label: str | None = None,
) -> SceneCollisionTargetShape:
    root = f"/World/envs/env_{env_id}/{source_name}"
    collision = (
        f"{root}/collision"
        if owner_robot_label is None
        else f"{root}/Robot/finger/collision"
    )
    return SceneCollisionTargetShape(
        env_id=env_id,
        category=category,
        source_name=source_name,
        owner_robot_label=owner_robot_label,
        root_path=root,
        collision_path=collision,
        rigid_body_path=(None if owner_robot_label is None else f"{root}/Robot/finger"),
    )


def _target_catalog() -> SceneCollisionTargetCatalog:
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


def _roots(*, env_ids: tuple[int, ...] = (0, 1)) -> RobotRootWorldTransformBatch:
    matrices = np.broadcast_to(np.eye(4), (len(env_ids), 4, 4)).copy()
    for row, env_id in enumerate(env_ids):
        matrices[row, 3, 0] = 10.0 * (env_id + 1)
    return RobotRootWorldTransformBatch(
        env_ids=env_ids,
        robot_label="left",
        imported_root_paths=tuple(
            f"/World/envs/env_{env_id}/left" for env_id in env_ids
        ),
        matrices=matrices,
    )


def _service(
    tmp_path: Path,
    *,
    overlap: _OverlapPort | None = None,
    roots: RobotRootWorldTransformBatch | None = None,
) -> tuple[ReferenceArticulatedCollisionQueryService, _OverlapPort]:
    resolved_overlap = overlap or _OverlapPort()
    service = ReferenceArticulatedCollisionQueryService(
        robot_catalog=_robot_catalog(),
        robot_label="left",
        kinematics_spec=_kinematics_spec(tmp_path),
        robot_root_transforms=roots or _roots(),
        joint_lower=np.asarray([-1.0], dtype=np.float64),
        joint_upper=np.asarray([1.0], dtype=np.float64),
        overlap_port=resolved_overlap,
    )
    return service, resolved_overlap


def test_reference_service_runs_validation_fk_composition_and_overlap(
    tmp_path: Path,
) -> None:
    service, overlap = _service(tmp_path)
    positions = np.asarray(
        [
            [[0.0], [1.0]],
            [[-1.0], [0.5]],
        ],
        dtype=np.float64,
    )

    result = service.query(
        joint_positions=positions,
        env_ids=(1, 0),
        target_catalog=_target_catalog(),
        targets=(CollisionQueryTarget.TABLE, CollisionQueryTarget.OTHER_ROBOT),
    )

    assert result.first_hit_sample.tolist() == [-1, -1]
    assert len(overlap.calls) == 1
    candidates = overlap.calls[0]["candidates"]
    assert isinstance(candidates, CandidateColliderTransformBatch)
    assert candidates.env_ids == (1, 0)
    assert candidates.collision_paths == (
        ("/World/envs/env_1/left/Robot/finger/collision",),
        ("/World/envs/env_0/left/Robot/finger/collision",),
    )
    np.testing.assert_allclose(candidates.matrices[0, 0, 0, 3, :3], (21, 0, 0))
    np.testing.assert_allclose(candidates.matrices[1, 0, 0, 3, :3], (11, 0, 0))


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (np.asarray([[[1.01]]], dtype=np.float64), "outside articulation limits"),
        (np.asarray([[[0.0]]], dtype=np.float32), "root matrix dtype"),
        (np.zeros((1, 0, 1), dtype=np.float64), "sample axis cannot be empty"),
        (np.zeros((2, 1, 1), dtype=np.float64), "must have shape"),
    ],
)
def test_reference_service_rejects_invalid_path_before_overlap(
    tmp_path: Path,
    positions: np.ndarray,
    message: str,
) -> None:
    service, overlap = _service(tmp_path)

    with pytest.raises((TypeError, ValueError), match=message):
        service.query(
            joint_positions=positions,
            env_ids=(0,),
            target_catalog=_target_catalog(),
            targets=(CollisionQueryTarget.TABLE,),
        )

    assert overlap.calls == []
    service.close()


def test_reference_service_requires_complete_root_identity_and_matching_owner(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cover every catalog env"):
        _service(tmp_path, roots=_roots(env_ids=(0,)))
    with pytest.raises(ValueError, match="overlap port does not match"):
        _service(tmp_path, overlap=_OverlapPort("right"))


def test_reference_service_close_is_idempotent_and_blocks_queries(
    tmp_path: Path,
) -> None:
    service, overlap = _service(tmp_path)

    service.close()
    service.close()

    assert service.closed is True
    assert overlap.close_count == 1
    with pytest.raises(RuntimeError, match="service is closed"):
        service.query(
            joint_positions=np.zeros((1, 1, 1), dtype=np.float64),
            env_ids=(0,),
            target_catalog=_target_catalog(),
            targets=(CollisionQueryTarget.TABLE,),
        )
