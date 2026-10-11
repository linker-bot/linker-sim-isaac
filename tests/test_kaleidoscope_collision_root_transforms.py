from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.collision_shapes import (
    CONVEX_DECOMPOSITION,
    PhysxCollisionShapeMetadata,
)
from linkerbot_sim.isaac.scene.pose import PrimWorldTransform
from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    RobotCollisionShape,
)
from linkerbot_sim.kaleidoscope.collision_root_transforms import (
    capture_robot_root_world_transforms,
)


IDENTITY = tuple(np.eye(4, dtype=np.float64).reshape(-1))


def _catalog() -> RobotCollisionCatalog:
    shapes: list[RobotCollisionShape] = []
    for env_id in range(2):
        env_root = f"/World/envs/env_{env_id}"
        imported_root = f"{env_root}/left"
        articulation = f"{imported_root}/Robot"
        rigid_body = f"{articulation}/finger"
        collision = f"{rigid_body}/collision"
        shapes.append(
            RobotCollisionShape(
                env_id=env_id,
                robot_id=0,
                robot_label="left",
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
        robot_labels=("left",),
        shapes=tuple(shapes),
    )


class _Scanner:
    def __init__(self, values: tuple[object, ...] | None = None) -> None:
        self.values = values
        self.requests: list[tuple[object, tuple[str, ...]]] = []

    def __call__(
        self,
        stage: object,
        paths: tuple[str, ...],
    ) -> tuple[object, ...]:
        self.requests.append((stage, paths))
        if self.values is not None:
            return self.values
        result = []
        for path in paths:
            env_id = int(path.split("env_")[1].split("/")[0])
            matrix = np.eye(4, dtype=np.float64)
            matrix[3, :3] = (env_id * 10.0, 2.0, 3.0)
            result.append(
                PrimWorldTransform(
                    prim_path=path,
                    matrix_row_major=tuple(matrix.reshape(-1)),
                )
            )
        return tuple(result)


def test_capture_binds_selected_env_paths_and_complete_world_matrices() -> None:
    stage = object()
    scanner = _Scanner()

    result = capture_robot_root_world_transforms(
        stage,
        robot_catalog=_catalog(),
        robot_label="left",
        env_ids=(1, 0),
        scanner=scanner,
        dtype=np.dtype(np.float32),
    )

    assert scanner.requests == [
        (
            stage,
            (
                "/World/envs/env_1/left",
                "/World/envs/env_0/left",
            ),
        )
    ]
    assert result.env_ids == (1, 0)
    assert result.robot_label == "left"
    assert result.imported_root_paths == scanner.requests[0][1]
    assert result.matrices.dtype == np.float32
    np.testing.assert_allclose(result.matrices[:, 3, :3], ((10, 2, 3), (0, 2, 3)))
    assert result.matrices.flags.writeable is False


def test_capture_rejects_scanner_path_order_or_value_drift() -> None:
    root_0 = PrimWorldTransform(
        prim_path="/World/envs/env_0/left",
        matrix_row_major=IDENTITY,
    )
    root_1 = PrimWorldTransform(
        prim_path="/World/envs/env_1/left",
        matrix_row_major=IDENTITY,
    )
    with pytest.raises(RuntimeError, match="path order"):
        capture_robot_root_world_transforms(
            object(),
            robot_catalog=_catalog(),
            robot_label="left",
            env_ids=(0, 1),
            scanner=_Scanner((root_1, root_0)),
        )
    with pytest.raises(RuntimeError, match="wrong row count"):
        capture_robot_root_world_transforms(
            object(),
            robot_catalog=_catalog(),
            robot_label="left",
            env_ids=(0, 1),
            scanner=_Scanner((root_0,)),
        )
    with pytest.raises(TypeError, match="invalid value"):
        capture_robot_root_world_transforms(
            object(),
            robot_catalog=_catalog(),
            robot_label="left",
            env_ids=(0,),
            scanner=_Scanner((object(),)),
        )


@pytest.mark.parametrize(
    ("env_ids", "message"),
    [
        ((0, 0), "duplicates"),
        ((2,), "outside robot catalog range"),
        ((), "must contain ints"),
    ],
)
def test_capture_rejects_invalid_env_selection(
    env_ids: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        capture_robot_root_world_transforms(
            object(),
            robot_catalog=_catalog(),
            robot_label="left",
            env_ids=env_ids,
            scanner=_Scanner(),
        )
