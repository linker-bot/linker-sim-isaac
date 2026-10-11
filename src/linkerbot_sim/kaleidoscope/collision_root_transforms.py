"""Bind live USD robot-root transforms to collision-query identities."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from linkerbot_sim.isaac.scene.pose import (
    PrimWorldTransform,
    read_prim_world_transforms,
)
from linkerbot_sim.kaleidoscope.collision_catalog import RobotCollisionCatalog
from linkerbot_sim.kaleidoscope.collision_kinematics import (
    RobotRootWorldTransformBatch,
)


RobotRootTransformScanner = Callable[
    [object, Sequence[str]],
    Sequence[PrimWorldTransform],
]


def capture_robot_root_world_transforms(
    stage: object,
    *,
    robot_catalog: RobotCollisionCatalog,
    robot_label: str,
    env_ids: Sequence[int],
    scanner: RobotRootTransformScanner = read_prim_world_transforms,
    dtype: np.dtype = np.dtype(np.float64),
) -> RobotRootWorldTransformBatch:
    """Capture exact imported-root matrices after scene assembly.

    This adapter reads USD and must run on the Isaac owner thread. It does not
    step physics or read articulation state.
    """

    if not isinstance(robot_catalog, RobotCollisionCatalog):
        raise TypeError("robot_catalog must be RobotCollisionCatalog")
    if robot_label not in robot_catalog.robot_labels:
        raise KeyError(f"unknown robot label: {robot_label!r}")
    selected_envs = _selected_env_ids(env_ids, num_envs=robot_catalog.num_envs)
    if not callable(scanner):
        raise TypeError("scanner must be callable")
    resolved_dtype = np.dtype(dtype)
    if resolved_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise TypeError("dtype must be float32 or float64")

    imported_root_paths = tuple(
        robot_catalog.shapes_for(env_id=env_id, robot_label=robot_label)[
            0
        ].imported_root_path
        for env_id in selected_envs
    )
    scanned = tuple(scanner(stage, imported_root_paths))
    if len(scanned) != len(imported_root_paths):
        raise RuntimeError("root-transform scanner returned the wrong row count")

    matrix_rows: list[tuple[float, ...]] = []
    for expected_path, value in zip(imported_root_paths, scanned, strict=True):
        if not isinstance(value, PrimWorldTransform):
            raise TypeError("root-transform scanner returned an invalid value")
        if value.prim_path != expected_path:
            raise RuntimeError(
                "root-transform scanner path order does not match catalog: "
                f"{value.prim_path!r} != {expected_path!r}"
            )
        matrix_rows.append(value.matrix_row_major)

    matrices = np.asarray(matrix_rows, dtype=resolved_dtype).reshape(
        len(selected_envs),
        4,
        4,
    )
    return RobotRootWorldTransformBatch(
        env_ids=selected_envs,
        robot_label=robot_label,
        imported_root_paths=imported_root_paths,
        matrices=np.ascontiguousarray(matrices),
    )


def _selected_env_ids(values: Sequence[int], *, num_envs: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("env_ids must be a sequence of ints")
    result = tuple(values)
    if not result or any(type(value) is not int for value in result):
        raise ValueError("env_ids must contain ints")
    if len(set(result)) != len(result):
        raise ValueError("env_ids must not contain duplicates")
    if any(value < 0 or value >= num_envs for value in result):
        raise ValueError("env_id is outside robot catalog range")
    return result


__all__ = [
    "RobotRootTransformScanner",
    "capture_robot_root_world_transforms",
]
