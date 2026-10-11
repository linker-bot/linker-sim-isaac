"""Single-robot internal reference service for articulated collision queries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    SceneCollisionTargetCatalog,
)
from linkerbot_sim.kaleidoscope.collision_hits import AggregatedCollisionQueryHits
from linkerbot_sim.kaleidoscope.collision_kinematics import (
    FullArticulatedForwardKinematics,
    FullArticulatedKinematicsSpec,
    RobotRootWorldTransformBatch,
    compose_candidate_collider_world_transforms,
)
from linkerbot_sim.kaleidoscope.collision_query import CollisionQueryTarget


@runtime_checkable
class CandidateOverlapPort(Protocol):
    robot_label: str

    def execute(
        self,
        *,
        candidates: object,
        target_catalog: SceneCollisionTargetCatalog,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> AggregatedCollisionQueryHits: ...

    def close(self) -> None: ...


class ReferenceArticulatedCollisionQueryService:
    """Validate a NumPy joint path and execute exact PhysX collider queries."""

    def __init__(
        self,
        *,
        robot_catalog: RobotCollisionCatalog,
        robot_label: str,
        kinematics_spec: FullArticulatedKinematicsSpec,
        robot_root_transforms: RobotRootWorldTransformBatch,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        overlap_port: CandidateOverlapPort,
    ) -> None:
        if not isinstance(robot_catalog, RobotCollisionCatalog):
            raise TypeError("robot_catalog must be RobotCollisionCatalog")
        if robot_label not in robot_catalog.robot_labels:
            raise KeyError(f"unknown robot label: {robot_label!r}")
        if not isinstance(kinematics_spec, FullArticulatedKinematicsSpec):
            raise TypeError("kinematics_spec must be FullArticulatedKinematicsSpec")
        if not isinstance(robot_root_transforms, RobotRootWorldTransformBatch):
            raise TypeError(
                "robot_root_transforms must be RobotRootWorldTransformBatch"
            )
        if robot_root_transforms.robot_label != robot_label:
            raise ValueError("robot root transforms do not match service robot label")
        if not isinstance(overlap_port, CandidateOverlapPort):
            raise TypeError("overlap_port must implement CandidateOverlapPort")
        if overlap_port.robot_label != robot_label:
            raise ValueError("overlap port does not match service robot label")

        expected_envs = set(range(robot_catalog.num_envs))
        if set(robot_root_transforms.env_ids) != expected_envs:
            raise ValueError(
                "robot root transforms must cover every catalog env exactly"
            )
        root_row_by_env: dict[int, int] = {}
        for row, env_id in enumerate(robot_root_transforms.env_ids):
            expected_root = robot_catalog.shapes_for(
                env_id=env_id,
                robot_label=robot_label,
            )[0].imported_root_path
            if robot_root_transforms.imported_root_paths[row] != expected_root:
                raise ValueError(
                    f"robot root transform path does not match catalog env {env_id}"
                )
            root_row_by_env[env_id] = row

        width = len(kinematics_spec.command_joint_names)
        dtype = robot_root_transforms.matrices.dtype
        lower = _joint_limit_vector(
            joint_lower,
            name="joint_lower",
            width=width,
            dtype=dtype,
        )
        upper = _joint_limit_vector(
            joint_upper,
            name="joint_upper",
            width=width,
            dtype=dtype,
        )
        if np.any(lower >= upper):
            raise ValueError("joint_lower must be strictly below joint_upper")

        self._robot_catalog = robot_catalog
        self._robot_label = robot_label
        self._kinematics_spec = kinematics_spec
        self._forward_kinematics = FullArticulatedForwardKinematics(kinematics_spec)
        self._root_transforms = robot_root_transforms
        self._root_row_by_env = root_row_by_env
        self._joint_lower = lower
        self._joint_upper = upper
        self._overlap_port = overlap_port
        self._closed = False

    @property
    def robot_label(self) -> str:
        return self._robot_label

    @property
    def command_joint_names(self) -> tuple[str, ...]:
        return self._kinematics_spec.command_joint_names

    @property
    def closed(self) -> bool:
        return self._closed

    def query(
        self,
        *,
        joint_positions: np.ndarray,
        env_ids: Sequence[int],
        target_catalog: SceneCollisionTargetCatalog,
        targets: tuple[CollisionQueryTarget, ...],
    ) -> AggregatedCollisionQueryHits:
        """Execute ``[selected envs, samples, command joints]`` candidates."""

        self._require_open()
        selected_envs = _selected_env_ids(
            env_ids,
            num_envs=self._robot_catalog.num_envs,
        )
        if not isinstance(target_catalog, SceneCollisionTargetCatalog):
            raise TypeError("target_catalog must be SceneCollisionTargetCatalog")
        if target_catalog.num_envs != self._robot_catalog.num_envs:
            raise ValueError("target and robot catalog num_envs do not match")
        if target_catalog.robot_labels != self._robot_catalog.robot_labels:
            raise ValueError("target and robot catalog labels do not match")
        for env_id in selected_envs:
            target_catalog.targets_for(
                env_id=env_id,
                robot_label=self._robot_label,
                targets=targets,
            )
        positions = _candidate_joint_positions(
            joint_positions,
            env_count=len(selected_envs),
            joint_count=len(self.command_joint_names),
            dtype=self._root_transforms.matrices.dtype,
        )
        violation = np.argwhere(
            (positions < self._joint_lower[None, None, :])
            | (positions > self._joint_upper[None, None, :])
        )
        if violation.size:
            env_row, sample_index, joint_index = (int(value) for value in violation[0])
            raise ValueError(
                "candidate joint position is outside articulation limits: "
                f"env_id={selected_envs[env_row]}, sample={sample_index}, "
                f"joint={self.command_joint_names[joint_index]!r}"
            )

        link_transforms = self._forward_kinematics.compute(positions)
        root_rows = tuple(self._root_row_by_env[env_id] for env_id in selected_envs)
        selected_roots = RobotRootWorldTransformBatch(
            env_ids=selected_envs,
            robot_label=self._robot_label,
            imported_root_paths=tuple(
                self._root_transforms.imported_root_paths[row] for row in root_rows
            ),
            matrices=np.ascontiguousarray(
                self._root_transforms.matrices[np.asarray(root_rows, dtype=np.int64)]
            ),
        )
        candidates = compose_candidate_collider_world_transforms(
            robot_catalog=self._robot_catalog,
            robot_label=self._robot_label,
            kinematics_spec=self._kinematics_spec,
            link_transforms=link_transforms,
            robot_root_transforms=selected_roots,
        )
        return self._overlap_port.execute(
            candidates=candidates,
            target_catalog=target_catalog,
            targets=targets,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._overlap_port.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "reference articulated collision query service is closed"
            )


def _joint_limit_vector(
    value: np.ndarray,
    *,
    name: str,
    width: int,
    dtype: np.dtype,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.shape != (width,):
        raise ValueError(f"{name} must have shape [command joints]")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use robot root matrix dtype")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.ascontiguousarray(value.copy())
    result.setflags(write=False)
    return result


def _candidate_joint_positions(
    value: np.ndarray,
    *,
    env_count: int,
    joint_count: int,
    dtype: np.dtype,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("joint_positions must be a NumPy array")
    if value.ndim != 3:
        raise ValueError(
            "joint_positions must have shape [selected envs, samples, command joints]"
        )
    if value.shape[0] != env_count or value.shape[2] != joint_count:
        raise ValueError(
            "joint_positions must have shape [selected envs, samples, command joints]"
        )
    if value.shape[1] < 1:
        raise ValueError("joint_positions sample axis cannot be empty")
    if value.dtype != dtype:
        raise TypeError("joint_positions must use robot root matrix dtype")
    if not np.all(np.isfinite(value)):
        raise ValueError("joint_positions must contain only finite values")
    return np.ascontiguousarray(value)


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
    "CandidateOverlapPort",
    "ReferenceArticulatedCollisionQueryService",
]
