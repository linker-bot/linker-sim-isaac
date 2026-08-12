"""Metadata-bound PhysX CPU task-space observations for hybrid control.

Isaac's array rows are never inferred from a DOF offset. The adapter binds
link, incoming-joint and DOF names once, then validates every sampled shape.
All returned spatial values use ``[x, y, z, rx, ry, rz]`` ordering and are
expressed in the world control frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from linkerbot_sim.controllers.hybrid_force_position import TaskSpaceObservation
from linkerbot_sim.robots.tcp_binding import PhysicalTcpBinding
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


class PhysxTaskSpaceError(RuntimeError):
    code = "hybrid_task_space_unavailable"


class PhysxTaskSpaceMetadataError(PhysxTaskSpaceError):
    pass


class PhysxTaskSpaceSensorError(PhysxTaskSpaceError):
    code = "hybrid_sensor_invalid"


@dataclass(frozen=True, slots=True)
class PhysxTaskSpaceBinding:
    parent_body_name: str
    incoming_joint_name: str
    body_state_row: int
    jacobian_body_row: int
    reaction_row: int
    arm_column_indices: tuple[int, ...]
    arm_joint_names: tuple[str, ...]
    joint_child_position: tuple[float, float, float]
    joint_child_orientation_wxyz: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "parent_body_name": self.parent_body_name,
            "incoming_joint_name": self.incoming_joint_name,
            "body_state_row": self.body_state_row,
            "jacobian_body_row": self.jacobian_body_row,
            "reaction_row": self.reaction_row,
            "arm_column_indices": list(self.arm_column_indices),
            "arm_joint_names": list(self.arm_joint_names),
        }


@dataclass(frozen=True, slots=True)
class _JointFrame:
    name: str
    child_body_path: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


class PhysxTaskSpacePort:
    """Read one articulation's physical TCP state without leaking Isaac tensors."""

    reaction_sign_to_environment_on_tool = 1.0

    def __init__(
        self,
        articulation: object,
        stage: object,
        arm_joint_names: tuple[str, ...] | list[str],
        tcp_binding: PhysicalTcpBinding,
    ) -> None:
        if not isinstance(tcp_binding, PhysicalTcpBinding):
            raise TypeError("tcp_binding must be PhysicalTcpBinding")
        names = tuple(str(name) for name in arm_joint_names)
        if len(names) < 6 or len(set(names)) != len(names):
            raise PhysxTaskSpaceMetadataError(
                "arm_joint_names must contain at least six unique names"
            )
        self.articulation = articulation
        self.stage = stage
        self.tcp_binding = tcp_binding
        self._owners = _metadata_owners(articulation)
        self._require_cpu_fixed_base(articulation, self._owners)

        body_positions, _ = self._read_body_poses()
        body_names = _required_names(
            self._owners,
            ("task_space_body_names", "body_names", "link_names"),
            label="articulation body names",
        )
        if len(body_names) != body_positions.shape[0]:
            raise PhysxTaskSpaceMetadataError(
                "body-name metadata does not match link-state rows"
            )
        body_state_row = _unique_name_index(
            body_names,
            tcp_binding.parent_frame_name,
            label="TCP parent body",
        )

        dof_names = _required_names(
            self._owners,
            ("task_space_dof_names", "dof_names"),
            label="articulation DOF names",
        )
        arm_columns = tuple(
            _unique_name_index(dof_names, name, label="arm DOF") for name in names
        )

        joint_frame = _incoming_joint_frame(
            stage,
            articulation=articulation,
            child_body_path=tcp_binding.parent_body_path,
        )
        jacobian = self._read_jacobians()
        jacobian_names = _row_names(
            self._owners,
            explicit_attributes=("jacobian_body_names",),
            row_count=jacobian.shape[0],
            fallback_names=body_names,
            stage=stage,
            articulation=articulation,
            label="Jacobian body rows",
        )
        jacobian_row = _unique_name_index(
            jacobian_names,
            tcp_binding.parent_frame_name,
            label="TCP Jacobian body",
        )
        if jacobian.shape[2] != len(dof_names):
            raise PhysxTaskSpaceMetadataError(
                "Jacobian columns do not match articulation DOF-name metadata"
            )

        reaction = self._read_reaction_wrenches()
        reaction_row = _reaction_row(
            self._owners,
            row_count=reaction.shape[0],
            body_names=body_names,
            parent_body_name=tcp_binding.parent_frame_name,
            incoming_joint_name=joint_frame.name,
        )
        self.binding = PhysxTaskSpaceBinding(
            parent_body_name=tcp_binding.parent_frame_name,
            incoming_joint_name=joint_frame.name,
            body_state_row=body_state_row,
            jacobian_body_row=jacobian_row,
            reaction_row=reaction_row,
            arm_column_indices=arm_columns,
            arm_joint_names=names,
            joint_child_position=joint_frame.position,
            joint_child_orientation_wxyz=joint_frame.orientation_wxyz,
        )
        self._body_names = body_names
        self._dof_names = dof_names
        self._expected_jacobian_shape = jacobian.shape
        self._expected_reaction_shape = reaction.shape
        self._sequence = 0

    def observe(self) -> TaskSpaceObservation:
        """Read and normalize one finite, CPU-resident task-space sample."""

        body_positions, body_orientations = self._read_body_poses()
        body_twists = self._read_body_twists()
        jacobians = self._read_jacobians()
        reactions = self._read_reaction_wrenches()
        if jacobians.shape != self._expected_jacobian_shape:
            raise PhysxTaskSpaceSensorError(
                "PhysX Jacobian shape changed after task-space binding"
            )
        if reactions.shape != self._expected_reaction_shape:
            raise PhysxTaskSpaceSensorError(
                "PhysX reaction-wrench shape changed after task-space binding"
            )
        q = _finite_vector(
            _call_getter(self.articulation, "get_joint_positions"),
            len(self._dof_names),
            label="joint positions",
        )
        qd = _finite_vector(
            _call_getter(self.articulation, "get_joint_velocities"),
            len(self._dof_names),
            label="joint velocities",
        )

        row = self.binding.body_state_row
        parent_position = body_positions[row]
        parent_orientation = body_orientations[row]
        parent_twist = body_twists[row]
        parent_rotation = quaternion_matrix_wxyz(parent_orientation)
        offset_rotation = rpy_matrix(self.tcp_binding.offset_rpy)
        offset_world = parent_rotation @ np.asarray(
            self.tcp_binding.offset_xyz, dtype=float
        )
        tcp_position = parent_position + offset_world
        tcp_orientation = quaternion_multiply_wxyz(
            parent_orientation,
            matrix_quaternion_wxyz(offset_rotation),
        )
        tcp_twist = parent_twist.copy()
        tcp_twist[:3] = parent_twist[:3] + np.cross(parent_twist[3:], offset_world)

        jacobian = jacobians[self.binding.jacobian_body_row]
        tcp_jacobian = jacobian.copy()
        tcp_jacobian[:3] = jacobian[:3] - skew(offset_world) @ jacobian[3:]
        tcp_jacobian = tcp_jacobian[:, self.binding.arm_column_indices]

        joint_position_world = parent_position + parent_rotation @ np.asarray(
            self.binding.joint_child_position,
            dtype=float,
        )
        joint_rotation_world = parent_rotation @ quaternion_matrix_wxyz(
            self.binding.joint_child_orientation_wxyz
        )
        reaction_local = reactions[self.binding.reaction_row]
        force_world = (
            self.reaction_sign_to_environment_on_tool
            * joint_rotation_world
            @ reaction_local[:3]
        )
        moment_joint_world = (
            self.reaction_sign_to_environment_on_tool
            * joint_rotation_world
            @ reaction_local[3:]
        )
        moment_tcp_world = moment_joint_world - np.cross(
            tcp_position - joint_position_world,
            force_world,
        )

        sequence = self._sequence
        self._sequence += 1
        arm_columns = np.asarray(self.binding.arm_column_indices, dtype=int)
        return TaskSpaceObservation(
            position=tcp_position,
            orientation_wxyz=tcp_orientation,
            twist=tcp_twist,
            jacobian=tcp_jacobian,
            joint_positions=q[arm_columns],
            joint_velocities=qd[arm_columns],
            external_wrench_environment_on_tool=np.concatenate(
                [force_world, moment_tcp_world]
            ),
            sequence=sequence,
        )

    def status(self) -> dict[str, object]:
        return {
            "available": True,
            "backend": "physx",
            "execution": "cpu",
            "tcp_frame_name": self.tcp_binding.tcp_frame_name,
            "reference_frames": ["world"],
            "wrench_sign": "environment_on_tool",
            "binding": self.binding.as_dict(),
        }

    def _read_body_poses(self) -> tuple[np.ndarray, np.ndarray]:
        for owner in self._owners:
            for name in ("get_link_world_poses", "get_body_world_poses"):
                callback = getattr(owner, name, None)
                if callable(callback):
                    positions, orientations = callback()
                    return _pose_arrays(
                        positions, orientations, quaternion_order="wxyz"
                    )

        physics_view = _physics_view(self.articulation)
        callback = getattr(physics_view, "get_link_transforms", None)
        if not callable(callback):
            raise PhysxTaskSpaceMetadataError(
                "PhysX articulation does not expose per-link world transforms"
            )
        transforms = _strip_batch(_array(callback()), dimensions=2)
        if transforms.ndim != 2 or transforms.shape[1] != 7:
            raise PhysxTaskSpaceSensorError(
                "PhysX link transforms must have shape (body_count, 7)"
            )
        # PhysX tensor transforms store quaternion components as xyzw.
        xyzw = transforms[:, 3:]
        wxyz = np.column_stack((xyzw[:, 3], xyzw[:, :3]))
        return _pose_arrays(transforms[:, :3], wxyz, quaternion_order="wxyz")

    def _read_body_twists(self) -> np.ndarray:
        for owner in self._owners:
            for name in ("get_link_velocities", "get_body_velocities"):
                callback = getattr(owner, name, None)
                if callable(callback):
                    return _matrix_rows(callback(), 6, label="link velocities")
        physics_view = _physics_view(self.articulation)
        callback = getattr(physics_view, "get_link_velocities", None)
        if not callable(callback):
            raise PhysxTaskSpaceMetadataError(
                "PhysX articulation does not expose per-link velocities"
            )
        return _matrix_rows(callback(), 6, label="link velocities")

    def _read_jacobians(self) -> np.ndarray:
        value = _call_first(self._owners, ("get_jacobians",))
        result = _strip_batch(_array(value), dimensions=3)
        if result.ndim != 3 or result.shape[1] != 6:
            raise PhysxTaskSpaceSensorError(
                "PhysX Jacobians must have shape (body_count, 6, dof_count)"
            )
        _require_finite(result, label="PhysX Jacobians")
        return result

    def _read_reaction_wrenches(self) -> np.ndarray:
        value = _call_first(
            self._owners,
            ("get_measured_joint_forces", "get_link_incoming_joint_force"),
        )
        return _matrix_rows(value, 6, label="measured reaction wrenches")

    @staticmethod
    def _require_cpu_fixed_base(
        articulation: object, owners: tuple[object, ...]
    ) -> None:
        device = str(getattr(articulation, "device", "cpu")).casefold()
        if "cuda" in device or "gpu" in device:
            raise PhysxTaskSpaceMetadataError(
                "hybrid task-space control requires a CPU articulation"
            )
        if _fixed_base_value(owners) is not True:
            raise PhysxTaskSpaceMetadataError(
                "hybrid task-space control requires a fixed-base articulation"
            )


def skew(vector: object) -> np.ndarray:
    x, y, z = _finite_vector(vector, 3, label="skew vector")
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rpy_matrix(rpy: object) -> np.ndarray:
    roll, pitch, yaw = _finite_vector(rpy, 3, label="rpy")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def quaternion_matrix_wxyz(quaternion: object) -> np.ndarray:
    w, x, y, z = _normalized_quaternion(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_quaternion_wxyz(matrix: object) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("rotation matrix must be finite with shape (3, 3)")
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = np.asarray(
            [
                0.25 * scale,
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(value)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
            result = np.asarray(
                [
                    (value[2, 1] - value[1, 2]) / scale,
                    0.25 * scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
            result = np.asarray(
                [
                    (value[0, 2] - value[2, 0]) / scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    0.25 * scale,
                    (value[1, 2] + value[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
            result = np.asarray(
                [
                    (value[1, 0] - value[0, 1]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                    (value[1, 2] + value[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return _normalized_quaternion(result)


def quaternion_multiply_wxyz(left: object, right: object) -> np.ndarray:
    lw, lx, ly, lz = _normalized_quaternion(left)
    rw, rx, ry, rz = _normalized_quaternion(right)
    return _normalized_quaternion(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _metadata_owners(articulation: object) -> tuple[object, ...]:
    owners: list[object] = [articulation]
    view = getattr(articulation, "_articulation_view", None)
    if view is not None:
        owners.append(view)
        metadata = getattr(view, "_metadata", None)
        if metadata is not None:
            owners.append(metadata)
    return tuple(owners)


def _physics_view(articulation: object) -> object:
    """Single compatibility boundary for Isaac's low-level PhysX tensor view."""

    view = getattr(articulation, "_articulation_view", None)
    physics_view = None if view is None else getattr(view, "_physics_view", None)
    if physics_view is None:
        physics_view = getattr(articulation, "_physics_view", None)
    if physics_view is None:
        raise PhysxTaskSpaceMetadataError(
            "PhysX articulation has no initialized tensor view"
        )
    return physics_view


def _required_names(
    owners: tuple[object, ...],
    attributes: tuple[str, ...],
    *,
    label: str,
) -> tuple[str, ...]:
    for owner in owners:
        for attribute in attributes:
            value = getattr(owner, attribute, None)
            if value is None:
                continue
            names = tuple(str(item) for item in value)
            if (
                not names
                or any(not name for name in names)
                or len(set(names)) != len(names)
            ):
                raise PhysxTaskSpaceMetadataError(
                    f"{label} must be non-empty and unique"
                )
            return names
    raise PhysxTaskSpaceMetadataError(f"missing {label}")


def _optional_names(
    owners: tuple[object, ...], attributes: tuple[str, ...]
) -> tuple[str, ...] | None:
    for owner in owners:
        for attribute in attributes:
            value = getattr(owner, attribute, None)
            if value is not None:
                names = tuple(str(item) for item in value)
                if len(set(names)) != len(names):
                    raise PhysxTaskSpaceMetadataError(
                        f"{attribute} contains duplicate names"
                    )
                return names
    return None


def _fixed_base_value(owners: tuple[object, ...]) -> bool | None:
    for owner in owners:
        for attribute in ("fixed_base", "is_fixed_base"):
            value = getattr(owner, attribute, None)
            if callable(value):
                value = value()
            if value is not None:
                return bool(value)
    return None


def _indexed_link_names(
    owners: tuple[object, ...], body_names: tuple[str, ...]
) -> tuple[str, ...] | None:
    for owner in owners:
        value = getattr(owner, "link_indices", None)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise PhysxTaskSpaceMetadataError(
                "articulation link_indices metadata must be a mapping"
            )
        indices: dict[str, int] = {}
        for raw_name, raw_index in value.items():
            name = str(raw_name)
            if isinstance(raw_index, bool):
                raise PhysxTaskSpaceMetadataError(
                    "articulation link_indices metadata contains a boolean index"
                )
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise PhysxTaskSpaceMetadataError(
                    "articulation link_indices metadata contains a non-integer index"
                ) from exc
            if index != raw_index:
                raise PhysxTaskSpaceMetadataError(
                    "articulation link_indices metadata contains a non-integer index"
                )
            indices[name] = index
        if set(indices) != set(body_names):
            raise PhysxTaskSpaceMetadataError(
                "articulation link_indices names do not match body names"
            )
        if set(indices.values()) != set(range(len(body_names))):
            raise PhysxTaskSpaceMetadataError(
                "articulation link_indices must be a contiguous zero-based mapping"
            )
        ordered = tuple(
            name for name, _index in sorted(indices.items(), key=lambda item: item[1])
        )
        if ordered != body_names:
            raise PhysxTaskSpaceMetadataError(
                "body-name rows do not follow articulation link_indices"
            )
        return ordered
    return None


def _row_names(
    owners: tuple[object, ...],
    *,
    explicit_attributes: tuple[str, ...],
    row_count: int,
    fallback_names: tuple[str, ...],
    stage: object,
    articulation: object,
    label: str,
) -> tuple[str, ...]:
    explicit = _optional_names(owners, explicit_attributes)
    if explicit is not None:
        if len(explicit) != row_count:
            raise PhysxTaskSpaceMetadataError(
                f"{label} metadata length does not match array rows"
            )
        return explicit
    if len(fallback_names) == row_count:
        return fallback_names
    # PhysX defines fixed-base Jacobian rows as link-index order with link 0
    # omitted. Bind that documented rule through shared_metatype.link_indices.
    indexed_names = _indexed_link_names(owners, fallback_names)
    if len(fallback_names) - 1 == row_count and indexed_names is not None:
        if _fixed_base_value(owners) is not True:
            raise PhysxTaskSpaceMetadataError(
                f"{label} omit link 0 but articulation metadata is not fixed-base"
            )
        return indexed_names[1:]
    # Minimal test doubles may not expose PhysX metatype. Resolve their unique
    # fixed root from USD relationships while retaining the same named mapping.
    roots = _bodies_without_incoming_joint(
        stage,
        articulation=articulation,
        body_names=fallback_names,
    )
    if len(fallback_names) - 1 == row_count and len(roots) == 1:
        return tuple(name for name in fallback_names if name != roots[0])
    raise PhysxTaskSpaceMetadataError(
        f"cannot bind names to {label}; explicit metadata is required "
        f"(body_names={len(fallback_names)}, rows={row_count}, roots={roots!r})"
    )


def _reaction_row(
    owners: tuple[object, ...],
    *,
    row_count: int,
    body_names: tuple[str, ...],
    parent_body_name: str,
    incoming_joint_name: str,
) -> int:
    joint_names = _optional_names(owners, ("reaction_joint_names",))
    if joint_names is not None:
        if len(joint_names) != row_count:
            raise PhysxTaskSpaceMetadataError(
                "reaction_joint_names does not match reaction rows"
            )
        return _unique_name_index(
            joint_names, incoming_joint_name, label="reaction incoming joint"
        )
    reaction_bodies = _optional_names(owners, ("reaction_body_names",))
    if reaction_bodies is None and len(body_names) == row_count:
        reaction_bodies = body_names
    if reaction_bodies is None or len(reaction_bodies) != row_count:
        raise PhysxTaskSpaceMetadataError(
            "reaction rows require explicit joint/body-name metadata"
        )
    return _unique_name_index(
        reaction_bodies, parent_body_name, label="reaction child body"
    )


def _incoming_joint_frame(
    stage: object,
    *,
    articulation: object,
    child_body_path: str,
) -> _JointFrame:
    from pxr import Usd, UsdPhysics

    root_path = str(getattr(articulation, "prim_path", ""))
    root = stage.GetPrimAtPath(root_path) if root_path else None
    if root is None or not root.IsValid():
        # Mirror imports may expose a nested articulation root; walking from the
        # stage pseudo-root is still name/relationship based and remains strict.
        root = stage.GetPseudoRoot()
    matches = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        targets = tuple(
            str(path) for path in prim.GetRelationship("physics:body1").GetTargets()
        )
        if targets == (child_body_path,):
            matches.append(prim)
    if len(matches) != 1:
        raise PhysxTaskSpaceMetadataError(
            f"TCP parent body must have one incoming USD joint; found {len(matches)}"
        )
    prim = matches[0]
    joint = UsdPhysics.Joint(prim)
    position = _vec3_attr(joint.GetLocalPos1Attr().Get(), default=(0.0, 0.0, 0.0))
    orientation = _quat_attr(joint.GetLocalRot1Attr().Get())
    return _JointFrame(
        name=str(prim.GetName()),
        child_body_path=child_body_path,
        position=position,
        orientation_wxyz=orientation,
    )


def _bodies_without_incoming_joint(
    stage: object,
    *,
    articulation: object,
    body_names: tuple[str, ...],
) -> tuple[str, ...]:
    from pxr import Usd, UsdPhysics

    root_path = str(getattr(articulation, "prim_path", ""))
    root = stage.GetPrimAtPath(root_path) if root_path else None
    if root is None or not root.IsValid():
        root = stage.GetPseudoRoot()
    child_names = set()
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        for target in prim.GetRelationship("physics:body1").GetTargets():
            child_names.add(str(target).rsplit("/", 1)[-1])
    return tuple(name for name in body_names if name not in child_names)


def _vec3_attr(
    value: Any, *, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if value is None:
        return default
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise PhysxTaskSpaceMetadataError("USD joint local position is invalid")
    return result


def _quat_attr(value: Any) -> tuple[float, float, float, float]:
    if value is None:
        return (1.0, 0.0, 0.0, 0.0)
    result = _normalized_quaternion(
        (float(value.GetReal()), *(float(item) for item in value.GetImaginary()))
    )
    return tuple(float(item) for item in result)


def _call_first(owners: tuple[object, ...], names: tuple[str, ...]) -> object:
    for owner in owners:
        for name in names:
            callback = getattr(owner, name, None)
            if callable(callback):
                return callback()
    raise PhysxTaskSpaceMetadataError(
        f"PhysX articulation is missing getter {list(names)}"
    )


def _call_getter(owner: object, name: str) -> object:
    callback = getattr(owner, name, None)
    if not callable(callback):
        raise PhysxTaskSpaceMetadataError(f"articulation is missing {name}()")
    return callback()


def _array(value: object) -> np.ndarray:
    try:
        return np.asarray(tensor_like_to_numpy(value, dtype=float), dtype=float)
    except (TypeError, ValueError) as exc:
        raise PhysxTaskSpaceSensorError(
            f"PhysX task-space getter returned a non-CPU array: {exc}"
        ) from exc


def _strip_batch(value: np.ndarray, *, dimensions: int) -> np.ndarray:
    result = value
    while result.ndim > dimensions and result.shape[0] == 1:
        result = result[0]
    return result


def _finite_vector(value: object, length: int, *, label: str) -> np.ndarray:
    result = _strip_batch(_array(value), dimensions=1).reshape(-1)
    if result.shape != (length,):
        raise PhysxTaskSpaceSensorError(
            f"{label} must have shape ({length},), got {result.shape}"
        )
    _require_finite(result, label=label)
    return result


def _matrix_rows(value: object, width: int, *, label: str) -> np.ndarray:
    result = _strip_batch(_array(value), dimensions=2)
    if result.ndim != 2 or result.shape[1] != width:
        raise PhysxTaskSpaceSensorError(f"{label} must have shape (row_count, {width})")
    _require_finite(result, label=label)
    return result


def _pose_arrays(
    positions: object,
    orientations: object,
    *,
    quaternion_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    position_array = _matrix_rows(positions, 3, label="link positions")
    orientation_array = _matrix_rows(orientations, 4, label="link orientations")
    if quaternion_order == "xyzw":
        orientation_array = np.column_stack(
            (orientation_array[:, 3], orientation_array[:, :3])
        )
    normalized = np.vstack([_normalized_quaternion(item) for item in orientation_array])
    if normalized.shape[0] != position_array.shape[0]:
        raise PhysxTaskSpaceSensorError(
            "link position/orientation row counts do not match"
        )
    return position_array, normalized


def _normalized_quaternion(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise ValueError("quaternion cannot be zero")
    return result / norm


def _require_finite(value: np.ndarray, *, label: str) -> None:
    if not np.all(np.isfinite(value)):
        raise PhysxTaskSpaceSensorError(f"{label} contains NaN/Infinity")


def _unique_name_index(names: tuple[str, ...], value: str, *, label: str) -> int:
    matches = tuple(index for index, name in enumerate(names) if name == value)
    if len(matches) != 1:
        raise PhysxTaskSpaceMetadataError(
            f"{label} {value!r} must match exactly one metadata name"
        )
    return matches[0]


__all__ = [
    "PhysxTaskSpaceBinding",
    "PhysxTaskSpaceError",
    "PhysxTaskSpaceMetadataError",
    "PhysxTaskSpacePort",
    "PhysxTaskSpaceSensorError",
    "matrix_quaternion_wxyz",
    "quaternion_matrix_wxyz",
    "quaternion_multiply_wxyz",
    "rpy_matrix",
    "skew",
]
