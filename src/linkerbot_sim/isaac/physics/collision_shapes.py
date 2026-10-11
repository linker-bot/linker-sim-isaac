"""Read-only discovery of articulated PhysX collision-shape metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol


CONVEX_DECOMPOSITION = "convexDecomposition"


@dataclass(frozen=True, slots=True)
class PhysxCollisionShapeMetadata:
    """Stable USD identity and link-local transform for one active collider.

    ``shape_to_rigid_body_matrix_row_major`` is the complete Gf affine
    transform from shape-local coordinates into the owning rigid-body frame.
    Keeping all 16 values preserves authored scale as well as rigid pose.
    """

    collision_path: str
    rigid_body_path: str
    geometry_type: str
    mesh_approximation: str | None
    shape_to_rigid_body_matrix_row_major: tuple[float, ...]

    def __post_init__(self) -> None:
        _absolute_prim_path(self.collision_path, label="collision_path")
        _absolute_prim_path(self.rigid_body_path, label="rigid_body_path")
        if not isinstance(self.geometry_type, str) or not self.geometry_type:
            raise ValueError("geometry_type must be a non-empty string")
        if self.mesh_approximation is not None and (
            not isinstance(self.mesh_approximation, str) or not self.mesh_approximation
        ):
            raise ValueError("mesh_approximation must be None or a non-empty string")
        matrix = self.shape_to_rigid_body_matrix_row_major
        if not isinstance(matrix, tuple) or len(matrix) != 16:
            raise ValueError(
                "shape_to_rigid_body_matrix_row_major must be a tuple of 16 values"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in matrix
        ):
            raise ValueError(
                "shape_to_rigid_body_matrix_row_major must contain finite numbers"
            )
        if any(
            not math.isclose(float(matrix[index]), 0.0, abs_tol=1e-12)
            for index in (3, 7, 11)
        ) or not math.isclose(float(matrix[15]), 1.0, abs_tol=1e-12):
            raise ValueError(
                "shape_to_rigid_body_matrix_row_major must be an affine Gf matrix"
            )
        if self.geometry_type == "Mesh" and self.mesh_approximation is None:
            raise ValueError("Mesh collision metadata requires mesh_approximation")
        if self.geometry_type != "Mesh" and self.mesh_approximation is not None:
            raise ValueError(
                "non-Mesh collision metadata cannot define mesh_approximation"
            )


@dataclass(frozen=True, slots=True)
class PhysxSceneCollisionIdentity:
    """Exact USD identity for one active scene-object collider.

    Static collision geometry is valid without ``RigidBodyAPI``.  Its
    ``rigid_body_path`` is therefore ``None`` while ``collision_path`` remains
    the authoritative identity used by overlap-hit filtering.
    """

    collision_path: str
    rigid_body_path: str | None

    def __post_init__(self) -> None:
        _absolute_prim_path(self.collision_path, label="collision_path")
        if self.rigid_body_path is not None:
            _absolute_prim_path(self.rigid_body_path, label="rigid_body_path")


class _CollisionShapeInspector(Protocol):
    def inspect(
        self,
        articulation_root_path: str,
    ) -> Sequence[PhysxCollisionShapeMetadata]: ...


class _SceneCollisionIdentityInspector(Protocol):
    def inspect(
        self,
        object_root_path: str,
    ) -> Sequence[PhysxSceneCollisionIdentity]: ...


def discover_articulated_collision_shapes(
    stage: object,
    articulation_root_path: str,
    *,
    required_mesh_approximation: str = CONVEX_DECOMPOSITION,
    inspector: _CollisionShapeInspector | None = None,
) -> tuple[PhysxCollisionShapeMetadata, ...]:
    """Discover active articulated colliders and fail closed on ambiguity."""

    root_path = _absolute_prim_path(
        articulation_root_path,
        label="articulation_root_path",
    )
    if (
        not isinstance(required_mesh_approximation, str)
        or not required_mesh_approximation
    ):
        raise ValueError("required_mesh_approximation must be a non-empty string")
    resolved_inspector = inspector or _UsdCollisionShapeInspector(stage)
    shapes = tuple(resolved_inspector.inspect(root_path))
    if not shapes:
        raise RuntimeError(f"articulation has no active collision GPrims: {root_path}")

    by_path: dict[str, PhysxCollisionShapeMetadata] = {}
    for shape in shapes:
        if not isinstance(shape, PhysxCollisionShapeMetadata):
            raise TypeError(
                "collision-shape inspector must return "
                "PhysxCollisionShapeMetadata values"
            )
        if not _is_descendant_or_self(shape.collision_path, root_path):
            raise RuntimeError(
                f"collision prim is outside articulation root: {shape.collision_path}"
            )
        if not _is_descendant_or_self(shape.rigid_body_path, root_path):
            raise RuntimeError(
                f"rigid body is outside articulation root: {shape.rigid_body_path}"
            )
        if shape.collision_path == shape.rigid_body_path:
            raise RuntimeError(
                "collision GPrim also owns RigidBodyAPI and cannot be used as a "
                f"query-shadow source: {shape.collision_path}"
            )
        if (
            shape.geometry_type == "Mesh"
            and shape.mesh_approximation != required_mesh_approximation
        ):
            raise RuntimeError(
                "collision mesh approximation mismatch: "
                f"path={shape.collision_path!r}, "
                f"actual={shape.mesh_approximation!r}, "
                f"required={required_mesh_approximation!r}"
            )
        if shape.collision_path in by_path:
            raise RuntimeError(
                f"duplicate collision prim identity: {shape.collision_path}"
            )
        by_path[shape.collision_path] = shape
    return tuple(by_path[path] for path in sorted(by_path))


def discover_scene_collision_identities(
    stage: object,
    object_root_path: str,
    *,
    inspector: _SceneCollisionIdentityInspector | None = None,
) -> tuple[PhysxSceneCollisionIdentity, ...]:
    """Discover exact active collider identities below one known object root.

    Unlike articulated shadow-source discovery, target discovery does not
    constrain mesh approximation or require a rigid-body owner.  It only
    establishes identities that PhysX overlap callbacks may report.
    """

    root_path = _absolute_prim_path(object_root_path, label="object_root_path")
    resolved_inspector = inspector or _UsdSceneCollisionIdentityInspector(stage)
    identities = tuple(resolved_inspector.inspect(root_path))
    if not identities:
        raise RuntimeError(f"scene object has no active collision GPrims: {root_path}")

    by_path: dict[str, PhysxSceneCollisionIdentity] = {}
    for identity in identities:
        if not isinstance(identity, PhysxSceneCollisionIdentity):
            raise TypeError(
                "scene collision inspector must return "
                "PhysxSceneCollisionIdentity values"
            )
        if not _is_descendant_or_self(identity.collision_path, root_path):
            raise RuntimeError(
                "scene collision prim is outside object root: "
                f"{identity.collision_path}"
            )
        if identity.rigid_body_path is not None and not _is_descendant_or_self(
            identity.rigid_body_path,
            root_path,
        ):
            raise RuntimeError(
                "scene collision rigid body is outside object root: "
                f"{identity.rigid_body_path}"
            )
        if identity.collision_path in by_path:
            raise RuntimeError(
                f"duplicate scene collision prim identity: {identity.collision_path}"
            )
        by_path[identity.collision_path] = identity
    return tuple(by_path[path] for path in sorted(by_path))


class _UsdCollisionShapeInspector:
    """USD traversal implementation loaded only inside an Isaac runtime."""

    def __init__(self, stage: object) -> None:
        try:
            from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        except ImportError as exc:
            raise RuntimeError(
                "PhysX collision-shape discovery requires an active Isaac USD runtime"
            ) from exc
        if stage is None:
            raise ValueError("stage is required")
        self._stage = stage
        self._Sdf = Sdf
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._UsdPhysics = UsdPhysics

    def inspect(
        self,
        articulation_root_path: str,
    ) -> Sequence[PhysxCollisionShapeMetadata]:
        root = self._stage.GetPrimAtPath(self._Sdf.Path(articulation_root_path))
        if not root.IsValid():
            raise RuntimeError(
                f"articulation root prim does not exist: {articulation_root_path}"
            )
        cache = self._UsdGeom.XformCache()
        shapes: list[PhysxCollisionShapeMetadata] = []
        for prim in self._Usd.PrimRange(
            root,
            self._Usd.TraverseInstanceProxies(),
        ):
            if not prim.HasAPI(self._UsdPhysics.CollisionAPI):
                continue
            collision_api = self._UsdPhysics.CollisionAPI(prim)
            if collision_api.GetCollisionEnabledAttr().Get() is False:
                continue
            if not prim.IsA(self._UsdGeom.Gprim):
                raise RuntimeError(
                    f"active CollisionAPI prim is not a GPrim: {prim.GetPath()}"
                )
            rigid_body = self._nearest_rigid_body(prim)
            if rigid_body is None:
                raise RuntimeError(
                    "articulated collision GPrim has no RigidBodyAPI ancestor: "
                    f"{prim.GetPath()}"
                )
            relative = (
                cache.GetLocalToWorldTransform(prim)
                * cache.GetLocalToWorldTransform(rigid_body).GetInverse()
            )
            matrix = tuple(
                float(relative[row][column]) for row in range(4) for column in range(4)
            )
            geometry_type = str(prim.GetTypeName())
            approximation = None
            if geometry_type == "Mesh":
                mesh_api = self._UsdPhysics.MeshCollisionAPI(prim)
                approximation = str(mesh_api.GetApproximationAttr().Get() or "")
            shapes.append(
                PhysxCollisionShapeMetadata(
                    collision_path=str(prim.GetPath()),
                    rigid_body_path=str(rigid_body.GetPath()),
                    geometry_type=geometry_type,
                    mesh_approximation=approximation,
                    shape_to_rigid_body_matrix_row_major=matrix,
                )
            )
        return tuple(shapes)

    def _nearest_rigid_body(self, prim: object) -> object | None:
        current = prim
        while current is not None and current.IsValid():
            if current.HasAPI(self._UsdPhysics.RigidBodyAPI):
                return current
            parent = current.GetParent()
            if parent == current or not parent.IsValid():
                break
            current = parent
        return None


class _UsdSceneCollisionIdentityInspector:
    """USD traversal for scene-object identities, loaded only inside Isaac."""

    def __init__(self, stage: object) -> None:
        try:
            from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        except ImportError as exc:
            raise RuntimeError(
                "PhysX scene collision discovery requires an active Isaac USD runtime"
            ) from exc
        if stage is None:
            raise ValueError("stage is required")
        self._stage = stage
        self._Sdf = Sdf
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._UsdPhysics = UsdPhysics

    def inspect(
        self,
        object_root_path: str,
    ) -> Sequence[PhysxSceneCollisionIdentity]:
        root = self._stage.GetPrimAtPath(self._Sdf.Path(object_root_path))
        if not root.IsValid():
            raise RuntimeError(
                f"scene object root prim does not exist: {object_root_path}"
            )
        identities: list[PhysxSceneCollisionIdentity] = []
        for prim in self._Usd.PrimRange(
            root,
            self._Usd.TraverseInstanceProxies(),
        ):
            if not prim.HasAPI(self._UsdPhysics.CollisionAPI):
                continue
            collision_api = self._UsdPhysics.CollisionAPI(prim)
            if collision_api.GetCollisionEnabledAttr().Get() is False:
                continue
            if not prim.IsA(self._UsdGeom.Gprim):
                raise RuntimeError(
                    f"active scene CollisionAPI prim is not a GPrim: {prim.GetPath()}"
                )
            rigid_body = self._nearest_rigid_body_within(prim, root)
            identities.append(
                PhysxSceneCollisionIdentity(
                    collision_path=str(prim.GetPath()),
                    rigid_body_path=(
                        None if rigid_body is None else str(rigid_body.GetPath())
                    ),
                )
            )
        return tuple(identities)

    def _nearest_rigid_body_within(
        self,
        prim: object,
        root: object,
    ) -> object | None:
        current = prim
        while current is not None and current.IsValid():
            if current.HasAPI(self._UsdPhysics.RigidBodyAPI):
                return current
            if current == root:
                break
            parent = current.GetParent()
            if parent == current or not parent.IsValid():
                break
            current = parent
        return None


def _absolute_prim_path(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "//" in value
    ):
        raise ValueError(f"{label} must be a canonical absolute USD prim path")
    return value


def _is_descendant_or_self(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


__all__ = [
    "CONVEX_DECOMPOSITION",
    "PhysxCollisionShapeMetadata",
    "PhysxSceneCollisionIdentity",
    "discover_articulated_collision_shapes",
    "discover_scene_collision_identities",
]
