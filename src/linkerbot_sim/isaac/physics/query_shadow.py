"""Lifecycle owner for query-only PhysX shadow GPrims."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import re
from typing import Protocol


_SHADOW_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DEFAULT_ROOT_PATH = "/World/__linkerbot_collision_query__"


class _ShadowStageBackend(Protocol):
    def create_root(self, root_path: str) -> None: ...

    def create_shadow(
        self,
        source_path: str,
        shadow_path: str,
    ) -> Sequence[str]: ...

    def set_world_transform(
        self,
        shadow_path: str,
        matrix_row_major: tuple[float, ...],
    ) -> None: ...

    def remove_prim(self, prim_path: str) -> None: ...


@dataclass(frozen=True, slots=True)
class PhysxQueryShadow:
    """Identity of one query-only GPrim owned by one shadow owner."""

    prim_path: str
    source_collision_path: str
    query_prim_paths: tuple[str, ...]
    _owner_token: object = field(repr=False)


class PhysxQueryShadowOwner:
    """Create, position, and remove a private subtree of query-only GPrims.

    This owner only prepares USD query inputs. It does not call scene queries,
    compute robot forward kinematics, or mutate an active articulation.
    """

    def __init__(
        self,
        stage: object,
        *,
        root_path: str = _DEFAULT_ROOT_PATH,
        backend: _ShadowStageBackend | None = None,
    ) -> None:
        self._root_path = _absolute_prim_path(root_path, label="root_path")
        self._backend = backend or _UsdShadowStageBackend(stage)
        self._owner_token = object()
        self._shadows: dict[str, PhysxQueryShadow] = {}
        self._closed = False
        self._backend.create_root(self._root_path)

    @property
    def root_path(self) -> str:
        return self._root_path

    @property
    def shadows(self) -> tuple[PhysxQueryShadow, ...]:
        return tuple(self._shadows.values())

    @property
    def closed(self) -> bool:
        return self._closed

    def create_shadow(
        self,
        *,
        name: str,
        source_collision_path: str,
    ) -> PhysxQueryShadow:
        """Create one disabled-collision GPrim referencing canonical geometry."""

        self._require_open()
        if not isinstance(name, str) or _SHADOW_NAME_RE.fullmatch(name) is None:
            raise ValueError("shadow name must match ^[a-z][a-z0-9_]*$")
        source_path = _absolute_prim_path(
            source_collision_path,
            label="source_collision_path",
        )
        if name in self._shadows:
            raise ValueError(f"shadow name already exists: {name!r}")
        shadow_path = f"{self._root_path}/{name}"
        try:
            query_prim_paths = tuple(
                self._backend.create_shadow(source_path, shadow_path)
            )
            if not query_prim_paths:
                raise RuntimeError(
                    f"query shadow has no convex query pieces: {shadow_path}"
                )
            expected_prefix = f"{shadow_path}/"
            for query_path in query_prim_paths:
                canonical = _absolute_prim_path(
                    query_path,
                    label="query_prim_path",
                )
                if canonical != shadow_path and not canonical.startswith(
                    expected_prefix
                ):
                    raise RuntimeError(
                        f"query shadow piece escapes its owner path: {canonical}"
                    )
        except BaseException as exc:
            try:
                self._backend.remove_prim(shadow_path)
            except BaseException as cleanup_exc:
                exc.add_note(
                    f"query-shadow partial-create cleanup failed: {cleanup_exc!r}"
                )
            raise
        shadow = PhysxQueryShadow(
            prim_path=shadow_path,
            source_collision_path=source_path,
            query_prim_paths=query_prim_paths,
            _owner_token=self._owner_token,
        )
        self._shadows[name] = shadow
        return shadow

    def set_world_transform(
        self,
        shadow: PhysxQueryShadow,
        *,
        matrix_row_major: Sequence[float],
    ) -> None:
        """Set a complete shape-local to world affine transform.

        All 16 Gf row-major values are retained so authored collider scale is
        not lost when candidate link FK is combined with the collider-local
        transform.
        """

        self._require_owned_shadow(shadow)
        matrix = _affine_matrix(matrix_row_major)
        self._backend.set_world_transform(
            shadow.prim_path,
            matrix,
        )

    def remove_shadow(self, shadow: PhysxQueryShadow) -> None:
        """Remove one owned shadow while keeping the owner open."""

        name = self._require_owned_shadow(shadow)
        self._backend.remove_prim(shadow.prim_path)
        del self._shadows[name]

    def close(self) -> None:
        """Idempotently remove the complete private query subtree."""

        if self._closed:
            return
        self._backend.remove_prim(self._root_path)
        self._shadows.clear()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PhysX query-shadow owner is closed")

    def _require_owned_shadow(self, shadow: PhysxQueryShadow) -> str:
        self._require_open()
        if not isinstance(shadow, PhysxQueryShadow):
            raise TypeError("shadow must be a PhysxQueryShadow")
        if shadow._owner_token is not self._owner_token:
            raise ValueError("shadow is not owned by this query-shadow owner")
        name = shadow.prim_path.rsplit("/", maxsplit=1)[-1]
        if self._shadows.get(name) is not shadow:
            raise ValueError("shadow is no longer owned by this query-shadow owner")
        return name


class _UsdShadowStageBackend:
    """Isaac/USD implementation hidden behind the pure owner tests."""

    def __init__(self, stage: object) -> None:
        try:
            from pxr import (
                Gf,
                PhysicsSchemaTools,
                Sdf,
                UsdGeom,
                UsdPhysics,
                UsdUtils,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PhysX query shadows require an active Isaac Sim USD runtime"
            ) from exc
        if stage is None:
            raise ValueError("stage is required")
        self._stage = stage
        self._Gf = Gf
        self._PhysicsSchemaTools = PhysicsSchemaTools
        self._Sdf = Sdf
        self._UsdGeom = UsdGeom
        self._UsdPhysics = UsdPhysics
        self._UsdUtils = UsdUtils
        self._transform_ops: dict[str, object] = {}

    def create_root(self, root_path: str) -> None:
        path = self._Sdf.Path(root_path)
        if self._stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(
                f"query-shadow root already exists; refusing to replace it: {root_path}"
            )
        root = self._UsdGeom.Scope.Define(self._stage, path).GetPrim()
        if not root.IsValid():
            raise RuntimeError(f"failed to create query-shadow root: {root_path}")

    def create_shadow(
        self,
        source_path: str,
        shadow_path: str,
    ) -> tuple[str, ...]:
        source = self._stage.GetPrimAtPath(self._Sdf.Path(source_path))
        if not source.IsValid():
            raise RuntimeError(f"source collision prim does not exist: {source_path}")
        if not source.IsA(self._UsdGeom.Gprim):
            raise RuntimeError(f"source collision prim is not a GPrim: {source_path}")
        if not source.HasAPI(self._UsdPhysics.CollisionAPI):
            raise RuntimeError(
                f"source GPrim does not have CollisionAPI: {source_path}"
            )
        if source.HasAPI(self._UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                "query-shadow source must not combine geometry and RigidBodyAPI: "
                f"{source_path}"
            )
        destination = self._stage.GetPrimAtPath(self._Sdf.Path(shadow_path))
        if destination.IsValid():
            raise RuntimeError(f"query-shadow prim already exists: {shadow_path}")

        mesh_collision = self._UsdPhysics.MeshCollisionAPI(source)
        approximation = mesh_collision.GetApproximationAttr().Get()
        if source.IsA(self._UsdGeom.Mesh) and approximation == "convexDecomposition":
            return self._create_convex_decomposition_shadow(
                source,
                shadow_path,
            )

        shadow = self._stage.OverridePrim(self._Sdf.Path(shadow_path))
        if not shadow.GetReferences().AddInternalReference(source.GetPath()):
            raise RuntimeError(
                f"failed to reference query-shadow source: {source_path}"
            )
        shadow = self._stage.GetPrimAtPath(self._Sdf.Path(shadow_path))
        if not shadow.IsValid() or not shadow.IsA(self._UsdGeom.Gprim):
            raise RuntimeError(f"referenced query shadow is not a GPrim: {shadow_path}")

        # The source's canonical geometry and mesh approximation are composed
        # through the internal reference. A stronger local false value prevents
        # this helper prim from becoming another physical scene collider.
        collision = self._UsdPhysics.CollisionAPI(shadow)
        collision.CreateCollisionEnabledAttr().Set(False)
        imageable = self._UsdGeom.Imageable(shadow)
        imageable.CreateVisibilityAttr().Set(self._UsdGeom.Tokens.invisible)
        imageable.CreatePurposeAttr().Set(self._UsdGeom.Tokens.guide)

        xform = self._UsdGeom.Xformable(shadow)
        xform.ClearXformOpOrder()
        transform_op = xform.AddTransformOp(
            precision=self._UsdGeom.XformOp.PrecisionDouble,
            opSuffix="queryWorldTransform",
        )
        transform_op.Set(self._Gf.Matrix4d(1.0))
        if not xform.SetXformOpOrder([transform_op], resetXformStack=True):
            raise RuntimeError(
                f"failed to freeze query-shadow world transform: {shadow_path}"
            )
        self._transform_ops[shadow_path] = transform_op
        return (shadow_path,)

    def _create_convex_decomposition_shadow(
        self,
        source: object,
        shadow_path: str,
    ) -> tuple[str, ...]:
        """Materialize PhysX-cooked convex pieces below one transform root."""

        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import (
            PhysxCollisionRepresentationResult,
        )

        returned: list[tuple[object, tuple[object, ...]]] = []

        def on_result(result: object, convexes: Sequence[object]) -> None:
            returned.append((result, tuple(convexes)))

        stage_id = self._UsdUtils.StageCache.Get().GetId(self._stage).ToLongInt()
        collision_prim_id = self._PhysicsSchemaTools.sdfPathToInt(source.GetPath())
        get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id,
            collision_prim_id=collision_prim_id,
            run_asynchronously=False,
            on_result=on_result,
        )
        if len(returned) != 1:
            raise RuntimeError(
                "PhysX synchronous convex-decomposition cooking did not return "
                f"one result: {source.GetPath()}"
            )
        result, convexes = returned[0]
        if result != PhysxCollisionRepresentationResult.RESULT_VALID or not convexes:
            raise RuntimeError(
                "PhysX failed to expose the active convex-decomposition "
                f"representation: source={source.GetPath()}, result={result!r}"
            )

        root = self._UsdGeom.Xform.Define(self._stage, shadow_path)
        root_imageable = self._UsdGeom.Imageable(root.GetPrim())
        root_imageable.CreateVisibilityAttr().Set(self._UsdGeom.Tokens.invisible)
        root_imageable.CreatePurposeAttr().Set(self._UsdGeom.Tokens.guide)
        transform_op = root.AddTransformOp(
            precision=self._UsdGeom.XformOp.PrecisionDouble,
            opSuffix="queryWorldTransform",
        )
        transform_op.Set(self._Gf.Matrix4d(1.0))
        if not root.SetXformOpOrder([transform_op], resetXformStack=True):
            raise RuntimeError(
                f"failed to freeze compound query-shadow transform: {shadow_path}"
            )
        self._transform_ops[shadow_path] = transform_op

        # Convex-decomposition piece order is not stable across cooking runs.
        ordered = sorted(
            convexes,
            key=lambda convex: (
                len(convex.indices),
                len(convex.vertices),
                len(convex.polygons),
                float(convex.vertices[0].x),
                float(convex.vertices[0].y),
                float(convex.vertices[0].z),
            ),
        )
        query_paths = []
        for index, convex in enumerate(ordered):
            query_path = f"{shadow_path}/piece_{index:04d}"
            mesh = self._UsdGeom.Mesh.Define(self._stage, query_path)
            points = [
                self._Gf.Vec3f(vertex.x, vertex.y, vertex.z)
                for vertex in convex.vertices
            ]
            face_vertex_counts = []
            face_vertex_indices = []
            for polygon in convex.polygons:
                start = int(polygon.index_base)
                count = int(polygon.num_vertices)
                face_vertex_counts.append(count)
                face_vertex_indices.extend(convex.indices[start : start + count])
            if not points or not face_vertex_counts or not face_vertex_indices:
                raise RuntimeError(
                    "PhysX returned an empty convex-decomposition piece: "
                    f"source={source.GetPath()}, index={index}"
                )
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
            mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
            mesh.GetSubdivisionSchemeAttr().Set(self._UsdGeom.Tokens.none)
            query_paths.append(query_path)
        return tuple(query_paths)

    def set_world_transform(
        self,
        shadow_path: str,
        matrix_row_major: tuple[float, ...],
    ) -> None:
        transform_op = self._transform_ops.get(shadow_path)
        if transform_op is None:
            raise RuntimeError(
                f"query-shadow transform op is unavailable: {shadow_path}"
            )
        matrix = self._Gf.Matrix4d(
            *matrix_row_major,
        )
        if not transform_op.Set(matrix):
            raise RuntimeError(
                f"failed to set query-shadow world transform: {shadow_path}"
            )

    def remove_prim(self, prim_path: str) -> None:
        prefix = f"{prim_path}/"
        for shadow_path in tuple(self._transform_ops):
            if shadow_path == prim_path or shadow_path.startswith(prefix):
                del self._transform_ops[shadow_path]
        path = self._Sdf.Path(prim_path)
        if self._stage.GetPrimAtPath(path).IsValid():
            self._stage.RemovePrim(path)


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


def _finite_tuple(
    value: Sequence[float],
    *,
    length: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    result: list[float] = []
    for component in value:
        if isinstance(component, bool):
            raise TypeError(f"{label} values must be real numbers")
        try:
            number = float(component)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} values must be real numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} values must be finite")
        result.append(number)
    return tuple(result)


def _affine_matrix(value: Sequence[float]) -> tuple[float, ...]:
    matrix = _finite_tuple(value, length=16, label="matrix_row_major")
    if any(
        not math.isclose(matrix[index], 0.0, rel_tol=0.0, abs_tol=1e-12)
        for index in (3, 7, 11)
    ) or not math.isclose(matrix[15], 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("matrix_row_major must be an affine Gf matrix")
    determinant = (
        matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
        - matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8])
        + matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8])
    )
    if determinant == 0.0:
        raise ValueError("matrix_row_major linear transform must be invertible")
    normalized = list(matrix)
    for index in (3, 7, 11):
        normalized[index] = 0.0
    normalized[15] = 1.0
    return tuple(normalized)


__all__ = ["PhysxQueryShadow", "PhysxQueryShadowOwner"]
