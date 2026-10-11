from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from linkerbot_sim.isaac.physics.collision_shapes import (
    CONVEX_DECOMPOSITION,
    PhysxCollisionShapeMetadata,
    PhysxSceneCollisionIdentity,
    discover_articulated_collision_shapes,
    discover_scene_collision_identities,
)


IDENTITY_MATRIX = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


class _FakeInspector:
    def __init__(self, shapes: tuple[object, ...]) -> None:
        self.shapes = shapes
        self.roots: list[str] = []

    def inspect(self, articulation_root_path: str) -> tuple[object, ...]:
        self.roots.append(articulation_root_path)
        return self.shapes


def _mesh(
    collision_path: str,
    *,
    rigid_body_path: str | None = None,
    approximation: str = CONVEX_DECOMPOSITION,
) -> PhysxCollisionShapeMetadata:
    return PhysxCollisionShapeMetadata(
        collision_path=collision_path,
        rigid_body_path=rigid_body_path or collision_path.rsplit("/", 1)[0],
        geometry_type="Mesh",
        mesh_approximation=approximation,
        shape_to_rigid_body_matrix_row_major=IDENTITY_MATRIX,
    )


def test_metadata_is_immutable_and_preserves_complete_affine_transform() -> None:
    shape = _mesh("/World/Robot/palm/collision")

    assert len(shape.shape_to_rigid_body_matrix_row_major) == 16
    assert shape.mesh_approximation == "convexDecomposition"
    with pytest.raises(FrozenInstanceError):
        shape.geometry_type = "Cube"  # type: ignore[misc]


def test_discovery_returns_stable_collision_path_order() -> None:
    later = _mesh("/World/Robot/finger/collision_b")
    earlier = _mesh("/World/Robot/finger/collision_a")
    inspector = _FakeInspector((later, earlier))

    result = discover_articulated_collision_shapes(
        object(),
        "/World/Robot",
        inspector=inspector,
    )

    assert inspector.roots == ["/World/Robot"]
    assert result == (earlier, later)


def test_usd_discovery_traverses_instance_proxy_colliders() -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Asset")
    body = UsdGeom.Xform.Define(stage, "/Asset/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    collider = UsdGeom.Mesh.Define(stage, "/Asset/body/collision").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr().Set(
        CONVEX_DECOMPOSITION
    )

    robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    robot.GetReferences().AddInternalReference(Sdf.Path("/Asset"))
    robot.SetInstanceable(True)

    proxy = stage.GetPrimAtPath("/World/Robot/body/collision")
    assert proxy.IsInstanceProxy()

    shapes = discover_articulated_collision_shapes(stage, "/World/Robot")
    identities = discover_scene_collision_identities(stage, "/World/Robot")

    assert tuple(shape.collision_path for shape in shapes) == (
        "/World/Robot/body/collision",
    )
    assert shapes[0].rigid_body_path == "/World/Robot/body"
    assert identities == (
        PhysxSceneCollisionIdentity(
            collision_path="/World/Robot/body/collision",
            rigid_body_path="/World/Robot/body",
        ),
    )


@pytest.mark.parametrize(
    "shape",
    [
        _mesh("/World/Other/collision", rigid_body_path="/World/Robot/link"),
        _mesh(
            "/World/Robot/link/collision",
            rigid_body_path="/World/Other/link",
        ),
    ],
)
def test_discovery_rejects_identity_outside_articulation_root(
    shape: PhysxCollisionShapeMetadata,
) -> None:
    with pytest.raises(RuntimeError, match="outside articulation root"):
        discover_articulated_collision_shapes(
            object(),
            "/World/Robot",
            inspector=_FakeInspector((shape,)),
        )


def test_discovery_rejects_mesh_approximation_mismatch() -> None:
    shape = _mesh(
        "/World/Robot/palm/collision",
        approximation="convexHull",
    )

    with pytest.raises(RuntimeError, match="approximation mismatch"):
        discover_articulated_collision_shapes(
            object(),
            "/World/Robot",
            inspector=_FakeInspector((shape,)),
        )


def test_discovery_rejects_collision_prim_that_is_also_rigid_body() -> None:
    shape = _mesh(
        "/World/Robot/palm",
        rigid_body_path="/World/Robot/palm",
    )

    with pytest.raises(RuntimeError, match="also owns RigidBodyAPI"):
        discover_articulated_collision_shapes(
            object(),
            "/World/Robot",
            inspector=_FakeInspector((shape,)),
        )


def test_discovery_rejects_duplicate_or_empty_inspector_results() -> None:
    shape = _mesh("/World/Robot/palm/collision")
    with pytest.raises(RuntimeError, match="duplicate collision"):
        discover_articulated_collision_shapes(
            object(),
            "/World/Robot",
            inspector=_FakeInspector((shape, shape)),
        )
    with pytest.raises(RuntimeError, match="no active collision"):
        discover_articulated_collision_shapes(
            object(),
            "/World/Robot",
            inspector=_FakeInspector(()),
        )


@pytest.mark.parametrize(
    ("geometry_type", "approximation"),
    [("Mesh", None), ("Cube", "convexDecomposition")],
)
def test_metadata_rejects_inconsistent_geometry_approximation(
    geometry_type: str,
    approximation: str | None,
) -> None:
    with pytest.raises(ValueError, match="mesh_approximation"):
        PhysxCollisionShapeMetadata(
            collision_path="/World/Robot/palm/collision",
            rigid_body_path="/World/Robot/palm",
            geometry_type=geometry_type,
            mesh_approximation=approximation,
            shape_to_rigid_body_matrix_row_major=IDENTITY_MATRIX,
        )


def test_metadata_rejects_nonfinite_or_wrong_size_transform() -> None:
    with pytest.raises(ValueError, match="tuple of 16 values"):
        PhysxCollisionShapeMetadata(
            collision_path="/World/Robot/palm/collision",
            rigid_body_path="/World/Robot/palm",
            geometry_type="Mesh",
            mesh_approximation=CONVEX_DECOMPOSITION,
            shape_to_rigid_body_matrix_row_major=(1.0,),
        )
    invalid = (*IDENTITY_MATRIX[:-1], float("nan"))
    with pytest.raises(ValueError, match="finite numbers"):
        PhysxCollisionShapeMetadata(
            collision_path="/World/Robot/palm/collision",
            rigid_body_path="/World/Robot/palm",
            geometry_type="Mesh",
            mesh_approximation=CONVEX_DECOMPOSITION,
            shape_to_rigid_body_matrix_row_major=invalid,
        )

    non_affine = (*IDENTITY_MATRIX[:3], 1.0, *IDENTITY_MATRIX[4:])
    with pytest.raises(ValueError, match="affine Gf matrix"):
        PhysxCollisionShapeMetadata(
            collision_path="/World/Robot/palm/collision",
            rigid_body_path="/World/Robot/palm",
            geometry_type="Mesh",
            mesh_approximation=CONVEX_DECOMPOSITION,
            shape_to_rigid_body_matrix_row_major=non_affine,
        )


def test_discovery_rejects_non_metadata_inspector_value() -> None:
    with pytest.raises(TypeError, match="PhysxCollisionShapeMetadata"):
        discover_articulated_collision_shapes(
            object(),
            "/World/Robot",
            inspector=_FakeInspector((object(),)),
        )


def test_scene_identity_discovery_allows_static_colliders_and_sorts_paths() -> None:
    later = PhysxSceneCollisionIdentity(
        collision_path="/World/Table/collision_b",
        rigid_body_path=None,
    )
    earlier = PhysxSceneCollisionIdentity(
        collision_path="/World/Table/collision_a",
        rigid_body_path="/World/Table/link",
    )
    inspector = _FakeInspector((later, earlier))

    result = discover_scene_collision_identities(
        object(),
        "/World/Table",
        inspector=inspector,
    )

    assert inspector.roots == ["/World/Table"]
    assert result == (earlier, later)


@pytest.mark.parametrize(
    "identity",
    [
        PhysxSceneCollisionIdentity("/World/Other/collision", None),
        PhysxSceneCollisionIdentity(
            "/World/Table/collision",
            "/World/Other/body",
        ),
    ],
)
def test_scene_identity_discovery_rejects_paths_outside_object_root(
    identity: PhysxSceneCollisionIdentity,
) -> None:
    with pytest.raises(RuntimeError, match="outside object root"):
        discover_scene_collision_identities(
            object(),
            "/World/Table",
            inspector=_FakeInspector((identity,)),
        )


def test_scene_identity_discovery_rejects_duplicate_empty_or_wrong_values() -> None:
    identity = PhysxSceneCollisionIdentity("/World/Table/collision", None)
    with pytest.raises(RuntimeError, match="duplicate scene collision"):
        discover_scene_collision_identities(
            object(),
            "/World/Table",
            inspector=_FakeInspector((identity, identity)),
        )
    with pytest.raises(RuntimeError, match="no active collision"):
        discover_scene_collision_identities(
            object(),
            "/World/Table",
            inspector=_FakeInspector(()),
        )
    with pytest.raises(TypeError, match="PhysxSceneCollisionIdentity"):
        discover_scene_collision_identities(
            object(),
            "/World/Table",
            inspector=_FakeInspector((object(),)),
        )
