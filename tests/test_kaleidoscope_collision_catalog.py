from __future__ import annotations

from types import SimpleNamespace

import pytest

from linkerbot_sim.isaac.physics.collision_shapes import (
    CONVEX_DECOMPOSITION,
    PhysxCollisionShapeMetadata,
    PhysxSceneCollisionIdentity,
)
from linkerbot_sim.kaleidoscope.collision_catalog import (
    RobotCollisionCatalog,
    SceneCollisionTargetCatalog,
    build_robot_collision_catalog,
    build_scene_collision_target_catalog,
)
from linkerbot_sim.kaleidoscope.collision_query import CollisionQueryTarget


IDENTITY = (
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


def _scene(*, num_envs: int = 2, strategy: str = "env_ids") -> object:
    env_roots = tuple(f"/World/envs/env_{index}" for index in range(num_envs))
    robots = []
    for robot_id, label in enumerate(("left", "right")):
        imported = tuple(f"{root}/{label}" for root in env_roots)
        robots.append(
            SimpleNamespace(
                robot_id=robot_id,
                label=label,
                imported_root_paths=imported,
                articulation_paths=tuple(f"{root}/Robot" for root in imported),
            )
        )
    return SimpleNamespace(
        collision_isolation_strategy=strategy,
        env_root_paths=env_roots,
        robots=tuple(robots),
    )


def _metadata(
    articulation_path: str,
    suffix: str,
    *,
    matrix: tuple[float, ...] = IDENTITY,
) -> PhysxCollisionShapeMetadata:
    link_suffix = suffix.rsplit("/", 1)[0]
    return PhysxCollisionShapeMetadata(
        collision_path=f"{articulation_path}{suffix}",
        rigid_body_path=f"{articulation_path}{link_suffix}",
        geometry_type="Mesh",
        mesh_approximation=CONVEX_DECOMPOSITION,
        shape_to_rigid_body_matrix_row_major=matrix,
    )


def _scanner(_stage: object, articulation_path: str):
    return (
        _metadata(articulation_path, "/palm/collision"),
        _metadata(articulation_path, "/finger/collision"),
    )


def _target_scene(*, strategy: str = "env_ids") -> object:
    scene = _scene(strategy=strategy)
    scene.object_handles = (
        SimpleNamespace(
            name="workstation",
            kind="rigid",
            config=SimpleNamespace(physics=SimpleNamespace(static=True)),
        ),
        SimpleNamespace(
            name="Tblock",
            kind="rigid",
            config=SimpleNamespace(physics=SimpleNamespace(static=False)),
        ),
    )
    scene.object_prim_paths = {
        "workstation": tuple(f"{root}/workstation" for root in scene.env_root_paths),
        "Tblock": tuple(f"{root}/Tblock" for root in scene.env_root_paths),
    }
    return scene


def _scene_scanner(_stage: object, object_root: str):
    rigid_body_path = None if object_root.endswith("workstation") else object_root
    return (
        PhysxSceneCollisionIdentity(
            collision_path=f"{object_root}/collision_b",
            rigid_body_path=rigid_body_path,
        ),
        PhysxSceneCollisionIdentity(
            collision_path=f"{object_root}/collision_a",
            rigid_body_path=rigid_body_path,
        ),
    )


def _target_catalog(*, scene: object | None = None) -> SceneCollisionTargetCatalog:
    resolved_scene = scene or _target_scene()
    robot_catalog = build_robot_collision_catalog(
        stage=object(),
        replicated_scene=resolved_scene,
        scanner=_scanner,
    )
    return build_scene_collision_target_catalog(
        stage=object(),
        replicated_scene=resolved_scene,
        robot_catalog=robot_catalog,
        dynamic_object_name="Tblock",
        scanner=_scene_scanner,
    )


def test_catalog_binds_exact_env_robot_and_articulation_identity() -> None:
    catalog = build_robot_collision_catalog(
        stage=object(),
        replicated_scene=_scene(),
        scanner=_scanner,
    )

    assert isinstance(catalog, RobotCollisionCatalog)
    assert catalog.num_envs == 2
    assert catalog.robot_labels == ("left", "right")
    assert len(catalog.shapes) == 8
    left_env_1 = catalog.shapes_for(env_id=1, robot_label="left")
    assert [shape.collision_suffix for shape in left_env_1] == [
        "/finger/collision",
        "/palm/collision",
    ]
    assert all(shape.env_id == 1 for shape in left_env_1)
    assert all(shape.robot_id == 0 for shape in left_env_1)
    assert all(
        shape.collision_path.startswith("/World/envs/env_1/left/Robot/")
        for shape in left_env_1
    )
    selected = catalog.shape_for_collision_path(left_env_1[0].collision_path)
    assert selected is left_env_1[0]


def test_catalog_uses_stable_env_then_scene_robot_then_path_order() -> None:
    catalog = build_robot_collision_catalog(
        stage=object(),
        replicated_scene=_scene(),
        scanner=_scanner,
    )

    keys = [
        (shape.env_id, shape.robot_label, shape.collision_suffix)
        for shape in catalog.shapes
    ]
    assert keys == sorted(
        keys,
        key=lambda item: (
            item[0],
            ("left", "right").index(item[1]),
            item[2],
        ),
    )


def test_catalog_rejects_non_physx_collision_isolation() -> None:
    with pytest.raises(ValueError, match="PhysX env_ids isolation"):
        build_robot_collision_catalog(
            stage=object(),
            replicated_scene=_scene(strategy="separate_worlds"),
            scanner=_scanner,
        )


def test_catalog_rejects_robot_paths_that_do_not_cover_all_envs() -> None:
    scene = _scene()
    scene.robots[0].articulation_paths = scene.robots[0].articulation_paths[:1]

    with pytest.raises(ValueError, match="exactly cover all envs"):
        build_robot_collision_catalog(
            stage=object(),
            replicated_scene=scene,
            scanner=_scanner,
        )


def test_catalog_rejects_articulation_outside_env_or_imported_root() -> None:
    scene = _scene(num_envs=1)
    scene.robots[0].imported_root_paths = ("/World/other/left",)
    scene.robots[0].articulation_paths = ("/World/other/left/Robot",)
    with pytest.raises(ValueError, match="outside its env root"):
        build_robot_collision_catalog(
            stage=object(), replicated_scene=scene, scanner=_scanner
        )

    scene = _scene(num_envs=1)
    scene.robots[0].articulation_paths = ("/World/envs/env_0/other/Robot",)
    with pytest.raises(ValueError, match="outside robot imported root"):
        build_robot_collision_catalog(
            stage=object(), replicated_scene=scene, scanner=_scanner
        )


def test_catalog_rejects_collision_identity_outside_articulation() -> None:
    def invalid_scanner(_stage: object, _articulation_path: str):
        return (_metadata("/World/other/Robot", "/palm/collision"),)

    with pytest.raises(RuntimeError, match="outside articulation path"):
        build_robot_collision_catalog(
            stage=object(),
            replicated_scene=_scene(num_envs=1),
            scanner=invalid_scanner,
        )


def test_catalog_rejects_cloned_topology_difference() -> None:
    def mismatched_scanner(_stage: object, articulation_path: str):
        if "env_0" in articulation_path:
            return (_metadata(articulation_path, "/palm/collision"),)
        return (_metadata(articulation_path, "/finger/collision"),)

    with pytest.raises(RuntimeError, match="topology differs from env 0"):
        build_robot_collision_catalog(
            stage=object(),
            replicated_scene=_scene(),
            scanner=mismatched_scanner,
        )


def test_catalog_rejects_cloned_local_transform_difference() -> None:
    changed = (2.0, *IDENTITY[1:])

    def mismatched_scanner(_stage: object, articulation_path: str):
        matrix = IDENTITY if "env_0" in articulation_path else changed
        return (_metadata(articulation_path, "/palm/collision", matrix=matrix),)

    with pytest.raises(RuntimeError, match="metadata differs from env 0"):
        build_robot_collision_catalog(
            stage=object(),
            replicated_scene=_scene(),
            scanner=mismatched_scanner,
        )


def test_catalog_rejects_empty_robot_scan() -> None:
    with pytest.raises(RuntimeError, match="has no collision shapes"):
        build_robot_collision_catalog(
            stage=object(),
            replicated_scene=_scene(num_envs=1),
            scanner=lambda _stage, _path: (),
        )


def test_catalog_lookup_rejects_unknown_env_robot_or_collision() -> None:
    catalog = build_robot_collision_catalog(
        stage=object(),
        replicated_scene=_scene(num_envs=1),
        scanner=_scanner,
    )
    with pytest.raises(ValueError, match="outside catalog range"):
        catalog.shapes_for(env_id=1, robot_label="left")
    with pytest.raises(KeyError, match="unknown robot label"):
        catalog.shapes_for(env_id=0, robot_label="missing")
    with pytest.raises(KeyError, match="unknown collision path"):
        catalog.shape_for_collision_path("/World/missing")


def test_catalog_rejects_duplicate_robot_identity() -> None:
    scene = _scene(num_envs=1)
    scene.robots[1].label = "left"
    with pytest.raises(ValueError, match="labels must be non-empty and unique"):
        build_robot_collision_catalog(
            stage=object(), replicated_scene=scene, scanner=_scanner
        )

    scene = _scene(num_envs=1)
    scene.robots[1].robot_id = 0
    with pytest.raises(ValueError, match="ids must be unique"):
        build_robot_collision_catalog(
            stage=object(), replicated_scene=scene, scanner=_scanner
        )


def test_scene_target_catalog_selects_exact_same_env_targets() -> None:
    catalog = _target_catalog()

    assert catalog.num_envs == 2
    assert catalog.robot_labels == ("left", "right")
    selected = catalog.targets_for(
        env_id=1,
        robot_label="left",
        targets=(
            CollisionQueryTarget.TABLE,
            CollisionQueryTarget.DYNAMIC_OBJECT,
            CollisionQueryTarget.OTHER_ROBOT,
        ),
    )

    assert len(selected) == 6
    assert all(shape.env_id == 1 for shape in selected)
    assert {shape.category for shape in selected} == set(CollisionQueryTarget)
    other_robot = [
        shape
        for shape in selected
        if shape.category is CollisionQueryTarget.OTHER_ROBOT
    ]
    assert {shape.owner_robot_label for shape in other_robot} == {"right"}
    assert all("/env_1/left/" not in shape.collision_path for shape in selected)


def test_scene_target_catalog_filters_and_classifies_by_full_path_equality() -> None:
    catalog = _target_catalog()
    targets = (CollisionQueryTarget.TABLE, CollisionQueryTarget.OTHER_ROBOT)
    paths = catalog.collision_paths_for(
        env_id=0,
        robot_label="right",
        targets=targets,
    )
    table_path = "/World/envs/env_0/workstation/collision_a"
    left_path = "/World/envs/env_0/left/Robot/finger/collision"

    assert table_path in paths
    assert left_path in paths
    assert (
        catalog.category_for_collision_path(
            table_path,
            env_id=0,
            robot_label="right",
            targets=targets,
        )
        is CollisionQueryTarget.TABLE
    )
    assert (
        catalog.category_for_collision_path(
            f"{table_path}_suffix",
            env_id=0,
            robot_label="right",
            targets=targets,
        )
        is None
    )
    assert (
        catalog.category_for_collision_path(
            "/World/envs/env_1/workstation/collision_a",
            env_id=0,
            robot_label="right",
            targets=targets,
        )
        is None
    )
    assert (
        catalog.category_for_collision_path(
            "/World/envs/env_0/right/Robot/finger/collision",
            env_id=0,
            robot_label="right",
            targets=targets,
        )
        is None
    )


def test_scene_target_catalog_rejects_newton_or_misaligned_robot_catalog() -> None:
    scene = _target_scene(strategy="separate_worlds")
    robot_catalog = build_robot_collision_catalog(
        stage=object(),
        replicated_scene=_target_scene(),
        scanner=_scanner,
    )
    with pytest.raises(ValueError, match="PhysX env_ids isolation"):
        build_scene_collision_target_catalog(
            stage=object(),
            replicated_scene=scene,
            robot_catalog=robot_catalog,
            dynamic_object_name="Tblock",
            scanner=_scene_scanner,
        )

    scene = _target_scene()
    scene.robots = tuple(reversed(scene.robots))
    with pytest.raises(ValueError, match="labels do not match"):
        build_scene_collision_target_catalog(
            stage=object(),
            replicated_scene=scene,
            robot_catalog=robot_catalog,
            dynamic_object_name="Tblock",
            scanner=_scene_scanner,
        )


def test_scene_target_catalog_rejects_wrong_object_roles_or_paths() -> None:
    scene = _target_scene()
    scene.object_handles = (
        SimpleNamespace(
            name="workstation",
            kind="rigid",
            config=SimpleNamespace(physics=SimpleNamespace(static=False)),
        ),
        scene.object_handles[1],
    )
    with pytest.raises(ValueError, match="configured static table"):
        _target_catalog(scene=scene)

    scene = _target_scene()
    scene.object_prim_paths["Tblock"] = scene.object_prim_paths["Tblock"][:1]
    with pytest.raises(ValueError, match="exactly cover all envs"):
        _target_catalog(scene=scene)

    scene = _target_scene()
    scene.object_prim_paths["workstation"] = (
        "/World/outside/workstation",
        scene.object_prim_paths["workstation"][1],
    )
    with pytest.raises(ValueError, match="outside its env root"):
        _target_catalog(scene=scene)


def test_scene_target_catalog_rejects_cloned_object_topology_difference() -> None:
    scene = _target_scene()
    robot_catalog = build_robot_collision_catalog(
        stage=object(),
        replicated_scene=scene,
        scanner=_scanner,
    )

    def mismatched_scanner(_stage: object, object_root: str):
        suffix = "collision_a" if "env_0" in object_root else "collision_changed"
        return (PhysxSceneCollisionIdentity(f"{object_root}/{suffix}", None),)

    with pytest.raises(RuntimeError, match="topology differs from env 0"):
        build_scene_collision_target_catalog(
            stage=object(),
            replicated_scene=scene,
            robot_catalog=robot_catalog,
            dynamic_object_name="Tblock",
            scanner=mismatched_scanner,
        )


def test_scene_target_catalog_rejects_invalid_selectors() -> None:
    catalog = _target_catalog()
    with pytest.raises(ValueError, match="outside catalog range"):
        catalog.targets_for(
            env_id=2,
            robot_label="left",
            targets=(CollisionQueryTarget.TABLE,),
        )
    with pytest.raises(KeyError, match="unknown robot label"):
        catalog.targets_for(
            env_id=0,
            robot_label="missing",
            targets=(CollisionQueryTarget.TABLE,),
        )
    with pytest.raises(ValueError, match="non-empty unique"):
        catalog.targets_for(env_id=0, robot_label="left", targets=())
