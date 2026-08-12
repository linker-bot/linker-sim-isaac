from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from linkerbot_sim.configuration.objects import (
    CapsuleRopePhysicsConfig,
    DynamicChainObjectProfileConfig,
    ObjectProfileConfig,
    RigidObjectPhysicsConfig,
    RigidObjectPlanningCollisionConfig,
    RigidObjectProfileConfig,
    object_profile_from_mapping,
)
from linkerbot_sim.configuration.catalog import _ConfigurationGraphReader
from linkerbot_sim.configuration.scenes import ObjectInstanceSettings
from linkerbot_sim.objects import runtime as object_runtime
from linkerbot_sim.objects.runtime import (
    runtime_object_prim_path,
    runtime_objects_from_settings,
)
from linkerbot_sim.utils.config import load_yaml


def _profile() -> ObjectProfileConfig:
    return object_profile_from_mapping(
        {
            "object": {
                "name": "fixture",
                "kind": "rigid",
                "source": "usd",
                "asset_path": "fixture.usd",
            }
        },
        profile_name="fixture",
    )


def _instance(profile: ObjectProfileConfig | None = None) -> ObjectInstanceSettings:
    parsed = ObjectInstanceSettings.from_mapping(
        {
            "name": "fixture",
            "object_profile": "fixture",
            "prim_path": "/World/Scene/Fixture",
            "root_pose": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        },
        label="scene.objects[0]",
    )
    return replace(parsed, resolved_profile=profile)


def _dynamic_profile() -> DynamicChainObjectProfileConfig:
    profile = object_profile_from_mapping(
        {
            "object": {
                "name": "rope",
                "kind": "dynamic_chain",
                "source": "usd",
                "asset_path": "rope.usda",
                "root_path": "/Rope",
                "state_summary": {"reference_body": "left_box"},
            }
        },
        profile_name="rope",
    )
    assert isinstance(profile, DynamicChainObjectProfileConfig)
    return profile


def test_all_bundled_object_profiles_parse_strictly() -> None:
    paths = sorted(Path("configs/objects").glob("*.yaml"))

    assert paths
    for path in paths:
        profile = object_profile_from_mapping(
            load_yaml(path), profile_name=path.stem, source=str(path)
        )
        assert profile.profile_name == path.stem


def test_rigid_profile_keeps_typed_import_physics_and_planning_settings() -> None:
    path = Path("configs/objects/TblockV1_default.yaml")
    profile = object_profile_from_mapping(
        load_yaml(path), profile_name=path.stem, source=str(path)
    )

    assert isinstance(profile, RigidObjectProfileConfig)
    assert isinstance(profile.physics, RigidObjectPhysicsConfig)
    assert profile.physics.material is not None
    assert profile.physics.material.static_friction == 0.8
    assert isinstance(profile.planning_collision, RigidObjectPlanningCollisionConfig)
    assert profile.planning_collision.shape == "cuboid"
    assert not hasattr(profile, "raw")


def test_capsule_rope_profile_keeps_typed_state_summary() -> None:
    path = Path("configs/objects/capsule_rope.yaml")
    profile = object_profile_from_mapping(
        load_yaml(path), profile_name=path.stem, source=str(path)
    )

    assert isinstance(profile, DynamicChainObjectProfileConfig)
    assert profile.root_path == "/CapsuleRope"
    assert profile.state_summary.reference_body == "left_box"
    assert isinstance(profile.physics, CapsuleRopePhysicsConfig)
    assert profile.physics.physx is not None
    assert profile.physics.physx.material is not None
    assert profile.physics.physx.material.friction_combine_mode == "average"
    assert profile.physics.physx.solver is not None
    assert profile.physics.physx.solver.position_iterations == 48
    assert profile.physics.physx.solver.velocity_iterations == 4


def test_catalog_projects_object_profile_to_discriminated_type() -> None:
    reader = _ConfigurationGraphReader(Path("configs"))

    rigid = reader.object_profile(
        instance_name="fixture",
        reference="workstation_armbase",
    )
    chain = reader.object_profile(
        instance_name="rope",
        reference="capsule_rope",
    )

    assert isinstance(rigid, RigidObjectProfileConfig)
    assert rigid.import_config.collision_approximation == "convex_decomposition"
    assert isinstance(chain, DynamicChainObjectProfileConfig)
    assert reader.sources["object.fixture"].name == "workstation_armbase.yaml"
    assert reader.sources["object.rope"].name == "capsule_rope.yaml"


def test_object_profile_rejects_scene_prim_path() -> None:
    with pytest.raises(ValueError, match=r"object\.prim_path"):
        object_profile_from_mapping(
            {
                "object": {
                    "kind": "rigid",
                    "source": "usd",
                    "asset_path": "fixture.usd",
                    "prim_path": "/World/Fixture",
                }
            },
            profile_name="fixture",
            source="object.yaml",
        )


@pytest.mark.parametrize(
    "section", ("import", "physics", "planning_collision", "state_summary")
)
def test_object_rejects_null_mapping_sections(section: str) -> None:
    with pytest.raises(ValueError, match=rf"object\.{section} must be a mapping"):
        object_profile_from_mapping(
            {
                "object": {
                    "kind": "rigid",
                    "source": "usd",
                    "asset_path": "fixture.usd",
                    section: None,
                }
            },
            profile_name="fixture",
        )


@pytest.mark.parametrize(
    ("mutation", "path"),
    (
        ({"typo": True}, "profile.typo"),
        ({"object": {"naem": "fixture"}}, "object.naem"),
        (
            {"object": {"physics": {"material": {"static_frictino": 0.5}}}},
            "object.physics.material.static_frictino",
        ),
        (
            {"object": {"planning_collision": {"paddding": 0.1}}},
            "object.planning_collision.paddding",
        ),
    ),
)
def test_object_rejects_nested_typos(mutation: dict[str, object], path: str) -> None:
    profile = load_yaml("configs/objects/TblockV1_default.yaml")
    if "object" not in mutation:
        profile.update(mutation)
    else:
        object_mutation = mutation["object"]
        assert isinstance(object_mutation, dict)
        for key, value in object_mutation.items():
            current = profile["object"].get(key)
            if isinstance(value, dict) and isinstance(current, dict):
                current.update(value)
            else:
                profile["object"][key] = value

    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        object_profile_from_mapping(profile, profile_name="fixture", source="typo.yaml")


def test_dynamic_chain_requires_named_reference_body() -> None:
    with pytest.raises(ValueError, match=r"state_summary\.reference_body is required"):
        object_profile_from_mapping(
            {
                "object": {
                    "name": "rope",
                    "kind": "dynamic_chain",
                    "source": "usd",
                    "asset_path": "rope.usda",
                    "root_path": "/Rope",
                }
            },
            profile_name="rope",
        )


def test_dynamic_chain_requires_explicit_asset_root_path() -> None:
    with pytest.raises(ValueError, match=r"object\.root_path is required"):
        object_profile_from_mapping(
            {
                "object": {
                    "name": "rope",
                    "kind": "dynamic_chain",
                    "source": "usd",
                    "asset_path": "rope.usda",
                    "state_summary": {"reference_body": "left_box"},
                }
            },
            profile_name="rope",
        )


def test_rigid_profile_rejects_conflicting_fixed_dynamic_semantics() -> None:
    with pytest.raises(ValueError, match=r"fix_base=true conflicts"):
        object_profile_from_mapping(
            {
                "object": {
                    "name": "fixture",
                    "kind": "rigid",
                    "source": "urdf",
                    "asset_path": "fixture.urdf",
                    "import": {"fix_base": True},
                    "physics": {"static": False},
                }
            },
            profile_name="fixture",
        )


@pytest.mark.parametrize(
    "import_settings",
    ({}, {"collision_approximation": "convex_hull"}),
)
def test_usd_object_profile_rejects_importer_settings(
    import_settings: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="not supported for USD assets"):
        object_profile_from_mapping(
            {
                "object": {
                    "name": "fixture",
                    "kind": "rigid",
                    "source": "usd",
                    "asset_path": "fixture.usda",
                    "import": import_settings,
                }
            },
            profile_name="fixture",
        )


def test_runtime_projection_requires_catalog_bound_profile() -> None:
    unresolved = _instance()
    with pytest.raises(TypeError, match="resolved ObjectProfileConfig"):
        runtime_objects_from_settings((unresolved,))

    instance = replace(unresolved, resolved_profile=_profile())
    runtime = runtime_objects_from_settings((instance,))[0]

    assert runtime.prim_path == "/World/Scene/Fixture"


def test_runtime_object_prim_path_prefers_canonical_stage_path() -> None:
    handle = SimpleNamespace(
        model=SimpleNamespace(
            prim_path="/World/canonical",
            imported_path="/World/imported",
        ),
        config=SimpleNamespace(prim_path="/World/configured"),
    )

    assert runtime_object_prim_path(handle) == "/World/canonical"


def test_runtime_object_prim_path_reads_dynamic_chain_root() -> None:
    root = SimpleNamespace(GetPath=lambda: "/World/rope")

    assert (
        runtime_object_prim_path(SimpleNamespace(model={"root": root}, config=None))
        == "/World/rope"
    )


@pytest.mark.parametrize(
    ("profile", "factory_name"),
    (
        (_profile(), "_add_rigid_object"),
        (_dynamic_profile(), "_add_capsule_rope_object"),
    ),
)
def test_newton_object_import_prepares_render_topology_after_asset_author(
    monkeypatch: pytest.MonkeyPatch,
    profile: ObjectProfileConfig,
    factory_name: str,
) -> None:
    events: list[object] = []
    handle = SimpleNamespace(
        model=SimpleNamespace(prim_path="/World/Object"),
        config=None,
    )
    monkeypatch.setattr(
        object_runtime,
        factory_name,
        lambda *_args, **_kwargs: events.append("asset_author") or handle,
    )
    monkeypatch.setattr(
        object_runtime,
        "prepare_newton_render_subtree",
        lambda **kwargs: events.append(
            ("render_topology", kwargs["stage"], kwargs["subtree_root"])
        ),
    )
    stage = object()

    actual = object_runtime.add_runtime_object(
        stage,
        config=SimpleNamespace(
            kind=profile.kind,
            source=profile.source,
            name="object",
            profile=profile,
        ),
        physics_backend="newton",
        prepare_newton_render_topology=True,
    )

    assert actual is handle
    assert events == [
        "asset_author",
        ("render_topology", stage, "/World/Object"),
    ]


def test_physx_object_import_does_not_author_newton_render_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = SimpleNamespace(
        model=SimpleNamespace(prim_path="/World/Object"),
        config=None,
    )
    monkeypatch.setattr(
        object_runtime,
        "_add_rigid_object",
        lambda *_args, **_kwargs: handle,
    )
    monkeypatch.setattr(
        object_runtime,
        "prepare_newton_render_subtree",
        lambda **_kwargs: pytest.fail("PhysX must not author Newton render topology"),
    )

    actual = object_runtime.add_runtime_object(
        object(),
        config=SimpleNamespace(
            kind="rigid", source="usd", name="object", profile=_profile()
        ),
        physics_backend="physx",
        prepare_newton_render_topology=False,
    )

    assert actual is handle


def test_headless_newton_object_import_keeps_original_xform_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = SimpleNamespace(
        model=SimpleNamespace(prim_path="/World/Object"),
        config=None,
    )
    monkeypatch.setattr(
        object_runtime,
        "_add_rigid_object",
        lambda *_args, **_kwargs: handle,
    )
    monkeypatch.setattr(
        object_runtime,
        "prepare_newton_render_subtree",
        lambda **_kwargs: pytest.fail(
            "headless Newton must keep the asset xform topology"
        ),
    )

    actual = object_runtime.add_runtime_object(
        object(),
        config=SimpleNamespace(
            kind="rigid", source="usd", name="object", profile=_profile()
        ),
        physics_backend="newton",
        prepare_newton_render_topology=False,
    )

    assert actual is handle
