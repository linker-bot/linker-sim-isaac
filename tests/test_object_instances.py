from __future__ import annotations

from pathlib import Path

import pytest

from linkerbot_sim.objects.runtime import runtime_objects_from_env_config
from linkerbot_sim.objects.config import (
    ObjectProfileConfig,
    expanded_object_mapping,
    load_object_profile,
    object_scene_instances_from_env_config,
    validate_object_profile,
)
from linkerbot_sim.utils.config import load_yaml


def _instance(
    name: str,
    *,
    profile: str = "fixture",
    prim_path: str | None = None,
    runtime_handle: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "object_profile": profile,
        "root_pose": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
    }
    if prim_path is not None:
        result["prim_path"] = prim_path
    if runtime_handle is not None:
        result["runtime_handle"] = runtime_handle
    return result


def _profile() -> ObjectProfileConfig:
    object_config: dict[str, object] = {
        "name": "fixture",
        "kind": "rigid",
        "source": "usd",
        "asset_path": "fixture.usd",
    }
    return ObjectProfileConfig.from_mapping(
        {"object": object_config}, profile_name="fixture"
    )


def test_object_instances_use_explicit_or_name_derived_paths() -> None:
    instances = object_scene_instances_from_env_config(
        {
            "objects": [
                _instance("fixture_a"),
                _instance("fixture_b", prim_path="/World/Fixtures/custom"),
            ]
        }
    )

    assert instances[0].prim_path is None
    assert instances[0].default_prim_path == "/World/Objects/fixture_a"
    assert instances[0].effective_prim_path == "/World/Objects/fixture_a"
    assert instances[1].effective_prim_path == "/World/Fixtures/custom"


def test_dynamic_chain_profile_parses_reference_body_state_summary() -> None:
    profile = ObjectProfileConfig.from_mapping(
        {
            "object": {
                "name": "rope",
                "kind": "dynamic_chain",
                "source": "usd",
                "asset_path": "rope.usda",
                "state_summary": {"reference_body": "left_box"},
            },
        },
        profile_name="rope",
    )

    assert profile.state_summary.reference_body == "left_box"


def test_all_bundled_object_profiles_load_strictly() -> None:
    for path in sorted(Path("configs/objects").glob("*.yaml")):
        load_object_profile(path)


def test_object_example_fields_are_consumed_by_runtime_parsers() -> None:
    profile = ObjectProfileConfig.from_profile("example")

    assert profile.kind == "dynamic_chain"
    assert profile.source == "usd"
    assert profile.root_path == "/CapsuleRope"
    assert profile.state_summary.reference_body == "left_box"
    assert profile.physics is not None
    assert profile.physics["solver_position_iterations"] == 48


def test_object_profile_rejects_unknown_prim_path() -> None:
    with pytest.raises(
        ValueError,
        match=r"object\.prim_path",
    ):
        validate_object_profile(
            {
                "object": {
                    "kind": "rigid",
                    "source": "usd",
                    "asset_path": "fixture.usd",
                    "prim_path": "/World/Fixture",
                },
            },
            source="object.yaml",
        )


@pytest.mark.parametrize(
    "section",
    ("import", "physics", "planning_collision", "state_summary"),
)
def test_object_rejects_null_non_nullable_mapping_sections(section: str) -> None:
    profile = {
        "object": {
            "kind": "rigid",
            "source": "usd",
            "asset_path": "fixture.usd",
            section: None,
        },
    }

    with pytest.raises(
        ValueError,
        match=rf"object\.{section} must be a mapping",
    ):
        validate_object_profile(profile, source="null-section.yaml")


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
def test_object_rejects_nested_typos_with_complete_path(
    mutation: dict[str, object], path: str
) -> None:
    profile = load_yaml("configs/objects/TblockV1_default.yaml")
    if "object" in mutation:
        object_mutation = mutation["object"]
        assert isinstance(object_mutation, dict)
        for key, value in object_mutation.items():
            if isinstance(value, dict) and isinstance(profile["object"].get(key), dict):
                profile["object"][key].update(value)
            else:
                profile["object"][key] = value
    else:
        profile.update(mutation)

    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        validate_object_profile(profile, source="typo.yaml", profile_name="fixture")


@pytest.mark.parametrize(
    ("profile", "path"),
    (
        (
            {
                "object": {
                    "kind": "rigid",
                    "source": "usd",
                    "asset_path": "fixture.usd",
                    "physics": {"static": "true"},
                },
            },
            "object.physics.static",
        ),
        (
            {
                "object": {
                    "kind": "rigid",
                    "source": "usd",
                    "asset_path": "fixture.usd",
                    "physics": {"material": {"static_friction": "0.5"}},
                },
            },
            "object.physics.material.static_friction",
        ),
        (
            {
                "object": {
                    "kind": "rigid",
                    "source": "usd",
                    "asset_path": "fixture.usd",
                    "physics": {"material": {"restitution": 1.1}},
                },
            },
            "object.physics.material.restitution",
        ),
        (
            {
                "object": {
                    "kind": "dynamic_chain",
                    "source": "usd",
                    "asset_path": "rope.usd",
                    "root_path": "/Rope",
                    "state_summary": {"reference_body": "body"},
                    "physics": {"solver_position_iterations": 1.5},
                },
            },
            "object.physics.solver_position_iterations",
        ),
    ),
)
def test_object_uses_strict_types_and_ranges(
    profile: dict[str, object], path: str
) -> None:
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        validate_object_profile(profile, source="invalid.yaml")


@pytest.mark.parametrize(
    ("kind", "state_summary", "message"),
    [
        ("dynamic_chain", {"reference_body": "/Bodies/left"}, "body name"),
        ("dynamic_chain", {"position_reduction": "mean"}, "unsupported"),
        ("rigid", {"reference_body": "body"}, "only supported"),
    ],
)
def test_object_profile_rejects_invalid_state_summary(
    kind: str,
    state_summary: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ObjectProfileConfig.from_mapping(
            {
                "object": {
                    "name": "object",
                    "kind": kind,
                    "source": "usd",
                    "asset_path": "object.usda",
                    "state_summary": state_summary,
                },
            },
            profile_name="object",
        )


def test_dynamic_chain_profile_requires_reference_body() -> None:
    with pytest.raises(ValueError, match=r"state_summary\.reference_body is required"):
        ObjectProfileConfig.from_mapping(
            {
                "object": {
                    "name": "rope",
                    "kind": "dynamic_chain",
                    "source": "usd",
                    "asset_path": "rope.usda",
                },
            },
            profile_name="rope",
        )


def test_object_instances_reject_duplicate_identity_handle_and_path() -> None:
    with pytest.raises(ValueError, match="Duplicate object name"):
        object_scene_instances_from_env_config(
            {"objects": [_instance("fixture"), _instance("fixture")]}
        )

    with pytest.raises(ValueError, match="Duplicate object runtime_handle"):
        object_scene_instances_from_env_config(
            {
                "objects": [
                    _instance("fixture_a", runtime_handle="target"),
                    _instance("fixture_b", runtime_handle="target"),
                ]
            }
        )

    with pytest.raises(ValueError, match="Duplicate object prim path"):
        object_scene_instances_from_env_config(
            {
                "objects": [
                    _instance("fixture_a", prim_path="/World/Fixtures/shared"),
                    _instance("fixture_b", prim_path="/World/Fixtures/shared"),
                ]
            }
        )

    with pytest.raises(ValueError, match="runtime_handle.*conflicts"):
        object_scene_instances_from_env_config(
            {
                "objects": [
                    _instance("fixture_a", runtime_handle="fixture_b"),
                    _instance("fixture_b"),
                ]
            }
        )


@pytest.mark.parametrize("name", ["", "fixture-a", "fixtures/a", "two words"])
def test_object_instances_reject_names_that_cannot_form_prim_paths(name: str) -> None:
    with pytest.raises(ValueError, match=r"objects\[0\]\.name"):
        object_scene_instances_from_env_config({"objects": [_instance(name)]})


@pytest.mark.parametrize(
    "prim_path",
    [None, "World/Fixture", "", "/", "/World//Fixture", "/World/Fixture/"],
)
def test_object_instances_reject_invalid_explicit_paths(prim_path: str | None) -> None:
    item = _instance("fixture")
    item["prim_path"] = prim_path
    with pytest.raises(ValueError, match=r"objects\[0\]\.prim_path"):
        object_scene_instances_from_env_config({"objects": [item]})


def test_expanded_object_mapping_uses_instance_path() -> None:
    instance = object_scene_instances_from_env_config(
        {"objects": [_instance("fixture", prim_path="/World/Scene/Fixture")]}
    )[0]
    profile = _profile()

    expanded = expanded_object_mapping(instance, profile)

    assert expanded["prim_path"] == "/World/Scene/Fixture"


def test_runtime_object_keeps_resolved_path_separate_from_profile() -> None:
    runtime_object = runtime_objects_from_env_config(
        {
            "objects": [
                _instance(
                    "fixture",
                    profile="workstation_armbase",
                    prim_path="/World/Scene/Fixture",
                )
            ]
        }
    )[0]

    assert runtime_object.prim_path == "/World/Scene/Fixture"
