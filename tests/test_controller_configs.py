from __future__ import annotations

from pathlib import Path

import pytest

from linkerbot_sim.app.runtime.single_scene_runtime import (
    _load_controller_profiles_cached,
)
from linkerbot_sim.controllers.config import (
    ControllerProfile,
    ControllerProfiles,
    _profile_from_mapping,
    joint_control_settings,
    load_controller_bundle,
    load_controller_profiles,
    physx_override_configs,
)
from linkerbot_sim.robots.classification import (
    RobotComponentMapping,
    component_for_name,
)
from linkerbot_sim.utils.config import load_yaml


def test_controller_profiles_split_arm_and_hand() -> None:
    profiles = load_controller_profiles("configs/controllers/default")
    assert profiles.arm.name == "arm"
    assert profiles.hand.name == "hand"
    assert profiles.arm.position_control["method"] == "implicit"
    assert profiles.arm.velocity_control
    assert profiles.hand.effort_control

    position = joint_control_settings(profiles, mode="position")
    assert position.component("AR5V2_L_arm_joint_1").mode == "position"
    assert position.component("AR5V2_L_arm_joint_1").method == "implicit"
    assert position.component("AR5V2_L_arm_joint_1").stiffness == (1000.0,)
    assert position.component("L6V1_L_hand_index_mcp_pitch").follower_stiffness == (
        50000.0,
    )

    velocity = joint_control_settings(profiles, mode="velocity")
    assert velocity.component("AR5V2_L_arm_joint_1").mode == "velocity"
    assert velocity.component("AR5V2_L_arm_joint_1").method == "explicit"
    assert velocity.component("L6V1_L_hand_index_mcp_pitch").follower_stiffness == (
        50000.0,
    )
    assert velocity.component("L6V1_L_hand_index_mcp_pitch").follower_damping == (40.0,)

    effort = joint_control_settings(profiles, mode="effort")
    assert effort.component("L6V1_L_hand_index_mcp_pitch").mode == "effort"
    assert effort.component("L6V1_L_hand_index_mcp_pitch").method == "direct"
    assert effort.component("L6V1_L_hand_index_mcp_pitch").effort_limit == 100.0
    assert effort.component("L6V1_L_hand_index_mcp_pitch").follower_stiffness == (
        50000.0,
    )
    assert effort.component("L6V1_L_hand_index_mcp_pitch").follower_damping == (40.0,)

    physx = physx_override_configs(profiles)
    assert set(physx) == {"default", "arm", "hand"}
    assert physx["hand"].follower_drive_stiffness_seed == 50000.0


def _write_controller_profile(
    path: Path, *, target: str, stiffness: float = 1000.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"target: {target}",
                "position_control:",
                "  method: implicit",
                "  active_joints:",
                f"    stiffness: {stiffness}",
            )
        ),
        encoding="utf-8",
    )


def test_controller_bundle_loads_canonical_default_directory() -> None:
    profiles = load_controller_bundle("default")

    assert profiles.arm.name == "arm"
    assert profiles.hand.name == "hand"


def test_controller_bundle_supports_optional_default_component(tmp_path: Path) -> None:
    bundle = tmp_path / "lab"
    _write_controller_profile(bundle / "arm_controller.yaml", target="arm")
    _write_controller_profile(bundle / "hand_controller.yaml", target="hand")
    _write_controller_profile(
        bundle / "default_controller.yaml",
        target="default",
        stiffness=321.0,
    )

    profiles = load_controller_bundle("lab", controllers_root=tmp_path)

    assert profiles.default is not None
    settings = joint_control_settings(profiles)
    assert settings.default.stiffness == (321.0,)
    assert physx_override_configs(profiles)["default"].drive_stiffness_seed == 321.0


@pytest.mark.parametrize("name", ("../default", "a/b", r"a\b", "", "."))
def test_controller_bundle_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="controller bundle name"):
        load_controller_bundle(name, controllers_root=tmp_path)


@pytest.mark.parametrize("field", ("typo", "physx", "implicit_position_drive"))
def test_controller_profile_rejects_unknown_top_level_fields(field: str) -> None:
    with pytest.raises(ValueError, match=rf"controller\[arm\]\.{field}"):
        _profile_from_mapping(
            "arm",
            {"target": "arm", field: {}},
        )


def test_scene_controller_bundle_cache_preserves_robot_order() -> None:
    calls: list[str] = []
    sentinels = {"shared": object(), "other": object()}

    def loader(name: str):
        calls.append(name)
        return sentinels[name]

    resolved = _load_controller_profiles_cached(
        ("shared", "other", "shared"), loader=loader
    )

    assert calls == ["shared", "other"]
    assert resolved == (
        sentinels["shared"],
        sentinels["other"],
        sentinels["shared"],
    )


def test_controller_profiles_require_directory_entrypoint() -> None:
    try:
        load_controller_profiles(
            {
                "arm": "configs/controllers/default/arm_controller.yaml",
                "hand": "configs/controllers/default/hand_controller.yaml",
            }
        )
    except TypeError:
        pass
    else:
        raise AssertionError("load_controller_profiles accepted mapping entrypoint")


def test_controller_profiles_reject_unknown_control_fields_and_methods() -> None:
    profile = ControllerProfile(
        name="arm",
        position_control={"type": "implicit"},
        velocity_control={},
        effort_control={},
    )
    profiles = load_controller_profiles("configs/controllers/default")
    patched_profiles = profiles.__class__(arm=profile, hand=profiles.hand)

    try:
        joint_control_settings(patched_profiles, mode="position")
    except ValueError as exc:
        assert "controller[arm].position_control.type" in str(exc)
    else:
        raise AssertionError("controller accepted unknown type field")

    profile = ControllerProfile(
        name="arm",
        position_control={"method": "implicit_drive"},
        velocity_control={},
        effort_control={},
    )
    patched_profiles = profiles.__class__(arm=profile, hand=profiles.hand)
    try:
        joint_control_settings(patched_profiles, mode="position")
    except ValueError as exc:
        assert "method must be one of" in str(exc)
    else:
        raise AssertionError("controller accepted unknown control method")


def test_robot_name_classification_supports_left_and_right() -> None:
    assert component_for_name("AR5V2_L_arm_link1") == "arm"
    assert component_for_name("AR5V2_R_arm_link1") == "arm"
    assert component_for_name("L6V1_L_hand_index_mcp_pitch") == "hand"
    assert component_for_name("L6V1_R_hand_index_mcp_pitch") == "hand"
    assert component_for_name("world") == "default"


def test_robot_name_classification_uses_category_token_not_device_prefix() -> None:
    assert component_for_name("UR10V3_R_arm_joint_2") == "arm"
    assert component_for_name("DexHandV2_L_hand_index_mcp") == "hand"
    assert component_for_name("mobilebaseV1_base_link") == "default"


def test_exact_component_groups_override_nonstandard_names_and_validate_ownership() -> (
    None
):
    mapping = RobotComponentMapping.from_profile(
        {
            "joint_groups": {
                "arm": ["axis_a"],
                "hand": ["axis_b"],
                "passive": ["axis_free"],
            },
            "rigid_body_groups": {
                "arm": ["body_a"],
                "hand": ["body_b"],
                "default": ["mount"],
            },
        }
    )

    assert mapping.joint_component("axis_a") == "arm"
    assert mapping.joint_component("axis_b") == "hand"
    assert mapping.joint_component("axis_free") == "default"
    assert mapping.rigid_body_component("body_b") == "hand"
    assert mapping.joint_component("fixture_arm_joint") == "arm"

    with pytest.raises(ValueError, match="multiple components"):
        RobotComponentMapping.from_profile(
            {"joint_groups": {"arm": ["same"], "hand": ["same"]}}
        )
    with pytest.raises(ValueError, match="unsupported keys: typo"):
        RobotComponentMapping.from_profile(
            {"rigid_body_groups": {"arm": [], "typo": []}}
        )


def test_controller_parameters_accept_scalar_sequence_and_exact_name_map() -> None:
    profile = ControllerProfile(
        name="arm",
        position_control={
            "method": "implicit",
            "active_joints": {
                "stiffness": [11.0, 12.0],
                "damping": {"axis_a": 1.0, "axis_b": 2.0},
                "max_force": 9.0,
            },
        },
        velocity_control={},
        effort_control={},
    )
    profiles = load_controller_profiles("configs/controllers/default")
    settings = joint_control_settings(
        profiles.__class__(arm=profile, hand=profiles.hand), mode="position"
    ).arm

    assert settings is not None
    assert settings.stiffness == (11.0, 12.0)
    assert settings.damping == {"axis_a": 1.0, "axis_b": 2.0}
    assert settings.max_force == 9.0


def test_controller_parameters_reject_unknown_fields_and_illegal_values() -> None:
    profiles = load_controller_profiles("configs/controllers/default")
    for active_joints, message in (
        ({"stifness": 1.0}, "unsupported keys: stifness"),
        ({"stiffness": -1.0}, "finite and non-negative"),
        ({"stiffness": []}, "cannot be empty"),
    ):
        profile = ControllerProfile(
            name="arm",
            position_control={
                "method": "implicit",
                "active_joints": active_joints,
            },
            velocity_control={},
            effort_control={},
        )
        with pytest.raises(ValueError, match=message):
            joint_control_settings(
                profiles.__class__(arm=profile, hand=profiles.hand), mode="position"
            )


def test_all_bundled_controller_yaml_loads_strictly() -> None:
    paths = sorted(Path("configs/controllers").glob("**/*.yaml"))

    assert paths
    for path in paths:
        data = load_yaml(path)
        target = data.get("target")
        assert isinstance(target, str)
        profile = _profile_from_mapping(target, data, source_path=path)
        for mode in ("position", "velocity", "effort"):
            settings = joint_control_settings(
                ControllerProfiles(arm=profile, hand=profile),
                mode=mode,
            )
            assert settings.arm is not None
            assert settings.arm.mode == mode


def test_controller_nested_typo_reports_complete_path_eagerly() -> None:
    with pytest.raises(
        ValueError,
        match=r"controller\[arm\]\.position_control\.active_joints\.stifness",
    ):
        _profile_from_mapping(
            "arm",
            {
                "target": "arm",
                "position_control": {"active_joints": {"stifness": 100.0}},
            },
        )


@pytest.mark.parametrize(
    ("field", "value", "path"),
    (
        ("target", 1, "controller[arm].target"),
        ("method", False, "position_control.method"),
        ("stiffness", True, "active_joints.stiffness"),
        ("stiffness", -1.0, "active_joints.stiffness"),
    ),
)
def test_controller_rejects_coerced_types_and_invalid_ranges(
    field: str,
    value: object,
    path: str,
) -> None:
    profile: dict[str, object] = {
        "target": "arm",
        "position_control": {
            "method": "implicit",
            "active_joints": {"stiffness": 100.0},
        },
    }
    if field == "target":
        profile[field] = value
    elif field == "method":
        position = profile["position_control"]
        assert isinstance(position, dict)
        position[field] = value
    else:
        position = profile["position_control"]
        assert isinstance(position, dict)
        active = position["active_joints"]
        assert isinstance(active, dict)
        active[field] = value

    with pytest.raises(ValueError, match=path.replace("[", r"\[").replace("]", r"\]")):
        _profile_from_mapping("arm", profile)


def test_controller_unknown_field_reports_source_context() -> None:
    source = "/tmp/controller.yaml"
    with pytest.raises(ValueError, match=r"controller\[arm\]\.typo") as exc_info:
        _profile_from_mapping(
            "arm", {"target": "arm", "typo": True}, source_path=source
        )

    assert source in str(exc_info.value)
