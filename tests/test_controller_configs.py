from __future__ import annotations

from linkerbot_sim.controllers.config import (
    ControllerProfile,
    _profile_from_mapping,
    joint_control_settings,
    load_controller_profiles,
    physx_override_configs,
)
from linkerbot_sim.robots.classification import component_for_name


def test_controller_profiles_split_arm_and_hand() -> None:
    profiles = load_controller_profiles("configs/controllers")
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


def test_controller_profiles_require_directory_entrypoint() -> None:
    try:
        load_controller_profiles(
            {
                "arm": "configs/controllers/arm_controller.yaml",
                "hand": "configs/controllers/hand_controller.yaml",
            }
        )
    except TypeError:
        pass
    else:
        raise AssertionError("load_controller_profiles accepted mapping entrypoint")


def test_controller_profiles_reject_robot_physx_fields() -> None:
    profile = {
        "target": "arm",
        "position_control": {"method": "implicit"},
        "physx": {"material": {"contact_static_friction": 0.9}},
    }

    try:
        _profile_from_mapping("arm", profile)
    except ValueError as exc:
        assert "move them to robot.physics.physx" in str(exc)
    else:
        raise AssertionError("controller accepted robot PhysX material fields")


def test_controller_profiles_reject_removed_method_aliases() -> None:
    profile = ControllerProfile(
        name="arm",
        position_control={"type": "implicit"},
        velocity_control={},
        effort_control={},
        physx={},
    )
    profiles = load_controller_profiles("configs/controllers")
    patched_profiles = profiles.__class__(arm=profile, hand=profiles.hand)

    try:
        joint_control_settings(patched_profiles, mode="position")
    except ValueError as exc:
        assert "removed field 'type'" in str(exc)
    else:
        raise AssertionError("controller accepted removed type field")

    profile = ControllerProfile(
        name="arm",
        position_control={"method": "implicit_drive"},
        velocity_control={},
        effort_control={},
        physx={},
    )
    patched_profiles = profiles.__class__(arm=profile, hand=profiles.hand)
    try:
        joint_control_settings(patched_profiles, mode="position")
    except ValueError as exc:
        assert "method must be one of" in str(exc)
    else:
        raise AssertionError("controller accepted removed implicit_drive alias")


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
