from __future__ import annotations

from manipulation_project.controllers.config import implicit_drive_settings, load_controller_profiles, physx_override_configs
from manipulation_project.robots.classification import component_for_name


def test_controller_profiles_split_arm_and_hand() -> None:
    profiles = load_controller_profiles("configs/controllers/implicit_position_drive.yaml")
    assert profiles.arm.name == "arm"
    assert profiles.hand.name == "hand"
    assert profiles.arm.velocity_control
    assert profiles.hand.effort_control

    drive = implicit_drive_settings(profiles)
    assert drive.component("AR5V2_L_arm_joint_1").stiffness == (1000.0,)
    assert drive.component("L6V1_L_hand_index_mcp_pitch").follower_stiffness == (50000.0,)

    physx = physx_override_configs(profiles)
    assert set(physx) == {"default", "arm", "hand"}
    assert physx["hand"].follower_drive_stiffness_seed == 50000.0


def test_robot_name_classification_supports_left_and_right() -> None:
    assert component_for_name("AR5V2_L_arm_link1") == "arm"
    assert component_for_name("AR5V2_R_arm_link1") == "arm"
    assert component_for_name("L6V1_L_hand_index_mcp_pitch") == "hand"
    assert component_for_name("L6V1_R_hand_index_mcp_pitch") == "hand"
    assert component_for_name("world") == "default"
