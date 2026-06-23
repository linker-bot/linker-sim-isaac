"""AR5/L6 实验使用的默认资产路径和关节名。

路径统一以仓库根目录的 assets/ 为基准，避免脚本入口到处硬编码。
关节名常量用于 IK 结果写回、控制器 joint group 和配置校验。
"""

from __future__ import annotations

from manipulation_project.utils.paths import ASSETS_ROOT


DEFAULT_AR5_MJCF = ASSETS_ROOT / "single_system" / "arm" / "AR5V2_L" / "AR5V2_L.xml"
DEFAULT_AR5_URDF = ASSETS_ROOT / "single_system" / "arm" / "AR5V2_L" / "AR5V2_L.urdf"
DEFAULT_AR5_LULA_DESCRIPTION = (
    ASSETS_ROOT / "single_system" / "arm" / "AR5V2_L" / "AR5V2_L_lula_robot_description.yaml"
)
DEFAULT_AR5_CUMOTION_XRDF = ASSETS_ROOT / "single_system" / "arm" / "AR5V2_L" / "AR5V2_L.xrdf"
DEFAULT_AR5_L6_MJCF = ASSETS_ROOT / "combined_system" / "AR5V2_L6V1_L" / "AR5V2_L6V1_L.xml"
DEFAULT_L6_MJCF = ASSETS_ROOT / "single_system" / "hand" / "L6V1_L" / "L6V1_L.xml"
DEFAULT_AR5_RIGHT_URDF = ASSETS_ROOT / "single_system" / "arm" / "AR5V2_R" / "AR5V2_R.urdf"
DEFAULT_L6_RIGHT_URDF = ASSETS_ROOT / "single_system" / "hand" / "L6V1_R" / "L6V1_R.urdf"
DEFAULT_CAPSULE_ROPE_USD = ASSETS_ROOT / "dynamic_env_objects" / "capsuleropeV1_default" / "capsuleropeV1_default.usda"


DEFAULT_ARM_JOINT_NAMES = (
    "AR5V2_L_arm_joint_1",
    "AR5V2_L_arm_joint_2",
    "AR5V2_L_arm_joint_3",
    "AR5V2_L_arm_joint_4",
    "AR5V2_L_arm_joint_5",
    "AR5V2_L_arm_joint_6",
    "AR5V2_L_arm_joint_7",
)

DEFAULT_HAND_MASTER_JOINT_NAMES = (
    "L6V1_L_hand_thumb_cmc_roll",
    "L6V1_L_hand_thumb_cmc_pitch",
    "L6V1_L_hand_index_mcp_pitch",
    "L6V1_L_hand_middle_mcp_pitch",
    "L6V1_L_hand_ring_mcp_pitch",
    "L6V1_L_hand_pinky_mcp_pitch",
)

DEFAULT_CONTROLLED_AR5_L6_JOINT_NAMES = DEFAULT_ARM_JOINT_NAMES + DEFAULT_HAND_MASTER_JOINT_NAMES
