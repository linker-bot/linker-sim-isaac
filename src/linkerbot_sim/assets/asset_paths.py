"""AR5/L6 实验使用的默认资产路径和关节名。

路径统一以仓库根目录的 ``assets/`` 为基准，避免脚本入口、动作配置和测试用例到处
硬编码相同相对路径。这里不检查文件是否存在，因为有些测试只需要导入常量，真正的
资产可用性应在加载机器人或构建后端上下文时检查。

职责边界:
    * 保存仓库内默认 MJCF/URDF/XRDF/USD 路径。
    * 保存常用控制关节名元组，作为数组列顺序的单一来源。
    * 不解析 YAML，不加载资产，不根据左右手动态切换路径。

顺序约定非常重要：这些关节名元组同时被 IK 结果写回、控制器 joint group、轨迹矩阵
和配置校验引用。凡是从 cuMotion、控制器或配置文件得到的关节向量，都需要显式按同一
名称顺序对齐后再写入 Isaac DOF target，不能只依赖“看起来相同”的资产内部顺序。
"""

from __future__ import annotations

from linkerbot_sim.utils.paths import ASSETS_ROOT


DEFAULT_AR5_MJCF = ASSETS_ROOT / "single_system" / "arm" / "AR5V2_L" / "AR5V2_L.xml"
DEFAULT_AR5_URDF = ASSETS_ROOT / "single_system" / "arm" / "AR5V2_L" / "AR5V2_L.urdf"
DEFAULT_AR5_CUMOTION_XRDF = (
    ASSETS_ROOT / "single_system" / "arm" / "AR5V2_L" / "AR5V2_L.xrdf"
)
DEFAULT_AR5_L6_MJCF = (
    ASSETS_ROOT / "combined_system" / "AR5V2_L6V1_L" / "AR5V2_L6V1_L.xml"
)
DEFAULT_L6_MJCF = ASSETS_ROOT / "single_system" / "hand" / "L6V1_L" / "L6V1_L.xml"
DEFAULT_AR5_RIGHT_URDF = (
    ASSETS_ROOT / "single_system" / "arm" / "AR5V2_R" / "AR5V2_R.urdf"
)
DEFAULT_AR5_RIGHT_CUMOTION_XRDF = (
    ASSETS_ROOT / "single_system" / "arm" / "AR5V2_R" / "AR5V2_R.xrdf"
)
DEFAULT_AR5_DUAL_CUMOTION_URDF = (
    ASSETS_ROOT / "combined_system" / "AR5V2_DUAL" / "AR5V2_DUAL.urdf"
)
DEFAULT_AR5_DUAL_CUMOTION_XRDF = (
    ASSETS_ROOT / "combined_system" / "AR5V2_DUAL" / "AR5V2_DUAL.xrdf"
)
DEFAULT_L6_RIGHT_URDF = (
    ASSETS_ROOT / "single_system" / "hand" / "L6V1_R" / "L6V1_R.urdf"
)
DEFAULT_CAPSULE_ROPE_USD = (
    ASSETS_ROOT
    / "dynamic_env_objects"
    / "capsuleropeV1_default"
    / "capsuleropeV1_default.usda"
)
DEFAULT_WORKSTATION_V1_ARMBASE_URDF = (
    ASSETS_ROOT
    / "static_env_objects"
    / "workstationV1_armbase"
    / "workstationV1_armbase.urdf"
)
DEFAULT_WORKSTATION_V1_TABLEBASE_URDF = (
    ASSETS_ROOT
    / "static_env_objects"
    / "workstationV1_tablebase"
    / "workstationV1_tablebase.urdf"
)


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

DEFAULT_CONTROLLED_AR5_L6_JOINT_NAMES = (
    DEFAULT_ARM_JOINT_NAMES + DEFAULT_HAND_MASTER_JOINT_NAMES
)
