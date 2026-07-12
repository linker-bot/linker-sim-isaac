"""机器人状态、关节分组和 mimic 关系工具。

robots 子包不直接持有 Isaac articulation，而是处理控制层需要知道的静态/半静态信息：
关节状态快照、DOF 名称分组、arm/hand 分类以及 MJCF equality/mimic 映射。这样控制器和
动作脚本在进入 Isaac runtime 前就能校验 DOF 名称、配置长度和从动关节约定。入口文件保持
轻量，避免导入时读取大型资产文件。
"""

from linkerbot_sim.robots.capabilities import (
    PlanningBindingConfig,
    PlanningCapability,
    RobotKind,
    robot_kind_from_profile,
)
from linkerbot_sim.robots.joint_groups import JointGroup, JointGroupLayout

__all__ = [
    "JointGroup",
    "JointGroupLayout",
    "PlanningBindingConfig",
    "PlanningCapability",
    "RobotKind",
    "robot_kind_from_profile",
]
