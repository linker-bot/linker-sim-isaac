"""机器人部件分类工具。

本模块按关节/刚体名称把 AR5 机械臂和 LinkerHand L6 灵巧手分组。
这些规则被 controller、USD/PhysX 覆盖和 solver 设置复用，避免不同模块各自
维护一套前缀判断。
"""

from __future__ import annotations


ARM_NAME_PREFIXES = ("AR5V2_L_arm_", "AR5V2_R_arm_")
HAND_NAME_PREFIXES = ("L6V1_L_hand_", "L6V1_R_hand_")


def is_arm_name(name: str) -> bool:
    """判断名称是否属于 AR5 机械臂。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        名称是否匹配已知 AR5 左/右臂前缀。
    """

    return any(str(name).startswith(prefix) for prefix in ARM_NAME_PREFIXES)


def is_hand_name(name: str) -> bool:
    """判断名称是否属于 LinkerHand L6 灵巧手。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        名称是否匹配已知 L6 左/右手前缀。
    """

    return any(str(name).startswith(prefix) for prefix in HAND_NAME_PREFIXES)


def component_for_name(name: str) -> str:
    """按名称返回 ``arm``、``hand`` 或 ``default``。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        分类字符串；未知名称返回 ``default``，调用方可使用回退参数。
    """

    if is_arm_name(name):
        return "arm"
    if is_hand_name(name):
        return "hand"
    return "default"
