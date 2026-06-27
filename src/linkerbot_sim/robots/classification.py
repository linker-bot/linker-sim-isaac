"""机器人部件分类工具。

本模块按资产命名规范中的 ``<single-system-name>_<category>_<local-name>`` 格式把关节/刚体
名称分组。``category`` 使用 ``arm``、``hand`` 等稳定字段，因此控制器、USD/PhysX 覆盖和
solver 设置不需要绑定具体设备型号。

职责边界:
    * 只基于名称 token 做轻量分类，不读取资产文件或配置。
    * 只把 ``arm``/``hand`` 作为当前控制参数分组，其它已知 token 仍会回退到 ``default``。
    * 未知或不符合规范的名称返回 ``default``，避免第三方 USD prim 或临时调试对象破坏导入流程。
"""

from __future__ import annotations


KNOWN_COMPONENTS = frozenset({"arm", "hand", "gripper", "sensor", "tool", "base"})


def component_token_from_name(name: str) -> str | None:
    """从规范实体名中提取部件字段。

    命名规范要求内部实体名形如
    ``<single-system-name>_<category>_<local-name>``，其中单体系统名自身可能
    带侧别字段。因此这里从左到右查找已知 category token，而不是匹配具体
    设备型号前缀。
    """

    # 从左到右查找已知 category，而不是假设型号前缀长度固定；这样 AR5V2_L、L6V1_L 等
    # 不同系统名都能复用同一分类逻辑。
    tokens = [token for token in str(name).split("_") if token]
    for token in tokens:
        if token in KNOWN_COMPONENTS:
            return token
    return None


def is_arm_name(name: str) -> bool:
    """判断名称是否属于机械臂。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        名称是否包含规范 category ``arm``。
    """

    return component_token_from_name(name) == "arm"


def is_hand_name(name: str) -> bool:
    """判断名称是否属于灵巧手。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        名称是否包含规范 category ``hand``。
    """

    return component_token_from_name(name) == "hand"


def component_for_name(name: str) -> str:
    """按名称返回 ``arm``、``hand`` 或 ``default``。

    参数:
        name: USD prim 名或 articulation DOF 名。
    返回:
        分类字符串；未知名称返回 ``default``，调用方可使用回退参数。
    """

    component = component_token_from_name(name)
    return component if component in {"arm", "hand"} else "default"
