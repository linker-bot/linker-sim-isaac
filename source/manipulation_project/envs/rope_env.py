"""绳体场景扩展点。

绳体对象的创建和 USD 资产生成已拆到 ``manipulation_project.objects.capsule_rope``。
本模块只保留环境层入口，并通过别名兼容旧导入路径。
"""

from __future__ import annotations

from manipulation_project.envs.base_env import BaseEnv
from manipulation_project.objects.capsule_rope import (
    CapsuleRopeConfig,
    add_capsule_rope_reference,
    create_rope_model,
    endpoint_center,
    write_capsule_rope_asset,
)


class RopeEnv(BaseEnv):
    """绳体场景扩展点。

    输入:
        当前没有构造参数。
    输出:
        后续可封装 rope reset、观测和场景级随机化接口。
    """

    pass


# 兼容旧代码中 ``RopeSceneConfig`` / ``create_rope_model`` 的导入路径。
RopeSceneConfig = CapsuleRopeConfig

__all__ = [
    "RopeEnv",
    "RopeSceneConfig",
    "CapsuleRopeConfig",
    "add_capsule_rope_reference",
    "create_rope_model",
    "endpoint_center",
    "write_capsule_rope_asset",
]
