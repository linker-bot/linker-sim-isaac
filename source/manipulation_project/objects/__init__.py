"""可复用仿真对象资产与对象生成工具。

objects 子包描述机器人以外的场景对象，例如 capsule rope、端点 box 或未来的夹具/障碍物。
这些模块负责把 YAML 参数转换成 USD prim、材料和 PhysX 关节设置；场景构建层只负责在合适
位置引用它们。

入口文件保持轻量：导入本包不会写 USD 文件、不会访问当前 stage，也不会创建 Isaac world。
真正的资产生成和引用动作由 ``write_capsule_rope_asset``、``add_capsule_rope_reference`` 等
显式函数触发，便于脚本和测试精确控制副作用。
"""

from manipulation_project.objects.capsule_rope import (
    CapsuleRopeConfig,
    add_capsule_rope_reference,
    create_rope_model,
    endpoint_center,
    write_capsule_rope_asset,
)

__all__ = [
    "CapsuleRopeConfig",
    "add_capsule_rope_reference",
    "create_rope_model",
    "endpoint_center",
    "write_capsule_rope_asset",
]
