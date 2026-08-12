"""Rigid object 配置与 stage importer 公共入口。

本包负责把 env YAML 中的 rigid object 配置转成 USD 引用、物理属性和规划碰撞描述。这里
只汇总稳定的对外类型与函数，具体 stage 写入逻辑由 ``importer`` 模块所有。
"""

from linkerbot_sim.objects.rigid.config import RigidObjectConfig
from linkerbot_sim.objects.rigid.importer import AddedRigidObject, add_rigid_objects

# 对象系统通过这些名称连接 env config、Isaac 场景构建和 cuRobo collision world。
__all__ = [
    "AddedRigidObject",
    "RigidObjectConfig",
    "add_rigid_objects",
]
