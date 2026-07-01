"""可复用仿真对象运行时工具。

objects 子包描述机器人以外的场景对象，例如 capsule rope、端点 cuboid 或未来的夹具/障碍物。
这些模块负责把 YAML 参数转换成运行时配置、引用已生成资产，并应用运行时物理覆盖。

入口文件保持轻量：导入本包不会写 USD 文件、不会访问当前 stage，也不会创建 Isaac world。
USD 资产生成属于离线工具层，不放在 ``src/linkerbot_sim`` runtime 包中。
"""

from linkerbot_sim.objects.config import (
    ObjectProfileConfig,
    ObjectSceneInstanceConfig,
    expanded_object_mapping,
    object_scene_instances_from_env_config,
)
from linkerbot_sim.objects.physics import ObjectMaterialConfig
from linkerbot_sim.objects.dynamic_chain.capsule_rope import (
    CapsuleRopeConfig,
    apply_capsule_rope_runtime_physics,
    add_capsule_rope_reference,
)
from linkerbot_sim.objects.rigid.runtime import (
    AddedRigidObject,
    RigidObjectConfig,
    RigidObjectMaterialConfig,
    RigidObjectPhysicsConfig,
    add_rigid_objects,
    rigid_objects_from_env_config,
)

__all__ = [
    "AddedRigidObject",
    "CapsuleRopeConfig",
    "ObjectMaterialConfig",
    "ObjectProfileConfig",
    "ObjectSceneInstanceConfig",
    "RigidObjectConfig",
    "RigidObjectMaterialConfig",
    "RigidObjectPhysicsConfig",
    "apply_capsule_rope_runtime_physics",
    "add_capsule_rope_reference",
    "add_rigid_objects",
    "expanded_object_mapping",
    "object_scene_instances_from_env_config",
    "rigid_objects_from_env_config",
]
