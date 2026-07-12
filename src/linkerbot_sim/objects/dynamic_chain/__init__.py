"""动态链式对象的 runtime 实现入口。

当前动态链对象主要是 capsule rope。这里仅导出配置、USD reference 添加和运行期物理覆盖
函数；具体 PhysX/material 细节留在 ``capsule_rope`` 模块中，避免对象注册入口过重。
"""

from linkerbot_sim.objects.dynamic_chain.capsule_rope import (
    CapsuleRopeConfig,
    add_capsule_rope_reference,
    apply_capsule_rope_runtime_physics,
)

# 保持导出列表精确，便于 env object registry 只依赖稳定对象入口。
__all__ = [
    "CapsuleRopeConfig",
    "add_capsule_rope_reference",
    "apply_capsule_rope_runtime_physics",
]
