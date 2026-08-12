"""严格 :class:`IsaacSessionSpec` 到具体 :class:`PhysicsRuntime` 的唯一工厂。"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from linkerbot_sim.isaac.physics.manager import install_physics_manager
from linkerbot_sim.isaac.physics.physx import PhysxRuntime
from linkerbot_sim.isaac.physics.runtime import PhysicsRuntime
from linkerbot_sim.isaac.spec import (
    IsaacNewtonCpuSpec,
    IsaacNewtonCudaSpec,
    IsaacPhysxCpuSpec,
    IsaacPhysxCudaSpec,
    IsaacSessionSpec,
)


def create_physics_runtime(
    *,
    app: object,
    stage: object,
    spec: IsaacSessionSpec,
    world_builder: Callable[..., object],
    fabric_output_configurer: Callable[..., Mapping[str, bool]],
) -> PhysicsRuntime:
    """构造规格声明的唯一物理 owner，并在失败时关闭半成品。

    factory 只有两个 engine 分支：PhysX CPU/CUDA 由 ``PhysxRuntime`` 独占 Isaac ``World``；
    Newton CPU/CUDA 由 ``NewtonRuntime`` 独占 model/state/control/solver。不存在
    extension-owned Newton、默认 backend 或参数形状 fallback。
    """

    if not isinstance(spec, IsaacSessionSpec):
        raise TypeError("spec must be IsaacSessionSpec")
    physics = spec.physics
    if isinstance(physics, (IsaacNewtonCpuSpec, IsaacNewtonCudaSpec)):
        from linkerbot_sim.isaac.physics.newton.manager import (
            NewtonRuntime,
        )

        runtime = NewtonRuntime(
            stage=stage,
            physics_spec=physics,
            device=spec.physics_device,
            physics_dt=spec.physics_dt,
            rendering_dt=spec.rendering_dt,
            gravity_z=spec.gravity_z,
            add_ground=spec.add_ground,
            ground_height=spec.ground_height,
            rendering_enabled=spec.render.enabled,
            render_callback=(app.update if spec.render.enabled else None),
            render_world_indices=spec.render.visible_world_indices,
        )
        try:
            install_physics_manager(runtime)
        except BaseException:
            # install 失败表示另一个 session 已拥有进程级 Newton view registry。新 runtime
            # 尚未暴露，必须自行释放，且绝不能关闭 registry 中身份不同的旧 owner。
            runtime.close()
            raise
        return runtime

    assert isinstance(physics, (IsaacPhysxCpuSpec, IsaacPhysxCudaSpec))
    fabric_outputs: Mapping[str, bool] | None = None
    if isinstance(physics, IsaacPhysxCudaSpec):
        fabric_outputs = fabric_output_configurer(
            rendering_required=spec.render.enabled
        )
    world = world_builder(spec=spec, fabric_outputs=fabric_outputs)
    return PhysxRuntime(world, kind=physics.kind)


__all__ = ["create_physics_runtime"]
