"""Isaac ``World``、视觉 prim 与场景级物理参数的基础设施边界。

``build_physx_world`` 只构造 PhysX owner；Newton 在自己的 manager 中从 USD stage
finalize model，不制造假的 Isaac ``World``。函数内延迟导入 Isaac/Omni，保证纯配置进程可
安全 import 本模块。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from linkerbot_sim.configuration.scenes import SceneVisualSettings
from linkerbot_sim.isaac.physics.physx_pipeline import (
    build_physx_world_kwargs,
    probe_physx_tensor_pipeline,
)
from linkerbot_sim.isaac.spec import (
    IsaacPhysxCpuSpec,
    IsaacPhysxCudaSpec,
    IsaacSessionSpec,
)


def configure_visuals(
    settings: SceneVisualSettings | None = None,
    *,
    configure_viewport: bool = True,
) -> None:
    """按显式视觉设置创建灯光，并可选配置默认 viewport。"""

    settings = settings or SceneVisualSettings()
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage while configuring visuals")

    if settings.key_light.enabled:
        key = UsdLux.DistantLight.Define(stage, Sdf.Path(settings.key_light.path))
        key.CreateIntensityAttr(float(settings.key_light.intensity))
        key.CreateAngleAttr(float(settings.key_light.angle))
        if settings.key_light.color is not None:
            key.CreateColorAttr(Gf.Vec3f(*settings.key_light.color))
        if settings.key_light.rotation_rpy is not None:
            rotation_deg = tuple(
                math.degrees(value) for value in settings.key_light.rotation_rpy
            )
            UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(
                Gf.Vec3f(*rotation_deg)
            )

    if settings.fill_light.enabled:
        fill = UsdLux.DomeLight.Define(stage, Sdf.Path(settings.fill_light.path))
        fill.CreateIntensityAttr(float(settings.fill_light.intensity))
        if settings.fill_light.color is not None:
            fill.CreateColorAttr(Gf.Vec3f(*settings.fill_light.color))

    if configure_viewport and settings.viewport.enabled:
        from linkerbot_sim.visualization.viewport import set_camera_view

        set_camera_view(
            eye=settings.viewport.eye,
            target=settings.viewport.target,
            camera_prim_path=settings.viewport.prim_path,
        )


def build_physx_world(
    *,
    spec: IsaacSessionSpec,
    fabric_outputs: Mapping[str, bool] | None = None,
) -> object:
    """从完整 session 规格创建并验收唯一 Isaac PhysX ``World``。

    CUDA 分支在创建 PhysicsScene 前写入 device/Fabric 参数；引擎容量不属于项目配置面。
    创建后只回读 runtime state。CPU 分支不携带 GPU 参数；两条路径都在 World 上设置重力和
    可选地面。
    """

    if not isinstance(spec, IsaacSessionSpec):
        raise TypeError("spec must be IsaacSessionSpec")
    physics = spec.physics
    if not isinstance(physics, (IsaacPhysxCpuSpec, IsaacPhysxCudaSpec)):
        raise TypeError("build_physx_world requires a PhysX session specification")

    from isaacsim.core.api.world import World

    world_kwargs: dict[str, object] = {
        "stage_units_in_meters": 1.0,
        "physics_dt": spec.physics_dt,
        "rendering_dt": spec.rendering_dt,
    }
    world_kwargs.update(build_physx_world_kwargs(spec))
    world = World(**world_kwargs)
    world.get_physics_context().set_gravity(float(spec.gravity_z))
    probe_physx_tensor_pipeline(
        world,
        spec,
        fabric_outputs=fabric_outputs,
    )
    if spec.add_ground:
        world.scene.add_default_ground_plane(z_position=float(spec.ground_height))
    return world


def set_physics_gravity(runtime: object, gravity_z: float) -> None:
    """在 reset 后通过 concrete runtime 恢复重力，不探测全局 engine。

    PhysX 的权威对象是 runtime 持有的 ``World/PhysicsContext``；Newton 的权威对象
    是 manager 持有的 Newton model。这里按 runtime identity 分派，禁止从 SimulationManager
    猜测 owner，也不接受旧 ``world.physics_manager`` facade。
    """

    backend = getattr(runtime, "backend", None)
    execution = getattr(runtime, "execution", None)
    if backend == "physx":
        world = getattr(runtime, "world", None)
        if world is None:
            raise TypeError("PhysX runtime must own a World")
        world.get_physics_context().set_gravity(float(gravity_z))
        return
    if backend == "newton" and execution in {"cpu", "cuda"}:
        setter = getattr(runtime, "set_gravity", None)
        if not callable(setter):
            raise TypeError("Newton runtime must implement set_gravity()")
        setter(float(gravity_z))
        return
    raise TypeError(
        "unsupported physics runtime while setting gravity: "
        f"backend={backend!r}, execution={execution!r}"
    )


__all__ = ["build_physx_world", "configure_visuals", "set_physics_gravity"]
