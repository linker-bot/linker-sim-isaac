"""基础 Isaac 场景构建工具。

本模块放置“所有动作脚本都可能复用”的最小场景初始化逻辑，例如灯光、GUI viewport、
Isaac ``World``、物理步长和重力设置。机器人、绳体、桌面等具体对象不在这里
创建，而是由对应的 robot/env/object 模块负责，这样可以避免动作脚本里重复写
Omni/USD 初始化代码，也让基础世界和具体场景对象保持解耦。

注意:
    这里的函数会直接操作当前 Isaac Sim / Omni USD context，因此调用前通常需要
    已经通过 ``SimulationApp`` 或项目的 launch helper 启动 Isaac 运行时。
    本模块仅设置基础 world、灯光和地面，不加载机器人资产，便于不同动作脚本复用同一
    场景初始化流程。
"""

from __future__ import annotations

import math

from linkerbot_sim.envs.visual_settings import SceneVisualSettings


def configure_visuals(
    settings: SceneVisualSettings | None = None,
    *,
    configure_viewport: bool = True,
) -> None:
    """添加基础灯光并设置默认视角。

    该函数只负责“看得见”的基础视觉环境，不参与物理仿真参数设置：

    - 按配置创建主方向光，用来提供清晰的主体照明。
    - 按配置创建 DomeLight，用来补环境亮度，减少全黑阴影。
    - 按配置调整默认 perspective viewport 的观察位置。

    参数:
        settings: 来自 env profile 的可选视觉配置；为 ``None`` 时使用默认值。
        configure_viewport: 是否导入 viewport API 并设置观察视角；headless camera 只需灯光。
    返回:
        无返回值；副作用是创建灯光 prim 并设置 viewport。
    """

    settings = settings or SceneVisualSettings()

    # Isaac/Omni 相关模块放在函数内部导入，避免普通单元测试或文档工具 import
    # 本模块时就强依赖 Isaac Sim 运行时。
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    import omni.usd

    # 获取当前 USD stage。所有灯光 prim 都会写入这个 stage。
    stage = omni.usd.get_context().get_stage()

    if settings.key_light.enabled:
        # 主光：DistantLight 类似“无限远方向光”，适合给机械臂和物体提供稳定轮廓。
        key = UsdLux.DistantLight.Define(stage, Sdf.Path(settings.key_light.path))
        key.CreateIntensityAttr(float(settings.key_light.intensity))
        # angle 越大阴影越柔和；较小值会让模型边缘更清楚。
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
        # 补光：DomeLight 从环境方向整体补亮，避免背光面过暗。
        fill = UsdLux.DomeLight.Define(stage, Sdf.Path(settings.fill_light.path))
        fill.CreateIntensityAttr(float(settings.fill_light.intensity))
        if settings.fill_light.color is not None:
            fill.CreateColorAttr(Gf.Vec3f(*settings.fill_light.color))

    if configure_viewport and settings.viewport.enabled:
        # eye 是 viewport 观察位置，target 是视线目标点，单位与 stage 一致，此处为 m。
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=Gf.Vec3d(*settings.viewport.eye),
            target=Gf.Vec3d(*settings.viewport.target),
            camera_prim_path=settings.viewport.prim_path,
        )


def build_world(
    *,
    physics_dt: float | None,
    rendering_dt: float | None,
    gravity_z: float,
    add_ground: bool = True,
    ground_height: float = 0.0,
):
    """创建带可选默认地面和重力的 Isaac ``World``。

    ``World`` 是 Isaac Sim 高层仿真入口，内部持有 stage、physics context、scene
    对象管理器等运行时状态。这里集中创建 ``World``，可以让不同脚本共享一致的
    单位、步长和重力设置。

    参数:
        physics_dt: 物理步长，单位 s；为 ``None`` 时使用 Isaac 默认值。
        rendering_dt: 渲染步长，单位 s；为 ``None`` 时使用 Isaac 默认值。
        gravity_z: z 方向重力加速度，单位 m/s^2。
        add_ground: 是否添加 Isaac 默认地面。
        ground_height: Isaac 默认地面的 z 高度，单位 m；仅在 ``add_ground`` 为 true 时生效。
    返回:
        已设置重力并按需创建默认地面的 ``World`` 实例。
    """

    # 延迟导入 Isaac API，原因同 configure_visuals：让非 Isaac 环境也能 import 项目模块。
    from isaacsim.core.api.world import World

    # stage_units_in_meters=1.0 表示 USD stage 的 1 个长度单位等于 1 m。
    # physics_dt/rendering_dt 可由 env yaml 通过频率换算得到；传 None 时沿用 Isaac 默认配置。
    world = World(
        stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=rendering_dt
    )

    # 当前项目约定 z 轴向上，因此重力通常是负值，例如 -9.81。
    world.get_physics_context().set_gravity(float(gravity_z))

    # 添加 Isaac 默认地面，给机器人、绳体等对象提供基础接触面；桌面/工装场景可关闭。
    if add_ground:
        world.scene.add_default_ground_plane(z_position=float(ground_height))
    return world
