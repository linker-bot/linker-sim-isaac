"""基础 Isaac 场景构建工具。

本模块放置“所有动作脚本都可能复用”的最小场景初始化逻辑，例如灯光、相机、
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


def configure_visuals() -> None:
    """添加基础灯光并设置默认视角。

    该函数只负责“看得见”的基础视觉环境，不参与物理仿真参数设置：

    - 在 ``/World/KeyLight`` 创建一盏主方向光，用来提供清晰的主体照明。
    - 在 ``/World/FillLight`` 创建一盏 DomeLight，用来补环境亮度，减少全黑阴影。
    - 调整默认 perspective viewport 的相机位置，使机器人和绳体区域更容易被看到。

    参数:
        无，直接操作当前 Omni/USD context。
    返回:
        无返回值；副作用是创建 ``/World/KeyLight``、``/World/FillLight`` 并设置 viewport。
    """

    # Isaac/Omni 相关模块放在函数内部导入，避免普通单元测试或文档工具 import
    # 本模块时就强依赖 Isaac Sim 运行时。
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Gf, Sdf, UsdLux
    import omni.usd

    # 获取当前 USD stage。所有灯光 prim 都会写入这个 stage。
    stage = omni.usd.get_context().get_stage()

    # 主光：DistantLight 类似“无限远方向光”，适合给机械臂和物体提供稳定轮廓。
    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    key.CreateIntensityAttr(1200.0)
    # angle 越大阴影越柔和；这里保持较小值，让模型边缘仍然清楚。
    key.CreateAngleAttr(0.5)

    # 补光：DomeLight 从环境方向整体补亮，避免背光面过暗。
    fill = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/FillLight"))
    fill.CreateIntensityAttr(250.0)

    # 设置默认视角。eye 是相机位置，target 是视线目标点，单位与 stage 一致，此处为 m。
    set_camera_view(
        eye=Gf.Vec3d(1.35, -1.65, 1.05),
        target=Gf.Vec3d(0.0, -0.1, 0.42),
        camera_prim_path="/OmniverseKit_Persp",
    )


def build_world(
    *, physics_dt: float | None, rendering_dt: float | None, gravity_z: float
):
    """创建带默认地面和重力的 Isaac ``World``。

    ``World`` 是 Isaac Sim 高层仿真入口，内部持有 stage、physics context、scene
    对象管理器等运行时状态。这里集中创建 ``World``，可以让不同脚本共享一致的
    单位、步长和重力设置。

    参数:
        physics_dt: 物理步长，单位 s；为 ``None`` 时使用 Isaac 默认值。
        rendering_dt: 渲染步长，单位 s；为 ``None`` 时使用 Isaac 默认值。
        gravity_z: z 方向重力加速度，单位 m/s^2。
    返回:
        已创建默认地面并设置重力的 ``World`` 实例。
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

    # 添加 Isaac 默认地面，给机器人、绳体等对象提供基础接触面。
    # 如果未来需要无地面场景，可以在此函数参数中扩展 add_ground 开关。
    world.scene.add_default_ground_plane()
    return world
