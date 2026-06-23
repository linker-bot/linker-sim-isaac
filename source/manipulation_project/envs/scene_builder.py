"""基础 Isaac 场景构建工具。"""

from __future__ import annotations


def configure_visuals() -> None:
    """添加基础灯光和默认相机。

    参数:
        无，直接操作当前 Omni/USD context。
    返回:
        无返回值；副作用是创建 ``/World/KeyLight``、``/World/FillLight`` 并设置 viewport。
    """

    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Gf, Sdf, UsdLux
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    key.CreateIntensityAttr(1200.0)
    key.CreateAngleAttr(0.5)
    fill = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/FillLight"))
    fill.CreateIntensityAttr(250.0)
    set_camera_view(
        eye=Gf.Vec3d(1.35, -1.65, 1.05),
        target=Gf.Vec3d(0.0, -0.1, 0.42),
        camera_prim_path="/OmniverseKit_Persp",
    )


def build_world(*, physics_dt: float | None, rendering_dt: float | None, gravity_z: float):
    """创建带默认地面和重力的 Isaac ``World``。

    参数:
        physics_dt: 物理步长，单位 s；为 ``None`` 时使用 Isaac 默认值。
        rendering_dt: 渲染步长，单位 s；为 ``None`` 时使用 Isaac 默认值。
        gravity_z: z 方向重力加速度，单位 m/s^2。
    返回:
        已创建默认地面并设置重力的 ``World`` 实例。
    """

    from isaacsim.core.api.world import World

    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=rendering_dt)
    world.get_physics_context().set_gravity(float(gravity_z))
    world.scene.add_default_ground_plane()
    return world
