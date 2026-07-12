"""Isaac SimulationApp 与 World 的会话装配工具。

动作脚本只关心“拿到一个可用的仿真会话”：app 生命周期、World 创建、stage 获取和 Isaac
runtime 类型句柄都集中在这里。Isaac/Omni 的重型 import 仍放在 ``SimulationApp`` 启动后，
避免普通配置解析测试误触发 Isaac 初始化。
"""

from __future__ import annotations

from dataclasses import dataclass

from linkerbot_sim.app.launch import launch_simulation_app
from linkerbot_sim.envs.settings import EnvRuntimeSettings
from linkerbot_sim.app.runtime.simulation_app_lifecycle import close_simulation_app
from linkerbot_sim.configs.runtime import SimulationAppSettings
from linkerbot_sim.envs.scene_builder import build_world, configure_visuals


@dataclass(frozen=True)
class SimulationSession:
    """Isaac 启动后脚本需要持有的一组运行时对象。

    ``app`` 由调用脚本在 ``finally`` 中关闭；``world`` 和 ``stage`` 用于后续导入资产；
    两个 type handle 传给 controller/execution 层，避免这些层在模块导入时直接 import Isaac。
    """

    app: object
    world: object
    stage: object
    articulation_action_type: object
    single_articulation_type: object


def create_simulation_session(
    *, simulation_app: SimulationAppSettings, settings: EnvRuntimeSettings
) -> SimulationSession:
    """启动 Isaac、创建 World，并返回 stage 和 runtime 类型句柄。

    如果 World 创建或延迟 import 失败，本函数会主动关闭刚启动的 ``SimulationApp``，避免异常
    路径留下 Kit 进程或扩展资源。正常路径下不关闭 app，由脚本统一管理生命周期。
    """

    app = launch_simulation_app(simulation_app)
    try:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        import omni.usd

        world = build_world(
            physics_dt=settings.physics_dt,
            rendering_dt=settings.rendering_dt(
                gui=simulation_app.gui,
                headless_dt_policy=(simulation_app.render.headless_dt_policy),
            ),
            gravity_z=settings.gravity_z,
            add_ground=settings.add_ground,
            ground_height=settings.ground_height,
        )
        if settings.requires_rendering(gui=simulation_app.gui):
            configure_visuals(
                settings.visuals,
                configure_viewport=simulation_app.gui,
            )
        return SimulationSession(
            app=app,
            world=world,
            stage=omni.usd.get_context().get_stage(),
            articulation_action_type=ArticulationAction,
            single_articulation_type=SingleArticulation,
        )
    except Exception:
        close_simulation_app(app)
        raise
