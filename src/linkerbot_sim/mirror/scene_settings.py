"""Mirror 资产装配桥接层使用的场景运行设置。

本模块只处理启动 ``World`` 所需的纯配置：物理步频、渲染步频、世界重力和默认地面。
它不 import Isaac/Omni，因此可以在普通 Python 单元测试中校验 strict config 的场景投影。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from linkerbot_sim.configuration.scenes import SceneVisualSettings
from linkerbot_sim.sensors import SceneSensorSettings


@dataclass(frozen=True)
class MirrorSceneRuntimeSettings:
    """动作脚本共享的 World 时间步和世界重力设置。

    ``physics_frequency`` 决定物理步长和轨迹采样节奏；``render_frequency`` 在 GUI 或
    headless camera output 模式下决定渲染步长。纯物理 headless 仍直接使用物理步长。
    """

    physics_frequency: float
    render_frequency: float
    gravity_z: float
    add_ground: bool = True
    ground_height: float = 0.0
    visuals: SceneVisualSettings = field(default_factory=SceneVisualSettings)
    sensors: SceneSensorSettings = field(default_factory=SceneSensorSettings)

    @property
    def physics_dt(self) -> float:
        """物理步长，单位秒。"""

        return 1.0 / self.physics_frequency

    def rendering_dt(
        self,
        *,
        gui: bool,
        headless_dt_policy: str = "camera_aware",
    ) -> float:
        """返回 Isaac World 使用的渲染步长。

        headless camera output 需要按配置渲染节奏生成传感器帧；没有 frame consumer 的
        纯物理 headless 则跟随 ``physics_dt``，避免无用渲染影响 smoke test。显式
        ``physics`` 策略会让 headless 渲染也跟随 physics cadence。
        """

        if headless_dt_policy not in {"camera_aware", "physics"}:
            raise ValueError("headless_dt_policy must be one of: camera_aware, physics")
        if not gui and headless_dt_policy == "physics":
            return self.physics_dt
        return (
            1.0 / self.render_frequency
            if self.requires_rendering(gui=gui)
            else self.physics_dt
        )

    def requires_rendering(self, *, gui: bool) -> bool:
        """判断 session 是否需要 GUI 或 camera output 渲染。"""

        return bool(gui) or self.sensors.has_output_consumers

    def validate(self) -> None:
        """校验 World 频率为正数，避免 Isaac 创建零步长或负步长。"""

        if self.physics_frequency <= 0 or self.render_frequency <= 0:
            raise ValueError("physics and render frequencies must be positive")
