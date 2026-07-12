"""从环境 YAML 解析仿真运行参数。

本模块只处理启动 ``World`` 所需的纯配置：物理步频、渲染步频、世界重力和默认地面。
它不 import Isaac/Omni，因此可以在普通 Python 单元测试或 dry-run 中提前校验 env 配置。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite

from linkerbot_sim.envs.visual_settings import SceneVisualSettings
from linkerbot_sim.sensors import SceneSensorSettings


@dataclass(frozen=True)
class EnvRuntimeSettings:
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

    @classmethod
    def from_env_config(
        cls,
        env_config: Mapping[str, object],
    ) -> "EnvRuntimeSettings":
        """解析 ``env`` 分组。

        世界级频率和重力统一来自 env profile；机器人自身重力策略不在这里处理，而是在
        robot YAML 中声明。
        """

        env = env_mapping(env_config)
        config = cls(
            physics_frequency=_finite_float(
                env.get("physics_frequency", 600.0),
                "env.physics_frequency",
            ),
            render_frequency=_finite_float(
                env.get("render_frequency", 100.0),
                "env.render_frequency",
            ),
            gravity_z=_finite_float(env.get("gravity_z", -9.81), "env.gravity_z"),
            add_ground=_strict_bool(env.get("add_ground", True), "env.add_ground"),
            ground_height=_finite_float(
                env.get("ground_height", 0.0), "env.ground_height"
            ),
            visuals=SceneVisualSettings.from_env_config(env_config),
            sensors=SceneSensorSettings.from_env_config(env_config),
        )
        config.validate()
        return config

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


def env_mapping(config: Mapping[str, object]) -> Mapping[str, object]:
    """读取顶层 ``env`` mapping，并为结构错误给出一致提示。"""

    env = config.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("Environment config must contain top-level env mapping")
    return env


def _strict_bool(value: object, label: str) -> bool:
    """严格解析 YAML 布尔值，不接受 truthy 字符串或整数替代。"""

    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _finite_float(value: object, label: str) -> float:
    """解析有限 YAML 数值，并显式拒绝 Python 中属于整数子类的布尔值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number
