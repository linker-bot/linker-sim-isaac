"""从环境 YAML 和命令行覆盖中解析仿真运行参数。

本模块只处理启动 ``World`` 所需的纯配置：物理步频、渲染步频、世界重力和默认地面。
它不 import Isaac/Omni，因此可以在普通 Python 单元测试或 dry-run 中提前校验 env 配置。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvRuntimeSettings:
    """动作脚本共享的 World 时间步和世界重力设置。

    ``physics_frequency`` 决定物理步长和轨迹采样节奏；``render_frequency`` 只在 GUI 模式下
    决定渲染步长，headless 模式会直接使用物理步长，避免渲染频率影响 smoke test。
    """

    physics_frequency: float
    render_frequency: float
    gravity_z: float
    add_ground: bool = True

    @classmethod
    def from_env_config(
        cls,
        env_config: Mapping[str, object],
        *,
        physics_frequency_override: float | None = None,
        render_frequency_override: float | None = None,
        gravity_z_override: float | None = None,
    ) -> "EnvRuntimeSettings":
        """解析 ``env`` 分组，并套用可选命令行覆盖。

        覆盖优先级为 CLI 参数高于 YAML。这里仍允许 CLI 覆盖世界级频率和重力，因为这些属于
        运行实验时常见的调试参数；机器人自身重力策略不在这里处理，而是在 robot YAML 中声明。
        """

        env = env_mapping(env_config)
        config = cls(
            physics_frequency=float(
                physics_frequency_override
                if physics_frequency_override is not None
                else env.get("physics_frequency", 600.0)
            ),
            render_frequency=float(
                render_frequency_override
                if render_frequency_override is not None
                else env.get("render_frequency", 100.0)
            ),
            gravity_z=float(
                gravity_z_override
                if gravity_z_override is not None
                else env.get("gravity_z", -9.81)
            ),
            add_ground=bool(env.get("add_ground", True)),
        )
        config.validate()
        return config

    @property
    def physics_dt(self) -> float:
        """物理步长，单位秒。"""

        return 1.0 / self.physics_frequency

    def rendering_dt(self, *, gui: bool) -> float:
        """返回 Isaac World 使用的渲染步长。

        headless 模式不需要独立渲染节奏，直接跟随 ``physics_dt``，这样 dry-run/smoke test 的
        时间推进更容易和控制采样对齐。
        """

        return 1.0 / self.render_frequency if gui else self.physics_dt

    def validate(self) -> None:
        if self.physics_frequency <= 0 or self.render_frequency <= 0:
            raise ValueError("physics and render frequencies must be positive")


def env_mapping(config: Mapping[str, object]) -> Mapping[str, object]:
    """读取顶层 ``env`` mapping，并为结构错误给出一致提示。"""

    env = config.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("Environment config must contain top-level env mapping")
    return env
