"""执行层 runtime 数据结构。

runtime 对象只保存执行动作所需的 Isaac 引用，例如 articulation、simulation world、关节
控制器和可选日志器。它不负责资源生命周期，也不生成目标或轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExecutionRuntime:
    """执行步骤所需的 Isaac runtime 对象。

    Isaac runtime 即执行仿真步骤时需要的一组 Isaac Sim 运行时对象。它只保存对象引用，
    不拥有这些对象的生命周期；创建/销毁 world、robot、controller 仍由动作脚本负责。
    """

    # Isaac articulation 对象，用于读取 DOF 数量、设置关节速度和采集实际关节状态。
    articulation: object
    # Isaac simulation world，负责 ``step`` 推进物理仿真并提供 physics dt。
    simulation_world: object
    # Isaac 的 ArticulationAction 类型/工厂，关节控制器用它构造每帧下发的 action。
    articulation_action_type: object
    # 项目关节控制器，负责把完整 DOF 目标转换成主动关节和 mimic follower 的 Isaac action。
    joint_controller: object
    # 可选 SimulationApp；GUI/交互运行时用于检测窗口是否仍在运行，测试中可以为 ``None``。
    simulation_app: object | None
    # 是否在 ``world.step`` 时渲染画面；headless 或测试场景通常为 ``False``。
    render_enabled: bool
    # 可选驱动日志器，记录下发目标、实际关节状态和控制误差等逐步数据。
    drive_logger: object | None = None
    # 可选状态 observer，在 world.step 后从主线程采样机器人和场景状态。
    state_observer: object | None = None
    # 可选 camera observer，在 world.step 后从主线程采样 sensor camera。
    camera_observer: object | None = None


class ExecutionStep(Protocol):
    """可顺序执行的执行步骤协议。

    所有实现都接收同一个 ``ExecutionRuntime`` 和全局 step，并返回更新后的 step。协议不
    要求实现一定推进 world；例如控制模式切换步骤只改变 runtime 配置并原样返回 step。
    """

    phase: str

    def run(self, runtime: ExecutionRuntime, step: int) -> int:
        """执行步骤并返回新的全局 step。"""
