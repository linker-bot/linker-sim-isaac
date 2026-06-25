"""仿真执行层入口。

execution 层只负责把已经规划或采样好的目标下发到 Isaac world，并在 physics step 后
记录实际状态。它不生成轨迹、不求解 IK，也不读取任务配置；这种分层便于把同一条
``JointTrajectory`` 用于 GUI 演示、headless 测试或日志回放。入口文件保持轻量，只暴露
执行层概念，不主动创建 world。
"""

from manipulation_project.execution.joint_trajectory_executor import (
    execute_joint_trajectory,
)

__all__ = ["execute_joint_trajectory"]
