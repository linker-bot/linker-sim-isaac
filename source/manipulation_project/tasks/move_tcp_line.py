"""TCP 直线移动任务占位模块。

底层笛卡尔直线采样和 IK 组件已经存在；完整 Isaac 执行任务会在关节目标任务稳定后
迁移到这里。
"""

from __future__ import annotations


class MoveTcpLineTask:
    """未来的 TCP 直线任务类。

    输入:
        后续会接收起点/终点 TCP 位姿、IK 配置和控制器配置。
    输出:
        当前为空实现，不产生轨迹或仿真动作。
    """

    pass
