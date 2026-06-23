"""TCP 点到目标位姿任务占位模块。

可复用的 IK 和轨迹基础件已经在 ``ik/`` 与 ``trajectories/`` 中实现；这里保留任务类
入口，后续把完整 Isaac 执行流程从脚本迁入时不会破坏调用路径。
"""

from __future__ import annotations


class MoveTcpToPoseTask:
    """未来的 TCP 点到点任务类。

    输入:
        后续会接收目标 TCP 位置/姿态、求解器配置和执行时长。
    输出:
        当前为空实现，不返回规划结果。
    """

    pass
