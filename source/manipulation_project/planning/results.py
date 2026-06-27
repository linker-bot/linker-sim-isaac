"""运动解算结果数据结构。

结果对象用于把不同规划/运动学后端的状态归一化给任务层。后端可以把原始错误码写入
``status``，把可读说明写入 ``message`` 或 ``diagnostics``；任务层只需要先检查
``success``，再决定是否执行轨迹、回退到保守动作或向用户报告失败。

职责边界:
    * 不保存求解器对象或 GPU 资源，只保存可序列化/可打印的结果摘要。
    * 不重新排列关节顺序；``joint_positions``/``path`` 的列顺序必须由生成它的后端说明。
    * 不在失败时抛异常；失败是规划问题的常见输出，异常留给配置错误或运行时错误。

单位约定:
    位置误差使用 m；关节位置使用 rad；姿态误差的具体度量由后端定义，但应在同一后端内
    保持可比较，便于阈值和日志诊断。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PlanningDiagnostics:
    """规划和解算诊断信息。

    ``metrics`` 用于保存耗时、迭代次数、误差等数值，避免把后端特有字段散落到结果类。键名
    由具体后端定义，但应保持可打印、可写入日志，不存放 pybind 对象或 numpy 大矩阵。
    """

    status: str = ""
    message: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class IKResult:
    """逆运动学结果。

    ``joint_positions`` 按后端关节顺序排列；失败时可以为空或回退到 warm start。位置误差单位
    为米，姿态误差单位由后端定义但应保持可比较。调用方不能只看 ``joint_positions`` 是否
    非空来判断可执行性，必须先检查 ``success``。
    """

    joint_positions: np.ndarray
    success: bool
    position_error: float
    orientation_error: float | None = None
    message: str = ""
    status: str = ""
    num_solutions: int = 1


@dataclass(frozen=True)
class MotionResult:
    """路径级运动规划结果。

    ``path`` 通常是离散 C-space 关节路径，``trajectory`` 保存后端生成的时间参数化轨迹对象。
    二者允许同时存在，也允许在失败时都为空；调用方应先检查 ``success``，再消费轨迹字段。
    ``path`` 和 ``trajectory`` 均保持后端关节顺序，不自动扩展成 Isaac 完整 DOF。
    """

    path: np.ndarray | None
    trajectory: object | None
    success: bool
    status: str
    diagnostics: PlanningDiagnostics = field(default_factory=PlanningDiagnostics)
