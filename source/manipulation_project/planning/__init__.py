"""运动规划请求、结果和后端无关数据结构。

planning 子包是 tasks 与具体求解后端之间的稳定契约。请求对象只表达目标位姿、初值、碰撞
对象和容差，结果对象只表达成功状态、关节解、误差和诊断信息。

该入口不创建任何求解器，也不导入 Isaac runtime；它只重新导出轻量 dataclass。这样后续替换
cuMotion、加入采样规划器或 mock 求解器时，不需要改动任务层 public API。单位和顺序约定由
各请求/结果类 docstring 说明，任务层应先检查 ``success`` 再消费关节解。
"""

from manipulation_project.planning.collision_objects import CollisionObject
from manipulation_project.planning.requests import (
    IKRequest,
    MotionRequest,
    OrientationMode,
    PoseTarget,
    TcpLineRequest,
)
from manipulation_project.planning.results import (
    IKResult,
    MotionResult,
    PlanningDiagnostics,
    TcpLineDiagnostics,
    TcpLinePlan,
)

__all__ = [
    "CollisionObject",
    "IKRequest",
    "IKResult",
    "MotionRequest",
    "MotionResult",
    "OrientationMode",
    "PlanningDiagnostics",
    "PoseTarget",
    "TcpLineDiagnostics",
    "TcpLinePlan",
    "TcpLineRequest",
]
