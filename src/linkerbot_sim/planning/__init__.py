"""运动规划请求、结果和后端无关数据结构。

planning 子包是动作脚本与具体求解后端之间的稳定契约。请求对象只表达目标位姿、初值、碰撞
对象和容差，结果对象只表达成功状态、关节解、误差和诊断信息。

该入口不创建任何求解器，也不导入 Isaac runtime；它只重新导出轻量 dataclass。单位和顺序
约定由各请求/结果类 docstring 说明，调用方应先检查 ``success`` 再消费关节解。
"""

from linkerbot_sim.planning.backend import (
    PlannerBackend,
    PlannerBackendName,
    PlanningRequest,
    normalize_planner_backend,
)
from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.planning.frames import FrameTransformer, PoseInRobotBase
from linkerbot_sim.planning.linear_backend import LinearPlannerBackend
from linkerbot_sim.planning.requests import (
    IKRequest,
    LinearPosePathRequest,
    MotionRequest,
    OrientationMode,
    PoseTarget,
    TaskSpacePath,
    TcpLineSegment,
    TcpPoseSequenceSegment,
)
from linkerbot_sim.planning.results import (
    IKResult,
    MotionResult,
    PlanningDiagnostics,
)

__all__ = [
    "CollisionObject",
    "FrameTransformer",
    "IKRequest",
    "IKResult",
    "LinearPosePathRequest",
    "MotionRequest",
    "MotionResult",
    "OrientationMode",
    "PlannerBackend",
    "PlannerBackendName",
    "PlanningRequest",
    "PlanningDiagnostics",
    "PoseTarget",
    "PoseInRobotBase",
    "TaskSpacePath",
    "TcpLineSegment",
    "TcpPoseSequenceSegment",
    "LinearPlannerBackend",
    "normalize_planner_backend",
]
