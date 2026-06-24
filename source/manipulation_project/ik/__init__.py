"""逆运动学兼容入口。

新代码应优先使用 ``manipulation_project.planning`` 和
``manipulation_project.backends.cumotion``。
"""

from manipulation_project.backends.cumotion.inverse_kinematics import CuMotionInverseKinematics
from manipulation_project.ik.cumotion_solver import CuMotionIKSolver
from manipulation_project.ik.ik_request import IKRequest
from manipulation_project.ik.ik_result import IKResult
from manipulation_project.ik.solver_factory import IKSolver, make_ik_solver

__all__ = [
    "CuMotionIKSolver",
    "CuMotionInverseKinematics",
    "IKRequest",
    "IKResult",
    "IKSolver",
    "make_ik_solver",
]
