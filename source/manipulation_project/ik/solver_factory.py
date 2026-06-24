"""cuMotion IK 创建入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from manipulation_project.planning.requests import IKRequest
from manipulation_project.planning.results import IKResult


class IKSolver(Protocol):
    """各个 IK 后端共同实现的最小接口。

    输入:
        后端实现需要在初始化时加载机器人描述、URDF 和目标 frame。
    输出:
        ``solve`` 接收 ``IKRequest`` 并返回统一 ``IKResult``。
    """

    backend: str

    def joint_names(self) -> list[str]:
        """返回 IK C-space 中的主动关节名。

        返回:
            关节名列表，顺序与 ``IKResult.joint_positions`` 一致。
        """

    def frame_names(self) -> list[str]:
        """返回后端可用的 frame 名。

        返回:
            后端机器人模型中可用于 FK/IK 的 frame 名列表。
        """

    def solve(self, request: IKRequest) -> IKResult:
        """求解一个 IK 请求。

        参数:
            request: 单个 TCP 目标和求解容差。
        返回:
            ``IKResult``。
        """


def make_ik_solver(
    backend_or_robot_description_path: str | Path,
    robot_description_path: str | Path | None = None,
    urdf_path: str | Path | None = None,
    *,
    tcp_frame_name: str,
    cspace_seeds: np.ndarray | None = None,
    ccd_max_iterations: int | None = None,
    bfgs_max_iterations: int | None = None,
    orientation_weight: float | None = None,
) -> IKSolver:
    """创建 cuMotion IK 解算器。

    参数:
        backend_or_robot_description_path: 旧调用中为 ``cumotion``；新调用中可直接传 XRDF 路径。
        robot_description_path: cuMotion XRDF 路径。
        urdf_path: 机器人 URDF 路径，可能是临时附加 TCP 后的 URDF。
        tcp_frame_name: 求解目标 TCP frame 名。
        cspace_seeds: 可选 IK C-space seeds，shape ``(N, dof)`` 或 ``(dof,)``。
        ccd_max_iterations: 可选 CCD 最大迭代次数。
        bfgs_max_iterations: 可选 BFGS 最大迭代次数。
        orientation_weight: 可选姿态误差权重。
    返回:
        实现 ``IKSolver`` 协议的 cuMotion 实例。
    """

    first = backend_or_robot_description_path
    if robot_description_path is None:
        raise TypeError("robot_description_path and urdf_path are required")
    if str(first).strip().lower() == "cumotion":
        xrdf_path = robot_description_path
        if urdf_path is None:
            raise TypeError("urdf_path is required")
    else:
        if urdf_path is not None:
            raise ValueError("Unsupported IK backend; only cumotion is supported")
        xrdf_path = first
        urdf_path = robot_description_path

    kwargs = {
        "tcp_frame_name": tcp_frame_name,
        "cspace_seeds": cspace_seeds,
        "ccd_max_iterations": ccd_max_iterations,
        "bfgs_max_iterations": bfgs_max_iterations,
        "orientation_weight": orientation_weight,
    }
    from manipulation_project.ik.cumotion_solver import CuMotionIKSolver

    return CuMotionIKSolver(xrdf_path, urdf_path, **kwargs)
