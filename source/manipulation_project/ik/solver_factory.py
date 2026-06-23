"""IK 后端选择工厂。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from manipulation_project.ik.ik_request import IKRequest
from manipulation_project.ik.ik_result import IKResult


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


def is_cumotion_available() -> bool:
    """检查当前 Python 环境是否安装了 cuMotion。

    参数:
        无。
    返回:
        能 import ``cumotion`` 时为 ``True``，否则为 ``False``。
    """

    try:
        import cumotion  # noqa: F401
    except ImportError:
        return False
    return True


def make_ik_solver(
    backend: str,
    robot_description_path: str | Path,
    urdf_path: str | Path,
    *,
    frame_name: str,
    default_cspace_seeds: np.ndarray | None = None,
    ccd_max_iterations: int | None = None,
    bfgs_max_iterations: int | None = None,
    orientation_weight: float | None = None,
) -> IKSolver:
    """创建 IK 后端。

    参数:
        backend: ``auto``、``cumotion`` 或 ``lula``。
        robot_description_path: 后端机器人描述路径；cuMotion 通常为 XRDF，Lula 为 YAML。
        urdf_path: 机器人 URDF 路径，可能是临时附加 TCP 后的 URDF。
        frame_name: 求解目标 frame 名。
        default_cspace_seeds: 可选默认 IK seeds，shape ``(N, dof)`` 或 ``(dof,)``。
        ccd_max_iterations: 可选 CCD 最大迭代次数。
        bfgs_max_iterations: 可选 BFGS 最大迭代次数。
        orientation_weight: 可选姿态误差权重。
    返回:
        实现 ``IKSolver`` 协议的后端实例。``auto`` 会优先 cuMotion，缺失时退回 Lula。
    """

    normalized = backend.strip().lower()
    if normalized == "auto":
        normalized = "cumotion" if is_cumotion_available() else "lula"

    kwargs = {
        "frame_name": frame_name,
        "default_cspace_seeds": default_cspace_seeds,
        "ccd_max_iterations": ccd_max_iterations,
        "bfgs_max_iterations": bfgs_max_iterations,
        "orientation_weight": orientation_weight,
    }
    if normalized == "cumotion":
        from manipulation_project.ik.cumotion_solver import CuMotionIKSolver

        return CuMotionIKSolver(robot_description_path, urdf_path, **kwargs)
    if normalized == "lula":
        from manipulation_project.ik.lula_solver import LulaIKSolver

        return LulaIKSolver(robot_description_path, urdf_path, **kwargs)
    raise ValueError(f"Unsupported IK backend {backend!r}; expected auto, cumotion, or lula")
