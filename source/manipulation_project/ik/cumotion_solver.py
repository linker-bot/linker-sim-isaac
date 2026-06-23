"""NVIDIA cuMotion IK 的轻量封装。

本模块把项目内部统一的 IKRequest / IKResult 转换为 cuMotion Python API。
cuMotion 使用 XRDF + URDF 加载机器人；目标姿态在项目内保持 wxyz 四元数，
调用 cuMotion 前转换成 Rotation3。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipulation_project.ik.ik_request import IKRequest
from manipulation_project.ik.ik_result import IKResult
from manipulation_project.utils.math_utils import quat_wxyz_to_matrix


def _seed_list(seeds: np.ndarray) -> list[np.ndarray]:
    """把单个 seed 或 seed 矩阵规范成 cuMotion 接受的 seed list。

    参数:
        seeds: shape ``(dof,)`` 或 ``(N, dof)`` 的数组。
    返回:
        ndarray 列表，每项都是一个 C-space seed。
    """

    seed_array = np.asarray(seeds, dtype=float)
    if seed_array.ndim == 1:
        return [seed_array]
    return [np.asarray(seed, dtype=float) for seed in seed_array]


class CuMotionIKSolver:
    """使用 cuMotion 求解 TCP IK 的后端实现。

    输入:
        robot_description_path: cuMotion XRDF 路径。
        urdf_path: 机器人 URDF 路径。
        frame_name: 要求解的 TCP frame 名。
        default_cspace_seeds/ccd_max_iterations/bfgs_max_iterations/orientation_weight:
        传给 ``cumotion.IkConfig`` 的求解参数。
    输出:
        ``solve`` 返回项目统一的 ``IKResult``。
    """

    def __init__(
        self,
        robot_description_path: str | Path,
        urdf_path: str | Path,
        *,
        frame_name: str,
        default_cspace_seeds: np.ndarray | None = None,
        ccd_max_iterations: int | None = None,
        bfgs_max_iterations: int | None = None,
        orientation_weight: float | None = None,
    ) -> None:
        """加载 cuMotion 机器人模型并初始化 IK 配置。

        参数:
            robot_description_path: XRDF 文件路径。
            urdf_path: URDF 文件路径。
            frame_name: TCP frame 名。
            default_cspace_seeds: 可选默认种子。
            ccd_max_iterations: 可选 CCD 最大迭代次数。
            bfgs_max_iterations: 可选 BFGS 最大迭代次数。
            orientation_weight: 可选姿态权重。
        返回:
            无返回值；缺少 cuMotion 包时抛出 ``ImportError``。
        """

        try:
            import cumotion
        except ImportError as exc:
            raise ImportError(
                "cuMotion is not installed in this Python environment. Install the NVIDIA "
                "cuMotion wheel from https://github.com/nvidia-isaac/cumotion/releases, "
                "or run with --ik-backend lula to use the legacy Isaac Sim Lula backend."
            ) from exc

        self.cumotion = cumotion
        self.backend = "cumotion"
        self.frame_name = frame_name
        self.robot_description = cumotion.load_robot_from_file(str(robot_description_path), str(urdf_path))
        self.kinematics = self.robot_description.kinematics()
        self.config = cumotion.IkConfig()
        self.orientation_weight = orientation_weight
        if default_cspace_seeds is not None:
            self.config.cspace_seeds = _seed_list(np.asarray(default_cspace_seeds, dtype=float))
        if ccd_max_iterations is not None:
            self.config.ccd_max_iterations = int(ccd_max_iterations)
        if bfgs_max_iterations is not None:
            self.config.bfgs_max_iterations = int(bfgs_max_iterations)
        if orientation_weight is not None:
            self.config.ccd_orientation_weight = float(orientation_weight)
            self.config.bfgs_orientation_weight = float(orientation_weight)

    def joint_names(self) -> list[str]:
        """返回 cuMotion C-space 中的主动关节名。

        返回:
            关节名列表，顺序与 cuMotion C-space 坐标一致。
        """

        return [str(self.kinematics.cspace_coord_name(index)) for index in range(self.kinematics.num_cspace_coords())]

    def frame_names(self) -> list[str]:
        """返回 cuMotion 可查询的全部 frame 名。

        返回:
            frame 名列表。
        """

        return [str(name) for name in self.kinematics.frame_names()]

    def _target_pose(self, request: IKRequest):
        """把项目 ``IKRequest`` 转为 cuMotion ``Pose3``。

        参数:
            request: 项目内部 IK 请求。
        返回:
            cuMotion ``Pose3``；无姿态目标时只包含 translation。
        """

        position = np.asarray(request.target_position, dtype=float).reshape(3)
        if request.target_orientation is None:
            return self.cumotion.Pose3.from_translation(position)
        rotation_matrix = quat_wxyz_to_matrix(request.target_orientation)
        return self.cumotion.Pose3(self.cumotion.Rotation3.from_matrix(rotation_matrix), position)

    def solve(self, request: IKRequest) -> IKResult:
        """求解单个 IK 请求。

        参数:
            request: TCP 位置/姿态目标、warm start 和容差。
        返回:
            ``IKResult``，关节顺序与 ``joint_names`` 一致。
        """

        if request.warm_start is not None:
            self.config.cspace_seeds = [np.asarray(request.warm_start, dtype=float)]
        self.config.position_tolerance = float(request.position_tolerance)
        if request.target_orientation is None:
            self.config.orientation_tolerance = 1.0e9
            self.config.ccd_orientation_weight = 0.0
            self.config.bfgs_orientation_weight = 0.0
        else:
            self.config.orientation_tolerance = float(request.orientation_tolerance)
            if self.orientation_weight is not None:
                self.config.ccd_orientation_weight = float(self.orientation_weight)
                self.config.bfgs_orientation_weight = float(self.orientation_weight)

        result = self.cumotion.solve_ik(self.kinematics, self._target_pose(request), self.frame_name, self.config)
        q = np.asarray(result.cspace_position, dtype=float)
        position_error = float(result.position_error)
        orientation_error = max(
            float(result.x_axis_orientation_error),
            float(result.y_axis_orientation_error),
            float(result.z_axis_orientation_error),
        )
        if result.success:
            self.config.cspace_seeds = [q]
        return IKResult(
            joint_positions=q,
            success=bool(result.success),
            position_error=position_error,
            orientation_error=orientation_error,
        )
