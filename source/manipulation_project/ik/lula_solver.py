"""Isaac Sim Lula IK 的兼容后端封装。

cuMotion 是默认迁移方向，但本机或旧项目可能暂时只有 Isaac Sim 自带 Lula。
保留该后端可以用 --ik-backend lula 做 smoke test 和回归对照。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipulation_project.ik.ik_request import IKRequest
from manipulation_project.ik.ik_result import IKResult


class LulaIKSolver:
    """使用 Isaac Lula 求解 TCP IK 的后端实现。

    输入:
        robot_description_path: Lula 机器人描述 YAML 路径。
        urdf_path: 机器人 URDF 路径。
        frame_name: 要求解的 TCP frame 名。
        default_cspace_seeds/ccd_max_iterations/bfgs_max_iterations/orientation_weight:
        传给 Lula solver 的求解参数。
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
        """启用 Lula 扩展并加载 IK solver。

        参数:
            robot_description_path: Lula 机器人描述路径。
            urdf_path: URDF 路径。
            frame_name: TCP frame 名。
            default_cspace_seeds: 可选默认 C-space seeds。
            ccd_max_iterations: 可选 CCD 最大迭代次数。
            bfgs_max_iterations: 可选 BFGS 最大迭代次数。
            orientation_weight: 可选姿态权重。
        返回:
            无返回值；副作用是启用 Isaac motion generation 扩展。
        """

        from isaacsim.core.utils.extensions import enable_extension
        import omni.kit.app

        enable_extension("isaacsim.robot_motion.motion_generation")
        omni.kit.app.get_app().update()

        from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

        self.backend = "lula"
        self.frame_name = frame_name
        self.solver = LulaKinematicsSolver(str(robot_description_path), str(urdf_path))
        if ccd_max_iterations is not None:
            self.solver.ccd_max_iterations = int(ccd_max_iterations)
        if bfgs_max_iterations is not None:
            self.solver.bfgs_max_iterations = int(bfgs_max_iterations)
        if orientation_weight is not None:
            self.solver.ccd_orientation_weight = float(orientation_weight)
            self.solver.bfgs_orientation_weight = float(orientation_weight)
        if default_cspace_seeds is not None:
            self.solver.set_default_cspace_seeds(np.asarray(default_cspace_seeds, dtype=float))

    def joint_names(self) -> list[str]:
        """返回 Lula C-space 中的主动关节名。

        返回:
            关节名列表，顺序与 Lula IK 输出一致。
        """

        return list(self.solver.get_joint_names())

    def frame_names(self) -> list[str]:
        """返回 Lula 可查询的全部 frame 名。

        返回:
            frame 名列表。
        """

        return list(self.solver.get_all_frame_names())

    def solve(self, request: IKRequest) -> IKResult:
        """求解单个 IK 请求。

        参数:
            request: TCP 位置/姿态目标、warm start 和容差。
        返回:
            ``IKResult``，position_error 通过 FK 后验计算。
        """

        q, success = self.solver.compute_inverse_kinematics(
            self.frame_name,
            np.asarray(request.target_position, dtype=float),
            None if request.target_orientation is None else np.asarray(request.target_orientation, dtype=float),
            warm_start=None if request.warm_start is None else np.asarray(request.warm_start, dtype=float),
            position_tolerance=float(request.position_tolerance),
            orientation_tolerance=float(request.orientation_tolerance),
        )
        q = np.asarray(q, dtype=float)
        achieved_position, _achieved_rotation = self.solver.compute_forward_kinematics(self.frame_name, q)
        position_error = float(np.linalg.norm(np.asarray(request.target_position, dtype=float) - achieved_position))
        return IKResult(joint_positions=q, success=bool(success), position_error=position_error)
