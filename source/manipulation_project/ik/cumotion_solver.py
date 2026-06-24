"""兼容入口：cuMotion IK 已迁移到 ``backends.cumotion``。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manipulation_project.backends.cumotion.context import CuMotionConfig, CuMotionContext
from manipulation_project.backends.cumotion.inverse_kinematics import CuMotionInverseKinematics


class CuMotionIKSolver(CuMotionInverseKinematics):
    """旧类名兼容封装。"""

    def __init__(
        self,
        xrdf_path: str | Path,
        urdf_path: str | Path,
        *,
        tcp_frame_name: str,
        cspace_seeds: np.ndarray | None = None,
        ccd_max_iterations: int | None = None,
        bfgs_max_iterations: int | None = None,
        orientation_weight: float | None = None,
    ) -> None:
        config = CuMotionConfig(
            xrdf_path=xrdf_path,
            urdf_path=urdf_path,
            default_tcp_frame=tcp_frame_name,
            cspace_seeds=cspace_seeds,
            ccd_max_iterations=180 if ccd_max_iterations is None else int(ccd_max_iterations),
            bfgs_max_iterations=80 if bfgs_max_iterations is None else int(bfgs_max_iterations),
            orientation_weight=0.25 if orientation_weight is None else float(orientation_weight),
        )
        super().__init__(CuMotionContext(config), tcp_frame_name=tcp_frame_name)
