"""cuMotion 机器人模型和共享资源上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CuMotionConfig:
    """cuMotion 后端配置。"""

    xrdf_path: str | Path
    urdf_path: str | Path
    flange_frame: str = "AR5V2_L_arm_flan_link"
    default_tcp_frame: str | None = None
    cspace_seeds: np.ndarray | None = None
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.75
    ccd_max_iterations: int = 180
    bfgs_max_iterations: int = 80
    orientation_weight: float = 0.25


class CuMotionContext:
    """缓存 cuMotion robot description 和 kinematics。"""

    def __init__(self, config: CuMotionConfig) -> None:
        try:
            import cumotion
        except ImportError as exc:
            raise ImportError(
                "cuMotion is not installed in this Python environment. Install the NVIDIA "
                "cuMotion package from https://github.com/nvidia-isaac/cumotion/releases."
            ) from exc

        self.cumotion = cumotion
        self.config = config
        self.robot_description = cumotion.load_robot_from_file(str(config.xrdf_path), str(config.urdf_path))
        self.kinematics = self.robot_description.kinematics()

    def joint_names(self) -> list[str]:
        """返回 cuMotion C-space 主动关节名。"""

        return [str(self.kinematics.cspace_coord_name(index)) for index in range(self.kinematics.num_cspace_coords())]

    def frame_names(self) -> list[str]:
        """返回 cuMotion 可查询 frame 名。"""

        return [str(name) for name in self.kinematics.frame_names()]

    def has_frame(self, frame_name: str) -> bool:
        """检查 frame 是否存在。"""

        return str(frame_name) in set(self.frame_names())

    def make_inverse_kinematics(self, *, tcp_frame_name: str | None = None):
        """创建逆运动学组件。"""

        from manipulation_project.backends.cumotion.inverse_kinematics import CuMotionInverseKinematics

        return CuMotionInverseKinematics(self, tcp_frame_name=tcp_frame_name or self.config.default_tcp_frame)

    def make_forward_kinematics(self):
        """创建正运动学组件。"""

        from manipulation_project.backends.cumotion.forward_kinematics import CuMotionForwardKinematics

        return CuMotionForwardKinematics(self)
