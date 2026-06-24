"""cuMotion 正运动学封装。"""

from __future__ import annotations

import numpy as np


class CuMotionForwardKinematics:
    """封装 FK 和 frame/joint 查询。"""

    def __init__(self, context) -> None:
        self.context = context
        self.kinematics = context.kinematics

    def joint_names(self) -> list[str]:
        """返回 C-space 关节名。"""

        return self.context.joint_names()

    def frame_names(self) -> list[str]:
        """返回 frame 名。"""

        return self.context.frame_names()

    def compute_pose(self, joint_positions, frame_name: str):
        """返回 cuMotion ``Pose3``。"""

        return self.kinematics.pose(np.asarray(joint_positions, dtype=float).reshape(-1), str(frame_name))

    def compute_position(self, joint_positions, frame_name: str) -> np.ndarray:
        """返回 frame 在 base 下的位置。"""

        pose = self.compute_pose(joint_positions, frame_name)
        return np.asarray(pose.translation, dtype=float).reshape(3)
