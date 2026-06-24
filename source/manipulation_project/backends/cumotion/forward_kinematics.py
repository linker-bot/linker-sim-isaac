"""cuMotion 正运动学封装。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardKinematicsPose:
    """项目内部使用的 FK 位姿结果。

    position: frame 在机器人 base 下的位置，shape ``(3,)``，单位 m。
    orientation: frame 在机器人 base 下的姿态，wxyz 四元数，shape ``(4,)``。
    rotation_matrix: 同一姿态的旋转矩阵，shape ``(3, 3)``。
    """

    position: np.ndarray
    orientation: np.ndarray
    rotation_matrix: np.ndarray


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

    def compute_cumotion_pose(self, joint_positions, frame_name: str):
        """返回 cuMotion ``Pose3``。"""

        return self.kinematics.pose(np.asarray(joint_positions, dtype=float).reshape(-1), str(frame_name))

    def compute_pose(self, joint_positions, frame_name: str) -> ForwardKinematicsPose:
        """返回 frame 在 base 下的完整位姿。

        姿态统一转换为项目外部接口常用的 wxyz 四元数，同时保留旋转矩阵，便于
        IK 请求、日志和后续笛卡尔轨迹生成直接复用。
        """

        cumotion_pose = self.compute_cumotion_pose(joint_positions, frame_name)
        rotation = cumotion_pose.rotation
        return ForwardKinematicsPose(
            position=np.asarray(cumotion_pose.translation, dtype=float).reshape(3),
            orientation=np.asarray([rotation.w(), rotation.x(), rotation.y(), rotation.z()], dtype=float),
            rotation_matrix=np.asarray(rotation.matrix(), dtype=float).reshape(3, 3),
        )

    def compute_position(self, joint_positions, frame_name: str) -> np.ndarray:
        """返回 frame 在 base 下的位置。"""

        pose = self.compute_pose(joint_positions, frame_name)
        return pose.position.copy()

    def compute_orientation(self, joint_positions, frame_name: str) -> np.ndarray:
        """返回 frame 在 base 下的姿态，wxyz 四元数。"""

        pose = self.compute_pose(joint_positions, frame_name)
        return pose.orientation.copy()
