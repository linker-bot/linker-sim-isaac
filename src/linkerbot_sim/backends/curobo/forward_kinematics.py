"""cuRobo 正运动学封装。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linkerbot_sim.utils.rotations import quat_wxyz_to_matrix


@dataclass(frozen=True)
class ForwardKinematicsPose:
    """项目内部使用的 FK 位姿结果。"""

    position: np.ndarray
    orientation: np.ndarray
    rotation_matrix: np.ndarray


class CuroboForwardKinematics:
    """把 ``CuroboContext.compute_tcp_poses`` 封装成单点 FK API。"""

    def __init__(self, context) -> None:
        """保存共享 cuRobo context。"""

        self.context = context

    def joint_names(self) -> list[str]:
        """返回 FK 输入向量使用的 C-space 关节名顺序。"""

        return self.context.joint_names()

    def frame_names(self) -> list[str]:
        """返回当前 context 注册的 tool frames。"""

        return self.context.frame_names()

    def compute_pose(self, joint_positions, frame_name: str) -> ForwardKinematicsPose:
        """返回 frame 在 cuRobo base 下的完整位姿。"""

        positions, orientations = self.context.compute_tcp_poses(
            np.asarray(joint_positions, dtype=float).reshape(1, -1),
            tcp_frame_name=str(frame_name),
        )
        quat = np.asarray(orientations, dtype=float).reshape(-1, 4)[0]
        return ForwardKinematicsPose(
            position=np.asarray(positions, dtype=float).reshape(-1, 3)[0],
            orientation=quat,
            rotation_matrix=quat_wxyz_to_matrix(quat),
        )

    def compute_position(self, joint_positions, frame_name: str) -> np.ndarray:
        """返回 frame 在 base 下的位置副本。"""

        return self.compute_pose(joint_positions, frame_name).position.copy()

    def compute_orientation(self, joint_positions, frame_name: str) -> np.ndarray:
        """返回 frame 在 base 下的 wxyz 四元数副本。"""

        return self.compute_pose(joint_positions, frame_name).orientation.copy()
