"""cuMotion 正运动学封装。

FK 封装只读取 cuMotion kinematics，不修改机器人状态。输入关节向量必须按 cuMotion
C-space 关节顺序排列；输出位姿位于机器人 base/world 约定坐标系下，位置单位为 m，姿态在
项目边界统一转换为 ``wxyz`` 四元数。

职责边界:
    * 不从 Isaac articulation 读取关节状态；调用方必须先按名称取出 cuMotion C-space 子向量。
    * 不缓存 FK 结果，避免连续仿真中误用过期姿态。
    * 不做 TCP 偏移推导；frame 必须已经存在于 cuMotion robot description。
"""

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
    """封装 FK 和 frame/joint 查询。

    该类通常由 ``CuMotionContext.make_forward_kinematics`` 创建，共享已加载的 robot
    description。它不缓存单次 FK 结果，避免调用方误用过期关节状态。
    """

    def __init__(self, context) -> None:
        """保存共享上下文和 kinematics 引用。

        参数:
            context: 已加载 cuMotion robot description 的 ``CuMotionContext``。
        """

        self.context = context
        self.kinematics = context.kinematics

    def joint_names(self) -> list[str]:
        """返回 FK 输入向量使用的 C-space 关节名顺序。

        调用 ``compute_*`` 时传入的关节数组必须按该顺序排列，通常是从完整 Isaac DOF 中按
        名称抽取出的子向量。
        """

        return self.context.joint_names()

    def frame_names(self) -> list[str]:
        """返回当前机器人描述中可查询 FK 的 frame 名。

        只有这些 frame 可以作为 ``compute_pose`` 的 ``frame_name``；自定义 TCP 需要先写入
        URDF/robot description 后才会出现在这里。
        """

        return self.context.frame_names()

    def compute_cumotion_pose(self, joint_positions, frame_name: str):
        """返回后端原生 cuMotion ``Pose3``。

        ``joint_positions`` 会被拉平成一维数组，但长度是否匹配 C-space 由 cuMotion
        kinematics 报错。需要项目统一 pose 结构时使用 ``compute_pose``。
        """

        return self.kinematics.pose(
            np.asarray(joint_positions, dtype=float).reshape(-1), str(frame_name)
        )

    def compute_pose(self, joint_positions, frame_name: str) -> ForwardKinematicsPose:
        """返回 frame 在 base 下的完整位姿。

        姿态统一转换为项目外部接口常用的 wxyz 四元数，同时保留旋转矩阵，便于
        IK 请求、日志和后续笛卡尔轨迹生成直接复用。
        """

        # cuMotion rotation 对象暴露 w/x/y/z 方法和 matrix；这里同时导出四元数和矩阵，
        # 让后续 IK、轨迹采样和调试日志不需要再次访问后端对象。
        cumotion_pose = self.compute_cumotion_pose(joint_positions, frame_name)
        rotation = cumotion_pose.rotation
        return ForwardKinematicsPose(
            position=np.asarray(cumotion_pose.translation, dtype=float).reshape(3),
            orientation=np.asarray(
                [rotation.w(), rotation.x(), rotation.y(), rotation.z()], dtype=float
            ),
            rotation_matrix=np.asarray(rotation.matrix(), dtype=float).reshape(3, 3),
        )

    def compute_position(self, joint_positions, frame_name: str) -> np.ndarray:
        """返回 frame 在 base 下的位置副本，shape 为 ``(3,)``。

        这是 ``compute_pose(...).position`` 的便捷入口，返回 copy，调用方修改它不会影响
        缓存或后端对象。
        """

        pose = self.compute_pose(joint_positions, frame_name)
        return pose.position.copy()

    def compute_orientation(self, joint_positions, frame_name: str) -> np.ndarray:
        """返回 frame 在 base 下的姿态副本，格式为 wxyz 四元数。

        这是 ``compute_pose(...).orientation`` 的便捷入口，适合直接作为 IK 请求的姿态目标。
        """

        pose = self.compute_pose(joint_positions, frame_name)
        return pose.orientation.copy()
