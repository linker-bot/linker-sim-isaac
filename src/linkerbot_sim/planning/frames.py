"""显式执行 world、env、robot base 与 TCP 之间的 pose/offset 变换。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.utils.math_utils import make_rpy_transform, make_transform
from linkerbot_sim.utils.rotations import (
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
)


SUPPORTED_REFERENCE_FRAMES = frozenset({"world", "env", "robot_base", "tcp"})


@dataclass(frozen=True)
class PoseInRobotBase:
    """已经转换到 robot-base-local 的 position 与可选 wxyz orientation。"""

    position: np.ndarray
    orientation_wxyz: np.ndarray | None


@dataclass(frozen=True)
class FrameTransformer:
    """把 task-space request 数据转换到 robot-base-local planning frame。

    所有输入 transform 都采用 ``target_from_source`` 约定；offset 只应用 rotation，pose
    同时应用 rotation/translation。TCP frame 需要命令开始时冻结的当前 TCP pose。
    """

    world_from_robot_base: np.ndarray
    world_from_env: np.ndarray
    robot_base_from_tcp: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "world_from_robot_base",
            _pose_matrix(self.world_from_robot_base, "world_from_robot_base"),
        )
        object.__setattr__(
            self,
            "world_from_env",
            _pose_matrix(self.world_from_env, "world_from_env"),
        )
        if self.robot_base_from_tcp is not None:
            object.__setattr__(
                self,
                "robot_base_from_tcp",
                _pose_matrix(self.robot_base_from_tcp, "robot_base_from_tcp"),
            )

    @classmethod
    def from_root_pose(
        cls,
        root_pose: RootPoseConfig,
        *,
        world_from_env: np.ndarray | None = None,
        tcp_position_in_base: np.ndarray | None = None,
        tcp_orientation_wxyz_in_base: np.ndarray | None = None,
    ) -> "FrameTransformer":
        """从 robot root pose、env origin 和可选当前 TCP pose 构造 transformer。"""

        world_from_base = make_rpy_transform(root_pose.xyz, root_pose.rpy)
        base_from_tcp = None
        if tcp_position_in_base is not None:
            base_from_tcp = make_transform(
                tcp_position_in_base,
                (
                    None
                    if tcp_orientation_wxyz_in_base is None
                    else quat_wxyz_to_matrix(tcp_orientation_wxyz_in_base)
                ),
            )
        return cls(
            world_from_robot_base=world_from_base,
            world_from_env=(
                np.eye(4, dtype=float)
                if world_from_env is None
                else np.asarray(world_from_env, dtype=float)
            ),
            robot_base_from_tcp=base_from_tcp,
        )

    def pose_to_robot_base(
        self,
        *,
        position: np.ndarray,
        orientation_wxyz: np.ndarray | None,
        reference_frame: str,
    ) -> PoseInRobotBase:
        """把指定 frame 中的绝对 pose 转换到 robot base；orientation 可不约束。"""

        source = self._world_from_frame(reference_frame)
        world_pose = source.copy()
        world_pose[:3, 3] = (
            source[:3, :3] @ np.asarray(position, dtype=float).reshape(3)
            + source[:3, 3]
        )
        if orientation_wxyz is None:
            orientation = None
        else:
            world_pose[:3, :3] = source[:3, :3] @ quat_wxyz_to_matrix(orientation_wxyz)
            base_pose = np.linalg.inv(self.world_from_robot_base) @ world_pose
            orientation = matrix_to_quat_wxyz(base_pose[:3, :3])
        base_position = (np.linalg.inv(self.world_from_robot_base) @ world_pose)[:3, 3]
        return PoseInRobotBase(base_position, orientation)

    def offset_to_robot_base(
        self,
        offset: np.ndarray,
        *,
        offset_frame: str,
    ) -> np.ndarray:
        """只旋转相对 offset 到 robot base，不引入 frame translation。"""

        source = self._world_from_frame(offset_frame)
        world_offset = source[:3, :3] @ np.asarray(offset, dtype=float).reshape(3)
        return self.world_from_robot_base[:3, :3].T @ world_offset

    def _world_from_frame(self, frame: str) -> np.ndarray:
        """解析 public frame 名称到 world transform，并校验 TCP pose 是否可用。"""

        normalized = str(frame).lower()
        if normalized not in SUPPORTED_REFERENCE_FRAMES:
            raise ValueError(
                f"reference frame must be one of {sorted(SUPPORTED_REFERENCE_FRAMES)}"
            )
        if normalized == "world":
            return np.eye(4, dtype=float)
        if normalized == "env":
            return self.world_from_env
        if normalized == "robot_base":
            return self.world_from_robot_base
        if self.robot_base_from_tcp is None:
            raise ValueError("tcp frame transform requires a current TCP pose")
        return self.world_from_robot_base @ self.robot_base_from_tcp


def _pose_matrix(value: np.ndarray, label: str) -> np.ndarray:
    """复制并校验 shape=(4,4) 的 homogeneous transform。"""

    matrix = np.asarray(value, dtype=float).reshape(4, 4).copy()
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{label} must be a homogeneous transform")
    return matrix


__all__ = ["FrameTransformer", "PoseInRobotBase", "SUPPORTED_REFERENCE_FRAMES"]
