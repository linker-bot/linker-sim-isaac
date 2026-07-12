"""没有 cuRobo link spheres 时使用的保守 robot root envelope。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.utils.math_utils import make_rpy_transform


class RobotEnvelopeProvider:
    """把 profile 中显式声明的球体固定到机器人 root。"""

    def __init__(
        self,
        *,
        robot_id: int,
        label: str,
        root_pose: RootPoseConfig,
        spheres: Sequence[tuple[str, np.ndarray, float]],
    ) -> None:
        self.robot_id = int(robot_id)
        self.label = str(label)
        self.root_pose = root_pose
        self.spheres = tuple(spheres)

    @classmethod
    def from_profile(
        cls,
        *,
        robot_id: int,
        label: str,
        root_pose: RootPoseConfig,
        robot_profile: Mapping[str, object],
    ) -> "RobotEnvelopeProvider | None":
        """从 ``robot.planning_collision.spheres`` 创建保守 root envelope。"""

        collision = robot_profile.get("planning_collision")
        if not isinstance(collision, Mapping):
            return None
        values = collision.get("spheres")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("robot.planning_collision.spheres must be a list")
        spheres = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise ValueError("robot.planning_collision.spheres[] must be a mapping")
            radius = float(value.get("radius", 0.0))
            if radius <= 0:
                raise ValueError(
                    "robot planning collision sphere radius must be positive"
                )
            spheres.append(
                (
                    str(value.get("name", f"sphere_{index}")),
                    np.asarray(value.get("center", ()), dtype=float).reshape(3),
                    radius,
                )
            )
        return cls(
            robot_id=robot_id,
            label=label,
            root_pose=root_pose,
            spheres=spheres,
        )

    def collision_objects(self) -> tuple[CollisionObject, ...]:
        """把 root-local spheres 变换到 world，并附加 robot ownership 名称。"""

        root = _root_pose_matrix(self.root_pose)
        result = []
        for name, center, radius in self.spheres:
            pose = root.copy()
            pose[:3, 3] = root[:3, :3] @ center + root[:3, 3]
            result.append(
                CollisionObject(
                    name=f"robot_{self.robot_id}_{name}",
                    shape="sphere",
                    pose=pose,
                    size=(radius,),
                )
            )
        return tuple(result)


def _root_pose_matrix(pose: RootPoseConfig) -> np.ndarray:
    """构造 envelope 使用的 robot root world transform。"""

    return make_rpy_transform(pose.xyz, pose.rpy)


__all__ = ["RobotEnvelopeProvider"]
