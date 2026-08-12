"""Mirror 在没有 cuRobo link spheres 时使用的保守机器人根包络。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configuration.robots import RobotProfileSettings
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
        robot_profile: RobotProfileSettings,
    ) -> "RobotEnvelopeProvider | None":
        """从 ``robot.planning_collision.spheres`` 创建保守 root envelope。"""

        if not isinstance(robot_profile, RobotProfileSettings):
            raise TypeError("robot_profile must be RobotProfileSettings")
        collision = robot_profile.planning_collision
        if collision is None:
            return None
        spheres = tuple(
            (
                sphere.name,
                np.asarray(sphere.center, dtype=float),
                sphere.radius,
            )
            for sphere in collision.spheres
        )
        return cls(
            robot_id=robot_id,
            label=label,
            root_pose=root_pose,
            spheres=spheres,
        )

    def collision_objects(self) -> tuple[CollisionObject, ...]:
        """把 root-local spheres 变换到 world，并附加 robot ownership 名称。"""

        root = make_rpy_transform(self.root_pose.xyz, self.root_pose.rpy)
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


__all__ = ["RobotEnvelopeProvider"]
