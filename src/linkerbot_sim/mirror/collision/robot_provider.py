"""把 Mirror 机器人当前 joint state 转换为规划碰撞球。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from linkerbot_sim.mirror.collision.envelope_provider import (
    RobotEnvelopeProvider,
)
from linkerbot_sim.mirror.collision.urdf_kinematics import _UrdfKinematics
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.planning.mounted_geometry import (
    MountedCollisionModel,
    cover_box_with_spheres,
)
from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.math_utils import make_rpy_transform
from linkerbot_sim.utils.paths import repo_path
from linkerbot_sim.utils.tensors import tensor_like_to_numpy


class RobotObstacleProvider:
    """按当前 articulation joint state 物化 profile 中的 link collision spheres。"""

    def __init__(
        self,
        *,
        robot_id: int,
        label: str,
        articulation: object,
        root_pose: RootPoseConfig,
        urdf_path: str | Path,
        collision_spheres: Mapping[str, Sequence[Mapping[str, object]]],
    ) -> None:
        self.robot_id = int(robot_id)
        self.label = str(label)
        self.articulation = articulation
        self.root_pose = root_pose
        self.urdf_path = repo_path(urdf_path)
        self._kinematics = _UrdfKinematics(self.urdf_path)
        self._spheres = _parse_collision_spheres(collision_spheres)
        self._mounted_boxes = ()
        self._flange_frame = None

    @classmethod
    def from_robot_profile(
        cls,
        *,
        robot_id: int,
        label: str,
        articulation: object,
        root_pose: RootPoseConfig,
        profile: RobotProfileSettings,
        stage: object | None = None,
        root_path: str | None = None,
    ) -> "RobotObstacleProvider | RobotEnvelopeProvider | None":
        """优先创建 link-sphere provider，缺少 cuRobo model 时回退到 root envelope。"""

        if not isinstance(profile, RobotProfileSettings):
            raise TypeError("profile must be RobotProfileSettings")
        robot = profile.curobo.robot
        if robot is None:
            return RobotEnvelopeProvider.from_profile(
                robot_id=robot_id,
                label=label,
                root_pose=root_pose,
                robot_profile=profile,
            )
        config_path = robot.robot_config_path
        urdf_path = robot.urdf_path
        if config_path is None or urdf_path is None:
            return None
        config = load_yaml(config_path)
        spheres = _nested_kinematics(config).get("collision_spheres")
        if not isinstance(spheres, Mapping) or not spheres:
            return None
        provider = cls(
            robot_id=robot_id,
            label=label,
            articulation=articulation,
            root_pose=root_pose,
            urdf_path=urdf_path,
            collision_spheres=spheres,
        )
        mounted = profile.planning_collision
        if mounted is not None and mounted.mounted_urdf is not None:
            if stage is None or root_path is None or robot.flange_frame is None:
                raise ValueError(
                    "mounted planning geometry requires stage, robot root and flange_frame"
                )
            from linkerbot_sim.mirror.collision.collider_bounds import (
                read_link_collider_boxes,
            )

            provider._kinematics = _UrdfKinematics(repo_path(mounted.mounted_urdf))
            links = provider._kinematics.link_transforms({})
            if robot.flange_frame not in links:
                raise ValueError(
                    f"mounted URDF is missing flange frame {robot.flange_frame!r}"
                )
            descendants = {robot.flange_frame}
            for _ in provider._kinematics.joints:
                descendants.update(
                    j.child
                    for j in provider._kinematics.joints
                    if j.parent in descendants
                )
            boxes = read_link_collider_boxes(stage, root_path, set(links))
            provider._mounted_boxes = tuple(
                box for box in boxes if box.link_name in descendants
            )
            if not provider._mounted_boxes:
                raise ValueError(
                    "mounted planning geometry has no colliders below the flange"
                )
            provider._flange_frame = robot.flange_frame
            # Replace the legacy generic gripper sphere, not the arm's link spheres.
            provider._spheres.pop(robot.flange_frame, None)
        return provider

    def _link_transforms(self):
        names = tuple(str(name) for name in getattr(self.articulation, "dof_names", ()))
        values = tensor_like_to_numpy(
            self.articulation.get_joint_positions(), dtype=float
        ).reshape(-1)
        if len(names) != values.size or not np.all(np.isfinite(values)):
            raise ValueError(f"robot {self.label!r} DOF names/positions are invalid")
        return self._kinematics.link_transforms(dict(zip(names, values, strict=True)))

    def mounted_collision_model(self) -> MountedCollisionModel | None:
        """Freeze hand shape in flange coordinates; candidate arm FK moves this model."""
        if self._flange_frame is None:
            return None
        links = self._link_transforms()
        inverse = np.linalg.inv(links[self._flange_frame])
        spheres = []
        sphere_links = []
        for box in self._mounted_boxes:
            pose = inverse @ links[box.link_name]
            cells = cover_box_with_spheres(box.size)
            cells[:, :3] = (cells[:, :3] + box.center) @ pose[:3, :3].T + pose[:3, 3]
            # Suppress harmless FK float noise; changes below a nanometre do not
            # rebuild CUDA models as the arm moves with unchanged hand shape.
            spheres.extend(tuple(float(v) for v in row) for row in cells.round(9))
            sphere_links.extend([box.link_name] * len(cells))
        return MountedCollisionModel(
            self.robot_id,
            self._flange_frame,
            tuple(spheres),
            tuple(sorted({box.link_name for box in self._mounted_boxes})),
            sphere_links=tuple(sphere_links),
        )

    def flange_world_pose(self) -> np.ndarray:
        if self._flange_frame is None:
            raise ValueError("robot has no configured mounted collision geometry")
        return (
            make_rpy_transform(self.root_pose.xyz, self.root_pose.rpy)
            @ self._link_transforms()[self._flange_frame]
        )

    def collision_objects(self) -> tuple[CollisionObject, ...]:
        """按 articulation 当前 joint state 计算所有 link sphere 的 world pose。"""

        links = self._link_transforms()
        root = make_rpy_transform(self.root_pose.xyz, self.root_pose.rpy)
        result = []
        for link_name, spheres in self._spheres.items():
            if link_name not in links:
                raise ValueError(
                    f"collision sphere link {link_name!r} is absent from {self.urdf_path}"
                )
            link_pose = root @ links[link_name]
            for index, (center, radius) in enumerate(spheres):
                pose = link_pose.copy()
                pose[:3, 3] = link_pose[:3, :3] @ center + link_pose[:3, 3]
                result.append(
                    CollisionObject(
                        name=f"robot_{self.robot_id}_{link_name}_{index}",
                        shape="sphere",
                        pose=pose,
                        size=(radius,),
                    )
                )
        for box in self._mounted_boxes:
            pose = root @ links[box.link_name]
            pose[:3, 3] += pose[:3, :3] @ np.asarray(box.center)
            result.append(
                CollisionObject(
                    name=f"robot_{self.robot_id}_{box.name}",
                    shape="cuboid",
                    pose=pose,
                    size=box.size,
                )
            )
        return tuple(result)


def _nested_kinematics(config: Mapping[str, object]) -> Mapping[str, object]:
    """读取 cuRobo YAML 中必需的 ``robot_cfg.kinematics`` mapping。"""

    robot_cfg = config.get("robot_cfg")
    if not isinstance(robot_cfg, Mapping):
        raise ValueError("cuRobo robot config requires robot_cfg mapping")
    kinematics = robot_cfg.get("kinematics")
    if not isinstance(kinematics, Mapping):
        raise ValueError("cuRobo robot config requires robot_cfg.kinematics")
    return kinematics


def _parse_collision_spheres(values):
    """解析每个 link 的正半径 collision sphere，并冻结 center array。"""

    result = {}
    for link_name, entries in values.items():
        parsed = []
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"collision_spheres.{link_name} must be a list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"collision_spheres.{link_name}[] must be a mapping")
            center = np.asarray(entry.get("center", ()), dtype=float).reshape(3)
            radius = float(entry.get("radius", 0.0))
            if radius <= 0:
                continue
            parsed.append((center, radius))
        if parsed:
            result[str(link_name)] = tuple(parsed)
    return result


__all__ = ["RobotObstacleProvider"]
