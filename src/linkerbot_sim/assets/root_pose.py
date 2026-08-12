"""机器人 root pose 数据与 USD/MJCF fixed-root 写入。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RootPoseConfig:
    """机器人或场景对象 root 在世界坐标下的固定位姿。"""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def is_identity(self) -> bool:
        """是否为零平移、零旋转。"""

        return self.xyz == (0.0, 0.0, 0.0) and self.rpy == (0.0, 0.0, 0.0)


def apply_root_pose_transform(
    stage,
    root_path: str,
    pose: RootPoseConfig,
    *,
    prepare_newton_render_topology: bool = False,
) -> None:
    """按显式 render intent 写 root xform，不混用两套 op topology。"""

    if prepare_newton_render_topology:
        from linkerbot_sim.isaac.physics.newton.render import (
            author_newton_render_root_pose,
        )

        author_newton_render_root_pose(
            stage=stage,
            root_path=root_path,
            pose=pose,
        )
        return

    from pxr import Gf, Sdf, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not prim.IsValid():
        raise RuntimeError(f"Cannot apply root_pose; prim not found: {root_path}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pose.xyz))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*tuple(np.degrees(pose.rpy))))


def apply_root_pose(
    stage,
    root_path: str,
    pose: RootPoseConfig,
    *,
    prepare_newton_render_topology: bool = False,
) -> None:
    """写入 root prim 的世界位姿，并同步 MJCF fixed-base world anchor。"""

    apply_root_pose_transform(
        stage,
        root_path,
        pose,
        prepare_newton_render_topology=prepare_newton_render_topology,
    )
    apply_mjcf_fixed_root_joint_pose(stage, root_path, pose)


def apply_mjcf_fixed_root_joint_pose(
    stage, root_path: str, pose: RootPoseConfig
) -> None:
    """只同步 MJCF fixed-base joint 的 world anchor，不移动机器人 Xform。"""

    from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz
    from pxr import Gf, Sdf, UsdPhysics

    quat = rpy_xyz_to_quat_wxyz(pose.rpy)
    world_anchor_pos = Gf.Vec3f(*pose.xyz)
    world_anchor_rot = Gf.Quatf(
        float(quat[0]),
        Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3])),
    )
    for joint_path in mjcf_fixed_root_joint_paths_without_body0(stage, root_path):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(Sdf.Path(joint_path)))
        joint.CreateLocalPos0Attr().Set(world_anchor_pos)
        joint.CreateLocalRot0Attr().Set(world_anchor_rot)


def mjcf_fixed_root_joint_paths_without_body0(stage, root_path: str) -> tuple[str, ...]:
    """返回 root 子树内 body0 为空的 MJCF fixed-base joints。"""

    from pxr import Sdf, Usd

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        return ()
    result: list[str] = []
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "PhysicsFixedJoint":
            continue
        if not prim.GetName().startswith("rootJoint_"):
            continue
        if not prim.GetRelationship("physics:body0").GetTargets():
            result.append(str(prim.GetPath()))
    return tuple(result)


__all__ = [
    "RootPoseConfig",
    "apply_mjcf_fixed_root_joint_pose",
    "apply_root_pose",
    "apply_root_pose_transform",
    "mjcf_fixed_root_joint_paths_without_body0",
]
