"""机器人 root pose 数据与 USD/MJCF fixed-root 写入。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RootPoseConfig:
    """机器人或场景对象 root 在世界坐标下的固定位姿。"""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "RootPoseConfig":
        """从可选 ``root_pose`` mapping 解析弧度制 xyz/rpy。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("root_pose must be a mapping")
        return cls(
            xyz=_vec3_from_mapping(data, "xyz"),
            rpy=_vec3_from_mapping(data, "rpy"),
        )

    def is_identity(self) -> bool:
        """是否为零平移、零旋转。"""

        return self.xyz == (0.0, 0.0, 0.0) and self.rpy == (0.0, 0.0, 0.0)


def apply_root_pose(stage, root_path: str, pose: RootPoseConfig) -> None:
    """写入 root prim 的世界位姿，并同步 MJCF fixed-base world anchor。"""

    from pxr import Gf, Sdf, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not prim.IsValid():
        raise RuntimeError(f"Cannot apply root_pose; prim not found: {root_path}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pose.xyz))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*tuple(np.degrees(pose.rpy))))
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


def _vec3_from_mapping(
    data: Mapping[str, object], key: str
) -> tuple[float, float, float]:
    """读取缺省为零的 xyz/rpy 三元组，并拒绝非 sequence 或错误长度。"""

    value = data.get(key, (0.0, 0.0, 0.0))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a length-3 sequence")
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{key} must contain exactly 3 values")
    return values


__all__ = [
    "RootPoseConfig",
    "apply_mjcf_fixed_root_joint_pose",
    "apply_root_pose",
    "mjcf_fixed_root_joint_paths_without_body0",
]
