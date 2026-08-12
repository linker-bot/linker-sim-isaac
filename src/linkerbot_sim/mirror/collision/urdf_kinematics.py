"""Mirror 规划碰撞球使用的轻量 URDF forward kinematics。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from linkerbot_sim.utils.math_utils import make_rpy_transform


@dataclass(frozen=True)
class _UrdfJoint:
    """轻量 URDF joint：只保留 collision sphere FK 所需字段。"""

    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class _UrdfKinematics:
    """只覆盖 collision sphere 定位所需的 fixed/revolute/prismatic 关节。"""

    def __init__(self, path: Path) -> None:
        root = ET.parse(path).getroot()
        joints = []
        child_links = set()
        parent_links = set()
        for element in root.findall("joint"):
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                continue
            parent_name = str(parent.attrib["link"])
            child_name = str(child.attrib["link"])
            joints.append(
                _UrdfJoint(
                    name=str(element.attrib.get("name", "")),
                    joint_type=str(element.attrib.get("type", "fixed")),
                    parent=parent_name,
                    child=child_name,
                    origin=_urdf_origin(element.find("origin")),
                    axis=_urdf_axis(element.find("axis")),
                )
            )
            child_links.add(child_name)
            parent_links.add(parent_name)
        roots = sorted(parent_links - child_links)
        if not roots:
            raise ValueError(f"URDF has no root link: {path}")
        self.root_link = roots[0]
        self.joints = tuple(joints)

    def link_transforms(self, values: Mapping[str, float]) -> dict[str, np.ndarray]:
        """从 root 开始拓扑展开 joint tree，返回每个 link 的 root-local transform。"""

        transforms = {self.root_link: np.eye(4, dtype=float)}
        unresolved = list(self.joints)
        while unresolved:
            progress = False
            remaining = []
            for joint in unresolved:
                parent = transforms.get(joint.parent)
                if parent is None:
                    remaining.append(joint)
                    continue
                transforms[joint.child] = (
                    parent
                    @ joint.origin
                    @ _joint_motion(joint, values.get(joint.name, 0.0))
                )
                progress = True
            if not progress:
                missing = [joint.name for joint in remaining]
                raise ValueError(f"URDF joint tree cannot be resolved: {missing}")
            unresolved = remaining
        return transforms


def _urdf_origin(element) -> np.ndarray:
    """解析 URDF origin xyz/rpy；缺失时返回 identity。"""

    if element is None:
        return np.eye(4, dtype=float)
    xyz = np.fromstring(element.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.attrib.get("rpy", "0 0 0"), sep=" ")
    return make_rpy_transform(xyz, rpy)


def _urdf_axis(element) -> np.ndarray:
    """解析并归一化 URDF joint axis，拒绝零向量。"""

    axis = (
        np.asarray([1.0, 0.0, 0.0])
        if element is None
        else np.fromstring(element.attrib.get("xyz", "1 0 0"), sep=" ")
    )
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("URDF joint axis cannot be zero")
    return axis / norm


def _joint_motion(joint: _UrdfJoint, value: float) -> np.ndarray:
    """按 joint type 构造 revolute/continuous rotation 或 prismatic translation。"""

    result = np.eye(4, dtype=float)
    if joint.joint_type in {"revolute", "continuous"}:
        result[:3, :3] = Rotation.from_rotvec(joint.axis * float(value)).as_matrix()
    elif joint.joint_type == "prismatic":
        result[:3, 3] = joint.axis * float(value)
    return result
