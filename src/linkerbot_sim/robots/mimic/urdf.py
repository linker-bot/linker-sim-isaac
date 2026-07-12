"""URDF ``<mimic>`` 关节关系的严格解析。

解析器只检查 ``<robot>`` 的直接 ``<joint>`` 子元素，验证关节名唯一、主关节存在、数值有限
且 follower 链无环，并保留文档声明顺序。结果是纯 Python 不可变数据，不依赖 URDF importer
或 Isaac stage。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class UrdfJointMimic:
    """一条 URDF 从属关节到主关节的仿射关系。

    关系为 ``dependent = multiplier * master + offset``；两个数值保证为有限 float。
    ``dependent_joint`` 与 ``master_joint`` 在解析结果中都引用同一 URDF 已声明的关节。
    """

    dependent_joint: str
    master_joint: str
    multiplier: float = 1.0
    offset: float = 0.0

    @property
    def polycoef(self) -> tuple[float, float]:
        """按共享的常数项优先顺序返回 ``(offset, multiplier)``。"""

        return (self.offset, self.multiplier)


def parse_urdf_joint_mimics(path: str | Path | None) -> list[UrdfJointMimic]:
    """按文档顺序解析 URDF mimic 声明。

    参数:
        path: URDF 文件路径；``None`` 或不存在的普通文件表示无关系并返回空列表。
    返回:
        已验证、保持源文档顺序的 :class:`UrdfJointMimic` 列表。
    异常:
        OSError: 路径存在但读取失败。
        xml.etree.ElementTree.ParseError: XML 格式错误。
        ValueError: 关节名/引用重复或缺失、数值非有限、自引用或依赖链有环。
    副作用:
        仅读取 XML 文件，不修改资产。
    """

    if path is None:
        return []
    urdf_path = Path(path)
    if not urdf_path.is_file():
        return []
    root = ET.parse(urdf_path).getroot()
    # URDF 运动学关节是 robot 的直接子元素。transmission/gazebo 扩展中的后代
    # ``joint`` 只是引用，若递归搜索会把它们误判为重复声明。
    joints = list(root.findall("./joint"))
    joint_names: set[str] = set()
    for joint in joints:
        name = str(joint.get("name", "")).strip()
        if not name:
            if joint.find("mimic") is not None:
                raise ValueError(f"URDF mimic joint in {urdf_path} requires a name")
            continue
        if name in joint_names:
            raise ValueError(f"URDF joint {name!r} is declared more than once")
        joint_names.add(name)

    relations: list[UrdfJointMimic] = []
    dependent_names: set[str] = set()
    for joint in joints:
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        dependent = str(joint.get("name", "")).strip()
        master = str(mimic.get("joint", "")).strip()
        if not dependent or not master:
            raise ValueError(
                f"URDF mimic in {urdf_path} requires dependent and master joint names"
            )
        if dependent == master:
            raise ValueError(f"URDF joint {dependent!r} cannot mimic itself")
        if master not in joint_names:
            raise ValueError(
                f"URDF joint {dependent!r} mimics unknown joint {master!r}"
            )
        if dependent in dependent_names:
            raise ValueError(f"URDF joint {dependent!r} declares mimic more than once")
        multiplier = _finite_float(
            mimic.get("multiplier", "1"),
            label=f"URDF mimic {dependent!r} multiplier",
        )
        offset = _finite_float(
            mimic.get("offset", "0"),
            label=f"URDF mimic {dependent!r} offset",
        )
        dependent_names.add(dependent)
        relations.append(
            UrdfJointMimic(
                dependent_joint=dependent,
                master_joint=master,
                multiplier=multiplier,
                offset=offset,
            )
        )
    _validate_acyclic_relations(relations, urdf_path=urdf_path)
    return relations


def _validate_acyclic_relations(
    relations: list[UrdfJointMimic], *, urdf_path: Path
) -> None:
    """拒绝无法按依赖顺序求值的 follower 环。"""

    master_by_dependent = {
        relation.dependent_joint: relation.master_joint for relation in relations
    }
    for dependent in master_by_dependent:
        chain: list[str] = []
        current = dependent
        while current in master_by_dependent:
            if current in chain:
                cycle = chain[chain.index(current) :] + [current]
                raise ValueError(
                    f"URDF mimic cycle in {urdf_path}: {' -> '.join(cycle)}"
                )
            chain.append(current)
            current = master_by_dependent[current]


def _finite_float(value: str, *, label: str) -> float:
    """把 XML 属性严格转换为有限 float，并保留字段标签用于定位。"""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number, got {value!r}") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


__all__ = ["UrdfJointMimic", "parse_urdf_joint_mimics"]
