"""MJCF equality/frictionloss 解析与多项式关系。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import xml.etree.ElementTree as ET


def evaluate_polycoef(
    polycoef: tuple[float, ...],
    master_position: float,
) -> float:
    """按 MuJoCo 常数项在前的 ``polycoef`` 计算 follower 位置。"""

    q = float(master_position)
    return float(sum(coef * (q**power) for power, coef in enumerate(polycoef)))


def evaluate_polycoef_derivative(
    polycoef: tuple[float, ...],
    master_position: float,
) -> float:
    """计算 ``polycoef`` 对 master position 的一阶导数。"""

    q = float(master_position)
    return float(
        sum(
            power * coef * (q ** (power - 1))
            for power, coef in enumerate(polycoef[1:], start=1)
        )
    )


@dataclass(frozen=True)
class MjcfJointEquality:
    """一条 ``joint1=f(joint2)`` MJCF equality 关系。"""

    name: str
    dependent_joint: str
    master_joint: str
    polycoef: tuple[float, ...]

    def evaluate_position(self, master_position: float) -> float:
        """按 MJCF equality polynomial 从 master position 计算 follower position。"""

        return evaluate_polycoef(self.polycoef, master_position)

    def evaluate_velocity(
        self,
        master_position: float,
        master_velocity: float,
    ) -> float:
        """对 equality polynomial 求导，计算 follower velocity。"""

        return float(
            evaluate_polycoef_derivative(self.polycoef, master_position)
            * float(master_velocity)
        )


def parse_mjcf_joint_equalities(
    path: str | Path | None,
) -> list[MjcfJointEquality]:
    """按 XML 顺序读取 ``equality/joint``；无 MJCF 文件时返回空列表。"""

    if path is None:
        return []
    mjcf_path = Path(path)
    if not mjcf_path.is_file():
        return []
    root = ET.parse(mjcf_path).getroot()
    joint_names = {
        name
        for joint in root.findall("./worldbody//joint")
        if (name := str(joint.get("name", "")).strip())
    }
    equalities: list[MjcfJointEquality] = []
    dependent_names: set[str] = set()
    for element in root.findall("./equality/joint"):
        dependent_joint = str(element.get("joint1", "")).strip()
        master_joint = str(element.get("joint2", "")).strip()
        # MuJoCo also permits a one-joint equality that fixes joint1 to a
        # polynomial constant. It is not a follower relation for this runtime.
        if not dependent_joint or not master_joint:
            continue
        equality_name = str(element.get("name", f"{dependent_joint}_mimic")).strip()
        if dependent_joint == master_joint:
            raise ValueError(
                f"MJCF joint {dependent_joint!r} cannot mimic itself in {mjcf_path}"
            )
        for role, joint_name in (
            ("joint1", dependent_joint),
            ("joint2", master_joint),
        ):
            if joint_name not in joint_names:
                raise ValueError(
                    f"MJCF equality {equality_name!r} {role} references unknown "
                    f"joint {joint_name!r} in {mjcf_path}"
                )
        if dependent_joint in dependent_names:
            raise ValueError(
                f"MJCF joint {dependent_joint!r} is the follower in more than one "
                f"equality in {mjcf_path}"
            )
        polycoef_text = element.get("polycoef", "0 1 0 0 0")
        try:
            polycoef = tuple(float(value) for value in polycoef_text.split())
        except ValueError as exc:
            raise ValueError(
                "Invalid polycoef for MJCF equality "
                f"{element.get('name', '')!r}: {polycoef_text!r}"
            ) from exc
        if not polycoef:
            polycoef = (0.0, 1.0)
        if not all(isfinite(value) for value in polycoef):
            raise ValueError(
                f"MJCF equality {equality_name!r} polycoef must contain only "
                f"finite values, got {polycoef_text!r}"
            )
        dependent_names.add(dependent_joint)
        equalities.append(
            MjcfJointEquality(
                name=equality_name,
                dependent_joint=dependent_joint,
                master_joint=master_joint,
                polycoef=polycoef,
            )
        )
    _validate_acyclic_equalities(equalities, mjcf_path=mjcf_path)
    return equalities


def _validate_acyclic_equalities(
    equalities: list[MjcfJointEquality], *, mjcf_path: Path
) -> None:
    """拒绝最终递归依赖自身的 follower 有向图。

    从每个 dependent 沿 master 链追踪；一旦重复进入当前链中的关节，就报告完整环路。
    """

    master_by_dependent = {
        equality.dependent_joint: equality.master_joint for equality in equalities
    }
    for dependent in master_by_dependent:
        chain: list[str] = []
        current = dependent
        while current in master_by_dependent:
            if current in chain:
                cycle = chain[chain.index(current) :] + [current]
                raise ValueError(
                    f"MJCF mimic cycle in {mjcf_path}: {' -> '.join(cycle)}"
                )
            chain.append(current)
            current = master_by_dependent[current]


def parse_mjcf_joint_frictionloss(
    path: str | Path | None,
) -> dict[str, float]:
    """读取 ``joint name -> non-negative frictionloss``。"""

    if path is None:
        return {}
    mjcf_path = Path(path)
    if not mjcf_path.is_file():
        return {}
    root = ET.parse(mjcf_path).getroot()
    friction_by_name: dict[str, float] = {}
    for joint in root.iter("joint"):
        joint_name = joint.get("name")
        frictionloss_text = joint.get("frictionloss")
        if not joint_name or frictionloss_text is None:
            continue
        try:
            frictionloss = float(frictionloss_text or 0.0)
        except ValueError as exc:
            raise ValueError(
                f"Invalid MJCF frictionloss for joint {joint_name!r}: "
                f"{frictionloss_text!r}"
            ) from exc
        if frictionloss < 0:
            raise ValueError(
                f"MJCF frictionloss for joint {joint_name!r} cannot be negative: "
                f"{frictionloss:g}"
            )
        friction_by_name[joint_name] = frictionloss
    return friction_by_name


def mjcf_equality_follower_joint_names(path: str | Path | None) -> set[str]:
    """返回所有 equality ``joint1`` follower names。"""

    return {equality.dependent_joint for equality in parse_mjcf_joint_equalities(path)}


def expand_targets_with_mjcf_equalities(
    targets: dict[str, float],
    path: str | Path | None,
) -> dict[str, float]:
    """把稀疏 master targets 扩展为包含可推导 follower 的新字典。"""

    expanded = dict(targets)
    for equality in parse_mjcf_joint_equalities(path):
        if equality.master_joint not in expanded:
            continue
        expanded[equality.dependent_joint] = equality.evaluate_position(
            float(expanded[equality.master_joint])
        )
    return expanded


__all__ = [
    "MjcfJointEquality",
    "evaluate_polycoef",
    "evaluate_polycoef_derivative",
    "expand_targets_with_mjcf_equalities",
    "mjcf_equality_follower_joint_names",
    "parse_mjcf_joint_equalities",
    "parse_mjcf_joint_frictionloss",
]
