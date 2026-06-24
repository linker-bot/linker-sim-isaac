"""MJCF equality/mimic 关系解析与运行时目标映射。

灵巧手的某些从动关节通过 MJCF ``<equality><joint ...>`` 描述 mimic
关系。Isaac 导入后不一定会自动帮控制器维护这些从动目标，因此这里把 MJCF
里的多项式关系解析出来，在运行时显式生成 follower 关节的目标位置和速度。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def evaluate_polycoef(polycoef: tuple[float, ...], master_position: float) -> float:
    """按 MJCF ``polycoef`` 多项式计算位置。

    参数:
        polycoef: 多项式系数 ``(a0, a1, a2, ...)``。
        master_position: 主动关节位置，单位 rad。
    返回:
        ``a0 + a1*q + a2*q^2 + ...`` 的计算结果。
    """

    q = float(master_position)
    return float(sum(coef * (q**power) for power, coef in enumerate(polycoef)))


def evaluate_polycoef_derivative(polycoef: tuple[float, ...], master_position: float) -> float:
    """按 MJCF ``polycoef`` 多项式计算对主动关节位置的导数。

    参数:
        polycoef: 多项式系数 ``(a0, a1, a2, ...)``。
        master_position: 主动关节位置，单位 rad。
    返回:
        ``a1 + 2*a2*q + 3*a3*q^2 + ...`` 的计算结果。
    """

    q = float(master_position)
    return float(sum(power * coef * (q ** (power - 1)) for power, coef in enumerate(polycoef[1:], start=1)))


@dataclass(frozen=True)
class MjcfJointEquality:
    """一条 MJCF joint equality 关系。

    输入字段:
        name: equality 名称。
        dependent_joint: 从动关节名，对应 MJCF ``joint1``。
        master_joint: 主动关节名，对应 MJCF ``joint2``。
        polycoef: MJCF 多项式系数。
    输出:
        ``evaluate_position`` 和 ``evaluate_velocity`` 可按多项式关系计算 follower 目标。
    """

    name: str
    dependent_joint: str
    master_joint: str
    polycoef: tuple[float, ...]

    def evaluate_position(self, master_position: float) -> float:
        """按 MJCF ``polycoef`` 计算 follower 位置。

        参数:
            master_position: 主动关节位置，单位 rad。
        返回:
            从动关节目标位置，单位 rad。
        """

        return evaluate_polycoef(self.polycoef, master_position)

    def evaluate_velocity(self, master_position: float, master_velocity: float) -> float:
        """按多项式导数计算 follower 速度。

        参数:
            master_position: 主动关节位置，单位 rad。
            master_velocity: 主动关节速度，单位 rad/s。
        返回:
            从动关节目标速度，单位 rad/s。
        """

        return float(evaluate_polycoef_derivative(self.polycoef, master_position) * float(master_velocity))


@dataclass(frozen=True)
class MimicFollowerControl:
    """从“实际主动 DOF 状态”到“从动 DOF 目标”的运行时映射。

    输入字段:
        dependent_joint/master_joint: 从动/主动关节名。
        dependent_index/master_index: 两者在完整 DOF 数组中的下标。
        polycoef: MJCF 多项式系数。
    输出:
        供 ``MimicFollowerTargetMapper`` 原地更新完整 follower 目标数组。
    """

    dependent_joint: str
    master_joint: str
    dependent_index: int
    master_index: int
    polycoef: tuple[float, ...]

    def evaluate_position(self, master_position: float) -> float:
        """按多项式计算 follower 位置。

        参数:
            master_position: 主动关节位置，单位 rad。
        返回:
            从动关节位置，单位 rad。
        """

        return evaluate_polycoef(self.polycoef, master_position)

    def evaluate_velocity(self, master_position: float, master_velocity: float) -> float:
        """按多项式导数计算 follower 速度。

        参数:
            master_position: 主动关节位置，单位 rad。
            master_velocity: 主动关节速度，单位 rad/s。
        返回:
            从动关节速度，单位 rad/s。
        """

        return float(evaluate_polycoef_derivative(self.polycoef, master_position) * float(master_velocity))

def parse_mjcf_joint_equalities(path: str | Path | None) -> list[MjcfJointEquality]:
    """读取 MJCF 文件中的 ``equality/joint`` mimic 关系。

    参数:
        path: MJCF 文件路径；为 ``None`` 或不存在时返回空列表。
    返回:
        ``MjcfJointEquality`` 列表，按 XML 中出现顺序排列。
    """

    if path is None:
        return []
    mjcf_path = Path(path)
    if not mjcf_path.is_file():
        return []

    root = ET.parse(mjcf_path).getroot()
    equalities: list[MjcfJointEquality] = []
    for element in root.findall("./equality/joint"):
        dependent_joint = element.get("joint1")
        master_joint = element.get("joint2")
        if not dependent_joint or not master_joint:
            continue
        polycoef_text = element.get("polycoef", "0 1 0 0 0")
        try:
            polycoef = tuple(float(value) for value in polycoef_text.split())
        except ValueError as exc:
            raise ValueError(f"Invalid polycoef for MJCF equality {element.get('name', '')!r}: {polycoef_text!r}") from exc
        if not polycoef:
            polycoef = (0.0, 1.0)
        equalities.append(
            MjcfJointEquality(
                name=element.get("name", f"{dependent_joint}_mimic"),
                dependent_joint=dependent_joint,
                master_joint=master_joint,
                polycoef=polycoef,
            )
        )
    return equalities


def parse_mjcf_joint_frictionloss(path: str | Path | None) -> dict[str, float]:
    """按关节名读取 MJCF 中的 ``frictionloss``。

    参数:
        path: MJCF 文件路径；为 ``None`` 或不存在时返回空字典。
    返回:
        ``关节名 -> frictionloss`` 映射，值必须非负。
    """

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
            raise ValueError(f"Invalid MJCF frictionloss for joint {joint_name!r}: {frictionloss_text!r}") from exc
        if frictionloss < 0:
            raise ValueError(f"MJCF frictionloss for joint {joint_name!r} cannot be negative: {frictionloss:g}")
        friction_by_name[joint_name] = frictionloss
    return friction_by_name


def mjcf_equality_follower_joint_names(path: str | Path | None) -> set[str]:
    """返回所有 mimic 从动关节名。

    参数:
        path: MJCF 文件路径。
    返回:
        从 ``equality/joint`` 的 ``joint1`` 收集到的关节名集合。
    """

    return {equality.dependent_joint for equality in parse_mjcf_joint_equalities(path)}


def expand_targets_with_mjcf_equalities(targets: dict[str, float], path: str | Path | None) -> dict[str, float]:
    """把稀疏主动关节目标扩展为包含从动关节目标的映射。

    参数:
        targets: ``关节名 -> 位置(rad)`` 的稀疏目标。
        path: MJCF 文件路径。
    返回:
        新字典；保留原目标，并为已知 master 生成 follower 目标。
    """

    expanded = dict(targets)
    for equality in parse_mjcf_joint_equalities(path):
        if equality.master_joint not in expanded:
            continue
        expanded[equality.dependent_joint] = equality.evaluate_position(float(expanded[equality.master_joint]))
    return expanded


def resolve_mimic_follower_controls(
    dof_names: list[str],
    mjcf_path: str | Path | None,
) -> list[MimicFollowerControl]:
    """生成基于完整 DOF 实际状态的 follower 映射。

    参数:
        dof_names: 完整 DOF 名称列表。
        mjcf_path: MJCF 文件路径。
    返回:
        ``MimicFollowerControl`` 列表；只包含 master/follower 都存在的关系。
    """

    dof_index_by_name = {name: index for index, name in enumerate(dof_names)}
    controls: list[MimicFollowerControl] = []
    for equality in parse_mjcf_joint_equalities(mjcf_path):
        dependent_index = dof_index_by_name.get(equality.dependent_joint)
        master_index = dof_index_by_name.get(equality.master_joint)
        if dependent_index is None or master_index is None:
            continue
        controls.append(
            MimicFollowerControl(
                dependent_joint=equality.dependent_joint,
                master_joint=equality.master_joint,
                dependent_index=int(dependent_index),
                master_index=int(master_index),
                polycoef=equality.polycoef,
            )
        )
    return controls


class MimicFollowerTargetMapper:
    """根据主动关节实际状态原地更新从动关节目标数组。

    输入:
        dof_names: 完整 DOF 名称列表。
        mjcf_path: MJCF 文件路径。
    输出:
        ``apply_from_actual`` 会直接修改传入的完整 target 数组。
    """

    def __init__(self, dof_names: list[str], mjcf_path: str | Path | None) -> None:
        """创建实际状态 follower 映射器。

        参数:
            dof_names: 完整 DOF 名称列表。
            mjcf_path: MJCF 文件路径。
        返回:
            无返回值；解析结果保存在 ``self.controls``。
        """

        self.controls = resolve_mimic_follower_controls(dof_names, mjcf_path)

    @property
    def relations(self) -> list[dict[str, float | int | str | tuple[float, ...]]]:
        """返回便于打印或写日志的关系元数据。

        返回:
            字典列表，每项包含 dependent/master 名称、索引和多项式系数。
        """

        return [
            {
                "dependent": control.dependent_joint,
                "master": control.master_joint,
                "dependent_index": control.dependent_index,
                "master_index": control.master_index,
                "polycoef": control.polycoef,
            }
            for control in self.controls
        ]

    def apply_from_actual(
        self,
        target_positions: np.ndarray,
        target_velocities: np.ndarray,
        actual_positions: np.ndarray,
        actual_velocities: np.ndarray,
    ) -> None:
        """用主动关节实际位置/速度原地覆盖 follower 目标。

        参数:
            target_positions: 完整 DOF 位置目标数组，会被原地修改。
            target_velocities: 完整 DOF 速度目标数组，会被原地修改。
            actual_positions: 完整 DOF 实际位置数组，单位 rad。
            actual_velocities: 完整 DOF 实际速度数组，单位 rad/s。
        返回:
            无返回值。
        """

        for control in self.controls:
            target_positions[control.dependent_index] = control.evaluate_position(actual_positions[control.master_index])
            target_velocities[control.dependent_index] = control.evaluate_velocity(
                actual_positions[control.master_index],
                actual_velocities[control.master_index],
            )
