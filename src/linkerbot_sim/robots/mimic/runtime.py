"""把 MJCF mimic 关系绑定到 articulation DOF 并更新 follower targets。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linkerbot_sim.robots.mimic.assets import parse_asset_mimic_relations
from linkerbot_sim.robots.mimic.mjcf import (
    evaluate_polycoef,
    evaluate_polycoef_derivative,
)


@dataclass(frozen=True)
class MimicFollowerControl:
    """一条已解析为完整 DOF indices 的 follower 控制关系。"""

    dependent_joint: str
    master_joint: str
    dependent_index: int
    master_index: int
    polycoef: tuple[float, ...]

    def evaluate_position(self, master_position: float) -> float:
        """按运行时 mimic polynomial 计算 follower position。"""

        return evaluate_polycoef(self.polycoef, master_position)

    def evaluate_velocity(
        self,
        master_position: float,
        master_velocity: float,
    ) -> float:
        """按 position relation 导数计算 follower velocity。"""

        return float(
            evaluate_polycoef_derivative(self.polycoef, master_position)
            * float(master_velocity)
        )


def resolve_mimic_follower_controls(
    dof_names: list[str],
    asset_path: str | Path | None,
) -> list[MimicFollowerControl]:
    """只绑定资产中 master/follower 都存在于当前 articulation 的关系。"""

    dof_index_by_name = {name: index for index, name in enumerate(dof_names)}
    controls: list[MimicFollowerControl] = []
    for equality in parse_asset_mimic_relations(asset_path):
        dependent_index = dof_index_by_name.get(equality.dependent_joint)
        master_index = dof_index_by_name.get(equality.master_joint)
        if dependent_index is None and master_index is None:
            continue
        if dependent_index is None or master_index is None:
            missing = (
                equality.dependent_joint
                if dependent_index is None
                else equality.master_joint
            )
            present = (
                equality.master_joint
                if dependent_index is None
                else equality.dependent_joint
            )
            raise ValueError(
                f"Mimic relation has joint {present!r} in the articulation but "
                f"is missing {missing!r}"
            )
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
    """用 master 实际状态原地更新 follower position/velocity targets。"""

    def __init__(
        self,
        dof_names: list[str],
        asset_path: str | Path | None,
    ) -> None:
        self.controls = resolve_mimic_follower_controls(dof_names, asset_path)

    @property
    def relations(self) -> list[dict[str, float | int | str | tuple[float, ...]]]:
        """返回日志和状态输出使用的稳定关系元数据。"""

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
        """按链式法则用 master 实际位置/速度覆盖 follower 目标。"""

        # 实际状态而非目标状态可让 follower 在 master 尚未跟踪完成时贴近物理姿态。
        for control in self.controls:
            target_positions[control.dependent_index] = control.evaluate_position(
                actual_positions[control.master_index]
            )
            target_velocities[control.dependent_index] = control.evaluate_velocity(
                actual_positions[control.master_index],
                actual_velocities[control.master_index],
            )


__all__ = [
    "MimicFollowerControl",
    "MimicFollowerTargetMapper",
    "resolve_mimic_follower_controls",
]
