"""cuRobo C-space 与项目 command-space 的关节名映射。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CuroboJointMapping:
    """按名称在 cuRobo C-space 与 controller command-space 间抽取/回填列。

    cuRobo 和 Isaac articulation 的 DOF 顺序不能假设相同；双臂、手部 overlay、mimic 关节
    更会让 command-space 成为 C-space 的 superset。这个 mapping 是后端替换时最重要的安全阀。
    """

    cspace_joint_names: tuple[str, ...]
    command_joint_names: tuple[str, ...]
    command_indices_for_cspace: tuple[int, ...]

    @classmethod
    def from_joint_names(
        cls,
        *,
        cspace_joint_names: Sequence[str],
        command_joint_names: Sequence[str],
    ) -> "CuroboJointMapping":
        """按关节名创建列映射。"""

        cspace_names = tuple(str(name) for name in cspace_joint_names)
        command_names = tuple(str(name) for name in command_joint_names)
        index_by_name = {name: index for index, name in enumerate(command_names)}
        missing = [name for name in cspace_names if name not in index_by_name]
        if missing:
            raise ValueError(
                "command_joint_names missing cuRobo C-space joints: "
                + ", ".join(missing)
            )
        return cls(
            cspace_joint_names=cspace_names,
            command_joint_names=command_names,
            command_indices_for_cspace=tuple(
                index_by_name[name] for name in cspace_names
            ),
        )

    @property
    def cspace_width(self) -> int:
        """返回 cuRobo C-space 维度。"""

        return len(self.cspace_joint_names)

    @property
    def command_width(self) -> int:
        """返回 command-space 维度。"""

        return len(self.command_joint_names)

    def command_to_cspace(self, command_positions: np.ndarray) -> np.ndarray:
        """从 command-space 数组抽取 C-space 列。"""

        command = _require_2d(command_positions, "command_positions")
        if command.shape[1] != self.command_width:
            raise ValueError("command_positions width must match command_joint_names")
        return np.ascontiguousarray(
            command[:, self.command_indices_for_cspace],
            dtype=float,
        )

    def cspace_to_command(
        self,
        cspace_positions: np.ndarray,
        *,
        base_command_positions: np.ndarray,
    ) -> np.ndarray:
        """把 C-space 解写回 command-space，非 C-space 列保留 base command。"""

        cspace = _require_2d(cspace_positions, "cspace_positions")
        base = _require_2d(base_command_positions, "base_command_positions")
        if cspace.shape[1] != self.cspace_width:
            raise ValueError("cspace_positions width must match cuRobo C-space")
        if base.shape[1] != self.command_width:
            raise ValueError("base_command_positions width must match command-space")
        if cspace.shape[0] != base.shape[0]:
            raise ValueError("cspace_positions and base_command_positions need same N")
        command = base.copy()
        command[:, self.command_indices_for_cspace] = cspace
        return command


def _require_2d(values: np.ndarray, label: str) -> np.ndarray:
    """把 1D/2D 输入规范化为二维 float 数组。"""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    return array
