"""灵巧手目标辅助函数。"""

from __future__ import annotations

from manipulation_project.robots.mimic import expand_targets_with_mjcf_equalities


def expanded_hand_pose(master_targets: dict[str, float], mjcf_path) -> dict[str, float]:
    """根据 MJCF equality 把 L6 手 master 关节目标扩展到 follower 关节。

    参数:
        master_targets: ``主动关节名 -> 目标位置(rad)`` 的稀疏映射。
        mjcf_path: L6 或 AR5+L6 MJCF 文件路径。
    返回:
        新字典，包含原 master 目标和可由 equality 推导出的 follower 目标。
    """

    return expand_targets_with_mjcf_equalities(master_targets, mjcf_path)
