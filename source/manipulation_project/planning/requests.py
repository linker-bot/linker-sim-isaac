"""运动解算请求数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manipulation_project.planning.collision_objects import CollisionObject


@dataclass(frozen=True)
class PoseTarget:
    """TCP 位姿目标。"""

    position: np.ndarray
    orientation: np.ndarray | None = None


@dataclass(frozen=True)
class IKRequest:
    """单个 TCP 逆运动学目标。"""

    target_position: np.ndarray
    target_orientation: np.ndarray | None = None
    tcp_frame_name: str | None = None
    tcp_type: str = "flange"
    warm_start: np.ndarray | None = None
    position_tolerance: float = 1.0e-3
    orientation_tolerance: float = 1.0e-2
    avoid_collisions: bool = False
    collision_objects: tuple[CollisionObject, ...] = ()


@dataclass(frozen=True)
class MotionRequest:
    """路径级运动规划请求。"""

    current_q: np.ndarray
    goal_q: np.ndarray | None = None
    goal_pose: PoseTarget | None = None
    tcp_frame_name: str | None = None
    collision_objects: tuple[CollisionObject, ...] = ()
    mode: str = "collision_aware"
