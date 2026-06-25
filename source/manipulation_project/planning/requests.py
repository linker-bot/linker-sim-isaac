"""运动解算请求数据结构。

请求对象是任务层与后端层之间的公共输入格式。它们不包含求解器实例，也不持有 Isaac
runtime 对象，只携带目标、初值、容差和碰撞对象。这样任务层可以用同一种数据结构调用
cuMotion、测试替身或未来其它规划后端。

单位/坐标约定:
    * 位置单位为 m；姿态使用 wxyz 四元数；角度容差使用 rad。
    * TCP 目标的坐标系由调用方和后端约定，通常是机器人 base/world 对齐后的坐标。
    * 关节向量顺序不在本文件固定，而由具体后端暴露的 ``joint_names`` 决定。

错误边界:
    数据类本身只表达请求，不主动求解，也不根据机器人模型检查目标是否可达。后端应把
    不可达、碰撞失败或数值异常归一化成 ``planning.results`` 中的结果对象。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manipulation_project.planning.collision_objects import CollisionObject


@dataclass(frozen=True)
class PoseTarget:
    """TCP 位姿目标。

    ``position`` 是目标点在机器人 base/world 约定坐标系下的位置；``orientation`` 为可选
    wxyz 四元数，留空表示只约束位置。
    """

    position: np.ndarray
    orientation: np.ndarray | None = None


@dataclass(frozen=True)
class IKRequest:
    """单个 TCP 逆运动学目标。

    ``warm_start`` 用于连续轨迹求解的上一帧关节解；``avoid_collisions`` 为真时后端应使用
    collision-aware 路径或 IK 模式。失败时由后端返回 ``IKResult``，不在数据类内抛错。
    """

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
    """路径级运动规划请求。

    ``current_q`` 和 ``goal_q`` 使用后端关节顺序；``goal_pose`` 用于任务空间目标。
    ``mode`` 是后端可解释的策略标签，默认表示优先考虑碰撞。
    """

    current_q: np.ndarray
    goal_q: np.ndarray | None = None
    goal_pose: PoseTarget | None = None
    tcp_frame_name: str | None = None
    collision_objects: tuple[CollisionObject, ...] = ()
    mode: str = "collision_aware"
