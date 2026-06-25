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
from typing import Literal

import numpy as np

from manipulation_project.planning.collision_objects import CollisionObject


OrientationMode = Literal["current", "target", "none"]


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


@dataclass(frozen=True)
class TcpLineRequest:
    """TCP 直线 IK 请求。

    该请求只描述目标和约束，不绑定具体求解器或 FK/IK 对象。后端实现负责解释
    ``current_joint_positions`` 的关节顺序；例如 cuMotion 后端使用 C-space 关节顺序。

    字段含义:
        tcp_frame_name: 要沿直线移动的 TCP frame 名称，必须能被后端 FK/IK 识别。
        current_joint_positions: 当前关节位置，使用后端约定的关节顺序。
        start_position: 直线起点；为 ``None`` 时后端应使用当前 FK 位置。
        target_position: 直线终点绝对位置；和 ``target_offset`` 二选一。
        target_offset: 相对起点的位移；和 ``target_position`` 二选一。
        orientation_mode: ``current`` 保持当前 TCP 姿态，``target`` 从当前姿态插值到目标
            姿态，``none`` 表示 IK 只约束位置。
        target_orientation/target_rpy: ``orientation_mode='target'`` 时的目标姿态。
        duration_s/sample_hz: 用于生成 waypoint 的时间轴。
        position_tolerance/orientation_tolerance: 逐点 IK 的容差，通常来自后端 config。
    """

    tcp_frame_name: str
    current_joint_positions: np.ndarray
    start_position: tuple[float, float, float] | None = None
    target_position: tuple[float, float, float] | None = None
    target_offset: tuple[float, float, float] | None = None
    orientation_mode: OrientationMode = "current"
    target_orientation: tuple[float, float, float, float] | None = None
    target_rpy: tuple[float, float, float] | None = None
    duration_s: float = 2.0
    sample_hz: float = 100.0
    position_tolerance: float = 1.0e-3
    orientation_tolerance: float = 1.0e-2

    def validate(self) -> None:
        """检查请求是否足够生成 TCP 直线 IK 路径。

        这里只校验不需要机器人模型即可判断的结构性约束。frame 是否存在、目标是否可达、
        IK 是否成功等问题由具体后端在执行时报告。
        """

        if not self.tcp_frame_name:
            raise ValueError("tcp_frame_name is required")
        if self.orientation_mode not in {"current", "target", "none"}:
            raise ValueError("orientation_mode must be one of: current, target, none")
        if (self.target_position is None) == (self.target_offset is None):
            raise ValueError(
                "Exactly one of target_position or target_offset must be provided"
            )
        if (
            self.orientation_mode == "target"
            and self.target_orientation is None
            and self.target_rpy is None
        ):
            raise ValueError(
                "target orientation requires target_orientation or target_rpy"
            )
        if self.duration_s < 0:
            raise ValueError("duration cannot be negative")
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
