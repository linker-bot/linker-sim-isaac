"""运动解算请求数据结构。

请求对象是动作脚本层与后端层之间的公共输入格式。它们不包含求解器实例，也不持有 Isaac
runtime 对象，只携带目标、初值、容差或路径几何。环境障碍由具体后端 context 和所选
planning pipeline 解释，这样动作脚本层可以用同一种数据结构调用 cuMotion 或测试替身。

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


OrientationMode = Literal["current", "target", "none"]
# Task-space specified path 只在请求层表达几何意图；具体如何调用 cuMotion
# TaskSpacePathSpec 由后端 adapter 决定。
TaskSpaceArcMode = Literal["tangent", "three_point"]
# Composite transition mode 暴露稳定字符串，避免把 cuMotion pybind enum 泄漏到动作脚本层。
CompositeTransitionMode = Literal["skip", "free", "linear_task_space"]


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

    ``warm_start_ik_cspace_seed`` 用于连续轨迹求解的上一帧 C-space 关节解；
    ``avoid_collisions`` 为真时后端应使用 collision-aware 路径或 IK 模式。
    环境障碍不放在请求里，支持环境管理的后端应从自身 context 读取当前 world。
    失败时由后端返回 ``IKResult``，不在数据类内抛错。
    """

    target_position: np.ndarray
    target_orientation: np.ndarray | None = None
    tcp_frame_name: str | None = None
    warm_start_ik_cspace_seed: np.ndarray | None = None
    position_tolerance: float = 1.0e-4
    orientation_tolerance: float = 1.0e-3
    avoid_collisions: bool = False

    def validate_structure(self) -> None:
        """检查 IK 请求的结构性约束。

        这里不判断目标是否可达，也不检查 frame 是否存在；这些依赖具体机器人模型。
        warm-start IK C-space seed 只检查非空，长度是否匹配 C-space 由具体后端在加载模型后检查。
        """

        np.asarray(self.target_position, dtype=float).reshape(3)
        if self.target_orientation is not None:
            np.asarray(self.target_orientation, dtype=float).reshape(4)
        if self.warm_start_ik_cspace_seed is not None:
            warm_start_ik_cspace_seed = np.asarray(
                self.warm_start_ik_cspace_seed, dtype=float
            ).reshape(-1)
            if warm_start_ik_cspace_seed.size == 0:
                raise ValueError("warm_start_ik_cspace_seed cannot be empty")
        if self.position_tolerance < 0 or self.orientation_tolerance < 0:
            raise ValueError("IK tolerances cannot be negative")
        if self.tcp_frame_name is not None and not str(self.tcp_frame_name):
            raise ValueError("tcp_frame_name cannot be empty")


@dataclass(frozen=True)
class MotionRequest:
    """路径级运动规划请求。

    ``current_q`` 和 ``goal_q`` 使用后端关节顺序；``goal_pose`` 用于任务空间目标。
    ``duration_s`` 只描述期望阶段时长，主要供 time-stamped trajectory generator 使用；
    是否使用环境障碍由 motion-planner pipeline 配置决定，而不是由请求对象携带。
    """

    current_q: np.ndarray
    goal_q: np.ndarray | None = None
    goal_pose: PoseTarget | None = None
    tcp_frame_name: str | None = None
    duration_s: float | None = None

    def validate_structure(self) -> None:
        """检查路径级请求是否描述了唯一目标。

        结构校验只保证数组形状、目标互斥关系和非负时长；frame 存在性、关节维度与
        pipeline 支持能力仍由后端 context/planner 判断。
        """

        # MotionRequest 不知道具体机器人模型，因此这里只做结构性检查：当前构型必须是非空
        # 1D 向量，目标必须且只能指定一种。具体关节名、frame 是否存在、碰撞世界是否可用，
        # 交给后端 context/planner 在执行时检查。
        current = np.asarray(self.current_q, dtype=float).reshape(-1)
        if current.size == 0:
            raise ValueError("current_q cannot be empty")
        if (self.goal_q is None) == (self.goal_pose is None):
            raise ValueError("Exactly one of goal_q or goal_pose must be provided")
        if self.goal_q is not None:
            # goal_q 和 current_q 必须使用同一后端关节顺序。这里能检查长度一致；顺序是否正确
            # 由调用方通过后端 ``joint_names()`` 做名称映射来保证。
            goal = np.asarray(self.goal_q, dtype=float).reshape(-1)
            if goal.size != current.size:
                raise ValueError(
                    f"goal_q expected {current.size} values, got {goal.size}"
                )
        if self.duration_s is not None and self.duration_s < 0:
            raise ValueError("duration_s cannot be negative")
        if self.goal_pose is not None:
            # PoseTarget 使用项目统一边界：position 为 3D 米制坐标，orientation 若提供则为
            # wxyz 四元数。这里只 reshape 触发清晰错误，不做归一化或可达性判断。
            np.asarray(self.goal_pose.position, dtype=float).reshape(3)
            if self.goal_pose.orientation is not None:
                np.asarray(self.goal_pose.orientation, dtype=float).reshape(4)


@dataclass(frozen=True)
class CSpaceWaypointPath:
    """调用方显式指定的一组 C-space waypoint。

    waypoint 的关节顺序必须和后端 ``joint_names()`` 一致。请求层会检查每个 waypoint 与
    ``current_q`` 同宽；真实 robot C-space 宽度、首点是否必须匹配 ``current_q`` 等后端策略在
    cuMotion adapter 中校验。
    """

    waypoints: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class TcpLineSegment:
    """specified-path 中的 TCP 直线段描述。

    ``target_position`` 表示 base/world 约定坐标系下的绝对终点，``target_offset`` 表示相对
    起点的位移；二者由使用该 segment 的 pipeline 解释为互斥目标。``orientation_mode`` 控制
    后端映射到 cuMotion PathSpec 的姿态约束：``none`` 只约束位置，``current`` 保持当前 tracked
    TCP 姿态，``target`` 使用 ``target_orientation`` 作为终点姿态。
    """

    start_position: np.ndarray | None = None
    target_position: np.ndarray | None = None
    target_offset: np.ndarray | None = None
    orientation_mode: OrientationMode = "current"
    target_orientation: np.ndarray | None = None


@dataclass(frozen=True)
class TcpRotationSegment:
    """specified-path 中的 TCP 原地旋转段描述。

    ``target_orientation`` 是 wxyz 四元数；后端会把它转换成 cuMotion ``Rotation3`` 并追加
    ``TaskSpacePathSpec.add_rotation(...)``。
    """

    target_orientation: np.ndarray


@dataclass(frozen=True)
class TcpArcSegment:
    """specified-path 中的 TCP 圆弧段描述。

    ``target_position`` 表示 base/world 约定坐标系下的绝对终点，``target_offset`` 表示相对
    当前 tracked TCP 位置的位移。``arc_mode='tangent'`` 使用当前路径切线和目标点定义圆弧；
    ``arc_mode='three_point'`` 还需要绝对 ``intermediate_position`` 或相对
    ``intermediate_offset``。若提供 ``target_orientation``，后端使用 cuMotion 的
    ``*_with_orientation_target`` 圆弧 API。
    """

    target_position: np.ndarray | None = None
    target_offset: np.ndarray | None = None
    intermediate_position: np.ndarray | None = None
    intermediate_offset: np.ndarray | None = None
    target_orientation: np.ndarray | None = None
    arc_mode: TaskSpaceArcMode = "tangent"
    constant_orientation: bool = True


@dataclass(frozen=True)
class TcpPoseSequenceSegment:
    """specified-path 中的一串完整 TCP pose 直线段描述。

    每个 pose 都必须带 orientation，因为该 segment 表示多段完整 pose 直线，而不是
    position-only 平移序列。需要只约束位置时应使用 ``TcpLineSegment(orientation_mode='none')``。
    """

    poses: tuple[PoseTarget, ...]
    blend_radius: float = 0.0


TaskSpaceSegment = (
    TcpLineSegment | TcpRotationSegment | TcpArcSegment | TcpPoseSequenceSegment
)


@dataclass(frozen=True)
class TaskSpacePath:
    """调用方显式指定的一组 task-space path segments。

    这些 segment 会在 cuMotion 后端映射到 ``TaskSpacePathSpec``，再通过官方 path conversion
    转成 C-space path。请求层不检查 frame 是否存在，也不判断 IK 可达性。
    """

    segments: tuple[TaskSpaceSegment, ...]


@dataclass(frozen=True)
class CompositePathPart:
    """Composite path 子段及其过渡方式。

    ``transition_mode`` 为 ``None`` 时使用后端配置中的
    ``specified_path.composite.default_transition_mode``。这里使用 wrapper 的原因是直接放
    ``CSpaceWaypointPath`` / ``TaskSpacePath`` 时仍能保持简洁。
    """

    path: CSpaceWaypointPath | TaskSpacePath
    transition_mode: CompositeTransitionMode | None = None


@dataclass(frozen=True)
class CompositePath:
    """混合 C-space 和 task-space 子路径的指定路径。

    每个 part 可以是裸子路径，也可以是 ``CompositePathPart`` 以指定段间 transition。后端会
    把它转换成 cuMotion ``CompositePathSpec``。
    """

    parts: tuple[CSpaceWaypointPath | TaskSpacePath | CompositePathPart, ...]


@dataclass(frozen=True)
class SpecifiedPathRequest:
    """指定路径规划请求。

    ``current_q`` 使用后端 C-space 关节顺序；``path`` 描述调用方指定的路径几何。
    cuMotion facade 支持 ``CSpaceWaypointPath``、``TaskSpacePath`` 和 ``CompositePath``，并将
    它们统一转换成 C-space joint path 后再做可选时间参数化。
    """

    current_q: np.ndarray
    path: CSpaceWaypointPath | TaskSpacePath | CompositePath
    tcp_frame_name: str | None = None
    duration_s: float | None = None

    def validate_structure(self) -> None:
        """检查指定路径请求的基础结构。

        该方法只做后端无关的 shape/枚举/互斥关系校验。C-space waypoint 会被检查为与
        ``current_q`` 同宽；真实 robot C-space 宽度、frame 存在性、首点容差和 task-space
        可达性都依赖具体 cuMotion context，留给后端 adapter 处理。
        """

        current = np.asarray(self.current_q, dtype=float).reshape(-1)
        if current.size == 0:
            raise ValueError("current_q cannot be empty")
        if self.duration_s is not None and self.duration_s < 0:
            raise ValueError("duration_s cannot be negative")
        if self.tcp_frame_name is not None and not str(self.tcp_frame_name):
            raise ValueError("tcp_frame_name cannot be empty")
        if isinstance(self.path, CSpaceWaypointPath):
            if len(self.path.waypoints) < 2:
                raise ValueError("CSpaceWaypointPath requires at least 2 waypoints")
            for index, waypoint in enumerate(self.path.waypoints):
                waypoint_array = np.asarray(waypoint, dtype=float).reshape(-1)
                if waypoint_array.size != current.size:
                    raise ValueError(
                        f"CSpaceWaypointPath waypoint {index} expected {current.size} values, "
                        f"got {waypoint_array.size}"
                    )
            return
        if isinstance(self.path, TaskSpacePath):
            if not self.path.segments:
                raise ValueError("TaskSpacePath requires at least one segment")
            for index, segment in enumerate(self.path.segments):
                _validate_task_space_segment(segment, f"path.segments[{index}]")
            return
        if isinstance(self.path, CompositePath):
            if not self.path.parts:
                raise ValueError("CompositePath requires at least one part")
            for index, part in enumerate(self.path.parts):
                _validate_composite_path_part(part, f"path.parts[{index}]")
            return
        raise ValueError(f"Unsupported specified path type: {type(self.path).__name__}")


def _validate_task_space_segment(segment: TaskSpaceSegment, label: str) -> None:
    """检查 task-space segment 的后端无关结构。

    这里保持“纯数据边界”：确保 numpy reshape 能得到预期维度、枚举值有效、必填字段存在。
    不创建任何 cuMotion 对象，也不根据机器人模型判断路径是否可执行。
    """

    if isinstance(segment, TcpLineSegment):
        _validate_tcp_line_segment(segment, label)
        return
    if isinstance(segment, TcpRotationSegment):
        np.asarray(segment.target_orientation, dtype=float).reshape(4)
        return
    if isinstance(segment, TcpArcSegment):
        if (segment.target_position is None) == (segment.target_offset is None):
            raise ValueError(
                f"{label} exactly one of target_position or target_offset must be provided"
            )
        if segment.target_position is not None:
            np.asarray(segment.target_position, dtype=float).reshape(3)
        if segment.target_offset is not None:
            np.asarray(segment.target_offset, dtype=float).reshape(3)
        if segment.arc_mode not in {"tangent", "three_point"}:
            raise ValueError(f"{label}.arc_mode must be one of: tangent, three_point")
        if segment.arc_mode == "three_point":
            if (segment.intermediate_position is None) == (
                segment.intermediate_offset is None
            ):
                raise ValueError(
                    f"{label} exactly one of intermediate_position or "
                    "intermediate_offset is required for three_point arc"
                )
        if segment.intermediate_position is not None:
            np.asarray(segment.intermediate_position, dtype=float).reshape(3)
        if segment.intermediate_offset is not None:
            np.asarray(segment.intermediate_offset, dtype=float).reshape(3)
        if segment.target_orientation is not None:
            np.asarray(segment.target_orientation, dtype=float).reshape(4)
        return
    if isinstance(segment, TcpPoseSequenceSegment):
        if not segment.poses:
            raise ValueError(f"{label}.poses requires at least one pose")
        if segment.blend_radius < 0:
            raise ValueError(f"{label}.blend_radius cannot be negative")
        for pose_index, pose in enumerate(segment.poses):
            np.asarray(pose.position, dtype=float).reshape(3)
            if pose.orientation is None:
                raise ValueError(
                    f"{label}.poses[{pose_index}].orientation is required"
                )
            np.asarray(pose.orientation, dtype=float).reshape(4)
        return
    raise ValueError(f"Unsupported task-space segment type: {type(segment).__name__}")


def _validate_tcp_line_segment(segment: TcpLineSegment, label: str) -> None:
    """检查 ``TcpLineSegment`` 的后端无关结构。

    绝对终点和相对 offset 必须二选一；target orientation 只在 ``orientation_mode='target'`` 时
    强制要求，但若调用方额外提供了 orientation，也会检查它是 wxyz 四元数形状。
    """

    if segment.orientation_mode not in {"current", "target", "none"}:
        raise ValueError(
            f"{label}.orientation_mode must be one of: current, target, none"
        )
    if (segment.target_position is None) == (segment.target_offset is None):
        raise ValueError(
            f"{label} exactly one of target_position or target_offset must be provided"
        )
    if segment.start_position is not None:
        np.asarray(segment.start_position, dtype=float).reshape(3)
    if segment.target_position is not None:
        np.asarray(segment.target_position, dtype=float).reshape(3)
    if segment.target_offset is not None:
        np.asarray(segment.target_offset, dtype=float).reshape(3)
    if segment.orientation_mode == "target" and segment.target_orientation is None:
        raise ValueError(f"{label}.target_orientation is required")
    if segment.target_orientation is not None:
        np.asarray(segment.target_orientation, dtype=float).reshape(4)


def _validate_composite_path_part(
    part: CSpaceWaypointPath | TaskSpacePath | CompositePathPart,
    label: str,
) -> None:
    """检查 composite path 子段结构。

    Composite 子段的 C-space waypoint 这里只检查非空，不检查宽度或首点连续性。宽度取决于
    robot C-space；首点连续性取决于配置和前序子段类型，均由 cuMotion adapter 处理。
    """

    if isinstance(part, CompositePathPart):
        if part.transition_mode is not None and part.transition_mode not in {
            "skip",
            "free",
            "linear_task_space",
        }:
            raise ValueError(
                f"{label}.transition_mode must be one of: skip, free, linear_task_space"
            )
        nested = part.path
    else:
        nested = part
    if isinstance(nested, CSpaceWaypointPath):
        if len(nested.waypoints) < 2:
            raise ValueError(
                f"{label}.path CSpaceWaypointPath requires at least 2 waypoints"
            )
        for waypoint_index, waypoint in enumerate(nested.waypoints):
            waypoint_array = np.asarray(waypoint, dtype=float).reshape(-1)
            if waypoint_array.size == 0:
                raise ValueError(
                    f"{label}.path.waypoints[{waypoint_index}] cannot be empty"
                )
        return
    if isinstance(nested, TaskSpacePath):
        if not nested.segments:
            raise ValueError(f"{label}.path TaskSpacePath requires at least one segment")
        for segment_index, segment in enumerate(nested.segments):
            _validate_task_space_segment(
                segment, f"{label}.path.segments[{segment_index}]"
            )
        return
    raise ValueError(f"Unsupported composite path part type: {type(nested).__name__}")
