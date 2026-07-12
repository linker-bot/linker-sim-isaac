"""运动解算请求数据结构。

请求对象是动作脚本层与后端层之间的公共输入格式。它们不包含求解器实例，也不持有 Isaac
runtime 对象，只携带目标、初值、容差或路径几何。环境障碍由具体后端 context 和所选
planning pipeline 解释，这样动作脚本层可以用同一种数据结构调用 cuRobo 或测试替身。

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
from typing import Literal, cast

import numpy as np


OrientationMode = Literal["free", "current", "target"]


def _finite_vector(values: object, width: int | None, label: str) -> np.ndarray:
    """把规划输入规范化为有限的一维向量。"""

    vector = np.asarray(values, dtype=float).reshape(-1)
    if width is not None and vector.size != width:
        raise ValueError(f"{label} must have shape ({width},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain finite values")
    return vector


def resolve_orientation_mode(
    *,
    requested_mode: object | None,
    requested_mode_is_explicit: bool,
    default_mode: object,
    target_orientation_present: bool,
    label: str = "orientation_mode",
    target_label: str = "target_orientation_quat_wxyz",
) -> OrientationMode:
    """解析请求的姿态约束语义，绝不静默忽略已提供的四元数。

    未显式选择 mode 但提供目标四元数时自动采用 ``target``；显式 mode 与四元数冲突，或
    选择 ``target`` 却缺少四元数时均拒绝请求。
    """

    if requested_mode_is_explicit:
        mode = str(requested_mode)
    elif target_orientation_present:
        mode = "target"
    else:
        mode = str(default_mode)
    if mode not in {"free", "current", "target"}:
        raise ValueError(f"{label} must be one of: free, current, target")
    if target_orientation_present and mode != "target":
        raise ValueError(
            f"{target_label} cannot be combined with {label}={mode!r}; "
            "use orientation_mode='target' or omit orientation_mode"
        )
    if mode == "target" and not target_orientation_present:
        raise ValueError(f"{label}='target' requires {target_label}")
    return cast(OrientationMode, mode)


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
    tolerance 为 ``None`` 时使用后端 profile。后端若不支持 per-request override，必须明确拒绝
    非 ``None`` 值，不能静默忽略。
    ``avoid_collisions`` 为真时后端应使用 collision-aware 路径或 IK 模式。
    环境障碍不放在请求里，支持环境管理的后端应从自身 context 读取当前 world。
    失败时由后端返回 ``IKResult``，不在数据类内抛错。
    """

    target_position: np.ndarray
    target_orientation: np.ndarray | None = None
    tcp_frame_name: str | None = None
    warm_start_ik_cspace_seed: np.ndarray | None = None
    position_tolerance: float | None = None
    orientation_tolerance: float | None = None
    avoid_collisions: bool = False

    def validate_structure(self) -> None:
        """检查 IK 请求的结构性约束。

        这里不判断目标是否可达，也不检查 frame 是否存在；这些依赖具体机器人模型。
        warm-start IK C-space seed 只检查非空，长度是否匹配 C-space 由具体后端在加载模型后检查。
        """

        _finite_vector(self.target_position, 3, "target_position")
        if self.target_orientation is not None:
            _finite_vector(self.target_orientation, 4, "target_orientation")
        if self.warm_start_ik_cspace_seed is not None:
            warm_start_ik_cspace_seed = _finite_vector(
                self.warm_start_ik_cspace_seed,
                None,
                "warm_start_ik_cspace_seed",
            )
            if warm_start_ik_cspace_seed.size == 0:
                raise ValueError("warm_start_ik_cspace_seed cannot be empty")
        for name, tolerance in (
            ("position_tolerance", self.position_tolerance),
            ("orientation_tolerance", self.orientation_tolerance),
        ):
            if tolerance is not None and (
                not np.isfinite(float(tolerance)) or float(tolerance) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.tcp_frame_name is not None and not str(self.tcp_frame_name):
            raise ValueError("tcp_frame_name cannot be empty")


@dataclass(frozen=True)
class MotionRequest:
    """路径级运动规划请求。

    ``current_q`` 和 ``goal_q`` 使用后端关节顺序；``goal_pose`` 用于任务空间目标。
    ``duration_s`` 描述期望阶段时长；``sample_dt_s`` 描述调用方希望得到的轨迹采样周期，
    通常应等于 physics dt。支持时间参数的后端应返回已经按该时间网格重采样的轨迹。
    ``avoid_collisions`` 表示调用方明确需要 collision-aware planner。后端如果没有完整
    robot collision model 或场景碰撞查询能力，应返回清晰失败，而不是退化成普通无碰撞规划。
    """

    current_q: np.ndarray
    goal_q: np.ndarray | None = None
    goal_pose: PoseTarget | None = None
    tcp_frame_name: str | None = None
    duration_s: float | None = None
    sample_dt_s: float | None = None
    avoid_collisions: bool = False

    def validate_structure(self) -> None:
        """检查路径级请求是否描述了唯一目标。

        结构校验只保证数组形状、目标互斥关系和非负时长；frame 存在性、关节维度与
        pipeline 支持能力仍由后端 context/planner 判断。
        """

        # MotionRequest 不知道具体机器人模型，因此这里只做结构性检查：当前构型必须是非空
        # 1D 向量，目标必须且只能指定一种。具体关节名、frame 是否存在、碰撞世界是否可用，
        # 交给后端 context/planner 在执行时检查。
        current = _finite_vector(self.current_q, None, "current_q")
        if current.size == 0:
            raise ValueError("current_q cannot be empty")
        if (self.goal_q is None) == (self.goal_pose is None):
            raise ValueError("Exactly one of goal_q or goal_pose must be provided")
        if self.goal_q is not None:
            # goal_q 和 current_q 必须使用同一后端关节顺序。这里能检查长度一致；顺序是否正确
            # 由调用方通过后端 ``joint_names()`` 做名称映射来保证。
            goal = _finite_vector(self.goal_q, None, "goal_q")
            if goal.size != current.size:
                raise ValueError(
                    f"goal_q expected {current.size} values, got {goal.size}"
                )
        if self.duration_s is not None and (
            not np.isfinite(float(self.duration_s)) or float(self.duration_s) < 0.0
        ):
            raise ValueError("duration_s must be finite and non-negative")
        if self.sample_dt_s is not None and (
            not np.isfinite(float(self.sample_dt_s)) or float(self.sample_dt_s) <= 0.0
        ):
            raise ValueError("sample_dt_s must be finite and positive")
        if self.goal_pose is not None:
            # PoseTarget 使用项目统一边界：position 为 3D 米制坐标，orientation 若提供则为
            # wxyz 四元数。这里只 reshape 触发清晰错误，不做归一化或可达性判断。
            _finite_vector(self.goal_pose.position, 3, "goal_pose.position")
            if self.goal_pose.orientation is not None:
                _finite_vector(
                    self.goal_pose.orientation,
                    4,
                    "goal_pose.orientation",
                )


@dataclass(frozen=True)
class TcpLineSegment:
    """线性 TCP 路径中的直线段描述。

    ``target_position`` 表示 base/world 约定坐标系下的绝对终点，``target_offset`` 表示相对
    起点的位移；二者由使用该 segment 的 pipeline 解释为互斥目标。``orientation_mode`` 控制
    线性采样时的姿态约束：``free`` 只约束 TCP 位置、不约束姿态，``current`` 保持当前
    tracked TCP 姿态，``target`` 使用 ``target_orientation`` 作为终点姿态。
    """

    start_position: np.ndarray | None = None
    target_position: np.ndarray | None = None
    target_offset: np.ndarray | None = None
    orientation_mode: OrientationMode = "free"
    target_orientation: np.ndarray | None = None


@dataclass(frozen=True)
class TcpPoseSequenceSegment:
    """线性路径中的一串完整 TCP pose 直线段描述。

    每个 pose 都必须带 orientation，因为该 segment 表示多段完整 pose 直线，而不是
    position-only 平移序列。需要只约束位置时应使用 ``TcpLineSegment(orientation_mode='free')``。
    """

    poses: tuple[PoseTarget, ...]
    blend_radius: float = 0.0


TaskSpaceSegment = TcpLineSegment | TcpPoseSequenceSegment


@dataclass(frozen=True)
class TaskSpacePath:
    """调用方显式指定的一组 task-space path segments。

    这些 segment 会在 cuRobo 后端离散成 TCP pose，再通过顺序 IK 转成 C-space path。
    请求层不检查 frame 是否存在，也不判断 IK 可达性。
    """

    segments: tuple[TaskSpaceSegment, ...]


@dataclass(frozen=True)
class LinearPosePathRequest:
    """线性 TCP 位姿路径请求。

    调用方只表达 TCP 的线性位姿运动，后端通过顺序 IK 把离散 TCP pose 转成 C-space
    waypoint。该请求不支持圆弧、composite transition 或 C-space waypoint 指定路径。
    """

    current_q: np.ndarray
    path: TaskSpacePath
    tcp_frame_name: str | None = None
    duration_s: float | None = None
    sample_dt_s: float | None = None
    avoid_collisions: bool = False

    def validate_structure(self) -> None:
        """检查线性位姿路径请求的基础结构。"""

        current = _finite_vector(self.current_q, None, "current_q")
        if current.size == 0:
            raise ValueError("current_q cannot be empty")
        if self.duration_s is not None and (
            not np.isfinite(float(self.duration_s)) or float(self.duration_s) < 0.0
        ):
            raise ValueError("duration_s must be finite and non-negative")
        if self.sample_dt_s is not None and (
            not np.isfinite(float(self.sample_dt_s)) or float(self.sample_dt_s) <= 0.0
        ):
            raise ValueError("sample_dt_s must be finite and positive")
        if self.tcp_frame_name is not None and not str(self.tcp_frame_name):
            raise ValueError("tcp_frame_name cannot be empty")
        if not isinstance(self.path, TaskSpacePath):
            raise ValueError("LinearPosePathRequest.path must be a TaskSpacePath")
        if not self.path.segments:
            raise ValueError("LinearPosePathRequest.path requires at least one segment")
        for index, segment in enumerate(self.path.segments):
            _validate_task_space_segment(segment, f"path.segments[{index}]")


def _validate_task_space_segment(segment: TaskSpaceSegment, label: str) -> None:
    """检查 task-space segment 的后端无关结构。

    这里保持“纯数据边界”：确保 numpy reshape 能得到预期维度、枚举值有效、必填字段存在。
    不创建任何 cuRobo 对象，也不根据机器人模型判断路径是否可执行。
    """

    if isinstance(segment, TcpLineSegment):
        _validate_tcp_line_segment(segment, label)
        return
    if isinstance(segment, TcpPoseSequenceSegment):
        if not segment.poses:
            raise ValueError(f"{label}.poses requires at least one pose")
        blend_radius = float(segment.blend_radius)
        if not np.isfinite(blend_radius):
            raise ValueError(f"{label}.blend_radius must be finite")
        if blend_radius < 0:
            raise ValueError(f"{label}.blend_radius cannot be negative")
        if blend_radius > 0:
            raise ValueError(
                f"{label}.blend_radius is not supported by linear pose paths"
            )
        for pose_index, pose in enumerate(segment.poses):
            _finite_vector(
                pose.position,
                3,
                f"{label}.poses[{pose_index}].position",
            )
            if pose.orientation is None:
                raise ValueError(f"{label}.poses[{pose_index}].orientation is required")
            _finite_vector(
                pose.orientation,
                4,
                f"{label}.poses[{pose_index}].orientation",
            )
        return
    raise ValueError(f"Unsupported task-space segment type: {type(segment).__name__}")


def _validate_tcp_line_segment(segment: TcpLineSegment, label: str) -> None:
    """检查 ``TcpLineSegment`` 的后端无关结构。

    绝对终点和相对 offset 必须二选一；target orientation 只在 ``orientation_mode='target'`` 时
    强制要求，但若调用方额外提供了 orientation，也会检查它是 wxyz 四元数形状。
    """

    if segment.orientation_mode not in {"free", "current", "target"}:
        raise ValueError(
            f"{label}.orientation_mode must be one of: free, current, target"
        )
    if (segment.target_position is None) == (segment.target_offset is None):
        raise ValueError(
            f"{label} exactly one of target_position or target_offset must be provided"
        )
    if segment.start_position is not None:
        _finite_vector(segment.start_position, 3, f"{label}.start_position")
    if segment.target_position is not None:
        _finite_vector(segment.target_position, 3, f"{label}.target_position")
    if segment.target_offset is not None:
        _finite_vector(segment.target_offset, 3, f"{label}.target_offset")
    if segment.orientation_mode == "target" and segment.target_orientation is None:
        raise ValueError(f"{label}.target_orientation is required")
    if segment.orientation_mode != "target" and segment.target_orientation is not None:
        raise ValueError(
            f"{label}.target_orientation cannot be combined with "
            f"orientation_mode={segment.orientation_mode!r}"
        )
    if segment.target_orientation is not None:
        _finite_vector(
            segment.target_orientation,
            4,
            f"{label}.target_orientation",
        )
