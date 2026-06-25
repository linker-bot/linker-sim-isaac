"""TCP 笛卡尔直线运动任务辅助函数。

该模块把“TCP 在 base 坐标下沿直线移动”的配置转换成控制器可执行的关节轨迹：先用 FK
读取当前 TCP 位姿，再在线段上逐点 IK 求解，并用上一点 IK 解热启动下一点。

职责边界:
    * 只负责生成关节空间命令轨迹，不直接推进 Isaac world。
    * 不创建机器人或控制器；调用方负责提供完整 DOF 名称、命令索引和当前关节状态。
    * 不做碰撞规划；若后端支持碰撞，可通过 ``IKRequest`` 扩展相关字段。

坐标/顺序约定:
    所有 position 都在 cuMotion robot base 坐标系下表达；当前示例资产固定在 world 原点，
    因此默认场景中它和 world 坐标一致。输出轨迹仍然是关节空间命令，列顺序按控制器
    ``command_indices`` 对应的 DOF 名称排列。如果 cuMotion 只覆盖机械臂 C-space，而组合
    articulation 还包含灵巧手 DOF，本模块会显式检查 IK 关节是否都属于命令空间，避免把
    未求解的 DOF 静默写错。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from manipulation_project.backends.cumotion.context import (
    CuMotionConfig,
    CuMotionContext,
)
from manipulation_project.planning.requests import IKRequest
from manipulation_project.trajectories.cartesian_waypoints import (
    sample_cartesian_pose_line,
)
from manipulation_project.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from manipulation_project.trajectories.types import JointTrajectory
from manipulation_project.utils.rotations import (
    normalize_quat_wxyz,
    rpy_xyz_deg_to_quat_wxyz,
)
from manipulation_project.utils.timing import sample_times


OrientationMode = Literal["current", "target", "none"]


@dataclass(frozen=True)
class MoveTcpLineConfig:
    """TCP 直线运动配置。

    坐标约定:
        所有 position 都在 cuMotion robot base 坐标系下表达；当前示例资产固定在
        world 原点，因此默认场景中它和 world 坐标一致。
    start_position: 起点位置；为 ``None`` 时使用当前 TCP FK 位置。
    target_position: 终点位置；和 ``target_offset`` 二选一。
    target_offset: 相对起点的位移；和 ``target_position`` 二选一。
    orientation_mode: ``current`` 保持起点 TCP 姿态，``target`` 从起点姿态
        slerp 到配置终点姿态，``none`` 表示 IK 只约束位置。
    """

    tcp_frame_name: str | None = None
    start_position: tuple[float, float, float] | None = None
    target_position: tuple[float, float, float] | None = None
    target_offset: tuple[float, float, float] | None = None
    orientation_mode: OrientationMode = "current"
    target_orientation: tuple[float, float, float, float] | None = None
    target_rpy_deg: tuple[float, float, float] | None = None
    duration_s: float = 2.0
    sample_hz: float = 100.0
    phase: str = "tcp_line"

    @classmethod
    def from_mapping(cls, data: dict) -> "MoveTcpLineConfig":
        """从 YAML 映射构造 TCP 直线配置。

        支持两种轨迹类型名：``tcp_line`` 是当前推荐名称，``cartesian_line`` 用于
        兼容早期配置。``tcp.frame_name`` 和顶层 ``tcp_frame_name`` 都可指定 TCP
        frame；后者优先级更高。
        """

        if "trajectory" not in data:
            raise ValueError(
                "TCP line config must contain top-level trajectory section"
            )
        trajectory = data["trajectory"]
        # 兼容旧配置名 ``cartesian_line``，但内部统一当作 TCP line 处理。TCP frame 可以
        # 放在 trajectory.tcp 子节，也可以顶层直接指定，便于简单 YAML 少写一层结构。
        trajectory_type = str(trajectory.get("type", "tcp_line"))
        if trajectory_type not in {"tcp_line", "cartesian_line"}:
            raise ValueError(
                f"TCP line trajectory type must be tcp_line or cartesian_line, got {trajectory_type!r}"
            )
        tcp = trajectory.get("tcp") or {}
        orientation_mode = str(trajectory.get("orientation_mode", "current")).lower()
        use_orientation = trajectory.get("use_orientation")
        if use_orientation is not None and not bool(use_orientation):
            orientation_mode = "none"
        return cls(
            tcp_frame_name=str(
                trajectory.get("tcp_frame_name") or tcp.get("frame_name") or ""
            )
            or None,
            start_position=_optional_vector3(trajectory.get("start_position")),
            target_position=_optional_vector3(trajectory.get("target_position")),
            target_offset=_optional_vector3(trajectory.get("target_offset")),
            orientation_mode=_orientation_mode(orientation_mode),
            target_orientation=_optional_quat(trajectory.get("target_orientation")),
            target_rpy_deg=_optional_vector3(trajectory.get("target_rpy_deg")),
            duration_s=float(trajectory.get("duration", cls.duration_s)),
            sample_hz=float(trajectory.get("sample_hz", cls.sample_hz)),
            phase=str(trajectory.get("phase", cls.phase)),
        )

    def validate(self) -> None:
        """检查配置是否足够构造 TCP 直线轨迹。

        这里做的是“任务语义”检查：例如终点只能用绝对位置或相对偏移中的一种；
        数组长度等结构性校验则在 ``_optional_*`` 解析函数里完成。
        """

        if not self.tcp_frame_name:
            raise ValueError("tcp_frame_name is required")
        if (self.target_position is None) == (self.target_offset is None):
            raise ValueError(
                "Exactly one of target_position or target_offset must be provided"
            )
        if (
            self.orientation_mode == "target"
            and self.target_orientation is None
            and self.target_rpy_deg is None
        ):
            raise ValueError(
                "target orientation requires target_orientation or target_rpy_deg"
            )
        if self.duration_s < 0:
            raise ValueError("duration cannot be negative")
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be positive")


@dataclass(frozen=True)
class TcpLineDiagnostics:
    """TCP 直线 IK 诊断信息。

    诊断只记录规划层关心的 TCP 起终点和最大 IK 位置误差，不保存每个 waypoint 的
    关节解，避免日志/打印输出过大。需要逐点关节数据时可直接读取返回的轨迹。
    """

    start_position: np.ndarray
    target_position: np.ndarray
    start_orientation: np.ndarray | None
    target_orientation: np.ndarray | None
    ik_joint_names: tuple[str, ...]
    max_position_error: float


def build_tcp_line_command_trajectory(
    *,
    dof_names: list[str],
    command_indices: np.ndarray,
    current_positions: np.ndarray,
    config: MoveTcpLineConfig,
    cumotion_config: CuMotionConfig | None = None,
    context=None,
) -> tuple[JointTrajectory, TcpLineDiagnostics]:
    """构建控制器命令空间的 TCP 直线关节轨迹。

    参数:
        dof_names: articulation 完整 DOF 名称，顺序和 ``current_positions`` 一致。
        command_indices: 控制器命令空间的 DOF 索引。
        current_positions: 当前完整 DOF 位置。
        config: TCP 直线运动配置。
        cumotion_config/context: 真实运行时传 ``cumotion_config``；测试可注入 fake context。
    返回:
        ``(JointTrajectory, TcpLineDiagnostics)``。轨迹的 ``joint_names`` 与
        ``command_indices`` 顺序一致，可直接交给 ``execute_joint_trajectory``。
    """

    config.validate()
    # 运行脚本会把控制器命令空间传进来。这里把它转成 numpy index 数组，后面所有
    # 轨迹输出都严格按这个顺序排列，这样可复用 joint target 的执行循环。
    command_indices = np.asarray(command_indices, dtype=int).reshape(-1)
    current = np.asarray(current_positions, dtype=float).reshape(-1)
    if current.size != len(dof_names):
        raise ValueError(
            f"current_positions expected {len(dof_names)} values, got {current.size}"
        )

    # 正常运行时从 CuMotionConfig 加载真实 cuMotion context；单元测试注入 fake
    # context，避免启动 Isaac/cuMotion，同时覆盖同一条任务数据流。
    if context is None:
        if cumotion_config is None:
            raise ValueError("cumotion_config is required when context is not provided")
        context = CuMotionContext(cumotion_config)

    # cuMotion 只求解机器人描述里的主动 C-space 关节。组合 articulation 里可能还有
    # 手部、mimic follower 等 DOF，所以必须先把 cuMotion 关节名映射回完整 DOF。
    ik_joint_names = context.joint_names()
    ik_indices = _indices_for_names(dof_names, ik_joint_names, label="cuMotion joints")
    _require_commanded_ik_joints(dof_names, command_indices, ik_indices)

    # FK 使用 cuMotion C-space 顺序，而不是 articulation 完整 DOF 顺序。若配置没有
    # 显式 start_position，就以当前 TCP pose 作为起点，保证 demo 可从任意当前姿态
    # 出发。
    current_cspace = current[ik_indices]
    fk = context.make_forward_kinematics()
    start_pose = fk.compute_pose(current_cspace, config.tcp_frame_name)
    start_position = np.asarray(
        config.start_position
        if config.start_position is not None
        else start_pose.position,
        dtype=float,
    )
    target_position = _target_position(start_position, config)
    start_orientation, target_orientation = _orientation_endpoints(
        start_pose.orientation, config
    )

    # 先在 trajectories 层生成任务空间 waypoint：位置走直线，姿态按 wxyz 四元数
    # slerp。若 orientation_mode=none，waypoint orientation 会是 None，IKRequest
    # 会只约束位置。
    solver = context.make_inverse_kinematics(tcp_frame_name=config.tcp_frame_name)
    position_tolerance = float(context.config.position_tolerance)
    orientation_tolerance = float(context.config.orientation_tolerance)
    times = sample_times(config.duration_s, config.sample_hz)
    waypoints = sample_cartesian_pose_line(
        times=times,
        start_position=start_position,
        target_position=target_position,
        start_orientation=start_orientation,
        target_orientation=target_orientation,
    )

    # 对每个 waypoint 单独 IK。第一帧若使用当前 TCP 作为起点，直接保留当前关节
    # 状态，不再求一次等价 IK，避免因解的冗余性导致起步时关节发生小跳变。
    full_targets = []
    position_errors = []
    warm_start = current_cspace
    for waypoint_index, waypoint in enumerate(waypoints):
        if waypoint_index == 0 and config.start_position is None:
            full_targets.append(current.copy())
            position_errors.append(0.0)
            continue
        result = solver.solve(
            IKRequest(
                target_position=waypoint.position,
                target_orientation=waypoint.orientation,
                tcp_frame_name=config.tcp_frame_name,
                warm_start=warm_start,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
        )
        if not result.success:
            raise RuntimeError(
                "TCP line IK failed at waypoint "
                f"{waypoint_index}/{len(waypoints) - 1}: status={result.status} "
                f"position_error={result.position_error}"
            )
        # 下一点使用上一点 IK 解热启动：这比每次用固定 seed 更容易得到连续的关节解，
        # 也是直线 waypoint IK 能平滑串起来的关键。
        warm_start = np.asarray(result.joint_positions, dtype=float).reshape(-1)

        # IK 只返回 cuMotion C-space 关节。执行器需要完整 articulation 目标，因此先
        # 从当前完整 DOF 复制一份，只覆盖 IK 关节；手部等非 IK 命令关节保持原值。
        full_target = current.copy()
        full_target[ik_indices] = warm_start
        full_targets.append(full_target)
        position_errors.append(float(result.position_error))

    # execute_joint_trajectory 接收的是 controller command space 轨迹，不是
    # 完整 DOF 轨迹。这里按 command_indices 裁剪列，再交给 trajectories 层统一
    # 由 position 采样构造 JointTrajectory。
    command_positions = np.asarray(
        [target[command_indices] for target in full_targets], dtype=float
    )
    trajectory = joint_trajectory_from_positions(
        times=times,
        positions=command_positions,
        phase=config.phase,
        joint_names=tuple(dof_names[int(index)] for index in command_indices),
    )
    diagnostics = TcpLineDiagnostics(
        start_position=start_position,
        target_position=target_position,
        start_orientation=None
        if start_orientation is None
        else start_orientation.copy(),
        target_orientation=None
        if target_orientation is None
        else target_orientation.copy(),
        ik_joint_names=tuple(ik_joint_names),
        max_position_error=max(position_errors),
    )
    return trajectory, diagnostics


def _target_position(
    start_position: np.ndarray, config: MoveTcpLineConfig
) -> np.ndarray:
    """解析 TCP 终点位置。

    ``target_position`` 是 base 坐标系下的绝对终点；``target_offset`` 是相对起点的
    位移。两者在 ``validate`` 中保证只会出现一个。
    """

    # 绝对终点和相对偏移在 validate 阶段已保证互斥；这里保留显式分支，使错误信息在未来
    # 直接调用该 helper 时仍然清晰。
    if config.target_position is not None:
        return np.asarray(config.target_position, dtype=float).reshape(3)
    if config.target_offset is not None:
        return np.asarray(start_position, dtype=float).reshape(3) + np.asarray(
            config.target_offset, dtype=float
        ).reshape(3)
    raise ValueError("Exactly one of target_position or target_offset must be provided")


def _orientation_endpoints(
    current_orientation: np.ndarray, config: MoveTcpLineConfig
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """根据姿态模式解析 slerp 的起点和终点四元数。

    返回 ``(None, None)`` 表示不约束姿态；返回同一个起终点表示保持当前姿态；
    返回不同起终点则由 ``slerp_quat_wxyz`` 生成逐点姿态。
    """

    # 姿态模式决定 IKRequest 是否传 orientation：None 表示后端只约束位置，可以在目标姿态
    # 难以满足时提高成功率；current/target 则用于保持或插值末端姿态。
    if config.orientation_mode == "none":
        return None, None
    start = normalize_quat_wxyz(current_orientation, label="current_orientation")
    if config.orientation_mode == "current":
        return start, start.copy()
    if config.target_orientation is not None:
        return start, normalize_quat_wxyz(
            config.target_orientation, label="target_orientation"
        )
    if config.target_rpy_deg is not None:
        return start, normalize_quat_wxyz(
            rpy_xyz_deg_to_quat_wxyz(config.target_rpy_deg), label="target_rpy_deg"
        )
    raise ValueError("target orientation requires target_orientation or target_rpy_deg")


def _indices_for_names(
    dof_names: list[str], joint_names: list[str], *, label: str
) -> np.ndarray:
    """把一组关节名解析成完整 DOF 索引，并在缺失时给出完整上下文。"""

    # 名称映射比直接假设索引更安全：cuMotion C-space 通常只是完整 articulation DOF 的子集。
    index_by_name = {name: index for index, name in enumerate(dof_names)}
    missing = [name for name in joint_names if name not in index_by_name]
    if missing:
        raise ValueError(
            f"{label} not found in articulation: {missing}. Available DOFs: {dof_names}"
        )
    return np.asarray([index_by_name[name] for name in joint_names], dtype=int)


def _require_commanded_ik_joints(
    dof_names: list[str], command_indices: np.ndarray, ik_indices: np.ndarray
) -> None:
    """确认控制器命令空间覆盖所有 IK 关节。

    如果某个 IK 关节没有被 controller 控制，轨迹即使求出来也无法下发到机器人。
    """

    command_index_set = {int(index) for index in command_indices}
    missing = [
        dof_names[int(index)]
        for index in ik_indices
        if int(index) not in command_index_set
    ]
    if missing:
        raise ValueError(
            f"Controller command joints must include all cuMotion IK joints; missing: {missing}"
        )


def _optional_vector3(value) -> tuple[float, float, float] | None:
    """解析可选 3D 向量；字符串 ``current`` 和缺省都表示使用当前 FK 值。"""

    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() == "current":
            return None
        raise ValueError(f"Expected a 3-vector or 'current', got {value!r}")
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"Expected 3 values, got {vector.size}")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _optional_quat(value) -> tuple[float, float, float, float] | None:
    """解析可选 wxyz 四元数。"""

    if value is None:
        return None
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 4:
        raise ValueError(f"Expected 4 quaternion values, got {vector.size}")
    return (float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3]))


def _orientation_mode(value: str) -> OrientationMode:
    """校验姿态模式并收窄类型。"""

    if value not in {"current", "target", "none"}:
        raise ValueError("orientation_mode must be one of: current, target, none")
    return value  # type: ignore[return-value]
