"""TCP 笛卡尔直线运动任务适配函数。

该模块把任务配置和 articulation 完整 DOF 状态转换成控制器可执行的关节轨迹。TCP 直线
本身的 FK/waypoint/IK/warm-start 逻辑位于 ``backends.cumotion.tcp_line``；这里保留
YAML 解析、cuMotion context 创建、完整 DOF 名称检查以及 controller command-space
轨迹裁剪。

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

import numpy as np

from manipulation_project.backends.cumotion.context import (
    CuMotionConfig,
    CuMotionContext,
)
from manipulation_project.backends.cumotion.tcp_line import plan_tcp_line_joint_path
from manipulation_project.planning.requests import (
    OrientationMode,
    TcpLineRequest,
)
from manipulation_project.planning.results import TcpLineDiagnostics
from manipulation_project.trajectories.joint_trajectory_builder import (
    joint_trajectory_from_positions,
)
from manipulation_project.trajectories.types import JointTrajectory


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
    target_rpy: 目标 RPY，固定轴 XYZ 顺序，单位 rad。
    """

    tcp_frame_name: str | None = None
    start_position: tuple[float, float, float] | None = None
    target_position: tuple[float, float, float] | None = None
    target_offset: tuple[float, float, float] | None = None
    orientation_mode: OrientationMode = "current"
    target_orientation: tuple[float, float, float, float] | None = None
    target_rpy: tuple[float, float, float] | None = None
    duration_s: float = 2.0
    sample_hz: float = 100.0
    phase: str = "tcp_line"

    @classmethod
    def from_mapping(cls, data: dict) -> "MoveTcpLineConfig":
        """从 YAML 映射构造 TCP 直线配置。

        TCP frame 通过 ``trajectory.tcp_frame_name`` 指定。
        """

        if "trajectory" not in data:
            raise ValueError(
                "TCP line config must contain top-level trajectory section"
            )
        trajectory = data["trajectory"]
        # 轨迹类型只接受接口名 ``tcp_line``，便于配置和运行日志保持一致。
        trajectory_type = str(trajectory.get("type", "tcp_line"))
        if trajectory_type != "tcp_line":
            raise ValueError(
                f"TCP line trajectory type must be tcp_line, got {trajectory_type!r}"
            )
        if "tcp" in trajectory:
            raise ValueError("trajectory.tcp is removed; use trajectory.tcp_frame_name")
        if "target_rpy_deg" in trajectory:
            raise ValueError("target_rpy_deg is removed; use target_rpy in radians")
        orientation_mode = str(trajectory.get("orientation_mode", "current")).lower()
        if "use_orientation" in trajectory:
            raise ValueError("use_orientation is removed; use orientation_mode='none'")
        return cls(
            tcp_frame_name=str(trajectory.get("tcp_frame_name") or "") or None,
            start_position=_optional_vector3(trajectory.get("start_position")),
            target_position=_optional_vector3(trajectory.get("target_position")),
            target_offset=_optional_vector3(trajectory.get("target_offset")),
            orientation_mode=_orientation_mode(orientation_mode),
            target_orientation=_optional_quat(trajectory.get("target_orientation")),
            target_rpy=_optional_vector3(trajectory.get("target_rpy")),
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
            and self.target_rpy is None
        ):
            raise ValueError(
                "target orientation requires target_orientation or target_rpy"
            )
        if self.duration_s < 0:
            raise ValueError("duration cannot be negative")
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be positive")


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
    tcp_frame_name = str(config.tcp_frame_name)
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

    # planning 层只处理后端关节顺序。这里先把 articulation 完整 DOF 切成 cuMotion
    # C-space，再把规划得到的 C-space 路径写回完整 DOF/命令空间。
    current_cspace = current[ik_indices]
    plan = plan_tcp_line_joint_path(
        context=context,
        request=TcpLineRequest(
            tcp_frame_name=tcp_frame_name,
            current_joint_positions=current_cspace,
            start_position=config.start_position,
            target_position=config.target_position,
            target_offset=config.target_offset,
            orientation_mode=config.orientation_mode,
            target_orientation=config.target_orientation,
            target_rpy=config.target_rpy,
            duration_s=config.duration_s,
            sample_hz=config.sample_hz,
            position_tolerance=float(context.config.position_tolerance),
            orientation_tolerance=float(context.config.orientation_tolerance),
        ),
    )

    full_targets = []
    for joint_positions in plan.joint_positions:
        full_target = current.copy()
        full_target[ik_indices] = joint_positions
        full_targets.append(full_target)

    # execute_joint_trajectory 接收的是 controller command space 轨迹，不是
    # 完整 DOF 轨迹。这里按 command_indices 裁剪列，再交给 trajectories 层统一
    # 由 position 采样构造 JointTrajectory。
    command_positions = np.asarray(
        [target[command_indices] for target in full_targets], dtype=float
    )
    trajectory = joint_trajectory_from_positions(
        times=plan.times,
        positions=command_positions,
        phase=config.phase,
        joint_names=tuple(dof_names[int(index)] for index in command_indices),
    )
    return trajectory, plan.diagnostics


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
