"""cuMotion 风格 TCP 直线 IK 辅助。

本模块位于 ``backends/cumotion``，负责“后端关节空间”里的 TCP 直线生成流程：
先用 FK 读取当前 TCP 位姿，再在 base 坐标系下采样一条任务空间直线，最后对每个
waypoint 单独求 IK，并用上一点的关节解作为下一点的 warm start。

职责边界:
    * 输入和输出关节向量都使用后端关节顺序，例如 cuMotion C-space 顺序。
    * 不理解 Isaac articulation 的完整 DOF 顺序，也不处理 controller command space。
      这些名称映射和裁剪由 ``tasks.move_tcp_line`` 适配层负责。
    * 不创建 cuMotion context，只消费调用方提供的最小 FK/IK context 协议；这样单元测试
      可以注入 fake context，而运行时可以传真实 ``CuMotionContext``。
    * 不做碰撞路径规划。若未来 cuMotion IK 请求需要碰撞对象，应从 ``IKRequest`` 扩展并
      在这里逐点透传。

坐标和单位:
    position 使用机器人 base 坐标系下的米制坐标；姿态使用项目边界约定的 wxyz 四元数；
    关节位置单位为 rad，时间单位为 s。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from manipulation_project.planning.requests import IKRequest, TcpLineRequest
from manipulation_project.planning.results import TcpLineDiagnostics, TcpLinePlan
from manipulation_project.trajectories.cartesian_waypoints import (
    sample_cartesian_pose_line,
)
from manipulation_project.utils.rotations import (
    normalize_quat_wxyz,
    rpy_xyz_to_quat_wxyz,
)
from manipulation_project.utils.timing import sample_times


class TcpLineKinematicsContext(Protocol):
    """TCP 直线 IK 所需的最小后端上下文协议。

    这里故意只描述 ``joint_names``、FK 和 IK 三个能力，不要求具体类型是
    ``CuMotionContext``。这样调用方既可以传真实后端，也可以在测试中传轻量替身。
    """

    def joint_names(self) -> list[str]:
        """返回后端关节顺序。

        ``planning.requests.TcpLineRequest.current_joint_positions`` 必须按这个顺序排列，
        返回的 ``planning.results.TcpLinePlan.joint_positions`` 也保持这个顺序。
        """

    def make_forward_kinematics(self):
        """返回 FK 对象。

        FK 对象需要提供 ``compute_pose(joint_positions, frame_name)``，其中
        ``joint_positions`` 使用后端关节顺序，返回 pose 至少包含 ``position`` 和
        ``orientation`` 字段。
        """

    def make_inverse_kinematics(self, *, tcp_frame_name: str | None = None):
        """返回 IK 求解器对象。

        IK 对象需要提供 ``solve(IKRequest)``，并返回项目统一的 ``IKResult``。
        """


def plan_tcp_line_joint_path(
    *, context: TcpLineKinematicsContext, request: TcpLineRequest
) -> TcpLinePlan:
    """按后端关节顺序生成 TCP 直线关节路径。

    流程:
        1. 校验请求，并读取后端关节名。
        2. 用当前关节位置做 FK，得到当前 TCP pose。
        3. 解析直线起点/终点和姿态端点。
        4. 根据 ``duration_s`` 和 ``sample_hz`` 采样任务空间 waypoint。
        5. 从当前关节位置开始逐点 IK，每个成功解作为下一点 warm start。

    参数:
        context: 提供后端关节名、FK 和 IK 的最小运动学上下文。
        request: TCP 直线请求；``current_joint_positions`` 必须按
            ``context.joint_names()`` 排列。
    返回:
        ``TcpLinePlan``，其中 ``joint_positions`` 每行都是后端关节顺序的一帧目标。
    """

    request.validate_structure()
    ik_joint_names = tuple(context.joint_names())
    current = np.asarray(request.current_joint_positions, dtype=float).reshape(-1)
    if current.size != len(ik_joint_names):
        raise ValueError(
            f"current_joint_positions expected {len(ik_joint_names)} values, got {current.size}"
        )

    fk = context.make_forward_kinematics()
    # FK 始终使用后端关节顺序；如果调用方传的是完整 articulation DOF，这里会在长度
    # 检查或后端 FK 中尽早暴露，而不会静默把关节写错位。
    start_pose = fk.compute_pose(current, request.tcp_frame_name)
    start_position = np.asarray(
        request.start_position
        if request.start_position is not None
        else start_pose.position,
        dtype=float,
    )
    target_position = _target_position(start_position, request)
    start_orientation, target_orientation = _orientation_endpoints(
        start_pose.orientation, request
    )

    solver = context.make_inverse_kinematics(tcp_frame_name=request.tcp_frame_name)
    times = sample_times(request.duration_s, request.sample_hz)
    # waypoint 位于任务空间：位置沿直线插值，姿态按 wxyz 四元数 slerp；如果不约束姿态，
    # 每个 waypoint 的 orientation 会是 None，后端 IKRequest 会只约束目标位置。
    waypoints = sample_cartesian_pose_line(
        times=times,
        start_position=start_position,
        target_position=target_position,
        start_orientation=start_orientation,
        target_orientation=target_orientation,
    )

    joint_targets = []
    position_errors = []
    warm_start_ik_cspace_seed = current
    for waypoint_index, waypoint in enumerate(waypoints):
        # 当起点来自当前 FK 时，第一帧已经等于当前关节状态，没有必要再求一次等价 IK。
        # 对冗余机械臂来说，重复求解可能返回另一个等价姿态，导致轨迹第一步小跳变。
        if waypoint_index == 0 and request.start_position is None:
            joint_targets.append(current.copy())
            position_errors.append(0.0)
            continue
        result = solver.solve(
            IKRequest(
                target_position=waypoint.position,
                target_orientation=waypoint.orientation,
                tcp_frame_name=request.tcp_frame_name,
                warm_start_ik_cspace_seed=warm_start_ik_cspace_seed,
                position_tolerance=request.position_tolerance,
                orientation_tolerance=request.orientation_tolerance,
            )
        )
        if not result.success:
            # IK 失败表示路径不可执行，直接抛出带 waypoint 序号和误差的异常，让任务层
            # 可以停止执行，而不是返回一条半截轨迹。
            raise RuntimeError(
                "TCP line IK failed at waypoint "
                f"{waypoint_index}/{len(waypoints) - 1}: status={result.status} "
                f"position_error={result.position_error}"
            )
        warm_start_ik_cspace_seed = np.asarray(
            result.joint_positions, dtype=float
        ).reshape(-1)
        # warm-start seed 更新为上一点成功解，使连续 waypoint 更容易落在同一个 IK 解分支上。
        joint_targets.append(warm_start_ik_cspace_seed.copy())
        position_errors.append(float(result.position_error))

    diagnostics = TcpLineDiagnostics(
        start_position=start_position,
        target_position=target_position,
        start_orientation=None
        if start_orientation is None
        else start_orientation.copy(),
        target_orientation=None
        if target_orientation is None
        else target_orientation.copy(),
        ik_joint_names=ik_joint_names,
        max_position_error=max(position_errors),
    )
    return TcpLinePlan(
        times=times,
        joint_positions=np.asarray(joint_targets, dtype=float),
        diagnostics=diagnostics,
    )


def _target_position(start_position: np.ndarray, request: TcpLineRequest) -> np.ndarray:
    """解析 TCP 终点位置。

    ``target_position`` 表示 base 坐标系下的绝对终点；``target_offset`` 表示相对起点的
    位移。两者互斥关系已在 ``validate`` 中检查，这里保留显式分支便于未来直接调用。
    """

    if request.target_position is not None:
        return np.asarray(request.target_position, dtype=float).reshape(3)
    if request.target_offset is not None:
        return np.asarray(start_position, dtype=float).reshape(3) + np.asarray(
            request.target_offset, dtype=float
        ).reshape(3)
    raise ValueError("Exactly one of target_position or target_offset must be provided")


def _orientation_endpoints(
    current_orientation: np.ndarray, request: TcpLineRequest
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """按姿态模式解析插值端点。

    返回 ``(None, None)`` 表示不约束姿态；返回相同起终点表示保持当前姿态；返回不同
    起终点表示后续 waypoint 会在两者之间做 slerp。
    """

    if request.orientation_mode == "none":
        return None, None
    start = normalize_quat_wxyz(current_orientation, label="current_orientation")
    if request.orientation_mode == "current":
        return start, start.copy()
    if request.target_orientation is not None:
        return start, normalize_quat_wxyz(
            request.target_orientation, label="target_orientation"
        )
    if request.target_rpy is not None:
        return start, normalize_quat_wxyz(
            rpy_xyz_to_quat_wxyz(request.target_rpy),
            label="target_rpy",
        )
    raise ValueError("target orientation requires target_orientation or target_rpy")
