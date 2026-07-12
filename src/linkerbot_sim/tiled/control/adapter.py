"""同步 tiled action 到完整 batch command target 的转换器。

本模块刻意不接入路径规划、MoveSpec 或异步队列。每个 action 都必须在进入 physics
tick 循环前转换成 batched joint target，然后固定 decimation 次数地调用
``world.step()``。这样才能保证所有 env 在同一个仿真时间线上同步推进。

支持的动作分三类:
    * 关节空间: ``joint_position_target``、``joint_delta_pos``、``hold``。
    * 末端空间: ``ee_pose_target``、``ee_delta_pos``、``ee_delta_pose``，它们先通过
      batched IK 转成关节目标。
    * 固定步长路径: ``ee_linear_path`` 按 ``sample_dt_s`` 对所有 env 做顺序 batched
      IK，再把关节 waypoint 重采样到统一 physics tick 网格。

graph search、trajectory optimization 或变长路径执行由异步 planner/timeline 层负责，不进入
这里的 ``step`` 热路径。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

from linkerbot_sim.planning.batch_ik import (
    BatchIKBackend,
    BatchIKResult,
    apply_ik_failure_fallback,
)
from linkerbot_sim.tiled.control.interpolation import (
    _interpolation_alpha,
    interpolate_joint_targets,
)
from linkerbot_sim.tiled.control.types import (
    TiledCommandAction,
    TiledCommandTarget,
    TiledCommandTrajectory,
)


class TiledIKRequestRejected(RuntimeError):
    """同步 IK 严格策略拒绝整条 request，并保留失败 env identity。"""

    def __init__(self, failed_env_ids: Sequence[int]) -> None:
        self.failed_env_ids = tuple(sorted({int(env_id) for env_id in failed_env_ids}))
        self.failure_policy = "reject_request"
        super().__init__(
            "synchronous IK request rejected; failed env_ids: "
            f"{list(self.failed_env_ids)}"
        )


def selected_ik_failure_env_ids(
    info: Mapping[str, object],
    *,
    env_ids: Sequence[int] | np.ndarray,
    num_envs: int,
) -> tuple[int, ...]:
    """从 IK info 中只提取当前 selector 覆盖的失败 env IDs。"""

    success_value = info.get("ik_success")
    if success_value is None:
        return ()
    success = np.asarray(success_value, dtype=bool).reshape(-1)
    if success.shape != (int(num_envs),):
        raise ValueError("ik_success must have shape (num_envs,)")
    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    return tuple(sorted(int(env_id) for env_id in selected if not success[env_id]))


class TiledCommandAdapter:
    """把同步 tiled action 转换成 batched command-space 关节目标。

    Adapter 持有少量运行时状态:
        * ``last_target``: ``hold`` 和 IK 失败 fallback 会用到的上一帧目标。
        * ``ik_solver``: 可选 batched IK 后端，只在 ``ee_*`` action 中使用。

    它不拥有 Isaac articulation，也不调用 ``world.step()``。这样可以把“动作语义”
    和“物理推进”分开，便于用 fake solver/articulation 做普通单元测试。
    """

    def __init__(
        self,
        *,
        num_envs: int,
        command_dim: int,
        default_decimation: int = 1,
        tcp_frame_name: str | None = None,
        ik_solver: BatchIKBackend | None = None,
        failure_policy: str = "hold_failed_env",
    ) -> None:
        """创建 action adapter。

        参数:
            num_envs: tiled env 数量。
            command_dim: controller command-space 维度，不包含 mimic follower。
            default_decimation: action 未显式指定 decimation 时使用的固定 tick 数。
            tcp_frame_name: 默认 TCP frame；单个 action 可覆盖。
            ik_solver: 末端空间 action 使用的 batched IK solver。
        """

        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        if int(command_dim) < 1:
            raise ValueError("command_dim must be positive")
        if int(default_decimation) < 1:
            raise ValueError("default_decimation must be positive")
        normalized_failure_policy = str(failure_policy).strip()
        if normalized_failure_policy not in {"hold_failed_env", "reject_request"}:
            raise ValueError(
                "failure_policy must be one of: hold_failed_env, reject_request"
            )
        self.num_envs = int(num_envs)
        self.command_dim = int(command_dim)
        self.default_decimation = int(default_decimation)
        self.tcp_frame_name = tcp_frame_name
        self.ik_solver = ik_solver
        self.failure_policy = normalized_failure_policy
        self.last_target: np.ndarray | None = None

    def action_to_joint_target(
        self,
        action: TiledCommandAction,
        *,
        current_positions: np.ndarray,
        current_tcp_positions: np.ndarray | None = None,
        current_tcp_orientations_wxyz: np.ndarray | None = None,
        env_origins: np.ndarray | None = None,
        update_last_target: bool = True,
    ) -> TiledCommandTarget:
        """把一个高层 action 转换成 batched command-space target。

        这里是 tiled step-control 的核心边界:
            * 关节 action 直接得到 ``(N, C)`` 目标。
            * 末端 action 先求 batched IK，再得到 ``(N, C)`` 目标。
            * 函数返回后，physics tick 循环只做插值、apply target 和 world.step。

        这样可以避免每个 env 在 step 中单独规划或执行不同长度的轨迹。
        """

        current = _batched_array(
            current_positions,
            self.num_envs,
            self.command_dim,
            "current_positions",
        )
        info: dict[str, np.ndarray] = {}
        if action.kind == "hold":
            # hold 优先保持上一帧 target；若还没有 target，则保持当前关节位置。
            target = (
                self.last_target.copy() if self.last_target is not None else current
            )
        elif action.kind == "joint_position_target":
            # 绝对关节目标，通常由 MPC 候选或上层控制器直接给出。
            target = _required_action_values(
                action.values,
                self.num_envs,
                self.command_dim,
                "joint_position_target.values",
            )
        elif action.kind == "joint_delta_pos":
            # 关节增量动作适合策略输出小动作；目标以当前 command-space 位置为基准。
            delta = _required_action_values(
                action.values,
                self.num_envs,
                self.command_dim,
                "joint_delta_pos.values",
            )
            target = current + delta
        elif action.kind == "ee_pose_target":
            # 绝对 TCP 位姿。默认 env-local，所以要把 env origin 加回 world/base 求解语义。
            pose = _required_action_values(
                action.values, self.num_envs, 7, "ee_pose_target.values"
            )
            target_positions = pose[:, :3].copy()
            if action.pose_reference_frame == "env":
                # env-local 广播是 tiled env 最常见语义：同一个 action 在每个局部场景
                # 内表示同一个目标，而不是所有 env 共享同一个 world 坐标点。
                target_positions = target_positions + _batched_array(
                    env_origins,
                    self.num_envs,
                    3,
                    "env_origins",
                )
            target = self._solve_ik(
                target_positions=target_positions,
                target_orientations_wxyz=pose[:, 3:7],
                seeds=current,
                action=action,
                info=info,
            )
        elif action.kind == "ee_delta_pos":
            # TCP 平移微动保持当前姿态，是 PushT/MPC 类 step-control 最常用的笛卡尔动作。
            delta = _required_action_values(
                action.values, self.num_envs, 3, "ee_delta_pos.values"
            )
            tcp_positions = _batched_array(
                current_tcp_positions,
                self.num_envs,
                3,
                "current_tcp_positions",
            )
            tcp_orientations = _batched_array(
                current_tcp_orientations_wxyz,
                self.num_envs,
                4,
                "current_tcp_orientations_wxyz",
            )
            target = self._solve_ik(
                target_positions=tcp_positions + delta,
                target_orientations_wxyz=tcp_orientations,
                seeds=current,
                action=action,
                info=info,
            )
        elif action.kind == "ee_delta_pose":
            # 位姿微动允许调用方同时给平移和旋转增量；旋转可以是 rotvec 或目标四元数。
            target_positions, target_orientations = _ee_delta_pose_target(
                values=action.values,
                current_tcp_positions=current_tcp_positions,
                current_tcp_orientations_wxyz=current_tcp_orientations_wxyz,
                num_envs=self.num_envs,
            )
            target = self._solve_ik(
                target_positions=target_positions,
                target_orientations_wxyz=target_orientations,
                seeds=current,
                action=action,
                info=info,
            )
        elif action.kind == "ee_linear_path":
            raise ValueError(
                "ee_linear_path requires linear_path_to_joint_trajectory()"
            )
        else:
            raise ValueError(f"Unsupported tiled command kind: {action.kind!r}")

        # 缓存目标用于下一次 hold，也作为 IK 失败时可选的稳定参考。
        if update_last_target:
            self.last_target = np.asarray(target, dtype=float).copy()
        return TiledCommandTarget(target, info=info)

    def decimation_for(self, action: TiledCommandAction) -> int:
        """返回当前 action 使用的 physics tick 展开倍数。"""

        if action.duration_s is not None:
            raise ValueError("duration_s requires runtime physics_dt resolution")

        return (
            self.default_decimation
            if action.decimation is None
            else int(action.decimation)
        )

    def interpolate_to(
        self,
        target: np.ndarray,
        *,
        start: np.ndarray,
        action: TiledCommandAction,
    ) -> np.ndarray:
        """返回一个 command step 内每个 physics tick 的 batched joint target。

        返回 shape 为 ``(steps, num_envs, command_dim)``。调用方应按第一维顺序下发
        target，每次下发所有 env 后只调用一次 ``world.step()``。
        """

        return interpolate_joint_targets(
            start=start,
            target=target,
            steps=self.decimation_for(action),
            mode=action.interpolation,
        )

    def linear_path_to_joint_trajectory(
        self,
        action: TiledCommandAction,
        *,
        steps: int,
        execution_steps: int | None = None,
        current_positions: np.ndarray,
        current_tcp_positions: np.ndarray,
        current_tcp_orientations_wxyz: np.ndarray,
        env_origins: np.ndarray | None = None,
        active_env_ids: np.ndarray | None = None,
        update_last_target: bool = True,
    ) -> TiledCommandTrajectory:
        """用逐 waypoint batched IK 生成固定 tick 的 TCP 直线轨迹。

        相对终点是 world-frame TCP 位移，绝对终点按 ``pose_reference_frame`` 解释。
        每个 waypoint 都使用上一 waypoint 的关节解作为 seed。某个 env 首次失败后，
        该行在剩余 tick 中保持最后一个成功 target；其它 env 继续求解，因此所有 env
        的 physics tick 数仍一致。
        """

        if action.kind != "ee_linear_path":
            raise ValueError("linear path trajectory requires kind='ee_linear_path'")
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be positive")
        output_steps = steps if execution_steps is None else int(execution_steps)
        if output_steps < 1:
            raise ValueError("execution_steps must be positive")
        seeds = _batched_array(
            current_positions,
            self.num_envs,
            self.command_dim,
            "current_positions",
        ).copy()
        start_joint_positions = seeds.copy()
        start_positions = _batched_array(
            current_tcp_positions,
            self.num_envs,
            3,
            "current_tcp_positions",
        )
        orientations = _normalize_quaternions(
            _batched_array(
                current_tcp_orientations_wxyz,
                self.num_envs,
                4,
                "current_tcp_orientations_wxyz",
            )
        )
        target_positions = _linear_path_target_positions(
            action,
            start_positions=start_positions,
            env_origins=env_origins,
            num_envs=self.num_envs,
        )
        target_orientations = _linear_path_target_orientations(
            action,
            start_orientations_wxyz=orientations,
            num_envs=self.num_envs,
        )
        selected = _active_env_mask(active_env_ids, self.num_envs)
        solving = selected.copy()
        path_success = np.ones(self.num_envs, dtype=bool)
        first_failure_step = np.full(self.num_envs, -1, dtype=int)
        completed_steps = np.zeros(self.num_envs, dtype=int)
        max_position_error = np.zeros(self.num_envs, dtype=float)
        max_orientation_error: np.ndarray | None = None
        trajectory = np.empty((steps, self.num_envs, self.command_dim), dtype=float)

        for step_index in range(steps):
            alpha = _interpolation_alpha(
                step_index=step_index,
                steps=steps,
                mode=action.interpolation,
            )
            result = self._solve_ik_result(
                target_positions=start_positions
                + alpha * (target_positions - start_positions),
                target_orientations_wxyz=(
                    None
                    if target_orientations is None
                    else _slerp_quaternions_wxyz(
                        orientations, target_orientations, alpha
                    )
                ),
                seeds=seeds,
                action=action,
            )
            attempted = solving.copy()
            max_position_error[attempted] = np.maximum(
                max_position_error[attempted], result.position_error[attempted]
            )
            if result.orientation_error is not None:
                if max_orientation_error is None:
                    max_orientation_error = np.zeros(self.num_envs, dtype=float)
                max_orientation_error[attempted] = np.maximum(
                    max_orientation_error[attempted],
                    result.orientation_error[attempted],
                )
            accepted = attempted & result.success
            failed = attempted & ~result.success
            first_failure_step[failed] = step_index + 1
            path_success[failed] = False
            completed_steps[accepted] += 1
            seeds = np.where(accepted[:, None], result.joint_positions, seeds)
            solving = accepted
            trajectory[step_index] = seeds

        info = {
            "ik_success": path_success,
            "ik_first_failure_step": first_failure_step,
            "ik_completed_steps": completed_steps,
            "ik_position_error": max_position_error,
        }
        if max_orientation_error is not None:
            info["ik_orientation_error"] = max_orientation_error
        if update_last_target:
            self.last_target = seeds.copy()
        return TiledCommandTrajectory(
            _resample_joint_waypoints(
                start_joint_positions,
                trajectory,
                execution_steps=output_steps,
                interpolation=action.interpolation,
            ),
            info=info,
        )

    def reset(self) -> None:
        """清空 adapter 内部缓存。

        reset 或 set_state_dict 后应调用本方法，避免上一 episode 的 hold target 泄漏到
        下一次 rollout。
        """

        self.last_target = None

    def _solve_ik(
        self,
        *,
        target_positions: np.ndarray,
        target_orientations_wxyz: np.ndarray | None,
        seeds: np.ndarray,
        action: TiledCommandAction,
        info: dict[str, np.ndarray],
    ) -> np.ndarray:
        """调用 batched IK，并对失败 env 应用 fallback。

        IK 失败是每个 env 独立的诊断，不能让整个 batch 崩溃。失败行默认保持传入的
        ``seeds``，通常就是当前关节位置或上一帧稳定 target。
        """

        result = self._solve_ik_result(
            target_positions=target_positions,
            target_orientations_wxyz=target_orientations_wxyz,
            seeds=seeds,
            action=action,
        )
        info["ik_success"] = result.success
        info["ik_position_error"] = result.position_error
        if result.orientation_error is not None:
            info["ik_orientation_error"] = result.orientation_error
        return apply_ik_failure_fallback(result, seeds)

    def _solve_ik_result(
        self,
        *,
        target_positions: np.ndarray,
        target_orientations_wxyz: np.ndarray | None,
        seeds: np.ndarray,
        action: TiledCommandAction,
    ) -> BatchIKResult:
        """执行一次 batched IK 并规范化后端结果。"""

        if self.ik_solver is None:
            raise RuntimeError(f"{action.kind} requires a batched IK solver")
        tcp_frame_name = action.tcp_frame_name or self.tcp_frame_name
        if tcp_frame_name is None:
            raise ValueError(f"{action.kind} requires tcp_frame_name")
        result = self.ik_solver.solve(
            target_positions=target_positions,
            target_orientations_wxyz=target_orientations_wxyz,
            seeds=seeds,
            tcp_frame_name=tcp_frame_name,
        )
        normalized = (
            result
            if isinstance(result, BatchIKResult)
            else BatchIKResult(
                joint_positions=result.joint_positions,
                success=result.success,
                position_error=result.position_error,
                orientation_error=result.orientation_error,
                status=result.status,
            )
        )
        if normalized.joint_positions.shape != np.asarray(seeds).shape:
            raise ValueError("batched IK joint_positions must match seeds shape")
        return normalized


def _active_env_mask(env_ids: np.ndarray | None, num_envs: int) -> np.ndarray:
    """把可选 env IDs 转成固定 batch mask。"""

    if env_ids is None:
        return np.ones(int(num_envs), dtype=bool)
    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    if selected.size == 0:
        raise ValueError("active_env_ids cannot be empty")
    if np.unique(selected).size != selected.size:
        raise ValueError("active_env_ids cannot contain duplicates")
    if np.any(selected < 0) or np.any(selected >= int(num_envs)):
        raise ValueError("active_env_ids contains out-of-range env id")
    mask = np.zeros(int(num_envs), dtype=bool)
    mask[selected] = True
    return mask


def _required_action_values(
    values: np.ndarray | None,
    num_envs: int,
    width: int,
    label: str,
) -> np.ndarray:
    """读取必需 action 数值，并按 env 维度规范化。"""

    if values is None:
        raise ValueError(f"{label} is required")
    return _batched_array(values, num_envs, width, label)


def _batched_array(
    values: np.ndarray | None,
    num_envs: int,
    width: int,
    label: str,
) -> np.ndarray:
    """把输入规范化为 ``(num_envs, width)``。

    第一维为 1 时允许广播。这是 tiled rollout 常见用法：同一个 action/state 写入所有
    env，用于验证“相同状态 + 相同动作 -> 相同 rollout”。
    """

    if values is None:
        raise ValueError(f"{label} is required")
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} must have shape (N, {width})")
    if array.shape[0] == 1 and num_envs != 1:
        return np.repeat(array, num_envs, axis=0)
    if array.shape[0] != num_envs:
        raise ValueError(f"{label} first dimension must be 1 or num_envs")
    return array.astype(float, copy=True)


def _ee_delta_pose_target(
    *,
    values: np.ndarray | None,
    current_tcp_positions: np.ndarray | None,
    current_tcp_orientations_wxyz: np.ndarray | None,
    num_envs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把 TCP 位姿增量转换成目标 TCP 位姿。

    ``values`` 支持两种 shape:
        * ``(N, 6)``: 前 3 维是平移增量，后 3 维是旋转向量增量。
        * ``(N, 7)``: 前 3 维是平移增量，后 4 维直接作为目标四元数。
    """

    action_values = np.asarray(values, dtype=float) if values is not None else None
    if action_values is None:
        raise ValueError("ee_delta_pose.values is required")
    if action_values.ndim == 1:
        action_values = action_values.reshape(1, -1)
    if action_values.ndim != 2 or action_values.shape[1] not in {6, 7}:
        raise ValueError("ee_delta_pose.values must have shape (N, 6) or (N, 7)")
    if action_values.shape[0] == 1 and num_envs != 1:
        action_values = np.repeat(action_values, num_envs, axis=0)
    if action_values.shape[0] != num_envs:
        raise ValueError("ee_delta_pose.values first dimension must be 1 or num_envs")
    tcp_positions = _batched_array(
        current_tcp_positions, num_envs, 3, "current_tcp_positions"
    )
    tcp_orientations = _batched_array(
        current_tcp_orientations_wxyz,
        num_envs,
        4,
        "current_tcp_orientations_wxyz",
    )
    target_positions = tcp_positions + action_values[:, :3]
    if action_values.shape[1] == 7:
        target_orientations = _normalize_quaternions(action_values[:, 3:7])
    else:
        target_orientations = _apply_rotvec_delta(
            tcp_orientations, action_values[:, 3:6]
        )
    return target_positions, target_orientations


def _normalize_quaternions(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """归一化 wxyz 四元数，并拒绝零四元数。"""

    quats = np.asarray(quaternions_wxyz, dtype=float)
    norms = np.linalg.norm(quats, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("quaternion values must be non-zero")
    return quats / norms[:, None]


def _linear_path_target_positions(
    action: TiledCommandAction,
    *,
    start_positions: np.ndarray,
    env_origins: np.ndarray | None,
    num_envs: int,
) -> np.ndarray:
    """把同步直线路径终点规范化为 world-frame batch。"""

    if action.target_position is not None:
        target = _required_action_values(
            action.target_position,
            num_envs,
            3,
            "ee_linear_path.target_position",
        ).copy()
        if action.pose_reference_frame == "env":
            target += _batched_array(env_origins, num_envs, 3, "env_origins")
        return target
    source = action.target_offset if action.target_offset is not None else action.values
    label = (
        "ee_linear_path.target_offset"
        if action.target_offset is not None
        else "ee_linear_path.values"
    )
    offset = _required_action_values(source, num_envs, 3, label)
    return np.asarray(start_positions, dtype=float) + offset


def _linear_path_target_orientations(
    action: TiledCommandAction,
    *,
    start_orientations_wxyz: np.ndarray,
    num_envs: int,
) -> np.ndarray | None:
    """返回直线路径终点姿态；None 表示 position-only IK。"""

    if action.orientation_mode == "free":
        return None
    if action.orientation_mode == "current":
        return np.asarray(start_orientations_wxyz, dtype=float).copy()
    assert action.target_orientation_wxyz is not None
    return _normalize_quaternions(
        _required_action_values(
            action.target_orientation_wxyz,
            num_envs,
            4,
            "ee_linear_path.target_orientation_quat_wxyz",
        )
    )


def _slerp_quaternions_wxyz(
    start: np.ndarray, target: np.ndarray, alpha: float | np.ndarray
) -> np.ndarray:
    """对一批 wxyz 四元数做最短弧 Slerp。"""

    start_q = _normalize_quaternions(np.asarray(start, dtype=float))
    target_q = _normalize_quaternions(np.asarray(target, dtype=float))
    if start_q.shape != target_q.shape:
        raise ValueError("Slerp start and target quaternion batches must match")
    progress = np.asarray(alpha, dtype=float)
    if progress.ndim == 0:
        progress = np.full(start_q.shape[0], float(progress), dtype=float)
    progress = progress.reshape(-1)
    if progress.shape != (start_q.shape[0],):
        raise ValueError("Slerp alpha must be scalar or have shape (N,)")
    progress = np.clip(progress, 0.0, 1.0)
    start_rotation = Rotation.from_quat(start_q[:, [1, 2, 3, 0]])
    target_rotation = Rotation.from_quat(target_q[:, [1, 2, 3, 0]])
    relative_rotvec = (target_rotation * start_rotation.inv()).as_rotvec()
    interpolated = (
        Rotation.from_rotvec(relative_rotvec * progress[:, None]) * start_rotation
    )
    return interpolated.as_quat()[:, [3, 0, 1, 2]]


def _resample_joint_waypoints(
    start: np.ndarray,
    waypoints: np.ndarray,
    *,
    execution_steps: int,
    interpolation: str,
) -> np.ndarray:
    """按相同 path progress 把 IK waypoint 重采样到 physics ticks。"""

    samples = np.asarray(waypoints, dtype=float)
    initial = np.asarray(start, dtype=float)
    if samples.ndim != 3 or initial.shape != samples.shape[1:]:
        raise ValueError("joint waypoint start/trajectory shapes do not match")
    target_steps = int(execution_steps)
    if target_steps < 1:
        raise ValueError("execution_steps must be positive")
    if target_steps == samples.shape[0]:
        return samples.copy()

    source_steps = samples.shape[0]
    anchors = np.concatenate([initial[None, :, :], samples], axis=0)
    source_progress = np.concatenate(
        [
            np.asarray([0.0], dtype=float),
            np.asarray(
                [
                    _interpolation_alpha(
                        step_index=index,
                        steps=source_steps,
                        mode=interpolation,
                    )
                    for index in range(source_steps)
                ],
                dtype=float,
            ),
        ]
    )
    target_progress = np.asarray(
        [
            _interpolation_alpha(
                step_index=index,
                steps=target_steps,
                mode=interpolation,
            )
            for index in range(target_steps)
        ],
        dtype=float,
    )
    interpolator = make_interp_spline(
        source_progress,
        anchors,
        k=1,
        axis=0,
    )
    return np.asarray(interpolator(target_progress), dtype=float)


def _apply_rotvec_delta(
    current_orientations_wxyz: np.ndarray, delta_rotvec: np.ndarray
) -> np.ndarray:
    """把旋转向量增量左乘到当前 wxyz 四元数上。"""

    current = _normalize_quaternions(current_orientations_wxyz)
    current_xyzw = current[:, [1, 2, 3, 0]]
    delta = Rotation.from_rotvec(np.asarray(delta_rotvec, dtype=float))
    composed = delta * Rotation.from_quat(current_xyzw)
    xyzw = composed.as_quat()
    return xyzw[:, [3, 0, 1, 2]]
