"""tiled runtime 的同步 command action 层。

本模块刻意不接入路径规划、MoveSpec 或异步队列。每个 action 都必须在进入 physics
tick 循环前转换成 batched joint target，然后固定 decimation 次数地调用
``world.step()``。这样才能保证所有 env 在同一个仿真时间线上同步推进。

支持的动作分两类:
    * 关节空间: ``joint_position_target``、``joint_delta_pos``、``hold``。
    * 末端空间: ``ee_pose_target``、``ee_delta_pos``、``ee_delta_pose``，它们先通过
      batched IK 转成关节目标。

如果将来需要 graph search、trajectory optimization 或变长路径执行，应放到旧 motion
runtime 或单独 planner 模块中，不要放进这里的 ``step`` 热路径。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from linkerbot_sim.tiled.batched_ik import (
    BatchedIKResult,
    BatchedIKSolver,
    apply_ik_failure_fallback,
)


SUPPORTED_COMMAND_KINDS = frozenset(
    {
        "hold",
        "joint_position_target",
        "joint_delta_pos",
        "ee_pose_target",
        "ee_delta_pos",
        "ee_delta_pose",
    }
)
SUPPORTED_INTERPOLATIONS = frozenset({"linear", "smoothstep"})
SUPPORTED_POSE_REFERENCE_FRAMES = frozenset({"env", "base", "world"})


@dataclass(frozen=True)
class TiledCommandAction:
    """作用于全部 tiled env 的固定步长 command action。

    字段语义:
        kind: action 类型，必须在 ``SUPPORTED_COMMAND_KINDS`` 中。
        values: action 数值，第一维是 env 维度；第一维为 1 时允许广播到所有 env。
        decimation: 一个高层 command step 展开成多少个 physics tick。
        interpolation: 关节目标插值方式；第一版只支持 linear/smoothstep。
        tcp_frame_name: 末端 IK 使用的 TCP frame。
        pose_reference_frame: 绝对末端位姿的参考系。默认 ``env``，避免多个 env
            广播同一个 world pose 后都指向同一个世界点。
    """

    kind: str
    values: np.ndarray | None = None
    decimation: int | None = None
    interpolation: str = "smoothstep"
    tcp_frame_name: str | None = None
    pose_reference_frame: str = "env"

    def __post_init__(self) -> None:
        """尽早校验 action 元数据，避免错误进入 runtime 热路径。"""

        if self.kind not in SUPPORTED_COMMAND_KINDS:
            raise ValueError(f"Unsupported tiled command kind: {self.kind!r}")
        if self.decimation is not None and int(self.decimation) < 1:
            raise ValueError("TiledCommandAction.decimation must be positive")
        if self.interpolation not in SUPPORTED_INTERPOLATIONS:
            raise ValueError(f"Unsupported interpolation mode: {self.interpolation!r}")
        if self.pose_reference_frame not in SUPPORTED_POSE_REFERENCE_FRAMES:
            raise ValueError(
                f"Unsupported pose_reference_frame: {self.pose_reference_frame!r}"
            )
        if self.kind == "hold" and self.values is not None:
            values = np.asarray(self.values)
            if values.size:
                raise ValueError("hold action values must be None or empty")


@dataclass(frozen=True)
class TiledCommandTarget:
    """由 command action 转换得到的 batched 关节目标。

    ``joint_positions`` 的 shape 固定为 ``(num_envs, command_dim)``。``info`` 用于
    携带 IK success mask、误差等非任务语义诊断，不放 reward/success 这类下游任务概念。
    """

    joint_positions: np.ndarray
    info: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """把 joint target 固定成二维 float 数组，保证 env 维度存在。"""

        q = np.asarray(self.joint_positions, dtype=float)
        if q.ndim != 2:
            raise ValueError("TiledCommandTarget.joint_positions must be 2D")
        object.__setattr__(self, "joint_positions", q)


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
        ik_solver: BatchedIKSolver | None = None,
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
        self.num_envs = int(num_envs)
        self.command_dim = int(command_dim)
        self.default_decimation = int(default_decimation)
        self.tcp_frame_name = tcp_frame_name
        self.ik_solver = ik_solver
        self.last_target: np.ndarray | None = None

    def action_to_joint_target(
        self,
        action: TiledCommandAction,
        *,
        current_positions: np.ndarray,
        current_tcp_positions: np.ndarray | None = None,
        current_tcp_orientations_wxyz: np.ndarray | None = None,
        env_origins: np.ndarray | None = None,
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
        else:
            raise ValueError(f"Unsupported tiled command kind: {action.kind!r}")

        # 缓存目标用于下一次 hold，也作为 IK 失败时可选的稳定参考。
        self.last_target = np.asarray(target, dtype=float).copy()
        return TiledCommandTarget(target, info=info)

    def decimation_for(self, action: TiledCommandAction) -> int:
        """返回当前 action 使用的 physics tick 展开倍数。"""

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
        if not isinstance(result, BatchedIKResult):
            result = BatchedIKResult(
                joint_positions=result.joint_positions,
                success=result.success,
                position_error=result.position_error,
                orientation_error=result.orientation_error,
                status=result.status,
            )
        info["ik_success"] = result.success
        info["ik_position_error"] = result.position_error
        if result.orientation_error is not None:
            info["ik_orientation_error"] = result.orientation_error
        return apply_ik_failure_fallback(result, seeds)


def interpolate_joint_targets(
    *,
    start: np.ndarray,
    target: np.ndarray,
    steps: int,
    mode: str = "smoothstep",
) -> np.ndarray:
    """在固定 physics step 数内插值 batched joint targets。

    ``smoothstep`` 会让起止速度更平滑，适合用作两个 command target 之间的默认过渡。
    这里不做速度/加速度限制规划；如果需要真正的轨迹优化，应走旧 motion runtime。
    """

    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if start_array.shape != target_array.shape or start_array.ndim != 2:
        raise ValueError("start and target must be 2D arrays with matching shape")
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be positive")
    if mode not in SUPPORTED_INTERPOLATIONS:
        raise ValueError(f"Unsupported interpolation mode: {mode!r}")
    result = np.empty((steps, *target_array.shape), dtype=float)
    for index in range(steps):
        alpha = float(index + 1) / float(steps)
        if mode == "smoothstep":
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        result[index] = start_array + (target_array - start_array) * alpha
    return result


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


def _apply_rotvec_delta(
    current_orientations_wxyz: np.ndarray, delta_rotvec: np.ndarray
) -> np.ndarray:
    """把旋转向量增量左乘到当前 wxyz 四元数上。"""

    from scipy.spatial.transform import Rotation

    current = _normalize_quaternions(current_orientations_wxyz)
    current_xyzw = current[:, [1, 2, 3, 0]]
    delta = Rotation.from_rotvec(np.asarray(delta_rotvec, dtype=float))
    composed = delta * Rotation.from_quat(current_xyzw)
    xyzw = composed.as_quat()
    return xyzw[:, [3, 0, 1, 2]]
