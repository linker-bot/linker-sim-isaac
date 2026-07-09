"""tiled runtime 的纯数据状态结构。

本模块只保存 batched state 的 shape 约定和轻量校验，不读取 Isaac articulation、
RigidPrim 或 USD stage。真实 runtime 后续可以把 Isaac view 读出的数组包装成这些
dataclass，再交给 evaluator、logger 或下游策略。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TiledRobotJointState:
    """一个逻辑机器人/单侧 articulation view 的 batched 关节状态。

    所有数组第一维都是 env 维度，第二维是该 articulation 的 DOF 维度。``joint_names``
    必须与第二维长度一致，避免上层把不同机器人或不同关节顺序的数据混用。
    """

    joint_names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    measured_efforts: np.ndarray | None = None
    applied_efforts: np.ndarray | None = None

    def __post_init__(self) -> None:
        """校验 qpos/qvel/effort shape 是否一致。"""

        positions = np.asarray(self.positions, dtype=float)
        velocities = np.asarray(self.velocities, dtype=float)
        if positions.ndim != 2:
            raise ValueError("TiledRobotJointState.positions must be 2D")
        if velocities.shape != positions.shape:
            raise ValueError("TiledRobotJointState.velocities must match positions")
        if len(self.joint_names) != positions.shape[1]:
            raise ValueError("joint_names length must match joint dimension")
        measured = _optional_matching_array(
            self.measured_efforts, positions.shape, "measured_efforts"
        )
        applied = _optional_matching_array(
            self.applied_efforts, positions.shape, "applied_efforts"
        )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "measured_efforts", measured)
        object.__setattr__(self, "applied_efforts", applied)


@dataclass(frozen=True)
class TiledObjectState:
    """一个命名 runtime object 的 batched 位姿状态。

    同时保存 world pose 和 env-local position。world pose 是 Isaac/PhysX 真实坐标；
    env-local position 更适合任务逻辑、MPC 打分和跨 env 比较。
    """

    name: str
    positions_world: np.ndarray
    orientations_wxyz: np.ndarray
    positions_local: np.ndarray

    @classmethod
    def from_world(
        cls,
        *,
        name: str,
        positions_world: np.ndarray,
        orientations_wxyz: np.ndarray,
        env_origins: np.ndarray,
    ) -> "TiledObjectState":
        """从 world 坐标和 env origins 构造 object state。

        orientations 暂时不转换到 local frame；第一阶段 tiled env 只通过平移 origin 分隔，
        所以 local/world 姿态一致。若未来 env root 支持旋转，再扩展这里的语义。
        """

        positions = np.asarray(positions_world, dtype=float)
        origins = np.asarray(env_origins, dtype=float)
        return cls(
            name=name,
            positions_world=positions,
            orientations_wxyz=orientations_wxyz,
            positions_local=positions - origins,
        )

    def __post_init__(self) -> None:
        """校验 object pose 的 env 维度一致。"""

        positions_world = _shape_array(self.positions_world, (-1, 3), "positions_world")
        orientations = _shape_array(
            self.orientations_wxyz, (positions_world.shape[0], 4), "orientations_wxyz"
        )
        positions_local = _shape_array(
            self.positions_local, positions_world.shape, "positions_local"
        )
        if not self.name:
            raise ValueError("TiledObjectState.name cannot be empty")
        object.__setattr__(self, "positions_world", positions_world)
        object.__setattr__(self, "orientations_wxyz", orientations)
        object.__setattr__(self, "positions_local", positions_local)


@dataclass(frozen=True)
class TiledState:
    """tiled runtime 每次 step/reset 返回的完整 batched 快照。

    ``robots`` 和 ``objects`` 按逻辑名称索引，例如 ``left``、``right``、``TBlock``。
    ``info`` 只放运行时诊断，如 ``ik_success``、IK error、env_ids；不放 reward、
    success 或 task metric。
    """

    step: int
    time_s: float
    robots: Mapping[str, TiledRobotJointState]
    objects: Mapping[str, TiledObjectState] = field(default_factory=dict)
    info: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验时间轴字段非负。"""

        if int(self.step) < 0:
            raise ValueError("TiledState.step cannot be negative")
        if float(self.time_s) < 0.0:
            raise ValueError("TiledState.time_s cannot be negative")


def broadcast_rows(values: np.ndarray, num_envs: int, *, label: str) -> np.ndarray:
    """把单行数组广播到 ``num_envs`` 行。

    state clone/MPC rollout 经常需要把一个 env 的状态复制到所有 env，再执行不同动作候选。
    这个 helper 只做 shape 处理，不改变调用方对字段语义的定义。
    """

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 1D or 2D array")
    if array.shape[0] == 1 and int(num_envs) != 1:
        return np.repeat(array, int(num_envs), axis=0)
    if array.shape[0] != int(num_envs):
        raise ValueError(f"{label} first dimension must be 1 or num_envs")
    return array.astype(float, copy=True)


def _optional_matching_array(
    values: np.ndarray | None, shape: tuple[int, ...], label: str
) -> np.ndarray | None:
    """校验可选数组是否与目标 shape 完全一致。"""

    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return array


def _shape_array(values: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    """校验数组 shape；shape 中 ``-1`` 表示该维度任意长度。"""

    array = np.asarray(values, dtype=float)
    if len(shape) != array.ndim:
        raise ValueError(f"{label} must have {len(shape)} dimensions")
    for actual, expected in zip(array.shape, shape, strict=True):
        if expected != -1 and actual != expected:
            raise ValueError(f"{label} must have shape {shape}")
    return array
