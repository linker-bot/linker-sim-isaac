"""轨迹数据类型。

项目内部的关节轨迹对齐 cuMotion ``Trajectory`` 的表达：一条轨迹由时间域和
关节空间 position / velocity / acceleration / jerk 采样矩阵组成。为了兼容
控制器和日志代码，``JointTrajectory`` 仍可迭代为 ``TrajectoryPoint``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryPoint:
    """一个离散轨迹采样点。

    输入字段:
        time_s: 采样点相对轨迹起点的时间，单位 s。
        joint_positions: 关节位置数组，单位 rad，顺序由所属轨迹的 ``joint_names`` 定义。
        joint_velocities: 可选关节速度数组，单位 rad/s，顺序同 ``joint_positions``。
        joint_accelerations: 可选关节加速度数组，单位 rad/s^2。
        joint_jerks: 可选关节 jerk 数组，单位 rad/s^3。
        tcp_position: 可选 TCP 世界或任务坐标位置，shape ``(3,)``，单位 m。
        tcp_orientation: 可选 TCP 姿态，通常为 wxyz 四元数。
        phase: 任务阶段名，用于日志标记。
    输出:
        dataclass 实例作为轨迹中的一个不可变点。
    """

    time_s: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray | None = None
    joint_accelerations: np.ndarray | None = None
    joint_jerks: np.ndarray | None = None
    tcp_position: np.ndarray | None = None
    tcp_orientation: np.ndarray | None = None
    phase: str = "trajectory"


@dataclass(frozen=True)
class TrajectoryEval:
    """某个时间点的轨迹求值结果。"""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray


class JointTrajectory:
    """矩阵存储的关节轨迹容器。

    输入:
        points: 至少包含一个 ``TrajectoryPoint`` 的列表，旧式构造方式。
        joint_names: 关节名元组，定义每个点中关节数组的顺序。
    输出:
        可迭代对象，迭代时逐个返回 ``TrajectoryPoint``。
    """

    def __init__(
        self,
        points: list[TrajectoryPoint] | None,
        joint_names: tuple[str, ...],
        *,
        times: np.ndarray | None = None,
        positions: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        accelerations: np.ndarray | None = None,
        jerks: np.ndarray | None = None,
        phases: tuple[str, ...] | None = None,
    ):
        """创建关节轨迹并校验形状。

        参数:
            points: 旧式轨迹点列表；如果提供 ``times``/``positions``，可为 ``None``。
            joint_names: 关节名元组，定义每个点中关节数组顺序。
            times/positions/velocities/accelerations/jerks:
                cuMotion 风格的采样矩阵，shape 分别为 ``(N,)`` 和 ``(N, dof)``。
        返回:
            无返回值；非法空轨迹会抛出 ``ValueError``。
        """

        if positions is None or times is None:
            if not points:
                raise ValueError("Trajectory must contain at least one point")
            times = np.asarray([point.time_s for point in points], dtype=float)
            positions = np.vstack([np.asarray(point.joint_positions, dtype=float).reshape(-1) for point in points])
            velocities = _stack_optional(points, "joint_velocities", positions.shape)
            accelerations = _stack_optional(points, "joint_accelerations", positions.shape)
            jerks = _stack_optional(points, "joint_jerks", positions.shape)
            phases = tuple(point.phase for point in points)
        self.joint_names = tuple(joint_names)
        self.times = np.asarray(times, dtype=float).reshape(-1)
        self.positions = np.asarray(positions, dtype=float)
        if self.positions.ndim != 2:
            raise ValueError("positions must have shape (N, dof)")
        if self.times.size != self.positions.shape[0]:
            raise ValueError("times length must match positions rows")
        if len(self.joint_names) != self.positions.shape[1]:
            raise ValueError(f"joint_names expected {self.positions.shape[1]} names, got {len(self.joint_names)}")
        if self.times.size == 0:
            raise ValueError("Trajectory must contain at least one sample")
        self.velocities = _matrix_or_zeros(velocities, self.positions.shape, "velocities")
        self.accelerations = _matrix_or_zeros(accelerations, self.positions.shape, "accelerations")
        self.jerks = _matrix_or_zeros(jerks, self.positions.shape, "jerks")
        self.phases = phases if phases is not None else tuple("trajectory" for _ in range(self.times.size))
        if len(self.phases) != self.times.size:
            raise ValueError("phases length must match trajectory samples")

    @classmethod
    def from_samples(
        cls,
        *,
        times: np.ndarray,
        positions: np.ndarray,
        joint_names: tuple[str, ...],
        velocities: np.ndarray | None = None,
        accelerations: np.ndarray | None = None,
        jerks: np.ndarray | None = None,
        phases: tuple[str, ...] | None = None,
    ) -> "JointTrajectory":
        """从 cuMotion 风格采样矩阵创建轨迹。"""

        return cls(
            None,
            joint_names,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            jerks=jerks,
            phases=phases,
        )

    @property
    def points(self) -> list[TrajectoryPoint]:
        """以旧式点列表形式返回轨迹。"""

        return [self.point(index) for index in range(len(self))]

    def point(self, index: int) -> TrajectoryPoint:
        """返回指定采样点。"""

        return TrajectoryPoint(
            time_s=float(self.times[index]),
            joint_positions=self.positions[index].copy(),
            joint_velocities=self.velocities[index].copy(),
            joint_accelerations=self.accelerations[index].copy(),
            joint_jerks=self.jerks[index].copy(),
            phase=self.phases[index],
        )

    def domain(self) -> tuple[float, float]:
        """返回轨迹时间域 ``(lower, upper)``。"""

        return float(self.times[0]), float(self.times[-1])

    def eval(self, time_s: float) -> np.ndarray:
        """返回指定时间的关节位置。"""

        return self.eval_all(time_s).position

    def eval_all(self, time_s: float) -> TrajectoryEval:
        """返回指定时间的位置、速度、加速度和 jerk。"""

        t = float(np.clip(time_s, self.times[0], self.times[-1]))
        return TrajectoryEval(
            position=_interp_rows(self.times, self.positions, t),
            velocity=_interp_rows(self.times, self.velocities, t),
            acceleration=_interp_rows(self.times, self.accelerations, t),
            jerk=_interp_rows(self.times, self.jerks, t),
        )

    def __iter__(self):
        """返回轨迹点迭代器。

        参数:
            无。
        返回:
            ``self.points`` 的迭代器。
        """

        for index in range(len(self)):
            yield self.point(index)

    def __len__(self) -> int:
        """返回轨迹点数量。

        参数:
            无。
        返回:
            轨迹点数量。
        """

        return int(self.times.size)


def _stack_optional(points: list[TrajectoryPoint], attr: str, shape: tuple[int, int]) -> np.ndarray | None:
    values = [getattr(point, attr) for point in points]
    if any(value is None for value in values):
        return None
    matrix = np.vstack([np.asarray(value, dtype=float).reshape(-1) for value in values])
    if matrix.shape != shape:
        raise ValueError(f"{attr} shape mismatch: expected {shape}, got {matrix.shape}")
    return matrix


def _matrix_or_zeros(values: np.ndarray | None, shape: tuple[int, int], label: str) -> np.ndarray:
    if values is None:
        return np.zeros(shape, dtype=float)
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != shape:
        raise ValueError(f"{label} shape mismatch: expected {shape}, got {matrix.shape}")
    return matrix


def _interp_rows(times: np.ndarray, values: np.ndarray, time_s: float) -> np.ndarray:
    if times.size == 1:
        return values[0].copy()
    return np.asarray([np.interp(time_s, times, values[:, col]) for col in range(values.shape[1])], dtype=float)
