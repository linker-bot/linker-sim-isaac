"""轨迹数据类型。

项目内部的关节轨迹对齐 cuMotion ``Trajectory`` 的表达：一条轨迹由时间域和
关节空间 position / velocity / acceleration / jerk 采样矩阵组成。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        times: 采样时间，shape ``(N,)``。
        positions: 关节位置采样矩阵，shape ``(N, dof)``。
        joint_names: 关节名元组，定义每个点中关节数组的顺序。
    输出:
        可按矩阵读取和按时间求值的轨迹对象。
    """

    def __init__(
        self,
        *,
        times: np.ndarray,
        positions: np.ndarray,
        joint_names: tuple[str, ...],
        velocities: np.ndarray | None = None,
        accelerations: np.ndarray | None = None,
        jerks: np.ndarray | None = None,
        phases: tuple[str, ...] | None = None,
    ):
        """创建关节轨迹并校验形状。

        参数:
            times/positions/velocities/accelerations/jerks:
                cuMotion 风格的采样矩阵，shape 分别为 ``(N,)`` 和 ``(N, dof)``。
            joint_names: 关节名元组，定义每列对应的关节。
        返回:
            无返回值；非法空轨迹会抛出 ``ValueError``。
        """

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
            times=times,
            positions=positions,
            joint_names=joint_names,
            velocities=velocities,
            accelerations=accelerations,
            jerks=jerks,
            phases=phases,
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

    def __len__(self) -> int:
        """返回轨迹点数量。

        参数:
            无。
        返回:
            轨迹点数量。
        """

        return int(self.times.size)


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
