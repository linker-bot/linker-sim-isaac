"""轨迹数据类型。

项目内部的关节轨迹对齐 cuRobo ``Trajectory`` 的表达：一条轨迹由时间域和
关节空间 position / velocity / acceleration / jerk 采样矩阵组成；effort 控制场景可额外携带
joint effort 采样矩阵。

矩阵行是时间采样点，列是 ``joint_names`` 定义的关节顺序；本类不关心这些列是否对应完整
articulation DOF 或控制器命令子空间。时间单位为秒，关节单位为弧度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryEval:
    """某个时间点的轨迹求值结果。

    四个数组 shape 相同，均按 ``JointTrajectory.joint_names`` 顺序排列。
    """

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    effort: np.ndarray


class JointTrajectory:
    """矩阵存储的关节轨迹容器。

    输入:
        times: 采样时间，shape ``(N,)``。
        positions: 关节位置采样矩阵，shape ``(N, dof)``。
        joint_names: 关节名元组，定义每个点中关节数组的顺序。
    输出:
        可按矩阵读取和按时间求值的轨迹对象。
    错误边界:
        构造时只校验矩阵维度、采样数和关节名数量；关节名是否存在于机器人由调用方负责。
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
        efforts: np.ndarray | None = None,
        phases: tuple[str, ...] | None = None,
    ):
        """创建关节轨迹并校验形状。

        参数:
            times/positions/velocities/accelerations/jerks:
                cuRobo 风格的采样矩阵，shape 分别为 ``(N,)`` 和 ``(N, dof)``。
            joint_names: 关节名元组，定义每列对应的关节。
        返回:
            无返回值；非法空轨迹会抛出 ``ValueError``。
        """

        # 先冻结关节名顺序，再校验所有矩阵列数。后续插值和执行层都只按列索引工作，
        # 因此构造阶段必须尽早发现名称数量与数据列数不一致的问题。
        self.joint_names = tuple(joint_names)
        self.times = np.asarray(times, dtype=float).reshape(-1)
        self.positions = np.asarray(positions, dtype=float)
        if self.positions.ndim != 2:
            raise ValueError("positions must have shape (N, dof)")
        if self.times.size != self.positions.shape[0]:
            raise ValueError("times length must match positions rows")
        if len(self.joint_names) != self.positions.shape[1]:
            raise ValueError(
                f"joint_names expected {self.positions.shape[1]} names, got {len(self.joint_names)}"
            )
        if self.times.size == 0:
            raise ValueError("Trajectory must contain at least one sample")
        if not np.all(np.isfinite(self.times)):
            raise ValueError("times must contain finite values")
        if self.times.size > 1 and np.any(np.diff(self.times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if not np.all(np.isfinite(self.positions)):
            raise ValueError("positions must contain finite values")
        self.times = self.times.copy()
        self.positions = self.positions.copy()
        # 速度/加速度/jerk 对执行器并非总是必需；缺省填零可保持对象接口完整，
        # 让日志和控制器无需为“没有速度曲线”的简单轨迹写特殊分支。
        self.velocities = _matrix_or_zeros(
            velocities, self.positions.shape, "velocities"
        )
        self.accelerations = _matrix_or_zeros(
            accelerations, self.positions.shape, "accelerations"
        )
        self.jerks = _matrix_or_zeros(jerks, self.positions.shape, "jerks")
        self.efforts = _matrix_or_zeros(efforts, self.positions.shape, "efforts")
        self.phases = (
            phases
            if phases is not None
            else tuple("trajectory" for _ in range(self.times.size))
        )
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
        efforts: np.ndarray | None = None,
        phases: tuple[str, ...] | None = None,
    ) -> "JointTrajectory":
        """从 cuRobo 风格采样矩阵创建轨迹。

        这是 ``__init__`` 的语义化别名，便于调用方表达“这些矩阵来自外部采样结果”。
        形状校验、缺省导数填零和 phase 校验仍由构造函数完成。
        """

        return cls(
            times=times,
            positions=positions,
            joint_names=joint_names,
            velocities=velocities,
            accelerations=accelerations,
            jerks=jerks,
            efforts=efforts,
            phases=phases,
        )

    def domain(self) -> tuple[float, float]:
        """返回轨迹时间域 ``(lower, upper)``。

        时间域直接取首尾采样点，不强制要求从 0 开始。
        """

        return float(self.times[0]), float(self.times[-1])

    def eval(self, time_s: float) -> np.ndarray:
        """返回指定时间的关节位置。

        超出时间域的查询会被钳制到首/末样本；需要导数时使用 ``eval_all``。
        """

        return self.eval_all(time_s).position

    def eval_all(self, time_s: float) -> TrajectoryEval:
        """返回指定时间的位置、速度、加速度和 jerk。

        各列独立线性插值，返回数组顺序与 ``joint_names`` 一致。effort 也会随同插值返回，
        即使调用方只用 position/velocity 播放轨迹。
        """

        # 对时间做钳制而不是抛错，便于执行循环在浮点舍入导致略微越界时仍返回端点。
        # 这与多数轨迹播放器“超出域保持首/末样本”的行为一致。
        query_time = float(time_s)
        if not np.isfinite(query_time):
            raise ValueError("time_s must be finite")
        t = float(np.clip(query_time, self.times[0], self.times[-1]))
        return TrajectoryEval(
            position=_interp_rows(self.times, self.positions, t),
            velocity=_interp_rows(self.times, self.velocities, t),
            acceleration=_interp_rows(self.times, self.accelerations, t),
            jerk=_interp_rows(self.times, self.jerks, t),
            effort=_interp_rows(self.times, self.efforts, t),
        )

    def __len__(self) -> int:
        """返回轨迹点数量。

        参数:
            无。
        返回:
            轨迹点数量。
        """

        return int(self.times.size)


def _matrix_or_zeros(
    values: np.ndarray | None, shape: tuple[int, int], label: str
) -> np.ndarray:
    """把可选采样矩阵规范化为指定 shape，缺省时填零。

    填零表示“没有提供该导数/effort 曲线”，让轨迹对象始终有完整字段。
    """

    if values is None:
        return np.zeros(shape, dtype=float)
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != shape:
        raise ValueError(
            f"{label} shape mismatch: expected {shape}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must contain finite values")
    return matrix.copy()


def _interp_rows(times: np.ndarray, values: np.ndarray, time_s: float) -> np.ndarray:
    """对每一列独立做一维线性插值。

    该 helper 只消费已经由 ``JointTrajectory`` 校验过的严格递增时间轴。
    """

    # 单点轨迹没有可插值区间，直接返回该点副本，避免 ``np.interp`` 在退化时间域上
    # 给出依赖实现细节的结果。
    if times.size == 1:
        return values[0].copy()
    return np.asarray(
        [np.interp(time_s, times, values[:, col]) for col in range(values.shape[1])],
        dtype=float,
    )
