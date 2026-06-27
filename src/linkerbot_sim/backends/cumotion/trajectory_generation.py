"""共享的 ``CSpaceTrajectoryGenerator`` 封装。

graph search 和 specified path 都会先得到一条离散 C-space waypoint path。这个模块负责把这条
path 交给 cuMotion ``CSpaceTrajectoryGenerator``，按配置生成连续时间参数化 trajectory。

注意：``CSpaceTrajectoryGenerator`` 只做时间参数化和运动学限位处理，不会重新搜索避障路径。
输入 path 是否安全必须由前面的 pipeline 或后验碰撞检查保证。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from linkerbot_sim.backends.cumotion.motion_planner_config import (
    TrajectoryGenerationConfig,
)
from linkerbot_sim.backends.cumotion.motion_planner_utils import (
    apply_config_params,
)


_SUPPORTED_LIMIT_KEYS = {
    "position_min",
    "position_max",
    "velocity",
    "acceleration",
    "jerk",
}


def generate_cspace_trajectory(
    context,
    joint_path: np.ndarray | None,
    config: TrajectoryGenerationConfig,
    *,
    duration_s: float | None = None,
):
    """用 cuMotion 对 C-space waypoint path 做时间参数化。

    返回值是 cuMotion ``Trajectory``。graph search 和 specified path 的成功结果必须能生成
    时间参数化轨迹；path 为空或 waypoint 少于两个时立即报错，避免上层把离散 path 当成
    可执行轨迹兜底。

    该函数不再提供 ``enabled`` 开关。只要 pipeline 的中间产物是 joint path，成功返回就必须
    带 trajectory；如果后端无法生成 trajectory，应显式失败，让动作脚本层重新选择 pipeline 或
    调整配置，而不是默默退回项目侧线性插值。
    """

    if joint_path is None:
        raise ValueError("joint_path is required to generate trajectory")
    path = np.asarray(joint_path, dtype=float)
    if path.ndim == 1:
        path = path.reshape(1, -1)
    if path.ndim != 2:
        raise ValueError("joint_path must have shape (N, dof)")
    if path.shape[0] < 2:
        raise ValueError("joint_path must contain at least two waypoints")
    if path.shape[1] != context.expected_cspace_width:
        raise ValueError(
            f"joint_path expected {context.expected_cspace_width} columns, got {path.shape[1]}"
        )

    # 用 kinematics 创建 generator 可以继承机器人 C-space 维度和默认限位；随后再应用动作/配置
    # 中的显式覆盖。
    generator = context.cumotion.create_cspace_trajectory_generator(
        context.kinematics
    )
    configure_trajectory_generator(context.cumotion, generator, config)
    waypoints = [np.asarray(row, dtype=float).reshape(-1) for row in path]
    if config.mode == "time_optimal":
        # time_optimal 由 cuMotion 根据限位和 solver 参数自行决定轨迹时长。
        return generator.generate_trajectory(waypoints)
    if config.mode != "time_stamped":
        raise ValueError(
            "trajectory_generation.mode must be one of: time_optimal, time_stamped"
        )
    if duration_s is None:
        raise ValueError(
            "duration_s is required when trajectory_generation.mode='time_stamped'"
        )
    if duration_s <= 0.0:
        raise ValueError(
            "duration_s must be positive when trajectory_generation.mode='time_stamped'"
        )
    return generator.generate_time_stamped_trajectory(
        waypoints,
        times_for_joint_path(path, float(duration_s)),
        trajectory_interpolation_mode(context.cumotion, config.interpolation_mode),
    )


def configure_trajectory_generator(
    cumotion, generator, config: TrajectoryGenerationConfig
) -> None:
    """把项目配置写入 cuMotion trajectory generator。

    limit 键名使用项目规范名：``position_min`` / ``position_max`` / ``velocity`` /
    ``acceleration`` / ``jerk``。未知键立即报错，避免用户误写配置后悄悄退回默认限位。
    """

    limits = _normalized_limits(config.limits)
    unknown_limit_keys = set(limits) - _SUPPORTED_LIMIT_KEYS
    if unknown_limit_keys:
        raise ValueError(
            f"Unsupported trajectory_generation.limits key(s): {sorted(unknown_limit_keys)}"
        )
    if "position_min" in limits or "position_max" in limits:
        # position limit 上下界必须成对出现；只设置一侧会让后端约束语义不完整。
        if "position_min" not in limits or "position_max" not in limits:
            raise ValueError(
                "trajectory_generation.limits position_min and position_max must be set together"
            )
        generator.set_position_limits(limits["position_min"], limits["position_max"])
    if "velocity" in limits:
        generator.set_velocity_limits(limits["velocity"])
    if "acceleration" in limits:
        generator.set_acceleration_limits(limits["acceleration"])
    if "jerk" in limits:
        generator.set_jerk_limits(limits["jerk"])
    apply_config_params(
        generator,
        config.solver_params,
        cumotion.CSpaceTrajectoryGenerator.SolverParamValue,
        setter_name="set_solver_param",
    )


def trajectory_interpolation_mode(cumotion, interpolation_mode: str):
    """把项目字符串映射成 cuMotion ``InterpolationMode`` 枚举。"""

    enum = cumotion.CSpaceTrajectoryGenerator.InterpolationMode
    if interpolation_mode == "linear":
        return enum.LINEAR
    if interpolation_mode == "cubic_spline":
        return enum.CUBIC_SPLINE
    raise ValueError(
        f"Unsupported trajectory_generation.interpolation_mode={interpolation_mode!r}"
    )


def times_for_joint_path(joint_path: np.ndarray, duration_s: float) -> list[float]:
    """按 C-space 路径段长度给 waypoint 分配时间戳。

    对重复点或总路径长度接近 0 的情况退回均匀分配，避免除以极小数导致时间戳不稳定。
    """

    if duration_s < 0:
        raise ValueError("duration_s cannot be negative")
    path = np.asarray(joint_path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 2:
        return [0.0]
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1.0e-12:
        return list(np.linspace(0.0, float(duration_s), path.shape[0]))
    return [float(value) for value in float(duration_s) * cumulative / total]


def _normalized_limits(limits: Mapping[str, object]) -> dict[str, np.ndarray]:
    """把 limit mapping 统一转成一维 float 数组。"""

    return {
        str(key): np.asarray(value, dtype=float).reshape(-1)
        for key, value in limits.items()
    }
