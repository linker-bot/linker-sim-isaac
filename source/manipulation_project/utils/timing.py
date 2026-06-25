"""时间采样与轨迹差分辅助函数。

本模块提供仿真循环更新间隔计算、轨迹采样时间生成、线性位置插值和采样矩阵差分等工具。
所有时间单位为 s，频率单位为 Hz。

职责边界:
    * 把用户给定的频率/时长转换为离散采样点或 physics step 间隔。
    * 对采样矩阵做简单后向差分，供日志和命令轨迹近似速度/加速度。
    * 不做滤波、限幅或动力学优化；需要严格轨迹约束时应由规划后端生成。
"""

from __future__ import annotations

import numpy as np


def interval_steps(update_frequency_hz: float, physics_dt: float) -> int:
    """计算控制命令更新间隔对应的 physics step 数。

    参数:
        update_frequency_hz: 控制器或日志希望更新的频率，单位 Hz。
        physics_dt: 仿真物理步长，单位 s，例如 ``1 / physics_frequency``。
    返回:
        至少为 1 的整数 step 间隔；返回值为 ``N`` 时表示每隔 ``N`` 个 physics step 更新一次。
    """

    if update_frequency_hz <= 0:
        raise ValueError("update_frequency_hz must be positive")
    if physics_dt <= 0:
        raise ValueError("physics_dt must be positive")
    # round 让目标频率尽量贴近 physics step 的整数倍；至少返回 1，避免频率高于物理频率时
    # 得到 0 步间隔。
    return max(1, int(round(1.0 / (update_frequency_hz * physics_dt))))


def sample_times(duration_s: float, sample_hz: float) -> np.ndarray:
    """按轨迹时长和采样频率生成采样时间序列。

    返回数组始终包含 ``0.0`` 和 ``duration_s``。当 ``duration_s`` 不是采样周期的整数倍时，
    最后一个采样点会被裁剪到精确的结束时间，避免轨迹超过目标时长。
    """

    duration = float(duration_s)
    if duration < 0:
        raise ValueError("duration_s cannot be negative")
    sample_hz = float(sample_hz)
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    sample_dt = 1.0 / sample_hz
    if duration == 0:
        return np.asarray([0.0], dtype=float)
    # ceil 保证覆盖终点；每个采样时刻用 min 钳制到 duration，避免最后一点超过配置时长。
    num_steps = max(1, int(np.ceil(duration / sample_dt)))
    return np.asarray([min(duration, step * sample_dt) for step in range(num_steps + 1)], dtype=float)


def sample_linear_positions(start_position, target_position, times: np.ndarray) -> np.ndarray:
    """根据采样时间在线段起点和终点之间插值位置。

    ``times`` 的最后一个值被视为轨迹总时长。返回矩阵 shape 为 ``(N, 3)``，
    每一行对应一个采样时刻的三维位置；总时长小于等于 0 时直接返回终点。
    """

    start = np.asarray(start_position, dtype=float).reshape(3)
    target = np.asarray(target_position, dtype=float).reshape(3)
    sample_times_array = np.asarray(times, dtype=float).reshape(-1)
    if sample_times_array.size == 0:
        raise ValueError("times must contain at least one sample")
    # 用最后一个采样时间作为总时长，可支持非均匀 times；alpha 逐行广播到 xyz 三列。
    duration = float(sample_times_array[-1])
    if duration <= 0:
        return target.reshape(1, 3)
    alpha = (sample_times_array / duration).reshape(-1, 1)
    return start.reshape(1, 3) + alpha * (target - start).reshape(1, 3)


def differentiate_samples(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """对采样矩阵按时间做一阶后向差分。

    ``values`` 形状应为 ``(N, dof)``，``times`` 长度应为 ``N``。返回矩阵与 ``values``
    同形状；第 0 个采样点没有前一帧可差分，因此导数保持 0。
    """

    samples = np.asarray(values, dtype=float)
    sample_times_array = np.asarray(times, dtype=float).reshape(-1)
    if samples.ndim != 2:
        raise ValueError("values must have shape (N, dof)")
    if samples.shape[0] != sample_times_array.size:
        raise ValueError("times length must match values rows")
    if samples.shape[0] == 1:
        return np.zeros_like(samples)

    result = np.zeros_like(samples)
    for index in range(1, samples.shape[0]):
        # dt 用极小正数下限保护重复时间戳，避免除零；这会产生很大的导数，提示输入采样异常。
        dt = max(1.0e-12, float(sample_times_array[index] - sample_times_array[index - 1]))
        result[index] = (samples[index] - samples[index - 1]) / dt
    return result
