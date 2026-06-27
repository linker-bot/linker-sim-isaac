"""轨迹差分辅助函数。

本模块提供采样矩阵差分工具。所有时间单位为 s。

职责边界:
    * 对采样矩阵做简单后向差分，供日志和命令轨迹近似速度/加速度。
    * 不做滤波、限幅或动力学优化；需要严格轨迹约束时应由规划后端生成。
"""

from __future__ import annotations

import numpy as np


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
        dt = max(
            1.0e-12, float(sample_times_array[index] - sample_times_array[index - 1])
        )
        result[index] = (samples[index] - samples[index - 1]) / dt
    return result
