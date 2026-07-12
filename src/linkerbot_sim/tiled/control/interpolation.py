"""固定 physics tick 数的 batch joint target 插值。"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.tiled.control.types import SUPPORTED_INTERPOLATIONS


def interpolate_joint_targets(
    *,
    start: np.ndarray,
    target: np.ndarray,
    steps: int,
    mode: str = "smoothstep",
) -> np.ndarray:
    """返回 shape ``(steps, num_envs, command_dim)`` 的固定 tick 轨迹。"""

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
        alpha = _interpolation_alpha(step_index=index, steps=steps, mode=mode)
        result[index] = start_array + (target_array - start_array) * alpha
    return result


def _interpolation_alpha(*, step_index: int, steps: int, mode: str) -> float:
    """返回一个 physics waypoint 的归一化进度。"""

    alpha = float(step_index + 1) / float(steps)
    if mode == "smoothstep":
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return alpha


__all__ = ["interpolate_joint_targets"]
