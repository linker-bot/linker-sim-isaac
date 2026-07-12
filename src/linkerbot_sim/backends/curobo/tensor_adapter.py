"""cuRobo tensor-like result 与 seed_config shape 适配。"""

from __future__ import annotations

import numpy as np


def tensor_like_to_numpy(value) -> np.ndarray:
    """把 torch tensor、numpy 或序列转换为 float ndarray。"""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def seed_config_from_state_or_seed(current_state: object | None, seed):
    """优先从 JointState.position 构造三维 cuRobo seed_config。"""

    if current_state is not None:
        position = getattr(current_state, "position", None)
        if position is not None:
            return as_curobo_seed_config(position)
    return None if seed is None else as_curobo_seed_config(seed)


def as_curobo_seed_config(value):
    """把 ``(D,)`` / ``(B,D)`` / ``(B,S,D)`` 规范为三维 seed。"""

    ndim = getattr(value, "ndim", None)
    if ndim is None:
        return as_curobo_seed_config(np.asarray(value, dtype=float))
    if ndim == 3:
        return _contiguous(value)
    if ndim == 2:
        if hasattr(value, "unsqueeze"):
            return _contiguous(value.unsqueeze(1))
        return np.ascontiguousarray(np.expand_dims(value, axis=1))
    if ndim == 1:
        if hasattr(value, "reshape"):
            return _contiguous(value.reshape(1, 1, -1))
        return np.ascontiguousarray(np.asarray(value, dtype=float).reshape(1, 1, -1))
    raise ValueError(f"cuRobo seed_config expects 1D/2D/3D seed, got ndim={ndim}")


def _contiguous(value):
    """保持 Torch tensor 类型调用 ``contiguous``，其它输入转为 contiguous ndarray。"""

    contiguous = getattr(value, "contiguous", None)
    if callable(contiguous):
        return contiguous()
    return np.ascontiguousarray(value)


__all__ = [
    "as_curobo_seed_config",
    "seed_config_from_state_or_seed",
    "tensor_like_to_numpy",
]
