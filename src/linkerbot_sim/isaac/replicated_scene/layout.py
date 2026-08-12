"""复制环境的 USD path 与确定性网格布局。"""

from __future__ import annotations

import math

import numpy as np


def environment_root_paths(settings: object, *, num_envs: int) -> tuple[str, ...]:
    """按 env id 顺序生成 source/destination 根路径。

    ``num_envs`` 是 composition root 已解析的最终数量，允许构造期显式 override；路径和
    prefix 来自根级环境事实，复制机制与 spacing 则由具体物理后端派生。
    """

    count = _positive_count(num_envs)
    base = _absolute_path(getattr(settings, "base_env_path"), label="base_env_path")
    prefix = str(getattr(settings, "env_prefix"))
    if not prefix or "/" in prefix:
        raise ValueError("env_prefix must be one non-empty USD path component")
    return tuple(f"{base}/{prefix}_{env_id}" for env_id in range(count))


def environment_origins(
    settings: object,
    *,
    num_envs: int,
    spacing_m: float,
) -> np.ndarray:
    """返回 row-major、近似方形网格的 world origins，形状为 ``(N, 3)``。

    返回值固定为 ``float32``，随后只上传一次到 Kaleidoscope 的 canonical CUDA buffer。
    计算发生在启动冷路径，不进入 observation/step 热路径。
    """

    count = _positive_count(num_envs)
    # spacing 是物理后端的复制机制事实：PhysX GridCloner 使用正间距，Newton
    # multi-world 使用零间距。环境配置只提供逻辑原点，不能反向选择后端策略。
    spacing = float(spacing_m)
    if not math.isfinite(spacing) or spacing < 0.0:
        raise ValueError("spacing_m must be finite and non-negative")
    origin = np.asarray(getattr(settings, "origin_xyz"), dtype=np.float32)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin_xyz must be a finite length-3 vector")
    per_row = max(1, int(math.ceil(math.sqrt(count))))
    env_ids = np.arange(count, dtype=np.int64)
    result = np.repeat(origin.reshape(1, 3), count, axis=0)
    # ``spacing == 0`` 是 separate-world Newton 的正式布局：所有 world 在各自坐标系
    # 使用同一原点。它们没有跨 world 接触，因此无需用大坐标换取物理隔离。
    result[:, 0] += (env_ids % per_row).astype(np.float32) * spacing
    result[:, 1] += (env_ids // per_row).astype(np.float32) * spacing
    return np.ascontiguousarray(result, dtype=np.float32)


def env_local_prim_path(env_root: str, original_path: str) -> str:
    """把单环境资产 path 映射到 source env namespace。"""

    root = _absolute_path(env_root, label="env_root")
    original = _absolute_path(original_path, label="original_path")
    if original == "/World":
        raise ValueError("/World itself cannot be namespaced into an environment")
    if original == root or original.startswith(root + "/"):
        return original
    suffix = (
        original[len("/World/") :] if original.startswith("/World/") else original[1:]
    )
    if not suffix:
        raise ValueError("original_path must contain a prim suffix")
    return f"{root}/{suffix}"


def relative_prim_suffix(env_root: str, prim_path: str) -> str:
    """取得已导入 prim 相对 source env root 的稳定 suffix。"""

    root = _absolute_path(env_root, label="env_root")
    path = _absolute_path(prim_path, label="prim_path")
    prefix = root + "/"
    if not path.startswith(prefix):
        raise ValueError(f"prim_path is not below environment root: {path}")
    return path[len(prefix) :]


def paths_from_suffix(roots: tuple[str, ...], suffix: str) -> tuple[str, ...]:
    """把 source env 内的一个 suffix 扩展到全部 clone。"""

    if not suffix or suffix.startswith("/") or "//" in suffix:
        raise ValueError("suffix must be a non-empty relative USD path")
    return tuple(f"{_absolute_path(root, label='env_root')}/{suffix}" for root in roots)


def _positive_count(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("num_envs must be a positive int")
    return value


def _absolute_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{label} must be an absolute USD path")
    result = value.rstrip("/") if value != "/" else value
    if "//" in result:
        raise ValueError(f"{label} cannot contain empty path components")
    return result


__all__ = [
    "env_local_prim_path",
    "environment_origins",
    "environment_root_paths",
    "paths_from_suffix",
    "relative_prim_suffix",
]
