"""tiled env 的 USD path 和 env origin 辅助函数。

当前项目的 robot/object profile 仍然写着单场景路径，例如 ``/World/Robot`` 或
``/World/TBlock``。tiled runtime 不能直接改这些 profile，否则会破坏旧单臂/双臂
入口。本模块提供运行时 path rewrite：把单场景 path 映射到某个 env root 下。

注意:
    这些 helper 只处理字符串和 numpy 数组，不触碰 USD stage。真正创建 prim、clone env
    和过滤碰撞的逻辑应放在后续 scene builder 中。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linkerbot_sim.tiled.config import TiledEnvConfig


def env_root_path(config: TiledEnvConfig, env_id: int) -> str:
    """返回单个 env 的 USD root path。

    例如 ``base_env_path=/World/envs``、``env_prefix=env``、``env_id=3`` 时，
    返回 ``/World/envs/env_3``。
    """

    env_id = int(env_id)
    if env_id < 0 or env_id >= config.num_envs:
        raise ValueError(f"env_id out of range: {env_id}")
    return f"{config.base_env_path.rstrip('/')}/{config.env_prefix}_{env_id}"


def env_root_paths(config: TiledEnvConfig) -> tuple[str, ...]:
    """按 env id 顺序返回所有 env root paths。"""

    return tuple(env_root_path(config, env_id) for env_id in range(config.num_envs))


def make_env_local_prim_path(env_root: str, original_prim_path: str) -> str:
    """把单 env profile 里的 prim path 改写到指定 env root 下。

    示例:
        ``/World/Robot`` -> ``/World/envs/env_0/Robot``
        ``/World/Foo/Bar`` -> ``/World/envs/env_0/Foo/Bar``

    规则:
        * 输入必须是 USD 绝对路径。
        * 如果路径已经位于 ``env_root`` 下，直接返回，避免重复嵌套。
        * ``/World`` 本身不能被 namespace，因为它代表整个 stage 的世界根。
        * 非 ``/World/...`` 的绝对路径会去掉开头 ``/`` 后挂到 env root 下，便于未来
          兼容其它顶层命名空间。
    """

    env_root = _normalize_absolute_path(env_root, label="env_root")
    original = _normalize_absolute_path(original_prim_path, label="original_prim_path")
    if original == "/World":
        raise ValueError("Cannot namespace /World itself as an env-local prim")
    if original == env_root or original.startswith(env_root + "/"):
        return original
    if original.startswith("/World/"):
        suffix = original[len("/World/") :]
    else:
        suffix = original.lstrip("/")
    if not suffix:
        raise ValueError("original_prim_path must contain a prim suffix")
    return f"{env_root}/{suffix}"


def env_origins(config: TiledEnvConfig) -> np.ndarray:
    """返回每个 env 的世界坐标 origin，shape 为 ``(N, 3)``。

    origin 只用于 env root 的世界位移和 world/env-local 坐标转换。构建 ``env_0``
    内部机器人/物体时不要把 origin 加到它们的 root_pose 上，否则 clone 后会重复偏移。
    """

    origins = np.zeros((config.num_envs, 3), dtype=float)
    per_row = config.effective_num_per_row
    for env_id in range(config.num_envs):
        row = env_id // per_row
        col = env_id % per_row
        origins[env_id, 0] = float(col) * float(config.spacing)
        origins[env_id, 1] = float(row) * float(config.spacing)
    return origins


def env_local_suffix(env_root: str, prim_path: str) -> str:
    """返回 ``prim_path`` 相对 ``env_root`` 的后缀。

    典型用途是先在 ``env_0`` 导入资产并找到真实 articulation root，然后记录 root
    相对 env root 的 suffix；clone 后再用同一个 suffix 拼出其它 env 的 articulation path。
    """

    env_root = _normalize_absolute_path(env_root, label="env_root")
    prim_path = _normalize_absolute_path(prim_path, label="prim_path")
    prefix = env_root + "/"
    if not prim_path.startswith(prefix):
        raise ValueError(f"prim_path is not under env_root: {prim_path}")
    return prim_path[len(prefix) :]


def prim_paths_from_suffix(
    roots: Sequence[str],
    suffix: str,
) -> tuple[str, ...]:
    """用一组 env roots 和相对 suffix 拼出每个 env 的 prim path。"""

    if not suffix or suffix.startswith("/"):
        raise ValueError("suffix must be non-empty and relative")
    return tuple(
        f"{_normalize_absolute_path(root, label='env_root')}/{suffix}" for root in roots
    )


def _normalize_absolute_path(path: str, *, label: str) -> str:
    """规范化 USD 绝对路径，去掉末尾斜杠并拒绝空路径段。"""

    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"{label} must be an absolute USD path")
    normalized = path.rstrip("/") if path != "/" else path
    if "//" in normalized:
        raise ValueError(f"{label} cannot contain empty path components")
    return normalized
