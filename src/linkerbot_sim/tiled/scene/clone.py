"""GridCloner 生命周期与 PhysX replication 兼容性处理。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from linkerbot_sim.assets.root_pose import mjcf_fixed_root_joint_paths_without_body0
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.scene.paths import env_origins
from linkerbot_sim.tiled.scene.types import ImportedTiledRobot
from linkerbot_sim.tiled.scene.utils import _print_status


def _clone_config_compatible_with_robots(
    *,
    stage: object,
    config: TiledEnvConfig,
    robots: Mapping[str, ImportedTiledRobot],
    status_prefix: str | None,
) -> TiledEnvConfig:
    """在 MJCF world-fixed root joint 存在时关闭不可靠的 physics replication。"""

    if not config.clone.replicate_physics:
        return config
    blocker_paths = _mjcf_world_fixed_root_joint_paths(stage=stage, robots=robots)
    if not blocker_paths:
        return config
    _print_status(
        status_prefix,
        "PHYSICS_REPLICATION_DISABLED "
        "reason=mjcf_fixed_root_joint_without_body0 "
        f"joints={list(blocker_paths)}",
    )
    return replace(
        config,
        clone=replace(config.clone, replicate_physics=False),
    )


def _mjcf_world_fixed_root_joint_paths(
    *,
    stage: object,
    robots: Mapping[str, ImportedTiledRobot],
) -> tuple[str, ...]:
    """收集会阻止 replication 正确更新 clone 位姿的 MJCF root joints。"""

    result: list[str] = []
    for robot in robots.values():
        if robot.asset_type != "mjcf":
            continue
        result.extend(
            mjcf_fixed_root_joint_paths_without_body0(
                stage, robot.imported_root_paths[0]
            )
        )
    return tuple(result)


def _clone_envs(
    *,
    stage: object,
    config: TiledEnvConfig,
    env_roots: Sequence[str],
) -> np.ndarray:
    """用 GridCloner 克隆 env_0，并返回每个 env root 的实际 world origin。"""

    from isaacsim.core.cloner import GridCloner

    cloner = GridCloner(
        spacing=float(config.spacing),
        num_per_row=(
            -1 if config.num_per_row is None else int(config.effective_num_per_row)
        ),
        stage=stage,
    )
    desired_origins = env_origins(config)
    position_offsets = desired_origins - _grid_cloner_default_positions(config)
    positions = cloner.clone(
        source_prim_path=env_roots[0],
        prim_paths=list(env_roots),
        position_offsets=position_offsets,
        replicate_physics=bool(config.clone.replicate_physics),
        base_env_path=config.base_env_path,
        root_path=_physics_replication_root_path(config),
        copy_from_source=bool(config.clone.copy_from_source),
        enable_env_ids=bool(config.clone.enable_env_ids),
    )
    return np.asarray(positions, dtype=float).reshape(config.num_envs, 3)


def _physics_replication_root_path(config: TiledEnvConfig) -> str:
    """返回 PhysX replicator 使用的 ``base/env_prefix_`` clone root 前缀。"""

    return f"{config.base_env_path}/{config.env_prefix}_"


def _grid_cloner_default_positions(config: TiledEnvConfig) -> np.ndarray:
    """复现 GridCloner 默认网格，用于对齐项目 row-major env origin。"""

    num_envs = int(config.num_envs)
    num_per_row = (
        int(np.sqrt(num_envs))
        if config.num_per_row is None
        else int(config.effective_num_per_row)
    )
    num_per_row = max(1, num_per_row)
    num_rows = int(np.ceil(num_envs / num_per_row))
    num_cols = int(np.ceil(num_envs / num_rows))
    row_offset = 0.5 * float(config.spacing) * (num_rows - 1)
    col_offset = 0.5 * float(config.spacing) * (num_cols - 1)
    positions = np.zeros((num_envs, 3), dtype=float)
    for env_id in range(num_envs):
        row = env_id // num_cols
        col = env_id % num_cols
        positions[env_id, 0] = row_offset - row * float(config.spacing)
        positions[env_id, 1] = col * float(config.spacing) - col_offset
    return positions
