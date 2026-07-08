"""GridCloner and collision filtering helpers for tiled Isaac scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from linkerbot_sim.assets.robot_loader import mjcf_fixed_root_joint_paths_without_body0
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.paths import env_origins
from linkerbot_sim.tiled.scene.types import ImportedTiledRobot
from linkerbot_sim.tiled.scene.utils import _print_status


def _clone_config_compatible_with_robots(
    *,
    stage: object,
    config: TiledEnvConfig,
    robots: Mapping[str, ImportedTiledRobot],
    status_prefix: str | None,
) -> TiledEnvConfig:
    """根据已导入机器人资产修正 clone 配置。

    当前 AR5+LinkerHand MJCF 是 fixed-base 资产，importer 会生成 ``body0`` 为空的
    ``rootJoint_*``。Isaac/PhysX replication 克隆这类 joint 时不会更新 clone 的
    local pose；如果继续启用 replication，env_1 机器人会在 reset 后偏离自己的 tile。
    因此这里在初始化阶段检测并关闭 replication，优先保证 tiled scene 的物理语义正确。
    """

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
    """收集会阻止 PhysX replication 正确更新 clone 位姿的 MJCF root joints。"""

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
    env_zero = env_roots[0]
    desired_origins = env_origins(config)
    position_offsets = desired_origins - _grid_cloner_default_positions(config)
    positions = cloner.clone(
        source_prim_path=env_zero,
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
    """返回 PhysX replication 需要的 clone root 前缀。

    ``GridCloner.generate_paths("/World/envs/env", N)`` 会生成
    ``/World/envs/env_0``，但它内部传给 PhysX replicator 的 root prefix 是
    ``/World/envs/env_``。这里直接复现该语义，避免 PhysX 把 clone 路径推导成
    ``/World/envs/env1``。
    """

    return f"{config.base_env_path}/{config.env_prefix}_"


def _filter_env_collisions(
    *,
    stage: object,
    config: TiledEnvConfig,
    env_roots: Sequence[str],
) -> bool:
    """按配置过滤不同 env root 之间的碰撞。

    Isaac ``GridCloner.filter_collisions`` 会为每个 env 创建一个
    ``PhysicsCollisionGroup``。PhysX 5.1 会对这种多 group authoring 反复打印
    ``Collisions are supported currently only in one collision group``。这里改用
    ``UsdPhysics.FilteredPairsAPI`` 直接在不同 env 的 articulation / rigid body /
    collider 之间建立 pair filter，避免 collision group，同时保留同一 env 内部和
    global prim 的正常碰撞。
    """

    if not config.clone.filter_collisions:
        return False

    participant_paths = [
        _collision_filter_participant_paths(stage=stage, env_root=env_root)
        for env_root in env_roots
    ]
    filtered_pair_count = _apply_env_pair_filters(stage, participant_paths)
    return filtered_pair_count > 0


def _collision_filter_participant_paths(*, stage: object, env_root: str) -> tuple[str, ...]:
    """收集一个 env 下可用于 ``FilteredPairsAPI`` 的 physics prim path。"""

    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(env_root)
    if not root.IsValid():
        return ()

    result: list[str] = []

    def visit(prim: object) -> None:
        if _is_collision_filter_participant(prim, UsdPhysics):
            result.append(str(prim.GetPath()))
            return
        for child in prim.GetChildren():
            visit(child)

    visit(root)
    return tuple(result)


def _is_collision_filter_participant(prim: object, usd_physics: object) -> bool:
    """判断 prim 是否能承载 ``FilteredPairsAPI``。"""

    return (
        prim.HasAPI(usd_physics.ArticulationRootAPI)
        or prim.HasAPI(usd_physics.RigidBodyAPI)
        or prim.HasAPI(usd_physics.CollisionAPI)
    )


def _apply_env_pair_filters(stage: object, participant_paths: Sequence[Sequence[str]]) -> int:
    """给不同 env 的 participant author pairwise collision filters。"""

    from pxr import Sdf, UsdPhysics

    authored = 0
    for source_env_index, source_paths in enumerate(participant_paths):
        for target_paths in participant_paths[source_env_index + 1 :]:
            for source_path in source_paths:
                source = stage.GetPrimAtPath(source_path)
                if not source.IsValid():
                    continue
                filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(source)
                rel = filtered_pairs.CreateFilteredPairsRel()
                for target_path in target_paths:
                    target = stage.GetPrimAtPath(target_path)
                    if not target.IsValid():
                        continue
                    rel.AddTarget(Sdf.Path(target_path))
                    authored += 1
    return authored


def _first_physics_scene_path(stage: object) -> str:
    """返回 stage 中第一个 UsdPhysics.Scene path。"""

    from pxr import UsdPhysics

    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return str(prim.GetPath())
    raise RuntimeError("No UsdPhysics.Scene found; create World before tiled scene")


def _global_collision_paths(stage: object, env_roots: Sequence[str]) -> list[str]:
    """收集需要和所有 env 保持碰撞的 stage-level prim。"""

    candidates = (
        "/World/defaultGroundPlane",
        "/World/GroundPlane",
        "/World/groundPlane",
        "/World/ground",
    )
    env_root_set = set(env_roots)
    return [
        path
        for path in candidates
        if path not in env_root_set and stage.GetPrimAtPath(path).IsValid()
    ]


def _grid_cloner_default_positions(config: TiledEnvConfig) -> np.ndarray:
    """复现 GridCloner 默认网格，用于把最终 env origin 对齐到项目 row-major 语义。"""

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
