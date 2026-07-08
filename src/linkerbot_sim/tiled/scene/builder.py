"""Top-level Isaac/PhysX tiled scene construction flow."""

from __future__ import annotations

from collections.abc import Mapping

from linkerbot_sim.app.runtime.objects import add_runtime_objects
from linkerbot_sim.configs.profiles import load_default_controller_profiles
from linkerbot_sim.controllers.config import ControllerProfiles
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.paths import env_root_paths
from linkerbot_sim.tiled.scene.clone import (
    _clone_config_compatible_with_robots,
    _clone_envs,
    _filter_env_collisions,
)
from linkerbot_sim.tiled.scene.objects import (
    _apply_per_env_object_pose_overrides,
    _tiled_object_prim_paths,
    env_local_runtime_object_configs,
)
from linkerbot_sim.tiled.scene.robots import _import_env_zero_robots
from linkerbot_sim.tiled.scene.root_pose import _apply_per_env_robot_root_pose_overrides
from linkerbot_sim.tiled.scene.types import IsaacTiledScene
from linkerbot_sim.tiled.scene.utils import _print_status
from linkerbot_sim.tiled.scene.views import _create_articulation_views


def build_isaac_tiled_scene(
    *,
    world: object,
    stage: object,
    env_config: Mapping[str, object],
    tiled_config: TiledEnvConfig,
    controller_profiles: ControllerProfiles | None = None,
    status_prefix: str | None = None,
) -> IsaacTiledScene:
    """在当前 Isaac stage 中构建 tiled scene。

    调用方必须已经创建 ``SimulationApp`` 和 ``World``。本函数只做 reset 前工作：
    导入 env_0、clone env roots、配置 collision filtering、创建 batched articulation
    views 并加入 ``world.scene``。调用方随后执行 ``world.reset()`` 初始化 PhysX view。
    """

    if tiled_config.num_envs < 1:
        raise ValueError("tiled_config.num_envs must be positive")
    controller_profiles = controller_profiles or load_default_controller_profiles()
    roots = env_root_paths(tiled_config)
    env_zero = roots[0]
    object_configs = env_local_runtime_object_configs(env_config, env_root=env_zero)

    _print_status(
        status_prefix,
        f"BUILD_BEGIN num_envs={tiled_config.num_envs} env_zero={env_zero}",
    )
    _define_env_zero(stage, env_zero)
    _print_status(status_prefix, "ENV_ZERO_DEFINED")
    object_handles = add_runtime_objects(
        stage,
        object_configs,
        status_prefix=status_prefix,
    )
    _print_status(status_prefix, f"OBJECTS_IMPORTED count={len(object_handles)}")
    robots = _import_env_zero_robots(
        stage=stage,
        env_config=env_config,
        tiled_config=tiled_config,
        controller_profiles=controller_profiles,
        env_roots=roots,
        status_prefix=status_prefix,
    )
    _print_status(status_prefix, f"ROBOTS_IMPORTED count={len(robots)}")
    effective_tiled_config = _clone_config_compatible_with_robots(
        stage=stage,
        config=tiled_config,
        robots=robots,
        status_prefix=status_prefix,
    )
    clone_positions = _clone_envs(
        stage=stage, config=effective_tiled_config, env_roots=roots
    )
    _print_status(status_prefix, f"CLONED positions={clone_positions.tolist()}")
    robot_root_pose_overrides_applied = _apply_per_env_robot_root_pose_overrides(
        stage=stage,
        robots=robots,
        env_origins=clone_positions,
        status_prefix=status_prefix,
    )
    _print_status(
        status_prefix,
        f"ROBOT_ROOT_POSE_OVERRIDES count={robot_root_pose_overrides_applied}",
    )
    object_prim_paths = _tiled_object_prim_paths(
        object_configs=object_configs,
        env_zero=env_zero,
        env_roots=roots,
    )
    object_pose_overrides_applied = _apply_per_env_object_pose_overrides(
        stage=stage,
        config=effective_tiled_config,
        object_prim_paths=object_prim_paths,
        status_prefix=status_prefix,
    )
    _print_status(
        status_prefix,
        f"OBJECT_POSE_OVERRIDES count={object_pose_overrides_applied}",
    )
    collision_applied = _filter_env_collisions(
        stage=stage,
        config=effective_tiled_config,
        env_roots=roots,
    )
    _print_status(status_prefix, f"COLLISION_FILTER applied={collision_applied}")
    articulation_views = _create_articulation_views(
        world=world,
        robots=robots,
    )
    _print_status(status_prefix, f"ARTICULATION_VIEWS count={len(articulation_views)}")
    return IsaacTiledScene(
        config=effective_tiled_config,
        env_root_paths=roots,
        env_origins=clone_positions.copy(),
        clone_positions=clone_positions,
        robots=robots,
        articulation_views=articulation_views,
        object_handles=object_handles,
        object_prim_paths=object_prim_paths,
        robot_root_pose_overrides_applied=robot_root_pose_overrides_applied,
        object_pose_overrides_applied=object_pose_overrides_applied,
        collision_filtering_applied=collision_applied,
    )


def _define_env_zero(stage: object, env_zero: str) -> None:
    """创建 env_0 root Xform，供后续导入资产和 GridCloner 使用。"""

    from pxr import Sdf, UsdGeom

    parent = Sdf.Path(env_zero).GetParentPath()
    if str(parent) != "/" and not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Scope.Define(stage, parent)
    UsdGeom.Xform.Define(stage, env_zero)
