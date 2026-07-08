"""Isaac/PhysX tiled scene builder package.

本子包是 tiled 配置进入真实 Isaac stage 的边界。它负责把单 env profile 改写到
``env_0`` 命名空间下，再用 Isaac ``GridCloner`` 克隆出所有 env，并创建 batched
``Articulation`` view。所有 Isaac/Omni import 都放在函数内部，保证普通单元测试
import 本包时不会启动 Kit。
"""

from linkerbot_sim.tiled.scene.builder import build_isaac_tiled_scene
from linkerbot_sim.tiled.scene.clone import (
    _clone_config_compatible_with_robots,
    _clone_envs,
    _filter_env_collisions,
    _first_physics_scene_path,
    _global_collision_paths,
    _grid_cloner_default_positions,
    _physics_replication_root_path,
)
from linkerbot_sim.tiled.scene.objects import (
    _apply_per_env_object_pose_overrides,
    _tiled_object_prim_paths,
    env_local_runtime_object_configs,
)
from linkerbot_sim.tiled.scene.robots import (
    _import_env_zero_robots,
    _import_robot_to_env_zero,
    env_local_robot_execution,
    tiled_robot_instances_from_env_config,
)
from linkerbot_sim.tiled.scene.root_pose import (
    _apply_per_env_robot_root_pose_overrides,
    _robot_world_root_pose,
)
from linkerbot_sim.tiled.scene.types import (
    ImportedTiledRobot,
    IsaacTiledScene,
    TiledArticulationView,
    TiledRobotInstance,
)
from linkerbot_sim.tiled.scene.views import (
    _command_joint_indices,
    _create_articulation_views,
    finalize_tiled_articulation_views,
)

__all__ = [
    "ImportedTiledRobot",
    "IsaacTiledScene",
    "TiledArticulationView",
    "TiledRobotInstance",
    "_apply_per_env_object_pose_overrides",
    "_apply_per_env_robot_root_pose_overrides",
    "_clone_config_compatible_with_robots",
    "_clone_envs",
    "_command_joint_indices",
    "_create_articulation_views",
    "_filter_env_collisions",
    "_first_physics_scene_path",
    "_global_collision_paths",
    "_grid_cloner_default_positions",
    "_import_env_zero_robots",
    "_import_robot_to_env_zero",
    "_physics_replication_root_path",
    "_robot_world_root_pose",
    "_tiled_object_prim_paths",
    "build_isaac_tiled_scene",
    "env_local_robot_execution",
    "env_local_runtime_object_configs",
    "finalize_tiled_articulation_views",
    "tiled_robot_instances_from_env_config",
]
