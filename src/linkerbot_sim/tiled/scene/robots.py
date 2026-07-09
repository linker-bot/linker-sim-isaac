"""Robot import helpers for tiled Isaac scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from linkerbot_sim.assets.robot_loader import (
    RobotExecutionConfig,
    apply_root_pose,
    dual_robot_scene_instances_from_env_config,
    import_robot_asset,
    robot_scene_instance_from_env_config,
)
from linkerbot_sim.assets.solver_overrides import (
    apply_solver_iteration_overrides,
    merge_solver_configs,
    scene_solver_settings,
)
from linkerbot_sim.assets.usd_overrides import (
    apply_robot_gravity_policy,
    apply_robot_usd_overrides,
)
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.controllers.config import ControllerProfiles, physx_override_configs
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.paths import (
    env_local_suffix,
    make_env_local_prim_path,
    prim_paths_from_suffix,
)
from linkerbot_sim.tiled.scene.types import ImportedTiledRobot, TiledRobotInstance
from linkerbot_sim.tiled.scene.utils import _print_status


def tiled_robot_instances_from_env_config(
    env_config: Mapping[str, object],
) -> tuple[TiledRobotInstance, ...]:
    """从 env profile 提取 tiled builder 支持的机器人实例。

    当前支持两类现有结构：``robots.single`` 和 ``robots.dual.left/right``。返回的
    ``name`` 是 tiled runtime 中的稳定逻辑名，双臂场景中分别为 ``left``/``right``。
    """

    robots = env_config.get("robots")
    if not isinstance(robots, Mapping):
        raise ValueError("Environment config must contain top-level robots mapping")
    if "single" in robots:
        instance = robot_scene_instance_from_env_config(env_config, "single")
        return (
            TiledRobotInstance(
                name="single",
                profile_name=instance.robot_profile,
                scene_instance=instance,
            ),
        )
    if "dual" in robots:
        instances = dual_robot_scene_instances_from_env_config(env_config)
        return tuple(
            TiledRobotInstance(
                name=side,
                profile_name=instance.robot_profile,
                scene_instance=instance,
            )
            for side, instance in instances.items()
        )
    raise ValueError("tiled scene builder requires robots.single or robots.dual")


def env_local_robot_execution(
    *,
    robot_profile_config: Mapping[str, object],
    scene_instance: object,
    env_root: str,
    robot_name: str,
) -> RobotExecutionConfig:
    """把一个 robot profile 改写为 env-local ``RobotExecutionConfig``。

    robot YAML 中的 ``prim_path`` 仍保持单臂/双臂 runtime 使用的 ``/World/...``；
    tiled builder 只在运行时替换成 ``/World/envs/env_0/...``，避免污染非 tiled 入口。
    """

    execution = RobotExecutionConfig.from_mapping(
        robot_profile_config,
        root_pose=scene_instance.root_pose,
    )
    robot = replace(
        execution.robot,
        name=robot_name,
        prim_path=make_env_local_prim_path(env_root, execution.robot.prim_path),
    )
    return replace(execution, robot=robot)


def _import_env_zero_robots(
    *,
    stage: object,
    env_config: Mapping[str, object],
    tiled_config: TiledEnvConfig,
    controller_profiles: ControllerProfiles,
    env_roots: Sequence[str],
    status_prefix: str | None,
) -> dict[str, ImportedTiledRobot]:
    """导入 env_0 下的所有机器人，并记录 clone 后每个 env 的路径。"""

    env_zero = env_roots[0]
    result: dict[str, ImportedTiledRobot] = {}
    for instance in tiled_robot_instances_from_env_config(env_config):
        robot_profile = load_profile_yaml("robot", instance.profile_name)
        execution = env_local_robot_execution(
            robot_profile_config=robot_profile,
            scene_instance=instance.scene_instance,
            env_root=env_zero,
            robot_name=f"tiled_{instance.name}",
        )
        imported = _import_robot_to_env_zero(
            stage=stage,
            env_config=env_config,
            robot_execution=execution,
            controller_profiles=controller_profiles,
        )
        articulation_suffix = env_local_suffix(env_zero, imported["articulation_path"])
        imported_root_suffix = env_local_suffix(
            env_zero, imported["imported_root_path"]
        )
        tiled_robot = ImportedTiledRobot(
            name=instance.name,
            profile_name=instance.profile_name,
            execution=execution,
            articulation_root_suffix=articulation_suffix,
            imported_root_suffix=imported_root_suffix,
            articulation_paths=prim_paths_from_suffix(env_roots, articulation_suffix),
            imported_root_paths=prim_paths_from_suffix(env_roots, imported_root_suffix),
            asset_path=imported["asset_path"],
            asset_type=execution.robot.asset_type,
            controlled_joints=tuple(execution.controlled_joints),
            gravity_policy=imported["gravity_policy"],
            gravity_counts=imported["gravity_counts"],
            solver_counts=imported["solver_counts"],
        )
        result[instance.name] = tiled_robot
        _print_status(
            status_prefix,
            "ROBOT "
            f"name={instance.name} profile={instance.profile_name} "
            f"articulation_suffix={articulation_suffix} "
            f"asset={tiled_robot.asset_path} gravity={tiled_robot.gravity_counts} "
            f"solver={tiled_robot.solver_counts}",
        )
    return result


def _import_robot_to_env_zero(
    *,
    stage: object,
    env_config: Mapping[str, object],
    robot_execution: RobotExecutionConfig,
    controller_profiles: ControllerProfiles,
) -> dict[str, object]:
    """执行单个机器人 reset 前的 USD/PhysX 导入和覆盖。"""

    articulation_path, asset_path, imported_root_path = import_robot_asset(
        robot_execution.robot
    )
    apply_root_pose(stage, imported_root_path, robot_execution.root_pose)
    physx_configs = robot_execution.robot.physx_overrides.apply_to_configs(
        physx_override_configs(controller_profiles)
    )
    apply_robot_usd_overrides(
        imported_root_path,
        physx_configs,
        driven_joint_names=tuple(robot_execution.controlled_joints),
        mjcf_path=asset_path if robot_execution.robot.asset_type == "mjcf" else None,
    )
    solver_config = merge_solver_configs(
        scene_solver_settings(env_config),
        robot_execution.robot.solver_iterations,
    )
    solver_counts = (
        apply_solver_iteration_overrides(stage, articulation_path, solver_config)
        if solver_config is not None
        else {"configured": 0}
    )
    gravity_policy = robot_execution.robot.gravity_policy
    gravity_counts = apply_robot_gravity_policy(imported_root_path, gravity_policy)
    return {
        "articulation_path": articulation_path,
        "imported_root_path": imported_root_path,
        "asset_path": asset_path,
        "gravity_policy": gravity_policy,
        "gravity_counts": gravity_counts,
        "solver_counts": solver_counts,
    }
