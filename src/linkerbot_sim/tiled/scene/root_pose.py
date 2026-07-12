"""cloned tiled scene 中 robot root pose override 的解析与写回。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from linkerbot_sim.assets.root_pose import (
    RootPoseConfig,
    apply_mjcf_fixed_root_joint_pose,
)
from linkerbot_sim.objects.physics import apply_root_pose_to_prim
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.scene.types import ImportedTiledRobot
from linkerbot_sim.tiled.scene.utils import _print_status


def _apply_per_env_robot_root_pose_overrides(
    *,
    stage: object,
    config: TiledEnvConfig,
    robots: Mapping[str, ImportedTiledRobot],
    env_origins: np.ndarray,
    status_prefix: str | None,
) -> int:
    """clone 后同步每个 env 的 robot Xform 和 fixed-base world anchor。

    robot prim 的 translate/rotate 使用 resolved env-local ``root_pose``，随各自的
    ``/World/envs/env_i`` root Xform 移动。MJCF fixed-base importer 额外生成的
    ``rootJoint_*`` 则是 world-anchor 语义；如果沿用 env_0 的 anchor，PhysX reset 时会把
    所有 tiled 机器人固定回同一个世界位置，因此还要用同一 local pose 加 env origin 写 anchor。
    """

    origins = np.asarray(env_origins, dtype=float).reshape(-1, 3)
    unknown_names = sorted(
        {
            name
            for per_env in config.per_env
            for name in per_env.robot_root_poses
            if name not in robots
        }
    )
    if unknown_names:
        raise ValueError(
            "tiled.per_env references unknown robot label(s): "
            + ", ".join(repr(name) for name in unknown_names)
        )

    applied = 0
    for robot_name, robot in robots.items():
        if len(robot.imported_root_paths) != origins.shape[0]:
            raise ValueError(
                f"robot {robot.name!r} root path count does not match env origins"
            )
        for env_id, imported_root_path in enumerate(robot.imported_root_paths):
            local_pose = config.robot_root_pose_for_env(
                env_id,
                robot_name,
                robot.execution.root_pose,
            )
            apply_root_pose_to_prim(stage, imported_root_path, local_pose)
            world_pose = _robot_world_root_pose(local_pose, origins[env_id])
            if robot.asset_type == "mjcf":
                apply_mjcf_fixed_root_joint_pose(stage, imported_root_path, world_pose)
            applied += 1
            _print_status(
                status_prefix,
                "ROBOT_ROOT_POSE "
                f"env_id={env_id} name={robot.name} "
                f"prim_path={imported_root_path} "
                f"local_xyz={list(local_pose.xyz)} "
                f"world_xyz={list(world_pose.xyz)} "
                f"rpy={list(world_pose.rpy)}",
            )
    return applied


def _robot_world_root_pose(
    local_pose: RootPoseConfig, env_origin: Sequence[float]
) -> RootPoseConfig:
    """把 env-local robot root pose 转成当前仅平移 env root 下的世界 pose。"""

    origin = np.asarray(env_origin, dtype=float).reshape(3)
    xyz = tuple(float(value) for value in (np.asarray(local_pose.xyz) + origin))
    return RootPoseConfig(xyz=xyz, rpy=local_pose.rpy)
