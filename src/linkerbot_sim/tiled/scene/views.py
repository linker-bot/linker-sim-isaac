"""tiled Isaac scene 的 batched articulation view 创建与 joint index 绑定。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from linkerbot_sim.robots.joint_groups import resolve_joint_indices
from linkerbot_sim.robots.mimic.runtime import resolve_mimic_follower_controls
from linkerbot_sim.tiled.scene.types import (
    ImportedTiledRobot,
    IsaacTiledScene,
    TiledArticulationView,
)


def _create_articulation_views(
    *,
    world: object,
    robots: Mapping[str, ImportedTiledRobot],
) -> dict[str, TiledArticulationView]:
    """创建 batched Articulation view 并加入 world.scene。"""

    from isaacsim.core.prims import Articulation

    views: dict[str, TiledArticulationView] = {}
    for name, robot in robots.items():
        view = world.scene.add(
            Articulation(
                prim_paths_expr=list(robot.articulation_paths),
                name=f"tiled_{name}_view",
                reset_xform_properties=False,
            )
        )
        views[name] = TiledArticulationView(
            name=name,
            view=view,
            articulation_paths=robot.articulation_paths,
            command_joint_names=(),
            command_joint_indices=np.asarray([], dtype=int),
            runtime_mimic_controls=(),
            robot_id=robot.robot_id,
            label=robot.label,
            controller_profile=robot.controller_profile,
        )
    return views


def finalize_tiled_articulation_views(scene: IsaacTiledScene) -> IsaacTiledScene:
    """在 ``world.reset()`` 后解析 DOF 名称并填充 command-space 索引。"""

    views: dict[str, TiledArticulationView] = {}
    for name, runtime in scene.articulation_views.items():
        robot = scene.robots[name]
        mimic_controls = resolve_mimic_follower_controls(
            list(runtime.view.dof_names), robot.mimic_path
        )
        command_indices = _command_joint_indices(
            dof_names=list(runtime.view.dof_names),
            controlled_joints=robot.controlled_joints,
            mimic_path=robot.mimic_path,
        )
        views[name] = replace(
            runtime,
            command_joint_names=tuple(
                runtime.view.dof_names[int(index)] for index in command_indices
            ),
            command_joint_indices=command_indices,
            # URDF importer authors a native PhysX MimicJointAPI. MJCF equality
            # followers need explicit batched targets in the tiled runtime.
            runtime_mimic_controls=(
                tuple(mimic_controls) if robot.asset_type == "mjcf" else ()
            ),
        )
    return replace(scene, articulation_views=views)


def _command_joint_indices(
    *,
    dof_names: Sequence[str],
    controlled_joints: Sequence[str],
    mimic_path: Path | None = None,
) -> np.ndarray:
    """解析主动 command joints，并剔除资产声明的 mimic follower。"""

    requested = resolve_joint_indices(list(dof_names), list(controlled_joints))
    controls = resolve_mimic_follower_controls(list(dof_names), mimic_path)
    follower_names = {control.dependent_joint for control in controls}
    follower_indices = {
        index for index, name in enumerate(dof_names) if name in follower_names
    }
    return np.asarray(
        [int(index) for index in requested if int(index) not in follower_indices],
        dtype=int,
    )
