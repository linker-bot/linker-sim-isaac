"""环境事实到 replicated PhysX stage 的专用构造器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np

from linkerbot_sim.configuration.controllers import ControllerProfiles
from linkerbot_sim.isaac.physics.core_api import create_articulation_core_view

from .assets import (
    define_source_environment,
    import_source_objects,
    import_source_robots,
    source_object_configs,
    validate_single_dynamic_rigid_object,
)
from .layout import (
    environment_origins,
    environment_root_paths,
    paths_from_suffix,
    relative_prim_suffix,
)
from .types import ImportedReplicatedRobot, ReplicatedPhysxScene


_PHYSX_GRID_SPACING_M = 3.0


@dataclass(frozen=True, slots=True)
class PhysxGridClonePlan:
    """由 PhysX builder 派生的固定 GridCloner 执行计划。

    三个 boolean 是经过真实 smoke 验证的 PhysX 接入合同，故意不接受 YAML 覆盖。
    ``spacing_m`` 同样属于当前 canonical Kaleidoscope scene 的 PhysX 布局策略；Newton
    不读取这个值，而是在自己的 plan 中派生共址 worlds。
    """

    env_root_paths: tuple[str, ...]
    env_origins: np.ndarray
    base_env_path: str
    env_prefix: str
    spacing_m: float = field(init=False, default=_PHYSX_GRID_SPACING_M)
    replicate_physics: Literal[True] = field(init=False, default=True)
    copy_from_source: Literal[True] = field(init=False, default=True)
    enable_env_ids: Literal[True] = field(init=False, default=True)

    @classmethod
    def from_environment_settings(
        cls,
        settings: object,
        *,
        num_envs: int,
    ) -> "PhysxGridClonePlan":
        return cls(
            env_root_paths=environment_root_paths(settings, num_envs=num_envs),
            env_origins=environment_origins(
                settings,
                num_envs=num_envs,
                spacing_m=_PHYSX_GRID_SPACING_M,
            ),
            base_env_path=str(getattr(settings, "base_env_path")),
            env_prefix=str(getattr(settings, "env_prefix")),
        )

    @property
    def root_path(self) -> str:
        return f"{self.base_env_path.rstrip('/')}/{self.env_prefix}_"


def build_replicated_physx_scene(
    *,
    stage: object,
    world: object,
    scene_settings: object,
    environment_settings: object,
    num_envs: int,
    dynamic_object_name: str,
    controller_bundle: str,
    controller_bundles: Mapping[str, ControllerProfiles],
    solver_type: str,
) -> ReplicatedPhysxScene:
    """导入 source env、执行 PhysX clone，并创建 reset 前 articulation views。

    本函数只接受已经创建好的 ``World``，不创建也不关闭它。调用方必须在返回后通过
    Session-owned ``PhysxRuntime.reset()`` 初始化 tensor entities，再调用
    :func:`finalize_replicated_robot_views` 冻结关节顺序和创建 rigid views。
    """

    plan = PhysxGridClonePlan.from_environment_settings(
        environment_settings,
        num_envs=num_envs,
    )
    roots = plan.env_root_paths
    desired_origins = plan.env_origins
    source_root = roots[0]
    object_configs = source_object_configs(scene_settings, env_root=source_root)
    validate_single_dynamic_rigid_object(
        object_configs,
        expected_name=dynamic_object_name,
    )
    # 对象状态闭包必须在首次 stage 写入前成立；失败时不会留下半个 source env。
    define_source_environment(
        stage,
        source_root,
        prepare_newton_render_topology=False,
    )
    object_handles = import_source_objects(
        stage,
        configs=object_configs,
        physics_backend="physx",
        prepare_newton_render_topology=False,
    )
    source_robots = import_source_robots(
        stage,
        scene_settings=scene_settings,
        env_root=source_root,
        controller_bundle=controller_bundle,
        controller_bundles=controller_bundles,
        solver_type=solver_type,
        physics_backend="physx",
        prepare_newton_render_topology=False,
        object_configs=object_configs,
    )
    actual_origins = _clone_environments(
        stage=stage,
        plan=plan,
    )
    if not np.allclose(actual_origins, desired_origins, rtol=0.0, atol=1.0e-5):
        raise RuntimeError(
            "GridCloner returned environment origins that differ from the strict "
            "PhysX GridCloner layout"
        )
    robots: list[ImportedReplicatedRobot] = []
    for source in source_robots:
        articulation_suffix = relative_prim_suffix(
            source_root, source.articulation_path
        )
        imported_suffix = relative_prim_suffix(source_root, source.imported_root_path)
        tcp_suffix = relative_prim_suffix(source_root, source.tcp_parent_body_path)
        articulation_paths = paths_from_suffix(roots, articulation_suffix)
        view = create_articulation_core_view(
            paths=articulation_paths,
            name=f"kaleidoscope_{source.label}_articulation",
            world_scene=getattr(world, "scene"),
            physics_backend="physx",
            controllable_dof_names=(
                None
                if source.controlled_joints == ("all",)
                else source.controlled_joints
            ),
        )
        robots.append(
            ImportedReplicatedRobot(
                robot_id=source.robot_id,
                label=source.label,
                profile_name=source.profile_name,
                profile=source.profile,
                controller_bundle_name=source.controller_bundle_name,
                controller_profiles=source.controller_profiles,
                execution=source.execution,
                asset_path=source.asset_path,
                asset_type=source.asset_type,
                articulation_paths=articulation_paths,
                imported_root_paths=paths_from_suffix(roots, imported_suffix),
                controlled_joints=source.controlled_joints,
                tcp_frame_name=source.tcp_frame_name,
                tcp_parent_frame_name=source.tcp_parent_frame_name,
                tcp_body_paths=paths_from_suffix(roots, tcp_suffix),
                tcp_offset_xyz=source.tcp_offset_xyz,
                tcp_offset_rpy=source.tcp_offset_rpy,
                articulation_view=view,
            )
        )
    object_paths = {
        config.name: paths_from_suffix(
            roots,
            relative_prim_suffix(source_root, config.prim_path),
        )
        for config in object_configs
    }
    return ReplicatedPhysxScene(
        env_root_paths=roots,
        env_origins=np.ascontiguousarray(actual_origins, dtype=np.float32),
        robots=tuple(robots),
        object_handles=object_handles,
        object_prim_paths=object_paths,
    )


def _clone_environments(
    *,
    stage: object,
    plan: PhysxGridClonePlan,
) -> np.ndarray:
    """调用 GridCloner，并把其中心化默认网格精确对齐到项目 row-major 原点。"""

    from isaacsim.core.cloner import GridCloner

    env_roots = plan.env_root_paths
    desired_origins = plan.env_origins
    count = len(env_roots)
    per_row = max(1, int(math.ceil(math.sqrt(count))))
    spacing = plan.spacing_m
    cloner = GridCloner(spacing=spacing, num_per_row=per_row, stage=stage)
    offsets = desired_origins - _grid_cloner_default_positions(
        num_envs=count,
        num_per_row=per_row,
        spacing=spacing,
    )
    positions = cloner.clone(
        source_prim_path=env_roots[0],
        prim_paths=list(env_roots),
        position_offsets=offsets,
        replicate_physics=plan.replicate_physics,
        base_env_path=plan.base_env_path,
        root_path=plan.root_path,
        copy_from_source=plan.copy_from_source,
        enable_env_ids=plan.enable_env_ids,
    )
    result = np.asarray(positions, dtype=np.float32)
    if result.shape != (count, 3) or not np.all(np.isfinite(result)):
        raise RuntimeError("GridCloner returned invalid environment origins")
    return np.ascontiguousarray(result)


def _grid_cloner_default_positions(
    *,
    num_envs: int,
    num_per_row: int,
    spacing: float,
) -> np.ndarray:
    """复现 Isaac GridCloner 的中心化网格，供 position_offsets 对齐。"""

    num_rows = int(math.ceil(num_envs / num_per_row))
    num_cols = int(math.ceil(num_envs / num_rows))
    row_offset = 0.5 * spacing * (num_rows - 1)
    col_offset = 0.5 * spacing * (num_cols - 1)
    result = np.zeros((num_envs, 3), dtype=np.float32)
    for env_id in range(num_envs):
        row = env_id // num_cols
        column = env_id % num_cols
        result[env_id, 0] = row_offset - row * spacing
        result[env_id, 1] = column * spacing - col_offset
    return result


__all__ = ["PhysxGridClonePlan", "build_replicated_physx_scene"]
