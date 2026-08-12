"""object runtime 共享的 stage 写入辅助函数。"""

from __future__ import annotations

from linkerbot_sim.assets.root_pose import RootPoseConfig, apply_root_pose_transform


def apply_root_pose_to_prim(
    stage,
    prim_path: str,
    pose: RootPoseConfig,
    *,
    prepare_newton_render_topology: bool = False,
) -> None:
    """把 scene root pose 写入对象 prim，并显式选择 render topology。"""

    apply_root_pose_transform(
        stage,
        prim_path,
        pose,
        prepare_newton_render_topology=prepare_newton_render_topology,
    )
