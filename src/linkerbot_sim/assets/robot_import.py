"""Isaac/Omni 机器人资产 importer 与 articulation root 查找。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from linkerbot_sim.assets.robot_config import AssetImportConfig, RobotAssetConfig


_live_import_directories: list[TemporaryDirectory] = []


def find_articulation_root(prim_path: str, *, require_rigid_body: bool = False) -> str:
    """在 importer 创建的 USD 子树中定位 articulation root。"""

    from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
    from pxr import Usd, UsdPhysics

    if not is_prim_path_valid(prim_path):
        raise RuntimeError(f"Stage prim was not created: {prim_path}")
    root = get_prim_at_path(prim_path)
    articulation_roots = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not articulation_roots:
        raise RuntimeError(f"No articulation root found under stage prim: {prim_path}")
    if require_rigid_body:
        rigid_roots = [
            prim for prim in articulation_roots if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not rigid_roots:
            raise RuntimeError(
                f"No rigid articulation root found under stage prim: {prim_path}"
            )
        return str(rigid_roots[0].GetPath())
    return str(articulation_roots[0].GetPath())


def configure_mjcf_import(
    mjcf_path: Path,
    prim_path: str,
    *,
    asset_import_config: AssetImportConfig | None = None,
) -> str:
    """配置 MJCF importer 并返回实际创建的 root prim path。"""

    import omni.kit.commands

    status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create MJCF import config")
    asset_import = asset_import_config or AssetImportConfig()
    import_config.set_fix_base(
        True if asset_import.fix_base is None else asset_import.fix_base
    )
    import_config.set_merge_fixed_joints(
        False
        if asset_import.merge_fixed_joints is None
        else asset_import.merge_fixed_joints
    )
    import_config.set_import_inertia_tensor(asset_import.import_inertia_tensor)
    # fingertip 等纯坐标 frame 使用 MJCF site，避免空 body 被导入成无质量刚体。
    import_config.set_import_sites(asset_import.import_sites)
    import_config.set_convex_decomp(asset_import.use_convex_decomposition())
    import_config.set_self_collision(asset_import.self_collision)
    source_prim_path = "/Robot"
    import_directory = TemporaryDirectory(prefix="linkerbot-sim-mjcf-")
    try:
        destination = Path(import_directory.name) / f"{mjcf_path.stem}.usd"
        status, imported_prim_path = omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=str(mjcf_path),
            import_config=import_config,
            prim_path=source_prim_path,
            dest_path=str(destination),
        )
        if not status:
            raise RuntimeError(f"Failed to import MJCF: {mjcf_path}")
        _reference_imported_prim_from_usd(
            destination,
            source_path=str(imported_prim_path or source_prim_path),
            target_path=prim_path,
        )
    except BaseException:
        import_directory.cleanup()
        raise
    _live_import_directories.append(import_directory)
    return prim_path


def configure_urdf_import(
    urdf_path: Path,
    *,
    prim_path: str | None = None,
    dest_path: Path | None = None,
    create_physics_scene: bool = False,
    drive_type: str = "position",
    get_articulation_root: bool = True,
    make_default_prim: bool = True,
    fix_base: bool | None = None,
    asset_import_config: AssetImportConfig | None = None,
) -> str:
    """配置 URDF importer，并可把完整导入子树映射到目标 stage path。"""

    if prim_path is not None and dest_path is not None:
        raise ValueError("prim_path and dest_path cannot both be provided")

    import omni.kit.commands
    from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import config")
    asset_import = asset_import_config or AssetImportConfig()
    import_config.merge_fixed_joints = (
        True
        if asset_import.merge_fixed_joints is None
        else asset_import.merge_fixed_joints
    )
    import_config.convex_decomp = asset_import.use_convex_decomposition()
    import_config.import_inertia_tensor = asset_import.import_inertia_tensor
    import_config.fix_base = (
        bool(fix_base)
        if fix_base is not None
        else (True if asset_import.fix_base is None else asset_import.fix_base)
    )
    import_config.self_collision = asset_import.self_collision
    import_config.distance_scale = 1.0
    if drive_type == "none":
        import_config.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_NONE
        import_config.default_drive_strength = 0.0
        import_config.default_position_drive_damping = 0.0
    elif drive_type == "position":
        import_config.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_POSITION
        import_config.default_drive_strength = 1.0e5
        import_config.default_position_drive_damping = 1.0e4
    else:
        raise ValueError(f"Unsupported URDF drive type: {drive_type}")
    # target path 可能嵌套在 /World 下，不能把导入时的临时 root 留作 stage default prim。
    import_config.make_default_prim = bool(make_default_prim and prim_path is None)
    import_config.create_physics_scene = create_physics_scene
    import_config.collision_from_visuals = asset_import.collision_from_visuals
    # URDF mimic 是线性约束，交给 PhysX 原生 MimicJointAPI 作为唯一执行者；
    # runtime 仍解析同一文件以从外部 command space 剔除 follower。
    import_config.parse_mimic = True
    command_args: dict[str, object] = {
        "urdf_path": str(urdf_path),
        "import_config": import_config,
    }
    if dest_path is not None:
        command_args["dest_path"] = str(dest_path)
        command_args["get_articulation_root"] = get_articulation_root
        status, imported_path = omni.kit.commands.execute(
            "URDFParseAndImportFile", **command_args
        )
        if not status or not imported_path:
            raise RuntimeError(f"Failed to import URDF: {urdf_path}")
        return str(imported_path)

    # File mode preserves importer-generated material and physics layers. Keep the
    # temporary directory alive for as long as the composed stage can reference it.
    import_directory = TemporaryDirectory(prefix="linkerbot-sim-urdf-")
    try:
        destination = Path(import_directory.name) / f"{urdf_path.stem}.usd"
        command_args["dest_path"] = str(destination)
        command_args["get_articulation_root"] = False
        status, imported_path = omni.kit.commands.execute(
            "URDFParseAndImportFile", **command_args
        )
        if not status or not imported_path:
            raise RuntimeError(f"Failed to import URDF: {urdf_path}")
        imported_root = str(imported_path)
        target_root = prim_path or imported_root
        _reference_imported_prim_from_usd(
            destination,
            source_path=imported_root,
            target_path=target_root,
        )
    except BaseException:
        import_directory.cleanup()
        raise
    _live_import_directories.append(import_directory)

    if get_articulation_root:
        return find_articulation_root(target_root)
    return target_root


def _reference_imported_prim_from_usd(
    source_usd_path: Path,
    *,
    source_path: str,
    target_path: str,
    destination_stage=None,
) -> None:
    """把 file-backed importer 结果映射到当前 stage 的最终路径。"""

    from pxr import Sdf, Usd

    if destination_stage is None:
        import omni.usd

        destination_stage = omni.usd.get_context().get_stage()
    if destination_stage is None:
        raise RuntimeError("No active USD stage for imported asset")

    source_stage = Usd.Stage.Open(str(source_usd_path))
    if source_stage is None or not source_stage.GetPrimAtPath(source_path).IsValid():
        raise RuntimeError(
            f"Imported USD prim was not created: {source_usd_path}:{source_path}"
        )

    target = Sdf.Path(target_path)
    if not target.IsAbsolutePath() or not target.IsPrimPath():
        raise ValueError(
            f"Imported asset target must be an absolute prim path: {target_path}"
        )
    if destination_stage.GetPrimAtPath(target).IsValid():
        raise RuntimeError(f"Imported asset target prim already exists: {target_path}")

    parent = target.GetParentPath()
    if (
        parent != Sdf.Path.absoluteRootPath
        and not destination_stage.GetPrimAtPath(parent).IsValid()
    ):
        destination_stage.DefinePrim(parent)
    mapped_prim = destination_stage.DefinePrim(target)
    if not mapped_prim.GetReferences().AddReference(str(source_usd_path), source_path):
        raise RuntimeError(
            f"Failed to map imported USD prim {source_path} to {target_path}"
        )


def release_imported_asset_files() -> None:
    """清理 file-backed importer 保留的临时分层 USD。"""

    while _live_import_directories:
        _live_import_directories.pop().cleanup()


def import_robot_asset(config: RobotAssetConfig) -> tuple[str, Path, str]:
    """导入机器人并返回 articulation path、资产路径和 imported root path。"""

    asset_path = config.asset_path.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Robot asset not found: {asset_path}")
    if config.asset_type == "mjcf":
        imported_path = configure_mjcf_import(
            asset_path,
            config.prim_path,
            asset_import_config=config.import_config,
        )
        articulation_path = find_articulation_root(
            imported_path, require_rigid_body=True
        )
        return articulation_path, asset_path, imported_path
    if config.asset_type == "urdf":
        articulation_path = configure_urdf_import(
            asset_path,
            prim_path=config.prim_path,
            drive_type=config.urdf_drive_type,
            asset_import_config=config.import_config,
        )
        return articulation_path, asset_path, config.prim_path
    raise ValueError(f"Unsupported robot asset type: {config.asset_type}")


__all__ = [
    "configure_mjcf_import",
    "configure_urdf_import",
    "find_articulation_root",
    "import_robot_asset",
    "release_imported_asset_files",
]
