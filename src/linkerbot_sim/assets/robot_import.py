"""Isaac/Omni 机器人资产 importer 与 articulation root 查找。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from linkerbot_sim.assets.robot_config import RobotAssetConfig
from linkerbot_sim.assets.root_pose import (
    RootPoseConfig,
    apply_root_pose,
    apply_root_pose_transform,
)
from linkerbot_sim.configuration.robots import AssetImportConfig
from linkerbot_sim.isaac.physics.backend import (
    active_physics_backend,
    normalize_physics_backend,
)
from linkerbot_sim.robots.mimic.mjcf import parse_mjcf_joint_equalities


_live_import_directories: list[TemporaryDirectory] = []

_COLLISION_TYPES = {
    "convex_decomposition": "Convex Decomposition",
    "convex_hull": "Convex Hull",
}


def find_articulation_root(prim_path: str, *, stage: object | None = None) -> str:
    """在 importer 创建的 USD 子树中定位 articulation root。"""

    from pxr import Usd, UsdPhysics

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage for articulation discovery")
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"Stage prim was not created: {prim_path}")
    articulation_roots = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        or prim.HasAPI("PhysicsArticulationRootAPI")
        or prim.HasAPI("NewtonArticulationRootAPI")
    ]
    if not articulation_roots:
        raise RuntimeError(f"No articulation root found under stage prim: {prim_path}")
    return str(articulation_roots[0].GetPath())


def configure_mjcf_import(
    mjcf_path: Path,
    prim_path: str,
    *,
    asset_import_config: AssetImportConfig | None = None,
    physics_backend: object | None = None,
    prepare_newton_render_topology: bool = False,
    root_pose: RootPoseConfig | None = None,
) -> str:
    """使用 Isaac 6 MJCF Importer 3.0 并按明确后端映射 root prim。"""

    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    backend = _resolved_physics_backend(physics_backend)
    asset_import = asset_import_config or AssetImportConfig()
    if asset_import.merge_fixed_joints is not None:
        raise ValueError("MJCF Importer 3.0 does not support merge_fixed_joints")
    if asset_import.collision_from_visuals:
        raise ValueError("MJCF import does not support collision_from_visuals")
    _validate_native_mjcf_mimics(mjcf_path)
    import_directory = TemporaryDirectory(prefix="linkerbot-sim-mjcf-")
    try:
        import_config = MJCFImporterConfig(
            mjcf_path=str(mjcf_path),
            # Importer 3.0 的 usd_path 是输出目录；返回值才是生成的 root USD。
            usd_path=import_directory.name,
            # Physics scene 由项目 World 统一创建，资产不得带入第二套 simulation settings。
            import_scene=False,
            collision_type=_COLLISION_TYPES[asset_import.collision_approximation],
            allow_self_collision=asset_import.self_collision,
            # 保持 5.1 项目机器人默认固定基座的可观测行为。
            fix_base=True if asset_import.fix_base is None else asset_import.fix_base,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
        destination = Path(MJCFImporter(import_config).import_mjcf())
        source_prim_path = _discover_imported_root_path(destination)
        reference_asset = (
            _prepare_newton_render_reference_asset(
                destination,
                source_path=source_prim_path,
                root_pose=root_pose or RootPoseConfig(),
                physics_backend=backend,
            )
            if prepare_newton_render_topology
            else destination
        )
        _reference_imported_prim_from_usd(
            reference_asset,
            source_path=source_prim_path,
            target_path=prim_path,
            physics_backend=backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
        )
        if backend == "newton":
            _deactivate_imported_mjcf_actuators(prim_path)
        _apply_mesh_collision_approximation(
            prim_path,
            approximation=asset_import.collision_approximation,
        )
    except BaseException:
        import_directory.cleanup()
        raise
    _live_import_directories.append(import_directory)
    return prim_path


def _validate_native_mjcf_mimics(mjcf_path: Path) -> None:
    """拒绝 NewtonMimicAPI 无法无损表达的非线性 MJCF equality。"""

    for equality in parse_mjcf_joint_equalities(mjcf_path):
        if any(coefficient != 0.0 for coefficient in equality.polycoef[2:]):
            raise ValueError(
                f"MJCF equality {equality.name!r} uses nonlinear polycoef "
                f"{equality.polycoef}; Isaac 6 native NewtonMimicAPI only supports "
                "an offset and linear multiplier"
            )


def configure_urdf_import(
    urdf_path: Path,
    *,
    prim_path: str | None = None,
    output_directory: Path | None = None,
    drive_type: str = "position",
    get_articulation_root: bool = True,
    fix_base: bool | None = None,
    asset_import_config: AssetImportConfig | None = None,
    allow_common_physics_variant: bool = False,
    physics_backend: object | None = None,
    prepare_newton_render_topology: bool = False,
    root_pose: RootPoseConfig | None = None,
) -> str:
    """使用 Isaac 6 URDF Importer 3.0 导入并按明确后端映射完整分层资产。"""

    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    backend = _resolved_physics_backend(physics_backend)
    asset_import = asset_import_config or AssetImportConfig()
    if drive_type == "none":
        target_type = "none"
        stiffness = 0.0
        damping = 0.0
    elif drive_type == "position":
        target_type = "position"
        stiffness = 1.0e5
        damping = 1.0e4
    else:
        raise ValueError(f"Unsupported URDF drive type: {drive_type}")

    # 临时输出目录及其 payload/material/physics 子层必须和 stage 引用同寿命。
    import_directory = (
        TemporaryDirectory(prefix="linkerbot-sim-urdf-")
        if output_directory is None
        else None
    )
    output_path = (
        Path(import_directory.name)
        if import_directory is not None
        else Path(output_directory).resolve()
    )
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        effective_fix_base = (
            fix_base
            if fix_base is not None
            else (True if asset_import.fix_base is None else asset_import.fix_base)
        )
        import_config = URDFImporterConfig(
            urdf_path=str(urdf_path),
            usd_path=str(output_path),
            merge_fixed_joints=(
                False
                if asset_import.merge_fixed_joints is None
                else asset_import.merge_fixed_joints
            ),
            collision_from_visuals=asset_import.collision_from_visuals,
            collision_type=_COLLISION_TYPES[asset_import.collision_approximation],
            allow_self_collision=asset_import.self_collision,
            fix_base=effective_fix_base,
            joint_drive_type="force",
            joint_target_type=target_type,
            override_joint_stiffness=stiffness,
            override_joint_damping=damping,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
        destination = Path(URDFImporter(import_config).import_urdf())
        imported_root = _discover_imported_root_path(destination)
        target_root = prim_path or imported_root
        reference_asset = (
            _prepare_newton_render_reference_asset(
                destination,
                source_path=imported_root,
                root_pose=root_pose or RootPoseConfig(),
                physics_backend=backend,
                allow_common_physics_variant=allow_common_physics_variant,
            )
            if prepare_newton_render_topology
            else destination
        )
        _reference_imported_prim_from_usd(
            reference_asset,
            source_path=imported_root,
            target_path=target_root,
            physics_backend=backend,
            allow_common_physics_variant=allow_common_physics_variant,
            prepare_newton_render_topology=prepare_newton_render_topology,
        )
        _apply_mesh_collision_approximation(
            target_root,
            approximation=asset_import.collision_approximation,
        )
    except BaseException:
        if import_directory is not None:
            import_directory.cleanup()
        raise
    if import_directory is not None:
        _live_import_directories.append(import_directory)

    if get_articulation_root:
        return find_articulation_root(target_root)
    return target_root


def _discover_imported_root_path(source_usd_path: Path) -> str:
    """从 Importer 3.0 返回的 root USD 发现可引用的资产 root prim。"""

    from pxr import Usd

    source_stage = Usd.Stage.Open(str(source_usd_path))
    if source_stage is None:
        raise RuntimeError(f"Failed to open imported USD: {source_usd_path}")
    default_prim = source_stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return str(default_prim.GetPath())
    roots = [
        prim for prim in source_stage.GetPseudoRoot().GetChildren() if prim.IsValid()
    ]
    if len(roots) != 1:
        paths = ", ".join(str(prim.GetPath()) for prim in roots) or "<none>"
        raise RuntimeError(
            f"Imported USD must have one root prim when no default prim is set: "
            f"{source_usd_path} (roots: {paths})"
        )
    return str(roots[0].GetPath())


def _apply_mesh_collision_approximation(
    root_path: str,
    *,
    approximation: str,
    stage=None,
) -> int:
    """对 importer 已有 collision mesh 显式写入标准 USD 近似策略。"""

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        raise RuntimeError(
            f"Cannot configure imported collision meshes; prim not found: {root_path}"
        )
    token = {
        "convex_decomposition": "convexDecomposition",
        "convex_hull": "convexHull",
    }[approximation]
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh_collision = (
            UsdPhysics.MeshCollisionAPI(prim)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            else UsdPhysics.MeshCollisionAPI.Apply(prim)
        )
        mesh_collision.CreateApproximationAttr().Set(token)
        count += 1
    return count


def _deactivate_imported_mjcf_actuators(
    root_path: str,
    *,
    stage=None,
) -> int:
    """在 Newton 路径禁用 importer 的 direct actuator，保留项目 DriveAPI 单执行者。"""

    from pxr import Sdf, Usd

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage for MJCF actuator cleanup")
    root_sdf_path = Sdf.Path(root_path)
    root = stage.GetPrimAtPath(root_sdf_path)
    if not root.IsValid():
        raise RuntimeError(
            f"Cannot clean imported MJCF actuators; prim not found: {root_path}"
        )

    actuators = [
        prim for prim in Usd.PrimRange(root) if str(prim.GetTypeName()) == "MjcActuator"
    ]
    for actuator in actuators:
        targets = tuple(actuator.GetRelationship("mjc:target").GetTargets())
        if len(targets) != 1:
            raise RuntimeError(
                "Newton MJCF actuator cleanup supports exactly one joint target: "
                f"actuator={actuator.GetPath()}, targets={list(targets)!r}"
            )
        target_path = targets[0]
        target_text = str(target_path)
        root_text = str(root_sdf_path).rstrip("/")
        if target_text != root_text and not target_text.startswith(f"{root_text}/"):
            raise RuntimeError(
                "Newton MJCF actuator target escapes the imported robot root: "
                f"actuator={actuator.GetPath()}, target={target_path}, root={root_path}"
            )
        target = stage.GetPrimAtPath(target_path)
        target_type = str(target.GetTypeName()) if target.IsValid() else ""
        if target_type not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}:
            raise RuntimeError(
                "Newton MJCF actuator cleanup only supports joint motors: "
                f"actuator={actuator.GetPath()}, target={target_path}, "
                f"target_type={target_type!r}"
            )

    # SetActive 会使 PrimRange iterator 失效，所以先 materialize 并完成全部校验。
    with Usd.EditContext(stage, stage.GetRootLayer()):
        for actuator in actuators:
            if not actuator.SetActive(False):
                raise RuntimeError(
                    f"Failed to deactivate imported MJCF actuator: {actuator.GetPath()}"
                )
    return len(actuators)


def _reference_imported_prim_from_usd(
    source_usd_path: Path,
    *,
    source_path: str,
    target_path: str,
    destination_stage=None,
    physics_backend: object | None = None,
    allow_common_physics_variant: bool = False,
    prepare_newton_render_topology: bool = False,
) -> None:
    """把 file-backed importer 结果映射到当前 stage 的最终路径。"""

    from pxr import Sdf, Usd, UsdGeom

    if destination_stage is None:
        import omni.usd

        destination_stage = omni.usd.get_context().get_stage()
    if destination_stage is None:
        raise RuntimeError("No active USD stage for imported asset")
    backend = _resolved_physics_backend(physics_backend)
    if prepare_newton_render_topology and backend != "newton":
        raise RuntimeError(
            "Newton render topology intent requires physics_backend='newton'"
        )

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
    mapped_prim = (
        UsdGeom.Xform.Define(destination_stage, target).GetPrim()
        if prepare_newton_render_topology
        else destination_stage.DefinePrim(target)
    )
    if prepare_newton_render_topology:
        # 在 reference 把源资产的 xformOpOrder 暴露给 Hydra 前，先以本地强 opinion
        # 发布最终 canonical topology。后续 scene root pose 只更新同一 matrix value。
        apply_root_pose_transform(
            destination_stage,
            str(target),
            RootPoseConfig(),
            prepare_newton_render_topology=True,
        )
    if not mapped_prim.GetReferences().AddReference(str(source_usd_path), source_path):
        raise RuntimeError(
            f"Failed to map imported USD prim {source_path} to {target_path}"
        )
    _select_imported_physics_variant(
        mapped_prim,
        physics_backend=backend,
        allow_common_physics_variant=allow_common_physics_variant,
    )


def _prepare_newton_render_reference_asset(
    source_usd_path: Path,
    *,
    source_path: str,
    root_pose: RootPoseConfig,
    physics_backend: object,
    allow_common_physics_variant: bool = False,
    physics_variant_required: bool = True,
    output_path: Path | None = None,
) -> Path:
    """在离线 wrapper 中完成 placed asset 的最终 Newton render topology。

    importer 输出仍保留后端中立分层；本函数只新建一层临时 wrapper。live stage 随后
    reference 的已经是 root pose 和全部 body matrix op 均固定的组合，因此 Hydra 从未
    观察到 importer 原始 ``translate/orient/scale`` body order。
    """

    from linkerbot_sim.isaac.physics.newton.render import (
        prepare_newton_render_subtree,
    )
    from pxr import Sdf, Usd, UsdGeom

    backend = _resolved_physics_backend(physics_backend)
    if backend != "newton":
        raise RuntimeError(
            "Newton render reference preparation requires physics_backend='newton'"
        )
    source = Path(source_usd_path).resolve()
    source_stage = Usd.Stage.Open(str(source))
    source_prim = (
        None if source_stage is None else source_stage.GetPrimAtPath(source_path)
    )
    if (
        source_prim is None
        or not bool(source_prim.IsValid())
        or not bool(source_prim.IsA(UsdGeom.Xformable))
    ):
        raise RuntimeError(
            f"Newton render source root is not Xformable: {source}:{source_path}"
        )
    wrapper = (
        source.with_name(f"{source.stem}.newton_render.usda")
        if output_path is None
        else Path(output_path).resolve()
    )
    stage = Usd.Stage.CreateNew(str(wrapper))
    if stage is None:
        raise RuntimeError(f"Failed to create Newton render wrapper: {wrapper}")
    root_sdf_path = Sdf.Path(source_path)
    root = UsdGeom.Xform.Define(stage, root_sdf_path)
    # 先发布本地强 root order，再添加 source reference；离线 stage 虽没有 Hydra，仍保持
    # 与 live reference 相同的 topology contract，避免 wrapper 自身携带过渡 order。
    apply_root_pose_transform(
        stage,
        source_path,
        root_pose,
        prepare_newton_render_topology=True,
    )
    if not root.GetPrim().GetReferences().AddReference(str(source), source_path):
        raise RuntimeError(
            f"Failed to create Newton render wrapper reference: {source}:{source_path}"
        )
    variant_names = tuple(
        str(name) for name in root.GetPrim().GetVariantSet("Physics").GetVariantNames()
    )
    if physics_variant_required or variant_names:
        _select_imported_physics_variant(
            root.GetPrim(),
            physics_backend=backend,
            allow_common_physics_variant=allow_common_physics_variant,
        )
    # reference/variant 已组合后再同步 fixed-base anchor；root topology 已固定，调用只会
    # 更新 matrix value，并为下面的 body world-matrix 采样提供最终 scene placement。
    apply_root_pose(
        stage,
        source_path,
        root_pose,
        prepare_newton_render_topology=True,
    )
    prepare_newton_render_subtree(stage=stage, subtree_root=source_path)
    stage.SetDefaultPrim(root.GetPrim())
    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"Failed to save Newton render wrapper: {wrapper}")
    return wrapper


def prepare_session_newton_render_reference_asset(
    source_usd_path: Path,
    *,
    root_pose: RootPoseConfig,
    physics_backend: object,
    source_path: str | None = None,
) -> Path:
    """为普通 USD 资产创建与当前 session 同寿命的 Newton render wrapper。

    source 资产可能只读，也可能在同一 scene 中被多次实例化，因此不能在资产旁写固定
    wrapper。每次调用都分配独立临时目录；目录只在 wrapper 完整保存后登记到 importer
    生命周期，并由 :func:`release_imported_asset_files` 在 native App 关闭前统一释放。
    """

    backend = _resolved_physics_backend(physics_backend)
    if backend != "newton":
        raise RuntimeError(
            "Newton render reference preparation requires physics_backend='newton'"
        )
    source = Path(source_usd_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"USD reference asset not found: {source}")
    resolved_source_path = source_path or _discover_imported_root_path(source)
    wrapper_directory = TemporaryDirectory(prefix="linkerbot-sim-newton-render-")
    wrapper_path = Path(wrapper_directory.name) / f"{source.stem}.newton_render.usda"
    try:
        prepared = _prepare_newton_render_reference_asset(
            source,
            source_path=resolved_source_path,
            root_pose=root_pose,
            physics_backend=backend,
            # 手写 USD object 通常没有 Physics variant；若存在，则必须明确提供
            # Newton 的 mujoco 或后端通用 physics 变体，不能沿用一个错误默认项。
            allow_common_physics_variant=True,
            physics_variant_required=False,
            output_path=wrapper_path,
        )
    except BaseException:
        wrapper_directory.cleanup()
        raise
    _live_import_directories.append(wrapper_directory)
    return prepared


def _select_imported_physics_variant(
    mapped_prim: object,
    *,
    physics_backend: object,
    allow_common_physics_variant: bool = False,
) -> str:
    """选择 Importer 3.0 的后端子层，禁止沿用错误的默认 variant。"""

    backend = normalize_physics_backend(physics_backend)
    expected = "mujoco" if backend == "newton" else "physx"
    variants = mapped_prim.GetVariantSet("Physics")
    names = tuple(str(name) for name in variants.GetVariantNames())
    by_normalized_name = {name.strip().lower(): name for name in names}
    selected = by_normalized_name.get(expected)
    if selected is None and allow_common_physics_variant:
        selected = by_normalized_name.get("physics")
    if selected is None:
        raise RuntimeError(
            "Imported robot does not provide the required Physics variant: "
            f"backend={backend!r}, required={expected!r}, available={list(names)!r}, "
            f"common_fallback={allow_common_physics_variant}"
        )
    if not variants.SetVariantSelection(selected):
        raise RuntimeError(
            f"Failed to select imported Physics variant {selected!r} for {backend}"
        )
    if variants.GetVariantSelection() != selected:
        raise RuntimeError(
            "Imported Physics variant selection did not persist: "
            f"expected={selected!r}, actual={variants.GetVariantSelection()!r}"
        )
    return selected


def release_imported_asset_files() -> None:
    """清理 file-backed importer 保留的临时分层 USD。"""

    while _live_import_directories:
        _live_import_directories.pop().cleanup()


def import_robot_asset(
    config: RobotAssetConfig,
    *,
    physics_backend: object | None = None,
    prepare_newton_render_topology: bool = False,
    root_pose: RootPoseConfig | None = None,
) -> tuple[str, Path, str]:
    """按明确物理后端导入机器人并返回 articulation、资产和导入根路径。"""

    asset_path = config.asset_path.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Robot asset not found: {asset_path}")
    if prepare_newton_render_topology and root_pose is None:
        raise ValueError(
            "Newton render robot import requires the resolved scene root_pose"
        )
    if config.asset_type == "mjcf":
        imported_path = configure_mjcf_import(
            asset_path,
            config.prim_path,
            asset_import_config=config.import_config,
            physics_backend=physics_backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
            root_pose=root_pose,
        )
        articulation_path = find_articulation_root(imported_path)
        return articulation_path, asset_path, imported_path
    if config.asset_type == "urdf":
        articulation_path = configure_urdf_import(
            asset_path,
            prim_path=config.prim_path,
            drive_type=config.urdf_drive_type,
            asset_import_config=config.import_config,
            physics_backend=physics_backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
            root_pose=root_pose,
        )
        return articulation_path, asset_path, config.prim_path
    raise ValueError(f"Unsupported robot asset type: {config.asset_type}")


def _resolved_physics_backend(value: object | None) -> str:
    """优先使用调用方拥有的后端事实，仅在省略时读取 runtime registry。"""

    return (
        active_physics_backend() if value is None else normalize_physics_backend(value)
    )


__all__ = [
    "configure_mjcf_import",
    "configure_urdf_import",
    "find_articulation_root",
    "import_robot_asset",
    "prepare_session_newton_render_reference_asset",
    "release_imported_asset_files",
]
