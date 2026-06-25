"""Isaac Sim 机器人资产导入工具。

本模块负责把 MJCF/URDF 资产导入当前 USD stage，并找到可交给 ``SingleArticulation``
使用的 articulation root。Isaac/Omni 相关导入刻意放在函数内部，这样普通 Python 测试可以
导入本包而不启动 Isaac。

职责边界:
    * 解析机器人资产配置并调用 Isaac importer。
    * 在导入后的 USD 子树中寻找 articulation root。
    * 不创建 ``World``，不配置控制器，不应用任务级初始姿态。

输入输出约定:
    路径字段可写仓库相对路径或绝对路径，最终通过 ``repo_path``/``resolve`` 规整；prim path
    必须是 USD 绝对路径（例如 ``/World/Robot``）。导入流程会修改当前 USD stage，因此调用方
    需要保证 ``SimulationApp`` 已启动且 stage 处于期望状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from manipulation_project.utils.paths import repo_path


@dataclass(frozen=True)
class RobotAssetConfig:
    """机器人资产导入配置。

    ``asset_path`` 指向 MJCF/URDF 文件，``prim_path`` 是导入到当前 USD stage 的目标路径。
    ``asset_type`` 只控制选择 MJCF importer 还是 URDF importer，不改变上层机器人控制 API。

    输入字段:
        asset_type: 资产类型，当前支持 ``mjcf`` 和 ``urdf``。
        asset_path: 资产文件路径，已解析为仓库相对/绝对 ``Path``。
        prim_path: 期望导入到 USD stage 的 prim 路径。
        name: 机器人逻辑名称，供上层记录或创建 articulation 时使用。
        urdf_drive_type: URDF importer 默认 drive 类型，支持 ``position`` 或 ``none``。
    输出:
        传给 ``import_robot_asset`` 后得到 articulation root 和实际资产路径。
    """

    asset_type: str
    asset_path: Path
    prim_path: str
    name: str = "robot"
    urdf_drive_type: str = "position"

    @classmethod
    def from_mapping(cls, data: dict) -> "RobotAssetConfig":
        """从 YAML 映射构造导入配置。

        参数:
            data: 完整配置，必须包含 ``robot`` 子配置字典。
        返回:
            ``RobotAssetConfig``；路径字段会通过 ``repo_path`` 解析。
        """

        # YAML 顶层保留 ``robot`` 子节，可以和 env/task/controller 配置并列；这里不接受
        # 裸映射，避免误把其它配置字典当成机器人资产配置。
        if "robot" not in data:
            raise ValueError("Robot config must contain top-level robot section")
        robot = data["robot"]
        return cls(
            asset_type=str(robot.get("asset_type", "mjcf")).lower(),
            asset_path=repo_path(robot["asset_path"]),
            prim_path=str(robot.get("prim_path", "/World/Robot")),
            name=str(robot.get("name", "robot")),
            urdf_drive_type=str(robot.get("urdf_drive_type", "position")),
        )


def find_articulation_root(prim_path: str, *, require_rigid_body: bool = False) -> str:
    """在指定 prim 子树下查找 articulation root。

    参数:
        prim_path: importer 创建的 USD 子树根路径。
        require_rigid_body: 为真时只接受同时具有 ``RigidBodyAPI`` 的 articulation root。
    返回:
        可传给 ``SingleArticulation`` 的 articulation root prim 路径字符串。
    """

    from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
    from pxr import Usd, UsdPhysics

    # importer 返回的路径有时是用户请求的根路径，有时是内部生成的子 prim；先确认路径
    # 存在，再遍历其子树寻找真正带 ArticulationRootAPI 的 prim。
    if not is_prim_path_valid(prim_path):
        raise RuntimeError(f"Stage prim was not created: {prim_path}")

    root = get_prim_at_path(prim_path)
    articulation_roots = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not articulation_roots:
        raise RuntimeError(f"No articulation root found under stage prim: {prim_path}")

    if require_rigid_body:
        # MJCF importer 常会在 articulation root 附近创建额外 Xform。执行控制时需要绑定到
        # 同时具备刚体语义的 root，才能被 Isaac articulation view 正确识别。
        rigid_roots = [prim for prim in articulation_roots if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not rigid_roots:
            raise RuntimeError(f"No rigid articulation root found under stage prim: {prim_path}")
        return str(rigid_roots[0].GetPath())
    return str(articulation_roots[0].GetPath())


def configure_mjcf_import(mjcf_path: Path, prim_path: str) -> str:
    """配置并导入 MJCF 资产。

    参数:
        mjcf_path: MJCF 文件路径。
        prim_path: 希望导入到 stage 的目标 prim 路径。
    返回:
        MJCF importer 实际创建的根 prim 路径。
    """

    import omni.kit.commands

    # MJCF importer 的配置对象由 Omni command 创建，不能直接实例化。配置在导入前一次性
    # 写入，导入后再由 usd_overrides/runtime controller 继续细化物理参数。
    status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create MJCF import config")

    import_config.set_fix_base(True)
    import_config.set_import_inertia_tensor(True)

    status, imported_prim_path = omni.kit.commands.execute(
        "MJCFCreateAsset",
        mjcf_path=str(mjcf_path),
        import_config=import_config,
        prim_path=prim_path,
    )
    if not status:
        raise RuntimeError(f"Failed to import MJCF: {mjcf_path}")
    return str(imported_prim_path or prim_path)


def configure_urdf_import(
    urdf_path: Path,
    *,
    dest_path: Path | None = None,
    create_physics_scene: bool = False,
    drive_type: str = "position",
) -> str:
    """配置并导入 URDF 资产。

    参数:
        urdf_path: URDF 文件路径。
        dest_path: 可选导出 USD 目标路径；为 ``None`` 时由 importer 决定。
        create_physics_scene: 是否让 importer 自动创建 physics scene。
        drive_type: 默认关节 drive 类型，支持 ``position`` 或 ``none``。
    返回:
        URDF importer 返回的 articulation root 路径。
    """

    import omni.kit.commands
    from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import config")

    # URDF 导入默认做凸分解和固定关节合并，得到更适合实时仿真的碰撞与关节树。
    # drive_type='none' 用于只需要几何/运动学、不希望 importer 预设控制参数的场景。
    import_config.merge_fixed_joints = True
    import_config.convex_decomp = True
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.self_collision = False
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
    import_config.make_default_prim = True
    import_config.create_physics_scene = create_physics_scene
    import_config.collision_from_visuals = False

    command_args = {
        "urdf_path": str(urdf_path),
        "import_config": import_config,
        "get_articulation_root": True,
    }
    if dest_path is not None:
        command_args["dest_path"] = str(dest_path)

    status, prim_path = omni.kit.commands.execute("URDFParseAndImportFile", **command_args)
    if not status or not prim_path:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")
    return str(prim_path)


def import_robot_asset(config: RobotAssetConfig) -> tuple[str, Path, str]:
    """按配置导入机器人资产。

    参数:
        config: 机器人资产导入配置。
    返回:
        ``(articulation_path, asset_path, imported_root_path)``：
        ``articulation_path`` 是 ``SingleArticulation`` 应绑定的 prim；
        ``asset_path`` 是实际使用的资产文件绝对路径；
        ``imported_root_path`` 是本次导入创建/覆盖的 USD 子树根路径。
    """

    # 在真正导入前检查文件存在，错误会指向用户配置的资产路径，而不是 Isaac importer
    # 内部更难读的失败信息。
    asset_path = config.asset_path.resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Robot asset not found: {asset_path}")

    if config.asset_type == "mjcf":
        imported_path = configure_mjcf_import(asset_path, config.prim_path)
        articulation_path = find_articulation_root(imported_path, require_rigid_body=True)
        return articulation_path, asset_path, imported_path

    if config.asset_type == "urdf":
        articulation_path = configure_urdf_import(asset_path, drive_type=config.urdf_drive_type)
        return articulation_path, asset_path, articulation_path

    raise ValueError(f"Unsupported robot asset type: {config.asset_type}")
