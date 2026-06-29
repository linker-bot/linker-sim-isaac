"""Isaac Sim 机器人资产导入工具。

本模块负责把 MJCF/URDF 资产导入当前 USD stage，并找到可交给 ``SingleArticulation``
使用的 articulation root。Isaac/Omni 相关导入刻意放在函数内部，这样普通 Python 测试可以
导入本包而不启动 Isaac。

职责边界:
    * 解析机器人资产配置并调用 Isaac importer。
    * 在导入后的 USD 子树中寻找 articulation root。
    * 不创建 ``World``，不配置控制器，不应用动作级初始姿态。

输入输出约定:
    路径字段可写仓库相对路径或绝对路径，最终通过 ``repo_path``/``resolve`` 规整；prim path
    必须是 USD 绝对路径（例如 ``/World/Robot``）。导入流程会修改当前 USD stage，因此调用方
    需要保证 ``SimulationApp`` 已启动且 stage 处于期望状态。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from linkerbot_sim.robots.classification import component_for_name
from linkerbot_sim.utils.paths import repo_path


DEFAULT_COLLISION_APPROXIMATION = "convex_decomposition"
SUPPORTED_COLLISION_APPROXIMATIONS = ("convex_decomposition", "convex_hull")


@dataclass(frozen=True)
class AssetImportConfig:
    """资产导入阶段的通用选项。

    这里的 ``collision_approximation`` 只作用于 Isaac importer 把 MJCF/URDF mesh 写入 USD
    的过程。它不是 cuMotion 规划模型配置，也不是导入后再额外 cooking 一次碰撞体的开关。
    """

    collision_approximation: str = DEFAULT_COLLISION_APPROXIMATION

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "AssetImportConfig":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        unsupported_keys = set(data) - {"collision_approximation"}
        if unsupported_keys:
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(f"{label} contains unsupported keys: {unsupported}")
        return cls(
            collision_approximation=_normalize_collision_approximation(
                data.get("collision_approximation", DEFAULT_COLLISION_APPROXIMATION),
                label=f"{label}.collision_approximation",
            )
        )

    def use_convex_decomposition(self) -> bool:
        """返回 MJCF/URDF importer 需要的 convex decomposition 布尔开关。"""

        return self.collision_approximation == "convex_decomposition"


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
        import_config: 资产导入选项，例如 collision approximation。
        gravity_policy: robot YAML 中的刚体重力策略，按 default/arm/hand 分组。
    输出:
        传给 ``import_robot_asset`` 后得到 articulation root 和实际资产路径。
    """

    asset_type: str
    asset_path: Path
    prim_path: str
    name: str = "robot"
    urdf_drive_type: str = "position"
    import_config: AssetImportConfig = field(default_factory=AssetImportConfig)
    gravity_policy: "RobotGravityPolicy" = field(
        default_factory=lambda: RobotGravityPolicy()
    )

    @classmethod
    def from_mapping(cls, data: dict) -> "RobotAssetConfig":
        """从 YAML 映射构造导入配置。

        参数:
            data: 完整配置，必须包含 ``robot`` 子配置字典。
        返回:
            ``RobotAssetConfig``；路径字段会通过 ``repo_path`` 解析。
        """

        # YAML 顶层保留 ``robot`` 子节，可以和 env/controller 等配置并列；这里不接受
        # 裸映射，避免误把其它配置字典当成机器人资产配置。
        if "robot" not in data:
            raise ValueError("Robot config must contain top-level robot section")
        robot = data["robot"]
        physics = _optional_mapping(robot, "physics", "robot")
        return cls(
            asset_type=str(robot.get("asset_type", "mjcf")).lower(),
            asset_path=repo_path(robot["asset_path"]),
            prim_path=str(robot.get("prim_path", "/World/Robot")),
            name=str(robot.get("name", "robot")),
            urdf_drive_type=str(robot.get("urdf_drive_type", "position")),
            import_config=AssetImportConfig.from_mapping(
                robot.get("import"), label="robot.import"
            ),
            gravity_policy=RobotGravityPolicy.from_mapping(
                physics.get("gravity") if physics is not None else None,
                label="robot.physics.gravity",
            ),
        )


@dataclass(frozen=True)
class RobotGravityPolicy:
    """机器人刚体重力策略，按 default/arm/hand 分组。

    ``false`` 表示给对应刚体写入 ``disableGravity=True``，``true`` 表示保留重力。
    分类基于刚体 prim 名称中的规范 category token；无法识别的名称走 ``default``。
    """

    default: bool = False
    arm: bool | None = None
    hand: bool | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "RobotGravityPolicy":
        """从 ``robot.physics.gravity`` 解析策略。

        只接受 mapping，避免 ``gravity: false`` 这类布尔简写重新变成第二套接口。
        """

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        unsupported_keys = set(data) - {"default", "arm", "hand"}
        if unsupported_keys:
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(f"{label} contains unsupported keys: {unsupported}")
        default = _optional_bool(data, "default", label)
        return cls(
            default=cls.default if default is None else default,
            arm=_optional_bool(data, "arm", label),
            hand=_optional_bool(data, "hand", label),
        )

    def enabled_for_name(self, name: str) -> bool:
        """按 USD prim/DOF 名称判断该刚体是否保留重力。"""

        return self.enabled_for_component(component_for_name(name))

    def enabled_for_component(self, component: str) -> bool:
        """按部件分类判断是否保留重力。"""

        if component == "arm" and self.arm is not None:
            return self.arm
        if component == "hand" and self.hand is not None:
            return self.hand
        return self.default

    def disables_all_known_components(self) -> bool:
        """是否对 default、arm、hand 都关闭了重力。

        该信息用于 reset 后同步调用 Isaac articulation runtime 的 ``disable_gravity()``。
        """

        return not self.enabled_for_component("default") and not (
            self.enabled_for_component("arm") or self.enabled_for_component("hand")
        )


@dataclass(frozen=True)
class RootPoseConfig:
    """机器人 root 在世界坐标下的固定位姿。"""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "RootPoseConfig":
        """从可选 ``root_pose`` mapping 解析 xyz/rpy。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("root_pose must be a mapping")
        return cls(
            xyz=_vec3_from_mapping(data, "xyz"),
            rpy=_vec3_from_mapping(data, "rpy"),
        )

    def is_identity(self) -> bool:
        """是否为默认零平移、零旋转。"""

        return self.xyz == (0.0, 0.0, 0.0) and self.rpy == (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RobotExecutionConfig:
    """单个 Isaac articulation 的导入配置和主动控制关节。"""

    robot: RobotAssetConfig
    controlled_joints: tuple[str, ...]
    root_pose: RootPoseConfig = RootPoseConfig()

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "RobotExecutionConfig":
        """从包含 ``robot`` 和可选 ``controlled_joints`` 的 mapping 解析单侧执行配置。"""

        config = dict(data)
        robot = RobotAssetConfig.from_mapping(config)
        controlled = _controlled_joints_from_mapping(config)
        return cls(
            robot=robot,
            controlled_joints=controlled,
            root_pose=RootPoseConfig.from_mapping(config.get("root_pose")),
        )


@dataclass(frozen=True)
class DualRobotExecutionConfig:
    """双 Isaac articulation 执行配置。"""

    left: RobotExecutionConfig
    right: RobotExecutionConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "DualRobotExecutionConfig":
        """从 ``robots.left/right`` 解析双机器人执行配置。"""

        robots = data.get("robots")
        if not isinstance(robots, Mapping):
            raise ValueError("Dual robot config must contain top-level robots mapping")
        return cls(
            left=RobotExecutionConfig.from_mapping(
                _required_mapping(robots, "left", "robots")
            ),
            right=RobotExecutionConfig.from_mapping(
                _required_mapping(robots, "right", "robots")
            ),
        )

    def side(self, side: str) -> RobotExecutionConfig:
        """返回指定侧执行配置。"""

        normalized = str(side).lower()
        if normalized == "left":
            return self.left
        if normalized == "right":
            return self.right
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def _controlled_joints_from_mapping(data: Mapping[str, object]) -> tuple[str, ...]:
    value = data.get("controlled_joints", ("all",))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("controlled_joints must be a sequence")
    joints = tuple(str(name) for name in value)
    if not joints:
        raise ValueError("controlled_joints cannot be empty")
    return joints


def _vec3_from_mapping(data: Mapping[str, object], key: str) -> tuple[float, float, float]:
    value = data.get(key, (0.0, 0.0, 0.0))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a length-3 sequence")
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{key} must contain exactly 3 values")
    return values


def _normalize_collision_approximation(value: object, *, label: str) -> str:
    normalized = str(value).strip()
    if normalized in SUPPORTED_COLLISION_APPROXIMATIONS:
        return normalized
    supported = ", ".join(SUPPORTED_COLLISION_APPROXIMATIONS)
    raise ValueError(f"{label} must be one of {supported}, got {value!r}")


def _required_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _optional_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _optional_bool(
    data: Mapping[str, object], key: str, parent_label: str
) -> bool | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"{parent_label}.{key} must be a boolean")
    return value


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
    articulation_roots = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not articulation_roots:
        raise RuntimeError(f"No articulation root found under stage prim: {prim_path}")

    if require_rigid_body:
        # MJCF importer 常会在 articulation root 附近创建额外 Xform。执行控制时需要绑定到
        # 同时具备刚体语义的 root，才能被 Isaac articulation view 正确识别。
        rigid_roots = [
            prim for prim in articulation_roots if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not rigid_roots:
            raise RuntimeError(
                f"No rigid articulation root found under stage prim: {prim_path}"
            )
        return str(rigid_roots[0].GetPath())
    return str(articulation_roots[0].GetPath())


def apply_root_pose(stage, root_path: str, pose: RootPoseConfig) -> None:
    """把机器人导入根 prim 摆到配置指定的世界位姿。"""

    from pxr import Gf, Sdf, UsdGeom

    prim = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not prim.IsValid():
        raise RuntimeError(f"Cannot apply root_pose; prim not found: {root_path}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pose.xyz))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*_radians_to_degrees(pose.rpy)))
    _apply_root_pose_to_mjcf_fixed_root_joints(stage, root_path, pose)


def _apply_root_pose_to_mjcf_fixed_root_joints(
    stage, root_path: str, pose: RootPoseConfig
) -> None:
    """同步 MJCF fixed-base root joint 的 world anchor。

    MJCF importer 在 ``fix_base=True`` 时会创建 ``rootJoint_*``，其 ``body0`` 为空，表示
    joint 另一端固定在 world。若只移动外层 Xform，reset 时 PhysX 会按 root joint 的默认
    world anchor 把 articulation base 拉回原点附近。因此 root pose 也要写到 joint 的
    ``localPos0/localRot0``。
    """

    from pxr import Gf, Sdf, Usd, UsdPhysics
    from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        return
    quat = rpy_xyz_to_quat_wxyz(pose.rpy)
    world_anchor_pos = Gf.Vec3f(*pose.xyz)
    world_anchor_rot = Gf.Quatf(
        float(quat[0]),
        Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3])),
    )
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "PhysicsFixedJoint":
            continue
        if not prim.GetName().startswith("rootJoint_"):
            continue
        body0_targets = prim.GetRelationship("physics:body0").GetTargets()
        if body0_targets:
            continue
        joint = UsdPhysics.Joint(prim)
        joint.CreateLocalPos0Attr().Set(world_anchor_pos)
        joint.CreateLocalRot0Attr().Set(world_anchor_rot)


def _radians_to_degrees(values: Sequence[float]) -> tuple[float, float, float]:
    import math

    return tuple(float(value) * 180.0 / math.pi for value in values)


def configure_mjcf_import(
    mjcf_path: Path,
    prim_path: str,
    *,
    asset_import_config: AssetImportConfig | None = None,
) -> str:
    """配置并导入 MJCF 资产。

    参数:
        mjcf_path: MJCF 文件路径。
        prim_path: 希望导入到 stage 的目标 prim 路径。
        asset_import_config: 资产导入选项。
    返回:
        MJCF importer 实际创建的根 prim 路径。
    """

    import omni.kit.commands

    # MJCF importer 的配置对象由 Omni command 创建，不能直接实例化。配置在导入前一次性
    # 写入，导入后再由 usd_overrides/runtime controller 继续细化物理参数。
    status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create MJCF import config")

    asset_import = asset_import_config or AssetImportConfig()
    import_config.set_fix_base(True)
    import_config.set_import_inertia_tensor(True)
    import_config.set_convex_decomp(asset_import.use_convex_decomposition())

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
    get_articulation_root: bool = True,
    make_default_prim: bool = True,
    fix_base: bool = True,
    asset_import_config: AssetImportConfig | None = None,
) -> str:
    """配置并导入 URDF 资产。

    参数:
        urdf_path: URDF 文件路径。
        dest_path: 可选导出 USD 目标路径；为 ``None`` 时由 importer 决定。
        create_physics_scene: 是否让 importer 自动创建 physics scene。
        drive_type: 默认关节 drive 类型，支持 ``position`` 或 ``none``。
        get_articulation_root: 是否要求 importer 返回 articulation root；环境物体可设为 ``False``。
        make_default_prim: 是否把导入资产设为默认 prim。
        fix_base: 是否让 URDF importer 固定 base；机器人导入默认固定，环境物体由 env 配置决定。
        asset_import_config: 资产导入选项。
    返回:
        URDF importer 返回的 prim 路径。
    """

    import omni.kit.commands
    from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create URDF import config")

    asset_import = asset_import_config or AssetImportConfig()
    # URDF 导入默认做凸分解和固定关节合并，得到更适合实时仿真的碰撞与关节树。
    # drive_type='none' 用于只需要几何/运动学、不希望 importer 预设控制参数的场景。
    import_config.merge_fixed_joints = True
    import_config.convex_decomp = asset_import.use_convex_decomposition()
    import_config.import_inertia_tensor = True
    import_config.fix_base = bool(fix_base)
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
    import_config.make_default_prim = bool(make_default_prim)
    import_config.create_physics_scene = create_physics_scene
    import_config.collision_from_visuals = False

    command_args = {
        "urdf_path": str(urdf_path),
        "import_config": import_config,
        "get_articulation_root": bool(get_articulation_root),
    }
    if dest_path is not None:
        command_args["dest_path"] = str(dest_path)

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", **command_args
    )
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
            drive_type=config.urdf_drive_type,
            asset_import_config=config.import_config,
        )
        return articulation_path, asset_path, articulation_path

    raise ValueError(f"Unsupported robot asset type: {config.asset_type}")
