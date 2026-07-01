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
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from linkerbot_sim.robots.classification import component_for_name
from linkerbot_sim.assets.solver_overrides import (
    SolverIterationConfig,
    robot_solver_settings,
)
from linkerbot_sim.assets.usd_overrides import PhysxOverrideConfig
from linkerbot_sim.utils.paths import repo_path


DEFAULT_COLLISION_APPROXIMATION = "convex_decomposition"
SUPPORTED_COLLISION_APPROXIMATIONS = ("convex_decomposition", "convex_hull")


@dataclass(frozen=True)
class AssetImportConfig:
    """资产导入阶段的通用选项。

    这里的 ``collision_approximation`` 只作用于 Isaac importer 把 MJCF/URDF mesh 写入 USD
    的过程。它不是 cuMotion 规划模型配置，也不是导入后再额外 cooking 一次碰撞体的开关。
    ``self_collision`` 只用于机器人 articulation 导入，表示是否让 Isaac/PhysX 在同一
    articulation 内部 link 之间生成自碰撞接触。
    """

    collision_approximation: str = DEFAULT_COLLISION_APPROXIMATION
    self_collision: bool = False

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        label: str,
        allow_self_collision: bool = False,
    ) -> "AssetImportConfig":
        """从 robot.import / object.import 映射解析 importer 选项。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        supported_keys = {"collision_approximation"}
        if allow_self_collision:
            supported_keys.add("self_collision")
        unsupported_keys = set(data) - supported_keys
        if unsupported_keys:
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(f"{label} contains unsupported keys: {unsupported}")
        return cls(
            collision_approximation=_normalize_collision_approximation(
                data.get("collision_approximation", DEFAULT_COLLISION_APPROXIMATION),
                label=f"{label}.collision_approximation",
            ),
            self_collision=_optional_bool(data, "self_collision", label) or False,
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
        physx_overrides: robot YAML 中的材料和刚体阻尼覆盖，按 default/arm/hand 分组。
        solver_iterations: robot YAML 中的刚体 solver iteration 覆盖，按 arm/hand 分组。
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
    physx_overrides: "RobotPhysxOverrides" = field(
        default_factory=lambda: RobotPhysxOverrides()
    )
    solver_iterations: SolverIterationConfig | None = None

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
                robot.get("import"),
                label="robot.import",
                allow_self_collision=True,
            ),
            gravity_policy=RobotGravityPolicy.from_mapping(
                physics.get("gravity") if physics is not None else None,
                label="robot.physics.gravity",
            ),
            physx_overrides=RobotPhysxOverrides.from_mapping(
                physics.get("physx") if physics is not None else None,
                label="robot.physics.physx",
            ),
            solver_iterations=robot_solver_settings(
                physics.get("solver") if physics is not None else None,
                label="robot.physics.solver",
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
class RobotPhysxComponentOverride:
    """单个部件的机器人 USD/PhysX 材料和刚体阻尼覆盖。

    字段为 ``None`` 表示不覆盖 controller/默认值生成的 ``PhysxOverrideConfig``。
    """

    contact_static_friction: float | None = None
    contact_dynamic_friction: float | None = None
    contact_restitution: float | None = None
    rigid_body_linear_damping: float | None = None
    rigid_body_angular_damping: float | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "RobotPhysxComponentOverride":
        """解析单个 default/arm/hand 分组的 PhysX 覆盖字段。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        unsupported_keys = set(data) - {"material", "rigid_body"}
        if unsupported_keys:
            unsupported = ", ".join(sorted(unsupported_keys))
            raise ValueError(f"{label} contains unsupported keys: {unsupported}")
        material = _optional_mapping(data, "material", label) or {}
        rigid_body = _optional_mapping(data, "rigid_body", label) or {}
        _reject_unsupported_keys(
            material,
            {
                "contact_static_friction",
                "contact_dynamic_friction",
                "contact_restitution",
            },
            f"{label}.material",
        )
        _reject_unsupported_keys(
            rigid_body,
            {"linear_damping", "angular_damping"},
            f"{label}.rigid_body",
        )
        return cls(
            contact_static_friction=_optional_non_negative_float(
                material, "contact_static_friction", f"{label}.material"
            ),
            contact_dynamic_friction=_optional_non_negative_float(
                material, "contact_dynamic_friction", f"{label}.material"
            ),
            contact_restitution=_optional_non_negative_float(
                material, "contact_restitution", f"{label}.material"
            ),
            rigid_body_linear_damping=_optional_non_negative_float(
                rigid_body, "linear_damping", f"{label}.rigid_body"
            ),
            rigid_body_angular_damping=_optional_non_negative_float(
                rigid_body, "angular_damping", f"{label}.rigid_body"
            ),
        )

    def merge(
        self, override: "RobotPhysxComponentOverride"
    ) -> "RobotPhysxComponentOverride":
        """返回 ``override`` 的非空字段覆盖当前对象后的新配置。"""

        return RobotPhysxComponentOverride(
            contact_static_friction=(
                self.contact_static_friction
                if override.contact_static_friction is None
                else override.contact_static_friction
            ),
            contact_dynamic_friction=(
                self.contact_dynamic_friction
                if override.contact_dynamic_friction is None
                else override.contact_dynamic_friction
            ),
            contact_restitution=(
                self.contact_restitution
                if override.contact_restitution is None
                else override.contact_restitution
            ),
            rigid_body_linear_damping=(
                self.rigid_body_linear_damping
                if override.rigid_body_linear_damping is None
                else override.rigid_body_linear_damping
            ),
            rigid_body_angular_damping=(
                self.rigid_body_angular_damping
                if override.rigid_body_angular_damping is None
                else override.rigid_body_angular_damping
            ),
        )

    def apply_to(self, config: PhysxOverrideConfig) -> PhysxOverrideConfig:
        """把非空覆盖字段叠加到完整 USD/PhysX 覆盖配置上。"""

        updates: dict[str, float] = {}
        for field_name in (
            "contact_static_friction",
            "contact_dynamic_friction",
            "contact_restitution",
            "rigid_body_linear_damping",
            "rigid_body_angular_damping",
        ):
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = value
        return replace(config, **updates) if updates else config


@dataclass(frozen=True)
class RobotPhysxOverrides:
    """机器人资产级 PhysX 覆盖，支持 default/arm/hand 分组。

    ``robot.physics.physx.material`` 和 ``robot.physics.physx.rigid_body`` 作为通用默认值；
    ``default``/``arm``/``hand`` 子节可覆盖对应部件。
    """

    default: RobotPhysxComponentOverride = field(
        default_factory=RobotPhysxComponentOverride
    )
    arm: RobotPhysxComponentOverride | None = None
    hand: RobotPhysxComponentOverride | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "RobotPhysxOverrides":
        """解析 robot.physics.physx，并合并通用默认值与分组覆盖。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_unsupported_keys(
            data,
            {"material", "rigid_body", "default", "arm", "hand"},
            label,
        )
        common = RobotPhysxComponentOverride.from_mapping(
            {
                key: data[key]
                for key in ("material", "rigid_body")
                if key in data
            },
            label=label,
        )
        default = common
        if "default" in data:
            default = default.merge(
                RobotPhysxComponentOverride.from_mapping(
                    _required_mapping(data, "default", label),
                    label=f"{label}.default",
                )
            )
        arm = (
            RobotPhysxComponentOverride.from_mapping(
                _required_mapping(data, "arm", label),
                label=f"{label}.arm",
            )
            if "arm" in data
            else None
        )
        hand = (
            RobotPhysxComponentOverride.from_mapping(
                _required_mapping(data, "hand", label),
                label=f"{label}.hand",
            )
            if "hand" in data
            else None
        )
        return cls(default=default, arm=arm, hand=hand)

    def apply_to_configs(
        self, configs: dict[str, PhysxOverrideConfig]
    ) -> dict[str, PhysxOverrideConfig]:
        """把 robot 资产物理覆盖叠加到 controller 生成的完整 PhysX 配置上。"""

        result = {
            name: self.default.apply_to(config) for name, config in configs.items()
        }
        if self.arm is not None and "arm" in result:
            result["arm"] = self.arm.apply_to(result["arm"])
        if self.hand is not None and "hand" in result:
            result["hand"] = self.hand.apply_to(result["hand"])
        return result


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
class RobotSceneInstanceConfig:
    """env 中声明的机器人实例。"""

    name: str
    robot_profile: str
    root_pose: RootPoseConfig

    @classmethod
    def from_mapping(
        cls, name: str, data: Mapping[str, object]
    ) -> "RobotSceneInstanceConfig":
        """从 ``env.robots.<name>`` 解析 robot profile 引用和场景摆放。"""

        label = f"robots.{name}"
        _reject_unsupported_keys(data, {"robot_profile", "root_pose"}, label)
        robot_profile = str(data.get("robot_profile", ""))
        if not robot_profile:
            raise ValueError(f"{label}.robot_profile is required")
        return cls(
            name=name,
            robot_profile=robot_profile,
            root_pose=RootPoseConfig.from_mapping(
                _required_mapping(data, "root_pose", label)
            ),
        )


@dataclass(frozen=True)
class RobotExecutionConfig:
    """单个 Isaac articulation 的导入配置和主动控制关节。"""

    robot: RobotAssetConfig
    controlled_joints: tuple[str, ...]
    root_pose: RootPoseConfig = RootPoseConfig()

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        root_pose: RootPoseConfig | None = None,
    ) -> "RobotExecutionConfig":
        """从包含 ``robot`` 和可选 ``controlled_joints`` 的 mapping 解析单侧执行配置。"""

        config = dict(data)
        if "root_pose" in config:
            raise ValueError("robot root_pose belongs under env robots")
        robot = RobotAssetConfig.from_mapping(config)
        controlled = _controlled_joints_from_mapping(config)
        return cls(
            robot=robot,
            controlled_joints=controlled,
            root_pose=root_pose or RootPoseConfig(),
        )


@dataclass(frozen=True)
class DualRobotExecutionConfig:
    """双 Isaac articulation 执行配置。"""

    left: RobotExecutionConfig
    right: RobotExecutionConfig

    @classmethod
    def from_robot_configs(
        cls,
        *,
        left: Mapping[str, object],
        right: Mapping[str, object],
        root_poses: Mapping[str, RootPoseConfig] | None = None,
    ) -> "DualRobotExecutionConfig":
        """从左右两个单 articulation robot profile 组装双机器人执行配置。"""

        root_poses = root_poses or {}
        return cls(
            left=RobotExecutionConfig.from_mapping(
                left,
                root_pose=root_poses.get("left"),
            ),
            right=RobotExecutionConfig.from_mapping(
                right,
                root_pose=root_poses.get("right"),
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


def robot_scene_instance_from_env_config(
    env_config: Mapping[str, object], name: str
) -> RobotSceneInstanceConfig:
    """从 env ``robots.<name>`` 读取机器人实例声明。"""

    robots = env_config.get("robots")
    if not isinstance(robots, Mapping):
        raise ValueError("Environment config must contain top-level robots mapping")
    return RobotSceneInstanceConfig.from_mapping(
        name,
        _required_mapping(robots, name, "robots"),
    )


def dual_robot_scene_instances_from_env_config(
    env_config: Mapping[str, object],
) -> dict[str, RobotSceneInstanceConfig]:
    """从 env ``robots.dual.left/right`` 读取双机器人场景实例声明。"""

    robots = env_config.get("robots")
    if not isinstance(robots, Mapping):
        raise ValueError("Environment config must contain top-level robots mapping")
    dual = _required_mapping(robots, "dual", "robots")
    return {
        side: RobotSceneInstanceConfig.from_mapping(
            f"dual.{side}",
            _required_mapping(dual, side, "robots.dual"),
        )
        for side in ("left", "right")
    }


def robot_root_pose_from_env_config(
    env_config: Mapping[str, object], name: str
) -> RootPoseConfig:
    """从 env ``robots.<name>.root_pose`` 读取机器人场景摆放。"""

    return robot_scene_instance_from_env_config(env_config, name).root_pose


def dual_robot_root_poses_from_env_config(
    env_config: Mapping[str, object],
) -> dict[str, RootPoseConfig]:
    """从 env ``robots.dual.left/right`` 读取双机器人场景摆放。"""

    return {
        side: instance.root_pose
        for side, instance in dual_robot_scene_instances_from_env_config(
            env_config
        ).items()
    }


def _controlled_joints_from_mapping(data: Mapping[str, object]) -> tuple[str, ...]:
    """读取 controlled_joints；缺省 ["all"] 表示由控制器自动选择。"""

    value = data.get("controlled_joints", ("all",))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("controlled_joints must be a sequence")
    joints = tuple(str(name) for name in value)
    if not joints:
        raise ValueError("controlled_joints cannot be empty")
    return joints


def _vec3_from_mapping(data: Mapping[str, object], key: str) -> tuple[float, float, float]:
    """从 mapping 中读取三维向量字段；缺省为零向量。"""

    value = data.get(key, (0.0, 0.0, 0.0))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a length-3 sequence")
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{key} must contain exactly 3 values")
    return values


def _normalize_collision_approximation(value: object, *, label: str) -> str:
    """规范化 importer collision approximation，并拒绝旧别名/未知值。"""

    normalized = str(value).strip()
    if normalized in SUPPORTED_COLLISION_APPROXIMATIONS:
        return normalized
    supported = ", ".join(SUPPORTED_COLLISION_APPROXIMATIONS)
    raise ValueError(f"{label} must be one of {supported}, got {value!r}")


def _required_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object]:
    """读取必填子 mapping，并把错误信息定位到 parent.key。"""

    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _optional_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object] | None:
    """读取可选子 mapping；字段缺失时返回 None。"""

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _optional_bool(
    data: Mapping[str, object], key: str, parent_label: str
) -> bool | None:
    """读取可选布尔字段；字段缺失时返回 None。"""

    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"{parent_label}.{key} must be a boolean")
    return value


def _optional_non_negative_float(
    data: Mapping[str, object], key: str, parent_label: str
) -> float | None:
    """读取可选非负浮点字段；字段缺失时返回 None。"""

    if key not in data:
        return None
    value = float(data[key])
    if value < 0.0:
        raise ValueError(f"{parent_label}.{key} cannot be negative")
    return value


def _reject_unsupported_keys(
    data: Mapping[str, object], allowed: set[str], label: str
) -> None:
    """拒绝配置中的未知 key，避免拼写错误被静默忽略。"""

    unsupported_keys = set(data) - allowed
    if unsupported_keys:
        unsupported = ", ".join(sorted(unsupported_keys))
        raise ValueError(f"{label} contains unsupported keys: {unsupported}")


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
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*tuple(np.degrees(pose.rpy))))
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
    import_config.set_self_collision(asset_import.self_collision)

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
