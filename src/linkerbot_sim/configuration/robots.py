"""``configs/robots`` 的唯一 strict mapping schema 与 typed settings。

本模块只解析后端无关的 YAML 数据，不导入 Isaac/Omni。调用方可以在普通 Python
环境中加载并校验机器人 profile；真正修改 USD stage 的操作由 ``robot_import`` 和
``root_pose`` 模块负责。

``validate_robot_profile`` 校验项目拥有的完整机器人 schema 与能力绑定。执行时使用的实例
路径和资产绝对路径由 ``assets.robot_config`` 投影；本模块不会创建资产、材质或 articulation。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from numbers import Real
from typing import Any

from linkerbot_sim.assets.solver_overrides import (
    SolverIterationConfig,
    robot_solver_settings,
)
from linkerbot_sim.assets.usd_overrides import RobotUsdOverrideConfig
from linkerbot_sim.backends.curobo.config import CuroboRobotConfig
from linkerbot_sim.configuration.controllers import normalize_controller_bundle_name
from linkerbot_sim.robots.classification import (
    RobotComponentMapping,
    component_for_name,
)
from linkerbot_sim.robots.capabilities import (
    PlanningBindingConfig,
    RobotKind,
    robot_kind_from_profile,
)
from linkerbot_sim.robots.joint_groups import JointGroupLayout


DEFAULT_COLLISION_APPROXIMATION = "convex_decomposition"
SUPPORTED_COLLISION_APPROXIMATIONS = ("convex_decomposition", "convex_hull")
_ROBOT_PROFILE_ROOT_KEYS = frozenset(
    {
        "robot",
        "curobo",
        "joint_groups",
        "rigid_body_groups",
        "controlled_joints",
    }
)
_ROBOT_KEYS = frozenset(
    {
        "kind",
        "name",
        "controller_profile",
        "asset_type",
        "asset_path",
        "urdf_drive_type",
        "import",
        "physics",
        "planning_collision",
    }
)
_CUROBO_ROBOT_PROFILE_KEYS = frozenset({"enabled", "planning_joint_group", "robot"})


@dataclass(frozen=True)
class RobotCuroboSettings:
    """机器人 profile 拥有的 cuRobo 能力绑定与模型资源。

    数值求解器参数与 CUDA 设备不属于机器人资产，分别由 ``configs/curobo`` 和 mode root
    拥有。本类型只保存是否启用、规划关节组以及已严格解析的机器人模型资源。
    """

    binding: PlanningBindingConfig
    robot: CuroboRobotConfig | None

    @classmethod
    def from_profile(
        cls,
        data: Mapping[str, object],
        *,
        kind: RobotKind,
    ) -> "RobotCuroboSettings":
        """从 robot YAML 的唯一 schema 边界构造 typed cuRobo 绑定。"""

        binding = PlanningBindingConfig.from_profile(data, kind=kind)
        if not binding.enabled:
            return cls(binding=binding, robot=None)
        curobo = _required_mapping(data, "curobo", "profile")
        robot = CuroboRobotConfig.from_mapping(
            _required_mapping(curobo, "robot", "curobo")
        )
        return cls(binding=binding, robot=robot)


@dataclass(frozen=True)
class RobotPlanningCollisionSphereSettings:
    """固定在机器人 root 坐标系中的一个保守规划碰撞球。"""

    name: str
    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class RobotPlanningCollisionSettings:
    """缺少 link collision spheres 时使用的机器人 root 包络。"""

    spheres: tuple[RobotPlanningCollisionSphereSettings, ...]

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
    ) -> "RobotPlanningCollisionSettings | None":
        """严格解析 ``robot.planning_collision``，缺省时返回 ``None``。"""

        if data is None:
            return None
        _reject_unsupported_keys(data, {"spheres"}, "robot.planning_collision")
        values = data.get("spheres")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("robot.planning_collision.spheres must be a sequence")
        if not values:
            raise ValueError("robot.planning_collision.spheres cannot be empty")
        spheres: list[RobotPlanningCollisionSphereSettings] = []
        for index, item in enumerate(values):
            label = f"robot.planning_collision.spheres[{index}]"
            if not isinstance(item, Mapping):
                raise ValueError(f"{label} must be a mapping")
            _reject_unsupported_keys(item, {"name", "center", "radius"}, label)
            radius = _finite_number(item.get("radius"), f"{label}.radius")
            if radius <= 0.0:
                raise ValueError(f"{label}.radius must be positive")
            spheres.append(
                RobotPlanningCollisionSphereSettings(
                    name=_non_empty_string(
                        item.get("name", f"sphere_{index}"), f"{label}.name"
                    ),
                    center=_finite_vector3(item.get("center"), f"{label}.center"),
                    radius=radius,
                )
            )
        names = tuple(sphere.name for sphere in spheres)
        if len(names) != len(set(names)):
            raise ValueError(
                "robot.planning_collision.spheres contains duplicate names"
            )
        return cls(spheres=tuple(spheres))


@dataclass(frozen=True)
class RobotProfileSettings:
    """一份完整 robot profile 的冻结、后端无关 typed 语义。

    catalog 在 YAML 边界完成全部 mapping 解析；资产、cuRobo、碰撞和产品 composition 只消费
    下列 typed 字段，不能再保留或二次解释原始文档。
    """

    name: str
    kind: RobotKind
    asset_type: str
    asset_path: str
    controller_profile: str | None
    urdf_drive_type: str
    import_config: "AssetImportConfig"
    gravity_policy: "RobotGravityPolicy"
    contact_material: "RobotContactMaterialSettings | None"
    physx: "RobotPhysxSettings"
    component_mapping: RobotComponentMapping
    joint_groups: JointGroupLayout
    controlled_joints: tuple[str, ...]
    curobo: RobotCuroboSettings
    planning_collision: RobotPlanningCollisionSettings | None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        source: str = "<robot profile>",
    ) -> "RobotProfileSettings":
        """严格校验 catalog 提供的文档并生成唯一 typed 表示。"""

        canonical = validate_robot_profile(data, source=source)
        robot = _required_mapping(canonical, "robot", "profile")
        physics = _optional_mapping(robot, "physics", "robot")
        asset_type = _non_empty_string(
            robot.get("asset_type", "mjcf"), "robot.asset_type"
        ).lower()
        kind = robot_kind_from_profile(canonical)
        layout = _validate_robot_component_groups(canonical, kind=kind)
        component_mapping = RobotComponentMapping.from_profile(canonical)
        return cls(
            name=_non_empty_string(robot.get("name", "robot"), "robot.name"),
            kind=kind,
            asset_type=asset_type,
            asset_path=_non_empty_string(robot["asset_path"], "robot.asset_path"),
            controller_profile=normalize_robot_controller_profile(robot),
            urdf_drive_type=_non_empty_string(
                robot.get("urdf_drive_type", "position"), "robot.urdf_drive_type"
            ).lower(),
            import_config=AssetImportConfig.from_mapping(
                _optional_mapping(robot, "import", "robot"),
                label="robot.import",
                asset_type=asset_type,
                allow_self_collision=True,
            ),
            gravity_policy=RobotGravityPolicy.from_mapping(
                _nested_mapping(physics, "gravity", "robot.physics"),
                label="robot.physics.gravity",
            ),
            contact_material=RobotContactMaterialSettings.from_mapping(
                _nested_mapping(physics, "material", "robot.physics"),
                label="robot.physics.material",
            ),
            physx=RobotPhysxSettings.from_mapping(
                _nested_mapping(physics, "physx", "robot.physics"),
                label="robot.physics.physx",
            ),
            component_mapping=component_mapping,
            joint_groups=layout,
            controlled_joints=_controlled_joint_names(canonical, layout=layout),
            curobo=RobotCuroboSettings.from_profile(canonical, kind=kind),
            planning_collision=RobotPlanningCollisionSettings.from_mapping(
                _optional_mapping(robot, "planning_collision", "robot")
            ),
        )


@dataclass(frozen=True)
class AssetImportConfig:
    """非 USD 资产 importer 的格式相关选项。

    ``collision_approximation`` 只接受当前支持的 convex decomposition/hull；
    ``self_collision`` 仅在调用边界明确允许时可配置。``fix_base`` 是三态开关；URDF 的
    ``merge_fixed_joints`` 为 ``None`` 时采用 Importer 3.0 的 ``false`` 默认值。Isaac 6 已移除
    惯量张量和 MJCF site/固定关节合并开关，配置入口会明确拒绝这些旧字段。
    """

    collision_approximation: str = DEFAULT_COLLISION_APPROXIMATION
    self_collision: bool = False
    fix_base: bool | None = None
    merge_fixed_joints: bool | None = None
    collision_from_visuals: bool = False

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        label: str,
        asset_type: str,
        allow_self_collision: bool = False,
    ) -> "AssetImportConfig":
        """从 ``robot.import`` 或 ``object.import`` 解析规范结构。

        参数:
            data: 可选 import mapping；``None`` 表示全部使用项目默认值。
            label: 完整配置路径，用于错误定位。
            asset_type: 当前资产格式，决定允许的 importer 字段。
            allow_self_collision: 当前配置域是否拥有 ``self_collision`` 字段。
        返回:
            冻结且已按资产格式验证的导入配置。
        异常:
            ValueError: 资产格式、mapping、字段类型/枚举或字段所有权不合法。
        """

        normalized_asset_type = str(asset_type).lower()
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        if normalized_asset_type == "usd":
            raise ValueError(f"{label} is not supported for USD assets")
        supported_keys = {
            "collision_approximation",
            "fix_base",
        }
        if normalized_asset_type == "urdf":
            supported_keys.update({"merge_fixed_joints", "collision_from_visuals"})
        elif normalized_asset_type != "mjcf":
            raise ValueError(f"{label} has unsupported asset type: {asset_type!r}")
        if allow_self_collision:
            supported_keys.add("self_collision")
        unsupported_keys = set(data) - supported_keys
        if unsupported_keys:
            unsupported = ", ".join(sorted(unsupported_keys))
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported_keys))
            raise ValueError(
                f"{label} contains unsupported keys: {unsupported} "
                f"(full paths: {paths})"
            )
        return cls(
            collision_approximation=_normalize_collision_approximation(
                data.get("collision_approximation", DEFAULT_COLLISION_APPROXIMATION),
                label=f"{label}.collision_approximation",
            ),
            self_collision=_optional_bool(data, "self_collision", label) or False,
            fix_base=_optional_nullable_bool(data, "fix_base", label),
            merge_fixed_joints=_optional_nullable_bool(
                data, "merge_fixed_joints", label
            ),
            collision_from_visuals=_bool_with_default(
                data,
                "collision_from_visuals",
                label,
                default=False,
            ),
        )


@dataclass(frozen=True)
class RobotGravityPolicy:
    """按 default/arm/hand 分组的机器人刚体重力策略。

    ``default`` 始终是最终布尔值；``arm`` 和 ``hand`` 为 ``None`` 时继承 default，显式
    布尔值则覆盖对应部件。该继承不修改对象，消费方通过查询方法得到最终策略。
    """

    default: bool = False
    arm: bool | None = None
    hand: bool | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "RobotGravityPolicy":
        """解析 ``robot.physics.gravity`` 的严格分组 mapping。

        参数:
            data: 可选 default/arm/hand mapping；``None`` 表示项目默认策略。
            label: 完整配置路径，用于错误定位。
        返回:
            冻结的继承式重力策略。
        异常:
            ValueError: 使用布尔简写、未知字段或非布尔叶子。
        """

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_unsupported_keys(data, {"default", "arm", "hand"}, label)
        default = _optional_bool(data, "default", label)
        return cls(
            default=cls.default if default is None else default,
            arm=_optional_bool(data, "arm", label),
            hand=_optional_bool(data, "hand", label),
        )

    def enabled_for_name(self, name: str) -> bool:
        """根据 USD prim 或 DOF 名称分类返回最终重力开关。

        参数:
            name: 待分类的实体名称。
        返回:
            对应 arm/hand/default 继承解析后的布尔值；无副作用。
        """

        return self.enabled_for_component(component_for_name(name))

    def enabled_for_component(self, component: str) -> bool:
        """返回规范部件的最终策略；无专用值时继承 ``default``。

        ``component`` 仅对 ``arm``/``hand`` 有专用分支，其它字符串均按 default 处理。
        """

        if component == "arm" and self.arm is not None:
            return self.arm
        if component == "hand" and self.hand is not None:
            return self.hand
        return self.default

    def disables_all_known_components(self) -> bool:
        """返回 default、arm、hand 的最终策略是否全部关闭；无副作用。"""

        return not self.enabled_for_component("default") and not (
            self.enabled_for_component("arm") or self.enabled_for_component("hand")
        )


@dataclass(frozen=True)
class RobotContactMaterialSettings:
    """两个物理后端共享的机器人标准 USD 接触材质。

    只要 ``robot.physics.material`` 存在就创建并绑定项目材质；缺失字段使用这里声明的
    项目默认值。PhysX combine mode 不属于本类型，由 ``RobotPhysxSettings`` 单独拥有。
    """

    contact_static_friction: float = 0.8
    contact_dynamic_friction: float = 0.6
    contact_restitution: float = 0.0

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "RobotContactMaterialSettings | None":
        """严格解析可选标准材质；``None`` 表示保留资产原材质。"""

        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_unsupported_keys(
            data,
            {
                "contact_static_friction",
                "contact_dynamic_friction",
                "contact_restitution",
            },
            label,
        )
        values = {
            field_name: (
                getattr(cls(), field_name)
                if field_name not in data
                else _non_negative_float(data[field_name], f"{label}.{field_name}")
            )
            for field_name in (
                "contact_static_friction",
                "contact_dynamic_friction",
                "contact_restitution",
            )
        }
        if values["contact_restitution"] > 1.0:
            raise ValueError(f"{label}.contact_restitution must be between 0 and 1")
        return cls(**values)

    def apply_to_configs(
        self, configs: Mapping[str, RobotUsdOverrideConfig]
    ) -> dict[str, RobotUsdOverrideConfig]:
        """把共享材质叠加到每个部件的通用 USD drive seed。"""

        return {
            name: replace(
                config,
                contact_static_friction=self.contact_static_friction,
                contact_dynamic_friction=self.contact_dynamic_friction,
                contact_restitution=self.contact_restitution,
                contact_material_override=True,
            )
            for name, config in configs.items()
        }


@dataclass(frozen=True)
class RobotPhysxComponentOverride:
    """单个机器人部件的 PhysX combine、刚体阻尼与关节摩擦增量覆盖。"""

    friction_combine_mode: str | None = None
    rigid_body_linear_damping: float | None = None
    rigid_body_angular_damping: float | None = None
    joint_friction: float | None = None
    follower_joint_friction: float | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "RobotPhysxComponentOverride":
        """解析一个 default/arm/hand 分组的 PhysX 增量字段。

        参数:
            data: 可选 ``material``/``rigid_body``/``joint`` mapping。
            label: 当前分组的完整配置路径。
        返回:
            缺失字段保持 ``None`` 的冻结覆盖对象。
        异常:
            ValueError: mapping、字段、数值范围或材料策略不合法。
        """

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_unsupported_keys(data, {"material", "rigid_body", "joint"}, label)
        material = _optional_mapping(data, "material", label) or {}
        rigid_body = _optional_mapping(data, "rigid_body", label) or {}
        joint = _optional_mapping(data, "joint", label) or {}
        _reject_unsupported_keys(
            material,
            {
                "friction_combine_mode",
            },
            f"{label}.material",
        )
        _reject_unsupported_keys(
            rigid_body,
            {"linear_damping", "angular_damping"},
            f"{label}.rigid_body",
        )
        _reject_unsupported_keys(
            joint,
            {"friction", "follower_friction"},
            f"{label}.joint",
        )
        return cls(
            friction_combine_mode=_robot_friction_combine_mode(
                material,
                label=f"{label}.material",
            ),
            rigid_body_linear_damping=_optional_non_negative_float(
                rigid_body, "linear_damping", f"{label}.rigid_body"
            ),
            rigid_body_angular_damping=_optional_non_negative_float(
                rigid_body, "angular_damping", f"{label}.rigid_body"
            ),
            joint_friction=_optional_non_negative_float(
                joint, "friction", f"{label}.joint"
            ),
            follower_joint_friction=_optional_non_negative_float(
                joint, "follower_friction", f"{label}.joint"
            ),
        )

    def merge(
        self, override: "RobotPhysxComponentOverride"
    ) -> "RobotPhysxComponentOverride":
        """用 ``override`` 的显式字段叠加当前对象并返回新实例。

        数值 ``None`` 不覆盖现值；显式 combine mode 与其它数值采用相同的增量语义。

        参数:
            override: 优先级更高的增量覆盖。
        返回:
            合并后的新冻结对象；两个输入均保持不变。
        """

        return RobotPhysxComponentOverride(
            friction_combine_mode=(
                self.friction_combine_mode
                if override.friction_combine_mode is None
                else override.friction_combine_mode
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
            joint_friction=(
                self.joint_friction
                if override.joint_friction is None
                else override.joint_friction
            ),
            follower_joint_friction=(
                self.follower_joint_friction
                if override.follower_joint_friction is None
                else override.follower_joint_friction
            ),
        )

    def apply_to(self, config: RobotUsdOverrideConfig) -> RobotUsdOverrideConfig:
        """把显式字段叠加到完整 USD/PhysX 配置并返回新对象。

        参数:
            config: controller 或通用默认值生成的完整 PhysX 配置。
        返回:
            叠加后的冻结配置；没有显式字段时原样返回 ``config``。
        副作用:
            无；不访问 stage，也不创建材质。
        """

        updates: dict[str, object] = {}
        for field_name in (
            "rigid_body_linear_damping",
            "rigid_body_angular_damping",
            "joint_friction",
            "follower_joint_friction",
        ):
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = value
        if self.friction_combine_mode is not None:
            updates["friction_combine_mode"] = self.friction_combine_mode
        return replace(config, **updates) if updates else config


@dataclass(frozen=True)
class RobotPhysxOverrides:
    """机器人资产级 default/arm/hand PhysX 覆盖集合。

    ``default`` 总是存在并作用于所有已知配置；``arm``、``hand`` 是可选增量，只覆盖结果
    字典中实际存在的对应部件。顶层通用 material/rigid_body/joint 先合入 default，再处理显式
    default 子组，从而保持确定的优先级。
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
        """解析通用默认值与 arm/hand 的增量覆盖。

        参数:
            data: 可选通用字段及 default/arm/hand 子组 mapping。
            label: 完整配置路径。
        返回:
            冻结的分部件覆盖集合；输入 ``None`` 时为空覆盖。
        异常:
            ValueError: 结构、未知字段或任一物理值不合法。
        """

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_unsupported_keys(
            data,
            {"material", "rigid_body", "joint", "default", "arm", "hand"},
            label,
        )
        common = RobotPhysxComponentOverride.from_mapping(
            {
                key: data[key]
                for key in ("material", "rigid_body", "joint")
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
                _required_mapping(data, "arm", label), label=f"{label}.arm"
            )
            if "arm" in data
            else None
        )
        hand = (
            RobotPhysxComponentOverride.from_mapping(
                _required_mapping(data, "hand", label), label=f"{label}.hand"
            )
            if "hand" in data
            else None
        )
        return cls(default=default, arm=arm, hand=hand)

    def apply_to_configs(
        self, configs: dict[str, RobotUsdOverrideConfig]
    ) -> dict[str, RobotUsdOverrideConfig]:
        """叠加资产 profile 覆盖，同时保留 controller 提供的其它参数。

        返回新字典，不修改传入 mapping 或其中的冻结配置；default 先应用到所有条目，
        arm/hand 再应用到同名条目。

        参数:
            configs: ``部件名 -> 完整 PhysX 配置`` mapping。
        返回:
            保留所有原键的全新字典；无文件或 stage 副作用。
        """

        result = {
            name: self.default.apply_to(config) for name, config in configs.items()
        }
        if self.arm is not None and "arm" in result:
            result["arm"] = self.arm.apply_to(result["arm"])
        if self.hand is not None and "hand" in result:
            result["hand"] = self.hand.apply_to(result["hand"])
        return result


@dataclass(frozen=True)
class RobotPhysxSettings:
    """robot.physics.physx 的唯一 typed leaf。

    资产材质、刚体、关节覆盖与 per-body solver iterations 都只由 PhysX composition
    消费。Newton 可以加载同一 robot profile，但不会把本 leaf 投影到 stage。
    """

    overrides: RobotPhysxOverrides = field(default_factory=RobotPhysxOverrides)
    solver_iterations: SolverIterationConfig | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "RobotPhysxSettings":
        """严格解析 PhysX 资产覆盖及 solver 子配置。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_unsupported_keys(
            data,
            {"material", "rigid_body", "joint", "default", "arm", "hand", "solver"},
            label,
        )
        override_keys = {"material", "rigid_body", "joint", "default", "arm", "hand"}
        overrides = RobotPhysxOverrides.from_mapping(
            {key: value for key, value in data.items() if key in override_keys},
            label=label,
        )
        solver = _optional_strict_mapping(data, "solver", label)
        _validate_robot_solver_mapping(solver, label=f"{label}.solver")
        return cls(
            overrides=overrides,
            solver_iterations=robot_solver_settings(solver, label=f"{label}.solver"),
        )


def validate_robot_profile(
    data: Mapping[str, Any],
    *,
    source: str = "<robot profile>",
) -> dict[str, Any]:
    """严格校验一份项目拥有的完整机器人 profile。

    参数:
        data: 包含 robot、cuRobo 绑定和关节分组的完整 mapping。
        source: 错误信息使用的来源标签或文件路径。
    返回:
        顶层已复制的规范字典；字段值不会被自动修正或隐式转换。
    异常:
        ValueError: schema、资产格式、部件分组、受控关节、规划绑定或物理范围不合法；
            错误统一带 ``source`` 前缀。
    副作用:
        无；不读取资产文件或启动后端。

    此边界只验证机器人 profile 自己拥有的 cuRobo 模型绑定。算法参数由所选 mode 的
    唯一 ``curobo`` profile 拥有，并在后续合并阶段校验。
    """

    try:
        if not isinstance(data, Mapping):
            raise ValueError("robot profile must be a mapping")
        canonical = dict(data)
        _reject_unsupported_keys(
            canonical,
            set(_ROBOT_PROFILE_ROOT_KEYS),
            "profile",
        )
        robot = _required_mapping(canonical, "robot", "profile")
        _validate_robot_section(robot)
        kind = robot_kind_from_profile(canonical)
        layout = _validate_robot_component_groups(canonical, kind=kind)
        _validate_controlled_joints(canonical, layout=layout)
        _validate_robot_curobo_section(canonical, kind=kind)
    except ValueError as exc:
        if str(exc).startswith(f"{source}:"):
            raise
        raise ValueError(f"{source}: {exc}") from exc
    return canonical


def _validate_robot_section(robot: Mapping[str, object]) -> None:
    """校验固定 robot/import/physics mapping 及各资产格式的字段所有权。"""

    _reject_unsupported_keys(robot, set(_ROBOT_KEYS), "robot")
    if "kind" not in robot:
        raise ValueError("robot.kind is required")
    _non_empty_string(robot["kind"], "robot.kind")
    RobotKind.parse(robot["kind"])
    if "name" in robot:
        _non_empty_string(robot["name"], "robot.name")
    if "asset_path" not in robot:
        raise ValueError("robot.asset_path is required")
    _non_empty_string(robot["asset_path"], "robot.asset_path")
    asset_type = _non_empty_string(
        robot.get("asset_type", "mjcf"),
        "robot.asset_type",
    ).lower()
    if asset_type not in {"mjcf", "urdf"}:
        raise ValueError("robot.asset_type must be 'mjcf' or 'urdf'")
    if "urdf_drive_type" in robot:
        if asset_type != "urdf":
            raise ValueError("robot.urdf_drive_type is only supported for URDF robots")
        drive_type = _non_empty_string(
            robot["urdf_drive_type"],
            "robot.urdf_drive_type",
        ).lower()
        if drive_type not in {"none", "position"}:
            raise ValueError("robot.urdf_drive_type must be 'none' or 'position'")

    import_settings = _optional_strict_mapping(robot, "import", "robot")
    AssetImportConfig.from_mapping(
        import_settings,
        label="robot.import",
        asset_type=asset_type,
        allow_self_collision=True,
    )
    physics = _optional_strict_mapping(robot, "physics", "robot")
    if physics is not None:
        _reject_unsupported_keys(
            physics,
            {"gravity", "material", "physx"},
            "robot.physics",
        )
        gravity = _optional_strict_mapping(physics, "gravity", "robot.physics")
        RobotGravityPolicy.from_mapping(gravity, label="robot.physics.gravity")
        material = _optional_strict_mapping(physics, "material", "robot.physics")
        RobotContactMaterialSettings.from_mapping(
            material,
            label="robot.physics.material",
        )
        physx = _optional_strict_mapping(physics, "physx", "robot.physics")
        RobotPhysxSettings.from_mapping(physx, label="robot.physics.physx")
    planning_collision = _optional_strict_mapping(
        robot,
        "planning_collision",
        "robot",
    )
    _validate_robot_planning_collision(planning_collision)


def _validate_robot_component_groups(
    data: Mapping[str, object],
    *,
    kind: RobotKind,
) -> JointGroupLayout:
    """校验关节/刚体精确名称分组，并解析机器人类型要求的关节布局。"""

    if "joint_groups" not in data:
        raise ValueError("joint_groups is required")
    joint_groups = _required_mapping(data, "joint_groups", "profile")
    component_mapping = RobotComponentMapping.from_profile(data)
    assert component_mapping.joints is not None
    command_names = (
        *component_mapping.joints.arm,
        *component_mapping.joints.hand,
        *component_mapping.joints.default,
    )
    layout = JointGroupLayout.resolve(
        kind=kind,
        command_joint_names=command_names,
        joint_groups=joint_groups,
    )
    if "rigid_body_groups" in data:
        rigid_body_groups = data["rigid_body_groups"]
        if rigid_body_groups is None:
            raise ValueError("rigid_body_groups must be a mapping")
        RobotComponentMapping.from_profile(data)
    return layout


def _validate_controlled_joints(
    data: Mapping[str, object],
    *,
    layout: JointGroupLayout,
) -> None:
    """校验可选运行时控制关节选择器，且不对名称做字符串强制转换。"""

    _controlled_joint_names(data, layout=layout)


def _controlled_joint_names(
    data: Mapping[str, object],
    *,
    layout: JointGroupLayout,
) -> tuple[str, ...]:
    """返回校验后的主动关节 selector；缺省使用唯一 ``all`` sentinel。"""

    if "controlled_joints" not in data:
        return ("all",)
    value = data["controlled_joints"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("controlled_joints must be a sequence")
    names = tuple(
        _non_empty_string(item, f"controlled_joints[{index}]")
        for index, item in enumerate(value)
    )
    if not names:
        raise ValueError("controlled_joints cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError("controlled_joints contains duplicate names")
    if "all" in {name.lower() for name in names}:
        if len(names) != 1 or names[0].lower() != "all":
            raise ValueError("controlled_joints 'all' must be the only selector")
        return ("all",)
    command_names = set(layout.arm) | set(layout.hand)
    unknown = sorted(set(names) - command_names)
    if unknown:
        raise ValueError(
            f"controlled_joints contains names outside arm/hand groups: {unknown}"
        )
    return names


def _validate_robot_curobo_section(
    data: Mapping[str, object],
    *,
    kind: RobotKind,
) -> None:
    """只校验机器人拥有的 cuRobo 模型字段，不接受算法默认参数。"""

    curobo = _required_mapping(data, "curobo", "profile")
    _reject_unsupported_keys(
        curobo,
        set(_CUROBO_ROBOT_PROFILE_KEYS),
        "curobo",
    )
    binding = PlanningBindingConfig.from_profile(data, kind=kind)
    if not binding.enabled:
        disabled_extras = set(curobo) - {"enabled"}
        if disabled_extras:
            paths = ", ".join(f"curobo.{key}" for key in sorted(disabled_extras))
            raise ValueError(
                f"disabled cuRobo binding cannot declare model fields: {paths}"
            )
        return
    if not isinstance(curobo.get("planning_joint_group"), str):
        raise ValueError("curobo.planning_joint_group must be a string")
    robot_model = _required_mapping(curobo, "robot", "curobo")
    parsed = CuroboRobotConfig.from_mapping(robot_model)
    if len(parsed.tool_frames) != len(set(parsed.tool_frames)):
        raise ValueError("curobo.robot.tool_frames contains duplicate names")


def _validate_robot_solver_mapping(
    solver: Mapping[str, object] | None,
    *,
    label: str,
) -> None:
    """在通用 solver 解析前拒绝 bool/string 到迭代次数的隐式转换。"""

    if solver is None:
        return
    _reject_unsupported_keys(solver, {"arm", "hand"}, label)
    for component in ("arm", "hand"):
        if component not in solver:
            continue
        group = solver[component]
        if not isinstance(group, Mapping):
            raise ValueError(f"{label}.{component} must be a mapping")
        _reject_unsupported_keys(
            group,
            {"position_iterations", "velocity_iterations"},
            f"{label}.{component}",
        )
        for key in ("position_iterations", "velocity_iterations"):
            if key in group:
                _non_negative_integer(
                    group[key],
                    f"{label}.{component}.{key}",
                )


def _validate_robot_planning_collision(
    data: Mapping[str, object] | None,
) -> None:
    """复用 typed parser 校验机器人根坐标系保守球体包络。"""

    RobotPlanningCollisionSettings.from_mapping(data)


def _optional_strict_mapping(
    data: Mapping[str, object],
    key: str,
    parent_label: str,
) -> Mapping[str, object] | None:
    """在当前 schema 边界读取可选但不允许显式 null 的 mapping。"""

    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _non_empty_string(value: object, label: str) -> str:
    """严格解析非空字符串，不把其它标量强制转换为文本。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _absolute_prim_path(value: object, label: str) -> str:
    """校验不含空段、尾斜杠且不是根节点的规范绝对 USD prim 路径。"""

    path = _non_empty_string(value, label)
    if not path.startswith("/") or path == "/" or path.endswith("/") or "//" in path:
        raise ValueError(f"{label} must be a canonical absolute USD path")
    return path


def _non_negative_integer(value: object, label: str) -> int:
    """严格解析非负整数，并显式排除 Python 的 bool 子类型。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _finite_number(value: object, label: str) -> float:
    """严格解析有限实数，并显式排除 bool。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _finite_vector3(value: object, label: str) -> tuple[float, float, float]:
    """严格解析恰含三个有限实数的向量。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of 3 numbers")
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 numbers")
    parsed = tuple(
        _finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    return parsed  # type: ignore[return-value]


def _optional_controller_profile(
    data: Mapping[str, object], key: str, parent_label: str
) -> str | None:
    """读取可选 controller bundle 名；null 与缺失表示由运行 profile 统一选择。"""

    value = data.get(key)
    if value is None:
        return None
    return normalize_controller_bundle_name(value, label=f"{parent_label}.{key}")


def normalize_robot_controller_profile(
    robot: Mapping[str, object],
) -> str | None:
    """读取 profile 的可选 controller bundle，供资产投影消费。"""

    return _optional_controller_profile(robot, "controller_profile", "robot")


def _normalize_collision_approximation(value: object, *, label: str) -> str:
    """规范化 Isaac collision approximation，并限制在已验证的两种模式。"""

    normalized = str(value).strip()
    if normalized in SUPPORTED_COLLISION_APPROXIMATIONS:
        return normalized
    supported = ", ".join(SUPPORTED_COLLISION_APPROXIMATIONS)
    raise ValueError(f"{label} must be one of {supported}, got {value!r}")


def _required_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object]:
    """读取必需 mapping，并在错误中保留完整配置路径。"""

    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _optional_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object] | None:
    """读取可选 mapping；字段缺失/null 时返回 None。"""

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _nested_mapping(
    parent: Mapping[str, object] | None,
    key: str,
    parent_label: str,
) -> Mapping[str, object] | None:
    """读取可选父 mapping 的子 mapping，避免各 typed parser 重复分支。"""

    return None if parent is None else _optional_mapping(parent, key, parent_label)


def _optional_bool(
    data: Mapping[str, object], key: str, parent_label: str
) -> bool | None:
    """严格读取可选 bool，不把 string/number 当作 truthy value。"""

    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"{parent_label}.{key} must be a boolean")
    return value


def _optional_nullable_bool(
    data: Mapping[str, object], key: str, parent_label: str
) -> bool | None:
    """读取 nullable bool；缺失或 null 表示保留消费方默认行为。"""

    if key not in data or data[key] is None:
        return None
    return _optional_bool(data, key, parent_label)


def _bool_with_default(
    data: Mapping[str, object],
    key: str,
    parent_label: str,
    *,
    default: bool,
) -> bool:
    """严格读取 bool，字段缺失时使用当前 schema 声明的默认值。"""

    value = _optional_bool(data, key, parent_label)
    return default if value is None else value


def _optional_non_negative_float(
    data: Mapping[str, object], key: str, parent_label: str
) -> float | None:
    """读取可选非负 float；字段缺失时保留 None 语义。"""

    if key not in data:
        return None
    return _non_negative_float(data[key], f"{parent_label}.{key}")


def _non_negative_float(value: object, label: str) -> float:
    """严格读取一个有限非负浮点数。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _robot_friction_combine_mode(
    material: Mapping[str, object], *, label: str
) -> str | None:
    """解析可选 PhysX combine 枚举；缺失或 null 表示不覆盖。"""

    if "friction_combine_mode" not in material:
        return None
    value = material["friction_combine_mode"]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label}.friction_combine_mode must be a string or null")
    normalized = value.lower()
    allowed = {"average", "min", "multiply", "max"}
    if normalized not in allowed:
        raise ValueError(
            f"{label}.friction_combine_mode must be one of {sorted(allowed)}, "
            f"or null, got {value!r}"
        )
    return normalized


def _reject_unsupported_keys(
    data: Mapping[str, object], allowed: set[str], label: str
) -> None:
    """拒绝 profile 未声明 key，防止拼写错误被静默忽略。"""

    unsupported_keys = sorted(str(key) for key in data if key not in allowed)
    if unsupported_keys:
        unsupported = ", ".join(sorted(unsupported_keys))
        paths = ", ".join(f"{label}.{key}" for key in unsupported_keys)
        raise ValueError(
            f"{label} contains unsupported keys: {unsupported} (full paths: {paths})"
        )


__all__ = [
    "AssetImportConfig",
    "RobotContactMaterialSettings",
    "RobotCuroboSettings",
    "RobotGravityPolicy",
    "RobotPlanningCollisionSettings",
    "RobotPlanningCollisionSphereSettings",
    "RobotProfileSettings",
    "RobotPhysxComponentOverride",
    "RobotPhysxOverrides",
    "RobotPhysxSettings",
    "normalize_robot_controller_profile",
    "validate_robot_profile",
]
