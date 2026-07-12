"""机器人资产导入与 PhysX 覆盖配置。

本模块只解析后端无关的 YAML 数据，不导入 Isaac/Omni。调用方可以在普通 Python
环境中加载并校验机器人 profile；真正修改 USD stage 的操作由 ``robot_import`` 和
``root_pose`` 模块负责。

职责分为两层：``validate_robot_profile`` 校验项目拥有的完整机器人 schema 与能力绑定，
``RobotAssetConfig.from_mapping`` 则结合场景实例的 prim path，提取 importer、重力、solver
和分部件 PhysX 覆盖。所有 dataclass 均冻结；解析不会创建资产、材质或 articulation。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from numbers import Real
from pathlib import Path
from typing import Any

from linkerbot_sim.assets.solver_overrides import (
    SolverIterationConfig,
    robot_solver_settings,
)
from linkerbot_sim.assets.usd_overrides import PhysxOverrideConfig
from linkerbot_sim.controllers.config import normalize_controller_bundle_name
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
from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.paths import repo_path


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
class AssetImportConfig:
    """非 USD 资产 importer 的格式相关选项。

    ``collision_approximation`` 只接受当前支持的 convex decomposition/hull；
    ``self_collision`` 仅在调用边界明确允许时可配置。``fix_base`` 与
    ``merge_fixed_joints`` 为 ``None`` 时保留具体 importer 的默认行为；其余布尔字段具有
    项目级明确默认值。字段可用性取决于 MJCF/URDF 格式，并在构造时严格校验。
    """

    collision_approximation: str = DEFAULT_COLLISION_APPROXIMATION
    self_collision: bool = False
    fix_base: bool | None = None
    merge_fixed_joints: bool | None = None
    collision_from_visuals: bool = False
    import_inertia_tensor: bool = True
    import_sites: bool = True

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
            "import_inertia_tensor",
        }
        if normalized_asset_type == "mjcf":
            supported_keys.update({"import_sites", "merge_fixed_joints"})
        elif normalized_asset_type == "urdf":
            supported_keys.update({"merge_fixed_joints", "collision_from_visuals"})
        else:
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
            import_inertia_tensor=_bool_with_default(
                data,
                "import_inertia_tensor",
                label,
                default=True,
            ),
            import_sites=_bool_with_default(
                data,
                "import_sites",
                label,
                default=True,
            ),
        )

    def use_convex_decomposition(self) -> bool:
        """返回 importer 布尔 API 是否应启用 convex decomposition。

        该查询不修改配置，也不调用 importer。
        """

        return self.collision_approximation == "convex_decomposition"


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
class RobotPhysxComponentOverride:
    """单个机器人部件的接触材料与刚体阻尼增量覆盖。

    数值字段为 ``None`` 时不改写上游 controller/USD 默认值。``contact_material_policy``
    区分 ``inherit``（本层不表态）、``preserve``（明确保留资产材质）和 ``override``
    （创建/更新项目材质）；只有 override 才消费 ``friction_combine_mode``。所有显式物理量
    均为有限非负数，实例冻结后可安全叠加。
    """

    contact_static_friction: float | None = None
    contact_dynamic_friction: float | None = None
    contact_restitution: float | None = None
    contact_material_policy: str = "inherit"
    friction_combine_mode: str | None = None
    rigid_body_linear_damping: float | None = None
    rigid_body_angular_damping: float | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None, *, label: str
    ) -> "RobotPhysxComponentOverride":
        """解析一个 default/arm/hand 分组的 PhysX 增量字段。

        参数:
            data: 可选 ``material``/``rigid_body`` mapping。
            label: 当前分组的完整配置路径。
        返回:
            缺失字段保持 ``None`` 或 ``inherit`` 的冻结覆盖对象。
        异常:
            ValueError: mapping、字段、数值范围或材料策略不合法。
        """

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_unsupported_keys(data, {"material", "rigid_body"}, label)
        material, material_policy = _robot_material_mapping(data, label=label)
        rigid_body = _optional_mapping(data, "rigid_body", label) or {}
        _reject_unsupported_keys(
            material,
            {
                "contact_static_friction",
                "contact_dynamic_friction",
                "contact_restitution",
                "friction_combine_mode",
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
            contact_material_policy=material_policy,
            friction_combine_mode=_robot_friction_combine_mode(
                material,
                label=f"{label}.material",
                material_policy=material_policy,
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
        """用 ``override`` 的显式字段叠加当前对象并返回新实例。

        数值 ``None`` 和材料 ``inherit`` 不覆盖现值；``preserve``/``override`` 是完整材料
        决策，会连同 combine mode 一起替换，避免残留上一层材料语义。

        参数:
            override: 优先级更高的增量覆盖。
        返回:
            合并后的新冻结对象；两个输入均保持不变。
        """

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
            contact_material_policy=(
                self.contact_material_policy
                if override.contact_material_policy == "inherit"
                else override.contact_material_policy
            ),
            friction_combine_mode=(
                self.friction_combine_mode
                if override.contact_material_policy == "inherit"
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
        )

    def apply_to(self, config: PhysxOverrideConfig) -> PhysxOverrideConfig:
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
            "contact_static_friction",
            "contact_dynamic_friction",
            "contact_restitution",
            "rigid_body_linear_damping",
            "rigid_body_angular_damping",
        ):
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = value
        if self.contact_material_policy == "preserve":
            updates["contact_material_override"] = False
        elif self.contact_material_policy == "override":
            updates["contact_material_override"] = True
            updates["friction_combine_mode"] = self.friction_combine_mode
        return replace(config, **updates) if updates else config


@dataclass(frozen=True)
class RobotPhysxOverrides:
    """机器人资产级 default/arm/hand PhysX 覆盖集合。

    ``default`` 总是存在并作用于所有已知配置；``arm``、``hand`` 是可选增量，只覆盖结果
    字典中实际存在的对应部件。顶层通用 material/rigid_body 先合入 default，再处理显式
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
            data, {"material", "rigid_body", "default", "arm", "hand"}, label
        )
        common = RobotPhysxComponentOverride.from_mapping(
            {key: data[key] for key in ("material", "rigid_body") if key in data},
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
        self, configs: dict[str, PhysxOverrideConfig]
    ) -> dict[str, PhysxOverrideConfig]:
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
class RobotAssetConfig:
    """单个场景机器人实例的资产与物理配置。

    ``asset_type``/``asset_path`` 定位项目资产，``prim_path`` 是该场景实例拥有的绝对
    USD 子树根；``name`` 是实例显示名，``controller_profile`` 可覆盖运行时默认 bundle。
    其余字段聚合格式相关 importer 选项、URDF drive 类型、重力、PhysX、solver 及部件映射。
    路径在构造时解析并验证，但文件存在性和实际导入由资产层负责。
    """

    asset_type: str
    asset_path: Path
    prim_path: str
    name: str = "robot"
    controller_profile: str | None = None
    urdf_drive_type: str = "position"
    import_config: AssetImportConfig = field(default_factory=AssetImportConfig)
    gravity_policy: RobotGravityPolicy = field(default_factory=RobotGravityPolicy)
    physx_overrides: RobotPhysxOverrides = field(default_factory=RobotPhysxOverrides)
    solver_iterations: SolverIterationConfig | None = None
    component_mapping: RobotComponentMapping = field(
        default_factory=RobotComponentMapping
    )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        prim_path: str,
        name: str | None = None,
    ) -> "RobotAssetConfig":
        """从机器人 profile 和场景实例信息解析资产配置。

        参数:
            data: 至少包含严格 ``robot`` section 的完整 profile mapping。
            prim_path: 场景实例拥有的绝对 USD prim 路径。
            name: 可选实例名；提供时覆盖 profile 中的机器人名称。
        返回:
            冻结的 :class:`RobotAssetConfig`，资产路径已按仓库规则解析。
        异常:
            ValueError: 缺少 robot section，路径、字段、importer 或物理配置不合法。
        副作用:
            无；不会确认资产存在，也不会修改 USD stage。
        """

        if "robot" not in data:
            raise ValueError("Robot config must contain top-level robot section")
        robot = _required_mapping(data, "robot", "profile")
        _reject_unsupported_keys(robot, set(_ROBOT_KEYS), "robot")
        physics = _optional_mapping(robot, "physics", "robot")
        asset_type = str(robot.get("asset_type", "mjcf")).lower()
        asset_path = robot.get("asset_path")
        if not isinstance(asset_path, (str, Path)):
            raise ValueError("robot.asset_path must be a path string")
        return cls(
            asset_type=asset_type,
            asset_path=repo_path(asset_path),
            prim_path=_absolute_prim_path(prim_path, "robot instance prim_path"),
            name=str(robot.get("name", "robot")) if name is None else name,
            controller_profile=_optional_controller_profile(
                robot, "controller_profile", "robot"
            ),
            urdf_drive_type=str(robot.get("urdf_drive_type", "position")),
            import_config=AssetImportConfig.from_mapping(
                _optional_mapping(robot, "import", "robot"),
                label="robot.import",
                asset_type=asset_type,
                allow_self_collision=True,
            ),
            gravity_policy=RobotGravityPolicy.from_mapping(
                (
                    _optional_mapping(physics, "gravity", "robot.physics")
                    if physics is not None
                    else None
                ),
                label="robot.physics.gravity",
            ),
            physx_overrides=RobotPhysxOverrides.from_mapping(
                (
                    _optional_mapping(physics, "physx", "robot.physics")
                    if physics is not None
                    else None
                ),
                label="robot.physics.physx",
            ),
            solver_iterations=robot_solver_settings(
                (
                    _optional_mapping(physics, "solver", "robot.physics")
                    if physics is not None
                    else None
                ),
                label="robot.physics.solver",
            ),
            component_mapping=RobotComponentMapping.from_profile(data),
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

    此边界只验证机器人 profile 自己拥有的 cuRobo 模型绑定。算法参数仍由所选
    ``configs/curobo`` profile 拥有，并在后续合并阶段校验。
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
        _validate_robot_physx_ranges(robot)
    except ValueError as exc:
        if str(exc).startswith(f"{source}:"):
            raise
        raise ValueError(f"{source}: {exc}") from exc
    return canonical


def load_robot_profile(path: str | Path) -> dict[str, Any]:
    """读取一个机器人 YAML 文件并返回严格校验后的 mapping。

    参数:
        path: profile 文件路径。
    返回:
        :func:`validate_robot_profile` 返回的规范字典。
    异常:
        FileNotFoundError/OSError: 文件不存在或无法读取。
        ValueError: YAML 或机器人 schema 不合法，错误中包含文件路径。
    副作用:
        仅读取文件，不修改配置或资产。
    """

    profile_path = Path(path)
    return validate_robot_profile(load_yaml(profile_path), source=str(profile_path))


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
            {"gravity", "physx", "solver"},
            "robot.physics",
        )
        gravity = _optional_strict_mapping(physics, "gravity", "robot.physics")
        RobotGravityPolicy.from_mapping(gravity, label="robot.physics.gravity")
        physx = _optional_strict_mapping(physics, "physx", "robot.physics")
        RobotPhysxOverrides.from_mapping(physx, label="robot.physics.physx")
        solver = _optional_strict_mapping(physics, "solver", "robot.physics")
        _validate_robot_solver_mapping(solver)
        robot_solver_settings(solver, label="robot.physics.solver")
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

    if "controlled_joints" not in data:
        return
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
        return
    command_names = set(layout.arm) | set(layout.hand)
    unknown = sorted(set(names) - command_names)
    if unknown:
        raise ValueError(
            f"controlled_joints contains names outside arm/hand groups: {unknown}"
        )


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
    from linkerbot_sim.backends.curobo.config import CuroboRobotConfig

    parsed = CuroboRobotConfig.from_mapping(robot_model)
    if len(parsed.tool_frames) != len(set(parsed.tool_frames)):
        raise ValueError("curobo.robot.tool_frames contains duplicate names")


def _validate_robot_solver_mapping(
    solver: Mapping[str, object] | None,
) -> None:
    """在通用 solver 解析前拒绝 bool/string 到迭代次数的隐式转换。"""

    if solver is None:
        return
    _reject_unsupported_keys(solver, {"arm", "hand"}, "robot.physics.solver")
    for component in ("arm", "hand"):
        if component not in solver:
            continue
        group = solver[component]
        if not isinstance(group, Mapping):
            raise ValueError(f"robot.physics.solver.{component} must be a mapping")
        _reject_unsupported_keys(
            group,
            {"position_iterations", "velocity_iterations"},
            f"robot.physics.solver.{component}",
        )
        for key in ("position_iterations", "velocity_iterations"):
            if key in group:
                _non_negative_integer(
                    group[key],
                    f"robot.physics.solver.{component}.{key}",
                )


def _validate_robot_planning_collision(
    data: Mapping[str, object] | None,
) -> None:
    """校验规划使用的机器人根坐标系保守球体包络。"""

    if data is None:
        return
    _reject_unsupported_keys(data, {"spheres"}, "robot.planning_collision")
    spheres = data.get("spheres")
    if not isinstance(spheres, Sequence) or isinstance(spheres, (str, bytes)):
        raise ValueError("robot.planning_collision.spheres must be a sequence")
    if not spheres:
        raise ValueError("robot.planning_collision.spheres cannot be empty")
    names: list[str] = []
    for index, item in enumerate(spheres):
        label = f"robot.planning_collision.spheres[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_unsupported_keys(item, {"name", "center", "radius"}, label)
        name = _non_empty_string(item.get("name", f"sphere_{index}"), f"{label}.name")
        names.append(name)
        _finite_vector3(item.get("center"), f"{label}.center")
        radius = _finite_number(item.get("radius"), f"{label}.radius")
        if radius <= 0.0:
            raise ValueError(f"{label}.radius must be positive")
    if len(names) != len(set(names)):
        raise ValueError("robot.planning_collision.spheres contains duplicate names")


def _validate_robot_physx_ranges(robot: Mapping[str, object]) -> None:
    """补充通用非负数解析器无法表达的 restitution 物理上界。"""

    physics = robot.get("physics")
    if not isinstance(physics, Mapping):
        return
    physx = physics.get("physx")
    if not isinstance(physx, Mapping):
        return
    for component in (None, "default", "arm", "hand"):
        component_data: object = physx if component is None else physx.get(component)
        if not isinstance(component_data, Mapping):
            continue
        material = component_data.get("material")
        if not isinstance(material, Mapping) or "contact_restitution" not in material:
            continue
        label = "robot.physics.physx"
        if component is not None:
            label = f"{label}.{component}"
        restitution = _finite_number(
            material["contact_restitution"],
            f"{label}.material.contact_restitution",
        )
        if restitution > 1.0:
            raise ValueError(
                f"{label}.material.contact_restitution must be between 0 and 1"
            )


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
    raw_value = data[key]
    if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
        raise ValueError(f"{parent_label}.{key} must be a number")
    value = float(raw_value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{parent_label}.{key} must be finite and non-negative")
    return value


def _robot_material_mapping(
    data: Mapping[str, object], *, label: str
) -> tuple[Mapping[str, object], str]:
    """解析材料 inherit/preserve/override，并区分字段缺失与显式 null。"""

    if "material" not in data:
        return {}, "inherit"
    value = data["material"]
    if value is None or (isinstance(value, str) and value.lower() == "preserve"):
        return {}, "preserve"
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.material must be a mapping, null, or 'preserve'")
    return value, "override"


def _robot_friction_combine_mode(
    material: Mapping[str, object], *, label: str, material_policy: str
) -> str | None:
    """解析 PhysX combine 枚举；null/preserve 表示保留材质 API 默认值。"""

    if material_policy != "override":
        return None
    value = material.get("friction_combine_mode", "average")
    if value is None or (isinstance(value, str) and value.lower() == "preserve"):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label}.friction_combine_mode must be a string or null")
    normalized = value.lower()
    allowed = {"average", "min", "multiply", "max"}
    if normalized not in allowed:
        raise ValueError(
            f"{label}.friction_combine_mode must be one of {sorted(allowed)}, "
            f"null, or 'preserve', got {value!r}"
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
    "RobotAssetConfig",
    "RobotGravityPolicy",
    "RobotPhysxComponentOverride",
    "RobotPhysxOverrides",
    "load_robot_profile",
    "validate_robot_profile",
]
