"""Isaac Lab 风格 tiled envs 的纯 Python 配置模型。

本模块只负责解析 env profile 里的 ``tiled`` 顶层分组，并做不依赖 Isaac/Omni
运行时的结构校验。这样 CLI、单元测试和后续下游 evaluator 可以先验证配置是否合理，
而不会因为 import 本包就启动 Isaac Sim。

设计约定:
    * tiled 配置只由 TiledSceneRuntime 消费；SingleSceneRuntime 不读取该分组。
    * 所有 USD path 都必须是绝对路径，避免后续 clone/filter collision 时路径语义含混。
    * tiled env 数量只来自 profile YAML 中的 ``tiled.num_envs``；CLI/调用方不再覆盖。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite

from linkerbot_sim.assets.root_pose import RootPoseConfig


_ENV_PROFILE_ROOT_KEYS = frozenset(
    {
        "env",
        "solver",
        "visuals",
        "sensors",
        "robots",
        "objects",
        "tiled",
    }
)


@dataclass(frozen=True)
class TiledLayoutConfig:
    """控制 tiled env 网格在 world 中的平移原点。

    env root rotation 暂不支持；未知字段会在解析时被拒绝，避免 YAML 看似生效但下游
    world/local 坐标换算仍按无旋转处理。
    """

    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "TiledLayoutConfig":
        """从 ``tiled.layout`` mapping 解析 world origin translation。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("tiled.layout must be a mapping")
        _reject_keys(data, {"origin_xyz"}, "tiled.layout")
        return cls(
            origin_xyz=_optional_vec3(
                data,
                "origin_xyz",
                cls.origin_xyz,
                "tiled.layout",
            )
        )


@dataclass(frozen=True)
class TiledPerEnvConfig:
    """单个 tiled env 的差异配置。

    TiledSceneRuntime 要求所有 env 拥有相同机器人和物体集合；这里仅允许按 env
    覆盖已有机器人/物体的 ``root_pose`` 和已有相机的 ``pose``。这样仍然可以使用
    ``GridCloner`` 复制同构 stage，再在 clone 后把每个 env 的对象和传感器移动到自己
    的局部位置。
    """

    env_id: int
    robot_root_poses: dict[str, RootPoseConfig] = field(default_factory=dict)
    object_root_poses: dict[str, RootPoseConfig] = field(default_factory=dict)
    camera_poses: dict[str, RootPoseConfig] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object], *, index: int
    ) -> "TiledPerEnvConfig":
        """解析 ``tiled.per_env[index]``。

        支持的 YAML 形态:

        ``objects.<object_name>.root_pose``:
            按对象名覆盖位姿。
        ``robots.<robot_name>.root_pose``:
            按机器人 label 覆盖 env-local 根位姿。
        ``cameras.<camera_name>.pose``:
            按相机名覆盖该 env 中传感器相机的局部位姿。
        """

        if not isinstance(data, Mapping):
            raise ValueError(f"tiled.per_env[{index}] must be a mapping")
        _reject_keys(
            data,
            {"env_id", "robots", "objects", "cameras", "metadata"},
            f"tiled.per_env[{index}]",
        )
        if "env_id" not in data:
            raise ValueError(f"tiled.per_env[{index}].env_id is required")
        env_id = _integer(data["env_id"], f"tiled.per_env[{index}].env_id")
        metadata = _metadata_mapping(
            data.get("metadata"), label=f"tiled.per_env[{index}].metadata"
        )
        return cls(
            env_id=env_id,
            robot_root_poses=_parse_robot_root_pose_overrides(
                data.get("robots"), label=f"tiled.per_env[{index}].robots"
            ),
            object_root_poses=_parse_object_root_pose_overrides(
                data.get("objects"), label=f"tiled.per_env[{index}].objects"
            ),
            camera_poses=_parse_camera_pose_overrides(
                data.get("cameras"), label=f"tiled.per_env[{index}].cameras"
            ),
            metadata=metadata,
        )


@dataclass(frozen=True)
class TiledCloneConfig:
    """控制 USD/PhysX 克隆行为的配置。

    tiled scene builder 固定使用 Isaac ``GridCloner``；这里仅保留实际可配置的
    clone 参数。
    """

    # replicate_physics 让 PhysX 复用克隆环境的物理结构，是 256+ env 性能路径的关键。
    replicate_physics: bool = True
    # copy_from_source=false 表示 clone 可以继承 source prim，减少 USD stage 冗余。
    copy_from_source: bool = False
    # enable_env_ids 用于 colocated env；当前布局通过 spacing 分隔 env，因此默认关闭。
    enable_env_ids: bool = False
    # 开启 env 间 collision filtering，避免相邻 tiled env 发生跨环境接触。
    filter_collisions: bool = True
    # 过滤策略：
    #   collision_groups —— 每个 env 一个 UsdPhysics.CollisionGroup，配合 scene 级
    #     invertCollisionGroupFilter 把 filteredGroups 语义变成白名单，env 只与自身和
    #     global(地面)碰撞。authoring 成本 O(E)，是 256+ env 的推荐路径。
    #   filtered_pairs —— 逐 prim 两两 FilteredPairsAPI，O(E²)，用于无法采用
    #     collision_groups 的运行环境。
    collision_filter_strategy: str = "collision_groups"
    # collision group / filtered-pair 根路径。
    collision_root_path: str = "/World/collisions"
    # null 表示从 stage 自动发现唯一 UsdPhysics.Scene；显式 path 用于多 scene stage。
    physics_scene_path: str | None = None
    # auto 扫描项目标准 ground prim；显式列表用于完全接管 global collider collection。
    global_collision_paths: str | tuple[str, ...] = "auto"
    # 在 auto 或显式列表之后追加 custom ground/fixture，并稳定去重。
    extra_global_collision_paths: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "TiledCloneConfig":
        """从 ``tiled.clone`` mapping 解析配置。

        未提供该分组时使用适合性能路径的默认值。未知 key 会被拒绝，避免 YAML
        拼写错误被静默忽略。
        """

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("tiled.clone must be a mapping")
        _reject_keys(
            data,
            {
                "replicate_physics",
                "copy_from_source",
                "enable_env_ids",
                "filter_collisions",
                "collision_filter_strategy",
                "collision_root_path",
                "physics_scene_path",
                "global_collision_paths",
                "extra_global_collision_paths",
            },
            "tiled.clone",
        )
        config = cls(
            replicate_physics=_optional_bool(
                data, "replicate_physics", cls.replicate_physics, "tiled.clone"
            ),
            copy_from_source=_optional_bool(
                data, "copy_from_source", cls.copy_from_source, "tiled.clone"
            ),
            enable_env_ids=_optional_bool(
                data, "enable_env_ids", cls.enable_env_ids, "tiled.clone"
            ),
            filter_collisions=_optional_bool(
                data, "filter_collisions", cls.filter_collisions, "tiled.clone"
            ),
            collision_filter_strategy=_normalize_collision_filter_strategy(
                _optional_non_empty_str(
                    data,
                    "collision_filter_strategy",
                    cls.collision_filter_strategy,
                    "tiled.clone",
                )
            ),
            collision_root_path=_optional_path(
                data,
                "collision_root_path",
                cls.collision_root_path,
                "tiled.clone",
            ),
            physics_scene_path=_optional_path_or_none(
                data,
                "physics_scene_path",
                "tiled.clone",
            ),
            global_collision_paths=_optional_usd_paths_or_auto(
                data,
                "global_collision_paths",
                "tiled.clone",
            ),
            extra_global_collision_paths=_optional_usd_path_tuple(
                data,
                "extra_global_collision_paths",
                (),
                "tiled.clone",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """校验 clone 配置中和 USD stage 相关的字段。"""

        _require_absolute_usd_path(
            self.collision_root_path, "tiled.clone.collision_root_path"
        )
        if self.collision_root_path == "/":
            raise ValueError("tiled.clone.collision_root_path cannot be '/'")
        if self.physics_scene_path is not None:
            _require_absolute_usd_path(
                self.physics_scene_path, "tiled.clone.physics_scene_path"
            )
        if self.global_collision_paths != "auto":
            for index, path in enumerate(self.global_collision_paths):
                _require_absolute_usd_path(
                    path, f"tiled.clone.global_collision_paths[{index}]"
                )
        for index, path in enumerate(self.extra_global_collision_paths):
            _require_absolute_usd_path(
                path, f"tiled.clone.extra_global_collision_paths[{index}]"
            )
        group_only_fields: list[str] = []
        if self.collision_root_path != type(self).collision_root_path:
            group_only_fields.append("collision_root_path")
        if self.physics_scene_path is not None:
            group_only_fields.append("physics_scene_path")
        if self.global_collision_paths != "auto":
            group_only_fields.append("global_collision_paths")
        if self.extra_global_collision_paths:
            group_only_fields.append("extra_global_collision_paths")
        if group_only_fields and (
            not self.filter_collisions
            or self.collision_filter_strategy != "collision_groups"
        ):
            names = ", ".join(group_only_fields)
            raise ValueError(
                f"tiled.clone fields {names} require filter_collisions=true and "
                "collision_filter_strategy=collision_groups"
            )


@dataclass(frozen=True)
class TiledDiagnosticsConfig:
    """选择 status/debug 中重点展示的 tiled env。"""

    # GUI/telemetry 调试时通常只需要看少数 env。这里记录想重点观察的 env id，
    # 不为每个 env 创建 viewport，也不减少物理或渲染开销。
    inspect_env_ids: tuple[int, ...] = (0,)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None
    ) -> "TiledDiagnosticsConfig":
        """从 ``tiled.diagnostics`` mapping 解析配置。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("tiled.diagnostics must be a mapping")
        _reject_keys(data, {"inspect_env_ids"}, "tiled.diagnostics")
        return cls(
            inspect_env_ids=_optional_int_tuple(
                data, "inspect_env_ids", cls.inspect_env_ids, "tiled.diagnostics"
            ),
        )

    def validate(self, *, num_envs: int) -> None:
        """校验 runtime 配置是否和 ``num_envs`` 匹配。"""

        if any(env_id < 0 or env_id >= num_envs for env_id in self.inspect_env_ids):
            raise ValueError(
                "tiled.diagnostics.inspect_env_ids contains out-of-range env id"
            )


@dataclass(frozen=True)
class TiledEnvConfig:
    """env profile 中 ``tiled`` 顶层配置的完整结果。

    ``enabled=False, num_envs=1`` 表示保持单 env 语义。省略 ``tiled`` 分组时，本类仍返回
    合法默认值，方便上层统一处理。
    """

    # enabled 只选择 TiledSceneRuntime 入口，不改变其它 runtime 的配置语义。
    enabled: bool = False
    # 并行环境数量。所有 batched 数组第一维都应等于该值。
    num_envs: int = 1
    # 所有 env root 的父路径，例如 /World/envs/env_0。
    base_env_path: str = "/World/envs"
    # 子 env 的命名前缀，最终路径形如 {base_env_path}/{env_prefix}_{env_id}。
    env_prefix: str = "env"
    # env root 之间的网格间距，必须大于机器人/物体可能活动范围，避免物理相互干扰。
    spacing: float = 2.0
    # 每行 env 数量；未配置时用接近正方形的紧凑网格。
    num_per_row: int | None = None
    # 目录型 env profile 中保存 per-env YAML 的相对目录；仅配置加载层使用。
    per_env_config_dir: str | None = None
    # 每个 env 的差异配置；仅允许覆盖已有机器人/物体的 root_pose 和相机 pose。
    per_env: tuple[TiledPerEnvConfig, ...] = ()
    layout: TiledLayoutConfig = field(default_factory=TiledLayoutConfig)
    clone: TiledCloneConfig = field(default_factory=TiledCloneConfig)
    diagnostics: TiledDiagnosticsConfig = field(default_factory=TiledDiagnosticsConfig)

    @classmethod
    def from_env_config(
        cls,
        env_config: Mapping[str, object],
    ) -> "TiledEnvConfig":
        """从完整 env profile 解析可选的 ``tiled`` 分组。

        env 数量以 YAML 中的 ``tiled.num_envs`` 为唯一来源；没有 ``tiled`` 分组时
        返回 disabled/single-env 默认配置。
        """

        if not isinstance(env_config, Mapping):
            raise ValueError("env profile must be a mapping")
        _reject_keys(env_config, _ENV_PROFILE_ROOT_KEYS, "env profile")
        if "tiled" not in env_config:
            config = cls()
            config.validate()
            return config
        data = env_config["tiled"]
        if not isinstance(data, Mapping):
            raise ValueError("tiled must be a mapping")
        _reject_keys(
            data,
            {
                "enabled",
                "num_envs",
                "base_env_path",
                "env_prefix",
                "spacing",
                "num_per_row",
                "per_env_config_dir",
                "per_env",
                "layout",
                "clone",
                "diagnostics",
            },
            "tiled",
        )
        num_envs = _positive_int(data.get("num_envs", cls.num_envs), "tiled.num_envs")
        enabled = _optional_bool(data, "enabled", cls.enabled, "tiled")
        config = cls(
            enabled=enabled,
            num_envs=num_envs,
            base_env_path=_optional_path(
                data, "base_env_path", cls.base_env_path, "tiled"
            ),
            env_prefix=_optional_non_empty_str(
                data, "env_prefix", cls.env_prefix, "tiled"
            ),
            spacing=_positive_float(data.get("spacing", cls.spacing), "tiled.spacing"),
            num_per_row=_optional_positive_int_or_none(data, "num_per_row", "tiled"),
            per_env_config_dir=_optional_relative_dir_or_none(
                data, "per_env_config_dir", "tiled"
            ),
            per_env=_parse_per_env_configs(
                data.get("per_env"),
                num_envs=num_envs,
            ),
            layout=TiledLayoutConfig.from_mapping(_optional_mapping(data, "layout")),
            clone=TiledCloneConfig.from_mapping(_optional_mapping(data, "clone")),
            diagnostics=TiledDiagnosticsConfig.from_mapping(
                _optional_mapping(data, "diagnostics")
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """执行跨字段校验，尽早暴露配置错误。"""

        if self.num_envs < 1:
            raise ValueError("tiled.num_envs must be >= 1")
        _require_absolute_usd_path(self.base_env_path, "tiled.base_env_path")
        if self.base_env_path == "/":
            raise ValueError("tiled.base_env_path cannot be '/'")
        if not self.env_prefix or "/" in self.env_prefix:
            raise ValueError("tiled.env_prefix must be non-empty and not contain '/'")
        if self.spacing <= 0.0:
            raise ValueError("tiled.spacing must be positive")
        if self.num_per_row is not None and self.num_per_row < 1:
            raise ValueError("tiled.num_per_row must be positive")
        _validate_per_env_configs(self.per_env, num_envs=self.num_envs)
        self.clone.validate()
        self.diagnostics.validate(num_envs=self.num_envs)

    def metadata_for_env(self, env_id: int) -> dict[str, object]:
        """返回一个 env 的 JSON-compatible metadata 副本。"""

        selected = int(env_id)
        if selected < 0 or selected >= self.num_envs:
            raise ValueError(f"env_id out of range: {selected}")
        for item in self.per_env:
            if item.env_id == selected:
                return dict(item.metadata)
        return {}

    def robot_root_pose_for_env(
        self,
        env_id: int,
        robot_name: str,
        default_pose: RootPoseConfig,
    ) -> RootPoseConfig:
        """解析一个 env 中机器人的 env-local root pose。

        未覆盖时返回机器人 scene instance 的 base pose；scene authoring、MJCF world anchor
        和 IK base frame 必须共同调用该 resolver，避免同一机器人出现多个位姿真相源。
        """

        selected = int(env_id)
        if selected < 0 or selected >= self.num_envs:
            raise ValueError(f"env_id out of range: {selected}")
        for item in self.per_env:
            if item.env_id == selected:
                return item.robot_root_poses.get(robot_name, default_pose)
        return default_pose

    @property
    def effective_num_per_row(self) -> int:
        """返回实际使用的网格每行 env 数。

        未显式配置时取 ``ceil(sqrt(num_envs))``，让 env roots 在 XY 平面上尽量形成
        近似正方形排布。该值只决定空间布局，不影响 batch 数组顺序。
        """

        if self.num_per_row is not None:
            return self.num_per_row
        # math 只在冷路径的配置解析里使用，避免其它热路径模块为一个默认值多 import。
        import math

        return max(1, int(math.ceil(math.sqrt(self.num_envs))))


def _optional_mapping(
    data: Mapping[str, object], key: str
) -> Mapping[str, object] | None:
    """读取可选 mapping，并给出带字段名的错误信息。"""

    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"tiled.{key} must be a mapping")
    return value


def _optional_bool(
    data: Mapping[str, object], key: str, default: bool, label: str
) -> bool:
    """读取可选布尔字段。"""

    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{key} must be a boolean")
    return value


def _optional_non_empty_str(
    data: Mapping[str, object], key: str, default: str, label: str
) -> str:
    """读取可选非空字符串字段。"""

    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _optional_path(
    data: Mapping[str, object], key: str, default: str, label: str
) -> str:
    """读取可选 USD path，并统一去掉末尾斜杠。"""

    value = _optional_non_empty_str(data, key, default, label)
    _require_absolute_usd_path(value, f"{label}.{key}")
    return value.rstrip("/") if value != "/" else value


def _optional_path_or_none(
    data: Mapping[str, object], key: str, label: str
) -> str | None:
    """读取可选 nullable USD path。"""

    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be null or a non-empty string")
    _require_absolute_usd_path(value, f"{label}.{key}")
    return value.rstrip("/") if value != "/" else value


def _optional_usd_paths_or_auto(
    data: Mapping[str, object], key: str, label: str
) -> str | tuple[str, ...]:
    """读取 canonical ``auto`` 或显式 USD path sequence。"""

    if key not in data:
        return "auto"
    value = data[key]
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"{label}.{key} must be 'auto' or a sequence of paths")
        return "auto"
    return _usd_path_tuple(value, label=f"{label}.{key}")


def _optional_usd_path_tuple(
    data: Mapping[str, object],
    key: str,
    default: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    """读取可选 USD path sequence。"""

    if key not in data:
        return tuple(str(path) for path in default)
    return _usd_path_tuple(data[key], label=f"{label}.{key}")


def _usd_path_tuple(value: object, *, label: str) -> tuple[str, ...]:
    """严格解析 USD path sequence，并规范化末尾斜杠。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of absolute USD paths")
    paths: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        _require_absolute_usd_path(item, f"{label}[{index}]")
        if item == "/":
            raise ValueError(f"{label}[{index}] cannot be '/'")
        paths.append(item.rstrip("/") if item != "/" else item)
    return tuple(paths)


def _optional_vec3(
    data: Mapping[str, object],
    key: str,
    default: Sequence[float],
    label: str,
) -> tuple[float, float, float]:
    """读取有限浮点三元组。"""

    value = data.get(key, default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label}.{key} must be a length-3 sequence")
    if len(value) != 3:
        raise ValueError(f"{label}.{key} must contain exactly 3 values")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{label}.{key} must contain finite numbers")
    result = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{label}.{key} must contain finite numbers")
    return result  # type: ignore[return-value]


def _optional_positive_int_or_none(
    data: Mapping[str, object], key: str, label: str
) -> int | None:
    """读取可选正整数；字段缺失或为 null 时返回 None。"""

    if key not in data or data[key] is None:
        return None
    return _positive_int(data[key], f"{label}.{key}")


def _optional_int_tuple(
    data: Mapping[str, object],
    key: str,
    default: Sequence[int],
    label: str,
) -> tuple[int, ...]:
    """读取整数序列并转成不可变 tuple。"""

    if key not in data:
        return tuple(int(value) for value in default)
    value = data[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label}.{key} must be a sequence of integers")
    result = tuple(
        _integer(item, f"{label}.{key}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{label}.{key} cannot contain duplicates")
    return result


def _normalize_collision_filter_strategy(value: object) -> str:
    """校验 tiled env 间碰撞过滤策略。"""

    strategy = str(value or "collision_groups").strip().lower()
    if strategy not in {"collision_groups", "filtered_pairs"}:
        raise ValueError(
            "tiled.clone.collision_filter_strategy must be one of "
            "collision_groups or filtered_pairs"
        )
    return strategy


def _optional_relative_dir_or_none(
    data: Mapping[str, object], key: str, label: str
) -> str | None:
    """读取可选相对目录名。

    目录型 env profile 用它指向 ``base.yaml`` 旁边的 per-env 文件夹。这里禁止绝对路径和
    ``..``，避免配置文件越过 ``configs/envs/<profile>/`` 边界。
    """

    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    parts = tuple(part for part in value.replace("\\", "/").split("/") if part)
    if value.startswith("/") or not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"{label}.{key} must be a relative directory")
    return "/".join(parts)


def _parse_per_env_configs(
    value: object,
    *,
    num_envs: int,
) -> tuple[TiledPerEnvConfig, ...]:
    """解析可选 per-env 差异列表。"""

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("tiled.per_env must be a sequence")
    result = tuple(
        TiledPerEnvConfig.from_mapping(item, index=index)
        for index, item in enumerate(value)
    )
    return tuple(sorted(result, key=lambda item: item.env_id))


def _metadata_mapping(value: object, *, label: str) -> dict[str, object]:
    """解析只允许 JSON-compatible 值的 opaque metadata。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        result[key] = _metadata_value(item, label=f"{label}.{key}")
    return result


def _metadata_value(value: object, *, label: str) -> object:
    """递归标准化 JSON scalar/list/object，拒绝非有限数和任意对象。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{label} must be finite")
        return value
    if isinstance(value, Mapping):
        return _metadata_mapping(value, label=label)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _metadata_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{label} must be JSON-compatible")


def _parse_object_root_pose_overrides(
    value: object,
    *,
    label: str,
) -> dict[str, RootPoseConfig]:
    """解析 per-env 对象位姿覆盖。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")

    result: dict[str, RootPoseConfig] = {}
    for object_name, override in value.items():
        if not isinstance(object_name, str) or not object_name:
            raise ValueError(f"{label} keys must be non-empty strings")
        name = object_name
        if "/" in name or "\\" in name:
            raise ValueError(f"{label}.{name} name must not contain path separators")
        item_label = f"{label}.{name}"
        if not isinstance(override, Mapping):
            raise ValueError(f"{item_label} must be a mapping")
        _reject_keys(override, {"root_pose"}, item_label)
        if "root_pose" not in override:
            raise ValueError(f"{item_label}.root_pose is required")
        pose_data = override["root_pose"]
        if not isinstance(pose_data, Mapping):
            raise ValueError(f"{item_label}.root_pose must be a mapping")
        result[name] = _root_pose_from_mapping(
            pose_data, label=f"{item_label}.root_pose"
        )
    return result


def _parse_robot_root_pose_overrides(
    value: object,
    *,
    label: str,
) -> dict[str, RootPoseConfig]:
    """解析 per-env 机器人 env-local 根位姿覆盖。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")

    result: dict[str, RootPoseConfig] = {}
    for robot_name, override in value.items():
        if not isinstance(robot_name, str) or not robot_name:
            raise ValueError(f"{label} keys must be non-empty strings")
        name = robot_name
        if "/" in name or "\\" in name:
            raise ValueError(f"{label}.{name} name must not contain path separators")
        item_label = f"{label}.{name}"
        if not isinstance(override, Mapping):
            raise ValueError(f"{item_label} must be a mapping")
        _reject_keys(override, {"root_pose"}, item_label)
        if "root_pose" not in override:
            raise ValueError(f"{item_label}.root_pose is required")
        pose_data = override["root_pose"]
        if not isinstance(pose_data, Mapping):
            raise ValueError(f"{item_label}.root_pose must be a mapping")
        _reject_keys(pose_data, {"xyz", "rpy"}, f"{item_label}.root_pose")
        missing_pose_fields = {"xyz", "rpy"} - set(pose_data)
        if missing_pose_fields:
            missing = ", ".join(sorted(missing_pose_fields))
            raise ValueError(
                f"{item_label}.root_pose requires complete xyz and rpy; missing: "
                f"{missing}"
            )
        result[name] = RootPoseConfig(
            xyz=_optional_vec3(
                pose_data,
                "xyz",
                RootPoseConfig.xyz,
                f"{item_label}.root_pose",
            ),
            rpy=_optional_vec3(
                pose_data,
                "rpy",
                RootPoseConfig.rpy,
                f"{item_label}.root_pose",
            ),
        )
    return result


def _parse_camera_pose_overrides(
    value: object,
    *,
    label: str,
) -> dict[str, RootPoseConfig]:
    """解析 per-env 相机位姿覆盖。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")

    result: dict[str, RootPoseConfig] = {}
    for camera_name, override in value.items():
        if not isinstance(camera_name, str) or not camera_name:
            raise ValueError(f"{label} keys must be non-empty strings")
        name = camera_name
        if "/" in name or "\\" in name:
            raise ValueError(f"{label}.{name} name must not contain path separators")
        item_label = f"{label}.{name}"
        if not isinstance(override, Mapping):
            raise ValueError(f"{item_label} must be a mapping")
        _reject_keys(override, {"pose"}, item_label)
        if "pose" not in override:
            raise ValueError(f"{item_label}.pose is required")
        pose_data = override["pose"]
        if not isinstance(pose_data, Mapping):
            raise ValueError(f"{item_label}.pose must be a mapping")
        result[name] = _root_pose_from_mapping(pose_data, label=f"{item_label}.pose")
    return result


def _validate_per_env_configs(
    per_env: Sequence[TiledPerEnvConfig], *, num_envs: int
) -> None:
    """校验 per-env 差异和 batch 数一致。"""

    seen: set[int] = set()
    for item in per_env:
        if item.env_id < 0 or item.env_id >= num_envs:
            raise ValueError("tiled.per_env contains out-of-range env_id")
        if item.env_id in seen:
            raise ValueError(f"tiled.per_env contains duplicate env_id {item.env_id}")
        seen.add(item.env_id)


def _require_absolute_usd_path(path: str, label: str) -> None:
    """校验 USD path 是绝对路径，且没有空路径段。"""

    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"{label} must be an absolute USD path")
    if "//" in path:
        raise ValueError(f"{label} cannot contain empty path components")


def _reject_keys(data: Mapping[str, object], allowed: set[str], label: str) -> None:
    """拒绝未知字段，防止配置拼写错误被静默吞掉。"""

    unsupported = set(data) - allowed
    if unsupported:
        names = sorted(str(key) for key in unsupported)
        keys = ", ".join(names)
        paths = ", ".join(f"{label}.{key}" for key in names)
        raise ValueError(
            f"{label} contains unsupported keys: {keys} (full paths: {paths})"
        )


def _root_pose_from_mapping(
    data: Mapping[str, object], *, label: str
) -> RootPoseConfig:
    """严格解析 root pose 映射，并在错误中保留调用方传入的完整配置路径。"""

    _reject_keys(data, {"xyz", "rpy"}, label)
    return RootPoseConfig(
        xyz=_optional_vec3(data, "xyz", RootPoseConfig.xyz, label),
        rpy=_optional_vec3(data, "rpy", RootPoseConfig.rpy, label),
    )


def _integer(value: object, label: str) -> int:
    """严格解析整数，不把布尔值、浮点数或字符串强制转换为整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _positive_int(value: object, label: str) -> int:
    parsed = _integer(value, label)
    if parsed < 1:
        raise ValueError(f"{label} must be positive")
    return parsed


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite positive number")
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return parsed
