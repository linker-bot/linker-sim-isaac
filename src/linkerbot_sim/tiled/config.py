"""Isaac Lab 风格 tiled envs 的纯 Python 配置模型。

本模块只负责解析 env profile 里的 ``tiled`` 顶层分组，并做不依赖 Isaac/Omni
运行时的结构校验。这样 CLI、单元测试和后续下游 evaluator 可以先验证配置是否合理，
而不会因为 import 本包就启动 Isaac Sim。

设计约定:
    * tiled 配置是旧 env profile 的增量扩展；旧单 env runtime 可以完全忽略它。
    * 所有 USD path 都必须是绝对路径，避免后续 clone/filter collision 时路径语义含混。
    * tiled env 数量只来自 profile YAML 中的 ``tiled.num_envs``；CLI/调用方不再覆盖。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from linkerbot_sim.assets.robot_loader import RootPoseConfig


@dataclass(frozen=True)
class TiledPerEnvConfig:
    """单个 tiled env 的差异配置。

    tiled runtime 第一阶段要求所有 env 拥有相同机器人和物体集合；这里仅允许按 env
    覆盖已有物体的 ``root_pose``。这样仍然可以使用 ``GridCloner`` 复制同构 stage，
    再在 clone 后把每个 env 的对象移动到自己的局部位置。
    """

    env_id: int
    object_root_poses: dict[str, RootPoseConfig] = field(default_factory=dict)
    metadata: Mapping[str, object] | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object], *, index: int
    ) -> "TiledPerEnvConfig":
        """解析 ``tiled.per_env[index]``。

        支持的 YAML 形态:

        ``objects.<object_name>.root_pose``:
            推荐写法，直接按对象名覆盖位姿。
        ``objects.overrides.<object_name>.root_pose``:
            兼容更显式的 override 分组写法。
        """

        if not isinstance(data, Mapping):
            raise ValueError(f"tiled.per_env[{index}] must be a mapping")
        _reject_keys(data, {"env_id", "objects", "metadata"}, f"tiled.per_env[{index}]")
        if "env_id" not in data:
            raise ValueError(f"tiled.per_env[{index}].env_id is required")
        env_id = int(data["env_id"])
        metadata = data.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(f"tiled.per_env[{index}].metadata must be a mapping")
        return cls(
            env_id=env_id,
            object_root_poses=_parse_object_root_pose_overrides(
                data.get("objects"), label=f"tiled.per_env[{index}].objects"
            ),
            metadata=metadata,
        )

@dataclass(frozen=True)
class TiledCloneConfig:
    """控制 USD/PhysX 克隆行为的配置。

    这些字段对应未来 scene builder 使用 ``GridCloner`` 时需要的选项。这里不直接
    import Isaac API，只保存“应该如何 clone”的意图。
    """

    # 是否优先使用 Isaac Sim 原生 GridCloner。保留开关是为了将来遇到资产兼容问题时，
    # 可以临时切换到较慢但更容易诊断的手工 clone 路径。
    use_grid_cloner: bool = True
    # replicate_physics 让 PhysX 复用克隆环境的物理结构，是 256+ env 性能路径的关键。
    replicate_physics: bool = True
    # copy_from_source=false 表示 clone 可以继承 source prim，减少 USD stage 冗余。
    copy_from_source: bool = False
    # enable_env_ids 是 colocated env 的高级能力；第一阶段通过 spacing 分隔 env，默认关闭。
    enable_env_ids: bool = False
    # 开启 env 间 collision filtering，避免相邻 tiled env 发生跨环境接触。
    filter_collisions: bool = True
    # GridCloner.filter_collisions 创建 collision groups 时使用的根路径。
    collision_root_path: str = "/World/collisions"

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None
    ) -> "TiledCloneConfig":
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
                "use_grid_cloner",
                "replicate_physics",
                "copy_from_source",
                "enable_env_ids",
                "filter_collisions",
                "collision_root_path",
            },
            "tiled.clone",
        )
        config = cls(
            use_grid_cloner=_optional_bool(
                data, "use_grid_cloner", cls.use_grid_cloner, "tiled.clone"
            ),
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
            collision_root_path=_optional_path(
                data,
                "collision_root_path",
                cls.collision_root_path,
                "tiled.clone",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """校验 clone 配置中和 USD stage 相关的字段。"""

        if not self.use_grid_cloner:
            raise ValueError(
                "tiled.clone.use_grid_cloner=false is not implemented; "
                "the current tiled scene builder always uses Isaac GridCloner"
            )
        _require_absolute_usd_path(
            self.collision_root_path, "tiled.clone.collision_root_path"
        )


@dataclass(frozen=True)
class TiledRuntimeConfig:
    """控制 clone 完成后的 tiled runtime 行为。

    这些选项影响 view/controller/state 这些运行时包装层，不影响 USD scene 的拓扑。
    """

    # 目标实现必须用 batched articulation view 读写状态；当前没有 per-env view fallback。
    use_batched_articulation_view: bool = True
    # GUI/telemetry 调试时通常只需要看少数 env。这里记录想重点观察的 env id，
    # 不为每个 env 创建 viewport，也不减少物理或渲染开销。
    inspect_env_ids: tuple[int, ...] = (0,)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | None
    ) -> "TiledRuntimeConfig":
        """从 ``tiled.runtime`` mapping 解析配置。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError("tiled.runtime must be a mapping")
        _reject_keys(
            data,
            {"use_batched_articulation_view", "inspect_env_ids"},
            "tiled.runtime",
        )
        return cls(
            use_batched_articulation_view=_optional_bool(
                data,
                "use_batched_articulation_view",
                cls.use_batched_articulation_view,
                "tiled.runtime",
            ),
            inspect_env_ids=_optional_int_tuple(
                data, "inspect_env_ids", cls.inspect_env_ids, "tiled.runtime"
            ),
        )

    def validate(self, *, num_envs: int) -> None:
        """校验 runtime 配置是否和 ``num_envs`` 匹配。"""

        if not self.use_batched_articulation_view:
            raise ValueError(
                "tiled.runtime.use_batched_articulation_view=false is not implemented; "
                "the current tiled runtime always uses batched articulation views"
            )
        if any(env_id < 0 or env_id >= num_envs for env_id in self.inspect_env_ids):
            raise ValueError("tiled.runtime.inspect_env_ids contains out-of-range env id")


@dataclass(frozen=True)
class TiledEnvConfig:
    """env profile 中 ``tiled`` 顶层配置的完整结果。

    ``enabled=False, num_envs=1`` 表示保持单 env 语义。即使旧 profile 没有 ``tiled``
    分组，本类也能返回一个合法默认值，方便上层统一处理。
    """

    # enabled 只是 tiled runtime 的入口开关；旧 runtime 不读取也不应该受它影响。
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
    # 每个 env 的差异配置。第一阶段只允许覆盖已有对象的 root_pose。
    per_env: tuple[TiledPerEnvConfig, ...] = ()
    clone: TiledCloneConfig = field(default_factory=TiledCloneConfig)
    runtime: TiledRuntimeConfig = field(default_factory=TiledRuntimeConfig)

    @classmethod
    def from_env_config(
        cls,
        env_config: Mapping[str, object],
    ) -> "TiledEnvConfig":
        """从完整 env profile 解析可选的 ``tiled`` 分组。

        env 数量以 YAML 中的 ``tiled.num_envs`` 为唯一来源；没有 ``tiled`` 分组时
        返回 disabled/single-env 默认配置。
        """

        data = env_config.get("tiled")
        if data is None:
            config = cls()
            config.validate()
            return config
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
                "clone",
                "runtime",
            },
            "tiled",
        )
        num_envs = int(data.get("num_envs", cls.num_envs))
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
            spacing=float(data.get("spacing", cls.spacing)),
            num_per_row=_optional_positive_int_or_none(
                data, "num_per_row", "tiled"
            ),
            per_env_config_dir=_optional_relative_dir_or_none(
                data, "per_env_config_dir", "tiled"
            ),
            per_env=_parse_per_env_configs(
                data.get("per_env"),
                num_envs=num_envs,
            ),
            clone=TiledCloneConfig.from_mapping(_optional_mapping(data, "clone")),
            runtime=TiledRuntimeConfig.from_mapping(
                _optional_mapping(data, "runtime")
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
        self.runtime.validate(num_envs=self.num_envs)

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

    value = data.get(key)
    if value is None:
        return None
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


def _optional_positive_int_or_none(
    data: Mapping[str, object], key: str, label: str
) -> int | None:
    """读取可选正整数；字段缺失或为 null 时返回 None。"""

    if key not in data or data[key] is None:
        return None
    value = int(data[key])
    if value < 1:
        raise ValueError(f"{label}.{key} must be positive")
    return value


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
    return tuple(int(item) for item in value)


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
    if "overrides" in value:
        if set(value) != {"overrides"}:
            raise ValueError(f"{label} cannot mix overrides with direct object keys")
        overrides = value["overrides"]
        if not isinstance(overrides, Mapping):
            raise ValueError(f"{label}.overrides must be a mapping")
        value = overrides

    result: dict[str, RootPoseConfig] = {}
    for object_name, override in value.items():
        name = str(object_name)
        if not name:
            raise ValueError(f"{label} contains an empty object name")
        item_label = f"{label}.{name}"
        if not isinstance(override, Mapping):
            raise ValueError(f"{item_label} must be a mapping")
        _reject_keys(override, {"root_pose"}, item_label)
        if "root_pose" not in override:
            raise ValueError(f"{item_label}.root_pose is required")
        pose_data = override["root_pose"]
        if not isinstance(pose_data, Mapping):
            raise ValueError(f"{item_label}.root_pose must be a mapping")
        result[name] = RootPoseConfig.from_mapping(pose_data)
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
        keys = ", ".join(sorted(unsupported))
        raise ValueError(f"{label} contains unsupported keys: {keys}")
