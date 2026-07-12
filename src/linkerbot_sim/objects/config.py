"""对象场景实例与对象 profile 的严格解析和合并边界。

env YAML 的 ``objects[]`` 只声明实例身份、root pose 与可选 prim path；资产来源、导入参数、
物理属性和规划碰撞属性只允许出现在 ``configs/objects/*.yaml``。本模块在创建任何 USD
对象前拒绝未知字段、无效组合和场景内身份冲突，并把两层配置合成底层导入所需映射。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configs.profiles import profile_path
from linkerbot_sim.utils.config import load_yaml


OBJECT_INSTANCE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_OBJECT_INSTANCE_PRIM_ROOT = "/World/Objects"
_OBJECT_PROFILE_ROOT_KEYS = frozenset({"object"})


@dataclass(frozen=True)
class ObjectSceneInstanceConfig:
    """env ``objects[]`` 中的一条场景实例声明。

    ``name`` 是场景内稳定身份并用于派生默认 prim path；``runtime_handle`` 是交互协议可用
    的可选别名；资产和物理配置由 ``object_profile`` 唯一引用，不能在实例层覆盖。
    """

    name: str
    object_profile: str
    root_pose: RootPoseConfig
    runtime_handle: str | None = None
    prim_path: str | None = None

    @property
    def default_prim_path(self) -> str:
        """按稳定 instance name 派生默认 USD prim path。"""

        return f"{DEFAULT_OBJECT_INSTANCE_PRIM_ROOT}/{self.name}"

    @property
    def effective_prim_path(self) -> str:
        """返回显式或 name-derived 路径。"""

        return self.prim_path or self.default_prim_path

    @classmethod
    def from_mapping(cls, data: object, *, index: int) -> "ObjectSceneInstanceConfig":
        """解析 ``env.objects[index]``，并拒绝所有 profile 层字段。

        ``index`` 只用于生成精确 YAML 路径的错误信息。返回前会规范化字符串和绝对 prim
        path；跨实例唯一性由列表解析完成后统一校验。
        """

        if not isinstance(data, Mapping):
            raise ValueError(f"objects[{index}] must be a mapping")
        allowed = {
            "name",
            "object_profile",
            "runtime_handle",
            "prim_path",
            "root_pose",
        }
        unsupported = set(data) - allowed
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                f"objects[{index}] contains scene-level unsupported keys: {names}; "
                "put object properties in configs/objects"
            )
        if "name" not in data:
            raise ValueError(f"objects[{index}].name is required")
        name_value = data["name"]
        if not isinstance(name_value, str):
            raise ValueError(f"objects[{index}].name must be a string")
        name = name_value.strip()
        if not name:
            raise ValueError(f"objects[{index}].name cannot be empty")
        if OBJECT_INSTANCE_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"objects[{index}].name must match [A-Za-z_][A-Za-z0-9_]*, got {name!r}"
            )
        if "object_profile" not in data:
            raise ValueError(f"objects[{index}].object_profile is required")
        object_profile_value = data["object_profile"]
        if not isinstance(object_profile_value, str):
            raise ValueError(f"objects[{index}].object_profile must be a string")
        object_profile = object_profile_value.strip()
        if not object_profile:
            raise ValueError(f"objects[{index}].object_profile cannot be empty")
        if "root_pose" not in data:
            raise ValueError(f"objects[{index}].root_pose is required")
        runtime_handle = data.get("runtime_handle")
        if runtime_handle is not None:
            if not isinstance(runtime_handle, str):
                raise ValueError(f"objects[{index}].runtime_handle must be a string")
            runtime_handle = runtime_handle.strip()
            if not runtime_handle:
                raise ValueError(f"objects[{index}].runtime_handle cannot be empty")
        return cls(
            name=name,
            object_profile=object_profile,
            runtime_handle=runtime_handle,
            prim_path=_optional_object_instance_prim_path(data, index=index),
            root_pose=RootPoseConfig.from_mapping(
                _required_mapping(data, "root_pose", f"objects[{index}]")
            ),
        )


@dataclass(frozen=True)
class ObjectStateSummaryConfig:
    """dynamic-chain 顶层 object state 的确定性摘要规则。

    动态链包含多个刚体，必须显式指定 ``reference_body``，才能稳定定义对外对象位姿；该
    字段是 body 名而不是完整 prim path。
    """

    reference_body: str | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object] | None,
        *,
        label: str,
    ) -> "ObjectStateSummaryConfig":
        """解析 ``object.state_summary``，当前只支持命名参考刚体。"""

        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        unsupported = set(data) - {"reference_body"}
        if unsupported:
            paths = ", ".join(f"{label}.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        value = data.get("reference_body")
        if value is None:
            return cls()
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}.reference_body must be a non-empty string")
        reference_body = value.strip()
        if "/" in reference_body or "\\" in reference_body:
            raise ValueError(
                f"{label}.reference_body must be a body name, not a prim path"
            )
        return cls(reference_body=reference_body)


@dataclass(frozen=True)
class ObjectProfileConfig:
    """``configs/objects/*.yaml`` 顶层 ``object`` 段的规范化配置。

    rigid 与 dynamic_chain 共用资产身份字段，但允许的 import、physics、collision 和
    state-summary 组合不同。``raw`` 保留已通过严格校验的原始映射，供下游需要完整配置时
    读取，不承担未知字段透传。
    """

    profile_name: str
    name: str
    kind: str
    source: str
    asset_path: str
    root_path: str | None = None
    urdf_drive_type: str = "none"
    import_config: Mapping[str, object] | None = None
    physics: Mapping[str, object] | None = None
    planning_collision: Mapping[str, object] | None = None
    state_summary: ObjectStateSummaryConfig = ObjectStateSummaryConfig()
    raw: Mapping[str, Any] | None = None

    @classmethod
    def from_profile(cls, name: str) -> "ObjectProfileConfig":
        """按 profile 名称加载 configs/objects/<name>.yaml。"""

        path = profile_path("object", name)
        return cls.from_mapping(load_yaml(path), profile_name=name, source=str(path))

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        profile_name: str,
        source: str | None = None,
    ) -> "ObjectProfileConfig":
        """解析 object profile，并拒绝 scene-only 字段进入 profile。"""

        source_label = source or f"Object profile {profile_name!r}"
        canonical = validate_object_profile(
            data, source=source_label, profile_name=profile_name
        )
        return cls._from_canonical_mapping(canonical, profile_name=profile_name)

    @classmethod
    def _from_canonical_mapping(
        cls, data: Mapping[str, Any], *, profile_name: str
    ) -> "ObjectProfileConfig":
        """解析已经完成嵌套校验的 canonical profile。"""

        unsupported_top_level = set(data) - _OBJECT_PROFILE_ROOT_KEYS
        if unsupported_top_level:
            names = ", ".join(sorted(str(key) for key in unsupported_top_level))
            raise ValueError(
                f"Object profile {profile_name!r} contains unsupported top-level "
                f"keys: {names}; runtime object properties belong under object"
            )
        if "object" not in data or not isinstance(data["object"], Mapping):
            raise ValueError(
                f"Object profile {profile_name!r} must contain top-level object mapping"
            )
        object_cfg = dict(data["object"])
        allowed = {
            "name",
            "kind",
            "source",
            "asset_path",
            "root_path",
            "urdf_drive_type",
            "import",
            "physics",
            "planning_collision",
            "state_summary",
        }
        unsupported = set(object_cfg) - allowed
        if unsupported:
            paths = ", ".join(f"object.{key}" for key in sorted(unsupported))
            raise ValueError(f"unsupported configuration field(s): {paths}")
        for key in ("kind", "source", "asset_path"):
            if key not in object_cfg:
                raise ValueError(
                    f"Object profile {profile_name!r} missing object.{key}"
                )
        kind = _required_non_empty_str(
            object_cfg["kind"], f"Object profile {profile_name!r} object.kind"
        ).lower()
        allowed_kinds = {"rigid", "dynamic_chain"}
        if kind not in allowed_kinds:
            raise ValueError(
                f"Object profile {profile_name!r} object.kind must be one of "
                f"{sorted(allowed_kinds)}, got {kind!r}"
            )
        state_summary = ObjectStateSummaryConfig.from_mapping(
            _optional_mapping(object_cfg, "state_summary", "object"),
            label="object.state_summary",
        )
        if kind != "dynamic_chain" and "state_summary" in object_cfg:
            raise ValueError(
                "object.state_summary is only supported for dynamic_chain objects"
            )
        if kind == "dynamic_chain" and state_summary.reference_body is None:
            raise ValueError(
                "dynamic_chain object.state_summary.reference_body is required"
            )
        source = _required_non_empty_str(
            object_cfg["source"], f"Object profile {profile_name!r} object.source"
        ).lower()
        allowed_sources = {"usd", "urdf"}
        if source not in allowed_sources:
            raise ValueError(
                f"Object profile {profile_name!r} object.source must be one of "
                f"{sorted(allowed_sources)}, got {source!r}"
            )
        root_path = (
            None
            if object_cfg.get("root_path") is None
            else _required_non_empty_str(
                object_cfg["root_path"],
                f"Object profile {profile_name!r} object.root_path",
            )
        )
        if root_path is not None and not root_path.startswith("/"):
            raise ValueError(
                f"Object profile {profile_name!r} object.root_path must be absolute"
            )
        return cls(
            profile_name=profile_name,
            name=_required_non_empty_str(
                object_cfg.get("name", profile_name),
                f"Object profile {profile_name!r} object.name",
            ),
            kind=kind,
            source=source,
            asset_path=_required_non_empty_str(
                object_cfg["asset_path"],
                f"Object profile {profile_name!r} object.asset_path",
            ),
            root_path=root_path,
            urdf_drive_type=_object_urdf_drive_type(
                object_cfg.get("urdf_drive_type", "none"),
                label=f"Object profile {profile_name!r} object.urdf_drive_type",
            ),
            import_config=_optional_mapping(object_cfg, "import", "object"),
            physics=_optional_mapping(object_cfg, "physics", "object"),
            planning_collision=_optional_mapping(
                object_cfg, "planning_collision", "object"
            ),
            state_summary=state_summary,
            raw=data,
        )


def validate_object_profile(
    data: Mapping[str, Any],
    *,
    source: str = "<object profile>",
    profile_name: str = "object",
) -> dict[str, Any]:
    """严格校验一份项目对象 profile，并返回规范化的独立字典。

    校验分两层：先检查本模块拥有的根字段和通用语义，再把 import/physics/planning 子段
    交给各自配置模型。任何异常都会附加 profile 来源，便于定位具体 YAML 文件。
    """

    try:
        if not isinstance(data, Mapping):
            raise ValueError("object profile must be a mapping")
        canonical = dict(data)
        _reject_unknown_paths(canonical, _OBJECT_PROFILE_ROOT_KEYS, "profile")
        parsed = ObjectProfileConfig._from_canonical_mapping(
            canonical, profile_name=profile_name
        )
        object_cfg = _required_mapping(canonical, "object", "profile")
        _validate_object_consumer_sections(object_cfg, parsed=parsed)
    except ValueError as exc:
        if str(exc).startswith(f"{source}:"):
            raise
        raise ValueError(f"{source}: {exc}") from exc
    return canonical


def load_object_profile(path: str | Path) -> dict[str, Any]:
    """从路径加载并严格校验 object profile。"""

    object_path = Path(path)
    return validate_object_profile(
        load_yaml(object_path),
        source=str(object_path),
        profile_name=object_path.stem,
    )


def _validate_object_consumer_sections(
    object_cfg: Mapping[str, object], *, parsed: ObjectProfileConfig
) -> None:
    from linkerbot_sim.assets.robot_config import AssetImportConfig
    from linkerbot_sim.objects.dynamic_chain.capsule_rope import (
        CapsuleRopePhysicsConfig,
    )
    from linkerbot_sim.objects.rigid.config import (
        RigidObjectPhysicsConfig,
        RigidObjectPlanningCollisionConfig,
    )

    import_settings = _optional_mapping(object_cfg, "import", "object")
    physics = _optional_mapping(object_cfg, "physics", "object")
    planning_collision = _optional_mapping(object_cfg, "planning_collision", "object")
    if parsed.kind == "rigid":
        if parsed.root_path is not None:
            raise ValueError("object.root_path is only supported for dynamic_chain")
        if parsed.source == "usd" and "urdf_drive_type" in object_cfg:
            raise ValueError(
                "object.urdf_drive_type is only supported for rigid URDF objects"
            )
        AssetImportConfig.from_mapping(
            import_settings,
            label="object.import",
            asset_type=parsed.source,
        )
        RigidObjectPhysicsConfig.from_mapping(physics, label="object.physics")
        RigidObjectPlanningCollisionConfig.from_mapping(
            planning_collision, label="object.planning_collision"
        )
        return

    if parsed.source != "usd":
        raise ValueError("dynamic_chain object.source must be 'usd'")
    if import_settings is not None:
        raise ValueError("object.import is only supported for rigid objects")
    if planning_collision is not None:
        raise ValueError(
            "object.planning_collision is only supported for rigid objects"
        )
    if "urdf_drive_type" in object_cfg:
        raise ValueError(
            "object.urdf_drive_type is only supported for rigid URDF objects"
        )
    CapsuleRopePhysicsConfig.from_mapping(physics, label="object.physics")


def object_scene_instances_from_env_config(
    env_config: Mapping[str, object],
) -> tuple[ObjectSceneInstanceConfig, ...]:
    """解析 env YAML 顶层 ``objects`` 列表。"""

    objects = env_config.get("objects", ())
    if objects is None:
        return ()
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("objects must be a sequence")
    instances = tuple(
        ObjectSceneInstanceConfig.from_mapping(item, index=index)
        for index, item in enumerate(objects)
    )
    _validate_object_scene_instances(instances)
    return instances


def expanded_object_mapping(
    instance: ObjectSceneInstanceConfig,
    profile: ObjectProfileConfig | None = None,
) -> dict[str, object]:
    """把场景实例与对象 profile 合成低层导入配置。

    实例层只贡献身份、prim path 和 root pose；其余属性全部来自 profile。返回值是新建的
    可变字典，不会把调用方对下游配置的修改写回冻结 dataclass。
    """

    profile = profile or ObjectProfileConfig.from_profile(instance.object_profile)
    data: dict[str, object] = {
        "name": instance.name,
        "source": profile.source,
        "asset_path": profile.asset_path,
        "prim_path": instance.effective_prim_path,
        "root_pose": {
            "xyz": list(instance.root_pose.xyz),
            "rpy": list(instance.root_pose.rpy),
        },
        "urdf_drive_type": profile.urdf_drive_type,
    }
    if profile.import_config is not None:
        data["import"] = dict(profile.import_config)
    if profile.physics is not None:
        data["physics"] = dict(profile.physics)
    if profile.planning_collision is not None:
        data["planning_collision"] = dict(profile.planning_collision)
    return data


def _optional_object_instance_prim_path(
    data: Mapping[str, object], *, index: int
) -> str | None:
    """读取 env object instance 的可选绝对 USD path。"""

    if "prim_path" not in data:
        return None
    if data["prim_path"] is None:
        raise ValueError(f"objects[{index}].prim_path must be a non-empty string")
    return _optional_absolute_prim_path(
        data["prim_path"], label=f"objects[{index}].prim_path"
    )


def _optional_absolute_prim_path(value: object, *, label: str) -> str | None:
    """校验可选绝对 USD prim path。"""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    path = value.strip()
    if not path.startswith("/") or path == "/" or path.endswith("/") or "//" in path:
        raise ValueError(f"{label} must be an absolute USD path")
    return path


def _validate_object_scene_instances(
    instances: Sequence[ObjectSceneInstanceConfig],
) -> None:
    """校验 instance name、runtime handle 和 effective path 的 scene 唯一性。"""

    names: dict[str, int] = {}
    runtime_handles: dict[str, int] = {}
    prim_paths: dict[str, int] = {}
    for index, instance in enumerate(instances):
        if instance.name in names:
            raise ValueError(
                f"Duplicate object name {instance.name!r} for objects indices "
                f"{names[instance.name]} and {index}"
            )
        names[instance.name] = index
        if instance.runtime_handle is not None:
            if instance.runtime_handle in runtime_handles:
                raise ValueError(
                    f"Duplicate object runtime_handle {instance.runtime_handle!r} "
                    f"for objects indices {runtime_handles[instance.runtime_handle]} "
                    f"and {index}"
                )
            runtime_handles[instance.runtime_handle] = index
        prim_path = instance.effective_prim_path
        if prim_path in prim_paths:
            raise ValueError(
                f"Duplicate object prim path {prim_path!r} for objects indices "
                f"{prim_paths[prim_path]} and {index}"
            )
        prim_paths[prim_path] = index
    for handle, index in runtime_handles.items():
        name_index = names.get(handle)
        if name_index is not None and name_index != index:
            raise ValueError(
                f"Object runtime_handle {handle!r} for objects[{index}] conflicts "
                f"with objects[{name_index}].name"
            )


def _required_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object]:
    """读取必填 mapping 字段，并生成带父路径的错误信息。"""

    value = data.get(key)
    if value is None:
        raise ValueError(f"{parent_label}.{key} is required")
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _optional_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object] | None:
    """读取可选 mapping 字段；缺省时返回 None。"""

    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _required_non_empty_str(value: object, label: str) -> str:
    """读取并去除必填字符串两端空白，空值按完整字段路径报错。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _object_urdf_drive_type(value: object, *, label: str) -> str:
    """规范化对象 URDF drive 类型并限制为当前支持的枚举。"""

    normalized = _required_non_empty_str(value, label).lower()
    if normalized not in {"none", "position"}:
        raise ValueError(f"{label} must be 'none' or 'position'")
    return normalized


def _reject_unknown_paths(
    data: Mapping[Any, Any], allowed: set[str] | frozenset[str], label: str
) -> None:
    """拒绝 mapping 中未由当前配置所有者声明的字段。"""

    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        paths = ", ".join(f"{label}.{key}" for key in unknown)
        raise ValueError(f"unsupported configuration field(s): {paths}")


__all__ = [
    "ObjectProfileConfig",
    "ObjectSceneInstanceConfig",
    "expanded_object_mapping",
    "load_object_profile",
    "object_scene_instances_from_env_config",
    "validate_object_profile",
]
