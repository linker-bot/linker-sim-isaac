"""Object scene instance 和 object profile 的统一解析。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from linkerbot_sim.assets.robot_loader import RootPoseConfig
from linkerbot_sim.configs.profiles import load_profile_yaml


@dataclass(frozen=True)
class ObjectSceneInstanceConfig:
    """env ``objects[]`` 中的场景实例声明。"""

    name: str
    object_profile: str
    root_pose: RootPoseConfig
    runtime_handle: str | None = None

    @classmethod
    def from_mapping(
        cls, data: object, *, index: int
    ) -> "ObjectSceneInstanceConfig":
        """解析 env.objects[index]，只允许 scene 层字段。"""

        if not isinstance(data, Mapping):
            raise ValueError(f"objects[{index}] must be a mapping")
        allowed = {"name", "object_profile", "runtime_handle", "root_pose"}
        unsupported = set(data) - allowed
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                f"objects[{index}] contains scene-level unsupported keys: {names}; "
                "put object properties in configs/objects"
            )
        if "name" not in data:
            raise ValueError(f"objects[{index}].name is required")
        name = str(data["name"])
        if not name:
            raise ValueError(f"objects[{index}].name cannot be empty")
        if "object_profile" not in data:
            raise ValueError(f"objects[{index}].object_profile is required")
        object_profile = str(data["object_profile"])
        if not object_profile:
            raise ValueError(f"objects[{index}].object_profile cannot be empty")
        if "root_pose" not in data:
            raise ValueError(f"objects[{index}].root_pose is required")
        runtime_handle = data.get("runtime_handle")
        if runtime_handle is not None:
            runtime_handle = str(runtime_handle)
            if not runtime_handle:
                raise ValueError(f"objects[{index}].runtime_handle cannot be empty")
        return cls(
            name=name,
            object_profile=object_profile,
            runtime_handle=runtime_handle,
            root_pose=RootPoseConfig.from_mapping(
                _required_mapping(data, "root_pose", f"objects[{index}]")
            ),
        )


@dataclass(frozen=True)
class ObjectProfileConfig:
    """``configs/objects/*.yaml`` 顶层 ``object`` 段的通用配置。"""

    profile_name: str
    name: str
    kind: str
    source: str
    asset_path: str
    prim_path: str
    root_path: str | None = None
    urdf_drive_type: str = "none"
    import_config: Mapping[str, object] | None = None
    physics: Mapping[str, object] | None = None
    raw: Mapping[str, Any] | None = None

    @classmethod
    def from_profile(cls, name: str) -> "ObjectProfileConfig":
        """按 profile 名称加载 configs/objects/<name>.yaml。"""

        return cls.from_mapping(load_profile_yaml("object", name), profile_name=name)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, profile_name: str
    ) -> "ObjectProfileConfig":
        """解析 object profile，并拒绝 scene-only 字段进入 profile。"""

        unsupported_top_level = set(data) - {"object"}
        if unsupported_top_level:
            names = ", ".join(sorted(unsupported_top_level))
            raise ValueError(
                f"Object profile {profile_name!r} contains unsupported top-level "
                f"keys: {names}; runtime object properties belong under object"
            )
        if "object" not in data or not isinstance(data["object"], Mapping):
            raise ValueError(
                f"Object profile {profile_name!r} must contain top-level object mapping"
            )
        object_cfg = dict(data["object"])
        if "root_pose" in object_cfg:
            raise ValueError("object profile root_pose is not allowed; put it in env")
        allowed = {
            "name",
            "kind",
            "source",
            "asset_path",
            "prim_path",
            "root_path",
            "urdf_drive_type",
            "import",
            "physics",
        }
        unsupported = set(object_cfg) - allowed
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                f"Object profile {profile_name!r} contains unsupported object keys: "
                f"{names}"
            )
        for key in ("kind", "source", "asset_path", "prim_path"):
            if key not in object_cfg:
                raise ValueError(f"Object profile {profile_name!r} missing object.{key}")
        kind = str(object_cfg["kind"]).lower()
        allowed_kinds = {"rigid", "dynamic_chain"}
        if kind not in allowed_kinds:
            raise ValueError(
                f"Object profile {profile_name!r} object.kind must be one of "
                f"{sorted(allowed_kinds)}, got {kind!r}"
            )
        source = str(object_cfg["source"]).lower()
        allowed_sources = {"usd", "urdf"}
        if source not in allowed_sources:
            raise ValueError(
                f"Object profile {profile_name!r} object.source must be one of "
                f"{sorted(allowed_sources)}, got {source!r}"
            )
        prim_path = str(object_cfg["prim_path"])
        if not prim_path.startswith("/"):
            raise ValueError(
                f"Object profile {profile_name!r} object.prim_path must be absolute"
            )
        root_path = (
            None if object_cfg.get("root_path") is None else str(object_cfg["root_path"])
        )
        if root_path is not None and not root_path.startswith("/"):
            raise ValueError(
                f"Object profile {profile_name!r} object.root_path must be absolute"
            )
        return cls(
            profile_name=profile_name,
            name=str(object_cfg.get("name", profile_name)),
            kind=kind,
            source=source,
            asset_path=str(object_cfg["asset_path"]),
            prim_path=prim_path,
            root_path=root_path,
            urdf_drive_type=str(object_cfg.get("urdf_drive_type", "none")),
            import_config=_optional_mapping(object_cfg, "import", "object"),
            physics=_optional_mapping(object_cfg, "physics", "object"),
            raw=data,
        )


def object_scene_instances_from_env_config(
    env_config: Mapping[str, object],
) -> tuple[ObjectSceneInstanceConfig, ...]:
    """解析 env YAML 顶层 ``objects`` 列表。"""

    from collections.abc import Sequence

    objects = env_config.get("objects", ())
    if objects is None:
        return ()
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("objects must be a sequence")
    return tuple(
        ObjectSceneInstanceConfig.from_mapping(item, index=index)
        for index, item in enumerate(objects)
    )


def expanded_object_mapping(
    instance: ObjectSceneInstanceConfig,
    profile: ObjectProfileConfig | None = None,
) -> dict[str, object]:
    """把 scene instance 和 object profile 合成低层导入配置。"""

    profile = profile or ObjectProfileConfig.from_profile(instance.object_profile)
    data: dict[str, object] = {
        "name": instance.name,
        "source": profile.source,
        "asset_path": profile.asset_path,
        "prim_path": profile.prim_path,
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
    return data


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

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value
