"""场景机器人实例身份与单 articulation 执行配置。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from linkerbot_sim.assets.robot_config import RobotAssetConfig
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.controllers.config import normalize_controller_bundle_name


ROBOT_INSTANCE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
DEFAULT_ROBOT_INSTANCE_PRIM_ROOT = "/World/Robots"


@dataclass(frozen=True)
class RobotSceneInstanceConfig:
    """env ``robots[]`` 中一个按 list 顺序编号的 canonical 实例。"""

    robot_profile: str
    root_pose: RootPoseConfig
    robot_id: int = 0
    label: str = ""
    prim_path: str | None = None
    controller_profile: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.robot_id, bool) or int(self.robot_id) < 0:
            raise ValueError("robot_id must be a non-negative integer")
        if not self.label:
            raise ValueError("robot label cannot be empty")
        if self.controller_profile is not None:
            normalized = normalize_controller_bundle_name(
                self.controller_profile,
                label="robot instance controller_profile",
            )
            object.__setattr__(self, "controller_profile", normalized)

    @property
    def default_prim_path(self) -> str:
        """按稳定 label 派生默认 USD prim path。"""

        return f"{DEFAULT_ROBOT_INSTANCE_PRIM_ROOT}/{self.label}"

    @property
    def effective_prim_path(self) -> str:
        """返回显式路径或 label-derived 默认路径。"""

        return self.prim_path or self.default_prim_path


@dataclass(frozen=True)
class RobotExecutionConfig:
    """单个 Isaac articulation 的资产、主动控制关节与 root pose。"""

    robot: RobotAssetConfig
    controlled_joints: tuple[str, ...]
    root_pose: RootPoseConfig = RootPoseConfig()

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        scene_instance: RobotSceneInstanceConfig,
    ) -> "RobotExecutionConfig":
        """把 robot profile 与唯一场景实例组合成执行配置。"""

        config = dict(data)
        robot = RobotAssetConfig.from_mapping(
            config,
            prim_path=scene_instance.effective_prim_path,
            name=scene_instance.label,
        )
        return cls(
            robot=robot,
            controlled_joints=_controlled_joints_from_mapping(config),
            root_pose=scene_instance.root_pose,
        )


def robot_instances_from_env_config(
    env_config: Mapping[str, object], *, allow_empty: bool = False
) -> tuple[RobotSceneInstanceConfig, ...]:
    """把 ``robots`` list 规范化为稠密 session ID 的实例序列。"""

    robots = env_config.get("robots")
    if not isinstance(robots, Sequence) or isinstance(robots, (str, bytes)):
        raise ValueError("Environment config must contain a top-level robots list")
    if not robots:
        if allow_empty:
            return ()
        raise ValueError("robots cannot be empty")
    instances = tuple(
        _canonical_robot_scene_instance(
            robot_id=index,
            source_label=f"robots[{index}]",
            data=item,
        )
        for index, item in enumerate(robots)
    )
    _validate_robot_scene_instances(instances)
    return instances


def resolve_controller_profile(
    scene_instance: RobotSceneInstanceConfig,
    robot_asset: RobotAssetConfig,
    runtime_default: str,
) -> str:
    """按 env instance、robot profile、runtime default 的顺序解析 bundle 名。"""

    selected = (
        scene_instance.controller_profile
        or robot_asset.controller_profile
        or runtime_default
    )
    return normalize_controller_bundle_name(
        selected,
        label=f"controller profile for robot {scene_instance.label!r}",
    )


def _canonical_robot_scene_instance(
    *, robot_id: int, source_label: str, data: object
) -> RobotSceneInstanceConfig:
    """把 env ``robots[]`` row 转成带稠密 session ID 的 canonical instance。"""

    if not isinstance(data, Mapping):
        raise ValueError(f"{source_label} must be a mapping")
    if "robot_id" in data:
        raise ValueError(
            f"{source_label}.robot_id is generated from robots list order and "
            "must not be configured"
        )
    _reject_unsupported_keys(
        data,
        {
            "robot_profile",
            "root_pose",
            "label",
            "prim_path",
            "controller_profile",
        },
        source_label,
    )
    profile_value = data.get("robot_profile")
    robot_profile = "" if profile_value is None else str(profile_value).strip()
    if not robot_profile:
        raise ValueError(f"{source_label}.robot_profile is required")
    label_value = data.get("label")
    if label_value is not None and not isinstance(label_value, str):
        raise ValueError(f"{source_label}.label must be a string")
    label = (
        label_value.strip()
        if label_value is not None
        else f"{robot_profile}_{robot_id}"
    )
    if not label:
        raise ValueError(f"{source_label}.label cannot be empty")
    if ROBOT_INSTANCE_LABEL_PATTERN.fullmatch(label) is None:
        raise ValueError(
            f"{source_label}.label must match [A-Za-z0-9_]+, got {label!r}"
        )
    prim_path = _optional_robot_instance_prim_path(data, source_label)
    return RobotSceneInstanceConfig(
        robot_profile=robot_profile,
        root_pose=RootPoseConfig.from_mapping(
            _required_mapping(data, "root_pose", source_label)
        ),
        robot_id=robot_id,
        label=label,
        prim_path=prim_path,
        controller_profile=_optional_controller_profile(data, source_label),
    )


def _optional_controller_profile(
    data: Mapping[str, object], source_label: str
) -> str | None:
    """读取 env robot instance 的可选 controller bundle 名。"""

    value = data.get("controller_profile")
    if value is None:
        return None
    return normalize_controller_bundle_name(
        value,
        label=f"{source_label}.controller_profile",
    )


def _optional_robot_instance_prim_path(
    data: Mapping[str, object], source_label: str
) -> str | None:
    """读取可选绝对 USD prim path；省略时由稳定 label 生成默认路径。"""

    if "prim_path" not in data:
        return None
    value = data["prim_path"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source_label}.prim_path must be a non-empty string")
    prim_path = value.strip()
    if (
        not prim_path.startswith("/")
        or prim_path == "/"
        or prim_path.endswith("/")
        or "//" in prim_path
    ):
        raise ValueError(
            f"{source_label}.prim_path must be a canonical absolute USD path"
        )
    return prim_path


def _validate_robot_scene_instances(
    instances: Sequence[RobotSceneInstanceConfig],
) -> None:
    """校验 ID 稠密顺序以及 label/prim path 在当前 session 内唯一。"""

    labels: dict[str, int] = {}
    prim_paths: dict[str, int] = {}
    for expected_id, instance in enumerate(instances):
        if instance.robot_id != expected_id:
            raise ValueError(
                "robot IDs must be dense and follow robots list order: "
                f"expected {expected_id}, got {instance.robot_id}"
            )
        if instance.label in labels:
            raise ValueError(
                f"Duplicate robot label {instance.label!r} for robot IDs "
                f"{labels[instance.label]} and {instance.robot_id}"
            )
        labels[instance.label] = instance.robot_id
        prim_path = instance.effective_prim_path
        if prim_path in prim_paths:
            raise ValueError(
                f"Duplicate robot prim path {prim_path!r} for robot IDs "
                f"{prim_paths[prim_path]} and {instance.robot_id}"
            )
        prim_paths[prim_path] = instance.robot_id


def _controlled_joints_from_mapping(data: Mapping[str, object]) -> tuple[str, ...]:
    """读取非空 controlled joint selector；默认保留显式 ``all`` sentinel。"""

    value = data.get("controlled_joints", ("all",))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("controlled_joints must be a sequence")
    joints = tuple(str(name) for name in value)
    if not joints:
        raise ValueError("controlled_joints cannot be empty")
    return joints


def _required_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object]:
    """读取 instance 必需 mapping，并保留 parent label diagnostics。"""

    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _reject_unsupported_keys(
    data: Mapping[str, object], allowed: set[str], label: str
) -> None:
    """拒绝 robot instance 未声明 key，保持 env schema 单一。"""

    unsupported_keys = set(data) - allowed
    if unsupported_keys:
        unsupported = ", ".join(sorted(unsupported_keys))
        raise ValueError(f"{label} contains unsupported keys: {unsupported}")


__all__ = [
    "RobotExecutionConfig",
    "RobotSceneInstanceConfig",
    "robot_instances_from_env_config",
]
