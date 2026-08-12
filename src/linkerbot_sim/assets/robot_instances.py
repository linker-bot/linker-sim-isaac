"""场景机器人实例身份与单 articulation 执行配置。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from linkerbot_sim.assets.robot_config import RobotAssetConfig
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configuration.controllers import normalize_controller_bundle_name
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.configuration.scenes import RobotInstanceSettings


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
        if (
            isinstance(self.robot_id, bool)
            or not isinstance(self.robot_id, int)
            or self.robot_id < 0
        ):
            raise ValueError("robot_id must be a non-negative integer")
        if not isinstance(self.robot_profile, str) or not self.robot_profile:
            raise ValueError("robot_profile must be a non-empty string")
        if (
            not isinstance(self.label, str)
            or ROBOT_INSTANCE_LABEL_PATTERN.fullmatch(self.label) is None
        ):
            raise ValueError("robot label must match [A-Za-z0-9_]+")
        if self.prim_path is not None and not _canonical_prim_path(self.prim_path):
            raise ValueError("robot prim_path must be a canonical absolute USD path")
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
    def from_profile(
        cls,
        profile: RobotProfileSettings,
        *,
        scene_instance: RobotSceneInstanceConfig,
    ) -> "RobotExecutionConfig":
        """把 robot profile 与唯一场景实例组合成执行配置。"""

        robot = RobotAssetConfig.from_profile(
            profile,
            prim_path=scene_instance.effective_prim_path,
            name=scene_instance.label,
        )
        return cls(
            robot=robot,
            controlled_joints=profile.controlled_joints,
            root_pose=scene_instance.root_pose,
        )


def robot_scene_instances_from_settings(
    values: Sequence[RobotInstanceSettings],
) -> tuple[RobotSceneInstanceConfig, ...]:
    """把 catalog 已解析的 scene robot identity 投影为稠密执行实例。"""

    if not all(isinstance(item, RobotInstanceSettings) for item in values):
        raise TypeError("values must contain RobotInstanceSettings")
    return tuple(
        RobotSceneInstanceConfig(
            robot_id=index,
            label=item.label,
            robot_profile=item.robot_profile,
            root_pose=RootPoseConfig(xyz=item.root_pose.xyz, rpy=item.root_pose.rpy),
            controller_profile=item.controller_profile,
        )
        for index, item in enumerate(values)
    )


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


def _canonical_prim_path(value: object) -> bool:
    """返回值是否为非根、无空段且无尾斜线的绝对 USD path。"""

    return bool(
        isinstance(value, str)
        and value.startswith("/")
        and value != "/"
        and not value.endswith("/")
        and "//" not in value
    )


__all__ = [
    "RobotExecutionConfig",
    "RobotSceneInstanceConfig",
    "robot_scene_instances_from_settings",
    "resolve_controller_profile",
]
