"""Kaleidoscope GPU-native 并行强化学习模式的根配置与组合校验。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from ..common import (
    ConfigurationError,
    as_float_tuple,
    as_int,
    as_string,
    reject_forbidden_keys,
    require_keys,
    strict_mapping,
)
from ..curobo import CuroboProfileSettings
from ..objects import RigidObjectProfileConfig
from ..physics import NewtonCudaSettings, PhysxCudaSettings
from ..scenes import KaleidoscopeSceneSettings
from ..tasks.kaleidoscope import (
    JointControlActionSettings,
    JointDeltaActionSettings,
    KaleidoscopeTaskSettings,
)
from .common import ComputeSettings

if TYPE_CHECKING:
    from linkerbot_sim.configuration.controllers import ControllerProfiles


# 这些 key fragment 代表 Kaleidoscope runtime 闭包不拥有的产品能力。递归检查补足了
# dataclass 的 exact-key 校验，使未来新增嵌套层时也不会意外引入已退役交互字段。
KALEIDOSCOPE_FORBIDDEN_KEY_FRAGMENTS = frozenset(
    {
        "render",
        "camera",
        "transport",
        "planner",
        "planning",
        "playback",
        "telemetry",
    }
)


def validate_kaleidoscope_closure(value: object, *, label: str) -> None:
    reject_forbidden_keys(
        value,
        forbidden_fragments=KALEIDOSCOPE_FORBIDDEN_KEY_FRAGMENTS,
        label=label,
    )


@dataclass(frozen=True)
class KaleidoscopeProfileReferences:
    """RL mode 的必选 profile，以及 EE/直线 action 才允许出现的 cuRobo profile。"""

    scene: str
    physics: str
    task: str
    curobo: str | None = None

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "profiles"
    ) -> "KaleidoscopeProfileReferences":
        mapping = strict_mapping(value, label=label)
        names = {"scene", "physics", "task"}
        require_keys(mapping, required=names, optional={"curobo"}, label=label)
        return cls(
            **{
                name: as_string(mapping[name], label=f"{label}.{name}")
                for name in names
            },
            curobo=(
                None
                if "curobo" not in mapping
                else as_string(mapping["curobo"], label=f"{label}.curobo")
            ),
        )


@dataclass(frozen=True)
class KaleidoscopeEnvironmentSettings:
    """与物理引擎无关的并行环境事实，由 mode root 唯一声明。

    PhysX 的 GridCloner 布局/隔离和 Newton 的 multi-world 复制是各自后端的固定实现
    策略，不是用户可交换的 profile。这里因此只保存两个后端共同消费且能独立配置的
    环境数量、USD 命名和逻辑原点。
    """

    num_envs: int
    base_env_path: str
    env_prefix: str
    origin_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        # 这些 dataclass 也属于公开 facade，不能只依赖 YAML factory 做校验；直接构造
        # 必须遵守与 from_mapping 完全相同的 strict contract。
        as_int(self.num_envs, label="environments.num_envs", minimum=1)
        as_string(self.base_env_path, label="environments.base_env_path")
        as_string(self.env_prefix, label="environments.env_prefix")
        origin_xyz = as_float_tuple(
            self.origin_xyz,
            label="environments.origin_xyz",
            length=3,
        )
        object.__setattr__(
            self,
            "origin_xyz",
            (origin_xyz[0], origin_xyz[1], origin_xyz[2]),
        )
        if not self.base_env_path.startswith("/"):
            raise ConfigurationError("environments.base_env_path must be an absolute USD path")
        if self.base_env_path == "/":
            raise ConfigurationError(
                "environments.base_env_path must be a non-root USD container path"
            )
        if "//" in self.base_env_path:
            raise ConfigurationError(
                "environments.base_env_path must not contain an empty USD path component"
            )
        if self.base_env_path != "/" and self.base_env_path.endswith("/"):
            raise ConfigurationError("environments.base_env_path must not end with '/'")
        if (
            not self.env_prefix
            or "/" in self.env_prefix
            or self.env_prefix in {".", ".."}
        ):
            raise ConfigurationError(
                "environments.env_prefix must be a single non-empty USD path component"
            )

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "environments"
    ) -> "KaleidoscopeEnvironmentSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "num_envs",
            "base_env_path",
            "env_prefix",
            "origin_xyz",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            num_envs=as_int(mapping["num_envs"], label=f"{label}.num_envs", minimum=1),
            base_env_path=as_string(
                mapping["base_env_path"], label=f"{label}.base_env_path"
            ),
            env_prefix=as_string(mapping["env_prefix"], label=f"{label}.env_prefix"),
            origin_xyz=as_float_tuple(
                mapping["origin_xyz"], label=f"{label}.origin_xyz", length=3
            ),  # type: ignore[arg-type]
        )


KaleidoscopePhysicsSettings: TypeAlias = PhysxCudaSettings | NewtonCudaSettings


@dataclass(frozen=True)
class KaleidoscopeConfig:
    """已解析的 GPU-native RL 配置图。

    ``physics`` 只接受 CUDA execution。CUDA 设备号只在 mode root 声明，具体
    physics、Torch、policy 和训练 adapter 均从该整数派生。
    """

    mode: Literal["kaleidoscope"]
    profiles: KaleidoscopeProfileReferences
    compute: ComputeSettings
    environments: KaleidoscopeEnvironmentSettings
    scene: KaleidoscopeSceneSettings
    physics: KaleidoscopePhysicsSettings
    task: KaleidoscopeTaskSettings
    controller_bundles: Mapping[str, "ControllerProfiles"]
    curobo: CuroboProfileSettings | None = None
    sources: Mapping[str, Path] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode != "kaleidoscope":
            raise ConfigurationError(
                f"KaleidoscopeConfig.mode must be 'kaleidoscope', got {self.mode!r}"
            )
        if not isinstance(self.physics, (PhysxCudaSettings, NewtonCudaSettings)):
            raise ConfigurationError(
                "Kaleidoscope only accepts PhysX CUDA or Newton CUDA; execution must be cuda"
            )
        if self.default_controller_bundle not in self.controller_bundles:
            raise ConfigurationError(
                "Kaleidoscope physics-derived default controller bundle did not enter the resolved configuration graph: "
                f"{self.default_controller_bundle!r}"
            )
        for robot in self.scene.robots:
            if robot.resolved_profile is None:
                raise ConfigurationError(
                    "Kaleidoscope scene.robots must be bound to a strict robot profile by the catalog"
                )
        if isinstance(self.physics, PhysxCudaSettings):
            if self.physics.use_fabric is not True:
                raise ConfigurationError("Kaleidoscope PhysX must enable Fabric")
            if self.physics.enable_scene_query_support:
                raise ConfigurationError(
                    "Kaleidoscope does not establish planner/collision-query paths, "
                    "enable_scene_query_support must be false"
                )
        dynamic_rigid_names: list[str] = []
        for item in self.scene.objects:
            profile = item.resolved_profile
            if profile is None:
                raise ConfigurationError(
                    "Kaleidoscope scene.objects must be bound to a strict object profile by the catalog"
                )
            if not isinstance(profile, RigidObjectProfileConfig):
                raise ConfigurationError(
                    "Kaleidoscope state schema supports only one dynamic rigid body, not dynamic_chain"
                )
            if not profile.physics.static:
                dynamic_rigid_names.append(item.name)
        if len(dynamic_rigid_names) != 1:
            raise ConfigurationError(
                "Kaleidoscope scene must contain exactly one non-static rigid object, "
                f"got {dynamic_rigid_names}"
            )
        if dynamic_rigid_names[0] != self.task.dynamic_object:
            raise ConfigurationError(
                "task.dynamic_object must name the unique non-static rigid object in the scene: "
                f"expected {dynamic_rigid_names[0]!r}, got {self.task.dynamic_object!r}"
            )
        needs_kinematics = not isinstance(
            self.task.action,
            (JointControlActionSettings, JointDeltaActionSettings),
        )
        if (self.profiles.curobo is None) != (self.curobo is None):
            raise ConfigurationError(
                "profiles.curobo and the resolved cuRobo profile must both be present or both be absent"
            )
        if needs_kinematics != (self.curobo is not None):
            raise ConfigurationError(
                "only EE/linear action must reference a cuRobo profile from the mode root; "
                "joint_control/joint_delta must not load an unused cuRobo configuration"
            )
        if self.curobo is not None:
            if self.curobo.motion_planner is not None:
                raise ConfigurationError(
                    "Kaleidoscope cuRobo profile must not declare motion_planner"
                )
            kinematics = self.curobo.kinematics
            if kinematics.collision_check:
                raise ConfigurationError(
                    "Kaleidoscope batch IK/linear motion does not assemble a collision world, "
                    "curobo.kinematics.collision_check must be false"
                )

    @property
    def cuda_device(self) -> int:
        return self.compute.cuda_device

    @property
    def torch_device(self) -> str:
        return f"cuda:{self.compute.cuda_device}"

    @property
    def default_controller_bundle(self) -> Literal["physx", "newton"]:
        """由物理引擎派生复制场景资产使用的默认 drive 标定。"""

        return self.physics.engine


def kaleidoscope_mode_from_mapping(
    value: object, *, label: str = "mode config"
) -> tuple[
    Literal["kaleidoscope"],
    KaleidoscopeProfileReferences,
    ComputeSettings,
    KaleidoscopeEnvironmentSettings,
]:
    validate_kaleidoscope_closure(value, label=label)
    mapping = strict_mapping(value, label=label)
    require_keys(
        mapping,
        required={"mode", "profiles", "compute", "environments"},
        label=label,
    )
    mode = as_string(mapping["mode"], label=f"{label}.mode", choices={"kaleidoscope"})
    return (
        mode,  # type: ignore[return-value]
        KaleidoscopeProfileReferences.from_mapping(
            mapping["profiles"], label=f"{label}.profiles"
        ),
        ComputeSettings.from_mapping(mapping["compute"], label=f"{label}.compute"),
        KaleidoscopeEnvironmentSettings.from_mapping(
            mapping["environments"], label=f"{label}.environments"
        ),
    )


__all__ = [
    "KaleidoscopeConfig",
    "KaleidoscopeEnvironmentSettings",
    "KaleidoscopePhysicsSettings",
    "KaleidoscopeProfileReferences",
    "kaleidoscope_mode_from_mapping",
    "validate_kaleidoscope_closure",
]
