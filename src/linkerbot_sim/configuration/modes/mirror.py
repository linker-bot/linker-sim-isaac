"""Mirror 单世界现实映像模式的严格根配置与组合校验。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..common import ConfigurationError, as_string, require_keys, strict_mapping
from ..control import HybridForcePositionSettings, MirrorControlSettings
from ..curobo import CuroboProfileSettings
from ..outputs import MirrorOutputsSettings
from ..physics import NewtonCpuSettings, NewtonCudaSettings, PhysxCpuSettings
from ..planning import MirrorPlanningSettings
from ..scenes import MirrorSceneSettings
from .common import ComputeSettings

if TYPE_CHECKING:
    from linkerbot_sim.configuration.controllers import ControllerProfiles


MirrorPhysicsSettings = PhysxCpuSettings | NewtonCpuSettings | NewtonCudaSettings


@dataclass(frozen=True)
class MirrorProfileReferences:
    """Mirror mode YAML 中允许出现的全部 profile 引用。"""

    scene: str
    physics: str
    control: str
    curobo: str
    planning: str
    outputs: str
    hybrid_control: str | None = None

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "profiles"
    ) -> "MirrorProfileReferences":
        mapping = strict_mapping(value, label=label)
        names = {"scene", "physics", "control", "curobo", "planning", "outputs"}
        require_keys(
            mapping,
            required=names,
            optional={"hybrid_control"},
            label=label,
        )
        return cls(
            **{
                name: as_string(mapping[name], label=f"{label}.{name}")
                for name in names
            },
            hybrid_control=(
                as_string(mapping["hybrid_control"], label=f"{label}.hybrid_control")
                if "hybrid_control" in mapping
                else None
            ),
        )


@dataclass(frozen=True)
class MirrorConfig:
    """Mirror 完整、不可变且已解析的配置图。

    ``physics`` 是判别 union，而不是一组可同时出现的 backend optional 字段。底层
    Newton runtime 仍可支持多 world；本产品在 session 投影时只派生一个 world。
    """

    mode: Literal["mirror"]
    profiles: MirrorProfileReferences
    compute: ComputeSettings
    scene: MirrorSceneSettings
    physics: MirrorPhysicsSettings
    control: MirrorControlSettings
    curobo: CuroboProfileSettings
    planning: MirrorPlanningSettings
    outputs: MirrorOutputsSettings
    hybrid_control: HybridForcePositionSettings | None
    controller_bundles: Mapping[str, "ControllerProfiles"]
    sources: Mapping[str, Path] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode != "mirror":
            raise ConfigurationError(
                f"MirrorConfig.mode must be 'mirror', got {self.mode!r}"
            )
        if not isinstance(
            self.physics,
            (PhysxCpuSettings, NewtonCpuSettings, NewtonCudaSettings),
        ):
            raise ConfigurationError(
                "Mirror only accepts PhysX CPU, Newton CPU, or Newton CUDA; "
                "PhysX CUDA is not yet in the Mirror support matrix"
            )
        if self.default_controller_bundle not in self.controller_bundles:
            raise ConfigurationError(
                "Mirror physics-derived default controller bundle did not enter the resolved configuration graph: "
                f"{self.default_controller_bundle!r}"
            )
        planner = self.curobo.motion_planner
        if planner is None:
            raise ConfigurationError("Mirror cuRobo profile must declare motion_planner")
        if planner.use_cuda_graph:
            raise ConfigurationError(
                "Mirror cuRobo motion_planner.use_cuda_graph must be false: "
                "the project-pinned cuRobo 0.8 runtime does not enable experimental CUDA graph reset, "
                "and Mirror does not rely on this global switch"
            )
        if (
            self.planning.request_defaults.avoid_collisions
            and not planner.collision_check
        ):
            raise ConfigurationError(
                "planning.request_defaults.avoid_collisions=true requires "
                "curobo.motion_planner.collision_check=true"
            )
        for robot in self.scene.robots:
            if robot.resolved_profile is None:
                raise ConfigurationError(
                    "Mirror scene.robots must be bound to a strict robot profile by the catalog"
                )
        if self.outputs.camera.enabled and not self.scene.cameras:
            raise ConfigurationError(
                "scene must declare at least one camera when outputs.camera.enabled=true"
            )
        if self.hybrid_control is not None:
            self._validate_hybrid_control()

    def _validate_hybrid_control(self) -> None:
        hybrid = self.hybrid_control
        assert hybrid is not None
        if not isinstance(self.physics, PhysxCpuSettings):
            raise ConfigurationError("hybrid_control only supports Mirror PhysX CPU in the first phase")
        if self.control.mode != "position":
            raise ConfigurationError(
                "hybrid_control requires the initial Mirror control.mode=position"
            )
        frequency = float(self.scene.physics_frequency_hz)
        if frequency < hybrid.minimum_physics_frequency_hz:
            raise ConfigurationError(
                "hybrid_control physics frequency is too low: "
                f"{frequency} < {hybrid.minimum_physics_frequency_hz}"
            )
        if hybrid.force.wrench_lpf_cutoff_hz >= frequency / 2.0:
            raise ConfigurationError(
                "hybrid_control wrench_lpf_cutoff_hz must be less than the physics Nyquist"
            )
        minimum_ramp_ticks = math.ceil(
            hybrid.limits.max_abs_joint_effort
            / (hybrid.limits.max_joint_effort_rate / frequency)
            - 1.0e-12
        )
        if hybrid.limits.ramp_down_ticks < minimum_ramp_ticks:
            raise ConfigurationError(
                "hybrid_control ramp_down_ticks is insufficient to reach zero within the effort-rate limit: "
                f"{hybrid.limits.ramp_down_ticks} < {minimum_ramp_ticks}"
            )
        for instance in self.scene.robots:
            profile = instance.resolved_profile
            assert profile is not None
            if not profile.joint_groups.arm:
                raise ConfigurationError(
                    f"hybrid robot {instance.label!r} must have a non-empty arm joint group"
                )
            if profile.gravity_policy.enabled_for_component("arm"):
                raise ConfigurationError(
                    f"hybrid robot {instance.label!r} must disable arm gravity in the first phase"
                )
            if not profile.curobo.binding.enabled or profile.curobo.robot is None:
                raise ConfigurationError(
                    f"hybrid robot {instance.label!r} is missing a physical TCP model"
                )
        for name, profiles in self.controller_bundles.items():
            arm = profiles.arm.effort_control
            hand = profiles.hand.position_control
            default_profile = (
                profiles.arm if profiles.default is None else profiles.default
            )
            default = default_profile.position_control
            if (arm.mode, arm.method) != ("effort", "direct"):
                raise ConfigurationError(
                    f"hybrid controller bundle {name!r} arm must be effort+direct"
                )
            if (hand.mode, hand.method) != ("position", "implicit"):
                raise ConfigurationError(
                    f"hybrid controller bundle {name!r} hand must be position+implicit"
                )
            if (default.mode, default.method) != ("position", "implicit"):
                raise ConfigurationError(
                    f"hybrid controller bundle {name!r} default must be position+implicit"
                )

    @property
    def cuda_device(self) -> int:
        """返回 cuRobo、RTX 与 Newton CUDA 可共同消费的唯一 CUDA 设备编号。"""

        return self.compute.cuda_device

    @property
    def torch_device(self) -> str:
        return f"cuda:{self.cuda_device}"

    @property
    def default_controller_bundle(self) -> Literal["physx", "newton"]:
        """由物理引擎派生资产导入时使用的默认 drive 标定。"""

        return self.physics.engine


def mirror_mode_from_mapping(
    value: object, *, label: str = "mode config"
) -> tuple[Literal["mirror"], MirrorProfileReferences, ComputeSettings]:
    """解析只包含模式名和 profile 引用的 Mirror mode 文件。"""

    mapping = strict_mapping(value, label=label)
    require_keys(
        mapping,
        required={"mode", "profiles", "compute"},
        label=label,
    )
    mode = as_string(mapping["mode"], label=f"{label}.mode", choices={"mirror"})
    compute = ComputeSettings.from_mapping(mapping["compute"], label=f"{label}.compute")
    return (
        mode,  # type: ignore[return-value]
        MirrorProfileReferences.from_mapping(
            mapping["profiles"], label=f"{label}.profiles"
        ),
        compute,
    )


__all__ = [
    "MirrorConfig",
    "MirrorPhysicsSettings",
    "MirrorProfileReferences",
    "mirror_mode_from_mapping",
]
