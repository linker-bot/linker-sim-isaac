"""产品配置与 Isaac runtime 之间的纯装配规格。

Mirror/Kaleidoscope 各自在 composition root 把严格产品配置投影为这里的 dataclass；本模块
不 import 任一产品 package 或 Isaac/Omni。这样物理基础设施只消费“启动一个 session 所需
的事实”，不会反向依赖产品配置结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, TypeAlias


PhysicsKind: TypeAlias = Literal[
    "physx_cpu",
    "physx_cuda",
    "newton_cpu",
    "newton_cuda",
]
ExperienceFamily: TypeAlias = Literal["mirror", "kaleidoscope"]


@dataclass(frozen=True)
class IsaacComputeSpec:
    """Isaac session 进程唯一的 CUDA 计算设备选择。"""

    cuda_device: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.cuda_device, "compute.cuda_device")

    @property
    def device(self) -> str:
        """返回 cuRobo、RTX、Torch 与 CUDA 物理共享的规范设备名。"""

        return f"cuda:{self.cuda_device}"


@dataclass(frozen=True)
class IsaacPhysxCpuSpec:
    """Mirror 的 PhysX CPU owner 判别项。"""

    kind: Literal["physx_cpu"] = "physx_cpu"

    def __post_init__(self) -> None:
        if self.kind != "physx_cpu":
            raise ValueError("physics.kind must be physx_cpu")


@dataclass(frozen=True)
class IsaacPhysxCudaSpec:
    """Kaleidoscope 的单设备 PhysX CUDA/Torch/Fabric 判别项。"""

    enable_scene_query_support: bool = False
    kind: Literal["physx_cuda"] = "physx_cuda"

    def __post_init__(self) -> None:
        if self.kind != "physx_cuda":
            raise ValueError("physics.kind must be physx_cuda")
        if type(self.enable_scene_query_support) is not bool:
            raise TypeError("physics.enable_scene_query_support must be boolean")


@dataclass(frozen=True)
class IsaacNewtonCudaSpec:
    """项目自有 Newton CUDA runtime 的装配参数。"""

    world_count: int = 1
    nconmax_per_world: int = 200
    njmax_per_world: int = 1200
    use_cuda_graph: bool = True
    substeps: int = 1
    iterations: int = 100
    line_search_iterations: int = 50
    constraint_solver: Literal["auto", "cg", "newton"] = "auto"
    contact_pipeline: Literal["auto", "mujoco", "newton"] = "auto"
    kind: Literal["newton_cuda"] = "newton_cuda"

    def __post_init__(self) -> None:
        if self.kind != "newton_cuda":
            raise ValueError("physics.kind must be newton_cuda")
        for name in (
            "world_count",
            "nconmax_per_world",
            "njmax_per_world",
            "substeps",
            "iterations",
            "line_search_iterations",
        ):
            _require_positive_int(getattr(self, name), f"physics.{name}")
        if type(self.use_cuda_graph) is not bool:
            raise TypeError("physics.use_cuda_graph must be boolean")
        if self.constraint_solver not in {"auto", "cg", "newton"}:
            raise ValueError("physics.constraint_solver is invalid")
        if self.contact_pipeline not in {"auto", "mujoco", "newton"}:
            raise ValueError("physics.contact_pipeline is invalid")


@dataclass(frozen=True)
class IsaacNewtonCpuSpec:
    """Mirror 单 world Newton CPU runtime 的装配参数。

    CPU execution 永远 eager 执行，因此该规格故意没有 ``use_cuda_graph``。上游 CPU
    ``SolverMuJoCo`` 不提供项目所需的 independent multi-world，也不消费 Newton contacts；
    这两个限制在纯规格边界收紧，避免无效组合进入 Kit/runtime 初始化。
    """

    world_count: int = 1
    nconmax_per_world: int = 200
    njmax_per_world: int = 1200
    substeps: int = 1
    iterations: int = 100
    line_search_iterations: int = 50
    constraint_solver: Literal["auto", "cg", "newton"] = "auto"
    contact_pipeline: Literal["auto", "mujoco"] = "auto"
    kind: Literal["newton_cpu"] = "newton_cpu"

    def __post_init__(self) -> None:
        if self.kind != "newton_cpu":
            raise ValueError("physics.kind must be newton_cpu")
        for name in (
            "world_count",
            "nconmax_per_world",
            "njmax_per_world",
            "substeps",
            "iterations",
            "line_search_iterations",
        ):
            _require_positive_int(getattr(self, name), f"physics.{name}")
        if self.world_count != 1:
            raise ValueError("Newton CPU requires world_count=1")
        if self.constraint_solver not in {"auto", "cg", "newton"}:
            raise ValueError("physics.constraint_solver is invalid")
        if self.contact_pipeline not in {"auto", "mujoco"}:
            raise ValueError("Newton CPU contact_pipeline must be auto or mujoco")


IsaacPhysicsSpec: TypeAlias = (
    IsaacPhysxCpuSpec | IsaacPhysxCudaSpec | IsaacNewtonCpuSpec | IsaacNewtonCudaSpec
)


@dataclass(frozen=True)
class IsaacAppSpec:
    """Kit 进程级应用开关。

    这里只描述 ``SimulationApp`` 本身的生命周期/窗口行为，不包含任务、
    camera、viewport 几何或产品输出策略。GPU 索引也不在此重复声明：它由 session
    根部的 ``compute`` 统一持有，从而避免 Kit、RTX 与物理运行时选择不同 GPU。
    """

    gui: bool = False
    hide_ui: bool | None = None
    disable_viewport_updates: bool | None = None
    fast_shutdown: bool | None = None
    material_sync_loads: bool = False
    hydra_material_sync_loads: bool = False

    def __post_init__(self) -> None:
        _require_bool(self.gui, "app.gui")
        for name in ("hide_ui", "disable_viewport_updates", "fast_shutdown"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"app.{name} must be boolean or None")
        _require_bool(self.material_sync_loads, "app.material_sync_loads")
        _require_bool(
            self.hydra_material_sync_loads,
            "app.hydra_material_sync_loads",
        )


@dataclass(frozen=True)
class IsaacRenderSpec:
    """Kit 需要预留的 renderer 资源事实。

    ``enabled`` 表示本 session 是否需要 RTX/viewport 闭包。它不表示每个 physics
    step 都必须 render，真正的渲染 cadence 仍由产品层持有。Kaleidoscope viewport
    通过 ``visible_world_indices`` 限定可见 world，避免调试一个环境时为全部并行环境
    维护 renderer-facing USD 状态；camera/SyntheticData 仍不是该模式的能力。
    """

    enabled: bool = False
    width: int = 640
    height: int = 480
    window_width: int = 1440
    window_height: int = 900
    renderer: str = "RaytracedLighting"
    anti_aliasing: int = 0
    samples_per_pixel_per_frame: int = 1
    denoiser: bool = False
    visible_world_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "render.enabled")
        for name in ("width", "height", "window_width", "window_height"):
            _require_positive_int(getattr(self, name), f"render.{name}")
        if not isinstance(self.renderer, str) or not self.renderer.strip():
            raise ValueError("render.renderer must not be empty")
        if type(self.anti_aliasing) is not int or self.anti_aliasing < 0:
            raise ValueError("render.anti_aliasing must be a non-negative integer")
        _require_positive_int(
            self.samples_per_pixel_per_frame,
            "render.samples_per_pixel_per_frame",
        )
        if type(self.denoiser) is not bool:
            raise TypeError("render.denoiser must be boolean")
        indices = self.visible_world_indices
        if indices is not None:
            if not self.enabled:
                raise ValueError(
                    "render.visible_world_indices requires render.enabled=true"
                )
            if not isinstance(indices, tuple) or not indices:
                raise TypeError(
                    "render.visible_world_indices must be a non-empty tuple"
                )
            if any(type(index) is not int or index < 0 for index in indices):
                raise ValueError(
                    "render.visible_world_indices must contain non-negative integers"
                )
            if len(set(indices)) != len(indices):
                raise ValueError("render.visible_world_indices must be unique")


@dataclass(frozen=True)
class IsaacSessionSpec:
    """创建一个 IsaacSession 所需的完整、产品无关事实。"""

    experience_family: ExperienceFamily
    compute: IsaacComputeSpec
    physics: IsaacPhysicsSpec
    physics_dt: float
    rendering_dt: float
    gravity_z: float
    add_ground: bool = True
    ground_height: float = 0.0
    app: IsaacAppSpec = field(default_factory=IsaacAppSpec)
    render: IsaacRenderSpec = field(default_factory=IsaacRenderSpec)

    def __post_init__(self) -> None:
        if not isinstance(
            self.experience_family, str
        ) or self.experience_family not in {
            "mirror",
            "kaleidoscope",
        }:
            raise ValueError("experience_family must be mirror or kaleidoscope")
        if not isinstance(
            self.physics,
            (
                IsaacPhysxCpuSpec,
                IsaacPhysxCudaSpec,
                IsaacNewtonCpuSpec,
                IsaacNewtonCudaSpec,
            ),
        ):
            raise TypeError("physics must be an Isaac physics specification")
        if not isinstance(self.compute, IsaacComputeSpec):
            raise TypeError("compute must be IsaacComputeSpec")
        if not isinstance(self.app, IsaacAppSpec):
            raise TypeError("app must be IsaacAppSpec")
        if not isinstance(self.render, IsaacRenderSpec):
            raise TypeError("render must be IsaacRenderSpec")
        _require_positive_finite(self.physics_dt, "physics_dt")
        _require_positive_finite(self.rendering_dt, "rendering_dt")
        _require_finite(self.gravity_z, "gravity_z")
        _require_finite(self.ground_height, "ground_height")
        _require_bool(self.add_ground, "add_ground")
        if self.app.gui and not self.render.enabled:
            raise ValueError("app.gui=true requires render.enabled=true")

        kind = self.physics.kind
        if self.experience_family == "mirror":
            if kind not in {"physx_cpu", "newton_cpu", "newton_cuda"}:
                raise ValueError(
                    "Mirror session only accepts physx_cpu, newton_cpu, or newton_cuda"
                )
            if (
                isinstance(self.physics, (IsaacNewtonCpuSpec, IsaacNewtonCudaSpec))
                and self.physics.world_count != 1
            ):
                raise ValueError("Mirror Newton session requires world_count=1")
            if self.render.visible_world_indices is not None:
                raise ValueError(
                    "Mirror render does not accept replicated world selection"
                )
        elif self.experience_family == "kaleidoscope":
            if not isinstance(
                self.physics,
                (IsaacPhysxCudaSpec, IsaacNewtonCudaSpec),
            ):
                raise ValueError(
                    "Kaleidoscope session only accepts physx_cuda or newton_cuda"
                )
            if self.render.enabled and self.render.visible_world_indices is None:
                raise ValueError(
                    "Kaleidoscope viewport requires explicit visible_world_indices"
                )
            if (
                isinstance(self.physics, IsaacNewtonCudaSpec)
                and self.render.visible_world_indices is not None
                and max(self.render.visible_world_indices) >= self.physics.world_count
            ):
                raise ValueError(
                    "Kaleidoscope visible world index exceeds Newton world_count"
                )
            if (
                isinstance(self.physics, IsaacPhysxCudaSpec)
                and self.physics.enable_scene_query_support
            ):
                raise ValueError("Kaleidoscope disables PhysX scene-query support")

    @property
    def physics_kind(self) -> PhysicsKind:
        """为 diagnostics/composition 返回已验证的判别项。"""

        return self.physics.kind

    @property
    def compute_device(self) -> str:
        """返回 Kit、RTX、cuRobo 与 GPU 物理共用的 CUDA 设备。"""

        return self.compute.device

    @property
    def physics_device(self) -> str:
        """从 physics 执行方式和根 compute 选择解析物理设备。"""

        if isinstance(self.physics, (IsaacPhysxCpuSpec, IsaacNewtonCpuSpec)):
            return "cpu"
        return self.compute_device

    @property
    def physics_execution(self) -> Literal["cpu", "cuda"]:
        """返回从 strict physics spec 派生的执行类别。"""

        if isinstance(self.physics, (IsaacPhysxCpuSpec, IsaacNewtonCpuSpec)):
            return "cpu"
        return "cuda"


def _require_nonnegative_int(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _require_bool(value: object, label: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{label} must be boolean")


def _require_positive_int(value: object, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _require_finite(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    if not isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def _require_positive_finite(value: object, label: str) -> None:
    _require_finite(value, label)
    if float(value) <= 0.0:
        raise ValueError(f"{label} must be positive")


__all__ = [
    "ExperienceFamily",
    "IsaacAppSpec",
    "IsaacComputeSpec",
    "IsaacNewtonCpuSpec",
    "IsaacNewtonCudaSpec",
    "IsaacPhysicsSpec",
    "IsaacPhysxCpuSpec",
    "IsaacPhysxCudaSpec",
    "IsaacRenderSpec",
    "IsaacSessionSpec",
    "PhysicsKind",
]
