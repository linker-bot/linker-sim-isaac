"""与产品模式正交的严格物理引擎配置。

公开配置只使用 ``engine`` 与 ``execution`` 描述引擎及其执行位置。具体 CUDA 编号由
mode root 的 :class:`ComputeSettings` 持有；world 数由产品 composition 派生，因此本模块
不会复制这两类事实。Newton runtime 同样只消费这些正交字段，不再暴露实现 selector。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict, cast

from .common import (
    ConfigurationError,
    as_bool,
    as_float,
    as_int,
    as_string,
    require_keys,
    strict_mapping,
)


PhysicsEngine: TypeAlias = Literal["physx", "newton"]
PhysicsExecution: TypeAlias = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class GpuMemoryBudget:
    """构造前和稳态验收使用的进程级显存门槛。"""

    max_simulator_process_mib: int
    min_free_floor_mib: int
    min_free_fraction_after_warmup: float
    max_steady_growth_mib: int

    def __post_init__(self) -> None:
        if (
            type(self.max_simulator_process_mib) is not int
            or self.max_simulator_process_mib <= 0
        ):
            raise ConfigurationError(
                "physics.memory.max_simulator_process_mib must be a positive integer"
            )
        if type(self.min_free_floor_mib) is not int or self.min_free_floor_mib <= 0:
            raise ConfigurationError(
                "physics.memory.min_free_floor_mib must be a positive integer"
            )
        if not 0.0 < self.min_free_fraction_after_warmup <= 1.0:
            raise ConfigurationError(
                "physics.memory.min_free_fraction_after_warmup must be within (0, 1]"
            )
        if (
            type(self.max_steady_growth_mib) is not int
            or self.max_steady_growth_mib < 0
        ):
            raise ConfigurationError(
                "physics.memory.max_steady_growth_mib must be a non-negative integer"
            )

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> "GpuMemoryBudget":
        mapping = strict_mapping(value, label=label)
        required = {
            "max_simulator_process_mib",
            "min_free_floor_mib",
            "min_free_fraction_after_warmup",
            "max_steady_growth_mib",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            max_simulator_process_mib=as_int(
                mapping["max_simulator_process_mib"],
                label=f"{label}.max_simulator_process_mib",
                minimum=1,
            ),
            min_free_floor_mib=as_int(
                mapping["min_free_floor_mib"],
                label=f"{label}.min_free_floor_mib",
                minimum=1,
            ),
            min_free_fraction_after_warmup=as_float(
                mapping["min_free_fraction_after_warmup"],
                label=f"{label}.min_free_fraction_after_warmup",
                strictly_positive=True,
                maximum=1.0,
            ),
            max_steady_growth_mib=as_int(
                mapping["max_steady_growth_mib"],
                label=f"{label}.max_steady_growth_mib",
                minimum=0,
            ),
        )


@dataclass(frozen=True)
class PhysxCpuSettings:
    """PhysX CPU 的公开配置；CUDA 设备仍由 mode root 独立持有。"""

    engine: Literal["physx"]
    execution: Literal["cpu"]
    solver_type: Literal["PGS", "TGS"]

    def __post_init__(self) -> None:
        if (
            self.engine != "physx"
            or self.execution != "cpu"
            or self.solver_type not in {"PGS", "TGS"}
        ):
            raise ConfigurationError(
                "PhysxCpuSettings must use engine=physx, execution=cpu and a valid solver"
            )

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, object], *, label: str
    ) -> "PhysxCpuSettings":
        require_keys(
            mapping,
            required={"engine", "execution", "solver_type"},
            label=label,
        )
        return cls(
            engine=as_string(
                mapping["engine"], label=f"{label}.engine", choices={"physx"}
            ),  # type: ignore[arg-type]
            execution=as_string(
                mapping["execution"], label=f"{label}.execution", choices={"cpu"}
            ),  # type: ignore[arg-type]
            solver_type=as_string(
                mapping["solver_type"],
                label=f"{label}.solver_type",
                choices={"PGS", "TGS"},
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PhysxCudaSettings:
    """PhysX CUDA 后端设置；具体 GPU 只从 mode root 派生。"""

    engine: Literal["physx"]
    execution: Literal["cuda"]
    solver_type: Literal["PGS", "TGS"]
    use_fabric: bool
    enable_scene_query_support: bool
    memory: GpuMemoryBudget

    def __post_init__(self) -> None:
        if (
            self.engine != "physx"
            or self.execution != "cuda"
            or self.solver_type not in {"PGS", "TGS"}
        ):
            raise ConfigurationError(
                "PhysxCudaSettings must use engine=physx, execution=cuda and a valid solver"
            )
        if self.use_fabric is not True:
            raise ConfigurationError("PhysX CUDA must enable use_fabric=true")
        if type(self.enable_scene_query_support) is not bool:
            raise ConfigurationError(
                "physics.enable_scene_query_support must be a boolean"
            )

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, object], *, label: str
    ) -> "PhysxCudaSettings":
        required = {
            "engine",
            "execution",
            "solver_type",
            "use_fabric",
            "enable_scene_query_support",
            "memory",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            engine=as_string(
                mapping["engine"], label=f"{label}.engine", choices={"physx"}
            ),  # type: ignore[arg-type]
            execution=as_string(
                mapping["execution"], label=f"{label}.execution", choices={"cuda"}
            ),  # type: ignore[arg-type]
            solver_type=as_string(
                mapping["solver_type"],
                label=f"{label}.solver_type",
                choices={"PGS", "TGS"},
            ),  # type: ignore[arg-type]
            use_fabric=as_bool(mapping["use_fabric"], label=f"{label}.use_fabric"),
            enable_scene_query_support=as_bool(
                mapping["enable_scene_query_support"],
                label=f"{label}.enable_scene_query_support",
            ),
            memory=GpuMemoryBudget.from_mapping(
                mapping["memory"], label=f"{label}.memory"
            ),
        )


_NEWTON_COMMON_REQUIRED_KEYS = frozenset(
    {
        "nconmax_per_world",
        "njmax_per_world",
        "substeps",
        "iterations",
        "line_search_iterations",
        "constraint_solver",
        "contact_pipeline",
    }
)
_NEWTON_POSITIVE_INT_FIELDS = (
    "nconmax_per_world",
    "njmax_per_world",
    "substeps",
    "iterations",
    "line_search_iterations",
)
_NEWTON_CONSTRAINT_SOLVERS = frozenset({"auto", "cg", "newton"})
_NEWTON_CONTACT_PIPELINES = frozenset({"auto", "mujoco", "newton"})


class _NewtonCommonValues(TypedDict):
    """CPU/CUDA Newton leaf 共用字段的已验证构造参数。"""

    nconmax_per_world: int
    njmax_per_world: int
    substeps: int
    iterations: int
    line_search_iterations: int
    constraint_solver: Literal["auto", "cg", "newton"]
    contact_pipeline: Literal["auto", "mujoco", "newton"]


def _newton_common_from_mapping(
    mapping: Mapping[str, object], *, label: str
) -> _NewtonCommonValues:
    """只解析两种 execution 共有的容量与求解器字段。"""

    return _NewtonCommonValues(
        nconmax_per_world=as_int(
            mapping["nconmax_per_world"],
            label=f"{label}.nconmax_per_world",
            minimum=1,
        ),
        njmax_per_world=as_int(
            mapping["njmax_per_world"],
            label=f"{label}.njmax_per_world",
            minimum=1,
        ),
        substeps=as_int(mapping["substeps"], label=f"{label}.substeps", minimum=1),
        iterations=as_int(
            mapping["iterations"], label=f"{label}.iterations", minimum=1
        ),
        line_search_iterations=as_int(
            mapping["line_search_iterations"],
            label=f"{label}.line_search_iterations",
            minimum=1,
        ),
        constraint_solver=cast(
            Literal["auto", "cg", "newton"],
            as_string(
                mapping["constraint_solver"],
                label=f"{label}.constraint_solver",
                choices=set(_NEWTON_CONSTRAINT_SOLVERS),
            ),
        ),
        contact_pipeline=cast(
            Literal["auto", "mujoco", "newton"],
            as_string(
                mapping["contact_pipeline"],
                label=f"{label}.contact_pipeline",
                choices=set(_NEWTON_CONTACT_PIPELINES),
            ),
        ),
    )


def _validate_newton_common(value: object, *, owner: str) -> None:
    """保护直接构造 dataclass 的路径，使 CPU/CUDA 采用同一组不变量。"""

    if getattr(value, "constraint_solver") not in _NEWTON_CONSTRAINT_SOLVERS:
        raise ConfigurationError(f"{owner}.constraint_solver is invalid")
    if getattr(value, "contact_pipeline") not in _NEWTON_CONTACT_PIPELINES:
        raise ConfigurationError(f"{owner}.contact_pipeline is invalid")
    for name in _NEWTON_POSITIVE_INT_FIELDS:
        field_value = getattr(value, name)
        if type(field_value) is not int or field_value < 1:
            raise ConfigurationError(f"physics.{name} must be an integer >= 1")


@dataclass(frozen=True)
class NewtonCudaSettings:
    """Newton CUDA 的每 world 容量与求解器配置。

    环境数量不属于 physics leaf：Mirror composition 派生一个 world，Kaleidoscope
    composition 从 mode root ``environments.num_envs`` 的最终值派生 world 数。
    """

    engine: Literal["newton"]
    execution: Literal["cuda"]
    nconmax_per_world: int
    njmax_per_world: int
    use_cuda_graph: bool
    substeps: int
    iterations: int
    line_search_iterations: int
    constraint_solver: Literal["auto", "cg", "newton"]
    contact_pipeline: Literal["auto", "mujoco", "newton"]

    def __post_init__(self) -> None:
        if self.engine != "newton" or self.execution != "cuda":
            raise ConfigurationError(
                "NewtonCudaSettings must use engine=newton, execution=cuda"
            )
        _validate_newton_common(self, owner="NewtonCudaSettings")
        if type(self.use_cuda_graph) is not bool:
            raise ConfigurationError("physics.use_cuda_graph must be a boolean")

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, object], *, label: str
    ) -> "NewtonCudaSettings":
        required = _NEWTON_COMMON_REQUIRED_KEYS | {
            "engine",
            "execution",
            "use_cuda_graph",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            engine=as_string(
                mapping["engine"], label=f"{label}.engine", choices={"newton"}
            ),  # type: ignore[arg-type]
            execution=as_string(
                mapping["execution"], label=f"{label}.execution", choices={"cuda"}
            ),  # type: ignore[arg-type]
            use_cuda_graph=as_bool(
                mapping["use_cuda_graph"], label=f"{label}.use_cuda_graph"
            ),
            **_newton_common_from_mapping(mapping, label=label),
        )


@dataclass(frozen=True)
class NewtonCpuSettings:
    """Mirror Newton CPU 的严格求解配置。

    CPU leaf 不声明只对 CUDA 有意义的 ``use_cuda_graph``。Newton 1.2.1 的 CPU
    ``SolverMuJoCo`` 也不会消费项目的 Newton contact pipeline，因此显式选择
    ``contact_pipeline=newton`` 会在配置边界直接失败；``auto`` 由 runtime 收敛为 MuJoCo
    contacts。
    """

    engine: Literal["newton"]
    execution: Literal["cpu"]
    nconmax_per_world: int
    njmax_per_world: int
    substeps: int
    iterations: int
    line_search_iterations: int
    constraint_solver: Literal["auto", "cg", "newton"]
    contact_pipeline: Literal["auto", "mujoco"]

    def __post_init__(self) -> None:
        if self.engine != "newton" or self.execution != "cpu":
            raise ConfigurationError(
                "NewtonCpuSettings must use engine=newton, execution=cpu"
            )
        _validate_newton_common(self, owner="NewtonCpuSettings")
        if self.contact_pipeline == "newton":
            raise ConfigurationError(
                "Newton CPU does not support contact_pipeline=newton; use auto or mujoco"
            )

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, object], *, label: str
    ) -> "NewtonCpuSettings":
        required = _NEWTON_COMMON_REQUIRED_KEYS | {"engine", "execution"}
        require_keys(mapping, required=required, label=label)
        common = _newton_common_from_mapping(mapping, label=label)
        if common["contact_pipeline"] == "newton":
            raise ConfigurationError(
                "Newton CPU does not support contact_pipeline=newton; use auto or mujoco"
            )
        return cls(
            engine=as_string(
                mapping["engine"], label=f"{label}.engine", choices={"newton"}
            ),  # type: ignore[arg-type]
            execution=as_string(
                mapping["execution"], label=f"{label}.execution", choices={"cpu"}
            ),  # type: ignore[arg-type]
            nconmax_per_world=common["nconmax_per_world"],
            njmax_per_world=common["njmax_per_world"],
            substeps=common["substeps"],
            iterations=common["iterations"],
            line_search_iterations=common["line_search_iterations"],
            constraint_solver=common["constraint_solver"],
            contact_pipeline=common["contact_pipeline"],
        )


PhysicsSettings: TypeAlias = (
    PhysxCpuSettings | PhysxCudaSettings | NewtonCpuSettings | NewtonCudaSettings
)


def physics_settings_from_mapping(
    value: object, *, label: str = "physics"
) -> PhysicsSettings:
    """按公开的 ``engine/execution`` 二元组构造物理 strict union。"""

    mapping = strict_mapping(value, label=label)
    engine = as_string(
        mapping.get("engine"),
        label=f"{label}.engine",
        choices={"newton", "physx"},
    )
    execution = as_string(
        mapping.get("execution"),
        label=f"{label}.execution",
        choices={"cpu", "cuda"},
    )
    if engine == "physx" and execution == "cpu":
        return PhysxCpuSettings.from_mapping(mapping, label=label)
    if engine == "physx" and execution == "cuda":
        return PhysxCudaSettings.from_mapping(mapping, label=label)
    if engine == "newton" and execution == "cpu":
        return NewtonCpuSettings.from_mapping(mapping, label=label)
    if engine == "newton" and execution == "cuda":
        return NewtonCudaSettings.from_mapping(mapping, label=label)
    raise ConfigurationError(
        f"unsupported physics engine/execution combination: {engine}/{execution}"
    )


__all__ = [
    "GpuMemoryBudget",
    "NewtonCpuSettings",
    "NewtonCudaSettings",
    "PhysicsEngine",
    "PhysicsExecution",
    "PhysicsSettings",
    "PhysxCpuSettings",
    "PhysxCudaSettings",
    "physics_settings_from_mapping",
]
