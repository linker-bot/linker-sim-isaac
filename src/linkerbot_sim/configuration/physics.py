"""与产品模式正交的严格物理引擎配置。

公开配置只使用 ``engine`` 与 ``execution`` 描述引擎及其执行位置。具体 CUDA 编号由
mode root 的 :class:`ComputeSettings` 持有；world 数由产品 composition 派生，因此本模块
不会复制这两类事实。Newton runtime 同样只消费这些正交字段，不再暴露实现 selector。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Literal, TypeAlias, TypedDict, cast

from .common import (
    ConfigurationError,
    as_bool,
    as_float,
    as_float_tuple,
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


@dataclass(frozen=True)
class NewtonContactMaterial:
    """One collision role; MuJoCo mixes the two colliding materials."""

    friction: tuple[float, ...]
    solimp: tuple[float, ...]
    condim: int

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> NewtonContactMaterial:
        data = strict_mapping(value, label=label)
        require_keys(data, required={"friction", "solimp", "condim"}, label=label)
        friction = as_float_tuple(data["friction"], label=f"{label}.friction", length=3)
        solimp = as_float_tuple(data["solimp"], label=f"{label}.solimp", length=5)
        condim = as_int(data["condim"], label=f"{label}.condim")
        if any(v < 0 for v in friction):
            raise ConfigurationError(f"{label}.friction must be non-negative")
        if not (
            0 < solimp[0] <= solimp[1] < 1
            and solimp[2] > 0
            and 0 < solimp[3] < 1
            and solimp[4] >= 1
        ):
            raise ConfigurationError(f"{label}.solimp is outside the supported range")
        if condim not in {1, 3, 4, 6}:
            raise ConfigurationError(f"{label}.condim must be 1, 3, 4 or 6")
        return cls(friction=friction, solimp=solimp, condim=condim)


@dataclass(frozen=True)
class NewtonContactSettings:
    """AR5/O6-scoped material policy and scene-wide friction solver options."""

    cone: str
    impratio: float
    solref: tuple[float, ...]
    friction_gain: float
    margin_m: float
    restitution: float
    hand: NewtonContactMaterial
    object: NewtonContactMaterial
    station: NewtonContactMaterial

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> NewtonContactSettings:
        data = strict_mapping(value, label=label)
        require_keys(
            data,
            required={
                "cone",
                "impratio",
                "solref",
                "friction_gain",
                "margin_m",
                "restitution",
                "hand",
                "object",
                "station",
            },
            label=label,
        )
        solref = as_float_tuple(data["solref"], label=f"{label}.solref", length=2)
        if any(v <= 0 for v in solref):
            raise ConfigurationError(
                f"{label}.solref requires positive time constant and damping ratio"
            )
        return cls(
            cone=as_string(
                data["cone"], label=f"{label}.cone", choices={"elliptic", "pyramidal"}
            ),
            impratio=as_float(
                data["impratio"], label=f"{label}.impratio", strictly_positive=True
            ),
            solref=solref,
            friction_gain=as_float(
                data["friction_gain"], label=f"{label}.friction_gain", minimum=0
            ),
            margin_m=as_float(data["margin_m"], label=f"{label}.margin_m", minimum=0),
            restitution=as_float(
                data["restitution"], label=f"{label}.restitution", minimum=0, maximum=1
            ),
            hand=NewtonContactMaterial.from_mapping(
                data["hand"], label=f"{label}.hand"
            ),
            object=NewtonContactMaterial.from_mapping(
                data["object"], label=f"{label}.object"
            ),
            station=NewtonContactMaterial.from_mapping(
                data["station"], label=f"{label}.station"
            ),
        )


_NEWTON_COMMON_REQUIRED_KEYS = frozenset(
    {
        "nconmax_per_world",
        "njmax_per_world",
        "default_contact_time_constant_s",
        "iterations",
        "line_search_iterations",
        "constraint_solver",
        "default_contact_gap_m",
        "solver_deterministic",
        "collision_deterministic",
        "contact_matching",
    }
)
_NEWTON_POSITIVE_INT_FIELDS = (
    "nconmax_per_world",
    "njmax_per_world",
    "iterations",
    "line_search_iterations",
)
_NEWTON_CONSTRAINT_SOLVERS = frozenset({"auto", "cg", "newton"})
_NEWTON_CONTACT_MATCHING = frozenset({"disabled", "latest", "sticky"})


class _NewtonCommonValues(TypedDict):
    """CPU/CUDA Newton leaf 共用字段的已验证构造参数。"""

    nconmax_per_world: int
    njmax_per_world: int
    default_contact_time_constant_s: float
    iterations: int
    line_search_iterations: int
    constraint_solver: Literal["auto", "cg", "newton"]
    default_contact_gap_m: float
    solver_deterministic: bool
    collision_deterministic: bool
    contact_matching: Literal["disabled", "latest", "sticky"]
    ar5_o6_contact: NewtonContactSettings | None


def _newton_common_from_mapping(
    mapping: Mapping[str, object], *, label: str
) -> _NewtonCommonValues:
    """只解析两种 execution 共有的容量与求解器字段。"""

    return _NewtonCommonValues(
        ar5_o6_contact=(
            NewtonContactSettings.from_mapping(
                mapping["ar5_o6_contact"], label=f"{label}.ar5_o6_contact"
            )
            if mapping.get("ar5_o6_contact") is not None
            else None
        ),
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
        default_contact_time_constant_s=as_float(
            mapping["default_contact_time_constant_s"],
            label=f"{label}.default_contact_time_constant_s",
            strictly_positive=True,
        ),
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
        default_contact_gap_m=as_float(
            mapping["default_contact_gap_m"],
            label=f"{label}.default_contact_gap_m",
            minimum=0.0,
        ),
        solver_deterministic=as_bool(
            mapping["solver_deterministic"], label=f"{label}.solver_deterministic"
        ),
        collision_deterministic=as_bool(
            mapping["collision_deterministic"], label=f"{label}.collision_deterministic"
        ),
        contact_matching=cast(
            Literal["disabled", "latest", "sticky"],
            as_string(
                mapping["contact_matching"],
                label=f"{label}.contact_matching",
                choices=set(_NEWTON_CONTACT_MATCHING),
            ),
        ),
    )


def _validate_newton_common(value: object, *, owner: str) -> None:
    """保护直接构造 dataclass 的路径，使 CPU/CUDA 采用同一组不变量。"""

    if getattr(value, "constraint_solver") not in _NEWTON_CONSTRAINT_SOLVERS:
        raise ConfigurationError(f"{owner}.constraint_solver is invalid")
    if getattr(value, "contact_matching") not in _NEWTON_CONTACT_MATCHING:
        raise ConfigurationError(f"{owner}.contact_matching is invalid")
    for name in ("solver_deterministic", "collision_deterministic"):
        if type(getattr(value, name)) is not bool:
            raise ConfigurationError(f"physics.{name} must be a boolean")
    gap = getattr(value, "default_contact_gap_m")
    if (
        isinstance(gap, bool)
        or not isinstance(gap, (int, float))
        or not math.isfinite(gap)
        or gap < 0
    ):
        raise ConfigurationError(
            "physics.default_contact_gap_m must be finite and >= 0"
        )
    for name in ("default_contact_time_constant_s",):
        value_s = getattr(value, name)
        if (
            isinstance(value_s, bool)
            or not isinstance(value_s, (int, float))
            or not math.isfinite(value_s)
            or value_s <= 0.0
        ):
            raise ConfigurationError(f"physics.{name} must be finite and positive")
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
    default_contact_time_constant_s: float
    iterations: int
    line_search_iterations: int
    constraint_solver: Literal["auto", "cg", "newton"]
    default_contact_gap_m: float
    solver_deterministic: bool
    collision_deterministic: bool
    contact_matching: Literal["disabled", "latest", "sticky"]
    ar5_o6_contact: NewtonContactSettings | None = None

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
        require_keys(
            mapping, required=required, optional={"ar5_o6_contact"}, label=label
        )
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
    """Mirror 单 world Newton 碰撞 + MuJoCo Warp CPU；不使用 CUDA graph。"""

    engine: Literal["newton"]
    execution: Literal["cpu"]
    nconmax_per_world: int
    njmax_per_world: int
    default_contact_time_constant_s: float
    iterations: int
    line_search_iterations: int
    constraint_solver: Literal["auto", "cg", "newton"]
    default_contact_gap_m: float
    solver_deterministic: bool
    collision_deterministic: bool
    contact_matching: Literal["disabled", "latest", "sticky"]
    ar5_o6_contact: NewtonContactSettings | None = None

    def __post_init__(self) -> None:
        if self.engine != "newton" or self.execution != "cpu":
            raise ConfigurationError(
                "NewtonCpuSettings must use engine=newton, execution=cpu"
            )
        _validate_newton_common(self, owner="NewtonCpuSettings")

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, object], *, label: str
    ) -> "NewtonCpuSettings":
        required = _NEWTON_COMMON_REQUIRED_KEYS | {"engine", "execution"}
        require_keys(
            mapping, required=required, optional={"ar5_o6_contact"}, label=label
        )
        common = _newton_common_from_mapping(mapping, label=label)
        return cls(
            engine=as_string(
                mapping["engine"], label=f"{label}.engine", choices={"newton"}
            ),  # type: ignore[arg-type]
            execution=as_string(
                mapping["execution"], label=f"{label}.execution", choices={"cpu"}
            ),  # type: ignore[arg-type]
            **common,
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
