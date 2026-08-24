"""cuRobo 数值后端的纯配置类型。

本模块只描述 cuRobo 的 IK 与 MotionPlanner 可调资源，不导入 cuRobo/Torch，也不拥有
``compute.cuda_device``。固定 task bundle 与 float32 dtype 属于 backend 合同。Mirror 的
请求默认值属于 planning 领域；Kaleidoscope 是否启用 EE action 属于 task/mode
composition，二者都不进入这里。
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import (
    ConfigurationError,
    as_bool,
    as_int,
    require_keys,
    strict_mapping,
)


@dataclass(frozen=True)
class CuroboCollisionCacheSettings:
    """cuRobo 场景碰撞缓存的固定容量。"""

    cuboid: int
    mesh: int

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str
    ) -> "CuroboCollisionCacheSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(mapping, required={"cuboid", "mesh"}, label=label)
        return cls(
            cuboid=as_int(mapping["cuboid"], label=f"{label}.cuboid", minimum=0),
            mesh=as_int(mapping["mesh"], label=f"{label}.mesh", minimum=0),
        )

    def as_backend_mapping(self) -> dict[str, int]:
        """返回后端 schema 使用的独立 mapping。"""

        return {"cuboid": self.cuboid, "mesh": self.mesh}


@dataclass(frozen=True)
class CuroboKinematicsSettings:
    """FK/IK 数值能力；不携带 backend selector 或 CUDA 设备编号。"""

    max_batch_size: int
    seed_count: int
    collision_check: bool
    use_cuda_graph: bool
    collision_cache: CuroboCollisionCacheSettings | None

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "curobo.kinematics"
    ) -> "CuroboKinematicsSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "max_batch_size",
            "seed_count",
            "collision_check",
            "use_cuda_graph",
        }
        require_keys(
            mapping,
            required=required,
            optional={"collision_cache"},
            label=label,
        )
        collision_check = as_bool(
            mapping["collision_check"], label=f"{label}.collision_check"
        )
        cache_raw = mapping.get("collision_cache")
        if collision_check and cache_raw is None:
            raise ConfigurationError(
                f"{label}.collision_cache must be declared when collision_check=true"
            )
        return cls(
            max_batch_size=as_int(
                mapping["max_batch_size"], label=f"{label}.max_batch_size", minimum=1
            ),
            seed_count=as_int(
                mapping["seed_count"], label=f"{label}.seed_count", minimum=1
            ),
            collision_check=collision_check,
            use_cuda_graph=as_bool(
                mapping["use_cuda_graph"], label=f"{label}.use_cuda_graph"
            ),
            collision_cache=(
                None
                if cache_raw is None
                else CuroboCollisionCacheSettings.from_mapping(
                    cache_raw, label=f"{label}.collision_cache"
                )
            ),
        )


@dataclass(frozen=True)
class CuroboMotionPlannerSettings:
    """cuRobo 单请求 MotionPlanner 的数值与碰撞能力。"""

    warmup: bool
    use_cuda_graph: bool
    ik_seed_count: int
    trajectory_seed_count: int
    collision_check: bool
    collision_cache: CuroboCollisionCacheSettings | None

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "curobo.motion_planner"
    ) -> "CuroboMotionPlannerSettings":
        mapping = strict_mapping(value, label=label)
        required = {
            "warmup",
            "use_cuda_graph",
            "ik_seed_count",
            "trajectory_seed_count",
            "collision_check",
        }
        require_keys(
            mapping,
            required=required,
            optional={"collision_cache"},
            label=label,
        )
        collision_check = as_bool(
            mapping["collision_check"], label=f"{label}.collision_check"
        )
        cache_raw = mapping.get("collision_cache")
        if collision_check and cache_raw is None:
            raise ConfigurationError(
                f"{label}.collision_cache must be declared when collision_check=true"
            )
        return cls(
            warmup=as_bool(mapping["warmup"], label=f"{label}.warmup"),
            use_cuda_graph=as_bool(
                mapping["use_cuda_graph"], label=f"{label}.use_cuda_graph"
            ),
            ik_seed_count=as_int(
                mapping["ik_seed_count"], label=f"{label}.ik_seed_count", minimum=1
            ),
            trajectory_seed_count=as_int(
                mapping["trajectory_seed_count"],
                label=f"{label}.trajectory_seed_count",
                minimum=1,
            ),
            collision_check=collision_check,
            collision_cache=(
                None
                if cache_raw is None
                else CuroboCollisionCacheSettings.from_mapping(
                    cache_raw, label=f"{label}.collision_cache"
                )
            ),
        )


@dataclass(frozen=True)
class CuroboProfileSettings:
    """一个 mode 可引用的完整 cuRobo 数值 profile。

    Mirror 要求 ``motion_planner`` 存在；Kaleidoscope 要求它不存在。该差异由各产品根
    校验，避免通用 parser 依赖产品包。
    """

    kinematics: CuroboKinematicsSettings
    motion_planner: CuroboMotionPlannerSettings | None = None

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "curobo"
    ) -> "CuroboProfileSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(
            mapping,
            required={"kinematics"},
            optional={"motion_planner"},
            label=label,
        )
        planner_raw = mapping.get("motion_planner")
        return cls(
            kinematics=CuroboKinematicsSettings.from_mapping(
                mapping["kinematics"], label=f"{label}.kinematics"
            ),
            motion_planner=(
                None
                if planner_raw is None
                else CuroboMotionPlannerSettings.from_mapping(
                    planner_raw, label=f"{label}.motion_planner"
                )
            ),
        )


__all__ = [
    "CuroboCollisionCacheSettings",
    "CuroboKinematicsSettings",
    "CuroboMotionPlannerSettings",
    "CuroboProfileSettings",
]
