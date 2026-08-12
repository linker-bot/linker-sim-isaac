"""Mirror 后端中立的 planning 请求默认策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .common import as_bool, as_float, as_string, require_keys, strict_mapping


@dataclass(frozen=True)
class MirrorPlanningRequestDefaults:
    """未在单次 Mirror motion 请求中覆盖时使用的默认值。"""

    duration_s: float
    sample_dt_s: float
    timeout_s: float
    avoid_collisions: bool
    force_collision_refresh: bool
    coordination: Literal["independent"]

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "planning.request_defaults"
    ) -> "MirrorPlanningRequestDefaults":
        mapping = strict_mapping(value, label=label)
        required = {
            "duration_s",
            "sample_dt_s",
            "timeout_s",
            "avoid_collisions",
            "force_collision_refresh",
            "coordination",
        }
        require_keys(mapping, required=required, label=label)
        return cls(
            duration_s=as_float(
                mapping["duration_s"],
                label=f"{label}.duration_s",
                strictly_positive=True,
            ),
            sample_dt_s=as_float(
                mapping["sample_dt_s"],
                label=f"{label}.sample_dt_s",
                strictly_positive=True,
            ),
            timeout_s=as_float(
                mapping["timeout_s"], label=f"{label}.timeout_s", strictly_positive=True
            ),
            avoid_collisions=as_bool(
                mapping["avoid_collisions"], label=f"{label}.avoid_collisions"
            ),
            force_collision_refresh=as_bool(
                mapping["force_collision_refresh"],
                label=f"{label}.force_collision_refresh",
            ),
            coordination=as_string(
                mapping["coordination"],
                label=f"{label}.coordination",
                choices={"independent"},
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class MirrorPlanningSettings:
    """Mirror planning 领域配置，不包含具体数值后端参数。"""

    request_defaults: MirrorPlanningRequestDefaults

    @classmethod
    def from_mapping(
        cls, value: object, *, label: str = "planning"
    ) -> "MirrorPlanningSettings":
        mapping = strict_mapping(value, label=label)
        require_keys(mapping, required={"request_defaults"}, label=label)
        return cls(
            request_defaults=MirrorPlanningRequestDefaults.from_mapping(
                mapping["request_defaults"], label=f"{label}.request_defaults"
            )
        )


__all__ = ["MirrorPlanningRequestDefaults", "MirrorPlanningSettings"]
