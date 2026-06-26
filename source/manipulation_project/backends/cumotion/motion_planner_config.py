"""cuMotion motion planner 的分组配置模型。

该模块定义 ``CuMotionMotionPlanner`` facade 使用的配置结构。顶层只负责选择规划
pipeline，各 pipeline 的参数分别放在作用域清晰的 dataclass 中：

* ``graph_search`` 只保存 graph ``MotionPlanner`` 相关参数。
* ``trajectory_generation`` 只保存 ``CSpaceTrajectoryGenerator`` 的时间参数化参数。
* ``trajectory_optimization`` 只保存 ``TrajectoryOptimizer`` 相关参数。
* ``specified_path`` 只保存指定路径族的默认行为。

这样可以避免一个参数名在不同 pipeline 中产生歧义。例如
``generate_interpolated_path`` 只属于 graph search，而不会影响 optimizer 或 specified path。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from manipulation_project.utils.paths import repo_path


PlanningPipeline = Literal[
    "graph_search",
    "specified_path",
    "trajectory_optimization",
]
TrajectoryGenerationMode = Literal["time_optimal", "time_stamped"]
TrajectoryInterpolationMode = Literal["linear", "cubic_spline"]
SpecifiedPathFamily = Literal["cspace_waypoints", "task_space_segments", "composite"]


@dataclass(frozen=True)
class GraphSearchConfig:
    """graph-based ``MotionPlanner`` 专属配置。

    ``generate_interpolated_path`` 只控制 cuMotion graph planner 是否返回/优先消费
    ``interpolated_path``；它不是 trajectory generator 的插值方式。
    ``use_environment_obstacles`` 为假时使用 context 的空 world view，不会清空已经同步到
    context 的真实环境。
    """

    generate_interpolated_path: bool = True
    use_environment_obstacles: bool = True
    motion_planner_config_path: Path | None = None
    motion_planner_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryGenerationConfig:
    """C-space path 时间参数化配置。

    该配置只服务于会先产生 ``joint_path`` 的 pipeline，例如 ``graph_search`` 和
    ``specified_path``。``trajectory_optimization`` 的主输出是 cuMotion ``Trajectory``。
    """

    enabled: bool = True
    mode: TrajectoryGenerationMode = "time_optimal"
    interpolation_mode: TrajectoryInterpolationMode = "cubic_spline"
    limits: Mapping[str, Any] = field(default_factory=dict)
    solver_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryOptimizationConfig:
    """cuMotion ``TrajectoryOptimizer`` 专属配置。

    optimizer pipeline 只执行 trajectory optimizer 本身。失败时返回 optimizer 的失败结果；
    如果调用方需要其它路线，应在任务层显式发起第二次规划。
    """

    config_path: Path | None = None
    use_environment_obstacles: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpecifiedPathConfig:
    """调用方指定路径族的配置。

    ``family`` 表示默认指定路径族。真实路径几何在 ``SpecifiedPathRequest.path`` 中表达，而不
    放在 config 中。当前实现支持 ``cspace_waypoints``；``task_space_segments`` 和
    ``composite`` 作为显式路径族保留在 schema 中，由 planner 在请求边界返回未实现错误。
    """

    family: SpecifiedPathFamily = "task_space_segments"
    validate_collision_after_generation: bool = False
    cspace_waypoints: Mapping[str, Any] = field(default_factory=dict)
    task_space_segments: Mapping[str, Any] = field(default_factory=dict)
    composite: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionPlannerBackendConfig:
    """``CuMotionMotionPlanner`` 顶层配置。

    顶层只选择 pipeline；各 pipeline 的细节参数都放入对应分组。默认值选择
    ``trajectory_optimization``，符合设计文档中“目标式请求优先用 optimizer”的策略。
    """

    planning_pipeline: PlanningPipeline = "trajectory_optimization"
    graph_search: GraphSearchConfig = field(default_factory=GraphSearchConfig)
    trajectory_generation: TrajectoryGenerationConfig = field(
        default_factory=TrajectoryGenerationConfig
    )
    trajectory_optimization: TrajectoryOptimizationConfig = field(
        default_factory=TrajectoryOptimizationConfig
    )
    specified_path: SpecifiedPathConfig = field(default_factory=SpecifiedPathConfig)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
        *,
        base_defaults: Mapping[str, Any] | None = None,
    ) -> "MotionPlannerBackendConfig":
        """从 YAML 风格 mapping 解析 planner 配置。

        ``base_defaults`` 是外部传入的默认值来源，例如 robot YAML 顶层的
        ``motion_planner_params`` 和 ``trajectory_limits``。显式分组配置优先级更高。
        """

        base = dict(base_defaults or {})
        # data=None 表示调用方使用默认 planner 配置；base_defaults 仍会作为各分组的默认值来源。
        if data is None:
            settings: Mapping[str, Any] = {}
        elif not isinstance(data, Mapping):
            raise ValueError("motion planner config must be a mapping")
        else:
            settings = data

        pipeline = str(
            settings.get("planning_pipeline", cls.planning_pipeline)
        ).strip()
        if pipeline not in {
            "graph_search",
            "specified_path",
            "trajectory_optimization",
        }:
            raise ValueError(
                "planning_pipeline must be one of: graph_search, specified_path, "
                "trajectory_optimization"
            )

        # 每个分组只接收自己的参数。不存在的分组视为空 mapping，从而使用 dataclass 默认值。
        graph_settings = _mapping(settings.get("graph_search"), "graph_search")
        trajectory_settings = _mapping(
            settings.get("trajectory_generation"), "trajectory_generation"
        )
        optimizer_settings = _mapping(
            settings.get("trajectory_optimization"), "trajectory_optimization"
        )
        specified_settings = _mapping(settings.get("specified_path"), "specified_path")

        # 显式分组配置优先于 base_defaults，便于调用方只覆盖需要调整的 planner 分组。
        graph_config_path = graph_settings.get(
            "motion_planner_config_path",
            base.get("motion_planner_config_path"),
        )
        graph_params = _merged_mapping(
            base.get("motion_planner_params"),
            graph_settings.get("motion_planner_params"),
        )
        trajectory_limits = _merged_mapping(
            base.get("trajectory_limits"), trajectory_settings.get("limits")
        )
        trajectory_solver_params = _merged_mapping(
            base.get("trajectory_solver_params"),
            trajectory_settings.get("solver_params"),
        )

        graph_search = GraphSearchConfig(
            generate_interpolated_path=bool(
                graph_settings.get(
                    "generate_interpolated_path",
                    GraphSearchConfig.generate_interpolated_path,
                )
            ),
            use_environment_obstacles=bool(
                graph_settings.get(
                    "use_environment_obstacles",
                    GraphSearchConfig.use_environment_obstacles,
                )
            ),
            motion_planner_config_path=_optional_repo_path(graph_config_path),
            motion_planner_params=graph_params,
        )

        trajectory_generation = TrajectoryGenerationConfig(
            enabled=bool(
                trajectory_settings.get("enabled", TrajectoryGenerationConfig.enabled)
            ),
            mode=_trajectory_generation_mode(
                trajectory_settings.get("mode", TrajectoryGenerationConfig.mode)
            ),
            interpolation_mode=_trajectory_interpolation_mode(
                trajectory_settings.get(
                    "interpolation_mode",
                    TrajectoryGenerationConfig.interpolation_mode,
                )
            ),
            limits=trajectory_limits,
            solver_params=trajectory_solver_params,
        )

        if "fallback_pipeline" in optimizer_settings:
            raise ValueError(
                "trajectory_optimization.fallback_pipeline is not supported; "
                "choose planning_pipeline=graph_search or retry explicitly in task code"
            )

        trajectory_optimization = TrajectoryOptimizationConfig(
            config_path=_optional_repo_path(optimizer_settings.get("config_path")),
            use_environment_obstacles=bool(
                optimizer_settings.get(
                    "use_environment_obstacles",
                    TrajectoryOptimizationConfig.use_environment_obstacles,
                )
            ),
            params=_params_mapping(optimizer_settings.get("params")),
        )

        family = str(
            specified_settings.get("family", SpecifiedPathConfig.family)
        ).strip()
        if family not in {"cspace_waypoints", "task_space_segments", "composite"}:
            raise ValueError(
                "specified_path.family must be one of: cspace_waypoints, "
                "task_space_segments, composite"
            )
        specified_path = SpecifiedPathConfig(
            family=family,
            validate_collision_after_generation=bool(
                specified_settings.get(
                    "validate_collision_after_generation",
                    SpecifiedPathConfig.validate_collision_after_generation,
                )
            ),
            cspace_waypoints=_params_mapping(
                specified_settings.get("cspace_waypoints")
            ),
            task_space_segments=_params_mapping(
                specified_settings.get("task_space_segments")
            ),
            composite=_params_mapping(specified_settings.get("composite")),
        )

        config = cls(
            planning_pipeline=pipeline,
            graph_search=graph_search,
            trajectory_generation=trajectory_generation,
            trajectory_optimization=trajectory_optimization,
            specified_path=specified_path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """校验枚举取值和 mapping 形状。

        这里不检查参数名是否被真实 cuMotion 接受；底层 ``set_param`` 返回 False 时会在具体
        pipeline 中抛出更接近后端的错误。
        """

        if self.planning_pipeline not in {
            "graph_search",
            "specified_path",
            "trajectory_optimization",
        }:
            raise ValueError(
                "planning_pipeline must be one of: graph_search, specified_path, "
                "trajectory_optimization"
            )
        _params_mapping(self.graph_search.motion_planner_params)
        _params_mapping(self.trajectory_optimization.params)
        _params_mapping(self.trajectory_generation.solver_params)
        _limit_mapping(self.trajectory_generation.limits)
        if self.trajectory_generation.mode not in {"time_optimal", "time_stamped"}:
            raise ValueError(
                "trajectory_generation.mode must be one of: time_optimal, time_stamped"
            )
        if self.trajectory_generation.interpolation_mode not in {
            "linear",
            "cubic_spline",
        }:
            raise ValueError(
                "trajectory_generation.interpolation_mode must be one of: linear, "
                "cubic_spline"
            )
        if self.specified_path.family not in {
            "cspace_waypoints",
            "task_space_segments",
            "composite",
        }:
            raise ValueError(
                "specified_path.family must be one of: cspace_waypoints, "
                "task_space_segments, composite"
            )
        _params_mapping(self.specified_path.cspace_waypoints)
        _params_mapping(self.specified_path.task_space_segments)
        _params_mapping(self.specified_path.composite)


def _mapping(value, label: str) -> Mapping[str, Any]:
    """解析可选分组 mapping。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _params_mapping(value) -> dict[str, Any]:
    """解析 cuMotion 参数 mapping，保留 value 原始类型交给 ParamValue 包装。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("cuMotion planner params must be a mapping")
    params = {}
    for key, param_value in value.items():
        key = str(key)
        if not key:
            raise ValueError("cuMotion planner param names cannot be empty")
        params[key] = param_value
    return params


def _limit_mapping(value) -> dict[str, np.ndarray]:
    """解析 trajectory generation limit，统一转成一维 float 数组。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("trajectory_generation.limits must be a mapping")
    limits = {}
    for key, limit_value in value.items():
        key = str(key)
        if not key:
            raise ValueError("trajectory generation limit names cannot be empty")
        array = np.asarray(limit_value, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError(f"trajectory generation limit {key!r} cannot be empty")
        limits[key] = array
    return limits


def _merged_mapping(*mappings) -> dict[str, Any]:
    """按顺序合并多个 mapping，后者覆盖前者。"""

    merged: dict[str, Any] = {}
    for mapping in mappings:
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise ValueError("cuMotion planner params must be mappings")
        merged.update({str(key): value for key, value in mapping.items()})
    return merged


def _optional_repo_path(value) -> Path | None:
    """按仓库根目录解析可选路径；空字符串视为未配置。"""

    if value is None or value == "":
        return None
    return repo_path(value)


def _trajectory_generation_mode(value) -> TrajectoryGenerationMode:
    """把用户配置中的轨迹生成模式归一化成内部固定字符串。"""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"time_optimal", "optimal", "default"}:
        return "time_optimal"
    if normalized in {"time_stamped", "timestamped", "time_stamp"}:
        return "time_stamped"
    raise ValueError(
        "trajectory_generation.mode must be one of: time_optimal, time_stamped"
    )


def _trajectory_interpolation_mode(value) -> TrajectoryInterpolationMode:
    """把用户配置中的插值模式归一化成内部固定字符串。"""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "linear":
        return "linear"
    if normalized in {"cubic", "cubic_spline", "spline"}:
        return "cubic_spline"
    raise ValueError(
        "trajectory_generation.interpolation_mode must be one of: linear, cubic_spline"
    )
