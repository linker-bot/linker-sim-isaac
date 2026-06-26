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

_TASK_SPACE_CONVERSION_KEYS = {
    "initial_s_step_size",
    "initial_s_step_size_delta",
    "min_s_step_size",
    "min_s_step_size_delta",
    "alpha",
    "max_iterations",
    "min_position_deviation",
    "max_position_deviation",
}
# 项目侧 composite transition 使用字符串表达，adapter 再映射到
# cumotion.CompositePathSpec.TransitionMode enum。
_COMPOSITE_TRANSITION_MODES = {"skip", "free", "linear_task_space"}


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
    放在 config 中。当前实现支持三类路径族：

    * ``cspace_waypoints``：调用方给出完整 C-space waypoint，后端用官方 CSpacePathSpec 生成
      LinearCSpacePath。
    * ``task_space_segments``：调用方给出 TCP 几何段，后端用 TaskSpacePathSpec 和官方 path
      conversion 转成 C-space。
    * ``composite``：调用方混合 C-space/task-space 子段，后端用 CompositePathSpec 转成统一
      C-space path。

    ``cspace_waypoints``、``task_space_segments`` 和 ``composite`` 保持 mapping 形态，便于 YAML
    直接表达官方 conversion 参数和少量项目侧策略键。
    """

    family: SpecifiedPathFamily = "task_space_segments"
    validate_collision_after_generation: bool = False
    cspace_waypoints: Mapping[str, Any] = field(default_factory=dict)
    task_space_segments: Mapping[str, Any] = field(default_factory=dict)
    composite: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionPlannerBackendConfig:
    """``CuMotionMotionPlanner`` 顶层配置。

    顶层只选择 pipeline；各 pipeline 的细节参数都放入对应分组。
    默认值选择``trajectory_optimization``。
    """

    # 选择本次 ``CuMotionMotionPlanner`` 使用哪条规划 pipeline。
    planning_pipeline: PlanningPipeline = "trajectory_optimization"
    # graph search pipeline 的专属参数：MotionPlanner 配置、是否返回 interpolated path、是否避障。
    graph_search: GraphSearchConfig = field(default_factory=GraphSearchConfig)
    # 对 graph/specifed path 产生的 C-space path 做时间参数化，生成可执行 trajectory。
    trajectory_generation: TrajectoryGenerationConfig = field(
        default_factory=TrajectoryGenerationConfig
    )
    # trajectory optimization pipeline 的专属参数：optimizer 配置、环境碰撞开关、后端参数覆盖。
    trajectory_optimization: TrajectoryOptimizationConfig = field(
        default_factory=TrajectoryOptimizationConfig
    )
    # specified path pipeline 的专属参数：默认路径族、路径转换策略和可选碰撞后验检查。
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
        _validate_specified_path_settings(self.specified_path)


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


def _validate_specified_path_settings(config: SpecifiedPathConfig) -> None:
    """校验 specified-path mapping 中项目侧已知的策略键。

    mapping 里的字段会在 adapter 中写入 cuMotion 对象。这里提前做白名单和数值范围检查，能把
    配置拼写错误停在 Python 层，而不是等 pybind conversion 失败后再倒查。
    """

    cspace_settings = _params_mapping(config.cspace_waypoints)
    # require_start_match 是项目侧保护策略，不是 cuMotion 官方字段；它控制 adapter 是否要求
    # C-space 路径首点与 request.current_q 一致。
    if "require_start_match" in cspace_settings and not isinstance(
        cspace_settings["require_start_match"], bool
    ):
        raise ValueError("specified_path.cspace_waypoints.require_start_match must be bool")
    if "start_match_tolerance" in cspace_settings:
        tolerance = float(cspace_settings["start_match_tolerance"])
        if tolerance < 0:
            raise ValueError(
                "specified_path.cspace_waypoints.start_match_tolerance cannot be negative"
            )

    task_settings = _params_mapping(config.task_space_segments)
    # conversion 子 mapping 对应 cuMotion TaskSpacePathConversionConfig 的公开字段。保持白名单
    # 可以避免用户把 IK 或其它 planner 参数误放进 conversion config。
    conversion = _mapping(
        task_settings.get("conversion"), "specified_path.task_space_segments.conversion"
    )
    unknown_conversion_keys = set(map(str, conversion)) - _TASK_SPACE_CONVERSION_KEYS
    if unknown_conversion_keys:
        raise ValueError(
            "Unsupported specified_path.task_space_segments.conversion key(s): "
            f"{sorted(unknown_conversion_keys)}"
        )
    _validate_conversion_numeric_ranges(conversion)
    # ik 子 mapping 只放 path conversion 的 IK seed 策略。实际 IK 容差/迭代次数复用
    # CuMotionConfig 上已有字段，由 path_spec_adapter._ik_config_for_path_conversion 复制。
    ik_settings = _mapping(
        task_settings.get("ik"), "specified_path.task_space_segments.ik"
    )
    if "use_current_q_as_seed" in ik_settings and not isinstance(
        ik_settings["use_current_q_as_seed"], bool
    ):
        raise ValueError(
            "specified_path.task_space_segments.ik.use_current_q_as_seed must be bool"
        )

    composite_settings = _params_mapping(config.composite)
    # 未单独指定 CompositePathPart.transition_mode 的子段会使用这个默认 transition。
    if "default_transition_mode" in composite_settings:
        transition_mode = str(composite_settings["default_transition_mode"])
        if transition_mode not in _COMPOSITE_TRANSITION_MODES:
            raise ValueError(
                "specified_path.composite.default_transition_mode must be one of: "
                "skip, free, linear_task_space"
            )


def _validate_conversion_numeric_ranges(conversion: Mapping[str, Any]) -> None:
    """校验 ``TaskSpacePathConversionConfig`` 的项目侧数值约束。

    这些范围来自 conversion 参数的基本数值语义：步长必须为正、迭代次数必须为正、误差上下界
    必须形成非空区间。真实 robot/路径是否能转换成功仍由 cuMotion conversion 判断。
    """

    for key in {
        "initial_s_step_size",
        "initial_s_step_size_delta",
        "min_s_step_size",
        "min_s_step_size_delta",
    }:
        if key in conversion and float(conversion[key]) <= 0:
            raise ValueError(
                f"specified_path.task_space_segments.conversion.{key} must be positive"
            )
    if "alpha" in conversion and float(conversion["alpha"]) <= 1:
        raise ValueError(
            "specified_path.task_space_segments.conversion.alpha must be greater than 1"
        )
    if "max_iterations" in conversion and int(conversion["max_iterations"]) <= 0:
        raise ValueError(
            "specified_path.task_space_segments.conversion.max_iterations must be positive"
        )
    if (
        "min_position_deviation" in conversion
        or "max_position_deviation" in conversion
    ):
        min_deviation = float(conversion.get("min_position_deviation", 0.001))
        max_deviation = float(conversion.get("max_position_deviation", 0.003))
        if min_deviation <= 0 or max_deviation <= min_deviation:
            raise ValueError(
                "specified_path.task_space_segments.conversion requires "
                "0 < min_position_deviation < max_position_deviation"
            )


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
