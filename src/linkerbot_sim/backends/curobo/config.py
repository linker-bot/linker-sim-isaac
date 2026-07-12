"""cuRobo 后端配置模型及项目拥有的严格 YAML 边界。

本文件只解析项目 YAML/Mapping，不导入 cuRobo。真实 robot config 是否可被 cuRobo 加载、
tool frame 是否存在、joint order 是否匹配等模型相关问题，交给 ``CuroboContext`` 或具体
solver 在运行时检查。所有 mapping 都拒绝未知键、隐式类型转换和任意第三方 task 路径；
只有项目验证过的 task bundle 能进入 backend，防止配置绕开已测试的求解器组合。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from linkerbot_sim.utils.paths import repo_path


SUPPORTED_COLLISION_CACHE_TYPES = frozenset({"cuboid", "mesh"})
SUPPORTED_CUROBO_DTYPES = frozenset({"float32"})
DEFAULT_CUROBO_TASK_BUNDLE = "curobo_v0_8_default"
_RAW_IK_TASK_KEYS = frozenset(
    {"optimizer_configs", "metrics_rollout", "transition_model"}
)
_RAW_MOTION_TASK_KEYS = frozenset(
    {
        "ik_optimizer_configs",
        "ik_transition_model",
        "metrics_rollout",
        "trajopt_optimizer_configs",
        "trajopt_transition_model",
        "graph_planner_config",
        "graph_planner_rollout",
        "graph_planner_transition_model",
    }
)
_GENERIC_RAW_TASK_PATH_KEYS = frozenset({"task_path", "task_file", "task_config_path"})
_DEVICE_KEYS = frozenset(
    {
        "device",
        "tensor_dtype",
        "collision_geometry_dtype",
        "collision_gradient_dtype",
        "collision_distance_dtype",
    }
)
_ROBOT_KEYS = frozenset(
    {
        "robot_config_path",
        "urdf_path",
        "base_link",
        "flange_frame",
        "tool_frames",
        "default_tcp_frame",
        "custom_tcps",
        "load_collision_spheres",
    }
)
_TCP_KEYS = frozenset({"frame_name", "parent_frame", "xyz", "rpy"})
_IK_KEYS = frozenset(
    {
        "num_seeds",
        "position_tolerance",
        "orientation_tolerance",
        "use_cuda_graph",
        "random_seed",
        "optimizer_collision_activation_distance",
        "store_debug",
        "override_optimizer_num_iters",
        "override_iters_for_multi_link_ik",
        "optimization_dt",
        "velocity_regularization_weight",
        "acceleration_regularization_weight",
        "success_requires_convergence",
        "seed_position_weight",
        "seed_orientation_weight",
        "seed_velocity_weight",
        "seed_acceleration_weight",
        "seed_solver_num_seeds",
        "max_batch_size",
        "multi_env",
        "max_goalset",
        "self_collision_check",
        "collision_cache",
    }
)
_MOTION_PLANNER_KEYS = frozenset(
    {
        "warmup",
        "num_ik_seeds",
        "num_trajopt_seeds",
        "position_tolerance",
        "orientation_tolerance",
        "use_cuda_graph",
        "random_seed",
        "optimizer_collision_activation_distance",
        "store_debug",
        "max_batch_size",
        "multi_env",
        "max_goalset",
        "self_collision_check",
        "collision_cache",
    }
)
_CUROBO_KEYS = frozenset(
    {
        "enabled",
        "planning_joint_group",
        "robot",
        "task_bundle",
        "device",
        "kinematics",
        "motion_planner",
    }
)


@dataclass(frozen=True)
class CuroboTaskBundle:
    """经过项目验证的 cuRobo task 文件组合。

    用户配置只选择 bundle 名；具体 task 路径和第三方版本约束留在代码中，避免任意
    cuRobo 内部 YAML 路径绕过项目兼容性验证。
    """

    name: str
    compatible_versions: frozenset[str]
    ik_optimizer_configs: tuple[str, ...]
    ik_metrics_rollout: str
    ik_transition_model: str
    motion_ik_optimizer_configs: tuple[str, ...]
    motion_ik_transition_model: str
    motion_metrics_rollout: str
    trajopt_optimizer_configs: tuple[str, ...]
    trajopt_transition_model: str
    graph_planner_config: str
    graph_planner_rollout: str
    graph_planner_transition_model: str

    @classmethod
    def named(cls, value: object = DEFAULT_CUROBO_TASK_BUNDLE) -> "CuroboTaskBundle":
        """解析唯一已验证的版本化 task bundle。"""

        name = _non_empty_str(value, "curobo.task_bundle")
        if name != DEFAULT_CUROBO_TASK_BUNDLE:
            raise ValueError(
                "curobo.task_bundle must be 'curobo_v0_8_default'; "
                f"unsupported bundle: {name!r}"
            )
        return cls(
            name=name,
            compatible_versions=frozenset({"0.8.0"}),
            ik_optimizer_configs=("ik/particle_ik.yml", "ik/lbfgs_ik.yml"),
            ik_metrics_rollout="metrics_base.yml",
            ik_transition_model="ik/transition_ik.yml",
            motion_ik_optimizer_configs=("ik/lbfgs_ik.yml",),
            motion_ik_transition_model="ik/transition_ik.yml",
            motion_metrics_rollout="metrics_base.yml",
            trajopt_optimizer_configs=("trajopt/lbfgs_bspline_trajopt.yml",),
            trajopt_transition_model="trajopt/transition_bspline_trajopt.yml",
            graph_planner_config="graph_planner/exact_graph_planner.yml",
            graph_planner_rollout="metrics_base.yml",
            graph_planner_transition_model=(
                "graph_planner/transition_graph_planner.yml"
            ),
        )

    def validate_curobo_version(self, version: object) -> None:
        """拒绝用未经验证的 cuRobo 版本消费 bundle。"""

        actual = str(version).strip()
        if actual not in self.compatible_versions:
            raise RuntimeError(
                f"cuRobo task bundle {self.name!r} requires cuRobo "
                f"version in {sorted(self.compatible_versions)!r}, "
                f"installed version is {actual!r}"
            )


@dataclass(frozen=True)
class CuroboTcpFrame:
    """需要写入 cuRobo/URDF 规划模型的固定 TCP frame。

    ``xyz`` 使用米，``rpy`` 使用弧度，均相对于 ``parent_frame``。frame 名称在同一 robot
    配置内必须唯一，实际父 frame 是否存在由 context 构建时结合模型校验。
    """

    frame_name: str
    parent_frame: str
    xyz: np.ndarray
    rpy: np.ndarray

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        default_parent_frame: str | None,
        label: str,
    ) -> "CuroboTcpFrame":
        """从 robot YAML 的 custom TCP 条目解析固定 frame。"""

        if not isinstance(data, Mapping):
            raise ValueError(f"{label} must be a mapping")
        _reject_unknown_keys(data, _TCP_KEYS, label)
        frame_name = _non_empty_str(
            _required_value(data, "frame_name", label=label),
            f"{label}.frame_name",
        )
        parent_value = data.get("parent_frame", default_parent_frame)
        if parent_value is None:
            raise ValueError(f"{label}.parent_frame is required")
        parent_frame = _non_empty_str(parent_value, f"{label}.parent_frame")
        return cls(
            frame_name=frame_name,
            parent_frame=parent_frame,
            xyz=_vector3(data.get("xyz", (0.0, 0.0, 0.0)), f"{label}.xyz"),
            rpy=_vector3(data.get("rpy", (0.0, 0.0, 0.0)), f"{label}.rpy"),
        )


@dataclass(frozen=True)
class CuroboDeviceConfig:
    """cuRobo tensor device 与各计算路径 dtype 配置。

    当前项目只验证了明确列入 ``SUPPORTED_CUROBO_DTYPES`` 的 dtype；解析阶段不会让 numpy
    或 torch 猜测类型，避免 collision distance/gradient 在不同精度间隐式转换。
    """

    device: str = "cuda:0"
    tensor_dtype: str = "float32"
    collision_geometry_dtype: str = "float32"
    collision_gradient_dtype: str = "float32"
    collision_distance_dtype: str = "float32"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CuroboDeviceConfig":
        """解析 ``curobo.device`` 分组。"""

        settings = _mapping_or_empty(data, "curobo.device")
        _reject_unknown_keys(settings, _DEVICE_KEYS, "curobo.device")
        return cls(
            device=_non_empty_str(
                settings.get("device", cls.device), "curobo.device.device"
            ),
            tensor_dtype=_curobo_dtype(
                settings.get("tensor_dtype", cls.tensor_dtype),
                "curobo.device.tensor_dtype",
            ),
            collision_geometry_dtype=_curobo_dtype(
                settings.get("collision_geometry_dtype", cls.collision_geometry_dtype),
                "curobo.device.collision_geometry_dtype",
            ),
            collision_gradient_dtype=_curobo_dtype(
                settings.get("collision_gradient_dtype", cls.collision_gradient_dtype),
                "curobo.device.collision_gradient_dtype",
            ),
            collision_distance_dtype=_curobo_dtype(
                settings.get("collision_distance_dtype", cls.collision_distance_dtype),
                "curobo.device.collision_distance_dtype",
            ),
        )


@dataclass(frozen=True)
class CuroboRobotConfig:
    """cuRobo robot model 的项目资源和 TCP 绑定配置。

    ``robot_config_path`` 可提供完整 cuRobo robot YAML，``urdf_path`` 可用于项目生成或补充
    模型；至少需要其中之一。若只给 URDF，可从唯一根 link 推断 ``base_link``。tool frame
    的模型存在性和 collision spheres 内容推迟到 context materialize 阶段校验。
    """

    robot_config_path: Path | None = None
    urdf_path: Path | None = None
    base_link: str | None = None
    flange_frame: str | None = None
    tool_frames: tuple[str, ...] = ()
    default_tcp_frame: str | None = None
    custom_tcp_frames: tuple[CuroboTcpFrame, ...] = ()
    load_collision_spheres: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CuroboRobotConfig":
        """解析 canonical ``curobo.robot`` mapping。"""

        settings = _mapping_or_empty(data, "curobo.robot")
        _reject_unknown_keys(settings, _ROBOT_KEYS, "curobo.robot")
        robot_config_path = _optional_repo_path(
            settings.get("robot_config_path"), "curobo.robot.robot_config_path"
        )
        urdf_path = _optional_repo_path(
            settings.get("urdf_path"), "curobo.robot.urdf_path"
        )
        base_link = _optional_str(settings.get("base_link"), "curobo.robot.base_link")
        if base_link is None and urdf_path is not None:
            base_link = _infer_urdf_root_link(urdf_path)
        tool_frames = _string_sequence(
            settings.get("tool_frames", ()), "curobo.robot.tool_frames"
        )
        default_tcp_frame = _optional_str(
            settings.get("default_tcp_frame"), "curobo.robot.default_tcp_frame"
        )
        flange_frame = _optional_str(
            settings.get("flange_frame"), "curobo.robot.flange_frame"
        )
        custom_tcp_frames = _parse_custom_tcp_frames(
            settings.get("custom_tcps"),
            default_parent_frame=flange_frame or default_tcp_frame,
            label="curobo.robot.custom_tcps",
        )
        config = cls(
            robot_config_path=robot_config_path,
            urdf_path=urdf_path,
            base_link=base_link,
            flange_frame=flange_frame,
            tool_frames=tool_frames,
            default_tcp_frame=default_tcp_frame,
            custom_tcp_frames=custom_tcp_frames,
            load_collision_spheres=_strict_bool(
                settings.get("load_collision_spheres", cls.load_collision_spheres),
                "curobo.robot.load_collision_spheres",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """校验 robot config 的静态结构。"""

        if self.robot_config_path is None and self.urdf_path is None:
            raise ValueError(
                "curobo robot config requires robot_config_path or urdf_path"
            )
        if self.urdf_path is not None and not str(self.urdf_path):
            raise ValueError("curobo.robot.urdf_path cannot be empty")
        if self.robot_config_path is not None and not str(self.robot_config_path):
            raise ValueError("curobo.robot.robot_config_path cannot be empty")
        if self.urdf_path is not None and self.base_link is None:
            raise ValueError("curobo.robot.base_link is required with urdf_path")
        if not self.tool_frames and self.default_tcp_frame is None:
            raise ValueError(
                "curobo robot config requires tool_frames or default_tcp_frame"
            )

    @property
    def resolved_tool_frames(self) -> tuple[str, ...]:
        """返回 cuRobo context 应注册/查询的 tool frames。"""

        if self.tool_frames:
            return self.tool_frames
        assert self.default_tcp_frame is not None
        return (self.default_tcp_frame,)


@dataclass(frozen=True)
class CuroboIkConfig:
    """cuRobo IK 的种子、收敛、正则化与批量容量参数。

    所有数值在解析时严格检查类型、有限性和取值范围。``collision_cache`` 声明预分配容量，
    不是动态扩容提示；实际场景需求超过容量时由 planning capability 明确报告不可用。
    """

    num_seeds: int = 32
    position_tolerance: float = 0.002
    orientation_tolerance: float = 0.01
    use_cuda_graph: bool = True
    random_seed: int = 123
    optimizer_collision_activation_distance: float = 0.01
    store_debug: bool = False
    override_optimizer_num_iters: dict[str, int | None] = field(
        default_factory=lambda: {"particle": None, "lbfgs": None}
    )
    override_iters_for_multi_link_ik: int | None = None
    optimization_dt: float | None = None
    velocity_regularization_weight: float | None = None
    acceleration_regularization_weight: float | None = None
    success_requires_convergence: bool = True
    seed_position_weight: float = 1.0
    seed_orientation_weight: float = 1.0
    seed_velocity_weight: float = 0.0
    seed_acceleration_weight: float = 0.0
    seed_solver_num_seeds: int = 32
    max_batch_size: int = 256
    multi_env: bool = False
    max_goalset: int = 1
    self_collision_check: bool = True
    collision_cache: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CuroboIkConfig":
        """解析 ``curobo.kinematics.ik`` 分组。"""

        settings = _mapping_or_empty(data, "curobo.kinematics.ik")
        _reject_raw_task_paths(
            settings,
            _RAW_IK_TASK_KEYS | _GENERIC_RAW_TASK_PATH_KEYS,
            label="curobo.kinematics.ik",
        )
        _reject_unknown_keys(settings, _IK_KEYS, "curobo.kinematics.ik")
        config = cls(
            num_seeds=_strict_int(
                settings.get("num_seeds", cls.num_seeds),
                "curobo.kinematics.ik.num_seeds",
            ),
            position_tolerance=_strict_float(
                settings.get("position_tolerance", cls.position_tolerance),
                "curobo.kinematics.ik.position_tolerance",
            ),
            orientation_tolerance=_strict_float(
                settings.get("orientation_tolerance", cls.orientation_tolerance),
                "curobo.kinematics.ik.orientation_tolerance",
            ),
            use_cuda_graph=_strict_bool(
                settings.get("use_cuda_graph", cls.use_cuda_graph),
                "curobo.kinematics.ik.use_cuda_graph",
            ),
            random_seed=_strict_int(
                settings.get("random_seed", cls.random_seed),
                "curobo.kinematics.ik.random_seed",
            ),
            optimizer_collision_activation_distance=_strict_float(
                settings.get(
                    "optimizer_collision_activation_distance",
                    cls.optimizer_collision_activation_distance,
                ),
                "curobo.kinematics.ik.optimizer_collision_activation_distance",
            ),
            store_debug=_strict_bool(
                settings.get("store_debug", cls.store_debug),
                "curobo.kinematics.ik.store_debug",
            ),
            override_optimizer_num_iters=_parse_optional_int_mapping(
                settings.get("override_optimizer_num_iters"),
                default=cls().override_optimizer_num_iters,
                label="curobo.kinematics.ik.override_optimizer_num_iters",
            ),
            override_iters_for_multi_link_ik=_optional_int(
                settings.get("override_iters_for_multi_link_ik"),
                "curobo.kinematics.ik.override_iters_for_multi_link_ik",
            ),
            optimization_dt=_optional_float(
                settings.get("optimization_dt"),
                "curobo.kinematics.ik.optimization_dt",
            ),
            velocity_regularization_weight=_optional_float(
                settings.get("velocity_regularization_weight"),
                "curobo.kinematics.ik.velocity_regularization_weight",
            ),
            acceleration_regularization_weight=_optional_float(
                settings.get("acceleration_regularization_weight"),
                "curobo.kinematics.ik.acceleration_regularization_weight",
            ),
            success_requires_convergence=_strict_bool(
                settings.get(
                    "success_requires_convergence",
                    cls.success_requires_convergence,
                ),
                "curobo.kinematics.ik.success_requires_convergence",
            ),
            seed_position_weight=_strict_float(
                settings.get("seed_position_weight", cls.seed_position_weight),
                "curobo.kinematics.ik.seed_position_weight",
            ),
            seed_orientation_weight=_strict_float(
                settings.get("seed_orientation_weight", cls.seed_orientation_weight),
                "curobo.kinematics.ik.seed_orientation_weight",
            ),
            seed_velocity_weight=_strict_float(
                settings.get("seed_velocity_weight", cls.seed_velocity_weight),
                "curobo.kinematics.ik.seed_velocity_weight",
            ),
            seed_acceleration_weight=_strict_float(
                settings.get(
                    "seed_acceleration_weight",
                    cls.seed_acceleration_weight,
                ),
                "curobo.kinematics.ik.seed_acceleration_weight",
            ),
            seed_solver_num_seeds=_strict_int(
                settings.get("seed_solver_num_seeds", cls.seed_solver_num_seeds),
                "curobo.kinematics.ik.seed_solver_num_seeds",
            ),
            max_batch_size=_strict_int(
                settings.get("max_batch_size", cls.max_batch_size),
                "curobo.kinematics.ik.max_batch_size",
            ),
            multi_env=_strict_bool(
                settings.get("multi_env", cls.multi_env),
                "curobo.kinematics.ik.multi_env",
            ),
            max_goalset=_strict_int(
                settings.get("max_goalset", cls.max_goalset),
                "curobo.kinematics.ik.max_goalset",
            ),
            self_collision_check=_strict_bool(
                settings.get("self_collision_check", cls.self_collision_check),
                "curobo.kinematics.ik.self_collision_check",
            ),
            collision_cache=_parse_int_mapping(
                settings.get("collision_cache"),
                label="curobo.kinematics.ik.collision_cache",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """校验 IK 数值参数。"""

        if self.num_seeds <= 0:
            raise ValueError("curobo.kinematics.ik.num_seeds must be positive")
        if self.seed_solver_num_seeds <= 0:
            raise ValueError(
                "curobo.kinematics.ik.seed_solver_num_seeds must be positive"
            )
        if self.max_batch_size <= 0 or self.max_goalset <= 0:
            raise ValueError("curobo IK batch sizes must be positive")
        if self.random_seed < 0:
            raise ValueError("curobo.kinematics.ik.random_seed cannot be negative")
        if (
            self.position_tolerance < 0
            or self.orientation_tolerance < 0
            or self.optimizer_collision_activation_distance < 0
        ):
            raise ValueError("curobo IK tolerances cannot be negative")
        _validate_non_negative_optional(
            self.override_iters_for_multi_link_ik,
            "curobo.kinematics.ik.override_iters_for_multi_link_ik",
        )
        _validate_positive_optional(
            self.optimization_dt,
            "curobo.kinematics.ik.optimization_dt",
        )
        _validate_non_negative_optional(
            self.velocity_regularization_weight,
            "curobo.kinematics.ik.velocity_regularization_weight",
        )
        _validate_non_negative_optional(
            self.acceleration_regularization_weight,
            "curobo.kinematics.ik.acceleration_regularization_weight",
        )
        if any(
            value < 0
            for value in (
                self.seed_position_weight,
                self.seed_orientation_weight,
                self.seed_velocity_weight,
                self.seed_acceleration_weight,
            )
        ):
            raise ValueError("curobo IK seed weights cannot be negative")


@dataclass(frozen=True)
class CuroboMotionPlannerConfig:
    """cuRobo MotionPlanner / BatchMotionPlanner 的求解和容量参数。

    单请求与批量路径共用该配置，``max_batch_size`` 与 ``max_goalset`` 因而是显式资源上限。
    解析只接受项目公开字段，第三方 task 文件选择由 ``CuroboTaskBundle`` 固定管理。
    """

    warmup: bool = True
    num_ik_seeds: int = 32
    num_trajopt_seeds: int = 4
    position_tolerance: float = 0.002
    orientation_tolerance: float = 0.01
    use_cuda_graph: bool = True
    random_seed: int = 123
    optimizer_collision_activation_distance: float = 0.01
    store_debug: bool = False
    max_batch_size: int = 256
    multi_env: bool = False
    max_goalset: int = 1
    self_collision_check: bool = True
    collision_cache: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "CuroboMotionPlannerConfig":
        """解析 ``curobo.motion_planner`` 分组。"""

        settings = _mapping_or_empty(data, "curobo.motion_planner")
        _reject_raw_task_paths(
            settings,
            _RAW_MOTION_TASK_KEYS | _GENERIC_RAW_TASK_PATH_KEYS,
            label="curobo.motion_planner",
        )
        _reject_unknown_keys(settings, _MOTION_PLANNER_KEYS, "curobo.motion_planner")
        config = cls(
            warmup=_strict_bool(
                settings.get("warmup", cls.warmup),
                "curobo.motion_planner.warmup",
            ),
            num_ik_seeds=_strict_int(
                settings.get("num_ik_seeds", cls.num_ik_seeds),
                "curobo.motion_planner.num_ik_seeds",
            ),
            num_trajopt_seeds=_strict_int(
                settings.get("num_trajopt_seeds", cls.num_trajopt_seeds),
                "curobo.motion_planner.num_trajopt_seeds",
            ),
            position_tolerance=_strict_float(
                settings.get("position_tolerance", cls.position_tolerance),
                "curobo.motion_planner.position_tolerance",
            ),
            orientation_tolerance=_strict_float(
                settings.get("orientation_tolerance", cls.orientation_tolerance),
                "curobo.motion_planner.orientation_tolerance",
            ),
            use_cuda_graph=_strict_bool(
                settings.get("use_cuda_graph", cls.use_cuda_graph),
                "curobo.motion_planner.use_cuda_graph",
            ),
            random_seed=_strict_int(
                settings.get("random_seed", cls.random_seed),
                "curobo.motion_planner.random_seed",
            ),
            optimizer_collision_activation_distance=_strict_float(
                settings.get(
                    "optimizer_collision_activation_distance",
                    cls.optimizer_collision_activation_distance,
                ),
                "curobo.motion_planner.optimizer_collision_activation_distance",
            ),
            store_debug=_strict_bool(
                settings.get("store_debug", cls.store_debug),
                "curobo.motion_planner.store_debug",
            ),
            max_batch_size=_strict_int(
                settings.get("max_batch_size", cls.max_batch_size),
                "curobo.motion_planner.max_batch_size",
            ),
            multi_env=_strict_bool(
                settings.get("multi_env", cls.multi_env),
                "curobo.motion_planner.multi_env",
            ),
            max_goalset=_strict_int(
                settings.get("max_goalset", cls.max_goalset),
                "curobo.motion_planner.max_goalset",
            ),
            self_collision_check=_strict_bool(
                settings.get("self_collision_check", cls.self_collision_check),
                "curobo.motion_planner.self_collision_check",
            ),
            collision_cache=_parse_int_mapping(
                settings.get("collision_cache"),
                label="curobo.motion_planner.collision_cache",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """校验 MotionPlanner 数值参数。"""

        if self.num_ik_seeds <= 0 or self.num_trajopt_seeds <= 0:
            raise ValueError("cuRobo planner seed counts must be positive")
        if self.max_batch_size <= 0 or self.max_goalset <= 0:
            raise ValueError("cuRobo planner batch sizes must be positive")
        if self.random_seed < 0:
            raise ValueError("curobo.motion_planner.random_seed cannot be negative")
        if (
            self.position_tolerance < 0
            or self.orientation_tolerance < 0
            or self.optimizer_collision_activation_distance < 0
        ):
            raise ValueError("cuRobo planner tolerances cannot be negative")


@dataclass(frozen=True)
class CuroboConfig:
    """项目侧完整且可直接用于 materialize context 的 cuRobo 后端配置。

    只有 ``curobo.enabled: true`` 且 planning group 为当前支持的 ``arm`` 时才能构造。各子段
    缺省值在这里一次性确定，runtime 不再读取原始 YAML 或执行隐藏覆盖。
    """

    robot: CuroboRobotConfig
    task_bundle: CuroboTaskBundle = field(
        default_factory=lambda: CuroboTaskBundle.named()
    )
    device: CuroboDeviceConfig = field(default_factory=CuroboDeviceConfig)
    ik: CuroboIkConfig = field(default_factory=CuroboIkConfig)
    motion_planner: CuroboMotionPlannerConfig = field(
        default_factory=CuroboMotionPlannerConfig
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CuroboConfig":
        """从包含顶层 ``curobo`` 段的 canonical 配置解析后端。"""

        settings = _mapping_or_empty(
            _required_value(data, "curobo", label="config"),
            "curobo",
        )
        _reject_raw_task_paths(
            settings,
            _RAW_IK_TASK_KEYS | _RAW_MOTION_TASK_KEYS | _GENERIC_RAW_TASK_PATH_KEYS,
            label="curobo",
        )
        enabled = _strict_bool(
            _required_value(settings, "enabled", label="curobo"),
            "curobo.enabled",
        )
        if not enabled:
            raise ValueError(
                "CuroboConfig cannot be materialized when curobo.enabled is false"
            )
        planning_joint_group = _non_empty_str(
            _required_value(settings, "planning_joint_group", label="curobo"),
            "curobo.planning_joint_group",
        ).lower()
        robot_settings = _mapping_or_empty(
            _required_value(settings, "robot", label="curobo"),
            "curobo.robot",
        )
        _reject_unknown_keys(settings, _CUROBO_KEYS, "curobo")
        if planning_joint_group != "arm":
            raise ValueError("curobo.planning_joint_group must be 'arm'")
        device_settings = _optional_section_mapping(settings, "device", "curobo")
        kinematics_settings = _optional_section_mapping(
            settings, "kinematics", "curobo"
        )
        _reject_unknown_keys(kinematics_settings, {"ik"}, "curobo.kinematics")
        ik_settings = _optional_section_mapping(
            kinematics_settings,
            "ik",
            "curobo.kinematics",
        )
        motion_planner_settings = _optional_section_mapping(
            settings,
            "motion_planner",
            "curobo",
        )
        config = cls(
            robot=CuroboRobotConfig.from_mapping(robot_settings),
            task_bundle=CuroboTaskBundle.named(
                settings.get("task_bundle", DEFAULT_CUROBO_TASK_BUNDLE)
            ),
            device=CuroboDeviceConfig.from_mapping(device_settings),
            ik=CuroboIkConfig.from_mapping(ik_settings),
            motion_planner=CuroboMotionPlannerConfig.from_mapping(
                motion_planner_settings
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """级联校验所有子配置。"""

        self.robot.validate()
        self.ik.validate()
        self.motion_planner.validate()


def _reject_raw_task_paths(
    settings: Mapping[str, Any],
    keys: frozenset[str],
    *,
    label: str,
) -> None:
    """拒绝任意 cuRobo 内部 task 文件路径。"""

    raw = sorted(set(settings) & keys)
    if raw:
        fields = ", ".join(f"{label}.{name}" for name in raw)
        raise ValueError(
            f"raw cuRobo task paths are not configurable ({fields}); "
            "select the versioned curobo.task_bundle instead"
        )


def _mapping_or_empty(data: Mapping[str, Any] | None, label: str) -> Mapping[str, Any]:
    """读取可选 mapping。"""

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return data


def _optional_section_mapping(
    data: Mapping[str, Any], key: str, parent_label: str
) -> Mapping[str, Any]:
    """缺少配置段时返回空 mapping，但拒绝显式 ``null`` 或其他类型。

    缺省表示使用 dataclass 当前默认值；显式 ``null`` 往往是拼写或生成配置错误，不能与
    缺省混为一谈。
    """

    if key not in data:
        return {}
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _optional_str(value: object | None, label: str) -> str | None:
    """严格读取可选非空字符串。"""

    if value is None:
        return None
    return _non_empty_str(value, label)


def _optional_float(value: object | None, label: str) -> float | None:
    """严格读取可选有限浮点数。"""

    return None if value is None else _strict_float(value, label)


def _optional_int(value: object | None, label: str) -> int | None:
    """严格读取可选整数。"""

    return None if value is None else _strict_int(value, label)


def _strict_bool(value: object, label: str) -> bool:
    """严格解析 bool，不接受 YAML/JSON 中的 truthy string。"""

    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _strict_int(value: object, label: str) -> int:
    """严格解析整数，不接受 bool、字符串或非整型浮点数。"""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _strict_float(value: object, label: str) -> float:
    """严格解析有限实数，不接受 bool 或数字字符串。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _non_empty_str(value: object, label: str) -> str:
    """严格解析非空字符串。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _curobo_dtype(value: object, label: str) -> str:
    """限制在项目已用 cuRobo 0.8.0 验证过的 tensor dtype。"""

    dtype = _non_empty_str(value, label)
    if dtype not in SUPPORTED_CUROBO_DTYPES:
        raise ValueError(
            f"{label} must be one of {sorted(SUPPORTED_CUROBO_DTYPES)}, got {dtype!r}"
        )
    return dtype


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    """严格解析非空字符串序列。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return tuple(
        _non_empty_str(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _vector3(value: object, label: str) -> np.ndarray:
    """严格解析长度为三的有限数值向量。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of 3 numbers")
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 numbers")
    return np.asarray(
        [_strict_float(item, f"{label}[{index}]") for index, item in enumerate(value)],
        dtype=float,
    )


def _required_value(data: Mapping[str, Any], key: str, *, label: str) -> Any:
    """读取必填字段。"""

    value = data.get(key)
    if value is None:
        raise ValueError(f"{label}.{key} is required")
    return value


def _optional_repo_path(value: object | None, label: str) -> Path | None:
    """把可选路径解析到仓库根目录。"""

    if value is None:
        return None
    return repo_path(_non_empty_str(value, label))


def _parse_custom_tcp_frames(
    value: object,
    *,
    default_parent_frame: str | None,
    label: str,
) -> tuple[CuroboTcpFrame, ...]:
    """解析 custom TCP frame 列表。"""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        parsed: list[CuroboTcpFrame] = []
        for frame_name, item in value.items():
            name = _non_empty_str(frame_name, f"{label} key")
            item_label = f"{label}.{name}"
            item_mapping = _mapping_or_empty(item, item_label)
            if "frame_name" in item_mapping:
                raise ValueError(
                    f"{item_label}.frame_name is not supported in a named mapping; "
                    f"use the key {name!r} as the frame name"
                )
            parsed.append(
                CuroboTcpFrame.from_mapping(
                    {"frame_name": name, **dict(item_mapping)},
                    default_parent_frame=default_parent_frame,
                    label=item_label,
                )
            )
        return tuple(parsed)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a mapping or a sequence")
    frames = tuple(
        CuroboTcpFrame.from_mapping(
            item,
            default_parent_frame=default_parent_frame,
            label=f"{label}[{index}]",
        )
        for index, item in enumerate(value)
    )
    names = [item.frame_name for item in frames]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate frame_name values")
    return frames


def _parse_int_mapping(value: object, *, label: str) -> dict[str, int]:
    """解析 obstacle cache 这类字符串到整数的映射。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    parsed = {
        _non_empty_str(key, f"{label} key"): _strict_int(item, f"{label}.{key}")
        for key, item in value.items()
    }
    if any(item < 0 for item in parsed.values()):
        raise ValueError(f"{label} values cannot be negative")
    unsupported = set(parsed) - SUPPORTED_COLLISION_CACHE_TYPES
    if unsupported:
        names = ", ".join(sorted(unsupported))
        supported = ", ".join(sorted(SUPPORTED_COLLISION_CACHE_TYPES))
        raise ValueError(
            f"{label} contains types unsupported by cuRobo v0.8.0: "
            f"{names}; supported types: {supported}"
        )
    return parsed


def _parse_optional_int_mapping(
    value: object,
    *,
    default: Mapping[str, int | None],
    label: str,
) -> dict[str, int | None]:
    """解析 optimizer iteration override，允许显式 ``null`` 表示使用 cuRobo 默认值。"""

    if value is None:
        return dict(default)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    _reject_unknown_keys(value, {"particle", "lbfgs"}, label)
    parsed: dict[str, int | None] = {}
    for key, item in value.items():
        parsed[str(key)] = None if item is None else _strict_int(item, f"{label}.{key}")
    if any(item is not None and item < 0 for item in parsed.values()):
        raise ValueError(f"{label} values cannot be negative")
    return parsed


def _reject_unknown_keys(
    data: Mapping[Any, Any], allowed: set[str] | frozenset[str], label: str
) -> None:
    """拒绝固定 mapping 的未知字段，并报告完整嵌套路径。"""

    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        paths = ", ".join(f"{label}.{key}" for key in unknown)
        raise ValueError(f"unsupported configuration field(s): {paths}")


def _validate_non_negative_optional(value: float | int | None, label: str) -> None:
    """校验可选非负数。"""

    if value is not None and value < 0:
        raise ValueError(f"{label} cannot be negative")


def _validate_positive_optional(value: float | int | None, label: str) -> None:
    """校验可选正数。"""

    if value is not None and value <= 0:
        raise ValueError(f"{label} must be positive")


def _infer_urdf_root_link(urdf_path: Path) -> str:
    """从 URDF 推断根 link，作为 cuRobo ``base_link`` 默认值。"""

    root = ET.parse(urdf_path).getroot()
    link_names = {
        str(link.get("name")) for link in root.findall("link") if link.get("name")
    }
    child_links = {
        str(child.get("link"))
        for child in root.findall("joint/child")
        if child.get("link")
    }
    candidates = sorted(link_names - child_links)
    if len(candidates) != 1:
        raise ValueError(
            f"cannot infer unique cuRobo base_link from URDF {urdf_path}: {candidates}"
        )
    return candidates[0]
