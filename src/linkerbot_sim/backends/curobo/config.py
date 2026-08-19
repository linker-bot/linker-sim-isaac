"""cuRobo 后端的 typed 配置与机器人模型 YAML 边界。

算法、设备和完整后端配置只能由已经严格解析的项目 profile 单向投影，不能在 backend
再次解释任意 mapping。这里仅保留机器人模型与 TCP frame 的 canonical mapping parser，
因为它们属于 ``configs/robots`` 的资产 schema。真实模型、joint order 与 tool frame
存在性仍由 ``CuroboContext`` 或具体 solver 在运行时校验。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from linkerbot_sim.utils.paths import repo_path


DEFAULT_CUROBO_TASK_BUNDLE = "curobo_v0_8_default"
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
            # Direct (kinematics) IK uses cuRobo's LBFGS optimizer ONLY -- the MPPI
            # particle stage (ik/particle_ik.yml) is skipped. cuRobo 0.8.0's multi-stage
            # direct IK crashes at scale: with >=8 parallel IK problems whose seed stage
            # doesn't fully converge, it falls back to the MPPI stage whose RobotCostManager
            # comes up with an empty cost registry (particle_ik.yml's multi_link_pose/bound
            # costs unregistered) -> torch.cat([]) in get_sum_cost_and_constraint. LBFGS-only
            # (what motion_ik_optimizer_configs already does) avoids that path, is faster
            # (one stage), and converges from the warm-started seed for our use. See the
            # DemoGrasp-port ticket ticket-curobo-mppi-ik-empty-cost.md.
            ik_optimizer_configs=("ik/lbfgs_ik.yml",),
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
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]

    def __post_init__(self) -> None:
        """把直接构造和 YAML 构造统一冻结为有限三元组。"""

        object.__setattr__(self, "xyz", _vector3(self.xyz, "curobo TCP xyz"))
        object.__setattr__(self, "rpy", _vector3(self.rpy, "curobo TCP rpy"))

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
    """从 mode root 投影出的 cuRobo CUDA device 与固定 float32 合同。"""

    device: str = "cuda:0"
    tensor_dtype: str = "float32"
    collision_geometry_dtype: str = "float32"
    collision_gradient_dtype: str = "float32"
    collision_distance_dtype: str = "float32"

    def validate(self) -> None:
        """校验根配置派生的设备事实及项目固定的 dtype 合同。"""

        prefix, separator, index = self.device.partition(":")
        if prefix != "cuda" or separator != ":" or not index.isdecimal():
            raise ValueError(
                "curobo.device.device must be a canonical non-negative CUDA device"
            )
        for field_name in (
            "tensor_dtype",
            "collision_geometry_dtype",
            "collision_gradient_dtype",
            "collision_distance_dtype",
        ):
            if getattr(self, field_name) != "float32":
                raise ValueError(f"curobo.device.{field_name} must be 'float32'")


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
        _validate_collision_cache(
            self.collision_cache,
            "curobo.kinematics.ik.collision_cache",
        )


@dataclass(frozen=True)
class CuroboMotionPlannerConfig:
    """cuRobo 单请求 MotionPlanner 的求解和容量参数。

    ``max_goalset`` 是显式目标集资源上限。解析只接受项目公开字段，第三方 task 文件选择
    由 ``CuroboTaskBundle`` 固定管理。
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
    max_goalset: int = 1
    self_collision_check: bool = True
    collision_cache: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        """校验 MotionPlanner 数值参数。"""

        if self.num_ik_seeds <= 0 or self.num_trajopt_seeds <= 0:
            raise ValueError("cuRobo planner seed counts must be positive")
        if self.max_goalset <= 0:
            raise ValueError("cuRobo planner max_goalset must be positive")
        if self.random_seed < 0:
            raise ValueError("curobo.motion_planner.random_seed cannot be negative")
        if (
            self.position_tolerance < 0
            or self.orientation_tolerance < 0
            or self.optimizer_collision_activation_distance < 0
        ):
            raise ValueError("cuRobo planner tolerances cannot be negative")
        _validate_collision_cache(
            self.collision_cache,
            "curobo.motion_planner.collision_cache",
        )


@dataclass(frozen=True)
class CuroboConfig:
    """项目侧完整且可直接用于 materialize context 的 cuRobo 后端配置。

    配置 catalog 先把 robot 与算法 profile 分别解析为 typed 对象；composition 再注入 mode
    root 的唯一 CUDA 设备并构造本类型。runtime 不读取原始 YAML 或执行隐藏覆盖。
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

    def validate(self) -> None:
        """级联校验所有子配置。"""

        self.robot.validate()
        self.device.validate()
        self.ik.validate()
        self.motion_planner.validate()


def _mapping_or_empty(data: Mapping[str, Any] | None, label: str) -> Mapping[str, Any]:
    """读取可选 mapping。"""

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return data


def _optional_str(value: object | None, label: str) -> str | None:
    """严格读取可选非空字符串。"""

    if value is None:
        return None
    return _non_empty_str(value, label)


def _strict_bool(value: object, label: str) -> bool:
    """严格解析 bool，不接受 YAML/JSON 中的 truthy string。"""

    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


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


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    """严格解析非空字符串序列。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return tuple(
        _non_empty_str(item, f"{label}[{index}]") for index, item in enumerate(value)
    )


def _vector3(value: object, label: str) -> tuple[float, float, float]:
    """严格解析长度为三的有限数值向量。"""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of 3 numbers")
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 numbers")
    parsed = tuple(
        _strict_float(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    return parsed  # type: ignore[return-value]


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


def _validate_collision_cache(cache: Mapping[str, int], label: str) -> None:
    """校验 typed config 中传给 cuRobo 0.8.0 的场景缓存容量。"""

    unsupported = sorted(set(cache) - {"cuboid", "mesh"})
    if unsupported:
        raise ValueError(
            f"{label} contains types unsupported by cuRobo v0.8.0: "
            f"{', '.join(unsupported)}; supported types: cuboid, mesh"
        )
    for shape, capacity in cache.items():
        if type(capacity) is not int or capacity < 0:
            raise ValueError(f"{label}.{shape} must be a non-negative integer")


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
