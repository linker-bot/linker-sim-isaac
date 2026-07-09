"""cuMotion 机器人模型和共享资源上下文。

``CuMotionContext`` 是后端对象的共享缓存：它加载 XRDF/URDF、持有 kinematics 对象和当前
环境的 cuMotion collision world，并为 FK/IK/planner 封装提供统一的关节名、frame 名和
world view 查询。cuMotion 导入被放到构造函数中，保证没有安装该库时仍可运行配置解析和
非后端相关测试。

职责边界:
    * 配置类只把 YAML/Mapping 中的路径、frame 名和后端参数规范化。
    * Context 负责加载机器人描述、维护当前环境 world，并创建 FK/IK/planner 包装器。
    * 具体求解误差、collision obstacle 适配和轨迹适配由同包其它模块负责。

坐标/顺序约定:
    cuMotion 的关节顺序来自 XRDF/URDF 解析后的 C-space，而不一定等于 Isaac articulation
    的完整 DOF 顺序。调用方必须通过 ``joint_names`` 做名称对齐后再把结果写回控制器。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np

from linkerbot_sim.backends.cumotion.tcp_frame import TcpFrame
from linkerbot_sim.backends.cumotion.tcp_urdf_builder import write_tcp_urdf_with_frames
from linkerbot_sim.planning.collision_objects import CollisionObject
from linkerbot_sim.utils.paths import repo_path
from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
)


@dataclass(frozen=True)
class CuMotionIkConfig:
    """cuMotion kinematics/IK 参数。

    这些字段最终会写入几何 IK 的 ``IkConfig``，或作为 collision-free IK 的 solver/config
    参数来源。容差单位为 m/rad，seed 按 cuMotion C-space 顺序排列。
    """

    # IK warm-start seed
    # 按 cuMotion C-space 关节顺序排列；可为单条 1D seed 或多条 2D seeds
    # 为 None 时项目侧不提供默认 seed，未提供 seed 时使用 cuMotion 默认初始化逻辑
    cspace_seeds: np.ndarray | None = None
    # IK 位置收敛容差，单位 m
    # 几何 IK 和 collision-free IK 都会使用该默认值
    position_tolerance: float = 0.005
    # IK 姿态收敛容差，单位 rad
    # 无姿态目标的请求会在 IK 层临时放宽/忽略该项
    orientation_tolerance: float = 0.75
    # cuMotion IK CCD 阶段最大迭代次数
    ccd_max_iterations: int = 180
    # cuMotion IK BFGS 精修阶段最大迭代次数
    bfgs_max_iterations: int = 80
    # IK 姿态误差权重
    orientation_weight: float = 0.25
    # collision-free IK 的额外后端参数，按名称写入 cuMotion solver config
    collision_free_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CuMotionIkConfig":
        """从 ``cumotion.kinematics.ik`` 解析 IK 配置。"""

        settings = _mapping_or_empty(data, "cumotion.kinematics.ik")
        _reject_removed_keys(
            settings,
            {"ik_cspace_seeds", "collision_free_ik_params"},
            label="cumotion.kinematics.ik",
        )
        config = cls(
            cspace_seeds=_parse_optional_cspace_seeds(settings.get("cspace_seeds")),
            position_tolerance=float(
                settings.get("position_tolerance", cls.position_tolerance)
            ),
            orientation_tolerance=float(
                settings.get("orientation_tolerance", cls.orientation_tolerance)
            ),
            ccd_max_iterations=int(
                settings.get("ccd_max_iterations", cls.ccd_max_iterations)
            ),
            bfgs_max_iterations=int(
                settings.get("bfgs_max_iterations", cls.bfgs_max_iterations)
            ),
            orientation_weight=float(
                settings.get("orientation_weight", cls.orientation_weight)
            ),
            collision_free_params=_parse_optional_params_mapping(
                settings.get("collision_free_params")
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """校验 IK 容差、迭代次数和可选 seeds/参数结构。"""

        if self.position_tolerance < 0 or self.orientation_tolerance < 0:
            raise ValueError("cuMotion kinematics.ik tolerances cannot be negative")
        if self.ccd_max_iterations <= 0 or self.bfgs_max_iterations <= 0:
            raise ValueError("cuMotion kinematics.ik iteration counts must be positive")
        if self.orientation_weight < 0:
            raise ValueError("kinematics.ik.orientation_weight cannot be negative")
        _parse_optional_cspace_seeds(self.cspace_seeds)
        _parse_optional_params_mapping(self.collision_free_params)


@dataclass(frozen=True)
class CuMotionFkConfig:
    """cuMotion FK 参数占位。

    当前 FK 封装没有可调后端参数，但保留该分组，让 YAML 和 dataclass 层次表达
    ``kinematics.fk`` / ``kinematics.ik`` 的边界。后续若要配置 base frame、输出 frame 策略等，
    可以在这里扩展。
    """

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CuMotionFkConfig":
        """解析 FK 分组；当前只允许空 mapping。"""

        _mapping_or_empty(data, "cumotion.kinematics.fk")
        return cls()

    def validate(self) -> None:
        """当前 FK 无可调字段，因此始终通过。"""

        return None


@dataclass(frozen=True)
class CuMotionKinematicsConfig:
    """cuMotion kinematics 分组配置。"""

    ik: CuMotionIkConfig = field(default_factory=CuMotionIkConfig)
    fk: CuMotionFkConfig = field(default_factory=CuMotionFkConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CuMotionKinematicsConfig":
        """解析 kinematics.ik/fk 两个子分组。"""

        settings = _mapping_or_empty(data, "cumotion.kinematics")
        config = cls(
            ik=CuMotionIkConfig.from_mapping(settings.get("ik")),
            fk=CuMotionFkConfig.from_mapping(settings.get("fk")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """级联校验 IK 和 FK 配置。"""

        self.ik.validate()
        self.fk.validate()


@dataclass(frozen=True)
class CuMotionConfig:
    """cuMotion 后端配置。

    路径字段指向 cuMotion 使用的 XRDF/URDF；frame 字段必须与 URDF link 名一致。kinematics
    和 motion planner 参数分别进入对应分组；本 dataclass 只保存项目侧可序列化的配置值。
    """

    # cuMotion XRDF 配置路径
    # 描述机器人 C-space、碰撞球、关节限制等后端规划信息
    xrdf_path: str | Path
    # cuMotion/URDF 机器人模型路径
    # 提供 link/frame 树、关节结构和 FK/IK 运动学描述
    urdf_path: str | Path
    # 默认末端 frame 名
    # 单臂配置通常填机械臂法兰；双臂融合模型不设置默认 frame，调用方必须显式选择 TCP。
    flange_frame: str | None
    # 默认 TCP frame 名；为空时回退到 flange_frame。该 frame 必须存在于最终 cuMotion context 中。
    default_tcp_frame: str | None = None
    # 从 YAML custom_tcps 解析得到的 fixed TCP frames。创建 context 前会 materialize 到派生 URDF。
    custom_tcp_frames: tuple[TcpFrame, ...] = ()
    # FK/IK 相关参数
    kinematics: CuMotionKinematicsConfig = field(default_factory=CuMotionKinematicsConfig)
    # motion-planner facade 的分组配置；为 None 时 context 会用上面的默认参数构造配置
    motion_planner: MotionPlannerBackendConfig | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, require_flange_frame: bool = True
    ) -> "CuMotionConfig":
        """从 robot YAML 或 cumotion 子映射构造后端配置。

        配置文件中的路径按仓库根目录解析；求解器容差、迭代次数和可选 seed
        都属于 ``cumotion``，动作配置只负责描述动作意图。支持传入完整 robot YAML，
        也支持只传 ``cumotion`` 子字典，便于测试复用。
        """

        # robot YAML 可能把后端字段直接放在顶层，也可能放在 ``cumotion`` 子节点下。
        # 这里统一取出后再校验必填项，让脚本和测试可以复用同一个构造入口。
        settings = data.get("cumotion", data)
        if not isinstance(settings, Mapping):
            raise ValueError("cuMotion config must be a mapping")

        # xrdf_path, urdf_path 是必须的；单臂入口默认还要求 flange_frame。
        missing = [
            key
            for key in ("xrdf_path", "urdf_path")
            if not settings.get(key)
        ]
        if require_flange_frame and not settings.get("flange_frame"):
            missing.append("flange_frame")
        if missing:
            raise ValueError(f"cuMotion config is missing required key(s): {missing}")

        # 路径立即按仓库根目录解析，避免后续后端对象依赖调用脚本当前工作目录。
        # seed 和 frame 名先保持配置形式；它们是否匹配机器人 C-space/URDF link 树，只有
        # cuMotion 加载模型后才能可靠判断。
        flange_frame = settings.get("flange_frame")
        default_tcp_frame = settings.get("default_tcp_frame")
        if flange_frame is not None and default_tcp_frame == flange_frame:
            default_tcp_frame = None
        custom_tcp_frames = _parse_custom_tcp_frames(
            settings.get("custom_tcps"),
            default_parent_frame=None if flange_frame is None else str(flange_frame),
            label="cumotion.custom_tcps",
        )

        _reject_removed_cumotion_fields(settings)
        kinematics = _kinematics_config_from_settings(settings)

        config = cls(
            xrdf_path=repo_path(settings["xrdf_path"]),
            urdf_path=repo_path(settings["urdf_path"]),
            flange_frame=None if flange_frame is None else str(flange_frame),
            default_tcp_frame=None
            if default_tcp_frame is None
            else str(default_tcp_frame),
            custom_tcp_frames=custom_tcp_frames,
            kinematics=kinematics,
            motion_planner=MotionPlannerBackendConfig.from_mapping(
                settings.get("motion_planner"),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """检查 cuMotion 后端配置字段。

        该检查只覆盖无需加载第三方库即可判断的边界条件，例如空路径、负容差和非正迭代
        次数。frame 是否真实存在、seed 长度是否匹配 C-space 等模型相关问题交给
        ``CuMotionContext`` 或具体求解器处理。
        """
        # 基础字段校验：这些值不依赖 cuMotion 模型，配置解析阶段即可判断是否合法。
        if not str(self.xrdf_path):
            raise ValueError("xrdf_path cannot be empty")
        if not str(self.urdf_path):
            raise ValueError("urdf_path cannot be empty")
        if self.flange_frame is not None and not self.flange_frame:
            raise ValueError("flange_frame cannot be empty")
        if self.default_tcp_frame is not None and not self.default_tcp_frame:
            raise ValueError("default_tcp_frame cannot be empty")
        _validate_unique_tcp_frames(self.custom_tcp_frames)

        # 解析函数并校验可选字段，确保手动构造 CuMotionConfig 时也满足同样约束。
        self.kinematics.validate()
        if self.motion_planner is not None:
            self.motion_planner.validate()


def _parse_optional_cspace_seeds(value) -> np.ndarray | None:
    """解析可选 C-space 种子数组。

    返回值可以是一条 seed（1D）或多条 seed（2D），均按 cuMotion C-space 关节顺序排列。
    这里只做形状和非空校验，具体长度由 ``CuMotionContext`` 加载机器人模型后检查。
    """

    if value is None:
        return None
    seeds = np.asarray(value, dtype=float)
    if seeds.ndim not in {1, 2} or seeds.size == 0:
        raise ValueError("cspace_seeds must be a non-empty 1D or 2D array")
    return seeds


def _mapping_or_empty(value, label: str) -> Mapping[str, Any]:
    """解析可选 mapping，用于分组配置入口。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _parse_custom_tcp_frames(
    value: object,
    *,
    default_parent_frame: str | None,
    label: str,
) -> tuple[TcpFrame, ...]:
    """解析 ``cumotion.custom_tcps`` 为已绑定 parent frame 的 fixed TCP frames。"""

    if value is None:
        return ()
    frames: list[TcpFrame] = []
    if isinstance(value, Mapping):
        iterable = value.items()
        for frame_name, frame_data in iterable:
            data = _mapping_or_empty(frame_data, f"{label}.{frame_name}")
            frames.append(
                _tcp_frame_from_mapping(
                    str(frame_name),
                    data,
                    default_parent_frame=default_parent_frame,
                    label=f"{label}.{frame_name}",
                )
            )
        return tuple(frames)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            data = _mapping_or_empty(item, f"{label}[{index}]")
            frame_name = data.get("frame_name")
            if not isinstance(frame_name, str) or not frame_name:
                raise ValueError(f"{label}[{index}].frame_name is required")
            frames.append(
                _tcp_frame_from_mapping(
                    frame_name,
                    data,
                    default_parent_frame=default_parent_frame,
                    label=f"{label}[{index}]",
                )
            )
        return tuple(frames)
    raise ValueError(f"{label} must be a mapping or a sequence")


def _tcp_frame_from_mapping(
    frame_name: str,
    data: Mapping[str, Any],
    *,
    default_parent_frame: str | None,
    label: str,
) -> TcpFrame:
    """从单个 custom TCP mapping 构造 ``TcpFrame``。"""

    if not frame_name:
        raise ValueError(f"{label}.frame_name cannot be empty")
    parent_frame = data.get("parent_frame", default_parent_frame)
    if not isinstance(parent_frame, str) or not parent_frame:
        raise ValueError(
            f"{label}.parent_frame is required when cumotion.flange_frame is not set"
        )
    return TcpFrame.from_xyz_rpy(
        frame_name=frame_name,
        parent_frame=parent_frame,
        xyz=data.get("xyz", (0.0, 0.0, 0.0)),
        rpy=data.get("rpy", (0.0, 0.0, 0.0)),
    )


def _validate_unique_tcp_frames(frames: Sequence[TcpFrame]) -> None:
    """校验自定义 TCP frame 名字唯一且 parent frame 非空。"""

    names: set[str] = set()
    for frame in frames:
        if not frame.frame_name:
            raise ValueError("custom TCP frame_name cannot be empty")
        if frame.frame_name in names:
            raise ValueError(f"Duplicate custom TCP frame: {frame.frame_name}")
        names.add(frame.frame_name)
        if not frame.parent_frame:
            raise ValueError(f"custom TCP {frame.frame_name!r} parent_frame cannot be empty")


def _kinematics_config_from_settings(
    settings: Mapping[str, Any]
) -> CuMotionKinematicsConfig:
    """解析 ``kinematics`` 分组。"""

    kinematics_settings = dict(_mapping_or_empty(settings.get("kinematics"), "kinematics"))
    ik_settings = dict(_mapping_or_empty(kinematics_settings.get("ik"), "kinematics.ik"))
    fk_settings = _mapping_or_empty(kinematics_settings.get("fk"), "kinematics.fk")
    return CuMotionKinematicsConfig(
        ik=CuMotionIkConfig.from_mapping(ik_settings),
        fk=CuMotionFkConfig.from_mapping(fk_settings),
    )


def _reject_removed_cumotion_fields(settings: Mapping[str, Any]) -> None:
    """拒绝旧版扁平 cuMotion 字段，提示用户使用当前分组 schema。"""

    removed_keys = {
        "ik_cspace_seeds",
        "cspace_seeds",
        "position_tolerance",
        "orientation_tolerance",
        "ccd_max_iterations",
        "bfgs_max_iterations",
        "orientation_weight",
        "collision_free_ik_params",
        "collision_free_params",
        "motion_planner_config_path",
        "motion_planner_params",
        "trajectory_limits",
        "trajectory_solver_params",
        "custom_tcp_frame",
    }
    _reject_removed_keys(settings, removed_keys, label="cumotion")


def _reject_removed_keys(
    settings: Mapping[str, Any], removed_keys: set[str], *, label: str
) -> None:
    """通用旧字段拒绝 helper。"""

    present = sorted(set(settings) & removed_keys)
    if present:
        raise ValueError(
            f"{label} contains removed field(s): {present}. "
            "Use the current grouped configuration schema instead."
        )


def _parse_optional_params_mapping(value) -> dict[str, Any]:
    """解析可选 cuMotion 参数映射。

    参数值会在具体后端对象中包装成 cuMotion 对应的 ``ParamValue``。这里不限制
    value 类型，保留官方 API 支持的 bool/int/float/list/str 等形态。
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("cuMotion params must be a mapping")
    params = {}
    for key, param_value in value.items():
        key = str(key)
        if not key:
            raise ValueError("cuMotion param names cannot be empty")
        params[key] = param_value
    return params


def materialize_cumotion_config(config: CuMotionConfig) -> CuMotionConfig:
    """把 YAML 声明的 custom TCP fixed frames materialize 到派生 URDF。

    运行时 JSON 只能选择已经存在的 frame；因此 ``custom_tcps`` 必须在创建 cuMotion context
    前写入 URDF。输出路径使用输入资源和 TCP 定义的 hash，稳定写入 ``.cache/cumotion``。
    """

    if not config.custom_tcp_frames:
        return config
    output_path = _custom_tcp_urdf_path(config)
    if not output_path.is_file():
        write_tcp_urdf_with_frames(
            config.urdf_path,
            output_path,
            config.custom_tcp_frames,
        )
    return replace(config, urdf_path=output_path, custom_tcp_frames=())


def default_tcp_frame_name(config: CuMotionConfig) -> str | None:
    """返回配置层默认 TCP frame，未配置时回退到 flange frame。"""

    return config.default_tcp_frame or config.flange_frame


def resolve_tcp_frame_name(
    context: object,
    *,
    tcp_frame_name: str | None = None,
    default_tcp_frame_name: str | None = None,
    label: str = "tcp_frame_name",
) -> str:
    """按统一优先级解析并校验 cuMotion frame。

    优先级为显式 ``tcp_frame_name``、调用方默认、context config 默认 TCP、flange frame。
    该函数不创建 frame；解析出的 frame 必须已经在 context 的 frame 集合中。
    """

    config = getattr(context, "config", None)
    frame_name = (
        tcp_frame_name
        or default_tcp_frame_name
        or (default_tcp_frame_name_from_config(config) if config is not None else None)
    )
    if frame_name is None:
        raise ValueError(f"{label} is required because this cuMotion config has no default frame")
    frame = str(frame_name)
    validate_cumotion_frame(context, frame, label=label)
    return frame


def default_tcp_frame_name_from_config(config: object) -> str | None:
    """从 CuMotionConfig-like 对象读取默认 frame。"""

    return (
        getattr(config, "default_tcp_frame", None)
        or getattr(config, "flange_frame", None)
    )


def validate_cumotion_frame(context: object, frame_name: str, *, label: str = "frame") -> None:
    """校验 frame 名非空，且在支持查询的 context 中必须存在。"""

    if not str(frame_name):
        raise ValueError(f"{label} cannot be empty")
    if hasattr(context, "has_frame") and not context.has_frame(str(frame_name)):
        raise ValueError(f"cuMotion frame {frame_name!r} not found")


def _custom_tcp_urdf_path(config: CuMotionConfig) -> Path:
    """为 custom TCP 派生 URDF 生成稳定缓存路径。"""

    base = Path(config.urdf_path)
    frames = [
        {
            "frame_name": frame.frame_name,
            "parent_frame": frame.parent_frame,
            "xyz": np.asarray(frame.xyz, dtype=float).reshape(3).tolist(),
            "rpy": np.asarray(frame.rpy, dtype=float).reshape(3).tolist(),
        }
        for frame in config.custom_tcp_frames
    ]
    digest = hashlib.sha256(
        json.dumps(
            {
                "urdf_path": str(base.resolve()),
                "urdf_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "frames": frames,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return repo_path(".cache/cumotion") / f"{base.stem}_custom_tcps_{digest}.urdf"


def urdf_link_names(urdf_path: str | Path) -> set[str]:
    """读取 URDF 中所有 link 名称。"""

    root = ET.parse(Path(urdf_path)).getroot()
    return {str(link.get("name")) for link in root.findall("link") if link.get("name")}


class CuMotionContext:
    """缓存 cuMotion robot description 和 kinematics。

    实例化该类是进入 cuMotion 后端的边界：如果第三方库缺失，会在这里给出明确错误；
    如果加载成功，后续 FK/IK 对象共享同一份 robot description，避免重复解析资产文件。
    """

    def __init__(self, config: CuMotionConfig) -> None:
        """延迟导入 cuMotion 并加载机器人描述文件。"""

        config = materialize_cumotion_config(config)

        # cuMotion 是可选且体量较大的依赖。把导入错误包装成项目语义更清晰的提示，
        # 可以让用户区分“后端未安装”和“配置文件写错”。
        try:
            import cumotion
        except ImportError as exc:
            raise ImportError(
                "cuMotion is not installed in this Python environment. Install the NVIDIA "
                "cuMotion package from https://github.com/nvidia-isaac/cumotion/releases."
            ) from exc

        self.cumotion = cumotion
        self.config = config
        # XRDF 提供 cuMotion 的语义配置，URDF 提供几何/运动链描述；二者必须同时加载，
        # 才能得到后续 FK/IK 共享的 kinematics 对象。
        self.robot_description = cumotion.load_robot_from_file(
            str(config.xrdf_path), str(config.urdf_path)
        )
        self.kinematics = self.robot_description.kinematics()
        self._joint_names = [
            str(self.kinematics.cspace_coord_name(index))
            for index in range(self.kinematics.num_cspace_coords())
        ]
        self.expected_cspace_width = len(self._joint_names)
        self._frame_names = [str(name) for name in self.kinematics.frame_names()]
        self._frame_name_set = set(self._frame_names)
        self._collision_world = None
        self._empty_collision_world = None
        self._validate_model_dependent_config()

    def joint_names(self) -> list[str]:
        """返回 cuMotion C-space 主动关节名。

        返回副本而不是内部列表，避免调用方无意修改 context 缓存。
        """

        return list(self._joint_names)

    def frame_names(self) -> list[str]:
        """返回 cuMotion 可查询 frame 名。

        名称来自加载后的 kinematics；自定义 TCP 必须先写入 URDF 才会出现在这里。
        """

        return list(self._frame_names)

    def has_frame(self, frame_name: str) -> bool:
        """检查 frame 是否存在。

        使用构造时缓存的 set，适合在 IK/planner 请求边界频繁做快速校验。
        """

        return str(frame_name) in self._frame_name_set

    def collision_world(self):
        """返回 context 当前管理的环境 collision world。

        如果还没有设置过环境，会创建一个空 ``CuMotionCollisionWorld``。因此 IK/planner 可以
        总是从 context 获取 world view，而不需要在求解点临时决定是否构造后端 ``World``。
        """

        if self._collision_world is None:
            return self.sync_collision_world(())
        return self._collision_world

    def sync_collision_world(
        self, collision_objects: Sequence[CollisionObject] = ()
    ):
        """用最新环境对象同步 context 持有的 cuMotion collision world。

        ``collision_objects`` 是当前环境快照，按名称增量同步到同一个后端 ``World``：
        新对象会添加，缺失对象会删除，已有对象会更新 pose/启停状态。动作脚本层可以在环境变化后
        调用本方法一次，后续 IK/planner 请求会复用同步后的 ``world_view``。
        """

        from linkerbot_sim.backends.cumotion.collision_world import (
            CuMotionCollisionWorld,
        )

        objects = tuple(collision_objects)
        if self._collision_world is None:
            self._collision_world = CuMotionCollisionWorld(self, objects)
        else:
            self._collision_world.sync(objects)
        return self._collision_world

    def clear_collision_world(self):
        """清空 context 当前环境，并返回清空后的 collision world。

        这会修改 context 管理的真实环境缓存；如果只想为某次几何规划临时使用空 world，
        应使用 ``empty_collision_world``，避免误删当前环境。
        """

        return self.sync_collision_world(())

    def empty_collision_world(self):
        """返回 context 复用的空 collision world。

        该 world 不包含环境 obstacle，专供几何/忽略环境障碍的规划分支使用。它与
        ``collision_world`` 维护的真实环境分离，并会在 context 内缓存复用；调用方不应向它
        添加或同步 obstacle，避免污染后续无障碍规划。
        """

        from linkerbot_sim.backends.cumotion.collision_world import (
            CuMotionCollisionWorld,
        )

        if self._empty_collision_world is None:
            self._empty_collision_world = CuMotionCollisionWorld(self, ())
        return self._empty_collision_world

    def _validate_model_dependent_config(self) -> None:
        """检查需要加载机器人模型后才能判断的配置。

        这里校验 seed/trajectory limit 宽度等于 C-space 维度，并确认默认 frame 存在。
        这样配置错误会在创建 context 时暴露，而不是等到某次 IK 或规划调用才由 pybind 抛错。
        """

        if self.config.kinematics.ik.cspace_seeds is not None:
            seeds = np.asarray(self.config.kinematics.ik.cspace_seeds, dtype=float)
            seed_width = seeds.size if seeds.ndim == 1 else seeds.shape[1]
            if seed_width != self.expected_cspace_width:
                raise ValueError(
                    "kinematics.ik.cspace_seeds width mismatch: "
                    f"expected_cspace_width {self.expected_cspace_width}, got {seed_width}"
                )
        if self.config.motion_planner is not None:
            for (
                key,
                values,
            ) in self.config.motion_planner.trajectory_generation.limits.items():
                if (
                    np.asarray(values, dtype=float).reshape(-1).size
                    != self.expected_cspace_width
                ):
                    raise ValueError(
                        "motion_planner.trajectory_generation.limits."
                        f"{key} expected_cspace_width {self.expected_cspace_width} values, "
                        f"got {np.asarray(values, dtype=float).reshape(-1).size}"
                    )
        for frame_name, label in (
            (self.config.flange_frame, "flange_frame"),
            (self.config.default_tcp_frame, "default_tcp_frame"),
        ):
            if frame_name and not self.has_frame(frame_name):
                raise ValueError(
                    f"cuMotion {label} {frame_name!r} not found in robot frames"
                )

    def make_inverse_kinematics(self, *, tcp_frame_name: str | None = None):
        """创建逆运动学组件。

        ``tcp_frame_name`` 为空时优先使用默认 TCP，否则回退到法兰 frame；返回对象会复用本 context 的
        ``robot_description`` 和 ``kinematics``。
        """

        from linkerbot_sim.backends.cumotion.inverse_kinematics import (
            CuMotionInverseKinematics,
        )

        frame_name = resolve_tcp_frame_name(
            self,
            tcp_frame_name=tcp_frame_name,
            label="tcp_frame_name",
        )
        return CuMotionInverseKinematics(
            self,
            tcp_frame_name=frame_name,
        )

    def make_forward_kinematics(self):
        """创建正运动学组件。

        FK wrapper 是轻量对象，主要负责把 cuMotion ``Pose3`` 转成项目统一 pose 结构。
        """

        from linkerbot_sim.backends.cumotion.forward_kinematics import (
            CuMotionForwardKinematics,
        )

        return CuMotionForwardKinematics(self)

    def make_motion_planner(
        self,
        *,
        tcp_frame_name: str | None = None,
        config: MotionPlannerBackendConfig | None = None,
    ):
        """创建路径级运动规划组件。

        ``config`` 是新分组配置模型；为空时使用 ``CuMotionConfig.motion_planner``，再
        回退到默认 ``MotionPlannerBackendConfig``。返回对象仍只处理 C-space 关节，不负责
        完整 articulation DOF 映射。
        """

        from linkerbot_sim.backends.cumotion.motion_planner import (
            CuMotionMotionPlanner,
        )

        frame_name = resolve_tcp_frame_name(
            self,
            tcp_frame_name=tcp_frame_name,
            label="tcp_frame_name",
        )
        return CuMotionMotionPlanner(
            self,
            tcp_frame_name=frame_name,
            config=config,
        )
