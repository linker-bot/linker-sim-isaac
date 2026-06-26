"""cuMotion 机器人模型和共享资源上下文。

``CuMotionContext`` 是后端对象的共享缓存：它加载 XRDF/URDF、持有 kinematics 对象，并为
FK/IK 封装提供统一的关节名和 frame 名查询。cuMotion 导入被放到构造函数中，保证没有安装
该库时仍可运行配置解析和非后端相关测试。

职责边界:
    * 配置类只把 YAML/Mapping 中的路径、frame 名和后端参数规范化。
    * Context 只负责加载机器人描述和创建 FK/IK/planner 包装器，不直接执行任务流程。
    * 具体求解误差、碰撞世界和轨迹适配由同包其它模块负责。

坐标/顺序约定:
    cuMotion 的关节顺序来自 XRDF/URDF 解析后的 C-space，而不一定等于 Isaac articulation
    的完整 DOF 顺序。调用方必须通过 ``joint_names`` 做名称对齐后再把结果写回控制器。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from manipulation_project.utils.paths import repo_path


@dataclass(frozen=True)
class CuMotionConfig:
    """cuMotion 后端配置。

    路径字段指向 cuMotion 使用的 XRDF/URDF；frame 字段必须与 URDF link 名一致。
    容差单位为米和弧度，seed 数组按 cuMotion C-space 关节顺序排列。planner/trajectory
    参数会在具体后端对象创建时写入 cuMotion config 或 generator；本 dataclass 只保存
    项目侧可序列化的配置值。
    """

    # cuMotion XRDF 配置路径
    # 描述机器人 C-space、碰撞球、关节限制等后端规划信息
    xrdf_path: str | Path
    # cuMotion/URDF 机器人模型路径
    # 提供 link/frame 树、关节结构和 FK/IK 运动学描述
    urdf_path: str | Path
    # 机械臂法兰 frame 名
    # 这是没有自定义 TCP 时的默认末端 frame，也是临时 TCP 的父 frame
    flange_frame: str
    # 自定义 TCP frame 名
    # 只有 URDF/XRDF 中已经包含额外工具坐标系，或任务临时写入 fixed TCP frame 时才设置
    custom_tcp_frame: str | None = None
    # IK warm-start seed
    # 按 cuMotion C-space 关节顺序排列；可为单条 1D seed 或多条 2D seeds
    ik_cspace_seeds: np.ndarray | None = None
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
    collision_free_ik_params: dict[str, Any] = field(default_factory=dict)
    # 可选 cuMotion MotionPlanner 配置文件路径
    # 为空时使用 cuMotion 默认 planner config
    motion_planner_config_path: str | Path | None = None
    # MotionPlanner 的额外参数覆盖，按名称写入 cuMotion motion planner config
    motion_planner_params: dict[str, Any] = field(default_factory=dict)
    # 轨迹生成的关节限制覆盖，例如速度/加速度/jerk 上限
    # 每项按 C-space 关节顺序排列
    trajectory_limits: dict[str, np.ndarray] = field(default_factory=dict)
    # C-space trajectory generator 的 solver 参数覆盖，例如迭代次数或平滑相关参数
    trajectory_solver_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CuMotionConfig":
        """从 robot YAML 或 cumotion 子映射构造后端配置。

        配置文件中的路径按仓库根目录解析；求解器容差、迭代次数和可选 seed
        都属于 ``cumotion``，任务配置只负责描述任务意图。支持传入完整 robot YAML，
        也支持只传 ``cumotion`` 子字典，便于测试复用。
        """

        # robot YAML 可能把后端字段直接放在顶层，也可能放在 ``cumotion`` 子节点下。
        # 这里统一取出后再校验必填项，让脚本和测试可以复用同一个构造入口。
        settings = data.get("cumotion", data)
        if not isinstance(settings, Mapping):
            raise ValueError("cuMotion config must be a mapping")
        if "default_tcp_frame" in settings:
            raise ValueError("default_tcp_frame is removed; use custom_tcp_frame")
        if "cspace_seeds" in settings:
            raise ValueError("cspace_seeds is removed; use ik_cspace_seeds")

        # xrdf_path, urdf_path, flange_frame 是必须的
        missing = [
            key
            for key in ("xrdf_path", "urdf_path", "flange_frame")
            if not settings.get(key)
        ]
        if missing:
            raise ValueError(f"cuMotion config is missing required key(s): {missing}")

        # 路径立即按仓库根目录解析，避免后续后端对象依赖调用脚本当前工作目录。
        # seed、trajectory limit 和 frame 名先保持配置形式；它们是否匹配机器人 C-space
        # 只有 cuMotion 加载模型后才能可靠判断。
        custom_tcp_frame = settings.get("custom_tcp_frame")
        if custom_tcp_frame == settings["flange_frame"]:
            custom_tcp_frame = None

        config = cls(
            xrdf_path=repo_path(settings["xrdf_path"]),
            urdf_path=repo_path(settings["urdf_path"]),
            flange_frame=str(settings["flange_frame"]),
            custom_tcp_frame=None
            if custom_tcp_frame is None
            else str(custom_tcp_frame),
            ik_cspace_seeds=_parse_optional_cspace_seeds(
                settings.get("ik_cspace_seeds")
            ),
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
            collision_free_ik_params=_parse_optional_params_mapping(
                settings.get("collision_free_ik_params")
            ),
            motion_planner_config_path=_parse_optional_repo_path(
                settings.get("motion_planner_config_path")
            ),
            motion_planner_params=_parse_optional_params_mapping(
                settings.get("motion_planner_params")
            ),
            trajectory_limits=_parse_optional_array_mapping(
                settings.get("trajectory_limits")
            ),
            trajectory_solver_params=_parse_optional_params_mapping(
                settings.get("trajectory_solver_params")
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
        if not self.flange_frame:
            raise ValueError("flange_frame cannot be empty")
        if self.custom_tcp_frame is not None and not self.custom_tcp_frame:
            raise ValueError("custom_tcp_frame cannot be empty")
        if self.position_tolerance < 0 or self.orientation_tolerance < 0:
            raise ValueError("cuMotion tolerances cannot be negative")
        if self.ccd_max_iterations <= 0 or self.bfgs_max_iterations <= 0:
            raise ValueError("cuMotion iteration counts must be positive")
        if self.orientation_weight < 0:
            raise ValueError("orientation_weight cannot be negative")

        # 解析函数并校验可选字段，确保手动构造 CuMotionConfig 时也满足同样约束。
        _parse_optional_cspace_seeds(self.ik_cspace_seeds)
        _parse_optional_params_mapping(self.collision_free_ik_params)
        _parse_optional_params_mapping(self.motion_planner_params)
        _parse_optional_array_mapping(self.trajectory_limits)
        _parse_optional_params_mapping(self.trajectory_solver_params)

        # 可选 planner 配置文件路径允许为 None，但如果显式传入 Path/字符串则不能是空值。
        if self.motion_planner_config_path is not None and not str(
            self.motion_planner_config_path
        ):
            raise ValueError("motion_planner_config_path cannot be empty")


def _parse_optional_cspace_seeds(value) -> np.ndarray | None:
    """解析可选 C-space 种子数组。

    返回值可以是一条 seed（1D）或多条 seed（2D），均按 cuMotion C-space 关节顺序排列。
    这里只做形状和非空校验，具体长度由 ``CuMotionContext`` 加载机器人模型后检查。
    """

    if value is None:
        return None
    seeds = np.asarray(value, dtype=float)
    if seeds.ndim not in {1, 2} or seeds.size == 0:
        raise ValueError("ik_cspace_seeds must be a non-empty 1D or 2D array")
    return seeds


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


def _parse_optional_array_mapping(value) -> dict[str, np.ndarray]:
    """解析可选 C-space limit 映射。

    ``trajectory_limits`` 中每个值都转为 1D float 数组，长度是否等于机器人 C-space
    维度由 ``CuMotionContext`` 加载模型后检查。
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("trajectory_limits must be a mapping")
    limits = {}
    for key, limit_value in value.items():
        key = str(key)
        if not key:
            raise ValueError("trajectory limit names cannot be empty")
        array = np.asarray(limit_value, dtype=float).reshape(-1)
        if array.size == 0:
            raise ValueError(f"trajectory limit {key!r} cannot be empty")
        limits[key] = array
    return limits


def _parse_optional_repo_path(value) -> Path | None:
    """按仓库根目录解析可选路径。

    空字符串视为未配置，便于 YAML 中显式保留键但关闭 config-file 覆盖。
    """

    if value is None or value == "":
        return None
    return repo_path(value)


class CuMotionContext:
    """缓存 cuMotion robot description 和 kinematics。

    实例化该类是进入 cuMotion 后端的边界：如果第三方库缺失，会在这里给出明确错误；
    如果加载成功，后续 FK/IK 对象共享同一份 robot description，避免重复解析资产文件。
    """

    def __init__(self, config: CuMotionConfig) -> None:
        """延迟导入 cuMotion 并加载机器人描述文件。"""

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
        self._frame_names = [str(name) for name in self.kinematics.frame_names()]
        self._frame_name_set = set(self._frame_names)
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

    def _validate_model_dependent_config(self) -> None:
        """检查需要加载机器人模型后才能判断的配置。

        这里校验 seed/trajectory limit 宽度等于 C-space 维度，并确认默认 frame 存在。
        这样配置错误会在创建 context 时暴露，而不是等到某次 IK 或规划调用才由 pybind 抛错。
        """

        expected_cspace_width = len(self._joint_names)
        if self.config.ik_cspace_seeds is not None:
            seeds = np.asarray(self.config.ik_cspace_seeds, dtype=float)
            seed_width = seeds.size if seeds.ndim == 1 else seeds.shape[1]
            if seed_width != expected_cspace_width:
                raise ValueError(
                    "ik_cspace_seeds width mismatch: "
                    f"expected_cspace_width {expected_cspace_width}, got {seed_width}"
                )
        for key, values in self.config.trajectory_limits.items():
            if np.asarray(values, dtype=float).reshape(-1).size != expected_cspace_width:
                raise ValueError(
                    f"trajectory_limits.{key} expected_cspace_width {expected_cspace_width} values, "
                    f"got {np.asarray(values, dtype=float).reshape(-1).size}"
                )
        for frame_name, label in (
            (self.config.flange_frame, "flange_frame"),
            (self.config.custom_tcp_frame, "custom_tcp_frame"),
        ):
            if frame_name and not self.has_frame(frame_name):
                raise ValueError(
                    f"cuMotion {label} {frame_name!r} not found in robot frames"
                )

    def make_inverse_kinematics(self, *, tcp_frame_name: str | None = None):
        """创建逆运动学组件。

        ``tcp_frame_name`` 为空时优先使用自定义 TCP，否则回退到法兰 frame；返回对象会复用本 context 的
        ``robot_description`` 和 ``kinematics``。
        """

        from manipulation_project.backends.cumotion.inverse_kinematics import (
            CuMotionInverseKinematics,
        )

        # tcp_frame_name, custom_tcp_frame, flange_frame, 从左到右取第一个有效值
        return CuMotionInverseKinematics(
            self,
            tcp_frame_name=(
                tcp_frame_name
                or self.config.custom_tcp_frame
                or self.config.flange_frame
            ),
        )

    def make_forward_kinematics(self):
        """创建正运动学组件。

        FK wrapper 是轻量对象，主要负责把 cuMotion ``Pose3`` 转成项目统一 pose 结构。
        """

        from manipulation_project.backends.cumotion.forward_kinematics import (
            CuMotionForwardKinematics,
        )

        return CuMotionForwardKinematics(self)

    def make_motion_planner(
        self,
        *,
        tcp_frame_name: str | None = None,
        generate_interpolated_path: bool = True,
        generate_trajectory: bool = True,
        trajectory_mode: str = "time_optimal",
        trajectory_interpolation_mode: str = "linear",
        motion_planner_params: Mapping[str, Any] | None = None,
        trajectory_limits: Mapping[str, Any] | None = None,
        trajectory_solver_params: Mapping[str, Any] | None = None,
    ):
        """创建路径级运动规划组件。

        构造参数会作为本次 planner 的覆盖配置；同名键优先级高于 ``CuMotionConfig`` 中的
        默认映射。返回对象仍只处理 C-space 关节，不负责完整 articulation DOF 映射。
        """

        from manipulation_project.backends.cumotion.motion_planner import (
            CuMotionMotionPlanner,
        )
        
        # tcp_frame_name, custom_tcp_frame, flange_frame, 从左到右取第一个有效值
        return CuMotionMotionPlanner(
            self,
            tcp_frame_name=(
                tcp_frame_name
                or self.config.custom_tcp_frame
                or self.config.flange_frame
            ),
            generate_interpolated_path=generate_interpolated_path,
            generate_trajectory=generate_trajectory,
            trajectory_mode=trajectory_mode,
            trajectory_interpolation_mode=trajectory_interpolation_mode,
            motion_planner_params=motion_planner_params,
            trajectory_limits=trajectory_limits,
            trajectory_solver_params=trajectory_solver_params,
        )
