"""cuMotion 路径级运动规划封装。

本模块把项目的 ``MotionRequest`` 转换成 cuMotion ``MotionPlanner`` 调用，并把结果
归一化为 ``MotionResult``。MotionPlanner 负责连续路径级避障搜索；返回的 path 是
cuMotion C-space 关节顺序。若可用，本封装还会用 cuMotion C-space trajectory generator
把搜索路径时间参数化，供上层通过 ``trajectory_adapter`` 采样。

职责边界:
    * 只处理后端 C-space，不映射 Isaac articulation 完整 DOF 或 controller command space。
    * 碰撞世界按请求构建为 cuMotion ``WorldView``；动态物体需要由调用方在请求层更新。
    * ``mode`` 只解释是否使用环境障碍；planner/trajectory 细节通过 context 或构造参数配置。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from manipulation_project.backends.cumotion.collision_world import make_collision_world
from manipulation_project.backends.cumotion.pose_adapter import target_pose
from manipulation_project.planning.requests import MotionRequest
from manipulation_project.planning.results import MotionResult, PlanningDiagnostics


_COLLISION_DISABLED_MODES = {
    # ``MotionRequest.mode`` 是后端可解释的策略标签，不做成 enum 是为了给任务层保留
    # 扩展空间。这里列出的模式表示“仍使用 cuMotion MotionPlanner，但不给它环境障碍”；
    # 机器人自身碰撞和关节限制仍由 cuMotion robot description / planner 配置决定。
    "geometric",
    "ignore_collision",
    "ignore_collisions",
    "no_collision",
    "no_collisions",
    "no_obstacle",
    "no_obstacles",
    "collision_unaware",
}


class CuMotionMotionPlanner:
    """使用 cuMotion ``MotionPlanner`` 规划路径级运动。

    该类封装 graph-based path search 和可选 C-space trajectory generation。输入输出都保持
    cuMotion C-space 关节顺序；完整 articulation DOF 裁剪和回填属于任务层职责。
    """

    def __init__(
        self,
        context,
        *,
        tcp_frame_name: str | None = None,
        generate_interpolated_path: bool = True,
        generate_trajectory: bool = True,
        trajectory_mode: str = "time_optimal",
        trajectory_interpolation_mode: str = "linear",
        motion_planner_params: Mapping[str, Any] | None = None,
        trajectory_limits: Mapping[str, Any] | None = None,
        trajectory_solver_params: Mapping[str, Any] | None = None,
    ) -> None:
        """创建 motion planner 封装。

        ``tcp_frame_name`` 是任务空间目标使用的工具 frame；关节空间目标也需要一个 frame
        来构造 cuMotion planner config，因此未指定时使用后端默认 TCP/frame。

        ``generate_interpolated_path`` 控制 cuMotion 是否额外返回插值后的路径采样。项目层优先
        消费这个较密的 path；如果关闭或后端未返回，则回退到 planner 的稀疏搜索节点。
        ``generate_trajectory`` 控制是否把 path 再交给 cuMotion trajectory generator 做时间
        参数化。调用方只需要离散路径时可以关闭它，省掉一次后处理。
        ``trajectory_mode`` 选择时间参数化入口：``time_optimal`` 使用 cuMotion 根据约束生成
        时间最优轨迹；``time_stamped`` 使用请求给定阶段时长为每个 waypoint 分配时间戳，再按
        ``trajectory_interpolation_mode`` 调用 cuMotion 的插值模式。

        参数覆盖采用“context 默认值 + 本次构造参数”的合并规则，构造参数同名键优先。
        ``motion_planner_params`` 写入 ``MotionPlannerConfig.set_param``；
        ``trajectory_limits`` 写入 trajectory generator 的 limit setter；
        ``trajectory_solver_params`` 写入 ``CSpaceTrajectoryGenerator.set_solver_param``。
        """

        self.context = context
        self.cumotion = context.cumotion
        self.tcp_frame_name = str(
            tcp_frame_name
            or context.config.custom_tcp_frame
            or context.config.flange_frame
        )
        self.generate_interpolated_path = bool(generate_interpolated_path)
        self.generate_trajectory = bool(generate_trajectory)
        self.trajectory_mode = _normalize_trajectory_mode(trajectory_mode)
        self.trajectory_interpolation_mode = _normalize_interpolation_mode(
            trajectory_interpolation_mode
        )
        self.motion_planner_params = _merged_mapping(
            getattr(context.config, "motion_planner_params", {}),
            motion_planner_params,
        )
        self.trajectory_limits = _merged_mapping(
            getattr(context.config, "trajectory_limits", {}),
            trajectory_limits,
        )
        self.trajectory_solver_params = _merged_mapping(
            getattr(context.config, "trajectory_solver_params", {}),
            trajectory_solver_params,
        )

    def joint_names(self) -> list[str]:
        """返回 planner 使用的 C-space 关节名。

        顺序与 ``MotionRequest.current_q``、``goal_q`` 和返回的 ``MotionResult.joint_path``
        完全一致。
        """

        return self.context.joint_names()

    def plan(self, request: MotionRequest) -> MotionResult:
        """根据 ``MotionRequest`` 调用 cuMotion MotionPlanner。

        请求必须提供当前 C-space 位置，并在 ``goal_q`` 和 ``goal_pose`` 中二选一。成功时返回
        离散 C-space path；若 ``generate_trajectory`` 为真，还会返回 cuMotion trajectory。
        """

        # MotionRequest 是后端无关的数据结构；进入 cuMotion 边界后先做无需加载机器人模型
        # 的结构校验，再用当前 context 的 C-space 维度校验关节向量长度。这里不猜测完整
        # articulation DOF 顺序，调用方必须已经把状态裁剪/重排成 ``context.joint_names()``。
        request.validate()
        frame_name = str(request.tcp_frame_name or self.tcp_frame_name)
        current = np.asarray(request.current_q, dtype=float).reshape(-1)
        self._validate_cspace_length(current, "current_q")

        # cuMotion MotionPlanner 接收的是 WorldView 快照，而不是 Isaac stage。项目层必须把
        # 需要参与规划的环境物体显式转换成 CollisionObject；mode 为 geometric 等值时传空
        # 世界，从而得到“不考虑环境障碍”的规划结果。
        collision_objects = _collision_objects_for_mode(request)
        collision_world = make_collision_world(self.context, collision_objects)
        config = self._motion_planner_config(frame_name, collision_world.world_view)
        planner = self.cumotion.create_motion_planner(config)

        # MotionRequest 允许两类互斥目标：
        # 1. goal_q: 后端 C-space 中的确定目标构型，直接走 JtRRT 的 c-space target。
        # 2. goal_pose: TCP 任务空间目标。只有 position 时走 translation target；带 orientation
        #    时构造成 Pose3 走 pose target。orientation 使用项目边界的 wxyz 四元数。
        if request.goal_q is not None:
            goal = np.asarray(request.goal_q, dtype=float).reshape(-1)
            self._validate_cspace_length(goal, "goal_q")
            results = planner.plan_to_cspace_target(
                current, goal, self.generate_interpolated_path
            )
            target_type = "cspace"
        else:
            pose = request.goal_pose
            if pose is None:
                raise ValueError("Exactly one of goal_q or goal_pose must be provided")
            if pose.orientation is None:
                translation = np.asarray(pose.position, dtype=float).reshape(3)
                results = planner.plan_to_translation_target(
                    current, translation, self.generate_interpolated_path
                )
                target_type = "translation"
            else:
                pose_target = target_pose(
                    self.cumotion, pose.position, pose.orientation
                )
                results = planner.plan_to_pose_target(
                    current, pose_target, self.generate_interpolated_path
                )
                target_type = "pose"

        return self._motion_result(
            results,
            target_type=target_type,
            frame_name=frame_name,
            num_collision_objects=len(collision_objects),
            duration_s=request.duration_s,
        )

    def _validate_cspace_length(self, values: np.ndarray, label: str) -> None:
        """确保请求关节向量长度匹配 cuMotion C-space。"""

        # cuMotion 只知道 XRDF/URDF 中的主动 C-space 关节，通常不包含灵巧手 mimic follower
        # 或组合 articulation 的全部 DOF。长度不匹配时立即报错，比把错误路径写回控制器安全。
        if values.size != self.context.expected_cspace_width:
            raise ValueError(
                f"{label} expected {self.context.expected_cspace_width} values, got {values.size}"
            )

    def _motion_result(
        self,
        results,
        *,
        target_type: str,
        frame_name: str,
        num_collision_objects: int,
        duration_s: float | None,
    ) -> MotionResult:
        """把 cuMotion planner results 归一化成项目 MotionResult。"""

        # cuMotion Results 中可能同时包含 sparse ``path`` 和较密的 ``interpolated_path``。
        # 控制执行通常更适合使用插值后的路径；如果后端没有生成它，则退回 sparse path。
        path_samples = _result_path_samples(
            results, prefer_interpolated=self.generate_interpolated_path
        )
        joint_path = _stack_path(path_samples)
        # path_found 为真但没有实际 path 时仍视为失败，避免上层拿到 success=True 却没有
        # 可执行关节路径。真实 cuMotion 暴露的是属性；测试替身可能用零参方法，统一用 _attr。
        success = (
            bool(_attr(results, "path_found", default=False))
            and joint_path is not None
        )
        trajectory = (
            self._generate_trajectory(joint_path, duration_s=duration_s)
            if success and self.generate_trajectory
            else None
        )
        # diagnostics 只放轻量可打印的数值摘要；原始 cuMotion trajectory/path 保留在
        # MotionResult 字段中，避免把后端对象混入 metrics。
        metrics = {
            "num_waypoints": float(0 if joint_path is None else joint_path.shape[0]),
            "num_collision_objects": float(num_collision_objects),
            "path_length": float(0.0 if joint_path is None else _path_length(joint_path)),
        }
        diagnostics = PlanningDiagnostics(
            status="SUCCESS" if success else "FAILED",
            message=(
                f"cuMotion MotionPlanner target={target_type} frame={frame_name}"
            ),
            metrics=metrics,
        )
        return MotionResult(
            joint_path=joint_path if success else None,
            trajectory=trajectory,
            success=success,
            status=diagnostics.status,
            diagnostics=diagnostics,
        )

    def _generate_trajectory(
        self, joint_path: np.ndarray | None, *, duration_s: float | None = None
    ):
        """用 cuMotion C-space trajectory generator 给路径做时间参数化。"""

        # 单点 path 没有时间参数化的意义，直接返回 None。两点及以上时，trajectory generator
        # 会生成 cuMotion Trajectory；项目已有 trajectory_adapter 负责按控制频率采样它。
        if joint_path is None or joint_path.shape[0] < 2:
            return None
        generator = self.cumotion.create_cspace_trajectory_generator(
            self.context.kinematics
        )
        self._configure_trajectory_generator(generator)
        waypoints = [np.asarray(row, dtype=float).reshape(-1) for row in joint_path]
        if self.trajectory_mode == "time_optimal":
            return generator.generate_trajectory(waypoints)
        if duration_s is None:
            raise ValueError(
                "duration_s is required when trajectory_mode='time_stamped'"
            )
        if duration_s <= 0.0:
            raise ValueError(
                "duration_s must be positive when trajectory_mode='time_stamped'"
            )
        times = _times_for_joint_path(joint_path, float(duration_s))
        return generator.generate_time_stamped_trajectory(
            waypoints,
            times,
            self._trajectory_interpolation_mode(),
        )

    def _trajectory_interpolation_mode(self):
        """返回 cuMotion ``CSpaceTrajectoryGenerator.InterpolationMode`` 枚举值。"""

        interpolation_mode = self.cumotion.CSpaceTrajectoryGenerator.InterpolationMode
        if self.trajectory_interpolation_mode == "linear":
            return interpolation_mode.LINEAR
        if self.trajectory_interpolation_mode == "cubic_spline":
            return interpolation_mode.CUBIC_SPLINE
        raise ValueError(
            f"Unsupported trajectory_interpolation_mode={self.trajectory_interpolation_mode!r}"
        )

    def _motion_planner_config(self, frame_name: str, world_view):
        """创建并配置 cuMotion ``MotionPlannerConfig``。

        优先使用 ``context.config.motion_planner_config_path`` 指向的配置文件；未配置时使用
        cuMotion 默认配置。随后统一应用 ``motion_planner_params``，让 YAML/default file 与
        任务级覆盖可以组合使用。
        """

        config_path = getattr(self.context.config, "motion_planner_config_path", None)
        if config_path:
            config = self.cumotion.create_motion_planner_config_from_file(
                Path(config_path),
                self.context.robot_description,
                frame_name,
                world_view,
            )
        else:
            config = self.cumotion.create_default_motion_planner_config(
                self.context.robot_description,
                frame_name,
                world_view,
            )
        _apply_config_params(
            config,
            self.motion_planner_params,
            self.cumotion.MotionPlannerConfig.ParamValue,
        )
        return config

    def _configure_trajectory_generator(self, generator) -> None:
        """应用 trajectory limits 和 solver params。

        支持的 limit 键为 ``position_min``、``position_max``、``velocity``、
        ``acceleration`` 和 ``jerk``，若设置位置上下界必须同时给出。
        """

        limits = _normalized_trajectory_limits(self.trajectory_limits)
        if "position_min" in limits or "position_max" in limits:
            if "position_min" not in limits or "position_max" not in limits:
                raise ValueError(
                    "trajectory_limits position_min and position_max must be set together"
                )
            generator.set_position_limits(
                limits["position_min"], limits["position_max"]
            )
        if "velocity" in limits:
            generator.set_velocity_limits(limits["velocity"])
        if "acceleration" in limits:
            generator.set_acceleration_limits(limits["acceleration"])
        if "jerk" in limits:
            generator.set_jerk_limits(limits["jerk"])
        _apply_config_params(
            generator,
            self.trajectory_solver_params,
            self.cumotion.CSpaceTrajectoryGenerator.SolverParamValue,
            setter_name="set_solver_param",
        )


def _collision_objects_for_mode(request: MotionRequest):
    """根据 ``mode`` 决定是否把环境障碍传给 cuMotion world。"""

    # 默认 ``collision_aware`` 会保留请求中的障碍物。这里仅用 mode 控制“环境障碍”是否传入
    # World；不是在这里开关 cuMotion 的所有碰撞逻辑。
    mode = str(request.mode or "").lower()
    if mode in _COLLISION_DISABLED_MODES:
        return ()
    return tuple(request.collision_objects)


def _merged_mapping(*mappings) -> dict[str, Any]:
    """合并可选映射，后面的映射覆盖前面的键。

    该函数用于实现 context 配置与任务级覆盖的优先级；值本身不做类型转换，保留给 cuMotion
    对应的 ``ParamValue`` 构造器处理。
    """

    merged: dict[str, Any] = {}
    for mapping in mappings:
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise ValueError("cuMotion parameter overrides must be mappings")
        merged.update({str(key): value for key, value in mapping.items()})
    return merged


def _apply_config_params(
    target, params: Mapping[str, Any], param_value_type, *, setter_name: str = "set_param"
) -> None:
    """把项目参数映射写入 cuMotion config/generator。

    cuMotion 的 ``set_param``/``set_solver_param`` 返回 ``False`` 时通常表示参数名或类型不被
    后端接受；这里立即抛出 ``ValueError``，避免后续规划悄悄退回默认参数。
    """

    setter = getattr(target, setter_name)
    for name, value in params.items():
        ok = setter(str(name), param_value_type(value))
        if ok is False:
            raise ValueError(f"cuMotion config rejected parameter {name!r}")


def _normalized_trajectory_limits(limits: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """规范化 trajectory limit 键名。

    项目配置允许少量同义键，例如 ``max_velocity`` 和 ``velocity_limit``。返回值只包含
    cuMotion wrapper 内部认可的规范键，未知键会报错以防拼写错误被忽略。
    """

    aliases = {
        "min_position": "position_min",
        "position_lower": "position_min",
        "lower_position": "position_min",
        "max_position": "position_max",
        "position_upper": "position_max",
        "upper_position": "position_max",
        "max_velocity": "velocity",
        "velocity_limit": "velocity",
        "max_acceleration": "acceleration",
        "acceleration_limit": "acceleration",
        "max_jerk": "jerk",
        "jerk_limit": "jerk",
    }
    normalized = {}
    for key, value in limits.items():
        normalized_key = aliases.get(str(key), str(key))
        normalized[normalized_key] = np.asarray(value, dtype=float).reshape(-1)
    unknown = set(normalized) - {
        "position_min",
        "position_max",
        "velocity",
        "acceleration",
        "jerk",
    }
    if unknown:
        raise ValueError(f"Unsupported trajectory_limits key(s): {sorted(unknown)}")
    return normalized


def _normalize_trajectory_mode(value: str) -> str:
    """规范化 cuMotion trajectory generator 模式。"""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"time_optimal", "optimal", "default"}:
        return "time_optimal"
    if normalized in {"time_stamped", "timestamped", "time_stamp"}:
        return "time_stamped"
    raise ValueError("trajectory_mode must be one of: time_optimal, time_stamped")


def _normalize_interpolation_mode(value: str) -> str:
    """规范化 cuMotion time-stamped trajectory 插值模式。"""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "linear":
        return "linear"
    if normalized in {"cubic", "cubic_spline", "spline"}:
        return "cubic_spline"
    raise ValueError(
        "trajectory_interpolation_mode must be one of: linear, cubic_spline"
    )


def _result_path_samples(results, *, prefer_interpolated: bool) -> list[np.ndarray]:
    """读取 cuMotion Results 中的 path，兼容属性和零参方法两种 fake/pybind 形态。"""

    # pybind 类型在不同版本/测试替身里可能把字段暴露成属性或方法；_attr 统一读取。
    # 当 prefer_interpolated=True 时优先使用后端插值路径，空列表则继续尝试 sparse path。
    names = ("interpolated_path", "path") if prefer_interpolated else ("path",)
    for name in names:
        samples = _attr(results, name, default=())
        if samples is None:
            continue
        if len(samples) > 0:
            return [np.asarray(sample, dtype=float).reshape(-1) for sample in samples]
    return []


def _stack_path(path_samples: Sequence[np.ndarray]) -> np.ndarray | None:
    """把路径采样列表堆叠成 ``(N, dof)`` 矩阵。"""

    # MotionResult 约定失败或无路径时 joint_path 为 None；成功时矩阵列顺序保持 cuMotion
    # C-space 顺序，由 ``CuMotionMotionPlanner.joint_names()`` 说明。
    if not path_samples:
        return None
    return np.vstack(
        [np.asarray(sample, dtype=float).reshape(1, -1) for sample in path_samples]
    )


def _path_length(joint_path: np.ndarray) -> float:
    """计算 C-space 路径长度。

    这是诊断用几何长度，不等于执行时长，也不考虑关节限速或加速度约束。
    """

    if joint_path.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(joint_path, axis=0), axis=1).sum())


def _times_for_joint_path(joint_path: np.ndarray, duration_s: float) -> list[float]:
    """按 C-space 路径长度给 waypoint 分配 ``[0, duration_s]`` 时间戳。"""

    if duration_s < 0:
        raise ValueError("duration_s cannot be negative")
    path = np.asarray(joint_path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 2:
        return [0.0]
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1.0e-12:
        return list(np.linspace(0.0, float(duration_s), path.shape[0]))
    return [float(value) for value in float(duration_s) * cumulative / total]


def _attr(obj, name: str, *, default=None):
    """兼容字段是属性或零参方法两种形式。

    真实 pybind 对象和测试 fake 在暴露结果字段时可能不一致；该 helper 让主逻辑不用关心
    ``results.path`` 与 ``results.path()`` 的区别。
    """

    value = getattr(obj, name, default)
    return value() if callable(value) else value
