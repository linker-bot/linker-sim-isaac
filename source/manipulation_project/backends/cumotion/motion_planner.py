"""cuMotion 路径级运动规划封装。

本模块把项目的 ``MotionRequest`` 转换成 cuMotion ``MotionPlanner`` 调用，并把结果
归一化为 ``MotionResult``。MotionPlanner 负责连续路径级避障搜索；返回的 path 是
cuMotion C-space 关节顺序。若可用，本封装还会用 cuMotion C-space trajectory generator
把搜索路径时间参数化，供上层通过 ``trajectory_adapter`` 采样。

职责边界:
    * 只处理后端 C-space，不映射 Isaac articulation 完整 DOF 或 controller command space。
    * 碰撞世界按请求临时构建，适合静态环境快照。
    * ``mode`` 只解释是否使用环境障碍；更细的 planner 参数仍由 cuMotion 默认配置管理。
"""

from __future__ import annotations

from collections.abc import Sequence

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
    """使用 cuMotion ``MotionPlanner`` 规划路径级运动。"""

    def __init__(
        self,
        context,
        *,
        tcp_frame_name: str | None = None,
        generate_interpolated_path: bool = True,
        generate_trajectory: bool = True,
    ) -> None:
        """创建 motion planner 封装。

        ``tcp_frame_name`` 是任务空间目标使用的工具 frame；关节空间目标也需要一个 frame
        来构造 cuMotion planner config，因此未指定时使用后端默认 TCP/frame。

        ``generate_interpolated_path`` 控制 cuMotion 是否额外返回插值后的路径采样。项目层优先
        消费这个较密的 path；如果关闭或后端未返回，则回退到 planner 的稀疏搜索节点。
        ``generate_trajectory`` 控制是否把 path 再交给 cuMotion trajectory generator 做时间
        参数化。调用方只需要离散路径时可以关闭它，省掉一次后处理。
        """

        self.context = context
        self.cumotion = context.cumotion
        self.tcp_frame_name = str(
            tcp_frame_name
            or context.config.default_tcp_frame
            or context.config.flange_frame
        )
        self.generate_interpolated_path = bool(generate_interpolated_path)
        self.generate_trajectory = bool(generate_trajectory)

    def joint_names(self) -> list[str]:
        """返回 planner C-space 关节名。"""

        return self.context.joint_names()

    def plan(self, request: MotionRequest) -> MotionResult:
        """根据 ``MotionRequest`` 调用 cuMotion MotionPlanner。"""

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
        config = self.cumotion.create_default_motion_planner_config(
            self.context.robot_description,
            frame_name,
            collision_world.world_view,
        )
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
        )

    def _validate_cspace_length(self, values: np.ndarray, label: str) -> None:
        """确保请求关节向量长度匹配 cuMotion C-space。"""

        # cuMotion 只知道 XRDF/URDF 中的主动 C-space 关节，通常不包含灵巧手 mimic follower
        # 或组合 articulation 的全部 DOF。长度不匹配时立即报错，比把错误路径写回控制器安全。
        expected = len(self.joint_names())
        if values.size != expected:
            raise ValueError(f"{label} expected {expected} values, got {values.size}")

    def _motion_result(
        self,
        results,
        *,
        target_type: str,
        frame_name: str,
        num_collision_objects: int,
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
            self._generate_trajectory(joint_path)
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

    def _generate_trajectory(self, joint_path: np.ndarray | None):
        """用 cuMotion C-space trajectory generator 给路径做时间参数化。"""

        # 单点 path 没有时间参数化的意义，直接返回 None。两点及以上时，trajectory generator
        # 会生成 cuMotion Trajectory；项目已有 trajectory_adapter 负责按控制频率采样它。
        if joint_path is None or joint_path.shape[0] < 2:
            return None
        generator = self.cumotion.create_cspace_trajectory_generator(
            self.context.kinematics
        )
        return generator.generate_trajectory(
            [np.asarray(row, dtype=float).reshape(-1) for row in joint_path]
        )


def _collision_objects_for_mode(request: MotionRequest):
    """根据 ``mode`` 决定是否把环境障碍传给 cuMotion world。"""

    # 默认 ``collision_aware`` 会保留请求中的障碍物。这里仅用 mode 控制“环境障碍”是否传入
    # World；不是在这里开关 cuMotion 的所有碰撞逻辑。
    mode = str(request.mode or "").lower()
    if mode in _COLLISION_DISABLED_MODES:
        return ()
    return tuple(request.collision_objects)


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
    """计算 C-space 路径长度。"""

    if joint_path.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(joint_path, axis=0), axis=1).sum())


def _attr(obj, name: str, *, default=None):
    """兼容字段是属性或零参方法两种形式。"""

    value = getattr(obj, name, default)
    return value() if callable(value) else value
