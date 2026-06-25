"""cuMotion 逆运动学封装。

本模块把项目的 ``IKRequest`` 转换成 cuMotion 几何 IK 或 collision-free IK 调用，并把结果
归一化为 ``IKResult``。目标位置单位为 m，目标姿态在项目边界使用 ``wxyz`` 四元数；关节
向量始终按 cuMotion C-space 关节名顺序返回。

职责边界:
    * 只封装单点 IK 求解，不把结果映射到 Isaac 完整 DOF。
    * 不推进仿真、不创建控制器；任务层负责把多个 IK 解串成轨迹。
    * collision-free 模式只根据请求构造静态碰撞世界，不维护长期规划状态。

连续性约定:
    实例会在成功求解后把解写回 ``cspace_seeds``，下一次调用默认使用上一帧解作为 seed。
    对 waypoint 序列而言，这比每次从固定 seed 开始更容易得到连续关节解。
"""

from __future__ import annotations

import numpy as np

from manipulation_project.backends.cumotion.collision_world import make_collision_world
from manipulation_project.backends.cumotion.pose_adapter import target_pose
from manipulation_project.planning.requests import IKRequest
from manipulation_project.planning.results import IKResult
from manipulation_project.utils.math_utils import quat_wxyz_to_matrix


def _seed_list(seeds: np.ndarray) -> list[np.ndarray]:
    """把 1D/2D seed 数组规范化为 cuMotion 接受的数组列表。"""

    seed_array = np.asarray(seeds, dtype=float)
    if seed_array.ndim == 1:
        return [seed_array]
    return [np.asarray(seed, dtype=float) for seed in seed_array]


class CuMotionInverseKinematics:
    """使用 cuMotion 求解 TCP 逆运动学。

    ``tcp_frame_name`` 必须是 cuMotion robot description 中存在的 link/frame；自定义 TCP
    通常由 ``tcp_urdf_builder`` 预先写入临时 URDF。实例会复用上一帧成功解作为 seed，
    以提高连续轨迹的求解稳定性。
    """

    def __init__(self, context, *, tcp_frame_name: str | None = None) -> None:
        """创建 IK 配置并填入默认 seed、容差和迭代次数。"""

        if tcp_frame_name is None:
            raise ValueError("tcp_frame_name is required for cuMotion IK")
        self.context = context
        self.cumotion = context.cumotion
        self.kinematics = context.kinematics
        self.tcp_frame_name = str(tcp_frame_name)
        self.config = self.cumotion.IkConfig()
        self.orientation_weight = context.config.orientation_weight
        # IkConfig 可复用，但每次请求会覆盖 warm_start 和容差；构造阶段只写全局默认值。
        if context.config.cspace_seeds is not None:
            self.config.cspace_seeds = _seed_list(np.asarray(context.config.cspace_seeds, dtype=float))
        self.config.ccd_max_iterations = int(context.config.ccd_max_iterations)
        self.config.bfgs_max_iterations = int(context.config.bfgs_max_iterations)
        self.config.ccd_orientation_weight = float(context.config.orientation_weight)
        self.config.bfgs_orientation_weight = float(context.config.orientation_weight)

    def joint_names(self) -> list[str]:
        """返回 IK C-space 关节名。"""

        return self.context.joint_names()

    def frame_names(self) -> list[str]:
        """返回可查询 frame 名。"""

        return self.context.frame_names()

    def solve(self, request: IKRequest) -> IKResult:
        """求解单个 IK 请求。"""

        # 几何 IK 更快，collision-free IK 更保守；请求对象显式选择路径，避免任务层依赖后端默认。
        if request.avoid_collisions:
            return self._solve_collision_free(request)
        return self._solve_geometric(request)

    def _target_pose(self, request: IKRequest):
        """构造 cuMotion 目标位姿对象。"""

        return target_pose(self.cumotion, request.target_position, request.target_orientation)

    def _apply_request_config(self, request: IKRequest) -> None:
        """把单次请求的 warm start 和容差写入可复用 IK config。"""

        # warm_start 是单次请求优先级最高的 seed，通常来自上一 waypoint 解。无姿态目标时把
        # orientation tolerance 放宽并把权重置 0，避免后端试图优化未指定的旋转。
        if request.warm_start is not None:
            self.config.cspace_seeds = [np.asarray(request.warm_start, dtype=float)]
        self.config.position_tolerance = float(request.position_tolerance)
        if request.target_orientation is None:
            self.config.orientation_tolerance = 1.0e9
            self.config.ccd_orientation_weight = 0.0
            self.config.bfgs_orientation_weight = 0.0
        else:
            self.config.orientation_tolerance = float(request.orientation_tolerance)
            self.config.ccd_orientation_weight = float(self.orientation_weight)
            self.config.bfgs_orientation_weight = float(self.orientation_weight)

    def _solve_geometric(self, request: IKRequest) -> IKResult:
        """调用 cuMotion 几何 IK，不显式考虑碰撞。"""

        self._apply_request_config(request)
        frame_name = request.tcp_frame_name or self.tcp_frame_name
        # solve_ik 返回 cuMotion C-space 顺序的关节向量；不要在后端内部重排，调用方会按
        # ``joint_names`` 映射回 articulation。
        result = self.cumotion.solve_ik(self.kinematics, self._target_pose(request), frame_name, self.config)
        q = np.asarray(result.cspace_position, dtype=float)
        orientation_error = max(
            float(result.x_axis_orientation_error),
            float(result.y_axis_orientation_error),
            float(result.z_axis_orientation_error),
        )
        if result.success:
            self.config.cspace_seeds = [q]
        return IKResult(
            joint_positions=q,
            success=bool(result.success),
            position_error=float(result.position_error),
            orientation_error=orientation_error,
            status="SUCCESS" if result.success else "FAILED",
        )

    def _solve_collision_free(self, request: IKRequest) -> IKResult:
        """调用 cuMotion collision-free IK，并用项目碰撞对象构造静态世界。"""

        frame_name = request.tcp_frame_name or self.tcp_frame_name
        # collision-free solver 每次根据请求中的静态对象创建世界视图，适合少量目标点验证；
        # 若未来需要高频规划，可在更高层缓存 collision_world。
        collision_world = make_collision_world(self.context, request.collision_objects)
        config = self.cumotion.create_default_collision_free_ik_solver_config(
            self.context.robot_description,
            frame_name,
            collision_world.world_view,
        )
        solver = self.cumotion.create_collision_free_ik_solver(config)
        translation = self.cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
            np.asarray(request.target_position, dtype=float).reshape(3)
        )
        if request.target_orientation is None:
            orientation = self.cumotion.CollisionFreeIkSolver.OrientationConstraint.none()
        else:
            orientation = self.cumotion.CollisionFreeIkSolver.OrientationConstraint.target(
                self.cumotion.Rotation3.from_matrix(quat_wxyz_to_matrix(request.target_orientation)),
                float(request.orientation_tolerance),
            )
        target = self.cumotion.CollisionFreeIkSolver.TaskSpaceTarget(translation, orientation)
        seeds = []
        if request.warm_start is not None:
            seeds.append(np.asarray(request.warm_start, dtype=float).reshape(-1))
        elif self.context.config.cspace_seeds is not None:
            seeds.extend(_seed_list(np.asarray(self.context.config.cspace_seeds, dtype=float)))
        results = solver.solve(target, seeds)
        status = results.status()
        success_status = self.cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
        positions = list(results.cspace_positions())
        success = status == success_status and bool(positions)
        if success:
            # collision-free 结果本身不一定给出同项目格式的位置误差，因此成功时用 FK 复算一次，
            # 让 IKResult 的 diagnostics 与几何 IK 保持同一单位和含义。
            q = np.asarray(positions[0], dtype=float).reshape(-1)
            self.config.cspace_seeds = [q]
            position_error = self._position_error(q, request, frame_name)
        else:
            q = np.asarray(request.warm_start, dtype=float).reshape(-1) if request.warm_start is not None else np.array([])
            position_error = float("inf")
        return IKResult(
            joint_positions=q,
            success=bool(success),
            position_error=float(position_error),
            orientation_error=None,
            status=str(status),
            num_solutions=len(positions),
        )

    def _position_error(self, joint_positions, request: IKRequest, frame_name: str) -> float:
        """用 FK 复算解的 TCP 位置误差，作为 collision-free IK 诊断值。"""

        pose = self.kinematics.pose(np.asarray(joint_positions, dtype=float).reshape(-1), frame_name)
        achieved = np.asarray(pose.translation, dtype=float).reshape(3)
        target = np.asarray(request.target_position, dtype=float).reshape(3)
        return float(np.linalg.norm(achieved - target))
