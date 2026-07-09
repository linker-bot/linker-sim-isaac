"""cuMotion 逆运动学封装。

本模块把项目的 ``IKRequest`` 转换成 cuMotion 几何 IK 或 collision-free IK 调用，并把结果
归一化为 ``IKResult``。目标位置单位为 m，目标姿态在项目边界使用 ``wxyz`` 四元数；关节
向量始终按 cuMotion C-space 关节名顺序返回。

职责边界:
    * 只封装单点 IK 求解，不把结果映射到 Isaac 完整 DOF。
    * 不推进仿真、不创建控制器；动作脚本层负责把多个 IK 解串成轨迹。
    * collision-free 模式使用 ``CuMotionContext`` 当前管理的环境 world；环境更新由 context
      的 ``sync_collision_world`` 入口完成。

连续性约定:
    实例会在成功求解后把解写回 cuMotion ``IkConfig.cspace_seeds``，下一次调用默认使用上一帧
    解作为 seed。
    对 waypoint 序列而言，这比每次从固定 seed 开始更容易得到连续关节解。
    如果请求、历史成功解和配置都没有提供 seed，则不在项目侧构造 fallback；未提供 seed
    时使用 cuMotion 默认初始化逻辑。
"""

from __future__ import annotations

import numpy as np
from linkerbot_sim.backends.cumotion.context import validate_cumotion_frame
from linkerbot_sim.backends.cumotion.pose_adapter import (
    pose_from_position_quat_wxyz,
    rotation_from_quat_wxyz,
)
from linkerbot_sim.planning.requests import IKRequest
from linkerbot_sim.planning.results import IKResult


def _ik_cspace_seeds_list(seeds: np.ndarray) -> list[np.ndarray]:
    """把 1D/2D seed 数组规范化为 cuMotion 接受的数组列表。"""

    seed_array = np.asarray(seeds, dtype=float)
    if seed_array.ndim == 1:
        return [seed_array]
    return [np.asarray(seed, dtype=float) for seed in seed_array]


class CuMotionInverseKinematics:
    """使用 cuMotion 求解 TCP 逆运动学。

    ``tcp_frame_name`` 必须是 cuMotion robot description 中存在的 link/frame；自定义 TCP
    通常来自 robot YAML 的 ``cumotion.custom_tcps``，并在 ``CuMotionContext`` 创建时写入
    派生 URDF。实例会复用上一帧成功解作为 seed，以提高连续轨迹的求解稳定性。
    """

    def __init__(self, context, *, tcp_frame_name: str | None = None) -> None:
        """创建 IK 配置并填入默认 seed、容差和迭代次数。

        ``tcp_frame_name`` 是后续请求的默认 frame；单次 ``IKRequest`` 仍可显式覆盖，但覆盖值
        也必须存在于当前 cuMotion robot description。
        """

        if tcp_frame_name is None:
            raise ValueError("tcp_frame_name is required for cuMotion IK")
        self.context = context
        self.cumotion = context.cumotion
        self.kinematics = context.kinematics
        self.tcp_frame_name = str(tcp_frame_name)
        self.config = self.cumotion.IkConfig()
        ik_config = context.config.kinematics.ik
        self.orientation_weight = ik_config.orientation_weight
        # IkConfig 可复用；构造阶段只写 YAML 中显式配置的默认 IK seeds。若未配置
        # kinematics.ik.cspace_seeds，则保持 cuMotion IkConfig 的原生默认值；首帧没有 warm start 时
        # 不在项目侧构造 seed，未提供 seed 时使用 cuMotion 默认初始化逻辑。
        if ik_config.cspace_seeds is not None:
            self.config.cspace_seeds = _ik_cspace_seeds_list(
                np.asarray(ik_config.cspace_seeds, dtype=float)
            )
        self.config.ccd_max_iterations = int(ik_config.ccd_max_iterations)
        self.config.bfgs_max_iterations = int(ik_config.bfgs_max_iterations)
        self.config.ccd_orientation_weight = float(ik_config.orientation_weight)
        self.config.bfgs_orientation_weight = float(ik_config.orientation_weight)

    def joint_names(self) -> list[str]:
        """返回 IK 输入 seed 和输出解使用的 C-space 关节名顺序。

        ``warm_start_ik_cspace_seed``、内部 seed 和 ``IKResult.joint_positions`` 都按该顺序
        排列；动作脚本层需要再按名称映射到 Isaac 完整 DOF。
        """

        return self.context.joint_names()

    def frame_names(self) -> list[str]:
        """返回当前机器人描述中 IK 可使用的 frame 名。

        如果自定义 TCP 没有出现在这里，说明 robot YAML/base URDF 没有声明该 frame，或
        ``CuMotionContext`` 尚未用派生 URDF 加载它。
        """

        return self.context.frame_names()

    def solve(self, request: IKRequest) -> IKResult:
        """求解单个 IK 请求。

        请求中 ``avoid_collisions`` 为真时走 collision-free solver，否则走几何 IK。返回的
        ``joint_positions`` 始终保持 C-space 顺序，调用方需要自行映射回完整 DOF。
        """

        # 先做后端无关的请求结构校验，例如目标位置/姿态维度和容差范围。
        request.validate_structure()
        # 再做依赖当前 cuMotion robot description 的检查，例如 TCP frame 是否存在、
        # warm-start IK C-space seed 是否匹配当前 C-space 宽度。
        self._validate_request_model_match(request)
        # 几何 IK 不考虑环境碰撞, collision-free IK 考虑环境碰撞
        if request.avoid_collisions:
            return self._solve_collision_free(request)
        return self._solve_geometric(request)

    def _validate_request_model_match(self, request: IKRequest) -> None:
        """检查与当前 cuMotion 机器人模型相关的请求字段。"""

        # TCP frame 是否存在。
        frame_name = str(request.tcp_frame_name or self.tcp_frame_name)
        validate_cumotion_frame(self.context, frame_name, label="tcp_frame_name")
        # warm-start IK C-space seed 是否匹配当前 C-space 宽度。
        if request.warm_start_ik_cspace_seed is not None:
            size = (
                np.asarray(request.warm_start_ik_cspace_seed, dtype=float)
                .reshape(-1)
                .size
            )
            if size != self.context.expected_cspace_width:
                raise ValueError(
                    "warm_start_ik_cspace_seed expected "
                    f"{self.context.expected_cspace_width} values, got {size}"
                )

    def _apply_request_config(self, request: IKRequest) -> None:
        """把单次请求的 warm start 和容差写入可复用 IK config。"""

        # warm-start IK C-space seed 是单次请求优先级最高的 seed，通常来自上一 waypoint 解。
        # 如果请求没有 seed，则不覆盖 self.config.cspace_seeds：它可能仍保存上一帧成功解或
        # YAML 默认 seed；如果这些来源也没有 seed，则使用 cuMotion 默认初始化逻辑。
        # 无姿态目标时把 orientation tolerance 放宽并把权重置 0，避免后端试图优化未指定的旋转。
        if request.warm_start_ik_cspace_seed is not None:
            self.config.cspace_seeds = [
                np.asarray(request.warm_start_ik_cspace_seed, dtype=float).reshape(-1)
            ]
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
        result = self.cumotion.solve_ik(
            self.kinematics,
            pose_from_position_quat_wxyz(
                self.cumotion, request.target_position, request.target_orientation
            ),
            frame_name,
            self.config,
        )
        q = np.asarray(result.cspace_position, dtype=float)
        # cuMotion 几何 IK 分别返回 TCP 坐标轴 x/y/z 的姿态误差；项目结果只保留一个标量，
        # 因此有姿态目标时取三者最大值作为最保守的 orientation_error。无姿态目标时语义上
        # 不约束旋转，返回 None。
        orientation_error = (
            None
            if request.target_orientation is None
            else max(
                float(result.x_axis_orientation_error),
                float(result.y_axis_orientation_error),
                float(result.z_axis_orientation_error),
            )
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
        """调用 cuMotion collision-free IK，并使用 context 管理的环境 world view。"""

        frame_name = request.tcp_frame_name or self.tcp_frame_name
        # Request 只表达“是否避障”，不携带障碍物数据；当前环境由 CuMotionContext 统一维护。
        # 动作脚本层在环境变化时调用 context.sync_collision_world(...)，这里直接读取最新 world view。
        collision_world = self.context.collision_world()
        config = self.cumotion.create_default_collision_free_ik_solver_config(
            self.context.robot_description,
            frame_name,
            collision_world.world_view,
        )
        _apply_config_params(
            config,
            self.context.config.kinematics.ik.collision_free_params,
            self.cumotion.CollisionFreeIkSolverConfig.ParamValue,
        )
        solver = self.cumotion.create_collision_free_ik_solver(config)
        translation = self.cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
            np.asarray(request.target_position, dtype=float).reshape(3),
            float(request.position_tolerance),
        )
        if request.target_orientation is None:
            orientation = (
                self.cumotion.CollisionFreeIkSolver.OrientationConstraint.none()
            )
        else:
            orientation = (
                self.cumotion.CollisionFreeIkSolver.OrientationConstraint.target(
                    rotation_from_quat_wxyz(self.cumotion, request.target_orientation),
                    float(request.orientation_tolerance),
                )
            )
        target = self.cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
            translation, orientation
        )
        seeds = []
        if request.warm_start_ik_cspace_seed is not None:
            seeds.append(
                np.asarray(request.warm_start_ik_cspace_seed, dtype=float).reshape(-1)
            )
        elif self.context.config.kinematics.ik.cspace_seeds is not None:
            seeds.extend(
                _ik_cspace_seeds_list(
                    np.asarray(
                        self.context.config.kinematics.ik.cspace_seeds,
                        dtype=float,
                    )
                )
            )
        # cuMotion collision-free IK 明确允许空 seed 列表；当请求和配置都没有 seed 时，
        # 这里传 []。未提供 seed 时使用 cuMotion 默认初始化逻辑。
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
            orientation_error = self._orientation_error(q, request, frame_name)
        else:
            q = (
                np.asarray(request.warm_start_ik_cspace_seed, dtype=float).reshape(-1)
                if request.warm_start_ik_cspace_seed is not None
                else np.array([])
            )
            position_error = float("inf")
            orientation_error = None
        return IKResult(
            joint_positions=q,
            success=bool(success),
            position_error=float(position_error),
            orientation_error=orientation_error,
            status=str(status),
            num_solutions=len(positions),
        )

    def _position_error(
        self, joint_positions, request: IKRequest, frame_name: str
    ) -> float:
        """用 FK 复算解的 TCP 位置误差，作为 collision-free IK 诊断值。"""

        pose = self.kinematics.pose(
            np.asarray(joint_positions, dtype=float).reshape(-1), frame_name
        )
        achieved = np.asarray(pose.translation, dtype=float).reshape(3)
        target = np.asarray(request.target_position, dtype=float).reshape(3)
        return float(np.linalg.norm(achieved - target))

    def _orientation_error(
        self, joint_positions, request: IKRequest, frame_name: str
    ) -> float | None:
        """用 FK 复算 TCP 姿态误差，单位 rad。"""

        if request.target_orientation is None:
            return None
        pose = self.kinematics.pose(
            np.asarray(joint_positions, dtype=float).reshape(-1), frame_name
        )
        target = rotation_from_quat_wxyz(self.cumotion, request.target_orientation)
        return float(self.cumotion.Rotation3.distance(target, pose.rotation))


def _apply_config_params(config, params: dict, param_value_type) -> None:
    """把项目配置中的参数写入 cuMotion config。

    cuMotion 返回 ``False`` 表示参数名或类型没有被接受；这里转成 ``ValueError``，让配置问题
    在第一次创建 solver 时清晰暴露。
    """

    for name, value in params.items():
        ok = config.set_param(str(name), param_value_type(value))
        if ok is False:
            raise ValueError(f"cuMotion config rejected parameter {name!r}")
