"""cuMotion 逆运动学封装。"""

from __future__ import annotations

import numpy as np

from manipulation_project.backends.cumotion.collision_world import make_collision_world
from manipulation_project.backends.cumotion.pose_adapter import target_pose
from manipulation_project.planning.requests import IKRequest
from manipulation_project.planning.results import IKResult
from manipulation_project.utils.math_utils import quat_wxyz_to_matrix


def _seed_list(seeds: np.ndarray) -> list[np.ndarray]:
    seed_array = np.asarray(seeds, dtype=float)
    if seed_array.ndim == 1:
        return [seed_array]
    return [np.asarray(seed, dtype=float) for seed in seed_array]


class CuMotionInverseKinematics:
    """使用 cuMotion 求解 TCP 逆运动学。"""

    backend = "cumotion"

    def __init__(self, context, *, tcp_frame_name: str | None = None) -> None:
        if tcp_frame_name is None:
            raise ValueError("tcp_frame_name is required for cuMotion IK")
        self.context = context
        self.cumotion = context.cumotion
        self.kinematics = context.kinematics
        self.tcp_frame_name = str(tcp_frame_name)
        self.config = self.cumotion.IkConfig()
        self.orientation_weight = context.config.orientation_weight
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

        if request.avoid_collisions:
            return self._solve_collision_free(request)
        return self._solve_geometric(request)

    def _target_pose(self, request: IKRequest):
        return target_pose(self.cumotion, request.target_position, request.target_orientation)

    def _apply_request_config(self, request: IKRequest) -> None:
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
        self._apply_request_config(request)
        frame_name = request.tcp_frame_name or self.tcp_frame_name
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
        frame_name = request.tcp_frame_name or self.tcp_frame_name
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
        pose = self.kinematics.pose(np.asarray(joint_positions, dtype=float).reshape(-1), frame_name)
        achieved = np.asarray(pose.translation, dtype=float).reshape(3)
        target = np.asarray(request.target_position, dtype=float).reshape(3)
        return float(np.linalg.norm(achieved - target))
