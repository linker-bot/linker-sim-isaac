"""World-frame IK wrappers for Isaac tiled interactive runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linkerbot_sim.app.interactive.tiled.command_utils import (
    _normalize_quaternions,
    _quat_inverse_rows,
    _quat_multiply_rows,
    _repeat_or_validate_rows,
)
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.tiled import BatchedIKResult
from linkerbot_sim.utils.rotations import rpy_xyz_to_matrix, rpy_xyz_to_quat_wxyz


@dataclass
class _WorldFrameBatchedIKSolver:
    """把 world/env 语义的 tiled IK 请求转换到 cuMotion robot base frame。"""

    solver: object
    root_positions_world: np.ndarray
    root_rotations_world_from_base: np.ndarray
    root_quats_world_wxyz: np.ndarray

    @property
    def tcp_frame_name(self) -> str:
        """返回底层 cuMotion solver 默认 TCP frame。"""

        return str(getattr(self.solver, "tcp_frame_name", ""))

    def solve(
        self,
        *,
        target_positions: np.ndarray,
        target_orientations_wxyz: np.ndarray | None,
        seeds: np.ndarray,
        tcp_frame_name: str,
    ) -> BatchedIKResult:
        """把 world target 转成 base-local 后调用真实 batched cuMotion IK。"""

        positions = np.asarray(target_positions, dtype=float).reshape(-1, 3)
        local_positions = self.world_positions_to_base(positions)
        local_orientations = None
        if target_orientations_wxyz is not None:
            local_orientations = self.world_orientations_to_base(
                np.asarray(target_orientations_wxyz, dtype=float).reshape(-1, 4)
            )
        return self.solver.solve(
            target_positions=local_positions,
            target_orientations_wxyz=local_orientations,
            seeds=seeds,
            tcp_frame_name=tcp_frame_name,
        )

    def command_tcp_world_poses(
        self,
        command_positions: np.ndarray,
        *,
        tcp_frame_name: str | None = None,
        env_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """用 cuMotion FK 计算 command positions 对应的 world TCP 位姿。"""

        local_positions, local_orientations = self.solver.compute_tcp_poses(
            command_positions,
            tcp_frame_name=tcp_frame_name or self.tcp_frame_name,
        )
        if env_ids is not None:
            selected = np.asarray(env_ids, dtype=int).reshape(-1)
            return (
                self._base_positions_to_world_for_envs(local_positions, selected),
                self._base_orientations_to_world_for_envs(local_orientations, selected),
            )
        return (
            self.base_positions_to_world(local_positions),
            self.base_orientations_to_world(local_orientations),
        )

    def base_positions_to_world(self, positions: np.ndarray) -> np.ndarray:
        """把 base-local 位置转成 world。"""

        local = np.asarray(positions, dtype=float).reshape(-1, 3)
        rotations = _repeat_or_validate_rows(
            self.root_rotations_world_from_base,
            local.shape[0],
            (3, 3),
            "root_rotations_world_from_base",
        )
        roots = _repeat_or_validate_rows(
            self.root_positions_world,
            local.shape[0],
            (3,),
            "root_positions_world",
        )
        return roots + np.einsum("nij,nj->ni", rotations, local)

    def world_positions_to_base(self, positions: np.ndarray) -> np.ndarray:
        """把 world 位置转成 robot base-local。"""

        world = np.asarray(positions, dtype=float).reshape(-1, 3)
        rotations = _repeat_or_validate_rows(
            self.root_rotations_world_from_base,
            world.shape[0],
            (3, 3),
            "root_rotations_world_from_base",
        )
        roots = _repeat_or_validate_rows(
            self.root_positions_world,
            world.shape[0],
            (3,),
            "root_positions_world",
        )
        return np.einsum("nji,nj->ni", rotations, world - roots)

    def base_orientations_to_world(self, orientations_wxyz: np.ndarray) -> np.ndarray:
        """把 base-local 姿态转成 world 姿态。"""

        local = _normalize_quaternions(
            np.asarray(orientations_wxyz, dtype=float).reshape(-1, 4)
        )
        roots = _repeat_or_validate_rows(
            self.root_quats_world_wxyz,
            local.shape[0],
            (4,),
            "root_quats_world_wxyz",
        )
        return _quat_multiply_rows(roots, local)

    def world_orientations_to_base(self, orientations_wxyz: np.ndarray) -> np.ndarray:
        """把 world 姿态转成 base-local 姿态。"""

        world = _normalize_quaternions(
            np.asarray(orientations_wxyz, dtype=float).reshape(-1, 4)
        )
        roots = _repeat_or_validate_rows(
            self.root_quats_world_wxyz,
            world.shape[0],
            (4,),
            "root_quats_world_wxyz",
        )
        return _quat_multiply_rows(_quat_inverse_rows(roots), world)

    def _base_positions_to_world_for_envs(
        self,
        positions: np.ndarray,
        env_ids: np.ndarray,
    ) -> np.ndarray:
        """把 selected env 的 base-local 位置转成 world。"""

        local = np.asarray(positions, dtype=float).reshape(-1, 3)
        selected = np.asarray(env_ids, dtype=int).reshape(-1)
        if selected.size != local.shape[0]:
            raise ValueError("env_ids length must match position rows")
        rotations = self.root_rotations_world_from_base[selected]
        roots = self.root_positions_world[selected]
        return roots + np.einsum("nij,nj->ni", rotations, local)

    def _base_orientations_to_world_for_envs(
        self,
        orientations_wxyz: np.ndarray,
        env_ids: np.ndarray,
    ) -> np.ndarray:
        """把 selected env 的 base-local 姿态转成 world。"""

        local = _normalize_quaternions(
            np.asarray(orientations_wxyz, dtype=float).reshape(-1, 4)
        )
        selected = np.asarray(env_ids, dtype=int).reshape(-1)
        if selected.size != local.shape[0]:
            raise ValueError("env_ids length must match orientation rows")
        roots = self.root_quats_world_wxyz[selected]
        return _quat_multiply_rows(roots, local)


def _create_isaac_ik_solvers(
    scene: object,
    robot_names: tuple[str, ...],
) -> dict[str, _WorldFrameBatchedIKSolver]:
    """为 selected Isaac tiled robots 创建真正 batched cuMotion IK solver。"""

    from linkerbot_sim.backends.cumotion import (
        BatchedCuMotionIKSolver,
        CuMotionConfig,
        CuMotionContext,
    )

    solvers: dict[str, _WorldFrameBatchedIKSolver] = {}
    for name in robot_names:
        robot = scene.robots[name]
        runtime = scene.articulation_views[name]
        robot_config = load_profile_yaml("robot", robot.profile_name)
        context = CuMotionContext(CuMotionConfig.from_mapping(robot_config))
        solver = BatchedCuMotionIKSolver(
            context,
            command_joint_names=runtime.command_joint_names,
        )
        root_positions, root_rotations, root_quats = _robot_root_world_frames(
            scene,
            name,
        )
        solvers[name] = _WorldFrameBatchedIKSolver(
            solver=solver,
            root_positions_world=root_positions,
            root_rotations_world_from_base=root_rotations,
            root_quats_world_wxyz=root_quats,
        )
    return solvers


def _robot_root_world_frames(
    scene: object,
    robot_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回每个 env 中机器人 base/root 的 world 位姿。"""

    robot = scene.robots[robot_name]
    local_pose = robot.execution.root_pose
    origins = np.asarray(scene.env_origins, dtype=float).reshape(scene.config.num_envs, 3)
    root_position = np.asarray(local_pose.xyz, dtype=float).reshape(1, 3) + origins
    rotation = rpy_xyz_to_matrix(local_pose.rpy)
    quat = rpy_xyz_to_quat_wxyz(local_pose.rpy)
    rotations = np.repeat(rotation.reshape(1, 3, 3), scene.config.num_envs, axis=0)
    quats = np.repeat(quat.reshape(1, 4), scene.config.num_envs, axis=0)
    return root_position, rotations, quats
