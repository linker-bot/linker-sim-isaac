"""Isaac TiledSceneRuntime 的 world/env/base 坐标转换与批量 IK 连接。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.command_utils import (
    _normalize_quaternions,
    _normalize_env_ids,
    _quat_inverse_rows,
    _quat_multiply_rows,
    _repeat_or_validate_rows,
    _selected_action_rows,
    _selected_rows,
)
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.planning.batch_ik import BatchIKResult
from linkerbot_sim.tiled.control.adapter import TiledCommandAdapter
from linkerbot_sim.tiled.control.types import TiledCommandAction
from linkerbot_sim.utils.rotations import rpy_xyz_to_matrix, rpy_xyz_to_quat_wxyz

if TYPE_CHECKING:
    from linkerbot_sim.app.interactive.tiled_scene.runtime.core import (
        TiledSceneRuntime,
    )


@dataclass
class _WorldFrameBatchIKBackend:
    """把 world/env 语义的 tiled IK 请求转换到底层 IK 后端 robot base frame。"""

    solver: object
    root_positions_world: np.ndarray
    root_rotations_world_from_base: np.ndarray
    root_quats_world_wxyz: np.ndarray

    @property
    def tcp_frame_name(self) -> str:
        """返回底层 solver 默认 TCP frame。"""

        return str(getattr(self.solver, "tcp_frame_name", ""))

    def solve(
        self,
        *,
        target_positions: np.ndarray,
        target_orientations_wxyz: np.ndarray | None,
        seeds: np.ndarray,
        tcp_frame_name: str,
    ) -> BatchIKResult:
        """把 world target 转成 base-local 后调用真实 batched IK。"""

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
        """用底层 FK 计算 command positions 对应的 world TCP 位姿。"""

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


def command_adapter(
    runtime: "TiledSceneRuntime",
    robot_name: str,
) -> TiledCommandAdapter:
    """返回机器人对应的 command adapter。"""

    if robot_name not in runtime.command_adapters:
        raise RuntimeError(f"robot {robot_name!r} has no tiled batched IK adapter")
    return runtime.command_adapters[robot_name]


def ik_solver(
    runtime: "TiledSceneRuntime",
    robot_name: str,
) -> _WorldFrameBatchIKBackend:
    """返回机器人对应的 world-frame IK solver。"""

    if robot_name not in runtime.ik_solvers:
        raise RuntimeError(f"robot {robot_name!r} has no tiled batched IK solver")
    return runtime.ik_solvers[robot_name]


def tcp_positions(
    runtime: "TiledSceneRuntime",
    robot_name: str,
) -> np.ndarray:
    """读取缓存的 world TCP positions，缺失时从当前关节状态刷新。"""

    ik_solver(runtime, robot_name)
    if robot_name not in runtime.tcp_positions_world:
        refresh_tcp_state(runtime, robot_name)
    return runtime.tcp_positions_world[robot_name]


def tcp_orientations(
    runtime: "TiledSceneRuntime",
    robot_name: str,
) -> np.ndarray:
    """读取缓存的 world TCP orientations，缺失时从当前关节状态刷新。"""

    ik_solver(runtime, robot_name)
    if robot_name not in runtime.tcp_orientations_wxyz:
        refresh_tcp_state(runtime, robot_name)
    return runtime.tcp_orientations_wxyz[robot_name]


def refresh_tcp_state(
    runtime: "TiledSceneRuntime",
    robot_name: str,
    *,
    env_ids: np.ndarray | None = None,
) -> None:
    """用 cuRobo FK 刷新 selected env 的 world TCP 位姿缓存。"""

    if robot_name not in runtime.ik_solvers:
        return
    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    solver = ik_solver(runtime, robot_name)
    articulation = runtime.scene.articulation_views[robot_name]
    measured_positions = np.asarray(
        articulation.view.get_joint_positions(
            indices=selected,
            joint_indices=articulation.command_joint_indices,
        ),
        dtype=float,
    )
    positions, orientations = solver.command_tcp_world_poses(
        measured_positions,
        env_ids=selected,
    )
    if robot_name not in runtime.tcp_positions_world:
        runtime.tcp_positions_world[robot_name] = np.zeros(
            (runtime.scene.config.num_envs, 3), dtype=float
        )
    if robot_name not in runtime.tcp_orientations_wxyz:
        runtime.tcp_orientations_wxyz[robot_name] = np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
            (runtime.scene.config.num_envs, 1),
        )
    runtime.tcp_positions_world[robot_name][selected, :] = positions
    runtime.tcp_orientations_wxyz[robot_name][selected, :] = orientations


def action_for_robot_reference(
    runtime: "TiledSceneRuntime",
    action: TiledCommandAction,
    *,
    robot_name: str,
    env_ids: np.ndarray,
) -> TiledCommandAction:
    """把绝对 base-local EE target 转换成 world target。"""

    if action.pose_reference_frame != "base" or action.kind not in {
        "ee_pose_target",
        "ee_linear_path",
    }:
        return action
    solver = ik_solver(runtime, robot_name)
    roots = solver.root_positions_world[env_ids]
    rotations = solver.root_rotations_world_from_base[env_ids]
    root_quats = solver.root_quats_world_wxyz[env_ids]
    if action.kind == "ee_pose_target":
        selected_values = _selected_action_rows(
            action.values, env_ids.size, 7, action.kind
        )
        world_positions = roots + np.einsum(
            "nij,nj->ni", rotations, selected_values[:, :3]
        )
        world_orientations = _quat_multiply_rows(
            root_quats,
            _normalize_quaternions(selected_values[:, 3:7]),
        )
        return replace(
            action,
            values=np.concatenate([world_positions, world_orientations], axis=1),
            pose_reference_frame="world",
        )
    world_positions = action.target_position
    if action.target_position is not None:
        selected_positions = _selected_rows(
            action.target_position,
            env_ids.size,
            3,
            "ee_linear_path.target_position",
        )
        world_positions = roots + np.einsum("nij,nj->ni", rotations, selected_positions)
    world_offset = action.target_offset
    if action.target_offset is not None:
        selected_offsets = _selected_rows(
            action.target_offset,
            env_ids.size,
            3,
            "ee_linear_path.target_offset",
        )
        world_offset = np.einsum("nij,nj->ni", rotations, selected_offsets)
    target_orientation = action.target_orientation_wxyz
    if target_orientation is not None:
        selected_orientations = _selected_rows(
            target_orientation,
            env_ids.size,
            4,
            "ee_linear_path.target_orientation_quat_wxyz",
        )
        target_orientation = _quat_multiply_rows(
            root_quats,
            _normalize_quaternions(selected_orientations),
        )
    return replace(
        action,
        target_position=world_positions,
        target_offset=world_offset,
        target_orientation_wxyz=target_orientation,
        pose_reference_frame="world",
    )


def _create_isaac_ik_solvers(
    scene: object,
    robot_names: tuple[str, ...],
    *,
    curobo_profile: object | None = None,
    cache_root: str | None = None,
) -> dict[str, _WorldFrameBatchIKBackend]:
    """为 selected Isaac tiled robots 创建 cuRobo batched IK solver。"""

    profile = (
        load_profile_yaml("curobo", "default")
        if curobo_profile is None
        else curobo_profile
    )
    solvers: dict[str, _WorldFrameBatchIKBackend] = {}
    for name in robot_names:
        robot = scene.robots[name]
        runtime = scene.articulation_views[name]
        robot_config = load_profile_yaml("robot", robot.profile_name)
        from linkerbot_sim.robots.capabilities import (
            PlanningBindingConfig,
            robot_kind_from_profile,
        )

        kind = robot_kind_from_profile(robot_config)
        binding = PlanningBindingConfig.from_profile(robot_config, kind=kind)
        if not binding.enabled:
            continue
        solver = _create_backend_ik_solver(
            robot_config=robot_config,
            command_joint_names=runtime.command_joint_names,
            curobo_profile=profile,
            cache_root=cache_root,
        )
        root_positions, root_rotations, root_quats = _robot_root_world_frames(
            scene,
            name,
        )
        solvers[name] = _WorldFrameBatchIKBackend(
            solver=solver,
            root_positions_world=root_positions,
            root_rotations_world_from_base=root_rotations,
            root_quats_world_wxyz=root_quats,
        )
    return solvers


def _create_backend_ik_solver(
    *,
    robot_config: object,
    command_joint_names: tuple[str, ...],
    curobo_profile: object | None = None,
    cache_root: str | None = None,
) -> object:
    """创建 command-space cuRobo batched IK solver。"""

    from linkerbot_sim.backends.curobo import (
        CuroboBatchIKSolver,
        robot_curobo_config,
    )
    from linkerbot_sim.backends.curobo.context import CuroboContext

    context = CuroboContext(
        robot_curobo_config(
            robot_config,
            curobo_profile=(
                load_profile_yaml("curobo", "default")
                if curobo_profile is None
                else curobo_profile
            ),
        ),
        cache_root=cache_root,
    )
    return CuroboBatchIKSolver(
        context,
        command_joint_names=command_joint_names,
    )


def _robot_root_world_frames(
    scene: object,
    robot_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回每个 env 中机器人 base/root 的 world 位姿。"""

    robot = scene.robots[robot_name]
    origins = np.asarray(scene.env_origins, dtype=float).reshape(
        scene.config.num_envs, 3
    )
    local_poses = tuple(
        scene.config.robot_root_pose_for_env(
            env_id,
            robot_name,
            robot.execution.root_pose,
        )
        for env_id in range(scene.config.num_envs)
    )
    root_position = (
        np.asarray([pose.xyz for pose in local_poses], dtype=float) + origins
    )
    rotations = np.asarray(
        [rpy_xyz_to_matrix(pose.rpy) for pose in local_poses], dtype=float
    ).reshape(scene.config.num_envs, 3, 3)
    quats = np.asarray(
        [rpy_xyz_to_quat_wxyz(pose.rpy) for pose in local_poses], dtype=float
    ).reshape(scene.config.num_envs, 4)
    return root_position, rotations, quats
