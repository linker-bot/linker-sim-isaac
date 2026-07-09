"""Pure Python tiled interactive runtime fake for protocol/unit tests."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.app.interactive.tiled.planning_messages import (
    load_interactive_hand_motion,
    load_interactive_trajectory,
    load_ready_planning_results,
    planning_request_from_message,
    single_trajectory_robot_name,
)
from linkerbot_sim.app.interactive.tiled.command_utils import (
    _action_for_selected_envs,
    _batched_values,
    _jsonable_mapping,
    _normalize_env_ids,
    _normalize_quaternions,
    _positive_decimation,
    _selected_int_rows,
    _selected_rows,
)
from linkerbot_sim.tiled import (
    BatchedIKResult,
    TiledCommandAction,
    TiledCommandAdapter,
    TiledEnvConfig,
    TiledPlannerManager,
    TiledPlanningResult,
    TiledTrajectoryBuffer,
    env_origins,
    env_root_paths,
)
from linkerbot_sim.snapshots.adapters import (
    clone_tiled_env_state,
    get_tiled_snapshot,
    set_tiled_snapshot,
)


DEBUG_TRAJECTORY_ROBOT = "debug"


@dataclass
class DebugBatchedIKSolver:
    """用于单元测试的 deterministic batched IK fake。"""

    fail_env_ids: frozenset[int] = frozenset()

    def solve(
        self,
        *,
        target_positions: np.ndarray,
        target_orientations_wxyz: np.ndarray | None,
        seeds: np.ndarray,
        tcp_frame_name: str,
    ) -> BatchedIKResult:
        """返回和 seeds 同形状的 fake IK 结果。"""

        positions = np.asarray(target_positions, dtype=float)
        seed_array = np.asarray(seeds, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("target_positions must have shape (N, 3)")
        if seed_array.ndim != 2 or seed_array.shape[0] != positions.shape[0]:
            raise ValueError("seeds must have shape (N, C)")

        q = seed_array.copy()
        write_width = min(q.shape[1], positions.shape[1])
        q[:, :write_width] = positions[:, :write_width]
        success = np.ones(positions.shape[0], dtype=bool)
        for env_id in self.fail_env_ids:
            if 0 <= env_id < success.shape[0]:
                success[env_id] = False
        orientation_error = None
        if target_orientations_wxyz is not None:
            orientation_error = np.zeros(positions.shape[0], dtype=float)
        return BatchedIKResult(
            joint_positions=q,
            success=success,
            position_error=np.zeros(positions.shape[0], dtype=float),
            orientation_error=orientation_error,
            status=tuple("SUCCESS" if ok else "DEBUG_FAILED" for ok in success),
        )


@dataclass
class DebugTiledInteractiveRuntime:
    """纯 Python tiled command-step runtime，仅用于协议和单元测试。"""

    env_name: str
    config: TiledEnvConfig
    adapter: TiledCommandAdapter
    physics_dt: float
    current_positions: np.ndarray
    current_tcp_positions: np.ndarray
    current_tcp_orientations_wxyz: np.ndarray
    origins: np.ndarray
    trajectory_buffer: TiledTrajectoryBuffer
    planner_manager: TiledPlannerManager
    episode_steps: np.ndarray
    episode_ids: np.ndarray
    quit_event: threading.Event
    step: int = 0

    @classmethod
    def create(
        cls,
        *,
        env_name: str,
        env_config: Mapping[str, object],
        command_dim: int,
        default_decimation: int,
        tcp_frame_name: str,
        ik_solver: DebugBatchedIKSolver | None,
        planner_workers: int = 2,
        max_pending_requests: int | None = 64,
        max_completed_results: int | None = 256,
    ) -> "DebugTiledInteractiveRuntime":
        """从 env profile 创建内存版 tiled runtime。"""

        tiled_config = TiledEnvConfig.from_env_config(env_config)
        env_section = env_config.get("env", {})
        physics_frequency = 240.0
        if isinstance(env_section, Mapping):
            physics_frequency = float(env_section.get("physics_frequency", physics_frequency))
        origins = env_origins(tiled_config)
        adapter = TiledCommandAdapter(
            num_envs=tiled_config.num_envs,
            command_dim=int(command_dim),
            default_decimation=int(default_decimation),
            tcp_frame_name=tcp_frame_name,
            ik_solver=ik_solver,
        )
        return cls(
            env_name=env_name,
            config=tiled_config,
            adapter=adapter,
            physics_dt=1.0 / max(float(physics_frequency), 1.0e-6),
            current_positions=np.zeros((tiled_config.num_envs, int(command_dim)), dtype=float),
            current_tcp_positions=origins.copy(),
            current_tcp_orientations_wxyz=np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
                (tiled_config.num_envs, 1),
            ),
            origins=origins,
            trajectory_buffer=TiledTrajectoryBuffer(num_envs=tiled_config.num_envs),
            planner_manager=TiledPlannerManager(
                max_workers=int(planner_workers),
                max_pending_requests=max_pending_requests,
                max_completed_results=max_completed_results,
            ),
            episode_steps=np.zeros(tiled_config.num_envs, dtype=int),
            episode_ids=np.zeros(tiled_config.num_envs, dtype=int),
            quit_event=threading.Event(),
        )

    @property
    def time_s(self) -> float:
        """返回按 physics_dt 推导的仿真时间。"""

        return float(self.step) * float(self.physics_dt)

    def status(self) -> dict[str, object]:
        """返回当前 tiled 内存状态快照。"""

        return {
            "event": "status",
            "env": self.env_name,
            "num_envs": self.config.num_envs,
            "command_dim": self.adapter.command_dim,
            "step": self.step,
            "time_s": self.time_s,
            "episode_steps": self.episode_steps.tolist(),
            "episode_ids": self.episode_ids.tolist(),
            "env_roots": list(env_root_paths(self.config)),
            "env_origins": self.origins.tolist(),
            "joint_positions": self.current_positions.tolist(),
            "tcp_positions_world": self.current_tcp_positions.tolist(),
            "tcp_orientations_wxyz": self.current_tcp_orientations_wxyz.tolist(),
        }

    def reset(self, env_ids: np.ndarray | None = None) -> dict[str, object]:
        """重置 selected env 状态并清空 adapter 的 hold target 缓存。"""

        selected = _normalize_env_ids(env_ids, self.config.num_envs)
        self.current_positions[selected, :] = 0.0
        self.current_tcp_positions[selected, :] = self.origins[selected, :]
        self.current_tcp_orientations_wxyz[selected, :] = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=float
        )
        self.episode_steps[selected] = 0
        self.episode_ids[selected] += 1
        self.adapter.reset()
        self.trajectory_buffer.clear(env_ids=selected)
        self.planner_manager.cancel_matching(env_ids=selected)
        return {
            "event": "reset",
            "accepted": True,
            "env_ids": selected.tolist(),
            "step": self.step,
            "time_s": self.time_s,
            "episode_steps": self.episode_steps.tolist(),
            "episode_ids": self.episode_ids.tolist(),
        }

    def step_action(
        self,
        action: TiledCommandAction,
        *,
        env_ids: np.ndarray | None = None,
        robot_names: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """执行一个同步 tiled command step。"""

        selected = _normalize_env_ids(env_ids, self.config.num_envs)
        action = _action_for_selected_envs(
            action=action,
            env_ids=selected,
            current_positions=self.current_positions,
            current_tcp_positions=self.current_tcp_positions,
            current_tcp_orientations_wxyz=self.current_tcp_orientations_wxyz,
            env_origins=self.origins,
        )
        start = self.current_positions.copy()
        target = self.adapter.action_to_joint_target(
            action,
            current_positions=self.current_positions,
            current_tcp_positions=self.current_tcp_positions,
            current_tcp_orientations_wxyz=self.current_tcp_orientations_wxyz,
            env_origins=self.origins,
        )
        target_positions = start.copy()
        target_positions[selected, :] = target.joint_positions[selected, :]
        trajectory = self.adapter.interpolate_to(target_positions, start=start, action=action)
        for tick_target in trajectory:
            self.current_positions = tick_target.copy()
            self.step += 1
            self.episode_steps[:] += 1
        self._update_debug_tcp_state(action, env_ids=selected)
        return {
            "event": "step",
            "accepted": True,
            "kind": action.kind,
            "env_ids": selected.tolist(),
            "ticks": int(trajectory.shape[0]),
            "step": self.step,
            "time_s": self.time_s,
            "episode_steps": self.episode_steps.tolist(),
            "joint_positions": self.current_positions.tolist(),
            "info": _jsonable_mapping(target.info),
        }

    def get_state(
        self,
        *,
        env_ids: np.ndarray | None = None,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """返回可裁剪的 JSON-compatible debug state。"""

        selected = _normalize_env_ids(env_ids, self.config.num_envs)
        payload = {
            "joint_positions": self.current_positions[selected].tolist(),
            "tcp_positions_world": self.current_tcp_positions[selected].tolist(),
            "tcp_orientations_wxyz": self.current_tcp_orientations_wxyz[selected].tolist(),
            "episode_steps": self.episode_steps[selected].tolist(),
            "episode_ids": self.episode_ids[selected].tolist(),
        }
        if fields is not None:
            payload = {key: value for key, value in payload.items() if key in fields}
        return {
            "event": "state",
            "accepted": True,
            "env_ids": selected.tolist(),
            "step": self.step,
            "time_s": self.time_s,
            "state": payload,
        }

    def set_state(
        self,
        state: Mapping[str, object],
        *,
        env_ids: np.ndarray | None = None,
    ) -> dict[str, object]:
        """写回 selected env 的 debug state，并清空 hold target 缓存。"""

        selected = _normalize_env_ids(env_ids, self.config.num_envs)
        if "joint_positions" in state:
            self.current_positions[selected, :] = _selected_rows(
                state["joint_positions"], selected.size, self.adapter.command_dim, "joint_positions"
            )
        if "tcp_positions_world" in state:
            self.current_tcp_positions[selected, :] = _selected_rows(
                state["tcp_positions_world"], selected.size, 3, "tcp_positions_world"
            )
        if "tcp_orientations_wxyz" in state:
            self.current_tcp_orientations_wxyz[selected, :] = _normalize_quaternions(
                _selected_rows(
                    state["tcp_orientations_wxyz"], selected.size, 4, "tcp_orientations_wxyz"
                )
            )
        if "episode_steps" in state:
            self.episode_steps[selected] = _selected_int_rows(
                state["episode_steps"], selected.size, "episode_steps"
            )
        if "episode_ids" in state:
            self.episode_ids[selected] = _selected_int_rows(
                state["episode_ids"], selected.size, "episode_ids"
            )
        self.adapter.reset()
        self.trajectory_buffer.clear(env_ids=selected)
        self.planner_manager.cancel_matching(env_ids=selected)
        return {
            "event": "set_state",
            "accepted": True,
            "env_ids": selected.tolist(),
            "step": self.step,
            "time_s": self.time_s,
        }

    def get_snapshot(self, *, env_id: int) -> dict[str, object]:
        """返回单个 debug env 的 runtime-neutral snapshot。

        fake runtime 使用真实 adapter 读快照，保证协议测试覆盖的 JSON 结构与 Isaac runtime
        保持一致。
        """

        snapshot = get_tiled_snapshot(self, env_id=int(env_id))
        return {
            "event": "snapshot",
            "accepted": True,
            "backend": "debug",
            "env_id": int(env_id),
            "step": self.step,
            "time_s": self.time_s,
            "snapshot": snapshot.as_dict(),
        }

    def set_snapshot(
        self,
        snapshot: Mapping[str, object],
        *,
        env_ids: np.ndarray,
        robot_map: Mapping[str, str] | None = None,
        strict: bool = True,
    ) -> dict[str, object]:
        """把 runtime-neutral snapshot 写回 selected debug env。

        这里不模拟 PhysX 对象，但仍走兼容性检查和多 env 广播逻辑，用来验证 tiled
        set_snapshot 协议行为。
        """

        result = set_tiled_snapshot(
            self,
            snapshot,
            env_ids=env_ids,
            robot_map=robot_map,
            strict=bool(strict),
        )
        return {
            **result.as_dict(),
            "backend": "debug",
            "step": self.step,
            "time_s": self.time_s,
        }

    def clone_state(
        self,
        *,
        source_env_id: int,
        target_env_ids: np.ndarray,
        strict: bool = True,
    ) -> dict[str, object]:
        """把一个 debug source env 克隆到多个 target env。

        fake clone 与真实 tiled 一样复用 get_snapshot + set_snapshot，避免测试出现专属路径。
        """

        result = clone_tiled_env_state(
            self,
            source_env_id=int(source_env_id),
            target_env_ids=target_env_ids,
            strict=bool(strict),
        )
        return {
            **result.as_dict(),
            "event": "state_cloned",
            "backend": "debug",
            "source_env_id": int(source_env_id),
            "target_env_ids": [int(env_id) for env_id in target_env_ids],
            "step": self.step,
            "time_s": self.time_s,
        }

    def load_trajectory(
        self,
        trajectory: Mapping[str, object],
        *,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """把已规划好的关节轨迹载入 debug runtime 的回放缓冲。"""

        selected = _normalize_env_ids(env_ids, self.config.num_envs)
        robot = robot_name or DEBUG_TRAJECTORY_ROBOT
        loaded = load_interactive_trajectory(
            self.trajectory_buffer,
            trajectory,
            env_ids=selected,
            robot_name=robot,
            current_positions=self.current_positions[selected],
            command_joint_names=tuple(f"joint_{index}" for index in range(self.adapter.command_dim)),
        )
        return {
            "event": "trajectory_loaded",
            "accepted": True,
            "backend": "debug",
            **loaded,
            "step": self.step,
            "time_s": self.time_s,
        }

    def step_trajectory(
        self,
        *,
        env_ids: np.ndarray | None = None,
        robot_names: tuple[str, ...] | None = None,
        decimation: int | None = None,
    ) -> dict[str, object]:
        """按 physics tick 回放一段 ready trajectory。"""

        planner_results, planner_loaded = self._collect_planner_results()
        selected = _normalize_env_ids(env_ids, self.config.num_envs)
        robot = single_trajectory_robot_name(robot_names, default=DEBUG_TRAJECTORY_ROBOT)
        ticks = _positive_decimation(decimation, default_decimation=self.adapter.default_decimation)
        result = None
        for _ in range(ticks):
            result = self.trajectory_buffer.step(
                robot_name=robot,
                current_positions=self.current_positions,
                dt_s=self.physics_dt,
                env_ids=selected,
            )
            self.current_positions = result.joint_positions.copy()
            self.step += 1
            self.episode_steps[:] += 1
        assert result is not None
        return {
            "event": "trajectory_step",
            "accepted": True,
            "backend": "debug",
            "ticks": int(ticks),
            "step": self.step,
            "time_s": self.time_s,
            "episode_steps": self.episode_steps.tolist(),
            "joint_positions": self.current_positions.tolist(),
            "trajectory": result.to_json(),
            "planner_ready": [item.to_json() for item in planner_results],
            "planner_loaded": planner_loaded,
        }

    def submit_plan(
        self,
        message: Mapping[str, object],
        *,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """提交 debug 关节空间规划请求。"""

        selected = _normalize_env_ids(env_ids, self.config.num_envs)
        robot = robot_name or DEBUG_TRAJECTORY_ROBOT
        request = planning_request_from_message(
            message,
            robot_name=robot,
            env_ids=selected,
            current_positions=self.current_positions[selected],
            command_joint_names=tuple(f"joint_{index}" for index in range(self.adapter.command_dim)),
            default_sample_dt_s=self.physics_dt,
            default_tcp_frame_name=self.adapter.tcp_frame_name,
        )
        request_id = self.planner_manager.submit(request)
        return {
            "event": "plan_submitted",
            "accepted": True,
            "backend": "debug",
            "request_id": request_id,
            "robot": robot,
            "env_ids": selected.tolist(),
            "duration_s": float(request.duration_s),
            "sample_dt_s": float(request.sample_dt_s),
            "segments": [segment.kind for segment in request.segments],
            "load_on_success": bool(request.load_on_success),
        }

    def submit_hand_motion(
        self,
        message: Mapping[str, object],
        *,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """提交 debug hand-only motion，并默认追加到 trajectory queue。"""

        selected = _normalize_env_ids(env_ids, self.config.num_envs)
        message_type = str(message.get("type", "hand"))
        loaded: list[dict[str, object]] = []
        if message_type == "dual_hand":
            for side in ("left", "right"):
                child = message.get(side)
                if child is None:
                    continue
                if not isinstance(child, Mapping):
                    raise ValueError(f"dual_hand.{side} must be a JSON object")
                loaded.append(
                    self._load_hand_payload(
                        child,
                        parent=message,
                        robot_name=side,
                        env_ids=selected,
                    )
                )
            if not loaded:
                raise ValueError("dual_hand requires left or right payload")
        else:
            loaded.append(
                self._load_hand_payload(
                    message,
                    parent=None,
                    robot_name=robot_name or DEBUG_TRAJECTORY_ROBOT,
                    env_ids=selected,
                )
            )
        return {
            "event": "hand_motion_queued",
            "accepted": True,
            "backend": "debug",
            "motions": loaded,
            "step": self.step,
            "time_s": self.time_s,
        }

    def planner_status(self, *, wait_timeout_s: float = 0.0) -> dict[str, object]:
        """收集 ready result，成功结果自动载入 trajectory buffer。"""

        ready, loaded = self._collect_planner_results(timeout_s=wait_timeout_s)
        return {
            "event": "planner_status",
            "accepted": True,
            "backend": "debug",
            "ready": [item.to_json() for item in ready],
            "loaded": loaded,
            "planner": self.planner_manager.status(),
        }

    def _load_hand_payload(
        self,
        payload: Mapping[str, object],
        *,
        parent: Mapping[str, object] | None,
        robot_name: str,
        env_ids: np.ndarray,
    ) -> dict[str, object]:
        """把单个 hand payload 写入 debug trajectory buffer。"""

        return load_interactive_hand_motion(
            self.trajectory_buffer,
            payload,
            env_ids=env_ids,
            robot_name=robot_name,
            current_positions=self.current_positions[env_ids],
            command_joint_names=tuple(
                f"joint_{index}" for index in range(self.adapter.command_dim)
            ),
            parent_payload=parent,
        )

    def cancel_plan(
        self,
        *,
        request_id: str | None = None,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """取消 debug 后台规划请求。"""

        if request_id is not None:
            result: object = self.planner_manager.cancel(request_id)
        else:
            result = self.planner_manager.cancel_matching(robot_name=robot_name, env_ids=env_ids)
        return {
            "event": "plan_cancelled",
            "accepted": True,
            "backend": "debug",
            "result": result,
        }

    def clear_completed(
        self,
        *,
        request_ids: str | tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """清理 debug planner completed result 缓存。"""

        return {
            "event": "completed_cleared",
            "accepted": True,
            "backend": "debug",
            "result": self.planner_manager.clear_completed(request_ids),
        }

    def trajectory_status(
        self,
        *,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """返回 debug 轨迹缓冲状态。"""

        return {
            "event": "trajectory_status",
            "accepted": True,
            "backend": "debug",
            "step": self.step,
            "time_s": self.time_s,
            "trajectory": self.trajectory_buffer.status(robot_name=robot_name, env_ids=env_ids),
        }

    def clear_trajectory(
        self,
        *,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """清理 debug 轨迹缓冲。"""

        selected = None if env_ids is None else _normalize_env_ids(env_ids, self.config.num_envs)
        cleared = self.trajectory_buffer.clear(robot_name=robot_name, env_ids=selected)
        return {
            "event": "trajectory_cleared",
            "accepted": True,
            "backend": "debug",
            "cleared": cleared,
            "step": self.step,
            "time_s": self.time_s,
        }

    def _update_debug_tcp_state(
        self,
        action: TiledCommandAction,
        *,
        env_ids: np.ndarray,
    ) -> None:
        """维护 debug TCP 状态，供下一条 ee_delta_* action 使用。"""

        if action.kind == "ee_pose_target":
            values = _batched_values(action.values, self.config.num_envs, 7, "values")
            positions = values[:, :3]
            if action.pose_reference_frame == "env":
                positions = positions + self.origins
            self.current_tcp_positions[env_ids, :] = positions[env_ids, :]
            self.current_tcp_orientations_wxyz[env_ids, :] = _normalize_quaternions(
                values[:, 3:7]
            )[env_ids, :]
            return
        if action.kind == "ee_delta_pos":
            delta = _batched_values(action.values, self.config.num_envs, 3, "values")
            self.current_tcp_positions[env_ids, :] = (
                self.current_tcp_positions[env_ids, :] + delta[env_ids, :]
            )
            return
        if action.kind == "ee_delta_pose":
            values = np.asarray(action.values, dtype=float)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.shape[0] == 1 and self.config.num_envs != 1:
                values = np.repeat(values, self.config.num_envs, axis=0)
            self.current_tcp_positions[env_ids, :] = (
                self.current_tcp_positions[env_ids, :] + values[env_ids, :3]
            )
            if values.shape[1] == 7:
                self.current_tcp_orientations_wxyz[env_ids, :] = _normalize_quaternions(
                    values[:, 3:7]
                )[env_ids, :]

    def _collect_planner_results(
        self,
        *,
        timeout_s: float = 0.0,
    ) -> tuple[tuple[TiledPlanningResult, ...], list[dict[str, object]]]:
        """收集 planner results，并把成功结果载入 trajectory buffer。"""

        results = self.planner_manager.collect_ready(timeout_s=timeout_s)
        loaded = load_ready_planning_results(self.trajectory_buffer, results)
        return results, loaded

    def close(self) -> None:
        """关闭 debug runtime 的后台 planner 线程池。"""

        self.planner_manager.shutdown()
