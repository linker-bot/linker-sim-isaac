"""Isaac/PhysX tiled interactive runtime."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from linkerbot_sim.app.interactive.tiled.isaac_ik_solver import (
    _WorldFrameBatchedIKSolver,
    _create_isaac_ik_solvers,
)
from linkerbot_sim.app.interactive.tiled.object_states import _read_tiled_object_states
from linkerbot_sim.app.interactive.tiled.object_states import (
    TiledDynamicChainObjectPoseView,
    _capture_tiled_object_pose_snapshot,
    _restore_tiled_object_pose_snapshot,
)
from linkerbot_sim.app.interactive.tiled.planning_messages import (
    load_interactive_hand_motion,
    load_interactive_trajectory,
    load_ready_planning_results,
    planning_request_from_message,
)
from linkerbot_sim.app.interactive.tiled.protocol import RobotSelection, _selected_robot_names
from linkerbot_sim.app.interactive.tiled.command_utils import (
    _action_decimation,
    _action_for_selected_envs,
    _action_width,
    _apply_joint_targets,
    _filter_isaac_state_fields,
    _jsonable_mapping,
    _normalize_env_ids,
    _normalize_quaternions,
    _positive_decimation,
    _quat_multiply_rows,
    _selected_action_rows,
    _selected_int_rows,
    _selected_rows,
)
from linkerbot_sim.tiled import (
    TiledCommandAction,
    TiledCommandAdapter,
    TiledEnvConfig,
    TiledPlannerManager,
    TiledPlanningResult,
    TiledTrajectoryBuffer,
)
from linkerbot_sim.snapshots.adapters import (
    clone_tiled_env_state,
    get_tiled_snapshot,
    set_tiled_snapshot,
)
from linkerbot_sim.sensors.camera_observer import CameraOutputHandle
from linkerbot_sim.sensors.camera_runtime import SensorCameraRuntime


@dataclass
class IsaacTiledInteractiveRuntime:
    """真实 Isaac tiled command-step runtime。"""

    env_name: str
    env_config: Mapping[str, object]
    session: object
    scene: object
    render: bool
    default_decimation: int
    robot_names: tuple[str, ...]
    episode_steps: np.ndarray
    episode_ids: np.ndarray
    initial_joint_positions: dict[str, np.ndarray]
    initial_joint_velocities: dict[str, np.ndarray]
    target_positions: dict[str, np.ndarray]
    initial_object_states: dict[str, dict[str, object]]
    command_adapters: dict[str, TiledCommandAdapter]
    ik_solvers: dict[str, _WorldFrameBatchedIKSolver]
    tcp_positions_world: dict[str, np.ndarray]
    tcp_orientations_wxyz: dict[str, np.ndarray]
    trajectory_buffer: TiledTrajectoryBuffer
    planner_manager: TiledPlannerManager
    sensor_cameras: tuple[SensorCameraRuntime, ...]
    camera_output: CameraOutputHandle | None
    quit_event: threading.Event
    object_pose_views: dict[str, object] = field(default_factory=dict)
    step: int = 0
    _closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        env_name: str,
        env_config: Mapping[str, object],
        gui: bool,
        default_decimation: int,
        planner_backend: str = "linear",
        planner_workers: int = 2,
        max_pending_requests: int | None = 64,
        max_completed_results: int | None = 256,
    ) -> "IsaacTiledInteractiveRuntime":
        """创建真实 Isaac tiled scene，并完成第一次 world reset/finalize。"""

        from linkerbot_sim.app.runtime.settings import EnvRuntimeSettings
        from linkerbot_sim.app.runtime.simulation_session import create_simulation_session
        from linkerbot_sim.configs.profiles import load_default_controller_profiles
        from linkerbot_sim.sensors.camera_observer import start_camera_output
        from linkerbot_sim.sensors.camera_runtime import (
            create_sensor_camera_runtimes,
            initialize_sensor_camera_runtimes,
        )
        from linkerbot_sim.tiled.cameras import tiled_sensor_camera_settings
        from linkerbot_sim.tiled.scene import build_isaac_tiled_scene, finalize_tiled_articulation_views
        from linkerbot_sim.tiled.scene.utils import _print_status
        from linkerbot_sim.utils.paths import repo_path

        tiled_config = TiledEnvConfig.from_env_config(env_config)
        runtime_settings = EnvRuntimeSettings.from_env_config(env_config)
        session = create_simulation_session(gui=bool(gui), settings=runtime_settings)
        scene = build_isaac_tiled_scene(
            world=session.world,
            stage=session.stage,
            env_config=env_config,
            tiled_config=tiled_config,
            controller_profiles=load_default_controller_profiles(),
            status_prefix="TILED_INTERACTIVE",
        )
        sensor_settings = tiled_sensor_camera_settings(
            runtime_settings.sensors,
            tiled_config=scene.config,
        )
        sensor_cameras = create_sensor_camera_runtimes(
            stage=session.stage,
            sensors=sensor_settings,
        )
        camera_output = start_camera_output(
            sensor_cameras,
            path_resolver=repo_path,
        )
        session.world.reset()
        session.world.get_physics_context().set_gravity(runtime_settings.gravity_z)
        scene = finalize_tiled_articulation_views(scene)
        initialize_sensor_camera_runtimes(sensor_cameras)
        for sensor_camera in sensor_cameras:
            _print_status(
                "TILED_INTERACTIVE",
                "SENSOR_CAMERA "
                f"name={sensor_camera.name} prim_path={sensor_camera.prim_path} "
                f"modalities={','.join(sensor_camera.settings.modalities)}",
            )
        if gui:
            _refresh_gui_view_after_scene_build(session, runtime_settings)
        selected = _selected_robot_names(scene.articulation_views, None)
        initial_positions = {
            name: np.asarray(scene.articulation_views[name].view.get_joint_positions())
            for name in selected
        }
        initial_velocities = {
            name: np.asarray(scene.articulation_views[name].view.get_joint_velocities())
            for name in selected
        }
        targets = {
            name: np.asarray(
                scene.articulation_views[name].view.get_joint_positions(
                    joint_indices=scene.articulation_views[name].command_joint_indices
                ),
                dtype=float,
            )
            for name in selected
        }
        ik_solvers = _create_isaac_ik_solvers(scene, selected)
        command_adapters = {
            name: TiledCommandAdapter(
                num_envs=scene.config.num_envs,
                command_dim=int(targets[name].shape[1]),
                default_decimation=max(1, int(default_decimation)),
                tcp_frame_name=ik_solvers[name].tcp_frame_name,
                ik_solver=ik_solvers[name],
            )
            for name in selected
        }
        tcp_positions = {}
        tcp_orientations = {}
        for name in selected:
            tcp_positions[name], tcp_orientations[name] = ik_solvers[
                name
            ].command_tcp_world_poses(targets[name])
        object_pose_views = _create_tiled_object_pose_views(scene)
        # object_pose_views 让动态对象读写走 Isaac RigidPrim batched API。特别是
        # dynamic_chain，需要按 child rigid body 保存/恢复，否则 root pose 不能代表整条链。
        planner_backend_obj = _create_tiled_planner_backend(
            planner_backend,
            scene=scene,
            ik_solvers=ik_solvers,
        )
        initial_object_states = _capture_tiled_object_pose_snapshot(
            stage=session.stage,
            object_prim_paths=scene.object_prim_paths,
            env_origins=scene.env_origins,
            env_ids=np.arange(scene.config.num_envs, dtype=int),
            object_pose_views=object_pose_views,
        )
        return cls(
            env_name=env_name,
            env_config=env_config,
            session=session,
            scene=scene,
            render=bool(gui) or camera_output is not None,
            default_decimation=max(1, int(default_decimation)),
            robot_names=selected,
            episode_steps=np.zeros(scene.config.num_envs, dtype=int),
            episode_ids=np.zeros(scene.config.num_envs, dtype=int),
            initial_joint_positions=initial_positions,
            initial_joint_velocities=initial_velocities,
            target_positions=targets,
            initial_object_states=initial_object_states,
            command_adapters=command_adapters,
            ik_solvers=ik_solvers,
            tcp_positions_world=tcp_positions,
            tcp_orientations_wxyz=tcp_orientations,
            trajectory_buffer=TiledTrajectoryBuffer(num_envs=scene.config.num_envs),
            planner_manager=TiledPlannerManager(
                backend=planner_backend_obj,
                max_workers=int(planner_workers),
                max_pending_requests=max_pending_requests,
                max_completed_results=max_completed_results,
            ),
            sensor_cameras=sensor_cameras,
            camera_output=camera_output,
            quit_event=threading.Event(),
            object_pose_views=object_pose_views,
        )

    @property
    def time_s(self) -> float:
        """返回按 physics dt 推导的全局仿真时间。"""

        return float(self.step) * float(self.session.world.get_physics_dt())

    def status(self) -> dict[str, object]:
        """返回真实 Isaac tiled scene 摘要。"""

        return {
            "event": "status",
            "backend": "isaac",
            "env": self.env_name,
            "num_envs": self.scene.config.num_envs,
            "step": self.step,
            "time_s": self.time_s,
            "episode_steps": self.episode_steps.tolist(),
            "episode_ids": self.episode_ids.tolist(),
            "env_roots": list(self.scene.env_root_paths),
            "env_origins": self.scene.env_origins.tolist(),
            "runtime": {
                "inspect_env_ids": list(self.scene.config.runtime.inspect_env_ids),
            },
            "robots": {
                name: {
                    "count": runtime.view.count,
                    "num_dof": runtime.view.num_dof,
                    "command_joints": list(runtime.command_joint_names),
                    "ik_tcp_frame": self._command_adapter(name).tcp_frame_name,
                }
                for name, runtime in self._selected_runtime_items(None)
            },
            "sensors": {
                "cameras": [
                    {
                        "name": camera.name,
                        "prim_path": camera.prim_path,
                        "modalities": list(camera.settings.modalities),
                    }
                    for camera in self.sensor_cameras
                ],
            },
        }

    def idle_step(self) -> None:
        """空闲时保持当前 target 并推进一次 Isaac world/render。

        tiled interactive 的主循环大多数时间在等待外部 JSONL 命令。GUI viewport 和
        Foxglove live state 都需要 Kit/PhysX 持续被主线程 pump；否则场景已经加载好，
        但窗口可能停在黑屏，telemetry 也只能看到启动时的静态快照。
        """

        for name, runtime in self._selected_runtime_items(None):
            _apply_joint_targets(
                runtime.view,
                self.target_positions[name],
                joint_indices=runtime.command_joint_indices,
            )
        self._step_world(phase="idle")
        for name, _runtime in self._selected_runtime_items(None):
            self._refresh_tcp_state(name)

    def _step_world(self, *, phase: str) -> None:
        """推进一次 world，并在主线程采样 tiled sensor camera。"""

        self.session.world.step(render=self.render)
        if self.camera_output is not None:
            self.camera_output.observer.observe(
                self.session.world,
                step=self.step,
                phase=phase,
            )
        self.step += 1
        self.episode_steps[:] += 1

    @property
    def idle_period_s(self) -> float:
        """返回交互空闲刷新周期，优先跟随 rendering dt。"""

        rendering_dt = getattr(self.session.world, "get_rendering_dt", None)
        if callable(rendering_dt):
            try:
                value = float(rendering_dt())
                if value > 0.0:
                    return value
            except Exception:
                pass
        physics_dt = getattr(self.session.world, "get_physics_dt", None)
        if callable(physics_dt):
            try:
                value = float(physics_dt())
                if value > 0.0:
                    return value
            except Exception:
                pass
        return 1.0 / 60.0

    def reset(self, env_ids: np.ndarray | None = None) -> dict[str, object]:
        """把 selected env 的机器人和对象状态写回初始化状态。"""

        selected = _normalize_env_ids(env_ids, self.scene.config.num_envs)
        for name, runtime in self._selected_runtime_items(None):
            runtime.view.set_joint_positions(self.initial_joint_positions[name][selected], indices=selected)
            runtime.view.set_joint_velocities(
                np.zeros_like(self.initial_joint_velocities[name][selected]), indices=selected
            )
            self.target_positions[name][selected, :] = self.initial_joint_positions[name][selected][
                :, runtime.command_joint_indices
            ]
            self._command_adapter(name).reset()
            self._refresh_tcp_state(name, env_ids=selected)
        objects_reset = _restore_tiled_object_pose_snapshot(
            stage=self.session.stage,
            object_prim_paths=self.scene.object_prim_paths,
            snapshot=self.initial_object_states,
            env_ids=selected,
            env_origins=self.scene.env_origins,
            object_pose_views=self.object_pose_views,
        )
        self.trajectory_buffer.clear(env_ids=selected)
        self.planner_manager.cancel_matching(env_ids=selected)
        self.episode_steps[selected] = 0
        self.episode_ids[selected] += 1
        return {
            "event": "reset",
            "accepted": True,
            "env_ids": selected.tolist(),
            "step": self.step,
            "time_s": self.time_s,
            "episode_steps": self.episode_steps.tolist(),
            "episode_ids": self.episode_ids.tolist(),
            "objects_reset": int(objects_reset),
        }

    def step_action(
        self,
        action: TiledCommandAction,
        *,
        env_ids: np.ndarray | None = None,
        robot_names: RobotSelection = None,
    ) -> dict[str, object]:
        """执行同步关节 command step。"""

        selected = _normalize_env_ids(env_ids, self.scene.config.num_envs)
        ticks = _action_decimation(action, default_decimation=self.default_decimation)
        info: dict[str, object] = {}
        selected_robots = self._selected_runtime_items(robot_names, require_explicit=True)
        trajectories: dict[str, tuple[object, np.ndarray, np.ndarray]] = {}
        for name, runtime in selected_robots:
            command_indices = runtime.command_joint_indices
            current = np.asarray(
                runtime.view.get_joint_positions(joint_indices=command_indices),
                dtype=float,
            )
            previous_target = self.target_positions[name].copy()
            if action.kind.startswith("ee_"):
                robot_action = self._action_for_robot_reference(action, robot_name=name, env_ids=selected)
                robot_action = _action_for_selected_envs(
                    action=robot_action,
                    env_ids=selected,
                    current_positions=current,
                    current_tcp_positions=self._tcp_positions(name),
                    current_tcp_orientations_wxyz=self._tcp_orientations(name),
                    env_origins=self.scene.env_origins,
                )
                target = self._command_adapter(name).action_to_joint_target(
                    robot_action,
                    current_positions=current,
                    current_tcp_positions=self._tcp_positions(name),
                    current_tcp_orientations_wxyz=self._tcp_orientations(name),
                    env_origins=self.scene.env_origins,
                )
                targets = previous_target.copy()
                targets[selected, :] = target.joint_positions[selected, :]
                self._command_adapter(name).last_target = targets.copy()
                start = previous_target.copy()
                start[selected, :] = current[selected, :]
                trajectories[name] = (
                    runtime,
                    self._command_adapter(name).interpolate_to(
                        targets,
                        start=start,
                        action=robot_action,
                    ),
                    command_indices,
                )
                info[name] = {
                    "command_width": int(command_indices.size),
                    "ik": _jsonable_mapping(target.info),
                    "ik_backend": getattr(
                        getattr(self._command_adapter(name), "ik_solver", None),
                        "tcp_frame_name",
                        "",
                    ),
                }
                continue
            width = _action_width(action, default_width=command_indices.size)
            joint_indices = command_indices[:width]
            start = previous_target[:, :width].copy()
            start[selected, :] = current[selected, :width]
            if action.kind == "hold":
                targets = previous_target[:, :width].copy()
            elif action.kind == "joint_position_target":
                targets = previous_target[:, :width].copy()
                targets[selected, :] = _selected_rows(
                    action.values, selected.size, width, f"{action.kind}.values"
                )
            elif action.kind == "joint_delta_pos":
                targets = previous_target[:, :width].copy()
                targets[selected, :] = current[selected, :width] + _selected_rows(
                    action.values, selected.size, width, f"{action.kind}.values"
                )
            else:
                raise ValueError(f"unsupported Isaac interactive action: {action.kind}")
            trajectories[name] = (
                runtime,
                self._command_adapter(name).interpolate_to(
                    targets,
                    start=start,
                    action=action,
                ),
                joint_indices,
            )
            info[name] = {"command_width": int(width)}
        for tick_index in range(ticks):
            for name, (runtime, trajectory, joint_indices) in trajectories.items():
                tick_targets = trajectory[tick_index]
                self.target_positions[name][:, : tick_targets.shape[1]] = tick_targets
                _apply_joint_targets(
                    runtime.view,
                    tick_targets,
                    joint_indices=joint_indices,
                )
            self._step_world(phase="action")
        for name, _ in selected_robots:
            self._refresh_tcp_state(name, env_ids=selected)
        return {
            "event": "step",
            "accepted": True,
            "backend": "isaac",
            "kind": action.kind,
            "env_ids": selected.tolist(),
            "robots": [name for name, _ in selected_robots],
            "ticks": int(ticks),
            "step": self.step,
            "time_s": self.time_s,
            "episode_steps": self.episode_steps.tolist(),
            "info": info,
        }

    def get_state(
        self,
        *,
        env_ids: np.ndarray | None = None,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """读取 selected env 的机器人和对象状态。"""

        selected = _normalize_env_ids(env_ids, self.scene.config.num_envs)
        robots = {}
        for name, runtime in self._selected_runtime_items(None):
            command_indices = runtime.command_joint_indices
            robots[name] = {
                "joint_names": list(runtime.command_joint_names),
                "joint_positions": np.asarray(
                    runtime.view.get_joint_positions(indices=selected, joint_indices=command_indices),
                    dtype=float,
                ).tolist(),
                "joint_velocities": np.asarray(
                    runtime.view.get_joint_velocities(indices=selected, joint_indices=command_indices),
                    dtype=float,
                ).tolist(),
                "tcp_positions_world": self._tcp_positions(name)[selected].tolist(),
                "tcp_orientations_wxyz": self._tcp_orientations(name)[selected].tolist(),
            }
        payload = {
            "robots": robots,
            "objects": self._object_states(selected),
            "episode_steps": self.episode_steps[selected].tolist(),
            "episode_ids": self.episode_ids[selected].tolist(),
        }
        if fields is not None:
            payload = _filter_isaac_state_fields(payload, fields)
        return {
            "event": "state",
            "accepted": True,
            "backend": "isaac",
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
        """写回 selected env 的机器人 command joint 状态。"""

        selected = _normalize_env_ids(env_ids, self.scene.config.num_envs)
        robots = state.get("robots", {})
        if robots is not None and not isinstance(robots, Mapping):
            raise ValueError("set_state.state.robots must be a JSON object")
        for name, runtime in self._selected_runtime_items(None):
            robot_state = robots.get(name, {}) if isinstance(robots, Mapping) else {}
            if not isinstance(robot_state, Mapping):
                raise ValueError(f"state.robots.{name} must be a JSON object")
            command_indices = runtime.command_joint_indices
            if "joint_positions" in robot_state:
                q = _selected_rows(
                    robot_state["joint_positions"],
                    selected.size,
                    command_indices.size,
                    f"robots.{name}.joint_positions",
                )
                runtime.view.set_joint_positions(q, indices=selected, joint_indices=command_indices)
                self.target_positions[name][selected, :] = q
                self._refresh_tcp_state(name, env_ids=selected)
            if "joint_velocities" in robot_state:
                dq = _selected_rows(
                    robot_state["joint_velocities"],
                    selected.size,
                    command_indices.size,
                    f"robots.{name}.joint_velocities",
                )
                runtime.view.set_joint_velocities(dq, indices=selected, joint_indices=command_indices)
            self._command_adapter(name).reset()
        if "episode_steps" in state:
            self.episode_steps[selected] = _selected_int_rows(
                state["episode_steps"], selected.size, "episode_steps"
            )
        if "episode_ids" in state:
            self.episode_ids[selected] = _selected_int_rows(
                state["episode_ids"], selected.size, "episode_ids"
            )
        self.trajectory_buffer.clear(env_ids=selected)
        self.planner_manager.cancel_matching(env_ids=selected)
        return {
            "event": "set_state",
            "accepted": True,
            "backend": "isaac",
            "env_ids": selected.tolist(),
            "step": self.step,
            "time_s": self.time_s,
        }

    def get_snapshot(self, *, env_id: int) -> dict[str, object]:
        """读取单个 env 的 runtime-neutral snapshot。

        该方法只做协议响应包装；实际读取逻辑在 ``snapshots.adapters`` 中，便于 debug
        tiled runtime 和真实 Isaac tiled runtime 共用同一套语义。
        """

        snapshot = get_tiled_snapshot(self, env_id=int(env_id))
        return {
            "event": "snapshot",
            "accepted": True,
            "backend": "isaac",
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
        """把 runtime-neutral snapshot 写回 selected env。

        ``env_ids`` 可以包含多个目标 env；adapter 会广播 source snapshot，并在写回后清理
        selected env 的 trajectory/planner 缓存。
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
            "backend": "isaac",
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
        """把一个 source env 的 snapshot 克隆到多个 target env。

        clone 不走 get_state/set_state 的 batched 原始状态，而是复用 runtime-neutral
        snapshot，因此后续 single/dual/tiled 的兼容性逻辑保持一致。
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
            "backend": "isaac",
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
        """把已规划好的关节轨迹载入 Isaac runtime 的回放缓冲。"""

        selected = _normalize_env_ids(env_ids, self.scene.config.num_envs)
        robot = self._single_selected_robot_name(robot_name)
        runtime = self.scene.articulation_views[robot]
        loaded = load_interactive_trajectory(
            self.trajectory_buffer,
            trajectory,
            env_ids=selected,
            robot_name=robot,
            current_positions=self.target_positions[robot][selected],
            command_joint_names=runtime.command_joint_names,
        )
        return {
            "event": "trajectory_loaded",
            "accepted": True,
            "backend": "isaac",
            **loaded,
            "step": self.step,
            "time_s": self.time_s,
        }

    def step_trajectory(
        self,
        *,
        env_ids: np.ndarray | None = None,
        robot_names: RobotSelection = None,
        decimation: int | None = None,
    ) -> dict[str, object]:
        """按 physics tick 回放 ready trajectory，并同步推进真实 tiled scene。"""

        planner_results, planner_loaded = self._collect_planner_results()
        selected = _normalize_env_ids(env_ids, self.scene.config.num_envs)
        selected_robots = self._selected_runtime_items(robot_names, require_explicit=True)
        ticks = _positive_decimation(decimation, default_decimation=self.default_decimation)
        last_results: dict[str, object] = {}
        for _ in range(ticks):
            for name, runtime in selected_robots:
                result = self.trajectory_buffer.step(
                    robot_name=name,
                    current_positions=self.target_positions[name],
                    dt_s=float(self.session.world.get_physics_dt()),
                    env_ids=selected,
                )
                self.target_positions[name][:, :] = result.joint_positions
                _apply_joint_targets(
                    runtime.view,
                    self.target_positions[name],
                    joint_indices=runtime.command_joint_indices,
                )
                last_results[name] = result.to_json()
            self._step_world(phase="trajectory")
            for name, _ in selected_robots:
                self._refresh_tcp_state(name, env_ids=selected)
        return {
            "event": "trajectory_step",
            "accepted": True,
            "backend": "isaac",
            "env_ids": selected.tolist(),
            "robots": [name for name, _ in selected_robots],
            "ticks": int(ticks),
            "step": self.step,
            "time_s": self.time_s,
            "episode_steps": self.episode_steps.tolist(),
            "trajectory": last_results,
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
        """提交 Isaac tiled 关节空间规划请求。"""

        selected = _normalize_env_ids(env_ids, self.scene.config.num_envs)
        robot = self._single_selected_robot_name(robot_name)
        runtime = self.scene.articulation_views[robot]
        request = planning_request_from_message(
            message,
            robot_name=robot,
            env_ids=selected,
            current_positions=self.target_positions[robot][selected],
            command_joint_names=runtime.command_joint_names,
            default_sample_dt_s=float(self.session.world.get_physics_dt()),
            default_tcp_frame_name=self._command_adapter(robot).tcp_frame_name,
        )
        request_id = self.planner_manager.submit(request)
        return {
            "event": "plan_submitted",
            "accepted": True,
            "backend": "isaac",
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
        """提交 Isaac tiled hand-only motion，并默认追加到 trajectory queue。"""

        selected = _normalize_env_ids(env_ids, self.scene.config.num_envs)
        message_type = str(message.get("type", "hand"))
        loaded: list[dict[str, object]] = []
        if message_type == "dual_hand":
            for side in ("left", "right"):
                child = message.get(side)
                if child is None:
                    continue
                if not isinstance(child, Mapping):
                    raise ValueError(f"dual_hand.{side} must be a JSON object")
                robot = self._single_selected_robot_name(side)
                loaded.append(
                    self._load_hand_payload(
                        child,
                        parent=message,
                        robot_name=robot,
                        env_ids=selected,
                    )
                )
            if not loaded:
                raise ValueError("dual_hand requires left or right payload")
        else:
            robot = self._single_selected_robot_name(robot_name)
            loaded.append(
                self._load_hand_payload(
                    message,
                    parent=None,
                    robot_name=robot,
                    env_ids=selected,
                )
            )
        return {
            "event": "hand_motion_queued",
            "accepted": True,
            "backend": "isaac",
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
            "backend": "isaac",
            "ready": [item.to_json() for item in ready],
            "loaded": loaded,
            "planner": self.planner_manager.status(),
        }

    def cancel_plan(
        self,
        *,
        request_id: str | None = None,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """取消 Isaac 后台规划请求。"""

        robot = None if robot_name is None else self._single_selected_robot_name(robot_name)
        if request_id is not None:
            result: object = self.planner_manager.cancel(request_id)
        else:
            result = self.planner_manager.cancel_matching(robot_name=robot, env_ids=env_ids)
        return {
            "event": "plan_cancelled",
            "accepted": True,
            "backend": "isaac",
            "result": result,
        }

    def clear_completed(
        self,
        *,
        request_ids: str | tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """清理 Isaac planner completed result 缓存。"""

        return {
            "event": "completed_cleared",
            "accepted": True,
            "backend": "isaac",
            "result": self.planner_manager.clear_completed(request_ids),
        }

    def trajectory_status(
        self,
        *,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """返回 Isaac 轨迹缓冲状态。"""

        robot = None if robot_name is None else self._single_selected_robot_name(robot_name)
        return {
            "event": "trajectory_status",
            "accepted": True,
            "backend": "isaac",
            "step": self.step,
            "time_s": self.time_s,
            "trajectory": self.trajectory_buffer.status(robot_name=robot, env_ids=env_ids),
        }

    def clear_trajectory(
        self,
        *,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """清理 Isaac 轨迹缓冲。"""

        selected = None if env_ids is None else _normalize_env_ids(env_ids, self.scene.config.num_envs)
        robot = None if robot_name is None else self._single_selected_robot_name(robot_name)
        cleared = self.trajectory_buffer.clear(robot_name=robot, env_ids=selected)
        return {
            "event": "trajectory_cleared",
            "accepted": True,
            "backend": "isaac",
            "cleared": cleared,
            "step": self.step,
            "time_s": self.time_s,
        }

    def close(self) -> None:
        """关闭 Isaac SimulationApp。"""

        from linkerbot_sim.app.runtime.simulation_app_lifecycle import close_simulation_app

        if self._closed:
            return
        self._closed = True
        try:
            if self.camera_output is not None:
                self.camera_output.close()
        finally:
            self.planner_manager.shutdown()
            close_simulation_app(self.session.app)

    def _command_adapter(self, robot_name: str) -> TiledCommandAdapter:
        """返回机器人对应的 command adapter。"""

        if robot_name not in self.command_adapters:
            raise RuntimeError(
                f"robot {robot_name!r} has no tiled BatchedCuMotionIKSolver adapter"
            )
        return self.command_adapters[robot_name]

    def _ik_solver(self, robot_name: str) -> _WorldFrameBatchedIKSolver:
        """返回机器人对应的 world-frame IK solver。"""

        if robot_name not in self.ik_solvers:
            raise RuntimeError(f"robot {robot_name!r} has no tiled BatchedCuMotionIKSolver")
        return self.ik_solvers[robot_name]

    def _tcp_positions(self, robot_name: str) -> np.ndarray:
        """读取缓存的 world TCP positions。"""

        if robot_name not in self.tcp_positions_world:
            self._refresh_tcp_state(robot_name)
        return self.tcp_positions_world[robot_name]

    def _tcp_orientations(self, robot_name: str) -> np.ndarray:
        """读取缓存的 world TCP orientations。"""

        if robot_name not in self.tcp_orientations_wxyz:
            self._refresh_tcp_state(robot_name)
        return self.tcp_orientations_wxyz[robot_name]

    def _refresh_tcp_state(
        self,
        robot_name: str,
        *,
        env_ids: np.ndarray | None = None,
    ) -> None:
        """用 cuMotion FK 刷新 selected env 的 world TCP 位姿缓存。"""

        selected = _normalize_env_ids(env_ids, self.scene.config.num_envs)
        solver = self._ik_solver(robot_name)
        runtime = self.scene.articulation_views[robot_name]
        measured_positions = np.asarray(
            runtime.view.get_joint_positions(
                indices=selected,
                joint_indices=runtime.command_joint_indices,
            ),
            dtype=float,
        )
        positions, orientations = solver.command_tcp_world_poses(
            measured_positions,
            env_ids=selected,
        )
        if robot_name not in self.tcp_positions_world:
            self.tcp_positions_world[robot_name] = np.zeros((self.scene.config.num_envs, 3), dtype=float)
        if robot_name not in self.tcp_orientations_wxyz:
            self.tcp_orientations_wxyz[robot_name] = np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
                (self.scene.config.num_envs, 1),
            )
        self.tcp_positions_world[robot_name][selected, :] = positions
        self.tcp_orientations_wxyz[robot_name][selected, :] = orientations

    def _action_for_robot_reference(
        self,
        action: TiledCommandAction,
        *,
        robot_name: str,
        env_ids: np.ndarray,
    ) -> TiledCommandAction:
        """把 ``ee_pose_target`` 的 base-local 位姿转成 world 位姿。"""

        if action.kind != "ee_pose_target" or action.pose_reference_frame != "base":
            return action
        selected_values = _selected_action_rows(action.values, env_ids.size, 7, action.kind)
        solver = self._ik_solver(robot_name)
        roots = solver.root_positions_world[env_ids]
        rotations = solver.root_rotations_world_from_base[env_ids]
        root_quats = solver.root_quats_world_wxyz[env_ids]
        world_positions = roots + np.einsum("nij,nj->ni", rotations, selected_values[:, :3])
        world_orientations = _quat_multiply_rows(
            root_quats,
            _normalize_quaternions(selected_values[:, 3:7]),
        )
        return TiledCommandAction(
            kind=action.kind,
            values=np.concatenate([world_positions, world_orientations], axis=1),
            decimation=action.decimation,
            interpolation=action.interpolation,
            tcp_frame_name=action.tcp_frame_name,
            pose_reference_frame="world",
        )

    def _object_states(self, env_ids: np.ndarray) -> dict[str, object]:
        """读取 selected env 中所有 runtime object 的 pose/state。"""

        return _read_tiled_object_states(
            stage=self.session.stage,
            object_prim_paths=self.scene.object_prim_paths,
            env_origins=self.scene.env_origins,
            env_ids=env_ids,
            object_pose_views=self.object_pose_views,
        )

    def _selected_runtime_items(
        self,
        robot_names: RobotSelection,
        *,
        require_explicit: bool = False,
    ) -> tuple[tuple[str, object], ...]:
        """返回选中的 articulation runtime items。"""

        available = {name: self.scene.articulation_views[name] for name in self.robot_names}
        if robot_names is None and require_explicit and len(self.robot_names) != 1:
            raise ValueError("robots is required when multiple tiled robots are available")
        selected = _selected_robot_names(available, robot_names)
        return tuple((name, self.scene.articulation_views[name]) for name in selected)

    def _single_selected_robot_name(self, robot_name: str | None) -> str:
        """解析需要单机器人语义的 trajectory/planner 消息。"""

        if robot_name is None:
            if len(self.robot_names) != 1:
                raise ValueError("robot is required when multiple tiled robots are selected")
            return self.robot_names[0]
        selected = _selected_robot_names(
            {name: self.scene.articulation_views[name] for name in self.robot_names},
            (robot_name,),
        )
        if len(selected) != 1:
            raise ValueError("exactly one robot is required")
        return selected[0]

    def _load_hand_payload(
        self,
        payload: Mapping[str, object],
        *,
        parent: Mapping[str, object] | None,
        robot_name: str,
        env_ids: np.ndarray,
    ) -> dict[str, object]:
        """把单个 hand payload 写入 Isaac trajectory buffer。"""

        runtime = self.scene.articulation_views[robot_name]
        return load_interactive_hand_motion(
            self.trajectory_buffer,
            payload,
            env_ids=env_ids,
            robot_name=robot_name,
            current_positions=self.target_positions[robot_name][env_ids],
            command_joint_names=runtime.command_joint_names,
            parent_payload=parent,
        )

    def _collect_planner_results(
        self,
        *,
        timeout_s: float = 0.0,
    ) -> tuple[tuple[TiledPlanningResult, ...], list[dict[str, object]]]:
        """收集 planner results，并把成功结果载入 trajectory buffer。"""

        results = self.planner_manager.collect_ready(timeout_s=timeout_s)
        loaded = load_ready_planning_results(self.trajectory_buffer, results)
        return results, loaded


def _create_tiled_object_pose_views(scene: object) -> dict[str, object]:
    """为 dynamic tiled objects 创建 batched rigid view，用于 reset/get_state/snapshot。

    普通 rigid object 每个 env 一个 row；dynamic-chain object 会把所有 child bodies 展平为
    env-major rows，并由 ``TiledDynamicChainObjectPoseView`` 负责局部/世界坐标转换。
    """

    object_paths = getattr(scene, "object_prim_paths", {}) or {}
    rigid_objects: list[tuple[str, tuple[str, ...]]] = []
    chain_objects: list[
        tuple[str, tuple[str, ...], tuple[str, ...], tuple[tuple[str, ...], ...]]
    ] = []
    for handle in getattr(scene, "object_handles", ()) or ():
        kind = str(getattr(handle, "kind", ""))
        model = getattr(handle, "model", None)
        name = str(getattr(handle, "name", ""))
        paths = tuple(str(path) for path in (object_paths.get(name) or ()))
        if not name or not paths:
            continue
        if kind == "rigid":
            if bool(getattr(model, "static", False)):
                continue
            rigid_objects.append((name, paths))
        elif kind == "dynamic_chain":
            # dynamic_chain 的 root prim 往往只是容器，真正可动的是 child rigid bodies；
            # 因此需要基于 env_0 推导 body suffix，再为所有 env 拼出完整 body path。
            body_names, body_suffixes = _dynamic_chain_body_suffixes(
                name=name,
                model=model,
                env_zero_root_path=paths[0],
            )
            body_paths_by_env = tuple(
                tuple(_join_prim_path_suffix(root_path, suffix) for suffix in body_suffixes)
                for root_path in paths
            )
            chain_objects.append((name, paths, body_names, body_paths_by_env))
    if not rigid_objects and not chain_objects:
        return {}
    try:
        from isaacsim.core.prims import RigidPrim
    except Exception as exc:
        raise RuntimeError(
            "isaacsim.core.prims.RigidPrim is required for tiled dynamic object state sync"
        ) from exc

    result: dict[str, object] = {}
    for name, paths in rigid_objects:
        try:
            view = RigidPrim(
                prim_paths_expr=list(paths),
                name=f"tiled_object_{_identifier_suffix(name)}",
                reset_xform_properties=False,
            )
            initialize = getattr(view, "initialize", None)
            if callable(initialize):
                initialize()
        except Exception as exc:
            raise RuntimeError(
                f"failed to create tiled object rigid view for {name!r}"
            ) from exc
        result[name] = view
    for name, _, body_names, body_paths_by_env in chain_objects:
        flat_body_paths = [
            body_path
            for env_body_paths in body_paths_by_env
            for body_path in env_body_paths
        ]
        try:
            view = RigidPrim(
                prim_paths_expr=flat_body_paths,
                name=f"tiled_object_{_identifier_suffix(name)}_bodies",
                reset_xform_properties=False,
            )
            initialize = getattr(view, "initialize", None)
            if callable(initialize):
                initialize()
        except Exception as exc:
            raise RuntimeError(
                f"failed to create tiled dynamic-chain object rigid view for {name!r}"
            ) from exc
        result[name] = TiledDynamicChainObjectPoseView(
            view=view,
            body_names=body_names,
            body_paths_by_env=body_paths_by_env,
        )
    return result


def _dynamic_chain_body_suffixes(
    *,
    name: str,
    model: object,
    env_zero_root_path: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """从 env_0 dynamic_chain model 推导每个 child rigid body 的 root-relative suffix。"""

    bodies = []
    if isinstance(model, Mapping):
        bodies = list(model.get("bodies", ()) or ())
    else:
        bodies = list(getattr(model, "bodies", ()) or ())
    body_names: list[str] = []
    body_suffixes: list[str] = []
    for body in bodies:
        path_getter = getattr(body, "GetPath", None)
        body_path = str(path_getter() if callable(path_getter) else body)
        body_suffix = _prim_path_suffix(env_zero_root_path, body_path)
        name_getter = getattr(body, "GetName", None)
        body_name = str(name_getter() if callable(name_getter) else body_path.rsplit("/", 1)[-1])
        body_names.append(body_name)
        body_suffixes.append(body_suffix)
    if not body_suffixes:
        raise RuntimeError(
            f"tiled dynamic-chain object {name!r} has no child rigid bodies for state sync"
        )
    return tuple(body_names), tuple(body_suffixes)


def _prim_path_suffix(root_path: str, child_path: str) -> str:
    """返回 child path 相对 root path 的 slash-prefixed suffix。"""

    root = str(root_path).rstrip("/")
    child = str(child_path)
    if child == root:
        return ""
    prefix = f"{root}/"
    if not child.startswith(prefix):
        raise RuntimeError(f"dynamic-chain body path {child!r} is not under {root!r}")
    return child[len(root) :]


def _join_prim_path_suffix(root_path: str, suffix: str) -> str:
    """把 tiled env 中的 object root 和 env_0 body suffix 拼回 child path。"""

    return f"{str(root_path).rstrip('/')}{suffix}"


def _identifier_suffix(value: str) -> str:
    """生成 Isaac view name 可用的保守后缀。"""

    suffix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    return suffix or "object"


def _create_tiled_planner_backend(
    planner_backend: str,
    *,
    scene: object,
    ik_solvers: dict[str, _WorldFrameBatchedIKSolver],
) -> object | None:
    """创建 async planner backend；``None`` 表示使用 manager 默认 linear backend。"""

    backend = str(planner_backend).strip().lower()
    if backend == "linear":
        return None
    if backend != "cumotion":
        raise ValueError("planner_backend must be one of: linear, cumotion")

    from linkerbot_sim.backends.cumotion import (
        CuMotionConfig,
        CuMotionContext,
        CuMotionJointPlannerBackend,
        CuMotionMotionPlanner,
    )
    from linkerbot_sim.app.motion.specs import specified_path_planner_config
    from linkerbot_sim.configs.profiles import load_profile_yaml
    from linkerbot_sim.planning.requests import SpecifiedPathRequest

    def _planner_factory(robot_name: str) -> object:
        """为指定 tiled robot 创建独立 cuMotion planner facade。"""

        robot = scene.robots[str(robot_name)]
        robot_config = load_profile_yaml("robot", robot.profile_name)
        context = CuMotionContext(CuMotionConfig.from_mapping(robot_config))
        tcp_frame_name = ik_solvers[str(robot_name)].tcp_frame_name
        return _TiledCuMotionPlannerFacade(
            context=context,
            tcp_frame_name=tcp_frame_name,
            planner_type=CuMotionMotionPlanner,
            specified_path_request_type=SpecifiedPathRequest,
            specified_path_config_fn=specified_path_planner_config,
        )

    return CuMotionJointPlannerBackend(_planner_factory)


class _TiledCuMotionPlannerFacade:
    """按 request 类型为 tiled async planner 选择 cuMotion pipeline。"""

    def __init__(
        self,
        *,
        context: object,
        tcp_frame_name: str,
        planner_type: type,
        specified_path_request_type: type,
        specified_path_config_fn: Callable[..., object],
    ) -> None:
        """保存 cuMotion context 和 request 分发所需的 planner 类型。"""

        self.context = context
        self.tcp_frame_name = tcp_frame_name
        self._planner_type = planner_type
        self._specified_path_request_type = specified_path_request_type
        self._specified_path_config_fn = specified_path_config_fn
        self._base_config = getattr(getattr(context, "config", None), "motion_planner", None)

    def joint_names(self) -> list[str]:
        """返回 planner C-space joint names。"""

        return self.context.joint_names()

    def plan(self, request: object) -> object:
        """按 request 类型创建一次性 cuMotion planner facade。"""

        tcp_frame_name = getattr(request, "tcp_frame_name", None) or self.tcp_frame_name
        config = self._base_config
        if isinstance(request, self._specified_path_request_type):
            config = self._specified_path_config_fn(self._base_config, path=request.path)
        return self._planner_type(
            self.context,
            tcp_frame_name=tcp_frame_name,
            config=config,
        ).plan(request)


def _refresh_gui_view_after_scene_build(session: object, runtime_settings: object) -> None:
    """场景加载完成后重新应用 GUI 视角，并 pump 几帧 Kit。

    ``create_simulation_session`` 会在空 stage 上创建灯光和设置 viewport；tiled scene
    导入和 clone 之后，Isaac 5.x 的 viewport 偶尔仍停留在未完成初始化的黑帧。这里在
    资产、clone 和 world reset 都完成后再应用一次配置，让 GUI 看到真实场景。
    """

    from linkerbot_sim.envs.scene_builder import configure_visuals

    configure_visuals(runtime_settings.visuals)
    app = getattr(session, "app", None)
    update = getattr(app, "update", None)
    if not callable(update):
        return
    for _ in range(3):
        update()
