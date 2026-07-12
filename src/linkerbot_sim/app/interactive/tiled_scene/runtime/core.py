"""Isaac/PhysX TiledSceneRuntime 的状态模型与公开 facade。"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.runtime.factory import (
    close_tiled_scene_runtime,
    create_tiled_scene_runtime,
)
from linkerbot_sim.app.interactive.tiled_scene.runtime.ik import (
    _WorldFrameBatchIKBackend,
    action_for_robot_reference as _action_for_robot_reference,
    command_adapter as _command_adapter,
    ik_solver as _ik_solver,
    refresh_tcp_state as _refresh_tcp_state,
    tcp_orientations as _tcp_orientations,
    tcp_positions as _tcp_positions,
)
from linkerbot_sim.app.interactive.tiled_scene.runtime.planning import (
    cancel_plan as _cancel_plan,
    clear_completed as _clear_completed,
    clear_trajectory as _clear_trajectory,
    load_trajectory as _load_trajectory,
    planner_status as _planner_status,
    step_trajectory as _step_trajectory,
    submit_hand_motion as _submit_hand_motion,
    submit_plan as _submit_plan,
    trajectory_status as _trajectory_status,
)
from linkerbot_sim.app.interactive.tiled_scene.runtime.state import (
    clone_state as _clone_state,
    get_snapshot as _get_snapshot,
    get_state as _get_state,
    set_snapshot as _set_snapshot,
    set_state as _set_state,
)
from linkerbot_sim.app.interactive.tiled_scene.runtime.stepping import (
    idle_period_s as _idle_period_s,
    idle_step as _idle_step,
    reset as _reset,
    step_action as _step_action,
    step_world as _step_world,
)
from linkerbot_sim.app.interactive.tiled_scene.selectors import (
    RobotSelection,
    selected_robot_names,
)
from linkerbot_sim.sensors.camera.observer import CameraOutputHandle
from linkerbot_sim.sensors.camera.runtime import SensorCameraRuntime
from linkerbot_sim.utils.output_paths import OutputPathPlan
from linkerbot_sim.tiled.control.adapter import TiledCommandAdapter
from linkerbot_sim.tiled.control.types import TiledCommandAction
from linkerbot_sim.tiled.planning.manager import TiledPlannerManager
from linkerbot_sim.tiled.playback.buffer import TiledTrajectoryBuffer

if TYPE_CHECKING:
    from linkerbot_sim.configs.runtime import (
        CameraOutputRuntimeSettings,
        PlannerRequestDefaults,
        PlaybackResourceSettings,
        RuntimeCommandDefaults,
        ShutdownSettings,
        SimulationAppSettings,
    )


@dataclass
class TiledSceneRuntime:
    """持有真实 Isaac tiled session，并把各领域操作显式委托给 service。"""

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
    ik_solvers: dict[str, _WorldFrameBatchIKBackend]
    tcp_positions_world: dict[str, np.ndarray]
    tcp_orientations_wxyz: dict[str, np.ndarray]
    trajectory_buffer: TiledTrajectoryBuffer
    planner_manager: TiledPlannerManager
    sensor_cameras: tuple[SensorCameraRuntime, ...]
    camera_output: CameraOutputHandle | None
    quit_event: threading.Event
    object_pose_views: dict[str, object] = field(default_factory=dict)
    planner_request_defaults: object | None = None
    command_defaults: object | None = None
    transport_status_provider: Callable[[], Mapping[str, object]] | None = None
    step: int = 0
    _closed: bool = False
    _planner_closed: bool = False
    _camera_closed: bool = False
    _ik_closed: bool = False
    _app_closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        env_name: str,
        env_config: Mapping[str, object],
        simulation_app: SimulationAppSettings,
        camera_output_settings: CameraOutputRuntimeSettings,
        shutdown_settings: ShutdownSettings,
        default_decimation: int,
        controller_bundle: str = "default",
        planner_workers: int = 2,
        max_pending_requests: int | None = 64,
        max_completed_results: int | None = 256,
        max_batch_problems: int = 64,
        oversize_request_policy: str = "split",
        failure_policy: str = "hold_failed_env",
        cache_root: str | None = None,
        planner_request_defaults: PlannerRequestDefaults | None = None,
        command_defaults: RuntimeCommandDefaults | None = None,
        playback_settings: PlaybackResourceSettings | None = None,
        planner_shutdown_timeout_s: float = 30.0,
        planner_backend: str = "curobo",
        curobo_profile: str = "default",
        joint_batch_mode: str = "auto",
        additional_output_path_plans: Sequence[OutputPathPlan] = (),
    ) -> "TiledSceneRuntime":
        """创建场景、相机、IK 和 planner，并完成首次 reset/finalize。"""

        return create_tiled_scene_runtime(
            cls,
            env_name=env_name,
            env_config=env_config,
            simulation_app=simulation_app,
            camera_output_settings=camera_output_settings,
            shutdown_settings=shutdown_settings,
            default_decimation=default_decimation,
            controller_bundle=controller_bundle,
            planner_workers=planner_workers,
            max_pending_requests=max_pending_requests,
            max_completed_results=max_completed_results,
            max_batch_problems=max_batch_problems,
            oversize_request_policy=oversize_request_policy,
            failure_policy=failure_policy,
            cache_root=cache_root,
            planner_request_defaults=planner_request_defaults,
            command_defaults=command_defaults,
            playback_settings=playback_settings,
            planner_shutdown_timeout_s=planner_shutdown_timeout_s,
            planner_backend=planner_backend,
            curobo_profile=curobo_profile,
            joint_batch_mode=joint_batch_mode,
            additional_output_path_plans=additional_output_path_plans,
        )

    @property
    def time_s(self) -> float:
        """返回按 physics dt 推导的全局仿真时间。"""

        return float(self.step) * float(self.session.world.get_physics_dt())

    def status(self) -> dict[str, object]:
        """返回 scene、机器人、传感器和 episode 的运行摘要。"""

        status = {
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
                "inspect_env_ids": list(self.scene.config.diagnostics.inspect_env_ids),
            },
            "per_env_metadata": [
                {
                    "env_id": item.env_id,
                    "metadata": dict(item.metadata),
                }
                for item in self.scene.config.per_env
                if item.metadata
            ],
            "robots": [
                {
                    "robot_id": articulation.robot_id,
                    "label": articulation.label or name,
                    "robot_profile": self.scene.robots[name].profile_name,
                    "controller_profile": self.scene.robots[name].controller_profile,
                    "kind": self.scene.robots[name].kind,
                    "supports_planning": self.scene.robots[name].supports_planning,
                    "count": articulation.view.count,
                    "num_dof": articulation.view.num_dof,
                    "command_joints": list(articulation.command_joint_names),
                    "ik_tcp_frame": self._command_adapter(name).tcp_frame_name,
                }
                for name, articulation in self._selected_runtime_items(None)
            ],
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
        if self.transport_status_provider is not None:
            status["transport"] = dict(self.transport_status_provider())
        telemetry_status = getattr(self, "telemetry_status_provider", None)
        if callable(telemetry_status):
            status["telemetry"] = dict(telemetry_status())
        planner_status = getattr(self.planner_manager, "status", None)
        if callable(planner_status):
            status["planner"] = planner_status()
        publisher = getattr(self.camera_output, "publisher", None)
        camera_status = getattr(publisher, "status", None)
        if callable(camera_status):
            status["camera_output"] = camera_status()
        return status

    def robot_name_for_id(self, robot_id: int) -> str:
        """把 session 内稳定 robot ID 解析为 tiled 内部标签。"""

        return self.scene.robot_label(robot_id)

    def idle_step(self) -> None:
        """保持当前关节目标并推进一个空闲 tick。"""

        _idle_step(self)

    def _step_world(self, *, phase: str) -> None:
        """在主线程推进一次 world 并采样相机。"""

        _step_world(self, phase=phase)

    @property
    def idle_period_s(self) -> float:
        """返回交互循环建议的空闲刷新周期。"""

        return _idle_period_s(self)

    def reset(self, env_ids: np.ndarray) -> dict[str, object]:
        """重置 selected env 的机器人、对象和调度状态。"""

        return _reset(self, env_ids)

    def step_action(
        self,
        action: TiledCommandAction,
        *,
        env_ids: np.ndarray,
        robot_names: RobotSelection = None,
    ) -> dict[str, object]:
        """同步执行 selected env/robot 的 command action。"""

        return _step_action(
            self,
            action,
            env_ids=env_ids,
            robot_names=robot_names,
        )

    def get_state(
        self,
        *,
        env_ids: np.ndarray,
        fields: tuple[str, ...] | None = None,
        include_efforts: bool = False,
    ) -> dict[str, object]:
        """读取 selected env 的 batched runtime state。"""

        return _get_state(
            self,
            env_ids=env_ids,
            fields=fields,
            include_efforts=include_efforts,
        )

    def set_state(
        self,
        state: Mapping[str, object],
        *,
        env_ids: np.ndarray,
    ) -> dict[str, object]:
        """写回 selected env 的 batched runtime state。"""

        return _set_state(self, state, env_ids=env_ids)

    def get_snapshot(self, *, env_id: int) -> dict[str, object]:
        """读取单个 env 的 runtime-neutral snapshot。"""

        return _get_snapshot(self, env_id=env_id)

    def set_snapshot(
        self,
        snapshot: Mapping[str, object],
        *,
        env_ids: np.ndarray,
        label_map: Mapping[str, str] | None = None,
        strict: bool = True,
    ) -> dict[str, object]:
        """把 runtime-neutral snapshot 写回 selected env。"""

        return _set_snapshot(
            self,
            snapshot,
            env_ids=env_ids,
            label_map=label_map,
            strict=strict,
        )

    def clone_state(
        self,
        *,
        source_env_id: int,
        target_env_ids: np.ndarray,
        strict: bool = True,
    ) -> dict[str, object]:
        """把一个 source env 克隆到多个 target env。"""

        return _clone_state(
            self,
            source_env_id=source_env_id,
            target_env_ids=target_env_ids,
            strict=strict,
        )

    def load_trajectory(
        self,
        trajectory: Mapping[str, object],
        *,
        env_ids: np.ndarray,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """把关节轨迹载入 selected robot/env 的回放缓冲。"""

        return _load_trajectory(
            self,
            trajectory,
            env_ids=env_ids,
            robot_name=robot_name,
        )

    def step_trajectory(
        self,
        *,
        env_ids: np.ndarray,
        robot_names: RobotSelection = None,
        decimation: int | None = None,
    ) -> dict[str, object]:
        """推进 selected robot/env 的轨迹回放。"""

        return _step_trajectory(
            self,
            env_ids=env_ids,
            robot_names=robot_names,
            decimation=decimation,
        )

    def submit_plan(
        self,
        message: Mapping[str, object],
        *,
        env_ids: np.ndarray,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """提交异步规划请求。"""

        return _submit_plan(
            self,
            message,
            env_ids=env_ids,
            robot_name=robot_name,
        )

    def submit_hand_motion(
        self,
        message: Mapping[str, object],
        *,
        env_ids: np.ndarray,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """提交 hand joint motion。"""

        return _submit_hand_motion(
            self,
            message,
            env_ids=env_ids,
            robot_name=robot_name,
        )

    def planner_status(self, *, wait_timeout_s: float = 0.0) -> dict[str, object]:
        """收集规划结果并返回 planner 状态。"""

        return _planner_status(self, wait_timeout_s=wait_timeout_s)

    def cancel_plan(
        self,
        *,
        request_id: str | None = None,
        env_ids: np.ndarray | None = None,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """取消匹配的后台规划请求。"""

        return _cancel_plan(
            self,
            request_id=request_id,
            env_ids=env_ids,
            robot_name=robot_name,
        )

    def clear_completed(
        self,
        *,
        request_ids: str | tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """清理 planner completed result 缓存。"""

        return _clear_completed(self, request_ids=request_ids)

    def trajectory_status(
        self,
        *,
        env_ids: np.ndarray,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """返回 selected trajectory buffer 状态。"""

        return _trajectory_status(
            self,
            env_ids=env_ids,
            robot_name=robot_name,
        )

    def clear_trajectory(
        self,
        *,
        env_ids: np.ndarray,
        robot_name: str | None = None,
    ) -> dict[str, object]:
        """清理 selected trajectory buffer。"""

        return _clear_trajectory(
            self,
            env_ids=env_ids,
            robot_name=robot_name,
        )

    def close(self) -> bool:
        """有界关闭 planner、cuRobo、camera 与 SimulationApp，并允许超时重试。"""

        return close_tiled_scene_runtime(self)

    def _command_adapter(self, robot_name: str) -> TiledCommandAdapter:
        """返回 robot command adapter；snapshot adapter 也依赖该边界。"""

        return _command_adapter(self, robot_name)

    def _ik_solver(self, robot_name: str) -> _WorldFrameBatchIKBackend:
        """返回 robot world-frame IK solver。"""

        return _ik_solver(self, robot_name)

    def _tcp_positions(self, robot_name: str) -> np.ndarray:
        """返回缓存的 world TCP positions。"""

        return _tcp_positions(self, robot_name)

    def _tcp_orientations(self, robot_name: str) -> np.ndarray:
        """返回缓存的 world TCP orientations。"""

        return _tcp_orientations(self, robot_name)

    def _refresh_tcp_state(
        self,
        robot_name: str,
        *,
        env_ids: np.ndarray | None = None,
    ) -> None:
        """刷新 selected env 的 TCP cache；snapshot restore 也依赖该边界。"""

        _refresh_tcp_state(self, robot_name, env_ids=env_ids)

    def _action_for_robot_reference(
        self,
        action: TiledCommandAction,
        *,
        robot_name: str,
        env_ids: np.ndarray,
    ) -> TiledCommandAction:
        """把 base-local EE target 转换为 world target。"""

        return _action_for_robot_reference(
            self,
            action,
            robot_name=robot_name,
            env_ids=env_ids,
        )

    def _selected_runtime_items(
        self,
        robot_names: RobotSelection,
        *,
        require_explicit: bool = False,
    ) -> tuple[tuple[str, object], ...]:
        """返回 selector 对应的 articulation runtime items。"""

        available = {
            name: self.scene.articulation_views[name] for name in self.robot_names
        }
        if robot_names is None and require_explicit and len(self.robot_names) != 1:
            raise ValueError(
                "robots is required when multiple tiled robots are available"
            )
        selected = selected_robot_names(available, robot_names)
        return tuple((name, self.scene.articulation_views[name]) for name in selected)

    def _single_selected_robot_name(self, robot_name: str | None) -> str:
        """解析需要单机器人语义的 trajectory/planner selector。"""

        if robot_name is None:
            if len(self.robot_names) != 1:
                raise ValueError(
                    "robot is required when multiple tiled robots are selected"
                )
            return self.robot_names[0]
        selected = selected_robot_names(
            {name: self.scene.articulation_views[name] for name in self.robot_names},
            (robot_name,),
        )
        if len(selected) != 1:
            raise ValueError("exactly one robot is required")
        return selected[0]
