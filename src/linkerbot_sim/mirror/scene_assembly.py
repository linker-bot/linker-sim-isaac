"""任意数量机器人 Mirror Isaac 场景的统一装配与资源集合。

创建顺序固定为启动 session、导入对象/机器人、一次 world reset、finalize controller、构建
registry/collision provider；失败时按相反顺序释放已创建资源。``MirrorSceneResources`` 是普通交互、
snapshot、telemetry 和 timeline executor 的共同根对象，不包含单/双机器人分支。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from linkerbot_sim.configuration.physics import (
    NewtonCpuSettings,
    NewtonCudaSettings,
    PhysxCpuSettings,
)
from linkerbot_sim.configuration.curobo import CuroboProfileSettings
from linkerbot_sim.configuration.modes.mirror import MirrorConfig
from linkerbot_sim.configuration.outputs import (
    CameraOutputSettings,
    LoggingOutputSettings,
    MirrorOutputsSettings,
    TelemetryOutputSettings,
)
from linkerbot_sim.configuration.control import HybridForcePositionSettings
from linkerbot_sim.configuration.scenes import (
    CameraSettings,
    DistantLightSettings,
    DomeLightSettings,
    MirrorSceneSettings,
    RobotInstanceSettings,
    SceneVisualSettings,
)
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.objects.runtime import (
    RuntimeObjectConfig,
    RuntimeObjectHandle,
    add_runtime_objects,
    runtime_object_handles_by_name,
    runtime_objects_from_settings,
)
from linkerbot_sim.objects.state_views import (
    SceneObjectStateView,
    create_scene_object_state_views,
)
from linkerbot_sim.mirror.robots import (
    CoordinationPolicy,
    RobotPlanningRegistry,
    RobotRegistry,
    RobotRuntime,
)
from linkerbot_sim.mirror.collision.registry import SceneCollisionRegistry
from linkerbot_sim.mirror.collision.robot_provider import RobotObstacleProvider
from linkerbot_sim.mirror.scene_settings import MirrorSceneRuntimeSettings
from linkerbot_sim.isaac.world import configure_visuals, set_physics_gravity
from linkerbot_sim.isaac.session import (
    IsaacSession,
    create_isaac_session_from_spec,
)
from linkerbot_sim.isaac.spec import (
    IsaacAppSpec,
    IsaacComputeSpec,
    IsaacNewtonCpuSpec,
    IsaacNewtonCudaSpec,
    IsaacPhysxCpuSpec,
    IsaacRenderSpec,
    IsaacSessionSpec,
)
from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    robot_scene_instances_from_settings,
    resolve_controller_profile,
)
from linkerbot_sim.assets.solver_overrides import SolverIterationConfig
from linkerbot_sim.backends.curobo.profile_merge import (
    curobo_config_from_profiles,
)
from linkerbot_sim.assets.instance_paths import validate_disjoint_instance_prim_paths
from linkerbot_sim.mirror.interface.state_stream import (
    InteractiveStateStreamConfig,
    start_interactive_state_stream,
)
from linkerbot_sim.mirror.motion.backend import MirrorTimelineBackend
from linkerbot_sim.mirror.control_mode import MirrorControlBinding
from linkerbot_sim.mirror.reset_runtime import MirrorResetOptions, reset_mirror_scene
from linkerbot_sim.snapshots.mirror_adapter import (
    get_mirror_snapshot,
    set_mirror_snapshot,
)
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.configuration.controllers import ControllerProfiles
from linkerbot_sim.execution.setup import (
    bind_imported_robot_articulation,
    finalize_robot_controller,
    import_execution_robot_to_stage,
)
from linkerbot_sim.telemetry.foxglove import FoxgloveTopicConfig, prepare_mcap_output
from linkerbot_sim.logging.csv_writer import (
    CsvOutputPlan,
    plan_csv_output,
)
from linkerbot_sim.logging.joint_logger import (
    JointTrackingLogger,
    joint_tracking_fieldnames,
)
from linkerbot_sim.logging.hybrid_control_logger import (
    HybridControlLogger,
    hybrid_control_fieldnames,
)
from linkerbot_sim.robots.capabilities import PlanningCapability
from linkerbot_sim.robots.joint_groups import JointGroupLayout
from linkerbot_sim.robots.tcp_binding import resolve_physical_tcp_binding
from linkerbot_sim.isaac.physics.physx_task_space import PhysxTaskSpacePort
from linkerbot_sim.sensors.camera.observer import (
    CameraOutputHandle,
    open_prepared_camera_output,
    prepare_camera_output,
)
from linkerbot_sim.sensors.camera.runtime import (
    SensorCameraRuntime,
    create_sensor_camera_runtimes,
    initialize_sensor_camera_runtimes,
)
from linkerbot_sim.sensors.camera.config import (
    SensorCameraIntrinsicsSettings,
    SensorCameraOutputSettings,
    SensorCameraSettings,
)
from linkerbot_sim.sensors.config import SceneSensorSettings
from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.output_paths import (
    OutputPathPlan,
    apply_output_path_plans,
    plan_output_file,
    timestamped_run_name,
    validate_output_path_plans,
)
from linkerbot_sim.utils.paths import repo_path


class MirrorPhysicsAdapter:
    """把 concrete PhysicsRuntime 适配给既有 Mirror asset/control consumers。

    adapter 只转发时间与 scene API，不持有或复制任何物理状态，也没有 ``close``；因此
    PhysX World 或 Newton Model/State 的唯一 owner 仍是 session 中的 concrete runtime。
    """

    def __init__(self, runtime: object) -> None:
        self.runtime = runtime

    @property
    def scene(self) -> object:
        return self.runtime.scene

    def reset(self) -> None:
        self.runtime.reset()

    def forward(self) -> None:
        self.runtime.forward()

    def step(self, *, render: bool = False) -> None:
        self.runtime.step(render=render)

    def render(self) -> None:
        self.runtime.render()

    def get_physics_dt(self) -> float:
        return float(self.runtime.get_physics_dt())

    def get_rendering_dt(self) -> float:
        return float(self.runtime.get_rendering_dt())


@dataclass
class MirrorSceneResources:
    """MirrorRuntime 借用的场景资源索引，不拥有 session 关闭权。

    该对象集中暴露 robot/object/sensor/collision/planning 句柄，供 snapshot、telemetry 和
    timeline executor 使用。每个可关闭资源都会单独登记到 ``MirrorAssembly``，最终只由
    ``MirrorRuntime`` 按逆依赖顺序关闭；这里故意不实现 ``close``。
    """

    session: IsaacSession
    physics: MirrorPhysicsAdapter
    scene: MirrorSceneSettings
    robot_registry: RobotRegistry
    planning_registry: RobotPlanningRegistry
    collision_registry: SceneCollisionRegistry
    runtime_object_configs: tuple[RuntimeObjectConfig, ...] = ()
    object_handles: tuple[RuntimeObjectHandle, ...] = ()
    objects: Mapping[str, RuntimeObjectHandle] = field(default_factory=dict)
    object_state_views: Mapping[str, SceneObjectStateView] = field(default_factory=dict)
    sensor_cameras: tuple[SensorCameraRuntime, ...] = ()
    camera_output: CameraOutputHandle | None = None
    state_observer: object | None = None
    camera_observer: object | None = None
    loggers: tuple[JointTrackingLogger, ...] = ()
    hybrid_control_logger: HybridControlLogger | None = None
    status_prefix: str | None = None
    control_mode_state_provider: Callable[[], object] | None = None
    hybrid_diagnostics_provider: Callable[[], Mapping[str, object]] | None = None
    _completed_physics_steps: int = field(default=0, init=False, repr=False)

    @property
    def robots_by_id(self) -> dict[int, RobotRuntime]:
        """返回 session robot ID 到 ``RobotRuntime`` 的主索引。"""

        return self.robot_registry.robots_by_id

    @property
    def robot_id_by_label(self) -> dict[str, int]:
        """返回稳定 label 到本次 session ID 的反向索引。"""

        return self.robot_registry.robot_id_by_label

    def robot(self, robot_id: int) -> RobotRuntime:
        """按 session ID 解析 robot。"""

        return self.robot_registry.robot(robot_id)

    def robot_by_label(self, label: str) -> RobotRuntime:
        """按稳定 label 解析 robot，供内部 snapshot/registry 逻辑使用。"""

        return self.robot_registry.robot_by_label(label)

    def claim_completed_step(self) -> int:
        """为刚完成的唯一 physics tick 分配全局、单调递增的零基序号。"""

        step = self._completed_physics_steps
        self._completed_physics_steps += 1
        return step

    def observe_after_step(
        self,
        *,
        step: int | None = None,
        phase: str = "idle",
        write_idle_logs: bool = True,
    ) -> int:
        """在 owner thread 的物理步后各采样一次日志、状态和相机。

        idle 与 timeline 必须共享 ``claim_completed_step``，否则二者交替时 telemetry 时间会
        倒退、camera 会因旧采样游标漏帧。timeline 已持有精确 control target，会先自行写
        日志并以 ``write_idle_logs=False`` 调用本方法；idle 则在这里记录最后一次目标。
        observer/日志异常会在物理已推进后原样传播，由上层 fail-stop 关闭路径处理。
        """

        sample_step = self.claim_completed_step() if step is None else int(step)
        if sample_step < 0:
            raise ValueError("Mirror output step 必须 >= 0")
        if write_idle_logs:
            self._write_idle_logs(step=sample_step, phase=phase)
        observe_state = getattr(self.state_observer, "observe", None)
        if callable(observe_state):
            observe_state(self, step=sample_step, phase=phase)
        observe_camera = getattr(self.camera_observer, "observe", None)
        if callable(observe_camera):
            observe_camera(self.physics, step=sample_step, phase=phase)
        return sample_step

    def reset_observation_clock(self) -> None:
        """reset 后让新 episode 的日志、状态和相机序号共同从零开始。"""

        self._completed_physics_steps = 0

    def hybrid_diagnostics(self) -> dict[str, object]:
        """返回 owner 已冻结的混合控制诊断；未绑定后端时明确报告 inactive。"""

        if self.hybrid_diagnostics_provider is None:
            return {"active": False}
        return dict(self.hybrid_diagnostics_provider())

    def _write_idle_logs(self, *, step: int, phase: str) -> None:
        """按 logger decimation 记录无新 drive update 的空闲物理步。"""

        physics_dt = float(self.physics.get_physics_dt())
        for robot in self.robots_by_id.values():
            execution = robot.execution
            logger = getattr(execution, "drive_logger", None)
            if logger is None or not logger.should_write(step):
                continue
            controller = execution.joint_controller
            targets = getattr(controller, "last_control_targets", None)
            if targets is None:
                targets = controller.build_control_targets()
            values = logger.collect_step_values(
                execution.articulation,
                controller,
                targets,
                controller.driven_indices,
            )
            logger.write(
                step=step,
                time_s=(step + 1) * physics_dt,
                phase=phase,
                drive_update=False,
                **values,
            )

    def status(self) -> dict[str, object]:
        """汇总 robot discovery、collision registry 与 planning context 状态。

        collision-aware capability 只有在 context 已 materialize 且同步到当前 scene version
        时才报告 true；读取 status 本身不会为了展示能力而创建 GPU context。
        """

        robots = []
        for robot in self.robot_registry.robots_by_id.values():
            status = robot.status()
            capability = self.planning_registry.collision_capability(robot.robot_id)
            if capability is not None:
                scene_current = (
                    capability.synced_scene_version == self.collision_registry.version
                    and not self.collision_registry.dirty
                )
                status["supports_collision_aware_planning"] = bool(
                    capability.available and scene_current
                )
                status["collision_capability"] = {
                    "robot_sphere_count": capability.robot_sphere_count,
                    "scene_checker_available": capability.scene_checker_available,
                    "required_cache": dict(capability.required_cache),
                    "configured_cache": dict(capability.configured_cache),
                    "cache_capacity_sufficient": (capability.cache_capacity_sufficient),
                    "synced_scene_version": capability.synced_scene_version,
                    "materialized_view_fingerprint": (
                        capability.materialized_view_fingerprint
                    ),
                    "missing_requirements": list(capability.missing_requirements),
                    "scene_version_current": scene_current,
                }
            robots.append(status)
        result = {
            "config_fingerprint": self.config_fingerprint,
            "robots": robots,
            "object_state": [
                {
                    "name": name,
                    "velocity_capability": view.velocity_capability,
                    "velocity_error": view.velocity_error,
                }
                for name, view in self.object_state_views.items()
            ],
            "collision": self.collision_registry.metrics(),
            "planning": self.planning_registry.metrics(),
        }
        publisher = getattr(self.camera_output, "publisher", None)
        camera_status = getattr(publisher, "status", None)
        camera_status_payload: dict[str, object] | None = None
        if callable(camera_status):
            camera_status_payload = dict(camera_status())
            result["camera_output"] = camera_status_payload
        telemetry_status = getattr(self, "telemetry_status_provider", None)
        if callable(telemetry_status):
            result["telemetry"] = dict(telemetry_status())
        return result

    @property
    def config_fingerprint(self) -> str:
        """计算 robot identity/profile/prim path 组合的 session 配置指纹。"""

        payload = {
            "robots": [
                {
                    "robot_id": robot.robot_id,
                    "label": robot.label,
                    "profile": robot.profile_name,
                    "controller_profile": robot.controller_profile,
                    "controller_profile_fingerprint": (
                        robot.controller_profile_fingerprint
                    ),
                    "profile_fingerprint": robot.profile_fingerprint,
                    "prim_path": robot.scene_instance.effective_prim_path,
                }
                for robot in self.robots_by_id.values()
            ]
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _load_controller_profiles_cached(
    names: tuple[str, ...],
    *,
    loader: Callable[[str], ControllerProfiles],
) -> tuple[ControllerProfiles, ...]:
    """按 bundle 名缓存 controller 配置，同时保留 robot list 顺序。"""

    cache: dict[str, ControllerProfiles] = {}
    result: list[ControllerProfiles] = []
    for name in names:
        if name not in cache:
            cache[name] = loader(name)
        result.append(cache[name])
    return tuple(result)


def create_mirror_scene_resources(
    *,
    scene: MirrorSceneSettings,
    session_spec: IsaacSessionSpec,
    output_settings: MirrorOutputsSettings,
    curobo_settings: CuroboProfileSettings,
    controller_bundles: Mapping[str, ControllerProfiles],
    controller_bundle: str,
    scene_solver: SolverIterationConfig | None = None,
    control_mode: str = "position",
    hybrid_control: HybridForcePositionSettings | None = None,
    cache_root: str | Path | None = None,
    hold_app: bool = False,
    status_prefix: str | None = None,
    additional_output_path_plans: Sequence[OutputPathPlan] = (),
    session_factory: Callable[..., IsaacSession] = create_isaac_session_from_spec,
) -> MirrorSceneResources:
    """围绕唯一一次 World reset 装配 object、全部 robot、sensor 和 registry。

    robot import 必须发生在 reset 前，controller finalize 和 batched handle 获取发生在 reset 后；
    任一步失败都会按资源依赖的逆序关闭已创建对象。所有输出路径先集中规划和冲突校验，
    再一次性创建目录/文件，避免场景后段校验失败时遗留部分输出。返回后，所有创建成功的
    资源都转移给 ``MirrorSceneResources``，调用方必须最终调用 ``close``。
    """

    if not isinstance(scene, MirrorSceneSettings):
        raise TypeError("scene must be MirrorSceneSettings")
    instances = robot_scene_instances_from_settings(scene.robots)
    resolved_robot_profiles = tuple(
        _resolved_robot_profile(item) for item in scene.robots
    )
    curobo_configs = tuple(
        (
            curobo_config_from_profiles(
                profile,
                curobo_settings=curobo_settings,
                cuda_device=session_spec.compute.cuda_device,
            )
            if profile.curobo.binding.enabled
            else None
        )
        for profile in resolved_robot_profiles
    )
    logging_settings = output_settings.logging
    camera_output_settings = output_settings.camera
    csv_policy = logging_settings.existing_data_policy
    csv_run_name = timestamped_run_name() if csv_policy == "timestamped_dir" else None
    csv_path_plans = {
        instance.robot_id: plan_output_file(
            path,
            policy=csv_policy,
            run_name=csv_run_name,
        )
        for instance in instances
        if (
            path := _robot_log_path(
                logging_settings,
                robot_id=instance.robot_id,
                label=instance.label,
            )
        )
        is not None
    }
    hybrid_log_path = _hybrid_control_log_path(logging_settings)
    hybrid_path_plan = (
        None
        if hybrid_log_path is None
        else plan_output_file(
            hybrid_log_path,
            policy=csv_policy,
            run_name=csv_run_name,
        )
    )
    hybrid_csv_output_plan = (
        None
        if hybrid_log_path is None or hybrid_path_plan is None
        else plan_csv_output(
            hybrid_log_path,
            hybrid_control_fieldnames(),
            existing_data_policy=csv_policy,
            timestamped_run_name=csv_run_name,
            path_plan=hybrid_path_plan,
        )
    )
    validate_output_path_plans(
        [*csv_path_plans.values()]
        + ([] if hybrid_path_plan is None else [hybrid_path_plan])
        + list(additional_output_path_plans)
    )
    settings = _mirror_scene_runtime_settings(
        scene,
        camera_output=output_settings.camera,
    )
    settings.sensors.validate_mirror_camera_scope()
    execution_configs = tuple(
        RobotExecutionConfig.from_profile(
            profile,
            scene_instance=instance,
        )
        for instance, profile in zip(instances, resolved_robot_profiles, strict=True)
    )
    runtime_object_configs = runtime_objects_from_settings(scene.objects)
    validate_disjoint_instance_prim_paths(
        robot_paths={
            instance.label: execution.robot.prim_path
            for instance, execution in zip(instances, execution_configs, strict=True)
        },
        object_paths={item.name: item.prim_path for item in runtime_object_configs},
    )
    controller_profile_names = tuple(
        resolve_controller_profile(instance, execution.robot, controller_bundle)
        for instance, execution in zip(instances, execution_configs, strict=True)
    )
    missing_controller_bundles = tuple(
        name
        for name in dict.fromkeys(controller_profile_names)
        if name not in controller_bundles
    )
    if missing_controller_bundles:
        raise ValueError(
            "resolved controller bundles are missing: "
            f"{list(missing_controller_bundles)}"
        )
    if not all(
        isinstance(controller_bundles[name], ControllerProfiles)
        for name in dict.fromkeys(controller_profile_names)
    ):
        raise TypeError("resolved controller bundles have invalid types")
    controller_profiles = _load_controller_profiles_cached(
        controller_profile_names,
        loader=controller_bundles.__getitem__,
    )

    session = session_factory(spec=session_spec)
    physics = MirrorPhysicsAdapter(session.physics_runtime)
    newton_runtime = session_spec.physics.kind in {"newton_cpu", "newton_cuda"}
    prepare_newton_render_topology = newton_runtime and bool(
        session_spec.render.enabled
    )
    camera_output: CameraOutputHandle | None = None
    sensor_cameras: tuple[SensorCameraRuntime, ...] = ()
    loggers: list[JointTrackingLogger] = []
    hybrid_control_logger: HybridControlLogger | None = None
    planning_registry: RobotPlanningRegistry | None = None
    try:
        # 灯光是 Mirror 场景事实，必须先于 camera render product 写入 stage；否则 depth
        # 仍可能有效而 RGB 只得到全黑图像。headless 相机不需要默认 GUI viewport，因此
        # viewport 更新严格跟随 app.gui，不能因启用了离屏渲染就隐式创建交互视角。
        if session_spec.render.enabled:
            configure_visuals(
                settings.visuals,
                configure_viewport=session_spec.app.gui,
            )
        if newton_runtime:
            # 先只创建 camera prim/viewport，让 Hydra engine 在场景尚无 imported instancer
            # 时完成初始化。SyntheticData sensor 仍由下方统一 initialize，严格晚于
            # Newton finalize/reset 与 object view 绑定。
            sensor_cameras = create_sensor_camera_runtimes(
                stage=session.stage,
                sensors=settings.sensors,
                physics_backend=session.physics_runtime.backend,
            )
        # Newton 的唯一模型必须包含完整 /World 拓扑。对象和所有机器人先导入 USD，
        # 再一次性 finalize manager；不能边导入边绑定 articulation，
        # 否则前一个 view 会引用尚未包含后续资产的旧 model/state 缓冲区。
        object_handles = add_runtime_objects(
            session.stage,
            runtime_object_configs,
            physics_backend=session.physics_runtime.backend,
            prepare_newton_render_topology=prepare_newton_render_topology,
            status_prefix=status_prefix,
        )
        objects = runtime_object_handles_by_name(object_handles)
        imported = tuple(
            import_execution_robot_to_stage(
                world=physics,
                stage=session.stage,
                single_articulation_type=session.single_articulation_type,
                robot_execution=execution,
                controller_profiles=profiles,
                scene_solver=scene_solver,
                physics_backend=session.physics_runtime.backend,
                prepare_newton_render_topology=prepare_newton_render_topology,
                defer_articulation_binding=newton_runtime,
            )
            for execution, profiles in zip(
                execution_configs, controller_profiles, strict=True
            )
        )

        physical_tcp_bindings = (
            tuple(
                resolve_physical_tcp_binding(
                    stage=session.stage,
                    imported_root_path=str(imported_robot.imported_root_path),
                    profile=profile,
                )
                for imported_robot, profile in zip(
                    imported, resolved_robot_profiles, strict=True
                )
            )
            if hybrid_control is not None
            else tuple(None for _ in imported)
        )

        if newton_runtime:
            imported = _initialize_newton_runtime_mirror(
                session=session,
                physics=physics,
                instances=instances,
                imported=imported,
                execution_configs=execution_configs,
                object_handles=object_handles,
            )

        if not newton_runtime:
            sensor_cameras = create_sensor_camera_runtimes(
                stage=session.stage,
                sensors=settings.sensors,
                physics_backend=session.physics_runtime.backend,
            )
        physics.reset()
        set_physics_gravity(session.physics_runtime, settings.gravity_z)
        object_state_views = create_scene_object_state_views(
            object_handles,
            physics_backend=session.physics_runtime.backend,
            stage=session.stage,
            immutable_static=newton_runtime,
        )
        # Newton 此时才激活早先预留 viewport 的 SyntheticData sensor；PhysX camera
        # 仍由其构造器完成 legacy 初始化。
        initialize_sensor_camera_runtimes(sensor_cameras)

        prepared_entries: list[tuple[Any, ...]] = []
        for (
            instance,
            profile,
            curobo_config,
            controller_profile_name,
            profiles,
            imported_robot,
        ) in zip(
            instances,
            resolved_robot_profiles,
            curobo_configs,
            controller_profile_names,
            controller_profiles,
            imported,
            strict=True,
        ):
            prepared = finalize_robot_controller(
                imported=imported_robot,
                controller_profiles=profiles,
                control_mode=control_mode,
            )
            command_names = tuple(prepared.joint_controller.command_joint_names)
            kind = profile.kind
            binding = profile.curobo.binding
            planning_names = (
                planning_joint_names_from_profile(profile) if binding.enabled else ()
            )
            declared_groups = profile.joint_groups
            groups = JointGroupLayout(
                command_joint_names=command_names,
                arm=declared_groups.arm,
                hand=declared_groups.hand,
                passive=declared_groups.passive,
            )
            groups.validate_kind(kind)
            if binding.enabled:
                groups.validate_planning_joints(planning_names)
            capability = PlanningCapability(
                kind=kind,
                backend_enabled=binding.enabled,
                planning_joint_group=binding.planning_joint_group,
                kinematics_binding_valid=(
                    not binding.enabled or curobo_config is not None
                ),
                arm_joint_mapping_valid=(
                    not binding.enabled or set(planning_names) == set(groups.arm)
                ),
            )
            driven_names = _robot_logger_joint_names(
                articulation=prepared.articulation,
                controller=prepared.joint_controller,
            )
            log_path = _robot_log_path(
                logging_settings,
                robot_id=instance.robot_id,
                label=instance.label,
            )
            csv_output_plan = (
                None
                if log_path is None
                else plan_csv_output(
                    log_path,
                    joint_tracking_fieldnames(driven_names, logging_settings),
                    existing_data_policy=csv_policy,
                    timestamped_run_name=csv_run_name,
                    path_plan=csv_path_plans[instance.robot_id],
                )
            )
            prepared_entries.append(
                (
                    instance,
                    profile,
                    controller_profile_name,
                    imported_robot,
                    prepared,
                    kind,
                    groups,
                    capability,
                    curobo_config,
                    csv_output_plan,
                )
            )

        csv_output_plans = tuple(
            entry[-1]
            for entry in prepared_entries
            if isinstance(entry[-1], CsvOutputPlan)
        )
        prepared_camera_output = prepare_camera_output(
            sensor_cameras,
            path_resolver=repo_path,
            settings=camera_output_settings,
            shutdown_timeout_s=camera_output_settings.shutdown_timeout_s,
        )
        apply_output_path_plans(
            [plan.path_plan for plan in csv_output_plans]
            + (
                []
                if hybrid_csv_output_plan is None
                else [hybrid_csv_output_plan.path_plan]
            )
            + list(prepared_camera_output.path_plans)
            + list(additional_output_path_plans)
        )
        if hybrid_log_path is not None and hybrid_csv_output_plan is not None:
            hybrid_control_logger = HybridControlLogger(
                hybrid_log_path,
                settings=logging_settings,
                physics_dt=physics.get_physics_dt(),
                timestamped_run_name=csv_run_name,
                output_plan=hybrid_csv_output_plan,
                paths_applied=True,
            )
        camera_output = open_prepared_camera_output(prepared_camera_output)

        robots: list[RobotRuntime] = []
        for (
            instance,
            profile,
            controller_profile_name,
            imported_robot,
            prepared,
            kind,
            groups,
            capability,
            curobo_config,
            csv_output_plan,
        ), physical_tcp_binding in zip(
            prepared_entries,
            physical_tcp_bindings,
            strict=True,
        ):
            logger = _make_robot_logger(
                logging_settings,
                robot_id=instance.robot_id,
                label=instance.label,
                articulation=prepared.articulation,
                controller=prepared.joint_controller,
                physics_dt=physics.get_physics_dt(),
                timestamped_run_name=csv_run_name,
                output_plan=(
                    csv_output_plan
                    if isinstance(csv_output_plan, CsvOutputPlan)
                    else None
                ),
                paths_applied=True,
            )
            loggers.append(logger)
            execution = ExecutionRuntime(
                articulation=prepared.articulation,
                simulation_world=physics,
                articulation_action_type=session.articulation_action_type,
                joint_controller=prepared.joint_controller,
                simulation_app=session.app if hold_app else None,
                render_enabled=(
                    session_spec.app.gui
                    or camera_output is not None
                    or (newton_runtime and bool(sensor_cameras))
                ),
                drive_logger=logger,
                camera_observer=(
                    None if camera_output is None else camera_output.observer
                ),
            )
            robots.append(
                RobotRuntime(
                    robot_id=instance.robot_id,
                    label=instance.label,
                    kind=kind,
                    profile_name=instance.robot_profile,
                    controller_profile=controller_profile_name,
                    profile_config=profile,
                    scene_instance=instance,
                    imported=imported_robot,
                    prepared=prepared,
                    execution=execution,
                    joint_groups=groups,
                    planning_capability=capability,
                    curobo_config=curobo_config,
                    controller_profiles=controller_bundles[controller_profile_name],
                    physical_tcp_binding=physical_tcp_binding,
                    task_space_port=(
                        None
                        if physical_tcp_binding is None
                        else PhysxTaskSpacePort(
                            prepared.articulation,
                            session.stage,
                            list(groups.arm),
                            physical_tcp_binding,
                        )
                    ),
                )
            )

        registry = RobotRegistry(tuple(robots))
        collision_registry = SceneCollisionRegistry()
        collision_registry.register_runtime_objects(
            object_handles,
            stage=session.stage,
            state_views=object_state_views,
        )
        for robot in robots:
            provider = RobotObstacleProvider.from_robot_profile(
                robot_id=robot.robot_id,
                label=robot.label,
                articulation=robot.articulation,
                root_pose=robot.scene_instance.root_pose,
                profile=robot.profile_config,
            )
            if provider is not None:
                collision_registry.register_provider(
                    f"robot:{robot.robot_id}",
                    provider,
                    owner_robot_id=robot.robot_id,
                    source="robot",
                )
        planning_registry = RobotPlanningRegistry(registry, cache_root=cache_root)
        runtime = MirrorSceneResources(
            session=session,
            physics=physics,
            scene=scene,
            robot_registry=registry,
            planning_registry=planning_registry,
            collision_registry=collision_registry,
            runtime_object_configs=runtime_object_configs,
            object_handles=object_handles,
            objects=objects,
            object_state_views=object_state_views,
            sensor_cameras=sensor_cameras,
            camera_output=camera_output,
            camera_observer=(None if camera_output is None else camera_output.observer),
            loggers=tuple(loggers),
            hybrid_control_logger=hybrid_control_logger,
            status_prefix=status_prefix,
        )
        _print_mirror_status(runtime)
        return runtime
    except BaseException as exc:
        traceback.print_exception(exc)
        sys.stderr.flush()
        if status_prefix is not None:
            print(
                f"{status_prefix}_BUILD_FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )
        _cleanup_failed_mirror_runtime(
            planning_registry=planning_registry,
            camera_output=camera_output,
            sensor_cameras=sensor_cameras,
            loggers=(
                *loggers,
                *(() if hybrid_control_logger is None else (hybrid_control_logger,)),
            ),
            session=session,
        )
        raise


def _initialize_newton_runtime_mirror(
    *,
    session: IsaacSession,
    physics: MirrorPhysicsAdapter,
    instances: Sequence[object],
    imported: Sequence[object],
    execution_configs: Sequence[RobotExecutionConfig],
    object_handles: Sequence[RuntimeObjectHandle],
) -> tuple[object, ...]:
    """先 finalize 唯一 ``/World`` model，再绑定全部 Newton articulation view。

    Mirror 的单场景可以包含多个机器人，但只能有一个 Newton world。所有 USD 资产必须在
    finalize 前到齐，否则先创建的 view 会借用缺失后续资产的旧 model/state storage。
    """

    if not imported:
        raise RuntimeError("Newton Mirror requires at least one robot")
    if len(instances) != len(imported) or len(imported) != len(execution_configs):
        raise RuntimeError("Newton Mirror robot metadata is incomplete")
    manager = session.physics_runtime
    initialize = getattr(manager, "initialize_worlds", None)
    if not callable(initialize):
        raise RuntimeError(
            "Newton physics manager does not provide scene initialization"
        )
    robots = {
        str(instance.label): imported_robot
        for instance, imported_robot in zip(instances, imported, strict=True)
    }
    initialize(
        env_root_paths=("/World",),
        env_origins=((0.0, 0.0, 0.0),),
        robots=robots,
        object_handles=object_handles,
    )
    if int(getattr(manager, "world_count", -1)) != 1:
        raise RuntimeError(
            "Newton Mirror must finalize exactly one world; "
            f"actual={getattr(manager, 'world_count', None)!r}"
        )
    return tuple(
        bind_imported_robot_articulation(
            imported_robot,
            world=physics,
            single_articulation_type=session.single_articulation_type,
            name=execution.robot.name,
        )
        for imported_robot, execution in zip(imported, execution_configs, strict=True)
    )


def _cleanup_failed_mirror_runtime(
    *,
    planning_registry: object | None,
    camera_output: object | None,
    sensor_cameras: Sequence[object],
    loggers: Sequence[object],
    session: IsaacSession,
) -> None:
    """尽力回滚启动期间已创建资源，且绝不覆盖原始创建异常。

    close 回调按依赖逆序执行；每个回调独立吞掉清理异常，以保证后续资源仍有机会释放。
    原始异常由调用方的 ``except`` 块原样重新抛出。
    """

    callbacks: list[Callable[[], object]] = []
    planning_close = getattr(planning_registry, "close", None)
    if callable(planning_close):
        callbacks.append(planning_close)
    camera_close = getattr(camera_output, "close", None)
    if callable(camera_close):
        callbacks.append(camera_close)
    callbacks.extend(
        close
        for camera in sensor_cameras
        if callable(close := getattr(camera, "close", None))
    )
    callbacks.extend(
        close for logger in loggers if callable(close := getattr(logger, "close", None))
    )
    # 失败回滚也必须由 canonical session 关闭物理 runtime 与 App，不能绕开 owner。
    callbacks.append(lambda: session.close(exit_code=1))
    for callback in callbacks:
        try:
            callback()
        except BaseException:
            pass


def planning_joint_names_from_profile(
    profile: RobotProfileSettings,
) -> tuple[str, ...]:
    """从 typed robot model 读取 active joints，不分配 cuRobo context。"""

    if not isinstance(profile, RobotProfileSettings):
        raise TypeError("profile must be RobotProfileSettings")
    robot = profile.curobo.robot
    if robot is None:
        return ()
    config_path = robot.robot_config_path
    if config_path is not None:
        data = load_yaml(config_path)
        robot_cfg = data.get("robot_cfg")
        if isinstance(robot_cfg, Mapping):
            kinematics = robot_cfg.get("kinematics")
            if isinstance(kinematics, Mapping):
                cspace = kinematics.get("cspace")
                if isinstance(cspace, Mapping):
                    values = cspace.get("joint_names", ())
                    return tuple(str(value) for value in values)
    urdf_path = robot.urdf_path
    if urdf_path is None:
        return ()
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    return tuple(
        str(joint.attrib["name"])
        for joint in root.findall("joint")
        if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}
    )


def _make_robot_logger(
    settings: LoggingOutputSettings,
    *,
    robot_id: int,
    label: str,
    articulation: object,
    controller: object,
    physics_dt: float,
    timestamped_run_name: str | None,
    output_plan: CsvOutputPlan | None = None,
    paths_applied: bool = False,
) -> JointTrackingLogger:
    """按 driven joint order 创建单 robot logger，并计算 flush step 周期。"""

    driven_names = _robot_logger_joint_names(
        articulation=articulation,
        controller=controller,
    )
    path = _robot_log_path(settings, robot_id=robot_id, label=label)
    return JointTrackingLogger(
        path,
        driven_names,
        settings=settings,
        flush_interval_steps=settings.flush_interval_steps(physics_dt),
        timestamped_run_name=timestamped_run_name,
        output_plan=output_plan,
        paths_applied=paths_applied,
    )


def _robot_logger_joint_names(
    *,
    articulation: object,
    controller: object,
) -> list[str]:
    """按 controller driven index 解析关节名，不提前打开 CSV writer。

    路径规划阶段需要精确表头来校验已有文件，但此时还不能产生文件副作用，因此这里只做
    纯索引映射。
    """

    indices = [int(index) for index in controller.driven_indices]
    dof_names = tuple(str(name) for name in articulation.dof_names)
    return [dof_names[index] for index in indices]


def _robot_log_path(
    settings: LoggingOutputSettings,
    *,
    robot_id: int,
    label: str,
) -> Path | None:
    """把共享 logging path 派生为包含 robot ID 与 label 的独立文件。"""

    if not settings.enabled or settings.joint_tracking_path is None:
        return None
    path = repo_path(settings.joint_tracking_path)
    return path.with_name(f"{path.stem}.{robot_id}.{label}{path.suffix}")


def _hybrid_control_log_path(settings: LoggingOutputSettings) -> Path | None:
    """解析单一 hybrid diagnostics CSV；禁用时不触碰配置路径。"""

    if not settings.enabled or not settings.log_hybrid_control:
        return None
    if settings.hybrid_control_path is None:
        raise RuntimeError("enabled hybrid control logging has no output path")
    return repo_path(settings.hybrid_control_path)


def _resolved_robot_profile(instance: RobotInstanceSettings) -> RobotProfileSettings:
    """返回 catalog 已绑定的 typed robot profile，缺失时立即失败。"""

    profile = instance.resolved_profile
    if not isinstance(profile, RobotProfileSettings):
        raise RuntimeError(
            f"Mirror robot {instance.label!r} 缺少 catalog 绑定的 resolved profile"
        )
    return profile


def _mirror_scene_runtime_settings(
    scene: MirrorSceneSettings,
    *,
    camera_output: CameraOutputSettings,
) -> MirrorSceneRuntimeSettings:
    """把 Mirror scene/output typed facts 组合为资产与 sensor runtime 设置。"""

    settings = MirrorSceneRuntimeSettings(
        physics_frequency=scene.physics_frequency_hz,
        render_frequency=scene.render_frequency_hz,
        gravity_z=scene.gravity_z,
        add_ground=scene.add_ground,
        ground_height=scene.ground_height,
        visuals=_scene_visual_settings(scene),
        sensors=SceneSensorSettings(
            cameras=tuple(
                _sensor_camera_settings(camera, output=camera_output)
                for camera in scene.cameras
            )
        ),
    )
    settings.validate()
    return settings


def _scene_visual_settings(scene: MirrorSceneSettings) -> SceneVisualSettings:
    """按 key/fill 的稳定语义把 typed 灯光投影到通用 visual settings。"""

    lights = {light.light_id: light for light in scene.lights}
    unsupported = sorted(set(lights) - {"key", "fill"})
    if unsupported:
        raise ValueError(f"Mirror scene contains unsupported light IDs: {unsupported}")
    key = lights.get("key")
    fill = lights.get("fill")
    if fill is not None and fill.angle is not None:
        raise ValueError("Mirror fill light cannot declare angle")
    key_settings = (
        DistantLightSettings()
        if key is None
        else DistantLightSettings(
            path=key.path,
            intensity=key.intensity,
            angle=DistantLightSettings().angle if key.angle is None else key.angle,
        )
    )
    fill_settings = (
        DomeLightSettings()
        if fill is None
        else DomeLightSettings(
            path=fill.path,
            intensity=fill.intensity,
        )
    )
    return SceneVisualSettings(
        viewport=scene.viewport,
        key_light=key_settings,
        fill_light=fill_settings,
    )


def _sensor_camera_settings(
    camera: CameraSettings,
    *,
    output: CameraOutputSettings,
) -> SensorCameraSettings:
    """把已校验的 scene camera 与全局 sink 直接组合为 runtime camera。"""

    intrinsics = camera.intrinsics
    return SensorCameraSettings(
        name=camera.camera_id,
        prim_path=camera.prim_path,
        parent_prim_path=camera.parent_prim_path,
        pose_xyz=camera.pose.xyz,
        pose_rpy=camera.pose.rpy,
        resolution=camera.resolution,
        frequency=camera.frequency_hz,
        modalities=camera.modalities,
        clipping_range=camera.clipping_range_m,
        intrinsics=(
            None
            if intrinsics is None
            else SensorCameraIntrinsicsSettings(
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                cx=intrinsics.cx,
                cy=intrinsics.cy,
            )
        ),
        output=_camera_output_settings(camera.camera_id, output),
    )


def _mirror_session_spec(config: MirrorConfig) -> IsaacSessionSpec:
    """把 strict Mirror config 单向投影为产品无关 Isaac session 规格。"""

    render = config.outputs.render
    physics: IsaacPhysxCpuSpec | IsaacNewtonCpuSpec | IsaacNewtonCudaSpec
    if isinstance(config.physics, PhysxCpuSettings):
        physics = IsaacPhysxCpuSpec()
    elif isinstance(config.physics, NewtonCpuSettings):
        source = config.physics
        physics = IsaacNewtonCpuSpec(
            # Mirror 的 world 数是产品语义，不在共享 physics leaf 中重复配置。
            world_count=1,
            nconmax_per_world=source.nconmax_per_world,
            njmax_per_world=source.njmax_per_world,
            substeps=source.substeps,
            iterations=source.iterations,
            line_search_iterations=source.line_search_iterations,
            constraint_solver=source.constraint_solver,
            contact_pipeline=source.contact_pipeline,
        )
    else:
        source = config.physics
        assert isinstance(source, NewtonCudaSettings)
        physics = IsaacNewtonCudaSpec(
            # Mirror 的 world 数是产品语义，不在共享 physics leaf 中重复配置。
            world_count=1,
            nconmax_per_world=source.nconmax_per_world,
            njmax_per_world=source.njmax_per_world,
            use_cuda_graph=source.use_cuda_graph,
            substeps=source.substeps,
            iterations=source.iterations,
            line_search_iterations=source.line_search_iterations,
            constraint_solver=source.constraint_solver,
            contact_pipeline=source.contact_pipeline,
        )
    render_resources = bool(
        render.enabled
        or render.gui
        or config.outputs.camera.enabled
        or config.scene.cameras
    )
    return IsaacSessionSpec(
        experience_family="mirror",
        compute=IsaacComputeSpec(cuda_device=config.compute.cuda_device),
        physics=physics,
        physics_dt=1.0 / config.scene.physics_frequency_hz,
        rendering_dt=1.0 / config.scene.render_frequency_hz,
        gravity_z=config.scene.gravity_z,
        add_ground=config.scene.add_ground,
        ground_height=config.scene.ground_height,
        app=IsaacAppSpec(gui=render.gui),
        render=IsaacRenderSpec(
            enabled=render_resources,
            width=render.width,
            height=render.height,
            window_width=render.width,
            window_height=render.height,
            renderer=render.renderer,
            samples_per_pixel_per_frame=render.samples_per_pixel_per_frame,
        ),
    )


def build_mirror_assembly(config: MirrorConfig) -> object:
    """从 strict config 创建 Mirror 的完整 engine-aware 资源图。

    构造成功后，session 与各资源的 close 权由 ``MirrorRuntime`` 接管；本函数返回的
    ``MirrorSceneResources`` 只是状态/时间线访问集合，不是第二个 session owner。
    """

    from linkerbot_sim.mirror.bootstrap import MirrorAssembly

    telemetry = config.outputs.telemetry
    telemetry_plan = (
        prepare_mcap_output(
            telemetry.mcap_path,
            existing_file_policy=telemetry.existing_data_policy,
        )
        if telemetry.enabled
        else None
    )
    for instance in config.scene.robots:
        _resolved_robot_profile(instance)

    for instance in config.scene.objects:
        profile = instance.resolved_profile
        if profile is None:
            raise RuntimeError(
                f"Mirror object {instance.name!r} 缺少 catalog 绑定的 resolved profile"
            )

    resources = create_mirror_scene_resources(
        scene=config.scene,
        session_spec=_mirror_session_spec(config),
        output_settings=config.outputs,
        curobo_settings=config.curobo,
        controller_bundles=config.controller_bundles,
        controller_bundle=config.default_controller_bundle,
        scene_solver=(
            SolverIterationConfig(solver_type=config.physics.solver_type)
            if isinstance(config.physics, PhysxCpuSettings)
            else None
        ),
        control_mode=config.control.mode,
        hybrid_control=config.hybrid_control,
        hold_app=False,
        status_prefix="MIRROR",
        additional_output_path_plans=(
            () if telemetry_plan is None else (telemetry_plan,)
        ),
    )

    try:
        _apply_scene_planning_startup(
            resources,
            coordination=config.planning.request_defaults.coordination,
        )
        state_stream = start_interactive_state_stream(
            resources,
            config=_state_stream_config(telemetry, mcap_plan=telemetry_plan),
            status_prefix="MIRROR",
        )
    except BaseException:
        _cleanup_failed_mirror_runtime(
            planning_registry=resources.planning_registry,
            camera_output=resources.camera_output,
            sensor_cameras=resources.sensor_cameras,
            loggers=(
                *resources.loggers,
                *(
                    ()
                    if resources.hybrid_control_logger is None
                    else (resources.hybrid_control_logger,)
                ),
            ),
            session=resources.session,
        )
        raise

    def capture() -> dict[str, object]:
        return get_mirror_snapshot(resources).as_dict()

    def restore(
        snapshot: Mapping[str, object],
        *,
        label_map: Mapping[str, str] | None = None,
        strict: bool = True,
    ) -> object:
        return set_mirror_snapshot(
            resources,
            snapshot,
            label_map=label_map,
            strict=strict,
        )

    try:
        motion = MirrorTimelineBackend(resources, config=config)
        resources.hybrid_diagnostics_provider = motion.hybrid_diagnostics
    except BaseException:
        if state_stream is not None:
            state_stream.close()
        _cleanup_failed_mirror_runtime(
            planning_registry=resources.planning_registry,
            camera_output=resources.camera_output,
            sensor_cameras=resources.sensor_cameras,
            loggers=(
                *resources.loggers,
                *(
                    ()
                    if resources.hybrid_control_logger is None
                    else (resources.hybrid_control_logger,)
                ),
            ),
            session=resources.session,
        )
        raise

    def resetter(*, hold_after_reset: bool = True) -> dict[str, object]:
        result = asdict(
            reset_mirror_scene(
                resources,
                options=MirrorResetOptions(hold_after_reset=hold_after_reset),
            )
        )
        step = motion.after_scene_reset(
            hold_duration_s=(
                config.control.idle_step_duration_s if hold_after_reset else None
            )
        )
        result["step"] = step
        result["message"] = (
            "runtime reset completed; "
            f"hold_after_reset={bool(hold_after_reset)}; step={step}"
        )
        return result

    controllers = tuple(
        robot.execution.joint_controller for robot in resources.robots_by_id.values()
    )
    control_bindings = tuple(
        MirrorControlBinding(
            label=robot.label,
            controller=robot.execution.joint_controller,
            controller_profiles=robot.controller_profiles,
            articulation_action_type=robot.execution.articulation_action_type,
        )
        for robot in resources.robots_by_id.values()
        if robot.controller_profiles is not None
    )
    if len(control_bindings) != len(resources.robots_by_id):
        raise RuntimeError("Mirror robot controller profile bindings are incomplete")
    return MirrorAssembly(
        session=resources.session,
        state_getter=capture,
        state_setter=lambda state, strict=True: restore(state, strict=strict),
        snapshot_capture=capture,
        snapshot_restore=restore,
        resetter=resetter,
        motion_backend=motion,
        collision_registry=resources.collision_registry,
        cameras=tuple(resources.sensor_cameras),
        camera_output=resources.camera_output,
        outputs=(
            *resources.loggers,
            *(
                ()
                if resources.hybrid_control_logger is None
                else (resources.hybrid_control_logger,)
            ),
            *(() if state_stream is None else (state_stream,)),
        ),
        controllers=controllers,
        control_bindings=control_bindings,
        views=tuple(resources.object_state_views.values()),
        scene_resources=resources,
    )


def _apply_scene_planning_startup(
    resources: MirrorSceneResources,
    *,
    coordination: CoordinationPolicy,
) -> tuple[int, ...]:
    """在 interactive 资源启动前执行 scene 声明的 planning 初始化策略。"""

    policy = resources.scene.planning_startup
    if policy == "lazy":
        return ()
    if policy != "prewarm":
        raise RuntimeError(f"unsupported Mirror planning startup policy: {policy!r}")
    snapshot = resources.collision_registry.snapshot()
    return resources.planning_registry.prewarm_interactive_planners(
        snapshot,
        coordination=coordination,
    )


def _camera_output_settings(
    camera_id: str,
    settings: CameraOutputSettings,
) -> SensorCameraOutputSettings:
    """把全局 camera sink 配置派生为一个相机的 typed 输出设置。"""

    if not settings.enabled:
        return SensorCameraOutputSettings()
    save_dir = (
        None
        if settings.save_root is None
        else str(Path(settings.save_root) / camera_id)
    )
    return SensorCameraOutputSettings(
        save_dir=save_dir,
        foxglove_topic_prefix=f"/cameras/{camera_id}",
        foxglove_live_host=settings.foxglove_live_host,
        foxglove_live_port=settings.foxglove_live_port,
        foxglove_mcap_path=settings.foxglove_mcap_path,
    )


def _state_stream_config(
    settings: TelemetryOutputSettings,
    *,
    mcap_plan: OutputPathPlan | None,
) -> InteractiveStateStreamConfig | None:
    """把 pure telemetry config 投影到 Mirror 状态流，不引入第二份默认值。"""

    if not settings.enabled:
        return None
    return InteractiveStateStreamConfig(
        rate_hz=settings.rate_hz,
        buffer_size=settings.buffer_size,
        drop_policy=settings.drop_policy,
        on_error=settings.on_error,
        include_joint_states=settings.include_joint_states,
        include_state_json=settings.include_state_json,
        include_scene_markers=settings.include_scene_markers,
        include_efforts=settings.include_efforts,
        include_objects=settings.include_objects,
        include_hybrid_control=settings.include_hybrid_control,
        topics=FoxgloveTopicConfig(
            joint_states=settings.topics.joint_states,
            scene=settings.topics.scene,
            state=settings.topics.state,
            hybrid_control=settings.topics.hybrid_control,
        ),
        foxglove_live_host=settings.foxglove_live_host,
        foxglove_live_port=settings.foxglove_live_port,
        foxglove_mcap_path=settings.mcap_path,
        mcap_existing_file_policy=settings.existing_data_policy,
        mcap_output_plan=mcap_plan,
        output_paths_applied=True,
        foxglove_joint_effort_field=(
            settings.joint_effort_field if settings.include_efforts else "none"
        ),  # type: ignore[arg-type]
        shutdown_timeout_s=settings.shutdown_timeout_s,
    )


def _print_mirror_status(runtime: MirrorSceneResources) -> None:
    """按 robot ID 输出稳定的启动状态行，供脚本和集成测试识别。"""

    if runtime.status_prefix is None:
        return
    for robot in runtime.robots_by_id.values():
        print(
            f"{runtime.status_prefix}_ROBOT "
            f"robot_id={robot.robot_id} label={robot.label} "
            f"profile={robot.profile_name} kind={robot.kind.value} "
            f"supports_planning={str(robot.supports_planning).lower()}",
            flush=True,
        )


__all__ = [
    "MirrorSceneResources",
    "build_mirror_assembly",
    "create_mirror_scene_resources",
    "planning_joint_names_from_profile",
]
