"""任意数量机器人 Isaac 场景的统一装配与生命周期所有者。

创建顺序固定为启动 session、导入对象/机器人、一次 world reset、finalize controller、构建
registry/collision provider；失败时按相反顺序释放已创建资源。``SingleSceneRuntime`` 是普通交互、
snapshot、telemetry 和 timeline executor 的共同根对象，不包含单/双机器人分支。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from linkerbot_sim.objects.runtime import (
    RuntimeObjectHandle,
    add_runtime_objects,
    runtime_object_handles_by_name,
    runtime_objects_from_env_config,
)
from linkerbot_sim.objects.state_views import (
    SceneObjectStateView,
    create_scene_object_state_views,
)
from linkerbot_sim.app.runtime.robot_registry import (
    RobotPlanningRegistry,
    RobotRegistry,
    RobotRuntime,
)
from linkerbot_sim.app.runtime.collision.registry import SceneCollisionRegistry
from linkerbot_sim.app.runtime.collision.robot_provider import RobotObstacleProvider
from linkerbot_sim.envs.settings import EnvRuntimeSettings
from linkerbot_sim.app.runtime.simulation_app_lifecycle import close_simulation_app
from linkerbot_sim.app.runtime.simulation_session import (
    SimulationSession,
    create_simulation_session,
)
from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    robot_instances_from_env_config,
    resolve_controller_profile,
)
from linkerbot_sim.backends.curobo.config import CuroboConfig
from linkerbot_sim.backends.curobo.profile_merge import (
    merged_robot_config_with_curobo_profile,
)
from linkerbot_sim.configs.instance_paths import validate_disjoint_instance_prim_paths
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.configs.runtime import (
    CameraOutputRuntimeSettings,
    OutputPolicySettings,
    ShutdownSettings,
    SimulationAppSettings,
)
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.controllers.config import (
    ControllerProfiles,
    load_controller_bundle,
)
from linkerbot_sim.execution.setup import (
    finalize_robot_controller,
    import_execution_robot_to_stage,
)
from linkerbot_sim.logging.config import (
    JointLoggingConfig,
    joint_logging_config_from_mapping,
)
from linkerbot_sim.logging.csv_writer import (
    CsvOutputPlan,
    plan_csv_output,
)
from linkerbot_sim.logging.joint_logger import (
    JointTrackingLogger,
    joint_tracking_fieldnames,
)
from linkerbot_sim.robots.capabilities import (
    PlanningBindingConfig,
    PlanningCapability,
    robot_kind_from_profile,
)
from linkerbot_sim.robots.joint_groups import JointGroupLayout
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
from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.output_paths import (
    OutputPathPlan,
    apply_output_path_plans,
    plan_output_file,
    timestamped_run_name,
    validate_output_path_plans,
)
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class SingleSceneRuntimeShutdownReport:
    """一次有界场景关闭尝试的结果。

    ``stopped=False`` 表示仍有异步资源存活；这些资源仍由 ``SingleSceneRuntime`` 持有，可在后续
    ``close`` 调用中重试，不能据此释放进程级 SimulationApp。
    """

    stopped: bool
    live_resources: tuple[str, ...] = ()


@dataclass
class SingleSceneRuntime:
    """拥有一个 SimulationSession 和任意数量 robot articulation 的根 runtime。

    该对象集中持有 object、sensor、logger、collision registry 与 planning context pool，
    因而也是 snapshot、telemetry、timeline executor 的资源生命周期边界。
    """

    session: SimulationSession
    env_config: Mapping[str, object]
    robot_registry: RobotRegistry
    planning_registry: RobotPlanningRegistry
    collision_registry: SceneCollisionRegistry
    object_handles: tuple[RuntimeObjectHandle, ...] = ()
    objects: Mapping[str, RuntimeObjectHandle] = field(default_factory=dict)
    object_state_views: Mapping[str, SceneObjectStateView] = field(default_factory=dict)
    sensor_cameras: tuple[SensorCameraRuntime, ...] = ()
    camera_output: CameraOutputHandle | None = None
    loggers: tuple[JointTrackingLogger, ...] = ()
    status_prefix: str | None = None
    shutdown_resources: dict[str, object] = field(default_factory=dict, repr=False)
    _closed: bool = False
    _planning_closed: bool = False
    _camera_closed: bool = False
    _loggers_closed: bool = False
    _closed_logger_indices: set[int] = field(default_factory=set, repr=False)
    _app_closed: bool = False

    @property
    def robots_by_id(self) -> dict[int, RobotRuntime]:
        """返回 session robot ID 到 ``RobotRuntime`` 的主索引。"""

        return self.robot_registry.robots_by_id

    @property
    def robot_id_by_label(self) -> dict[str, int]:
        """返回稳定 label 到本次 session ID 的反向索引。"""

        return self.robot_registry.robot_id_by_label

    @property
    def world(self):
        """暴露 session 唯一的 Isaac World。"""

        return self.session.world

    def robot(self, robot_id: int) -> RobotRuntime:
        """按 session ID 解析 robot。"""

        return self.robot_registry.robot(robot_id)

    def robot_by_label(self, label: str) -> RobotRuntime:
        """按稳定 label 解析 robot，供内部 snapshot/registry 逻辑使用。"""

        return self.robot_registry.robot_by_label(label)

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
        live_resources = set(getattr(self, "shutdown_resources", {}))
        if camera_status_payload is not None and bool(
            camera_status_payload.get("shutdown_timed_out", False)
        ):
            live_resources.add("camera_output")
        result["shutdown"] = {
            "closed": bool(getattr(self, "_closed", False)),
            "live_resources": sorted(live_resources),
        }
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
                    "profile_fingerprint": robot.profile_fingerprint,
                    "prim_path": robot.scene_instance.effective_prim_path,
                }
                for robot in self.robots_by_id.values()
            ]
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def retain_shutdown_resource(self, name: str, resource: object) -> None:
        """保留关闭超时的异步资源，使后续 ``close`` 可以重试回收。

        ``name`` 同时作为 ``status().shutdown.live_resources`` 的诊断身份；同名登记会用最新
        句柄替换旧值，因此调用方必须保证名称在 runtime 内唯一。
        """

        self.shutdown_resources[str(name)] = resource

    def close(self) -> SingleSceneRuntimeShutdownReport:
        """按依赖逆序有界关闭 runtime，并保留超时资源所有权。

        关闭顺序为额外异步资源、planning contexts、camera publisher、CSV logger，最后才是
        进程级 SimulationApp。任一子资源仍存活时不会提前关闭 app；可重试的成功阶段通过
        内部标志记录，后续调用不会重复关闭。同步关闭异常在尽量处理其他资源后重新抛出。
        """

        if self._closed:
            return SingleSceneRuntimeShutdownReport(stopped=True)
        live_resources: list[str] = []
        first_error: BaseException | None = None
        for name, resource in tuple(self.shutdown_resources.items()):
            try:
                stopped = _stop_retained_shutdown_resource(resource)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                live_resources.append(name)
                continue
            if stopped:
                self.shutdown_resources.pop(name, None)
            else:
                live_resources.append(name)
        if live_resources and first_error is None:
            return SingleSceneRuntimeShutdownReport(
                stopped=False,
                live_resources=tuple(sorted(live_resources)),
            )
        if not self._planning_closed:
            try:
                self.planning_registry.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._planning_closed = True
        if self.camera_output is not None and not self._camera_closed:
            try:
                camera_stopped = self.camera_output.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                if camera_stopped:
                    self._camera_closed = True
                else:
                    print(
                        "SCENE_RUNTIME_CAMERA_SHUTDOWN_TIMEOUT "
                        f"status={self.camera_output.publisher.status()}",
                        flush=True,
                    )
                    live_resources.append("camera_output")
        if not self._loggers_closed:
            closed_indices = getattr(self, "_closed_logger_indices", set())
            self._closed_logger_indices = closed_indices
            for index, logger in enumerate(self.loggers):
                if index in closed_indices:
                    continue
                try:
                    logger.close()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                else:
                    closed_indices.add(index)
            self._loggers_closed = len(closed_indices) == len(self.loggers)
        children_closed = bool(
            not self.shutdown_resources
            and self._planning_closed
            and (self.camera_output is None or self._camera_closed)
            and self._loggers_closed
        )
        if children_closed and not self._app_closed:
            try:
                close_simulation_app(self.session.app)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._app_closed = True
        if first_error is not None:
            raise first_error
        if live_resources:
            return SingleSceneRuntimeShutdownReport(
                stopped=False,
                live_resources=tuple(sorted(live_resources)),
            )
        self._closed = True
        return SingleSceneRuntimeShutdownReport(stopped=True)


def _stop_retained_shutdown_resource(resource: object) -> bool:
    """重试保留资源的 ``close``/``stop``，并统一不同返回契约。

    布尔值直接表示是否停止；带 ``stopped`` 属性的 report 和包含
    ``shutdown_timed_out`` 的 mapping 会转换为同一语义；无返回值视为同步关闭成功。
    """

    callback = getattr(resource, "close", None)
    if not callable(callback):
        callback = getattr(resource, "stop", None)
    if not callable(callback):
        return True
    result = callback()
    if isinstance(result, bool):
        return result
    stopped = getattr(result, "stopped", None)
    if isinstance(stopped, bool):
        return stopped
    if isinstance(result, Mapping):
        return not bool(result.get("shutdown_timed_out", False))
    return True


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


def create_single_scene_runtime(
    *,
    env: str = "scene1",
    env_config: Mapping[str, object] | None = None,
    simulation_app: SimulationAppSettings,
    camera_output_settings: CameraOutputRuntimeSettings,
    shutdown_settings: ShutdownSettings,
    output_settings: OutputPolicySettings | None = None,
    curobo_profile: str = "default",
    logging_profile: str = "default_logger",
    controller_bundle: str = "default",
    control_mode: str = "position",
    cache_root: str | Path | None = None,
    hold_app: bool = False,
    status_prefix: str | None = None,
    additional_output_path_plans: Sequence[OutputPathPlan] = (),
    session_factory: Callable[..., SimulationSession] = create_simulation_session,
    profile_loader: Callable[[str, str], Mapping[str, object]] = load_profile_yaml,
    controller_bundle_loader: Callable[[str], ControllerProfiles] = (
        load_controller_bundle
    ),
) -> SingleSceneRuntime:
    """围绕唯一一次 World reset 装配 object、全部 robot、sensor 和 registry。

    robot import 必须发生在 reset 前，controller finalize 和 batched handle 获取发生在 reset 后；
    任一步失败都会按资源依赖的逆序关闭已创建对象。所有输出路径先集中规划和冲突校验，
    再一次性创建目录/文件，避免场景后段校验失败时遗留部分输出。返回后，所有创建成功的
    资源都转移给 ``SingleSceneRuntime``，调用方必须最终调用 ``close``。
    """

    if env_config is None:
        env_config = profile_loader("env", env)
    instances = robot_instances_from_env_config(env_config)
    algorithm_profile = profile_loader("curobo", curobo_profile)
    robot_profiles = {
        profile_name: profile_loader("robot", profile_name)
        for profile_name in dict.fromkeys(
            instance.robot_profile for instance in instances
        )
    }
    profile_configs = tuple(
        merged_robot_config_with_curobo_profile(
            robot_profiles[instance.robot_profile],
            algorithm_profile,
        )
        for instance in instances
    )
    logging_config = joint_logging_config_from_mapping(
        profile_loader("logging", logging_profile)
    )
    output_settings = output_settings or OutputPolicySettings()
    csv_policy = output_settings.csv_existing_file_policy
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
                logging_config,
                robot_id=instance.robot_id,
                label=instance.label,
            )
        )
        is not None
    }
    validate_output_path_plans(list(csv_path_plans.values()))
    settings = EnvRuntimeSettings.from_env_config(env_config)
    settings.sensors.validate_single_scene_camera_scope()
    execution_configs = tuple(
        RobotExecutionConfig.from_mapping(profile, scene_instance=instance)
        for profile, instance in zip(profile_configs, instances, strict=True)
    )
    runtime_object_configs = runtime_objects_from_env_config(env_config)
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
    controller_profiles = _load_controller_profiles_cached(
        controller_profile_names,
        loader=controller_bundle_loader,
    )

    session = session_factory(simulation_app=simulation_app, settings=settings)
    camera_output: CameraOutputHandle | None = None
    loggers: list[JointTrackingLogger] = []
    planning_registry: RobotPlanningRegistry | None = None
    try:
        object_handles = add_runtime_objects(
            session.stage,
            runtime_object_configs,
            status_prefix=status_prefix,
        )
        objects = runtime_object_handles_by_name(object_handles)
        imported = tuple(
            import_execution_robot_to_stage(
                world=session.world,
                stage=session.stage,
                single_articulation_type=session.single_articulation_type,
                robot_execution=execution,
                controller_profiles=profiles,
                env_config=env_config,
            )
            for execution, profiles in zip(
                execution_configs, controller_profiles, strict=True
            )
        )

        sensor_cameras = create_sensor_camera_runtimes(
            stage=session.stage,
            sensors=settings.sensors,
        )
        session.world.reset()
        session.world.get_physics_context().set_gravity(settings.gravity_z)
        object_state_views = create_scene_object_state_views(object_handles)
        initialize_sensor_camera_runtimes(sensor_cameras)

        prepared_entries: list[tuple[Any, ...]] = []
        for instance, profile, controller_profile_name, profiles, imported_robot in zip(
            instances,
            profile_configs,
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
            kind = robot_kind_from_profile(profile)
            binding = PlanningBindingConfig.from_profile(profile, kind=kind)
            planning_names = (
                planning_joint_names_from_profile(profile) if binding.enabled else ()
            )
            groups = JointGroupLayout.resolve(
                kind=kind,
                command_joint_names=command_names,
                joint_groups=profile.get("joint_groups"),
                planning_joint_names=planning_names,
            )
            curobo_config = (
                CuroboConfig.from_mapping(profile) if binding.enabled else None
            )
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
                logging_config,
                robot_id=instance.robot_id,
                label=instance.label,
            )
            csv_output_plan = (
                None
                if log_path is None
                else plan_csv_output(
                    log_path,
                    joint_tracking_fieldnames(driven_names, logging_config),
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
            shutdown_timeout_s=shutdown_settings.camera_publisher_timeout_s,
        )
        apply_output_path_plans(
            [plan.path_plan for plan in csv_output_plans]
            + list(prepared_camera_output.path_plans)
            + list(additional_output_path_plans)
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
        ) in prepared_entries:
            logger = _make_robot_logger(
                logging_config,
                robot_id=instance.robot_id,
                label=instance.label,
                articulation=prepared.articulation,
                controller=prepared.joint_controller,
                physics_dt=float(session.world.get_physics_dt()),
                existing_data_policy=csv_policy,
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
                simulation_world=session.world,
                articulation_action_type=session.articulation_action_type,
                joint_controller=prepared.joint_controller,
                simulation_app=session.app if hold_app else None,
                render_enabled=simulation_app.gui or camera_output is not None,
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
                )
            )

        registry = RobotRegistry(tuple(robots))
        collision_registry = SceneCollisionRegistry()
        collision_registry.register_runtime_objects(
            object_handles,
            stage=session.stage,
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
        runtime = SingleSceneRuntime(
            session=session,
            env_config=env_config,
            robot_registry=registry,
            planning_registry=planning_registry,
            collision_registry=collision_registry,
            object_handles=object_handles,
            objects=objects,
            object_state_views=object_state_views,
            sensor_cameras=sensor_cameras,
            camera_output=camera_output,
            loggers=tuple(loggers),
            status_prefix=status_prefix,
        )
        _print_single_scene_status(runtime)
        return runtime
    except BaseException:
        _cleanup_failed_single_scene_runtime(
            planning_registry=planning_registry,
            camera_output=camera_output,
            loggers=loggers,
            app=session.app,
        )
        raise


def _cleanup_failed_single_scene_runtime(
    *,
    planning_registry: object | None,
    camera_output: object | None,
    loggers: Sequence[object],
    app: object,
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
        close for logger in loggers if callable(close := getattr(logger, "close", None))
    )
    callbacks.append(lambda: close_simulation_app(app))
    for callback in callbacks:
        try:
            callback()
        except BaseException:
            pass


def planning_joint_names_from_profile(
    profile: Mapping[str, object],
) -> tuple[str, ...]:
    """直接读取 model config/URDF 的 active joints，不分配 cuRobo context。"""

    curobo = profile.get("curobo")
    if not isinstance(curobo, Mapping):
        return ()
    robot = curobo.get("robot")
    if not isinstance(robot, Mapping):
        return ()
    config_path = robot.get("robot_config_path")
    if config_path is not None:
        data = load_yaml(repo_path(str(config_path)))
        robot_cfg = data.get("robot_cfg")
        if isinstance(robot_cfg, Mapping):
            kinematics = robot_cfg.get("kinematics")
            if isinstance(kinematics, Mapping):
                cspace = kinematics.get("cspace")
                if isinstance(cspace, Mapping):
                    values = cspace.get("joint_names", ())
                    return tuple(str(value) for value in values)
    urdf_path = robot.get("urdf_path")
    if urdf_path is None:
        return ()
    import xml.etree.ElementTree as ET

    root = ET.parse(repo_path(str(urdf_path))).getroot()
    return tuple(
        str(joint.attrib["name"])
        for joint in root.findall("joint")
        if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}
    )


def _make_robot_logger(
    config: JointLoggingConfig,
    *,
    robot_id: int,
    label: str,
    articulation: object,
    controller: object,
    physics_dt: float,
    existing_data_policy: str,
    timestamped_run_name: str | None,
    output_plan: CsvOutputPlan | None = None,
    paths_applied: bool = False,
) -> JointTrackingLogger:
    """按 driven joint order 创建单 robot logger，并计算 flush step 周期。"""

    driven_names = _robot_logger_joint_names(
        articulation=articulation,
        controller=controller,
    )
    path = _robot_log_path(config, robot_id=robot_id, label=label)
    return JointTrackingLogger(
        path,
        driven_names,
        flush_interval_steps=config.flush_interval_steps(physics_dt),
        config=config,
        existing_data_policy=existing_data_policy,
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
    config: JointLoggingConfig,
    *,
    robot_id: int,
    label: str,
) -> Path | None:
    """把共享 logging path 派生为包含 robot ID 与 label 的独立文件。"""

    if not config.enabled or config.joint_tracking_path is None:
        return None
    path = repo_path(config.joint_tracking_path)
    return path.with_name(f"{path.stem}.{robot_id}.{label}{path.suffix}")


def _print_single_scene_status(runtime: SingleSceneRuntime) -> None:
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
    "SingleSceneRuntime",
    "create_single_scene_runtime",
    "planning_joint_names_from_profile",
]
