"""单机器人 Isaac runtime 装配入口。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from linkerbot_sim.app.runtime.objects import (
    RuntimeObjectHandle,
    add_runtime_objects,
    runtime_object_handles_by_name,
    runtime_objects_from_env_config,
)
from linkerbot_sim.app.runtime.settings import EnvRuntimeSettings
from linkerbot_sim.app.runtime.simulation_app_lifecycle import close_simulation_app
from linkerbot_sim.app.runtime.simulation_session import SimulationSession, create_simulation_session
from linkerbot_sim.assets.robot_loader import (
    RobotExecutionConfig,
    robot_scene_instance_from_env_config,
)
from linkerbot_sim.backends.cumotion.context import CuMotionConfig
from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
)
from linkerbot_sim.backends.cumotion.profile_config import (
    merged_robot_config_with_cumotion_profile,
    motion_planner_config_from_profile,
    robot_cumotion_config,
)
from linkerbot_sim.configs.profiles import (
    load_default_controller_profiles,
    load_profile_yaml,
)
from linkerbot_sim.execution.runtime import ExecutionRuntime
from linkerbot_sim.execution.setup import (
    ImportedRobot,
    PreparedRobotRuntime,
    finalize_robot_controller,
    import_execution_robot_to_stage,
)
from linkerbot_sim.logging.config import (
    JointLoggingConfig,
    joint_logging_config_from_mapping,
    override_logging_config,
)
from linkerbot_sim.logging.joint_logger import JointTrackingLogger
from linkerbot_sim.robots.mimic import mjcf_equality_follower_joint_names
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class LoggingRuntimeOverrides:
    """CLI 对关节日志配置的覆盖。"""

    joint_tracking_path: Path | None = None
    interval_steps: int | None = None
    log_measured_effort: bool | None = None
    log_applied_effort: bool | None = None
    log_action_effort: bool | None = None
    log_command_effort: bool | None = None


@dataclass
class SingleRobotRuntime:
    """单机器人动作脚本所需的完整 runtime。"""

    session: SimulationSession
    execution: ExecutionRuntime
    robot_cumotion: CuMotionConfig
    motion_planner_config: MotionPlannerBackendConfig
    env_config: Mapping[str, object]
    robot_config: Mapping[str, object]
    imported_robot: ImportedRobot
    prepared_robot: PreparedRobotRuntime
    object_handles: tuple[RuntimeObjectHandle, ...]
    objects: Mapping[str, RuntimeObjectHandle]
    logging_config: JointLoggingConfig
    log_path: Path | None
    logger: JointTrackingLogger
    status_prefix: str | None = None
    _closed: bool = False

    @property
    def robot(self):
        return self.execution.articulation

    @property
    def world(self):
        return self.execution.simulation_world

    @property
    def controller(self):
        return self.execution.joint_controller

    @property
    def mjcf_path(self) -> Path | None:
        return self.prepared_robot.mjcf_path

    def mjcf_path_required(self, reason: str) -> Path:
        if self.mjcf_path is None:
            raise ValueError(f"{reason} requires an MJCF robot asset")
        return self.mjcf_path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.logger.close()
        close_simulation_app(self.session.app)


def create_single_robot_runtime(
    *,
    env: str = "scene1",
    cumotion_profile: str = "default",
    logging_profile: str = "default_logger",
    control_mode: str = "position",
    gui: bool = False,
    status_prefix: str | None = None,
    logging_overrides: LoggingRuntimeOverrides | None = None,
) -> SingleRobotRuntime:
    """从 profile 名称创建单机器人 runtime。"""

    logging_overrides = logging_overrides or LoggingRuntimeOverrides()

    cumotion_profile_data = load_profile_yaml("cumotion", cumotion_profile)
    env_config = load_profile_yaml("env", env)
    robot_instance = robot_scene_instance_from_env_config(env_config, "single")
    robot_config = merged_robot_config_with_cumotion_profile(
        load_profile_yaml("robot", robot_instance.robot_profile), cumotion_profile_data
    )
    controller_profiles = load_default_controller_profiles()
    logging_config = joint_logging_config_from_mapping(
        load_profile_yaml("logging", logging_profile)
    )
    logging_config = override_logging_config(
        logging_config,
        joint_tracking_path=logging_overrides.joint_tracking_path,
        interval_steps=logging_overrides.interval_steps,
        log_measured_effort=logging_overrides.log_measured_effort,
        log_applied_effort=logging_overrides.log_applied_effort,
        log_action_effort=logging_overrides.log_action_effort,
        log_command_effort=logging_overrides.log_command_effort,
    )

    runtime_settings = EnvRuntimeSettings.from_env_config(env_config)
    robot_execution = RobotExecutionConfig.from_mapping(
        robot_config,
        root_pose=robot_instance.root_pose,
    )
    robot_cumotion = robot_cumotion_config(robot_config)
    motion_planner_config = motion_planner_config_from_profile(cumotion_profile_data)
    runtime_object_configs = runtime_objects_from_env_config(env_config)

    session = create_simulation_session(gui=gui, settings=runtime_settings)
    logger: JointTrackingLogger | None = None
    try:
        object_handles = add_runtime_objects(
            session.stage,
            runtime_object_configs,
            status_prefix=status_prefix,
        )
        objects = runtime_object_handles_by_name(object_handles)

        imported = import_execution_robot_to_stage(
            world=session.world,
            stage=session.stage,
            single_articulation_type=session.single_articulation_type,
            robot_execution=robot_execution,
            controller_profiles=controller_profiles,
            env_config=env_config,
        )
        _print_status(status_prefix, f"GRAVITY {imported.gravity_counts}")
        _print_status(status_prefix, f"SOLVER {imported.solver_counts}")

        session.world.reset()
        session.world.get_physics_context().set_gravity(runtime_settings.gravity_z)

        prepared = finalize_robot_controller(
            imported=imported,
            controller_profiles=controller_profiles,
            control_mode=control_mode,
        )
        robot_articulation = prepared.articulation
        controller = prepared.joint_controller
        driven_joint_names = [
            list(robot_articulation.dof_names)[int(index)]
            for index in controller.driven_indices
        ]
        flush_interval_steps = logging_config.flush_interval_steps(
            float(session.world.get_physics_dt())
        )
        log_path = (
            None
            if not logging_config.enabled or logging_config.joint_tracking_path is None
            else repo_path(logging_config.joint_tracking_path)
        )
        logger = JointTrackingLogger(
            log_path,
            driven_joint_names,
            flush_interval_steps=flush_interval_steps,
            config=logging_config,
        )
        execution = ExecutionRuntime(
            articulation=robot_articulation,
            simulation_world=session.world,
            articulation_action_type=session.articulation_action_type,
            joint_controller=controller,
            simulation_app=session.app,
            render_enabled=gui,
            drive_logger=logger,
        )

        mimic_names = (
            ()
            if prepared.mjcf_path is None
            else sorted(mjcf_equality_follower_joint_names(prepared.mjcf_path))
        )
        _print_status(
            status_prefix,
            "IMPORTED "
            f"asset={prepared.asset_path} prim_path={imported.articulation_path} "
            f"num_dof={robot_articulation.num_dof} control_mode={control_mode} "
            f"mimic_joint_names={mimic_names} "
            f"follower_relations={controller.follower_mapper.relations}",
        )
        _print_status(
            status_prefix,
            "DOF_NAMES " + ", ".join(list(robot_articulation.dof_names)),
        )
        return SingleRobotRuntime(
            session=session,
            execution=execution,
            robot_cumotion=robot_cumotion,
            motion_planner_config=motion_planner_config,
            env_config=env_config,
            robot_config=robot_config,
            imported_robot=imported,
            prepared_robot=prepared,
            object_handles=object_handles,
            objects=objects,
            logging_config=logging_config,
            log_path=log_path,
            logger=logger,
            status_prefix=status_prefix,
        )
    except Exception:
        if logger is not None:
            logger.close()
        close_simulation_app(session.app)
        raise


def _print_status(status_prefix: str | None, message: str) -> None:
    if status_prefix is None:
        return
    print(f"{status_prefix}_{message}", flush=True)
