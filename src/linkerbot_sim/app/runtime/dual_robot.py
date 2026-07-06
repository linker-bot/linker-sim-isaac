"""双机器人 Isaac runtime 装配入口。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from linkerbot_sim.app.runtime.objects import (
    RuntimeObjectConfig,
    RuntimeObjectHandle,
    add_runtime_objects,
    runtime_object_handles_by_name,
    runtime_objects_from_env_config,
)
from linkerbot_sim.app.runtime.settings import EnvRuntimeSettings
from linkerbot_sim.app.runtime.simulation_app_lifecycle import close_simulation_app
from linkerbot_sim.app.runtime.simulation_session import SimulationSession, create_simulation_session
from linkerbot_sim.assets.robot_loader import (
    DualRobotExecutionConfig,
    dual_robot_scene_instances_from_env_config,
)
from linkerbot_sim.backends.cumotion.dual_urdf import dual_cumotion_config_from_sides
from linkerbot_sim.configs.profiles import (
    load_default_controller_profiles,
    load_profile_yaml,
)
from linkerbot_sim.execution.dual_runtime import DualRobotRuntime, RobotSideRuntime
from linkerbot_sim.execution.setup import (
    ImportedRobot,
    PreparedRobotRuntime,
    finalize_robot_controller,
    import_execution_robot_to_stage,
)
from linkerbot_sim.sensors.camera_runtime import (
    SensorCameraRuntime,
    create_sensor_camera_runtimes,
    initialize_sensor_camera_runtimes,
)
from linkerbot_sim.sensors.camera_observer import (
    CameraOutputHandle,
    start_camera_output,
)
from linkerbot_sim.utils.paths import repo_path


@dataclass(frozen=True)
class DualRobotRuntimeConfig:
    """不启动 Isaac 的双机器人 runtime 配置摘要。"""

    robot_config: Mapping[str, object]
    side_robot_configs: Mapping[str, Mapping[str, object]]
    dual_config: DualRobotExecutionConfig
    env_config: Mapping[str, object]
    runtime_settings: EnvRuntimeSettings
    object_configs: tuple[RuntimeObjectConfig, ...]


@dataclass
class DualRobotAppRuntime:
    """双机器人动作脚本所需的完整 app runtime。"""

    session: SimulationSession
    execution: DualRobotRuntime
    env_config: Mapping[str, object]
    robot_config: Mapping[str, object]
    side_robot_configs: Mapping[str, Mapping[str, object]]
    dual_config: DualRobotExecutionConfig
    imported: Mapping[str, ImportedRobot]
    prepared: Mapping[str, PreparedRobotRuntime]
    object_handles: tuple[RuntimeObjectHandle, ...]
    objects: Mapping[str, RuntimeObjectHandle]
    sensor_cameras: tuple[SensorCameraRuntime, ...]
    camera_output: CameraOutputHandle | None
    status_prefix: str | None = None
    _closed: bool = False

    @property
    def world(self):
        """返回当前 Isaac World。"""

        return self.execution.simulation_world

    @property
    def left(self) -> RobotSideRuntime:
        """返回左侧机器人运行时。"""

        return self.execution.left

    @property
    def right(self) -> RobotSideRuntime:
        """返回右侧机器人运行时。"""

        return self.execution.right

    def close(self) -> None:
        """关闭 SimulationApp；多次调用是安全的。"""

        if self._closed:
            return
        self._closed = True
        if self.camera_output is not None:
            self.camera_output.close()
        close_simulation_app(self.session.app)


def load_dual_robot_runtime_config(
    *,
    env: str = "scene2",
) -> DualRobotRuntimeConfig:
    """加载双机器人 runtime 所需配置，不启动 Isaac。"""

    env_config = load_profile_yaml("env", env)
    robot_instances = dual_robot_scene_instances_from_env_config(env_config)
    side_robot_configs = {
        side: load_profile_yaml("robot", instance.robot_profile)
        for side, instance in robot_instances.items()
    }
    robot_config = dual_cumotion_config_from_sides(
        left=side_robot_configs["left"],
        right=side_robot_configs["right"],
    )
    runtime_settings = EnvRuntimeSettings.from_env_config(env_config)
    return DualRobotRuntimeConfig(
        robot_config=robot_config,
        side_robot_configs=side_robot_configs,
        dual_config=DualRobotExecutionConfig.from_robot_configs(
            left=side_robot_configs["left"],
            right=side_robot_configs["right"],
            root_poses={
                name: instance.root_pose
                for name, instance in robot_instances.items()
            },
        ),
        env_config=env_config,
        runtime_settings=runtime_settings,
        object_configs=runtime_objects_from_env_config(env_config),
    )


def create_dual_robot_runtime(
    *,
    env: str = "scene2",
    control_mode: str = "position",
    gui: bool = False,
    hold_app: bool = False,
    status_prefix: str | None = None,
) -> DualRobotAppRuntime:
    """从 profile 名称创建双机器人 runtime。"""

    config = load_dual_robot_runtime_config(
        env=env,
    )
    controller_profiles = load_default_controller_profiles()
    session = create_simulation_session(gui=gui, settings=config.runtime_settings)
    camera_output: CameraOutputHandle | None = None
    try:
        object_handles = add_runtime_objects(
            session.stage,
            config.object_configs,
            status_prefix=status_prefix,
        )
        objects = runtime_object_handles_by_name(object_handles)

        imported = {
            side: import_execution_robot_to_stage(
                world=session.world,
                stage=session.stage,
                single_articulation_type=session.single_articulation_type,
                robot_execution=config.dual_config.side(side),
                controller_profiles=controller_profiles,
                env_config=config.env_config,
            )
            for side in ("left", "right")
        }
        for side, imported_side in imported.items():
            _print_status(
                status_prefix,
                f"{side.upper()}_GRAVITY {imported_side.gravity_counts}",
            )
            _print_status(
                status_prefix,
                f"{side.upper()}_SOLVER {imported_side.solver_counts}",
            )

        sensor_cameras = create_sensor_camera_runtimes(
            stage=session.stage,
            sensors=config.runtime_settings.sensors,
        )
        camera_output = start_camera_output(
            sensor_cameras,
            path_resolver=repo_path,
        )
        session.world.reset()
        session.world.get_physics_context().set_gravity(
            config.runtime_settings.gravity_z
        )
        initialize_sensor_camera_runtimes(sensor_cameras)
        for sensor_camera in sensor_cameras:
            _print_status(
                status_prefix,
                "SENSOR_CAMERA "
                f"name={sensor_camera.name} prim_path={sensor_camera.prim_path} "
                f"modalities={','.join(sensor_camera.settings.modalities)}",
            )

        prepared: dict[str, PreparedRobotRuntime] = {}
        side_runtimes: dict[str, RobotSideRuntime] = {}
        for side, imported_side in imported.items():
            prepared_side = finalize_robot_controller(
                imported=imported_side,
                controller_profiles=controller_profiles,
                control_mode=control_mode,
            )
            prepared[side] = prepared_side
            controller = prepared_side.joint_controller
            side_runtimes[side] = RobotSideRuntime(
                side=side,
                articulation=prepared_side.articulation,
                joint_controller=controller,
            )
            _print_status(
                status_prefix,
                f"IMPORTED side={side} asset={prepared_side.asset_path} "
                f"prim_path={imported_side.articulation_path} "
                f"num_dof={prepared_side.articulation.num_dof} "
                f"control_mode={control_mode} "
                f"command_joints={list(controller.command_joint_names)}",
            )
            _print_status(
                status_prefix,
                f"{side.upper()}_DOF_NAMES "
                + ", ".join(list(prepared_side.articulation.dof_names)),
            )

        execution = DualRobotRuntime(
            left=side_runtimes["left"],
            right=side_runtimes["right"],
            simulation_world=session.world,
            articulation_action_type=session.articulation_action_type,
            simulation_app=session.app if hold_app else None,
            render_enabled=gui or camera_output is not None,
            camera_observer=None if camera_output is None else camera_output.observer,
        )
        return DualRobotAppRuntime(
            session=session,
            execution=execution,
            env_config=config.env_config,
            robot_config=config.robot_config,
            side_robot_configs=config.side_robot_configs,
            dual_config=config.dual_config,
            imported=imported,
            prepared=prepared,
            object_handles=object_handles,
            objects=objects,
            sensor_cameras=sensor_cameras,
            camera_output=camera_output,
            status_prefix=status_prefix,
        )
    except Exception:
        if camera_output is not None:
            camera_output.close()
        close_simulation_app(session.app)
        raise


def _print_status(status_prefix: str | None, message: str) -> None:
    """按可选前缀输出机器可 grep 的 runtime 状态行。"""

    if status_prefix is None:
        return
    print(f"{status_prefix}_{message}", flush=True)
