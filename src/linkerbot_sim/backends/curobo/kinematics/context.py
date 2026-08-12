"""Kaleidoscope 专用的 cuRobo kinematics-only CUDA context。

本模块不会导入 MotionPlanner、BatchMotionPlanner、Scene 或项目碰撞世界。一个 context
只创建 Kinematics 与 InverseKinematics，并由一个 device batch IK adapter 独占关闭。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from linkerbot_sim.backends.curobo.config import (
    CuroboConfig,
)
from linkerbot_sim.backends.curobo.profile_merge import curobo_config_from_profiles
from linkerbot_sim.backends.curobo.robot_model import (
    materialize_curobo_config,
    materialized_robot_mapping,
)
from linkerbot_sim.backends.curobo.resources import curobo_task_resource_path
from linkerbot_sim.backends.curobo.runtime_imports import (
    ensure_torch_device_usable,
    import_curobo_module,
    import_curobo_public,
    import_torch_module,
    require_curobo_kernel_backend,
)
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.configuration.curobo import CuroboProfileSettings


class CuroboKinematicsContext:
    """一个机器人对应的长期 CUDA FK/IK owner。"""

    def __init__(
        self,
        config: CuroboConfig,
        *,
        cache_root: str | Path | None = None,
    ) -> None:
        source_urdf = config.robot.urdf_path
        curobo = import_curobo_module()
        config.task_bundle.validate_curobo_version(
            getattr(curobo, "__version__", "unknown")
        )
        self.kernel_backend = require_curobo_kernel_backend(expected="cuda_core")
        self.config = materialize_curobo_config(config, cache_root=cache_root)
        self._robot_asset_root_path = (
            None if source_urdf is None else source_urdf.parent
        )
        self.torch = import_torch_module()
        ensure_torch_device_usable(self.torch, self.config.device.device)
        # 只加载 kinematics/IK public modules。这里若出现 motion/scene module，就是产品
        # 闭包回归，架构测试会直接失败。
        self.types = import_curobo_public("types")
        self.kinematics_module = import_curobo_public("kinematics")
        self.ik_module = import_curobo_public("inverse_kinematics")
        self.robot_module = import_curobo_public("_src.types.robot")
        self.device_cfg = self._make_device_cfg()
        self.default_tcp_frame = (
            self.config.robot.default_tcp_frame
            or self.config.robot.resolved_tool_frames[0]
        )
        self.tool_frames = tuple(self.config.robot.resolved_tool_frames)
        self.kinematics = self._make_kinematics()
        self._ik_solver: object | None = None
        self._kinematics_closed = False
        self._closing_started = False
        self._closed = False

    @property
    def ik_solver(self) -> object:
        """首次访问时创建 IK solver；joint-only task 根本不构造本 context。"""

        if self._closed or self._closing_started:
            raise RuntimeError("cuRobo kinematics context teardown has started")
        if self._ik_solver is None:
            self._ik_solver = self._make_ik_solver()
        return self._ik_solver

    def joint_names(self) -> list[str]:
        names = getattr(self.kinematics, "joint_names")
        return list(names() if callable(names) else names)

    def frame_names(self) -> list[str]:
        return list(self.tool_frames)

    def joint_state_from_positions(self, positions: object) -> object:
        """保持已有 CUDA tensor 的 device/dtype，并构造 cuRobo JointState。"""

        if self.torch.is_tensor(positions):
            tensor = positions.to(
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
        else:
            tensor = self.torch.as_tensor(
                positions,
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
        return self.types.JointState.from_position(
            tensor.contiguous(),
            joint_names=self.joint_names(),
        )

    def close(self) -> None:
        """幂等释放 solver graph 与 kinematics；失败的资源保留以便重试。"""

        if self._closed:
            return
        self._closing_started = True
        first_error: BaseException | None = None
        solver = self._ik_solver
        if solver is not None:
            try:
                _destroy_if_supported(solver)
            except BaseException as exc:
                first_error = exc
            else:
                self._ik_solver = None
        if not self._kinematics_closed:
            try:
                _destroy_if_supported(self.kinematics)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note(
                        f"kinematics close also failed: {type(exc).__name__}: {exc}"
                    )
            else:
                self._kinematics_closed = True
        self._closed = self._ik_solver is None and self._kinematics_closed
        if first_error is not None:
            raise first_error

    def _make_device_cfg(self) -> object:
        torch = self.torch
        device = torch.device(self.config.device.device)
        return self.types.DeviceCfg(
            device=device,
            dtype=_torch_dtype(torch, self.config.device.tensor_dtype),
            collision_geometry_dtype=_torch_dtype(
                torch, self.config.device.collision_geometry_dtype
            ),
            collision_gradient_dtype=_torch_dtype(
                torch, self.config.device.collision_gradient_dtype
            ),
            collision_distance_dtype=_torch_dtype(
                torch, self.config.device.collision_distance_dtype
            ),
        )

    def _make_kinematics(self) -> object:
        robot = self.config.robot
        if robot.robot_config_path is not None:
            config = self.kinematics_module.KinematicsCfg.from_robot_yaml_file(
                str(robot.robot_config_path),
                tool_frames=list(self.tool_frames),
                device_cfg=self.device_cfg,
                urdf_path=None if robot.urdf_path is None else str(robot.urdf_path),
                # Kaleidoscope IK 明确不加载自碰撞 spheres。
                load_collision_spheres=False,
            )
        elif robot.urdf_path is not None and robot.base_link is not None:
            config = self.kinematics_module.KinematicsCfg.from_basic_urdf(
                str(robot.urdf_path),
                robot.base_link,
                list(self.tool_frames),
                device_cfg=self.device_cfg,
            )
        else:
            raise ValueError(
                "cuRobo kinematics requires robot_config_path or urdf_path + base_link"
            )
        return self.kinematics_module.Kinematics(config)

    def _make_ik_solver(self) -> object:
        robot = self.config.robot
        if robot.robot_config_path is not None:
            robot_input = materialized_robot_mapping(
                robot,
                tool_frames=self.tool_frames,
                asset_root_path=self._robot_asset_root_path,
            )
        elif robot.urdf_path is not None and robot.base_link is not None:
            robot_input = self.robot_module.RobotCfg.from_basic(
                urdf_path=str(robot.urdf_path),
                base_link=str(robot.base_link),
                tool_frames=list(self.tool_frames),
                device_cfg=self.device_cfg,
            )
        else:
            raise ValueError("cuRobo IK requires a robot model")
        ik = self.config.ik
        bundle = self.config.task_bundle
        solver_config = self.ik_module.InverseKinematicsCfg.create(
            robot=robot_input,
            optimizer_configs=[
                _task_config_path(path) for path in bundle.ik_optimizer_configs
            ],
            metrics_rollout=_task_config_path(bundle.ik_metrics_rollout),
            transition_model=_task_config_path(bundle.ik_transition_model),
            device_cfg=self.device_cfg,
            num_seeds=int(ik.num_seeds),
            position_tolerance=float(ik.position_tolerance),
            orientation_tolerance=float(ik.orientation_tolerance),
            use_cuda_graph=bool(ik.use_cuda_graph),
            random_seed=int(ik.random_seed),
            store_debug=False,
            override_optimizer_num_iters=dict(ik.override_optimizer_num_iters),
            override_iters_for_multi_link_ik=ik.override_iters_for_multi_link_ik,
            optimization_dt=ik.optimization_dt,
            velocity_regularization_weight=ik.velocity_regularization_weight,
            acceleration_regularization_weight=ik.acceleration_regularization_weight,
            success_requires_convergence=bool(ik.success_requires_convergence),
            seed_position_weight=float(ik.seed_position_weight),
            seed_orientation_weight=float(ik.seed_orientation_weight),
            seed_velocity_weight=float(ik.seed_velocity_weight),
            seed_acceleration_weight=float(ik.seed_acceleration_weight),
            seed_solver_num_seeds=int(ik.seed_solver_num_seeds),
            self_collision_check=False,
            scene_model=None,
            collision_cache=None,
            max_batch_size=int(ik.max_batch_size),
            multi_env=False,
            max_goalset=1,
            load_collision_spheres=False,
        )
        return self.ik_module.InverseKinematics(solver_config)


def kinematics_config_from_robot_profile(
    robot_profile: RobotProfileSettings,
    *,
    settings: CuroboProfileSettings,
    cuda_device: int,
) -> CuroboConfig:
    """把 robot 资产事实与 strict cuRobo profile 合成无碰撞 IK 配置。"""

    if not isinstance(robot_profile, RobotProfileSettings):
        raise TypeError("robot_profile must be RobotProfileSettings")
    if not isinstance(settings, CuroboProfileSettings):
        raise TypeError("settings must be CuroboProfileSettings")
    base = curobo_config_from_profiles(
        robot_profile,
        curobo_settings=settings,
        cuda_device=cuda_device,
    )
    # 即使通用 robot profile 为 Mirror 规划保存了 collision spheres，Kaleidoscope
    # composition 也在这里强制关闭；这不是忽略配置，而是产品能力闭包的显式裁剪。
    robot = replace(base.robot, load_collision_spheres=False)
    ik = replace(
        base.ik,
        multi_env=False,
        self_collision_check=False,
        collision_cache={},
    )
    result = replace(
        base,
        robot=robot,
        ik=ik,
    )
    result.validate()
    return result


def create_kinematics_context(
    *,
    robot_profile: RobotProfileSettings,
    settings: CuroboProfileSettings,
    cuda_device: int,
    cache_root: str | Path | None = None,
    context_type: type[CuroboKinematicsContext] = CuroboKinematicsContext,
) -> CuroboKinematicsContext:
    """创建独立 kinematics owner；factory 不创建 planner 或碰撞世界。"""

    if not isinstance(robot_profile, RobotProfileSettings):
        raise TypeError("robot_profile must be RobotProfileSettings")
    config = kinematics_config_from_robot_profile(
        robot_profile,
        settings=settings,
        cuda_device=cuda_device,
    )
    return context_type(config, cache_root=cache_root)


def _task_config_path(relative_path: str) -> str:
    return curobo_task_resource_path(relative_path)


def _torch_dtype(torch: object, name: str) -> object:
    dtype = getattr(torch, str(name), None)
    if dtype is None:
        raise ValueError(f"unknown Torch dtype for cuRobo: {name}")
    return dtype


def _destroy_if_supported(value: object) -> None:
    destroy = getattr(value, "destroy", None)
    if callable(destroy):
        destroy()


__all__ = [
    "CuroboKinematicsContext",
    "create_kinematics_context",
    "kinematics_config_from_robot_profile",
]
