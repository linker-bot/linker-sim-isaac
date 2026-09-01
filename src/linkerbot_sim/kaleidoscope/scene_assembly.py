"""Kaleidoscope strict config 到 PhysX CUDA/Newton CUDA scene 的启动链。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import math
import sys
import traceback

from linkerbot_sim.isaac.replicated_scene import (
    build_replicated_physx_scene,
    finalize_replicated_robot_views,
)
from linkerbot_sim.isaac.replicated_scene.views import (
    command_joint_limits,
    create_dynamic_object_rigid_view,
    create_tcp_rigid_views,
)
from linkerbot_sim.isaac.spec import (
    IsaacAppSpec,
    IsaacComputeSpec,
    IsaacNewtonCudaSpec,
    IsaacPhysxCudaSpec,
    IsaacRenderSpec,
    IsaacSessionSpec,
)
from linkerbot_sim.kaleidoscope.ik import EnvLocalDeviceBatchIKSolver
from linkerbot_sim.kaleidoscope.control_mode import KaleidoscopeControlBinding
from linkerbot_sim.kaleidoscope.physx_ports import (
    IsaacArticulationTensorPort,
    IsaacRigidObjectTensorPort,
)
from linkerbot_sim.robots.mimic.runtime import resolve_mimic_follower_controls
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz


def build_kaleidoscope_scene_assembly(
    *,
    config: object,
    num_envs: int,
    viewport: object | None = None,
    session_factory: Callable[..., object] | None = None,
    replicated_scene_builder: Callable[..., object] = build_replicated_physx_scene,
    newton_scene_builder: Callable[..., object] | None = None,
    ik_solver_factory: Callable[..., object] | None = None,
):
    """构造默认的 Session → replicated scene → raw CUDA ports。

    这是生产 composition 的重型边界；普通 ``linkerbot_sim.kaleidoscope`` facade 不会导入
    本模块。任一阶段失败都会按 solver → port/view → Session 逆序清理，原始构造异常
    保持为主异常，清理异常只作为 ``note`` 附加。
    """

    import torch

    from linkerbot_sim.kaleidoscope.isaac_adapter import KaleidoscopeSceneAssembly

    _validate_root_config(config, num_envs=num_envs)
    if session_factory is None:
        from linkerbot_sim.isaac.session import create_isaac_session_from_spec

        session_factory = create_isaac_session_from_spec
    physics_engine = str(config.physics.engine)
    session_spec = session_spec_from_config(
        config,
        num_envs=num_envs,
        viewport=viewport,
    )
    backend_kind = session_spec.physics_kind
    session: object | None = None
    robot_ports: list[object] = []
    control_bindings: list[KaleidoscopeControlBinding] = []
    object_port: object | None = None
    physics_state_port: object | None = None
    viewport_reconfigure: Callable[[], None] | None = None
    raw_views: list[object] = []
    solvers: dict[str, object] = {}
    try:
        session = session_factory(spec=session_spec)
        physics_runtime = getattr(session, "physics_runtime", None)
        if getattr(physics_runtime, "kind", None) != backend_kind:
            raise RuntimeError(
                "Kaleidoscope Session physics runtime differs from strict config: "
                f"actual={getattr(physics_runtime, 'kind', None)!r}, "
                f"expected={backend_kind!r}"
            )
        if physics_engine == "physx":
            world = getattr(physics_runtime, "world", None)
            if world is None:
                raise RuntimeError("physx_cuda runtime did not publish its owned World")
            replicated = replicated_scene_builder(
                stage=session.stage,
                world=world,
                scene_settings=config.scene,
                environment_settings=config.environments,
                num_envs=num_envs,
                dynamic_object_name=config.task.dynamic_object,
                controller_bundle=config.default_controller_bundle,
                controller_bundles=config.controller_bundles,
                solver_type=config.physics.solver_type,
            )
            raw_views.extend(robot.articulation_view for robot in replicated.robots)
            if viewport is not None:
                viewport_reconfigure = _configure_kaleidoscope_viewport(
                    stage=session.stage,
                    replicated=replicated,
                    viewport=viewport,
                )
            # World.reset 初始化 PhysX tensor entities。只能通过 Session-owned concrete
            # runtime 触发，不使用 ``session.world`` shim，也不创建第二个 World owner。
            physics_runtime.reset()
            replicated = finalize_replicated_robot_views(replicated)
            tcp_views = create_tcp_rigid_views(replicated)
            object_view = create_dynamic_object_rigid_view(
                replicated,
                object_name=config.task.dynamic_object,
            )
        else:
            from linkerbot_sim.isaac.replicated_scene import newton_builder

            if newton_scene_builder is None:
                newton_scene_builder = newton_builder.build_replicated_newton_scene
            replicated = newton_scene_builder(
                stage=session.stage,
                runtime=physics_runtime,
                scene_settings=config.scene,
                environment_settings=config.environments,
                num_envs=num_envs,
                dynamic_object_name=config.task.dynamic_object,
                controller_bundle=config.default_controller_bundle,
                controller_bundles=config.controller_bundles,
                prepare_newton_render_topology=viewport is not None,
            )
            raw_views.extend(robot.articulation_view for robot in replicated.robots)
            if viewport is not None:
                viewport_reconfigure = _configure_kaleidoscope_viewport(
                    stage=session.stage,
                    replicated=replicated,
                    viewport=viewport,
                )
            # Newton builder 已一次性 finalize 全部 worlds；reset 只恢复 manager-owned
            # generalized state/control，不重建 model 或创建 Isaac World。
            physics_runtime.reset()
            tcp_views = newton_builder.create_newton_tcp_rigid_views(
                replicated,
                runtime=physics_runtime,
            )
            object_view = newton_builder.create_newton_dynamic_object_rigid_view(
                replicated,
                runtime=physics_runtime,
                object_name=config.task.dynamic_object,
            )
        raw_views.extend((*tcp_views.values(), object_view))
        device = torch.device(config.torch_device)
        env_origins = torch.as_tensor(
            replicated.env_origins,
            device=device,
            dtype=torch.float32,
        ).contiguous()
        lower_rows: list[torch.Tensor] = []
        upper_rows: list[torch.Tensor] = []
        for robot in replicated.robots:
            assert robot.command_joint_indices is not None
            if physics_engine == "physx":
                command_indices = torch.as_tensor(
                    robot.command_joint_indices,
                    device=device,
                    dtype=torch.int64,
                ).contiguous()
                port = IsaacArticulationTensorPort(
                    label=robot.label,
                    view=robot.articulation_view,
                    tcp_view=tcp_views[robot.label],
                    command_joint_indices=command_indices,
                    device=device,
                    command_joint_names=tuple(robot.command_joint_names),
                    command_joint_indices_host=tuple(
                        int(index) for index in robot.command_joint_indices
                    ),
                    orientation_order="wxyz",
                    tcp_offset_xyz=robot.tcp_offset_xyz,
                    tcp_offset_rpy=robot.tcp_offset_rpy,
                    mimic_follower_controls=tuple(
                        resolve_mimic_follower_controls(
                            list(robot.articulation_view.dof_names),
                            robot.asset_path
                            if robot.asset_type in {"mjcf", "urdf"}
                            else None,
                        )
                    ),
                )
                raw_limits, host_indices = command_joint_limits(robot)
                limits = _copy_startup_metadata_to_cuda(
                    value=raw_limits,
                    device=device,
                )
                if (
                    limits.ndim != 3
                    or limits.shape[0] != num_envs
                    or limits.shape[2] != 2
                ):
                    raise RuntimeError(
                        f"robot {robot.label!r} returned invalid PhysX DOF limits"
                    )
                selected = limits[0].index_select(
                    0,
                    torch.as_tensor(host_indices, device=device, dtype=torch.int64),
                )
                lower, upper = selected[:, 0], selected[:, 1]
            else:
                from linkerbot_sim.kaleidoscope.newton_ports import (
                    NewtonArticulationTensorPort,
                )

                port = NewtonArticulationTensorPort(
                    label=robot.label,
                    view=robot.articulation_view,
                    tcp_view=tcp_views[robot.label],
                    command_dof_names=robot.command_joint_names,
                    device=device,
                    tcp_offset_xyz=robot.tcp_offset_xyz,
                    tcp_offset_rpy=robot.tcp_offset_rpy,
                )
                from linkerbot_sim.isaac.replicated_scene.newton_builder import (
                    newton_command_joint_limits,
                )

                lower, upper = newton_command_joint_limits(
                    robot,
                    runtime=physics_runtime,
                    device=device,
                )
            robot_ports.append(port)
            component_mapping = getattr(robot.profile, "component_mapping", None)
            controller_profiles = getattr(robot, "controller_profiles", None)
            if component_mapping is not None and controller_profiles is not None:
                command_names = tuple(str(name) for name in robot.command_joint_names)
                components = tuple(
                    component_mapping.joint_component(name) for name in command_names
                )
                control_bindings.append(
                    KaleidoscopeControlBinding(
                        label=robot.label,
                        port=port,
                        controller_profiles=controller_profiles,
                        command_joint_names=command_names,
                        components=components,
                    )
                )
            lower_rows.append(lower.clone())
            upper_rows.append(upper.clone())
        if physics_engine == "physx":
            object_port = IsaacRigidObjectTensorPort(
                label=str(config.task.dynamic_object),
                view=object_view,
                device=device,
                orientation_order="wxyz",
            )
        else:
            from linkerbot_sim.kaleidoscope.newton_ports import (
                NewtonRigidObjectTensorPort,
                NewtonSolverIntegrationTensorPort,
            )

            object_port = NewtonRigidObjectTensorPort(
                label=str(config.task.dynamic_object),
                view=object_view,
                device=device,
            )
            physics_state_port = NewtonSolverIntegrationTensorPort(
                runtime=physics_runtime,
                device=device,
            )
        physics_runtime.forward()
        first_env = torch.zeros(1, device=device, dtype=torch.int64)
        if physics_engine == "physx":
            for port in robot_ports:
                port.prepare_full_dof_reset(
                    port.read_all_joint_positions(first_env).reshape(-1)
                )
        nominal_joint_positions = torch.cat(
            [port.read_joint_positions(first_env).reshape(-1) for port in robot_ports]
        ).clone()
        joint_lower = torch.cat(lower_rows).contiguous()
        joint_upper = torch.cat(upper_rows).contiguous()
        torch._assert_async(
            torch.all(torch.isfinite(joint_lower) & torch.isfinite(joint_upper)),
            "command joint limits must be finite",
        )
        torch._assert_async(
            torch.all(joint_lower < joint_upper),
            "command joint lower limits must be below upper limits",
        )
        fixed_orientations: dict[str, torch.Tensor] = {}
        if config.curobo is not None:
            if int(config.curobo.kinematics.max_batch_size) < num_envs:
                raise ValueError(
                    "curobo.kinematics.max_batch_size must cover the final num_envs"
                )
            factory = ik_solver_factory or _create_device_ik_solver
            by_label = {item.label: item for item in config.scene.robots}
            replicated_by_label = {item.label: item for item in replicated.robots}
            for port in robot_ports:
                robot = replicated_by_label[port.label]
                root_pose = by_label[port.label].root_pose
                solvers[port.label] = factory(
                    robot=robot,
                    settings=config.curobo,
                    cuda_device=config.cuda_device,
                    device=device,
                    root_pose=root_pose,
                )
                _position, orientation = port.read_tcp_pose_wxyz(first_env)
                fixed_orientations[port.label] = orientation[0].clone()
        dynamic_instance = _dynamic_object_instance(config)
        return KaleidoscopeSceneAssembly(
            session=session,
            robot_ports=tuple(robot_ports),
            object_port=object_port,
            env_origins=env_origins,
            nominal_joint_positions=nominal_joint_positions,
            nominal_block_position_local=torch.tensor(
                dynamic_instance.root_pose.xyz,
                device=device,
                dtype=torch.float32,
            ),
            nominal_block_orientation_wxyz=torch.tensor(
                rpy_xyz_to_quat_wxyz(dynamic_instance.root_pose.rpy),
                device=device,
                dtype=torch.float32,
            ),
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            ik_solvers=solvers,
            fixed_orientations_wxyz=fixed_orientations,
            control_bindings=tuple(control_bindings),
            physics_state_port=physics_state_port,
            viewport_reconfigure=viewport_reconfigure,
        )
    except BaseException as exc:
        _rollback_partial_assembly(
            exc,
            solvers=solvers,
            robot_ports=robot_ports,
            object_port=object_port,
            physics_state_port=physics_state_port,
            raw_views=raw_views,
            session=session,
        )
        raise


def session_spec_from_config(
    config: object,
    *,
    num_envs: int | None = None,
    viewport: object | None = None,
) -> IsaacSessionSpec:
    """把 strict Kaleidoscope config 无损投影到 canonical Isaac Session spec。"""

    final_count = config.environments.num_envs if num_envs is None else num_envs
    # 这个函数也是可直接调用的 composition port，不能依赖上层 facade 已做过校验。
    # 精确类型检查防止 True 和 1.5 经 int() 静默变成一个环境。
    if type(final_count) is not int or final_count < 1:
        raise ValueError("num_envs must be a positive int")

    physics_source = config.physics
    selection = (str(physics_source.engine), str(physics_source.execution))
    if selection == ("physx", "cuda"):
        physics = IsaacPhysxCudaSpec(
            enable_scene_query_support=False,
        )
    elif selection == ("newton", "cuda"):
        physics = IsaacNewtonCudaSpec(
            world_count=final_count,
            nconmax_per_world=int(physics_source.nconmax_per_world),
            njmax_per_world=int(physics_source.njmax_per_world),
            use_cuda_graph=bool(physics_source.use_cuda_graph),
            substeps=int(physics_source.substeps),
            iterations=int(physics_source.iterations),
            line_search_iterations=int(physics_source.line_search_iterations),
            constraint_solver=str(physics_source.constraint_solver),
            contact_pipeline=str(physics_source.contact_pipeline),
        )
    else:
        raise ValueError(
            "Kaleidoscope session requires PhysX CUDA or Newton CUDA config"
        )
    physics_dt = 1.0 / float(config.scene.physics_frequency_hz)
    if viewport is None:
        rendering_dt = physics_dt
        app_spec = IsaacAppSpec(
            gui=False,
            hide_ui=True,
            disable_viewport_updates=True,
            fast_shutdown=True,
        )
        render_spec = IsaacRenderSpec(enabled=False)
    else:
        selected_env = int(getattr(viewport, "selected_env"))
        if selected_env < 0 or selected_env >= final_count:
            raise ValueError(
                "viewport.selected_env must be within the final environment batch"
            )
        action_ticks = int(config.task.action.physics_ticks_per_action)
        render_interval = int(getattr(viewport, "render_every_n_steps"))
        rendering_dt = physics_dt * action_ticks * render_interval
        app_spec = IsaacAppSpec(
            # 选择 viewport profile 就表示进入显式 human-viewer 产品边界；训练
            # composition 不加载该 profile，也不会创建窗口。
            gui=True,
            hide_ui=False,
            disable_viewport_updates=False,
            # standalone viewer 在成功 marker flush 后交给 Isaac 6 fast shutdown；
            # SimulationApp 会在 Notebook 内自行强制改回 graceful shutdown。
            fast_shutdown=True,
            material_sync_loads=True,
            hydra_material_sync_loads=True,
        )
        render_spec = IsaacRenderSpec(
            enabled=True,
            width=int(getattr(viewport, "width")),
            height=int(getattr(viewport, "height")),
            window_width=int(getattr(viewport, "window_width")),
            window_height=int(getattr(viewport, "window_height")),
            renderer=str(getattr(viewport, "renderer")),
            anti_aliasing=int(getattr(viewport, "anti_aliasing")),
            samples_per_pixel_per_frame=int(
                getattr(viewport, "samples_per_pixel_per_frame")
            ),
            denoiser=bool(getattr(viewport, "denoiser")),
            visible_world_indices=(selected_env,),
        )
    return IsaacSessionSpec(
        experience_family="kaleidoscope",
        compute=IsaacComputeSpec(cuda_device=int(config.compute.cuda_device)),
        physics=physics,
        physics_dt=physics_dt,
        rendering_dt=rendering_dt,
        gravity_z=float(config.scene.gravity_z),
        add_ground=bool(config.scene.add_ground),
        ground_height=float(config.scene.ground_height),
        app=app_spec,
        render=render_spec,
    )


def _configure_kaleidoscope_viewport(
    *,
    stage: object,
    replicated: object,
    viewport: object,
) -> Callable[[], None]:
    """只显示选中 env，并把 profile 的 env-local 视角平移到其 world origin。"""

    from pxr import Sdf, UsdGeom

    from linkerbot_sim.isaac.world import configure_visuals

    selected = int(getattr(viewport, "selected_env"))
    roots = tuple(getattr(replicated, "env_root_paths"))
    origins = getattr(replicated, "env_origins")
    shape = tuple(getattr(origins, "shape", ()))
    if selected < 0 or selected >= len(roots) or shape != (len(roots), 3):
        raise RuntimeError("Kaleidoscope viewport selection differs from scene layout")
    for env_id, path in enumerate(roots):
        prim = stage.GetPrimAtPath(Sdf.Path(path))
        if prim is None or not bool(prim.IsValid()):
            # Newton 只为选中 world 物化 renderer clone；其它 world 留在 CUDA model。
            if env_id == selected:
                raise RuntimeError(
                    f"selected Kaleidoscope viewport root is missing: {path}"
                )
            continue
        imageable = UsdGeom.Imageable(prim)
        if env_id == selected:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    visuals = getattr(viewport, "visuals")
    camera = visuals.viewport
    offset = tuple(float(value) for value in origins[selected])
    if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
        raise RuntimeError("Kaleidoscope selected viewport origin is not finite")
    translated = replace(
        camera,
        eye=tuple(float(camera.eye[index] + offset[index]) for index in range(3)),
        target=tuple(float(camera.target[index] + offset[index]) for index in range(3)),
    )
    configure_visuals(
        replace(visuals, viewport=translated),
        configure_viewport=True,
    )

    def reconfigure_camera() -> None:
        """在 lightweight Kit 完成首个 update 后恢复同一个确定性 world view。"""

        from linkerbot_sim.visualization.viewport import set_camera_view

        set_camera_view(
            eye=translated.eye,
            target=translated.target,
            camera_prim_path=translated.prim_path,
        )

    return reconfigure_camera


def _create_device_ik_solver(
    *,
    robot: object,
    settings: object,
    cuda_device: int,
    device: object,
    root_pose: object,
) -> object:
    """创建 context/raw adapter/env-frame wrapper 的唯一所有权链。"""

    import torch

    from linkerbot_sim.backends.curobo.kinematics import (
        CuroboDeviceBatchIKSolver,
        create_kinematics_context,
    )

    context = create_kinematics_context(
        robot_profile=robot.profile,
        settings=settings,
        cuda_device=cuda_device,
    )
    try:
        raw_solver = CuroboDeviceBatchIKSolver(
            context,
            tcp_frame_name=robot.tcp_frame_name,
            command_joint_names=robot.command_joint_names,
        )
    except BaseException:
        context.close()
        raise
    return EnvLocalDeviceBatchIKSolver(
        raw_solver,
        robot_root_position_local=torch.tensor(
            root_pose.xyz,
            device=device,
            dtype=torch.float32,
        ),
        robot_root_orientation_wxyz=torch.tensor(
            rpy_xyz_to_quat_wxyz(root_pose.rpy),
            device=device,
            dtype=torch.float32,
        ),
    )


def _validate_root_config(config: object, *, num_envs: int) -> None:
    if getattr(config, "mode", None) != "kaleidoscope":
        raise ValueError("scene assembly requires KaleidoscopeConfig")
    physics = getattr(config, "physics", None)
    selection = (
        str(getattr(physics, "engine", "")),
        str(getattr(physics, "execution", "")),
    )
    if selection not in {("physx", "cuda"), ("newton", "cuda")}:
        raise ValueError("scene assembly requires PhysX CUDA or Newton CUDA")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive int")


def _dynamic_object_instance(config: object) -> object:
    matches = tuple(
        item for item in config.scene.objects if item.name == config.task.dynamic_object
    )
    if len(matches) != 1:
        raise RuntimeError(
            "task.dynamic_object must resolve to exactly one scene object"
        )
    return matches[0]


def _rollback_partial_assembly(
    primary: BaseException,
    *,
    solvers: Mapping[str, object],
    robot_ports: list[object],
    object_port: object | None,
    physics_state_port: object | None,
    raw_views: list[object],
    session: object | None,
) -> None:
    """尽力回滚且永远不覆盖主构造异常。"""

    covered_views: set[int] = set()

    def attempt(label: str, callback: Callable[[], object]) -> None:
        try:
            callback()
        except BaseException as cleanup_error:
            primary.add_note(
                f"{label} cleanup failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )

    for label, solver in reversed(tuple(solvers.items())):
        close = getattr(solver, "close", None)
        if callable(close):
            attempt(f"IK solver {label}", close)
    if physics_state_port is not None:
        attempt("physics state port", physics_state_port.close)
    if object_port is not None:
        covered_views.add(id(object_port.view))
        attempt("dynamic object port", object_port.close)
    for port in reversed(robot_ports):
        covered_views.update((id(port.view), id(port.tcp_view)))
        attempt(f"robot port {port.label}", port.close)
    for view in reversed(raw_views):
        if id(view) in covered_views:
            continue
        close = getattr(view, "close", None)
        if callable(close):
            attempt("raw tensor view", close)
            continue
        invalidate = getattr(view, "invalidate", None)
        if callable(invalidate):
            attempt("raw tensor view", invalidate)
    if session is not None:
        # fast-shutdown Kit 会在 native close 中直接结束解释器，因此失败清理必须明确传
        # 非零状态，避免构造失败被 shell/CI 误判成成功。先输出主异常，因为真实 App.close
        # 不会返回到随后的裸 ``raise``；可返回的 fake Session 仍保留原异常对象和 note。
        traceback.print_exception(primary)
        sys.stderr.flush()
        print(
            f"KALEIDOSCOPE_SCENE_ASSEMBLY_FAILED {type(primary).__name__}: {primary}",
            flush=True,
        )
        attempt("IsaacSession", lambda: session.close(exit_code=1))


def _copy_startup_metadata_to_cuda(*, value: object, device: object):
    """把 PhysX 只在初始化期公开的结构元数据复制到常驻 CUDA buffer。

    Isaac Sim 6 即使 articulation view 使用 ``backend="torch", device="cuda"``，
    ``get_dof_limits()`` 仍可能返回 CPU tensor。关节上下限不是逐 step 的物理状态；允许
    在场景装配时做唯一一次 H2D，随后 ``joint_lower/joint_upper`` 始终留在 GPU。热路径
    state/pose/velocity 仍必须经过 :func:`as_torch_cuda` 的零拷贝严格门禁。
    """

    import torch

    if isinstance(value, torch.Tensor):
        source = value
    elif hasattr(value, "__dlpack__"):
        source = torch.from_dlpack(value)
    else:
        source = torch.as_tensor(value)
    return source.to(device=device, dtype=torch.float32).contiguous()


__all__ = ["build_kaleidoscope_scene_assembly", "session_spec_from_config"]
