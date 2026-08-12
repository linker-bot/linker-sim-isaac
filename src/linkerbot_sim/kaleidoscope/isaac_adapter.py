"""严格配置、Isaac scene assembly 与 KaleidoscopeRuntime 的 composition root。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import sys
import traceback

import torch

from linkerbot_sim.configuration import semantic_config_fingerprint
from linkerbot_sim.kaleidoscope.actions import (
    ActionMode,
    IKRuntimeAction,
    JointDeltaActionTerm,
    JointDeltaRuntimeAction,
    JointControlRuntimeAction,
    KinematicsRobotBinding,
    LinearRuntimeAction,
    action_spec_from_configuration,
)
from linkerbot_sim.kaleidoscope.control_mode import (
    KaleidoscopeControlBinding,
    KaleidoscopeControlModeCoordinator,
)
from linkerbot_sim.kaleidoscope.ik import DeviceBatchIKSolver
from linkerbot_sim.kaleidoscope.isaac_views import (
    KaleidoscopeTensorViews,
    PhysicsStateTensorPort,
    RigidObjectTensorPort,
    RobotTensorPort,
)
from linkerbot_sim.kaleidoscope.runtime import KaleidoscopeRuntime
from linkerbot_sim.kaleidoscope.state_api import KaleidoscopeStateAPI
from linkerbot_sim.kaleidoscope.tasks.tblock_push_v1 import (
    TBlockPushV1,
    TBlockPushV1Settings,
)


@dataclass(frozen=True, slots=True)
class KaleidoscopeSceneAssembly:
    """scene builder 交给 mode composition root 的全部、且仅有的资源。"""

    session: object
    robot_ports: tuple[RobotTensorPort, ...]
    object_port: RigidObjectTensorPort
    env_origins: torch.Tensor
    nominal_joint_positions: torch.Tensor
    nominal_block_position_local: torch.Tensor
    nominal_block_orientation_wxyz: torch.Tensor
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    ik_solvers: Mapping[str, DeviceBatchIKSolver]
    fixed_orientations_wxyz: Mapping[str, torch.Tensor]
    control_bindings: tuple[KaleidoscopeControlBinding, ...] = ()
    physics_state_port: PhysicsStateTensorPort | None = None
    viewport_reconfigure: Callable[[], None] | None = None


SceneAssemblyFactory = Callable[..., KaleidoscopeSceneAssembly]


def create_kaleidoscope_runtime(
    *,
    config: object,
    num_envs: int | None = None,
    viewport: object | None = None,
    assembly_factory: SceneAssemblyFactory | None = None,
) -> KaleidoscopeRuntime:
    """构造配置指定的 CUDA physics runtime；失败时逆序释放已取得资源。"""

    if getattr(config, "mode", None) != "kaleidoscope":
        raise ValueError("expected KaleidoscopeConfig")
    physics_selection = (
        str(getattr(config.physics, "engine", "")),
        str(getattr(config.physics, "execution", "")),
    )
    runtime_kind_by_selection = {
        ("physx", "cuda"): "physx_cuda",
        ("newton", "cuda"): "newton_cuda",
    }
    backend_kind = runtime_kind_by_selection.get(physics_selection)
    if backend_kind is None:
        raise ValueError(
            "Kaleidoscope scene assembly requires PhysX CUDA or Newton CUDA"
        )
    count = config.environments.num_envs if num_envs is None else num_envs
    # bool 是 int 的子类，浮点数也可被 int() 截断；composition 边界必须拒绝这两种
    # 隐式转换，否则调用方请求的 batch 形状会在创建 CUDA 资源前悄悄改变。
    if type(count) is not int or count < 1:
        raise ValueError("num_envs must be a positive int")
    if assembly_factory is None:
        from linkerbot_sim.kaleidoscope.scene_assembly import (
            build_kaleidoscope_scene_assembly,
        )

        assembly_factory = build_kaleidoscope_scene_assembly
    assembly = assembly_factory(
        config=config,
        num_envs=count,
        viewport=viewport,
    )
    views: KaleidoscopeTensorViews | None = None
    action_term = None
    try:
        physics_runtime = getattr(assembly.session, "physics_runtime", None)
        if getattr(physics_runtime, "kind", None) != backend_kind:
            raise RuntimeError(
                "scene assembly physics runtime differs from strict config: "
                f"actual={getattr(physics_runtime, 'kind', None)!r}, "
                f"expected={backend_kind!r}"
            )
        views = KaleidoscopeTensorViews(
            robot_ports=assembly.robot_ports,
            object_port=assembly.object_port,
            env_origins=assembly.env_origins,
            physics_state_port=assembly.physics_state_port,
        )
        command_dims = tuple(port.command_dim for port in assembly.robot_ports)
        robot_labels = tuple(port.label for port in assembly.robot_ports)
        action_spec = action_spec_from_configuration(
            config.task.action,
            robot_labels=robot_labels,
            command_dims=command_dims,
        )
        if (
            action_spec.mode
            in {
                ActionMode.JOINT_CONTROL,
                ActionMode.JOINT_DELTA,
            }
            and assembly.ik_solvers
        ):
            raise RuntimeError("joint actions must not allocate cuRobo IK solvers")
        settings = TBlockPushV1Settings.from_configuration(config.task)
        task = TBlockPushV1(
            num_envs=count,
            command_dim=views.command_dim,
            action_dim=action_spec.action_dim,
            robot_count=len(assembly.robot_ports),
            device=views.device,
            dtype=torch.float32,
            nominal_joint_positions=assembly.nominal_joint_positions,
            nominal_block_position=assembly.nominal_block_position_local,
            nominal_block_orientation_wxyz=assembly.nominal_block_orientation_wxyz,
            settings=settings,
        )
        supported_modes = (
            ("position", "velocity", "effort")
            if action_spec.mode is ActionMode.JOINT_CONTROL
            else ("position", "velocity")
        )
        control_mode = KaleidoscopeControlModeCoordinator(
            views=views,
            bindings=assembly.control_bindings,
            supported_modes=supported_modes,
        )
        action_term = _action_term(
            config=config,
            spec=action_spec,
            views=views,
            assembly=assembly,
            effort_limits=control_mode.command_effort_limits,
        )
        state_api = KaleidoscopeStateAPI(
            views.state_bindings(task.state_fields()),
            num_envs=count,
            rng_fields=("rng.key", "rng.counter"),
            compatibility_fingerprint=semantic_config_fingerprint(config),
        )
        state_api.bind_control_mode_provider(control_mode.get_mode)
        return KaleidoscopeRuntime(
            session=assembly.session,
            views=views,
            action_term=action_term,
            task=task,
            state_api=state_api,
            control_mode_coordinator=control_mode,
            ik_failure_penalty=settings.ik_failure_penalty,
            viewport=viewport,
            viewport_reconfigure=assembly.viewport_reconfigure,
        )
    except BaseException as primary:
        construction_error = primary

        def attempt(label: str, callback) -> None:
            try:
                callback()
            except BaseException as cleanup_error:
                construction_error.add_note(
                    f"{label} cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

        if action_term is not None:
            attempt("action term", action_term.close)
        else:
            attempt(
                "assembly IK solvers",
                lambda: _close_assembly_solvers(assembly.ik_solvers),
            )
        if views is not None:
            attempt("Kaleidoscope views", views.close)
        else:
            for port in (*assembly.robot_ports, assembly.object_port):
                attempt(f"tensor port {port.label}", port.close)
        # 真实 fast-shutdown Kit 会在 native close 中结束进程；构造失败必须使用非零
        # 状态，防止 CI 把失败清理误判为成功。关闭前先输出原始异常，因为 native close
        # 可能不返回；可返回的测试 Session 仍由裸 raise 保留同一个异常对象。
        traceback.print_exception(construction_error)
        sys.stderr.flush()
        print(
            "KALEIDOSCOPE_RUNTIME_CREATE_FAILED "
            f"{type(construction_error).__name__}: {construction_error}",
            flush=True,
        )
        attempt("IsaacSession", lambda: assembly.session.close(exit_code=1))
        raise


def _close_assembly_solvers(solvers: Mapping[str, DeviceBatchIKSolver]) -> None:
    """factory 接管 action term 前失败时，关闭 assembly 暂存的唯一 solver owners。"""

    first_error: BaseException | None = None
    closed: set[int] = set()
    for solver in reversed(tuple(solvers.values())):
        if id(solver) in closed:
            continue
        close = getattr(solver, "close", None)
        try:
            if not callable(close):
                raise TypeError("assembly IK solver must implement close()")
            close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            else:
                first_error.add_note(
                    f"additional solver close failure: {type(exc).__name__}: {exc}"
                )
        else:
            closed.add(id(solver))
    if first_error is not None:
        raise first_error


def _action_term(
    *,
    config: object,
    spec: object,
    views: KaleidoscopeTensorViews,
    assembly: KaleidoscopeSceneAssembly,
    effort_limits: torch.Tensor,
):
    physics_dt = 1.0 / float(config.scene.physics_frequency_hz)
    if spec.mode in {ActionMode.JOINT_CONTROL, ActionMode.JOINT_DELTA}:
        scale = torch.full_like(assembly.joint_lower, spec.scale)
        assert spec.clip is not None
        controller = JointDeltaActionTerm(
            lower=assembly.joint_lower,
            upper=assembly.joint_upper,
            scale=scale,
            clip=spec.clip,
            num_envs=views.num_envs,
            target=views.command_targets,
        )
        if spec.mode is ActionMode.JOINT_CONTROL:
            return JointControlRuntimeAction(
                controller,
                physics_ticks_per_action=spec.physics_ticks_per_action,
                velocity_scale_rad_s=spec.velocity_scale,
                effort_limits=effort_limits,
                effort_limit_fraction=spec.effort_limit_fraction,
                physics_dt=physics_dt,
            )
        return JointDeltaRuntimeAction(
            controller,
            physics_ticks_per_action=spec.physics_ticks_per_action,
            physics_dt=physics_dt,
            reference_velocity_limit_rad_s=spec.reference_velocity_limit,
        )
    bindings = _kinematics_bindings(spec, views=views, assembly=assembly)
    if spec.mode in {
        ActionMode.EE_LINEAR_PATH_POSITION,
        ActionMode.EE_LINEAR_PATH_FULL,
    }:
        return LinearRuntimeAction(
            spec=spec,
            bindings=bindings,
            command_dim=views.command_dim,
            physics_dt=physics_dt,
        )
    return IKRuntimeAction(
        spec=spec,
        bindings=bindings,
        command_dim=views.command_dim,
        physics_dt=physics_dt,
    )


def _kinematics_bindings(
    spec: object,
    *,
    views: KaleidoscopeTensorViews,
    assembly: KaleidoscopeSceneAssembly,
) -> tuple[KinematicsRobotBinding, ...]:
    action_slices = spec.robot_slices()
    command_slices = {item.label: item.columns for item in views.robot_columns}
    bindings = []
    for tcp_index, label in enumerate(spec.robot_labels):
        try:
            solver = assembly.ik_solvers[label]
        except KeyError as exc:
            raise RuntimeError(f"missing device IK solver for robot {label!r}") from exc
        bindings.append(
            KinematicsRobotBinding(
                label=label,
                action_slice=action_slices[label],
                command_slice=command_slices[label],
                tcp_index=tcp_index,
                solver=solver,
                fixed_orientation_wxyz=assembly.fixed_orientations_wxyz.get(label),
            )
        )
    return tuple(bindings)


__all__ = ["KaleidoscopeSceneAssembly", "create_kaleidoscope_runtime"]
