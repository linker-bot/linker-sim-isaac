"""Mirror 唯一 composition root。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from linkerbot_sim.configuration.physics import (
    NewtonCpuSettings,
    NewtonCudaSettings,
    PhysxCpuSettings,
)
from linkerbot_sim.configuration.modes.mirror import MirrorConfig
from linkerbot_sim.mirror.collision import MirrorCollisionOwner
from linkerbot_sim.mirror.controller import MirrorController
from linkerbot_sim.mirror.control_mode import (
    MirrorControlBinding,
    MirrorControlModeService,
)
from linkerbot_sim.mirror.hybrid_parameters import HybridParameterService
from linkerbot_sim.mirror.interface.admission import MirrorAdmissionQueue
from linkerbot_sim.mirror.motion import MirrorMotionOwner
from linkerbot_sim.mirror.rendering import CameraBundle, RenderCoordinator
from linkerbot_sim.mirror.reset import MirrorResetService
from linkerbot_sim.mirror.runtime import MirrorRuntime
from linkerbot_sim.mirror.snapshot import MirrorSnapshotService
from linkerbot_sim.mirror.state import MirrorStateService


@dataclass
class MirrorAssembly:
    """engine-aware 场景装配完成后交给产品根的资源集合。

    ``session`` 已持有 App/stage/physics runtime；其它字段只借用这些资源。assembly factory
    不能把 session close 权转交给 delegate 或任何 child。
    """

    session: object
    state_getter: Callable[[], Mapping[str, object]]
    state_setter: Callable[..., object]
    snapshot_capture: Callable[[], object]
    snapshot_restore: Callable[..., object]
    resetter: Callable[..., object]
    motion_backend: object
    collision_registry: object | None = None
    collision_contexts: tuple[object, ...] = ()
    cameras: tuple[object, ...] = ()
    camera_output: object | None = None
    outputs: tuple[object, ...] = ()
    controllers: tuple[object, ...] = ()
    views: tuple[object, ...] = ()
    scene_resources: object | None = None
    control_bindings: tuple[MirrorControlBinding, ...] = ()


def _backend_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw).casefold()


def _validate_physics_composition(config: MirrorConfig, session: object) -> None:
    physics = getattr(session, "physics_runtime", None)
    if physics is None:
        raise RuntimeError("Mirror assembly session 缺少 physics_runtime")
    backend = _backend_value(getattr(physics, "backend", ""))
    kind = str(getattr(physics, "kind", "")).casefold()
    execution = str(getattr(physics, "execution", "")).casefold()
    if isinstance(config.physics, PhysxCpuSettings):
        if backend != "physx" or kind != "physx_cpu":
            raise RuntimeError(
                "Mirror physx_cpu 配置得到错误 runtime："
                f"backend={backend!r}, kind={kind!r}"
            )
        return
    if not isinstance(config.physics, (NewtonCpuSettings, NewtonCudaSettings)):
        raise RuntimeError("MirrorConfig 包含未授权物理后端")
    expected_execution = config.physics.execution
    expected_kind = f"newton_{expected_execution}"
    if backend != "newton" or kind != expected_kind or execution != expected_execution:
        raise RuntimeError(
            "Mirror Newton 配置必须得到执行设备一致的项目 Newton runtime；"
            f"actual={backend}/{kind}/{execution}"
        )
    assert_single_world = getattr(physics, "assert_single_world", None)
    if not callable(assert_single_world):
        raise RuntimeError("Newton runtime 缺少 assert_single_world")
    assert_single_world(consumer="Mirror")


def _default_assembly_factory(config: MirrorConfig) -> MirrorAssembly:
    # 场景装配延迟导入，确保 ``import linkerbot_sim.mirror`` 不启动 Kit 或加载 Isaac。
    from linkerbot_sim.mirror.scene_assembly import build_mirror_assembly

    return build_mirror_assembly(config)


def _best_effort_rollback(assembly: MirrorAssembly) -> None:
    resources = (
        *assembly.outputs,
        assembly.camera_output,
        *assembly.cameras,
        assembly.motion_backend,
        assembly.collision_registry,
        *assembly.collision_contexts,
        *assembly.controllers,
        *assembly.views,
    )
    seen: set[int] = set()
    for resource in resources:
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                pass
    close_session = getattr(assembly.session, "close", None)
    if callable(close_session):
        try:
            close_session(exit_code=1)
        except TypeError:
            close_session()
        except BaseException:
            pass


def create_mirror_runtime(
    config: MirrorConfig,
    *,
    assembly_factory: Callable[[MirrorConfig], MirrorAssembly] | None = None,
) -> MirrorRuntime:
    """从 ``MirrorConfig`` 构造唯一 session/owner graph。

    admission 与 terminal history 的容量只能来自 strict interface profile；composition
    root 不提供调用参数默认值，避免 embedded/CLI 两条入口形成不同资源边界。
    """

    if not isinstance(config, MirrorConfig):
        raise TypeError("create_mirror_runtime 必须接收 MirrorConfig")
    factory = assembly_factory or _default_assembly_factory
    assembly = factory(config)
    try:
        _validate_physics_composition(config, assembly.session)
        state = MirrorStateService(
            getter=assembly.state_getter,
            setter=assembly.state_setter,
        )
        snapshots = MirrorSnapshotService(
            capture=assembly.snapshot_capture,
            restore=assembly.snapshot_restore,
        )
        reset = MirrorResetService(assembly.resetter)
        motion = MirrorMotionOwner(assembly.motion_backend)
        collision = MirrorCollisionOwner(
            registry=assembly.collision_registry,
            contexts=assembly.collision_contexts,
        )
        camera_bundle = (
            None
            if not assembly.cameras and assembly.camera_output is None
            else CameraBundle(
                cameras=assembly.cameras,
                output=assembly.camera_output,
            )
        )
        render_enabled = bool(
            config.outputs.render.enabled or camera_bundle is not None
        )
        rendering = (
            RenderCoordinator(
                physics_runtime=assembly.session.physics_runtime,
                cameras=camera_bundle,
            )
            if render_enabled
            else None
        )
        if rendering is not None:
            bind_render_frame = getattr(
                assembly.motion_backend,
                "bind_render_frame",
                None,
            )
            if callable(bind_render_frame):
                # Timeline 与 idle step 必须复用同一个 coordinator；尤其 Newton
                # 不能在 motion loop 中退回 manager.render() 的单 update 快捷路径。
                bind_render_frame(rendering.render_only)
        interface = config.control.interface
        admission = MirrorAdmissionQueue(
            capacity=interface.admission_capacity,
            terminal_capacity=interface.terminal_history_capacity,
        )
        control_mode = MirrorControlModeService(
            initial_mode=config.control.mode,
            bindings=assembly.control_bindings,
        )
        hybrid_parameters = HybridParameterService(config.hybrid_control)
        bind_hybrid_parameter_provider = getattr(
            assembly.motion_backend,
            "bind_hybrid_parameter_provider",
            None,
        )
        if callable(bind_hybrid_parameter_provider):
            bind_hybrid_parameter_provider(hybrid_parameters.snapshot)
        if assembly.scene_resources is not None and hasattr(
            assembly.scene_resources,
            "control_mode_state_provider",
        ):
            if assembly.scene_resources.control_mode_state_provider is not None:
                raise RuntimeError("Mirror snapshot control-mode provider 已绑定")
            assembly.scene_resources.control_mode_state_provider = control_mode.get_mode
        bind_control_mode_provider = getattr(
            assembly.motion_backend,
            "bind_control_mode_provider",
            None,
        )
        if callable(bind_control_mode_provider):
            bind_control_mode_provider(lambda: control_mode.get_mode().active_mode)
        controller = MirrorController(
            admission=admission,
            motion=motion,
            state=state,
            snapshots=snapshots,
            reset_service=reset,
            control_mode=control_mode,
            hybrid_parameters=hybrid_parameters,
        )
        return MirrorRuntime(
            config=config,
            session=assembly.session,
            controller=controller,
            state_service=state,
            snapshot_service=snapshots,
            reset_service=reset,
            motion=motion,
            collision=collision,
            control_mode=control_mode,
            rendering=rendering,
            outputs=assembly.outputs,
            controllers=assembly.controllers,
            views=assembly.views,
            scene_resources=assembly.scene_resources,
        )
    except BaseException:
        _best_effort_rollback(assembly)
        raise


__all__ = ["MirrorAssembly", "create_mirror_runtime"]
