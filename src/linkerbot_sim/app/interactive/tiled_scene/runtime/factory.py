"""Isaac TiledSceneRuntime 的构造、资源回滚与后端装配。"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.runtime.ik import (
    _WorldFrameBatchIKBackend,
    _create_isaac_ik_solvers,
)
from linkerbot_sim.app.interactive.tiled_scene.selectors import selected_robot_names
from linkerbot_sim.planning.backend import normalize_planner_backend
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.control.adapter import TiledCommandAdapter
from linkerbot_sim.tiled.planning.linear_backend import LinearJointPlannerBackend
from linkerbot_sim.tiled.planning.batching import normalize_joint_batch_mode
from linkerbot_sim.tiled.planning.manager import TiledPlannerManager
from linkerbot_sim.tiled.state.object_io import (
    capture_tiled_object_pose_snapshot,
)
from linkerbot_sim.tiled.state.object_views import (
    TiledDynamicChainObjectPoseView,
)
from linkerbot_sim.utils.output_paths import OutputPathPlan
from linkerbot_sim.tiled.playback.buffer import TiledTrajectoryBuffer

if TYPE_CHECKING:
    from linkerbot_sim.app.interactive.tiled_scene.runtime.core import (
        TiledSceneRuntime,
    )
    from linkerbot_sim.configs.runtime import SimulationAppSettings
    from linkerbot_sim.configs.runtime import (
        CameraOutputRuntimeSettings,
        PlaybackResourceSettings,
        ShutdownSettings,
    )


_RuntimeT = TypeVar("_RuntimeT", bound="TiledSceneRuntime")


def create_tiled_scene_runtime(
    runtime_type: type[_RuntimeT],
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
    planner_request_defaults: object | None = None,
    command_defaults: object | None = None,
    playback_settings: PlaybackResourceSettings | None = None,
    planner_shutdown_timeout_s: float = 30.0,
    planner_backend: str = "curobo",
    curobo_profile: str = "default",
    joint_batch_mode: str = "auto",
    additional_output_path_plans: Sequence[OutputPathPlan] = (),
) -> _RuntimeT:
    """完整创建 runtime；任一步失败都会逆序释放已经取得的资源。"""

    from linkerbot_sim.envs.settings import EnvRuntimeSettings
    from linkerbot_sim.app.runtime.simulation_session import (
        create_simulation_session,
    )
    from linkerbot_sim.configs.profiles import load_profile_yaml
    from linkerbot_sim.sensors.camera.observer import (
        open_prepared_camera_output,
        prepare_camera_output,
    )
    from linkerbot_sim.sensors.camera.runtime import (
        create_sensor_camera_runtimes,
        initialize_sensor_camera_runtimes,
    )
    from linkerbot_sim.tiled.scene.builder import build_isaac_tiled_scene
    from linkerbot_sim.tiled.scene.cameras import tiled_sensor_camera_settings
    from linkerbot_sim.tiled.scene.utils import _print_status
    from linkerbot_sim.tiled.scene.views import finalize_tiled_articulation_views
    from linkerbot_sim.utils.paths import repo_path
    from linkerbot_sim.utils.output_paths import apply_output_path_plans

    session: object | None = None
    camera_output: object | None = None
    planner_manager: TiledPlannerManager | None = None
    ik_solvers: dict[str, _WorldFrameBatchIKBackend] = {}
    try:
        tiled_config = TiledEnvConfig.from_env_config(env_config)
        selected_planner_backend = normalize_planner_backend(planner_backend)
        selected_joint_batch_mode = normalize_joint_batch_mode(joint_batch_mode)
        selected_curobo_profile = str(curobo_profile).strip()
        if not selected_curobo_profile:
            raise ValueError("curobo_profile must be a non-empty profile name")
        if playback_settings is None:
            from linkerbot_sim.configs.runtime import PlaybackResourceSettings

            playback_settings = PlaybackResourceSettings()
        runtime_settings = EnvRuntimeSettings.from_env_config(env_config)
        session = create_simulation_session(
            simulation_app=simulation_app,
            settings=runtime_settings,
        )
        scene = build_isaac_tiled_scene(
            world=session.world,
            stage=session.stage,
            env_config=env_config,
            tiled_config=tiled_config,
            controller_bundle=controller_bundle,
            status_prefix="TILED_SCENE_INTERACTIVE",
        )
        sensor_settings = tiled_sensor_camera_settings(
            runtime_settings.sensors,
            tiled_config=scene.config,
        )
        sensor_cameras = create_sensor_camera_runtimes(
            stage=session.stage,
            sensors=sensor_settings,
        )
        prepared_camera_output = prepare_camera_output(
            sensor_cameras,
            path_resolver=repo_path,
            settings=camera_output_settings,
            shutdown_timeout_s=shutdown_settings.camera_publisher_timeout_s,
        )
        apply_output_path_plans(
            list(prepared_camera_output.path_plans) + list(additional_output_path_plans)
        )
        camera_output = open_prepared_camera_output(prepared_camera_output)
        session.world.reset()
        session.world.get_physics_context().set_gravity(runtime_settings.gravity_z)
        scene = finalize_tiled_articulation_views(scene)
        initialize_sensor_camera_runtimes(sensor_cameras)
        for sensor_camera in sensor_cameras:
            _print_status(
                "TILED_SCENE_INTERACTIVE",
                "SENSOR_CAMERA "
                f"name={sensor_camera.name} prim_path={sensor_camera.prim_path} "
                f"modalities={','.join(sensor_camera.settings.modalities)}",
            )
        if simulation_app.gui:
            _refresh_gui_view_after_scene_build(session, runtime_settings)

        selected = selected_robot_names(scene.articulation_views, None)
        curobo_profile = load_profile_yaml(
            "curobo",
            selected_curobo_profile,
        )
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
        ik_solvers = _create_isaac_ik_solvers(
            scene,
            selected,
            curobo_profile=curobo_profile,
            cache_root=cache_root,
        )
        command_adapters = {
            name: TiledCommandAdapter(
                num_envs=scene.config.num_envs,
                command_dim=int(targets[name].shape[1]),
                default_decimation=max(1, int(default_decimation)),
                tcp_frame_name=(
                    None if name not in ik_solvers else ik_solvers[name].tcp_frame_name
                ),
                ik_solver=ik_solvers.get(name),
                failure_policy=failure_policy,
            )
            for name in selected
        }
        tcp_positions = {}
        tcp_orientations = {}
        for name, solver in ik_solvers.items():
            tcp_positions[name], tcp_orientations[name] = (
                solver.command_tcp_world_poses(targets[name])
            )
        object_pose_views = _create_tiled_object_pose_views(scene)
        planner_backend = _create_tiled_planner_backend(
            scene=scene,
            ik_solvers=ik_solvers,
            backend=selected_planner_backend,
            joint_batch_mode=selected_joint_batch_mode,
            curobo_profile=curobo_profile,
            cache_root=cache_root,
        )
        initial_object_states = capture_tiled_object_pose_snapshot(
            stage=session.stage,
            object_prim_paths=scene.object_prim_paths,
            env_origins=scene.env_origins,
            env_ids=np.arange(scene.config.num_envs, dtype=int),
            object_pose_views=object_pose_views,
        )
        planner_manager = TiledPlannerManager(
            backend=planner_backend,
            max_workers=int(planner_workers),
            max_pending_requests=max_pending_requests,
            max_completed_results=max_completed_results,
            max_batch_problems=max_batch_problems,
            oversize_request_policy=oversize_request_policy,
            shutdown_timeout_s=planner_shutdown_timeout_s,
        )
        return runtime_type(
            env_name=env_name,
            env_config=env_config,
            session=session,
            scene=scene,
            render=simulation_app.gui or camera_output is not None,
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
            trajectory_buffer=TiledTrajectoryBuffer(
                num_envs=scene.config.num_envs,
                max_queue_depth_per_env=playback_settings.max_queue_depth_per_env,
                max_samples_per_env=playback_settings.max_samples_per_env,
                max_duration_s_per_env=playback_settings.max_duration_s_per_env,
                overflow_policy=playback_settings.overflow_policy,
            ),
            planner_manager=planner_manager,
            sensor_cameras=sensor_cameras,
            camera_output=camera_output,
            quit_event=threading.Event(),
            object_pose_views=object_pose_views,
            planner_request_defaults=planner_request_defaults,
            command_defaults=command_defaults,
        )
    except BaseException:
        _close_runtime_resources(
            planner_manager=planner_manager,
            ik_solvers=ik_solvers,
            camera_output=camera_output,
            session=session,
            suppress_errors=True,
        )
        raise


def close_tiled_scene_runtime(
    runtime: "TiledSceneRuntime",
) -> bool:
    """按依赖顺序关闭 runtime；任何超时都保留 owner 状态供后续重试。"""

    if runtime._closed:
        return True
    first_error: BaseException | None = None
    live_resources: list[str] = []
    if not getattr(runtime, "_planner_closed", False):
        shutdown = getattr(runtime.planner_manager, "shutdown", None)
        if callable(shutdown):
            try:
                result = shutdown()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                if _shutdown_timed_out(result):
                    live_resources.append("planner")
                    print(
                        "TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT "
                        f"resource=planner result={result}",
                        flush=True,
                    )
                else:
                    runtime._planner_closed = True
        else:
            runtime._planner_closed = True
    if not getattr(runtime, "_camera_closed", False):
        close_camera = getattr(runtime.camera_output, "close", None)
        if callable(close_camera):
            try:
                result = close_camera()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                if _shutdown_timed_out(result):
                    live_resources.append("camera_output")
                    print(
                        "TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT "
                        f"resource=camera_output result={result}",
                        flush=True,
                    )
                else:
                    runtime._camera_closed = True
        else:
            runtime._camera_closed = True
    if not getattr(runtime, "_ik_closed", False):
        live_callbacks: list[str] = []
        callbacks = _ik_close_callbacks(runtime.ik_solvers)
        closed_owner_ids = getattr(runtime, "_closed_ik_owner_ids", set())
        runtime._closed_ik_owner_ids = closed_owner_ids
        for callback in callbacks:
            owner_id = _callback_owner_id(callback)
            if owner_id in closed_owner_ids:
                continue
            try:
                result = callback()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                continue
            if _shutdown_timed_out(result):
                live_callbacks.append(getattr(callback, "__qualname__", repr(callback)))
            else:
                closed_owner_ids.add(owner_id)
        if live_callbacks:
            print(
                "TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT "
                f"resource=ik live_callbacks={live_callbacks}",
                flush=True,
            )
            live_resources.extend(f"ik:{name}" for name in live_callbacks)
        if len(closed_owner_ids) == len(callbacks):
            runtime._ik_closed = True
    children_closed = bool(
        getattr(runtime, "_planner_closed", False)
        and getattr(runtime, "_camera_closed", False)
        and getattr(runtime, "_ik_closed", False)
    )
    if children_closed and not getattr(runtime, "_app_closed", False):
        from linkerbot_sim.app.runtime.simulation_app_lifecycle import (
            close_simulation_app,
        )

        app = getattr(runtime.session, "app", None)
        if app is not None:
            try:
                close_simulation_app(app)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                runtime._app_closed = True
        else:
            runtime._app_closed = True
    if first_error is not None:
        raise first_error
    if live_resources or not children_closed:
        return False
    runtime._closed = True
    return True


def _close_runtime_resources(
    *,
    planner_manager: object | None,
    ik_solvers: Mapping[str, object],
    camera_output: object | None,
    session: object | None,
    suppress_errors: bool,
) -> bool:
    """释放已创建资源，并确保某个 close 失败时仍继续关闭其余资源。"""

    from linkerbot_sim.app.runtime.simulation_app_lifecycle import (
        close_simulation_app,
    )

    callbacks = _ik_close_callbacks(ik_solvers)
    app = None if session is None else getattr(session, "app", None)

    first_error: BaseException | None = None
    timed_out_resources = False
    resources: list[tuple[str, Callable[[], object]]] = []
    shutdown = getattr(planner_manager, "shutdown", None)
    if callable(shutdown):
        resources.append(("planner", shutdown))
    close_camera = getattr(camera_output, "close", None)
    if callable(close_camera):
        resources.append(("camera_output", close_camera))
    resources.extend(
        (getattr(callback, "__qualname__", repr(callback)), callback)
        for callback in callbacks
    )
    if app is not None:
        resources.append(("simulation_app", lambda: close_simulation_app(app)))
    for name, callback in resources:
        try:
            result = callback()
            timed_out = _shutdown_timed_out(result)
            if timed_out:
                timed_out_resources = True
                print(
                    "TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT "
                    f"resource={name} result={result}",
                    flush=True,
                )
        except BaseException as exc:  # 关闭必须尽力完成后续资源
            if first_error is None:
                first_error = exc
    if first_error is not None and not suppress_errors:
        raise first_error
    if first_error is not None:
        return False
    if timed_out_resources:
        return False
    return True


def _shutdown_timed_out(result: object) -> bool:
    """统一判定 planner、camera 与测试资源返回的关闭超时结果。

    布尔 ``False`` 表示资源未能关闭；映射结果仅读取其 ``shutdown_timed_out`` 标志，
    其余返回值均视为正常完成。
    """

    if result is False:
        return True
    if isinstance(result, Mapping):
        return bool(result.get("shutdown_timed_out", False))
    return False


def _ik_close_callbacks(
    ik_solvers: Mapping[str, object],
) -> list[Callable[[], object]]:
    """返回去重后的 IK solver/context 关闭回调。"""

    callbacks: list[Callable[[], object]] = []
    seen: set[int] = set()
    for world_solver in ik_solvers.values():
        solver = getattr(world_solver, "solver", world_solver)
        close_owner = solver
        close = getattr(close_owner, "close", None)
        if not callable(close):
            close_owner = getattr(solver, "context", None)
            close = getattr(close_owner, "close", None)
        if not callable(close) or close_owner is None or id(close_owner) in seen:
            continue
        seen.add(id(close_owner))
        callbacks.append(close)
    return callbacks


def _callback_owner_id(callback: Callable[[], object]) -> int:
    """返回关闭回调所属对象的稳定标识，供超时后的去重重试使用。

    每次访问绑定方法都会产生新的方法对象，因此优先取 ``__self__``，普通 callable
    才以自身身份作为标识。
    """

    return id(getattr(callback, "__self__", callback))


def _create_tiled_object_pose_views(scene: object) -> dict[str, object]:
    """为动态 tiled objects 创建用于 state/reset/snapshot 的 batched rigid view。"""

    object_paths = getattr(scene, "object_prim_paths", {}) or {}
    rigid_objects: list[tuple[str, tuple[str, ...]]] = []
    chain_objects: list[
        tuple[
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, ...], ...],
            str | None,
        ]
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
            body_names, body_suffixes = _dynamic_chain_body_suffixes(
                name=name,
                model=model,
                env_zero_root_path=paths[0],
            )
            body_paths_by_env = tuple(
                tuple(
                    _join_prim_path_suffix(root_path, suffix)
                    for suffix in body_suffixes
                )
                for root_path in paths
            )
            state_summary = getattr(handle, "state_summary", None)
            reference_body = getattr(state_summary, "reference_body", None)
            if (
                not isinstance(reference_body, str)
                or body_names.count(reference_body) != 1
            ):
                raise ValueError(
                    f"tiled dynamic-chain object {name!r} state_summary.reference_body "
                    f"must identify exactly one body; got {reference_body!r} in "
                    f"{body_names!r}"
                )
            chain_objects.append(
                (name, paths, body_names, body_paths_by_env, reference_body)
            )
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
    for name, _, body_names, body_paths_by_env, reference_body in chain_objects:
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
            reference_body=reference_body,
        )
    return result


def _dynamic_chain_body_suffixes(
    *,
    name: str,
    model: object,
    env_zero_root_path: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """从 env_0 dynamic-chain model 推导 child rigid body 的相对路径。"""

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
        body_name = str(
            name_getter() if callable(name_getter) else body_path.rsplit("/", 1)[-1]
        )
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
    *,
    scene: object,
    ik_solvers: dict[str, _WorldFrameBatchIKBackend],
    backend: str,
    joint_batch_mode: str,
    curobo_profile: Mapping[str, object],
    cache_root: str | None = None,
) -> object:
    """按已校验的 TiledSceneRuntime 配置创建异步 planner backend。"""

    if backend == "linear":
        return LinearJointPlannerBackend()
    if backend == "curobo":
        return _create_curobo_tiled_planner_backend(
            scene=scene,
            ik_solvers=ik_solvers,
            joint_batch_mode=joint_batch_mode,
            curobo_profile=curobo_profile,
            cache_root=cache_root,
        )
    raise ValueError(f"Unsupported tiled planner backend: {backend!r}")


def _create_curobo_tiled_planner_backend(
    *,
    scene: object,
    ik_solvers: dict[str, _WorldFrameBatchIKBackend],
    joint_batch_mode: str,
    curobo_profile: Mapping[str, object],
    cache_root: str | None = None,
) -> object:
    """创建 cuRobo async tiled planner backend。"""

    from linkerbot_sim.backends.curobo import (
        CuroboMotionPlanner,
        robot_curobo_config,
    )
    from linkerbot_sim.backends.curobo.context import CuroboContext
    from linkerbot_sim.configs.profiles import load_profile_yaml
    from linkerbot_sim.tiled.planning.backends.curobo import (
        TiledCuroboPlanningBackend,
    )

    def _planner_factory(robot_name: str) -> object:
        """为每个 worker/request 创建独占 cuRobo context 与 scalar planner。"""

        robot = scene.robots[str(robot_name)]
        if str(robot_name) not in ik_solvers:
            raise RuntimeError(f"robot {robot_name!r} does not support cuRobo planning")
        robot_config = load_profile_yaml("robot", robot.profile_name)
        context = CuroboContext(
            robot_curobo_config(
                robot_config,
                curobo_profile=curobo_profile,
            ),
            cache_root=cache_root,
        )
        return CuroboMotionPlanner(
            context,
            tcp_frame_name=ik_solvers[str(robot_name)].tcp_frame_name,
        )

    return TiledCuroboPlanningBackend(
        _planner_factory,
        joint_batch_mode=joint_batch_mode,
    )


def _refresh_gui_view_after_scene_build(
    session: object, runtime_settings: object
) -> None:
    """场景 clone/reset 后重新应用 GUI 视角并 pump Kit 完成 viewport 初始化。"""

    from linkerbot_sim.envs.scene_builder import configure_visuals

    configure_visuals(runtime_settings.visuals)
    update = getattr(getattr(session, "app", None), "update", None)
    if not callable(update):
        return
    for _ in range(3):
        update()
