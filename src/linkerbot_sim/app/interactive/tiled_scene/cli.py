"""tiled interactive runtime 的 CLI、transport 与 telemetry 装配入口。"""

from __future__ import annotations

import argparse
import queue
import socketserver
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from linkerbot_sim.app.interactive.policies import (
    resolve_interactive_runtime_policy,
)
from linkerbot_sim.utils.json import strict_json_dumps
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.configs.runtime import (
    ResolvedRuntimeConfig,
    load_runtime_profile,
    resolve_runtime_config,
)

if TYPE_CHECKING:
    from linkerbot_sim.telemetry.tiled.config import TiledTelemetryConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 ``tiled_scene`` 交互脚本参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-profile", default="default_tiled_scene")
    parser.add_argument(
        "--dump-effective-config",
        action="store_true",
        help="print resolved values and their sources, then exit",
    )
    parser.add_argument("--env", default=None, help="env profile 名称")
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="打开 Isaac GUI",
    )
    parser.add_argument(
        "--default-decimation",
        type=int,
        default=None,
        help="action 未指定 decimation 时展开的 physics tick 数",
    )
    parser.add_argument(
        "--planner-workers",
        type=int,
        default=None,
        help="tiled async planner worker 数；planner 不访问 Isaac runtime，只消费状态快照",
    )
    parser.add_argument(
        "--max-pending-requests",
        type=int,
        default=None,
        help="最多允许同时排队/运行的 planner 请求数",
    )
    parser.add_argument(
        "--max-completed-results",
        type=int,
        default=None,
        help="planner completed result 缓存上限；设为 0 表示不保留 completed 摘要",
    )
    parser.add_argument(
        "--stdin",
        dest="stdin_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="关闭 stdin JSONL，只使用 TCP/telemetry 保持进程",
    )
    parser.add_argument(
        "--stdin-eof-policy",
        choices=("exit", "keep_alive"),
        default=None,
        help="stdin EOF 时退出或保持进程",
    )
    parser.add_argument(
        "--idle-physics-policy",
        choices=("pause", "hold_step"),
        default=None,
        help="空闲时暂停 physics 或保持 target 并推进",
    )
    parser.add_argument("--tcp-jsonl-host", default=None)
    parser.add_argument("--tcp-jsonl-port", type=int, default=None)
    parser.add_argument("--websocket-host", default=None)
    parser.add_argument("--websocket-port", type=int, default=None)
    parser.add_argument("--foxglove-live-host", default=None)
    parser.add_argument(
        "--foxglove-live-port",
        type=int,
        default=None,
        help="Foxglove live server port；tiled 日常调试建议 8767；不传则不开 telemetry live",
    )
    parser.add_argument("--foxglove-mcap-path", default=None)
    parser.add_argument(
        "--telemetry-env-ids",
        default=None,
        help="逗号分隔的 selected env ids，用于 tiled Foxglove/MCAP 输出",
    )
    parser.add_argument(
        "--telemetry-primary-env-id",
        type=int,
        default=None,
        help="标准 JointStates/scene topic 使用的 env id；必须属于 selected env ids",
    )
    parser.add_argument(
        "--telemetry-decimation",
        type=int,
        default=None,
        help="每隔多少 global step 发布一次 tiled telemetry；reset/set_state 总会发布",
    )
    parser.add_argument(
        "--telemetry-rate-hz",
        type=float,
        default=None,
        help="开启 Foxglove/MCAP 时的状态发布频率；设为 0 完全关闭 telemetry",
    )
    parser.add_argument(
        "--telemetry-full-batch-json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否发布 /tiled/state JSON payload",
    )
    parser.add_argument(
        "--telemetry-joint-states",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否为 runtime primary env 发布标准 JointStates",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """脚本入口。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    resolved, env_config = resolve_entry_config(args)
    if args.dump_effective_config:
        _dump_effective_config(args.runtime_profile, resolved)
        return
    print(
        "TILED_SCENE_INTERACTIVE_CONFIG "
        f"runtime_profile={args.runtime_profile} fingerprint={resolved.fingerprint}",
        flush=True,
    )
    run_interactive_mode(args, resolved=resolved, env_config=env_config)


def run_interactive_mode(
    args: argparse.Namespace,
    *,
    resolved: ResolvedRuntimeConfig | None = None,
    env_config: Mapping[str, object] | None = None,
) -> None:
    """创建 Tiled runtime，并装配 transport、telemetry 与主循环。"""

    if resolved is None or env_config is None:
        resolved, env_config = resolve_entry_config(args)
    policy = resolve_interactive_runtime_policy(
        stdin_eof_policy=resolved.interactive.stdin_eof_policy,
        idle_physics_policy=resolved.execution.idle_physics_policy,
        default_stdin_eof_policy="exit",
        default_idle_physics_policy="pause",
    )
    from linkerbot_sim.app.interactive.tiled_scene.telemetry_publish import (
        _create_telemetry,
        _runtime_num_envs,
    )
    from linkerbot_sim.app.interactive.tiled_scene.transport import (
        BoundedInteractiveRequestQueue,
        SharedTransportAdmission,
        _InteractiveControl,
        _InteractiveRequest,
        _quit_on_stdin_eof,
        combined_transport_status,
        run_interactive_loop,
        start_stdin_jsonl_reader,
        start_tcp_jsonl_server,
        start_websocket_server,
        stop_tcp_jsonl_server,
    )

    telemetry_config = _tiled_telemetry_config(resolved)
    telemetry_mcap_output_plan = None
    if telemetry_config is not None:
        from linkerbot_sim.telemetry.foxglove import prepare_mcap_output

        telemetry_mcap_output_plan = prepare_mcap_output(
            resolved.telemetry.mcap.path,
            existing_file_policy=resolved.output.mcap_existing_file_policy,
        )
    additional_output_path_plans = (
        () if telemetry_mcap_output_plan is None else (telemetry_mcap_output_plan,)
    )
    runtime = create_tiled_scene_runtime(
        env_name=resolved.profiles.env,
        env_config=env_config,
        simulation_app=resolved.simulation_app,
        camera_output_settings=resolved.camera_output,
        shutdown_settings=resolved.shutdown,
        default_decimation=resolved.execution.default_decimation,
        controller_bundle=resolved.profiles.controller_bundle,
        planner_workers=resolved.planner.resources.max_workers,
        max_pending_requests=resolved.planner.resources.max_pending_requests,
        max_completed_results=resolved.planner.resources.max_completed_results,
        max_batch_problems=cast(int, resolved.planner.resources.max_batch_problems),
        oversize_request_policy=resolved.planner.oversize_request_policy,
        failure_policy=resolved.planner.failure_policy,
        cache_root=resolved.paths.cache_root,
        planner_request_defaults=resolved.planner.request_defaults,
        command_defaults=resolved.execution.command_defaults,
        playback_settings=resolved.playback,
        planner_shutdown_timeout_s=(resolved.planner.resources.shutdown_timeout_s),
        planner_backend=resolved.planner.backend,
        curobo_profile=resolved.profiles.curobo,
        joint_batch_mode=resolved.planner.joint_batch_mode,
        additional_output_path_plans=additional_output_path_plans,
    )
    telemetry = None
    server: socketserver.ThreadingTCPServer | None = None
    websocket_server = None
    stdin_reader = None
    try:
        telemetry = _create_telemetry(
            telemetry_config,
            num_envs=_runtime_num_envs(runtime),
            live_host=resolved.telemetry.foxglove_live.host,
            live_port=(
                resolved.telemetry.foxglove_live.port
                if resolved.telemetry.foxglove_live.enabled
                else None
            ),
            mcap_path=resolved.telemetry.mcap.path,
            mcap_output_plan=telemetry_mcap_output_plan,
            output_paths_applied=telemetry_mcap_output_plan is not None,
        )
        runtime.telemetry_status_provider = (
            None if telemetry is None else telemetry.status
        )
        transport = resolved.interactive.transport
        admission = SharedTransportAdmission(max_connections=transport.max_connections)
        request_queue: queue.Queue[_InteractiveRequest | _InteractiveControl] = (
            BoundedInteractiveRequestQueue(capacity=transport.request_queue_capacity)
        )
        tcp = transport.tcp_jsonl
        if tcp.enabled:
            assert tcp.port is not None
            server = start_tcp_jsonl_server(
                request_queue,
                quit_event=runtime.quit_event,
                host=tcp.host,
                port=tcp.port,
                max_message_bytes=transport.max_message_bytes,
                max_connections=transport.max_connections,
                server_poll_interval_s=transport.server_poll_interval_s,
                response_poll_interval_s=transport.response_poll_interval_s,
                admission=admission,
            )
            print(
                f"TILED_SCENE_INTERACTIVE_TCP_JSONL host={tcp.host} port={tcp.port}",
                flush=True,
            )
        websocket = transport.websocket
        if websocket.enabled:
            assert websocket.port is not None
            websocket_server = start_websocket_server(
                request_queue,
                quit_event=runtime.quit_event,
                host=websocket.host,
                port=websocket.port,
                max_message_bytes=transport.max_message_bytes,
                max_connections=transport.max_connections,
                event_queue_capacity=transport.event_queue_capacity,
                startup_timeout_s=transport.startup_timeout_s,
                server_poll_interval_s=transport.server_poll_interval_s,
                response_poll_interval_s=transport.response_poll_interval_s,
                admission=admission,
            )
            print(
                "TILED_SCENE_INTERACTIVE_WEBSOCKET "
                f"host={websocket.host} port={websocket_server.bound_port}",
                flush=True,
            )

        def transport_status() -> dict[str, object]:
            """汇总当前请求队列和全部已启用远端 transport 的只读状态。"""

            return combined_transport_status(
                request_queue,
                tcp_server=server,
                websocket_server=websocket_server,
                admission=admission,
            )

        runtime.transport_status_provider = transport_status
        print("TILED_SCENE_INTERACTIVE_READY", flush=True)
        if resolved.interactive.stdin_enabled:
            stdin_reader = start_stdin_jsonl_reader(
                request_queue,
                max_message_bytes=transport.max_message_bytes,
                admission=admission,
                quit_on_eof=_quit_on_stdin_eof(
                    stdin_eof_policy=policy.stdin_eof_policy,
                    tcp_jsonl_port=tcp.port if tcp.enabled else None,
                    telemetry=telemetry,
                    keepalive_consumer_active=(
                        getattr(runtime, "camera_output", None) is not None
                        or websocket_server is not None
                    ),
                ),
            )
        run_interactive_loop(
            runtime,
            telemetry=telemetry,
            request_queue=request_queue,
            telemetry_rate_hz=resolved.telemetry.rate_hz,
            idle_physics_policy=policy.idle_physics_policy,
            idle_step_duration_s=resolved.execution.idle_step_duration_s,
            queue_poll_timeout_s=resolved.interactive.queue_poll_timeout_s,
            event_publisher=(
                None if websocket_server is None else websocket_server.publish_event
            ),
            transport_status_provider=transport_status,
        )
    finally:
        if stdin_reader is not None and not stdin_reader.stop(
            timeout_s=resolved.shutdown.transport_timeout_s
        ):
            print(
                "TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT "
                f"resource=stdin thread={stdin_reader.name}",
                flush=True,
            )
        if websocket_server is not None:
            websocket_status = websocket_server.stop(
                timeout_s=resolved.shutdown.transport_timeout_s
            )
            if websocket_status.get("thread_alive"):
                print(
                    "TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT "
                    f"resource=websocket status={websocket_status}",
                    flush=True,
                )
        if server is not None:
            tcp_status = stop_tcp_jsonl_server(
                server,
                timeout_s=resolved.shutdown.transport_timeout_s,
            )
            if (
                tcp_status.get("serve_thread_alive")
                or tcp_status.get("shutdown_thread_alive")
                or int(tcp_status.get("active_connections", 0)) > 0
            ):
                print(
                    "TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT "
                    f"resource=tcp_jsonl status={tcp_status}",
                    flush=True,
                )
        try:
            if telemetry is not None and not telemetry.close():
                print(
                    "TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT resource=telemetry "
                    f"status={telemetry.status()}",
                    flush=True,
                )
        finally:
            try:
                close = getattr(runtime, "close", None)
                if callable(close):
                    runtime_stopped = close()
                    if runtime_stopped is False:
                        status = getattr(runtime, "status", None)
                        print(
                            "TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT resource=runtime "
                            f"status={status() if callable(status) else None}",
                            flush=True,
                        )
            finally:
                print("TILED_SCENE_INTERACTIVE_EXIT", flush=True)


def resolve_entry_config(
    args: argparse.Namespace,
) -> tuple[ResolvedRuntimeConfig, Mapping[str, object]]:
    """在导入或创建 Isaac runtime 前解析 ``tiled_scene`` effective config。"""

    overrides = _cli_overrides(args)
    profile = load_runtime_profile(args.runtime_profile)
    env_name = profile.profiles.env if args.env is None else args.env
    env_config = load_profile_yaml("env", env_name)
    resolved = resolve_runtime_config(
        profile,
        cli_overrides=overrides,
        env_config=env_config,
        expected_mode="tiled_scene",
    )
    return resolved, env_config


def _cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    primary_env_id = args.telemetry_primary_env_id
    selected_env_ids: tuple[int, ...] | None = None
    if args.telemetry_env_ids is not None:
        from linkerbot_sim.telemetry.tiled.config import parse_env_ids

        selected_env_ids = parse_env_ids(args.telemetry_env_ids)
    overrides: dict[str, object] = {
        "profiles.env": args.env,
        "simulation_app.gui": args.gui,
        "execution.default_decimation": args.default_decimation,
        "planner.resources.max_workers": args.planner_workers,
        "planner.resources.max_pending_requests": args.max_pending_requests,
        "planner.resources.max_completed_results": args.max_completed_results,
        "interactive.stdin_enabled": args.stdin_enabled,
        "interactive.stdin_eof_policy": args.stdin_eof_policy,
        "execution.idle_physics_policy": args.idle_physics_policy,
        "interactive.transport.tcp_jsonl.host": args.tcp_jsonl_host,
        "interactive.transport.tcp_jsonl.port": args.tcp_jsonl_port,
        "interactive.transport.websocket.host": args.websocket_host,
        "interactive.transport.websocket.port": args.websocket_port,
        "telemetry.primary_env_id": primary_env_id,
        "telemetry.selected_env_ids": selected_env_ids,
        "telemetry.publish_decimation": args.telemetry_decimation,
        "telemetry.rate_hz": args.telemetry_rate_hz,
        "telemetry.include_state_json": args.telemetry_full_batch_json,
        "telemetry.include_joint_states": args.telemetry_joint_states,
        "telemetry.foxglove_live.host": args.foxglove_live_host,
        "telemetry.foxglove_live.port": args.foxglove_live_port,
        "telemetry.mcap.path": args.foxglove_mcap_path,
    }
    if args.tcp_jsonl_port is not None:
        overrides["interactive.transport.tcp_jsonl.enabled"] = True
    if args.websocket_port is not None:
        overrides["interactive.transport.websocket.enabled"] = True
    if args.foxglove_live_port is not None:
        overrides["telemetry.foxglove_live.enabled"] = True
    return overrides


def _tiled_telemetry_config(
    resolved: ResolvedRuntimeConfig,
) -> "TiledTelemetryConfig | None":
    """由解析完成的运行配置构造 tiled telemetry sink 配置。

    采样频率非正，或 JSON、JointStates、scene marker 三种输出均关闭时无需创建
    sink，返回 ``None``；否则原样传递已校验的 topic、筛选范围与错误策略。
    """

    telemetry = resolved.telemetry
    if telemetry.rate_hz <= 0.0 or not (
        telemetry.include_joint_states
        or telemetry.include_state_json
        or telemetry.include_scene_markers
    ):
        return None
    from linkerbot_sim.telemetry.foxglove import (
        FoxgloveTopicConfig,
    )
    from linkerbot_sim.telemetry.tiled.config import TiledTelemetryConfig

    return TiledTelemetryConfig(
        selected_env_ids=telemetry.selected_env_ids,
        primary_env_id=telemetry.primary_env_id,
        publish_decimation=telemetry.publish_decimation,
        include_full_batch_json=telemetry.include_state_json,
        include_standard_joint_states=telemetry.include_joint_states,
        include_scene_markers=telemetry.include_scene_markers,
        include_efforts=telemetry.include_efforts,
        include_objects=telemetry.include_objects,
        topics=FoxgloveTopicConfig(
            joint_states=telemetry.topics.joint_states,
            scene=telemetry.topics.scene,
            state=telemetry.topics.state,
        ),
        buffer_size=telemetry.buffer_size,
        drop_policy=telemetry.drop_policy,
        on_error=telemetry.on_error,
        shutdown_timeout_s=resolved.shutdown.state_publisher_timeout_s,
        mcap_existing_file_policy=resolved.output.mcap_existing_file_policy,
    )


def _dump_effective_config(
    runtime_profile: str,
    resolved: ResolvedRuntimeConfig,
) -> None:
    print(
        strict_json_dumps(
            {
                "runtime_profile": runtime_profile,
                "fingerprint": resolved.fingerprint,
                "effective": resolved.as_dict(),
                "sources": dict(resolved.sources),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def create_tiled_scene_runtime(**kwargs: object):
    """延迟导入 TiledSceneRuntime，保证配置 dry-run 不触发 runtime 装配。"""

    from linkerbot_sim.app.interactive.tiled_scene.runtime import (
        TiledSceneRuntime,
    )

    return TiledSceneRuntime.create(**kwargs)
