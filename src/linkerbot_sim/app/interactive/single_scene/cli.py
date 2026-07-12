"""普通 SingleSceneRuntime canonical 交互模式的 CLI 装配入口。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import cast

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 ``single_scene`` 交互、planner、transport 与 state stream 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-profile", default="default_single_scene")
    parser.add_argument(
        "--dump-effective-config",
        action="store_true",
        help="print resolved values and their sources, then exit",
    )
    parser.add_argument("--env", default=None)
    parser.add_argument("--curobo-profile", default=None)
    parser.add_argument(
        "--planner-backend",
        choices=("curobo", "linear"),
        default=None,
    )
    parser.add_argument("--logging-profile", default=None)
    parser.add_argument("--control-mode", default=None)
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--stdin-eof-policy",
        choices=("exit", "keep_alive"),
        default=None,
    )
    parser.add_argument(
        "--idle-physics-policy",
        choices=("pause", "hold_step"),
        default=None,
    )
    parser.add_argument("--tcp-jsonl-host", default=None)
    parser.add_argument("--tcp-jsonl-port", type=int, default=None)
    parser.add_argument("--websocket-host", default=None)
    parser.add_argument("--websocket-port", type=int, default=None)
    parser.add_argument("--state-rate-hz", type=float, default=None)
    parser.add_argument(
        "--state-include-efforts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--state-include-objects",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--foxglove-live-host", default=None)
    parser.add_argument("--foxglove-live-port", type=int, default=None)
    parser.add_argument("--foxglove-mcap-path", default=None)
    parser.add_argument(
        "--foxglove-joint-effort-field",
        choices=("none", "commanded", "measured", "applied"),
        default=None,
    )
    return parser.parse_args(argv)


def run_interactive_mode(
    args: argparse.Namespace,
    *,
    resolved: ResolvedRuntimeConfig | None = None,
    env_config: Mapping[str, object] | None = None,
) -> int:
    """创建 SingleSceneRuntime，运行交互主循环，并保证退出时释放 runtime。"""

    if resolved is None or env_config is None:
        resolved, env_config = resolve_entry_config(args)
    policy = resolve_interactive_runtime_policy(
        stdin_eof_policy=resolved.interactive.stdin_eof_policy,
        idle_physics_policy=resolved.execution.idle_physics_policy,
        default_stdin_eof_policy="exit",
        default_idle_physics_policy="hold_step",
    )
    from linkerbot_sim.app.interactive.single_scene.state_stream import (
        InteractiveStateStreamConfig,
    )
    from linkerbot_sim.telemetry.foxglove import (
        FoxgloveTopicConfig,
        prepare_mcap_output,
    )
    from linkerbot_sim.telemetry.foxglove_state import JointEffortField

    telemetry = resolved.telemetry
    telemetry_has_output = telemetry.rate_hz > 0.0 and (
        telemetry.include_joint_states
        or telemetry.include_state_json
        or telemetry.include_scene_markers
    )
    mcap_plan = (
        prepare_mcap_output(
            telemetry.mcap.path,
            existing_file_policy=resolved.output.mcap_existing_file_policy,
        )
        if telemetry_has_output
        else None
    )
    state_stream_config = InteractiveStateStreamConfig(
        rate_hz=telemetry.rate_hz,
        buffer_size=telemetry.buffer_size,
        drop_policy=telemetry.drop_policy,
        on_error=telemetry.on_error,
        include_joint_states=telemetry.include_joint_states,
        include_state_json=telemetry.include_state_json,
        include_scene_markers=telemetry.include_scene_markers,
        include_efforts=telemetry.include_efforts,
        include_objects=telemetry.include_objects,
        topics=FoxgloveTopicConfig(
            joint_states=telemetry.topics.joint_states,
            scene=telemetry.topics.scene,
            state=telemetry.topics.state,
        ),
        foxglove_live_host=telemetry.foxglove_live.host,
        foxglove_live_port=(
            telemetry.foxglove_live.port if telemetry.foxglove_live.enabled else None
        ),
        foxglove_mcap_path=telemetry.mcap.path,
        mcap_existing_file_policy=resolved.output.mcap_existing_file_policy,
        mcap_output_plan=mcap_plan,
        output_paths_applied=mcap_plan is not None,
        foxglove_joint_effort_field=cast(
            JointEffortField, telemetry.joint_effort_field
        ),
        shutdown_timeout_s=resolved.shutdown.state_publisher_timeout_s,
    )
    runtime = create_single_scene_runtime(
        env=resolved.profiles.env,
        env_config=env_config,
        simulation_app=resolved.simulation_app,
        camera_output_settings=resolved.camera_output,
        shutdown_settings=resolved.shutdown,
        output_settings=resolved.output,
        curobo_profile=resolved.profiles.curobo,
        logging_profile=resolved.profiles.logging,
        controller_bundle=resolved.profiles.controller_bundle,
        control_mode=resolved.execution.control_mode,
        cache_root=resolved.paths.cache_root,
        hold_app=policy.keeps_alive_on_stdin_eof,
        status_prefix="SINGLE_SCENE_INTERACTIVE",
        additional_output_path_plans=(() if mcap_plan is None else (mcap_plan,)),
    )

    tcp = resolved.interactive.transport.tcp_jsonl
    websocket = resolved.interactive.transport.websocket
    try:
        return run_single_scene_interactive_motion(
            runtime,
            stdin_enabled=resolved.interactive.stdin_enabled,
            tcp_jsonl_host=tcp.host,
            tcp_jsonl_port=tcp.port if tcp.enabled else None,
            websocket_host=websocket.host,
            websocket_port=websocket.port if websocket.enabled else None,
            planner_backend=resolved.planner.backend,
            policy=policy,
            interactive_settings=resolved.interactive,
            execution_settings=resolved.execution,
            planner_settings=resolved.planner,
            shutdown_settings=resolved.shutdown,
            state_stream_config=state_stream_config,
        )
    finally:
        shutdown_report = runtime.close()
        if shutdown_report is not None and not shutdown_report.stopped:
            print(
                "SINGLE_SCENE_INTERACTIVE_RUNTIME_SHUTDOWN_TIMEOUT "
                f"live_resources={list(shutdown_report.live_resources)}",
                flush=True,
            )


def resolve_entry_config(
    args: argparse.Namespace,
) -> tuple[ResolvedRuntimeConfig, Mapping[str, object]]:
    """在导入或创建 Isaac runtime 前解析 ``single_scene`` effective config。"""

    overrides = _cli_overrides(args)
    profile = load_runtime_profile(args.runtime_profile)
    env_name = profile.profiles.env if args.env is None else args.env
    env_config = load_profile_yaml("env", env_name)
    resolved = resolve_runtime_config(
        profile,
        cli_overrides=overrides,
        env_config=env_config,
        expected_mode="single_scene",
    )
    return resolved, env_config


def _cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {
        "profiles.env": args.env,
        "profiles.curobo": args.curobo_profile,
        "profiles.logging": args.logging_profile,
        "planner.backend": args.planner_backend,
        "execution.control_mode": args.control_mode,
        "simulation_app.gui": args.gui,
        "interactive.stdin_eof_policy": args.stdin_eof_policy,
        "execution.idle_physics_policy": args.idle_physics_policy,
        "interactive.transport.tcp_jsonl.host": args.tcp_jsonl_host,
        "interactive.transport.tcp_jsonl.port": args.tcp_jsonl_port,
        "interactive.transport.websocket.host": args.websocket_host,
        "interactive.transport.websocket.port": args.websocket_port,
        "telemetry.rate_hz": args.state_rate_hz,
        "telemetry.include_efforts": args.state_include_efforts,
        "telemetry.include_objects": args.state_include_objects,
        "telemetry.joint_effort_field": args.foxglove_joint_effort_field,
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


def create_single_scene_runtime(**kwargs: object):
    """延迟导入 SingleSceneRuntime，保证配置 dry-run 不触发 runtime 装配。"""

    from linkerbot_sim.app.runtime.single_scene_runtime import (
        create_single_scene_runtime as create,
    )

    return create(**kwargs)


def run_single_scene_interactive_motion(runtime: object, **kwargs: object) -> int:
    """延迟导入 canonical Single Scene 交互循环。"""

    from linkerbot_sim.app.interactive.single_scene.runtime import (
        run_single_scene_interactive_motion as run,
    )

    return run(runtime, **kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    """配置 stdout 行缓冲，执行 CLI，并输出最终 global step。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    resolved, env_config = resolve_entry_config(args)
    if args.dump_effective_config:
        _dump_effective_config(args.runtime_profile, resolved)
        return
    print(
        "SINGLE_SCENE_INTERACTIVE_CONFIG "
        f"runtime_profile={args.runtime_profile} fingerprint={resolved.fingerprint}",
        flush=True,
    )
    steps = run_interactive_mode(args, resolved=resolved, env_config=env_config)
    print(f"SINGLE_SCENE_INTERACTIVE_OK steps={steps}", flush=True)


__all__ = [
    "main",
    "parse_args",
    "resolve_entry_config",
    "run_interactive_mode",
]
