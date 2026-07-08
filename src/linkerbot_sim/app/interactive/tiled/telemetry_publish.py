"""Telemetry helpers for tiled interactive loop."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping

import numpy as np

from linkerbot_sim.telemetry.tiled import (
    TiledInteractiveTelemetrySink,
    TiledTelemetryConfig,
)


def _create_telemetry(
    args: argparse.Namespace,
    *,
    num_envs: int,
) -> TiledInteractiveTelemetrySink | None:
    """按 CLI 创建 tiled telemetry sink；未配置输出时保持完全关闭。"""

    if args.foxglove_live_port is None and args.foxglove_mcap_path is None:
        return None
    config = TiledTelemetryConfig.from_env_ids(
        args.telemetry_env_ids,
        publish_decimation=args.telemetry_decimation,
        topic_prefix=args.telemetry_topic_prefix,
        include_full_batch_json=args.telemetry_full_batch_json,
        include_standard_joint_states=args.telemetry_joint_states,
    )
    if any(env_id >= int(num_envs) for env_id in config.selected_env_ids):
        raise ValueError("telemetry env id out of range")
    sink = TiledInteractiveTelemetrySink.open(
        config=config,
        live_host=args.foxglove_live_host,
        live_port=args.foxglove_live_port,
        mcap_path=args.foxglove_mcap_path,
    )
    if sink is not None:
        print(
            "TILED_INTERACTIVE_TELEMETRY "
            f"env_ids={list(config.selected_env_ids)} "
            f"decimation={config.publish_decimation} "
            f"topic_prefix={config.topic_prefix} "
            f"live_port={args.foxglove_live_port} "
            f"mcap_path={args.foxglove_mcap_path}",
            flush=True,
        )
    return sink


def _runtime_num_envs(runtime: object) -> int:
    """从 debug 或 Isaac runtime 中读取 env 数量。"""

    config = getattr(runtime, "config", None)
    if config is not None and hasattr(config, "num_envs"):
        return int(config.num_envs)
    scene = getattr(runtime, "scene", None)
    scene_config = getattr(scene, "config", None)
    if scene_config is not None and hasattr(scene_config, "num_envs"):
        return int(scene_config.num_envs)
    raise ValueError("runtime does not expose num_envs")


def _publish_state_telemetry(
    telemetry: TiledInteractiveTelemetrySink | None,
    runtime: object,
    *,
    event: str,
    trigger_response: Mapping[str, object] | None = None,
) -> None:
    """从 runtime 主线程采样 selected env state 并发布到 Foxglove/MCAP。"""

    if telemetry is None:
        return
    try:
        state_response = runtime.get_state(
            env_ids=np.asarray(telemetry.config.selected_env_ids, dtype=int),
            fields=None,
        )
        telemetry.publish_interactive_state(
            state_response,
            event=event,
            trigger_response=trigger_response,
        )
    except Exception as exc:
        print(
            f"TILED_INTERACTIVE_TELEMETRY_FAILED {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _publish_response_telemetry(
    telemetry: TiledInteractiveTelemetrySink | None,
    runtime: object,
    response: Mapping[str, object],
) -> None:
    """把本次交互响应对应的最新 selected state 发布到 telemetry。"""

    if telemetry is None:
        return
    event = str(response.get("event", ""))
    if event in {"rejected", "quit"}:
        return
    try:
        _publish_state_telemetry(
            telemetry,
            runtime,
            event=event,
            trigger_response=response,
        )
    except Exception as exc:
        print(
            f"TILED_INTERACTIVE_TELEMETRY_FAILED {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
