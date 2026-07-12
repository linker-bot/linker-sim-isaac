"""tiled interactive 主循环的 telemetry 创建、采样与发布辅助函数。"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from linkerbot_sim.telemetry.tiled.config import TiledTelemetryConfig
from linkerbot_sim.telemetry.tiled.sink import TiledInteractiveTelemetrySink
from linkerbot_sim.utils.output_paths import OutputPathPlan


def _create_telemetry(
    config: TiledTelemetryConfig | None,
    *,
    num_envs: int,
    live_host: str,
    live_port: int | None,
    mcap_path: str | Path | None,
    mcap_output_plan: OutputPathPlan | None,
    output_paths_applied: bool,
) -> TiledInteractiveTelemetrySink | None:
    """从 resolved telemetry 配置创建 sink；未配置输出时保持完全关闭。"""

    if config is None:
        return None
    if live_port is None and mcap_path is None:
        return None
    if any(env_id >= int(num_envs) for env_id in config.selected_env_ids):
        raise ValueError("telemetry env id out of range")
    sink = TiledInteractiveTelemetrySink.open(
        config=config,
        live_host=live_host,
        live_port=live_port,
        mcap_path=mcap_path,
        mcap_output_plan=mcap_output_plan,
        output_paths_applied=output_paths_applied,
    )
    if sink is not None:
        print(
            "TILED_SCENE_INTERACTIVE_TELEMETRY "
            f"env_ids={list(config.selected_env_ids)} "
            f"primary_env_id={config.primary_env_id} "
            f"decimation={config.publish_decimation} "
            f"buffer_size={config.buffer_size} "
            f"drop_policy={config.drop_policy} "
            f"live_port={live_port} "
            f"mcap_enabled={str(mcap_path is not None).lower()}",
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
        config = telemetry.config
        get_state_kwargs: dict[str, object] = {
            "env_ids": np.asarray(config.selected_env_ids, dtype=int),
            "fields": (
                None
                if getattr(config, "include_objects", True)
                else ("robots", "episode_steps", "episode_ids")
            ),
        }
        if getattr(config, "include_efforts", False):
            get_state_kwargs["include_efforts"] = True
        state_response = runtime.get_state(**get_state_kwargs)
        telemetry.publish_interactive_state(
            state_response,
            event=event,
            trigger_response=trigger_response,
        )
    except Exception as exc:
        telemetry.record_error(exc)
        print(
            f"TILED_SCENE_INTERACTIVE_TELEMETRY_FAILED {type(exc).__name__}: {exc}",
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
            f"TILED_SCENE_INTERACTIVE_TELEMETRY_FAILED {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
