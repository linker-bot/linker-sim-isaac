"""交互式 TiledSceneRuntime 的 env step-control 入口。

这个包面向 Isaac Lab 风格 tiled envs，而不是单臂/双臂 motion runtime。热路径保持
tiled command step 的同步语义；轨迹和异步规划放在 runtime 外围，通过状态快照和
trajectory buffer 接入。
"""

from linkerbot_sim.app.interactive.tiled_scene.cli import main, parse_args
from linkerbot_sim.app.interactive.tiled_scene.action_messages import (
    parse_tiled_action,
)
from linkerbot_sim.app.interactive.tiled_scene.protocol import (
    handle_tiled_interactive_message,
)
from linkerbot_sim.app.interactive.tiled_scene.transport import (
    BoundedInteractiveRequestQueue,
    run_interactive_loop,
    start_stdin_jsonl_reader,
    start_tcp_jsonl_server,
    start_websocket_server,
    stop_tcp_jsonl_server,
)

__all__ = [
    "TiledSceneRuntime",
    "BoundedInteractiveRequestQueue",
    "handle_tiled_interactive_message",
    "main",
    "parse_args",
    "parse_tiled_action",
    "run_interactive_loop",
    "start_stdin_jsonl_reader",
    "start_tcp_jsonl_server",
    "start_websocket_server",
    "stop_tcp_jsonl_server",
]


def __getattr__(name: str):
    if name != "TiledSceneRuntime":
        raise AttributeError(name)
    from linkerbot_sim.app.interactive.tiled_scene.runtime import (
        TiledSceneRuntime,
    )

    return TiledSceneRuntime
