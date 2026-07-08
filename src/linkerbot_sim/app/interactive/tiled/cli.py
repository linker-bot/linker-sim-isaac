"""CLI entrypoint for tiled interactive runtime."""

from __future__ import annotations

import argparse
import queue
import socketserver
import sys

from linkerbot_sim.app.interactive.tiled.isaac_runtime import IsaacTiledInteractiveRuntime
from linkerbot_sim.app.interactive.tiled.telemetry_publish import _create_telemetry, _runtime_num_envs
from linkerbot_sim.app.interactive.tiled.transport import (
    _InteractiveControl,
    _InteractiveRequest,
    _quit_on_stdin_eof,
    run_interactive_loop,
    start_stdin_jsonl_reader,
    start_tcp_jsonl_server,
)
from linkerbot_sim.configs.profiles import load_profile_yaml


def parse_args() -> argparse.Namespace:
    """解析 tiled 交互脚本参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="scene3_tiled", help="env profile 名称")
    parser.add_argument("--gui", action="store_true", help="打开 Isaac GUI")
    parser.add_argument(
        "--default-decimation",
        type=int,
        default=2,
        help="action 未指定 decimation 时展开的 physics tick 数",
    )
    parser.add_argument(
        "--planner-workers",
        type=int,
        default=2,
        help="tiled async planner worker 数；planner 不访问 Isaac runtime，只消费状态快照",
    )
    parser.add_argument(
        "--planner-backend",
        choices=("linear", "cumotion"),
        default="linear",
        help="async planner backend；cumotion 支持 task-space/specified-path 段",
    )
    parser.add_argument(
        "--max-pending-requests",
        type=int,
        default=64,
        help="最多允许同时排队/运行的 planner 请求数",
    )
    parser.add_argument(
        "--max-completed-results",
        type=int,
        default=256,
        help="planner completed result 缓存上限；设为 0 表示不保留 completed 摘要",
    )
    parser.add_argument(
        "--stdin",
        dest="stdin_enabled",
        action="store_true",
        default=True,
        help="从 stdin 读取 JSONL 命令",
    )
    parser.add_argument(
        "--no-stdin",
        dest="stdin_enabled",
        action="store_false",
        help="关闭 stdin JSONL，只使用 TCP/telemetry 保持进程",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="stdin EOF 后仍保持交互进程，适合 IDE/后台启动或只看 Foxglove",
    )
    parser.add_argument("--tcp-jsonl-host", default="127.0.0.1")
    parser.add_argument("--tcp-jsonl-port", type=int, default=None)
    parser.add_argument("--foxglove-live-host", default="127.0.0.1")
    parser.add_argument(
        "--foxglove-live-port",
        type=int,
        default=None,
        help="Foxglove live server port；tiled 日常调试建议 8767；不传则不开 telemetry live",
    )
    parser.add_argument("--foxglove-mcap-path", default=None)
    parser.add_argument(
        "--telemetry-env-ids",
        default="0",
        help="逗号分隔的 selected env ids，用于 tiled Foxglove/MCAP 输出",
    )
    parser.add_argument(
        "--telemetry-decimation",
        type=int,
        default=1,
        help="每隔多少 global step 发布一次 tiled telemetry；reset/set_state 总会发布",
    )
    parser.add_argument(
        "--telemetry-rate-hz",
        type=float,
        default=10.0,
        help="开启 Foxglove/MCAP 时的周期状态发布频率；设为 0 关闭周期发布",
    )
    parser.add_argument(
        "--telemetry-topic-prefix",
        default="/tiled",
        help="tiled telemetry topic 前缀",
    )
    parser.add_argument(
        "--telemetry-full-batch-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否发布 /tiled/state JSON payload",
    )
    parser.add_argument(
        "--telemetry-joint-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否为第一个 selected env 发布标准 JointStates",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    env_config = load_profile_yaml("env", args.env)
    runtime = IsaacTiledInteractiveRuntime.create(
        env_name=args.env,
        env_config=env_config,
        gui=bool(args.gui),
        default_decimation=args.default_decimation,
        planner_backend=args.planner_backend,
        planner_workers=args.planner_workers,
        max_pending_requests=args.max_pending_requests,
        max_completed_results=args.max_completed_results,
    )
    telemetry = _create_telemetry(args, num_envs=_runtime_num_envs(runtime))
    request_queue: queue.Queue[_InteractiveRequest | _InteractiveControl] = queue.Queue()
    server: socketserver.ThreadingTCPServer | None = None
    try:
        if args.tcp_jsonl_port is not None:
            server = start_tcp_jsonl_server(
                request_queue,
                quit_event=runtime.quit_event,
                host=args.tcp_jsonl_host,
                port=args.tcp_jsonl_port,
            )
            print(
                "TILED_INTERACTIVE_TCP_JSONL "
                f"host={args.tcp_jsonl_host} port={args.tcp_jsonl_port}",
                flush=True,
            )
        print("TILED_INTERACTIVE_READY", flush=True)
        if args.stdin_enabled:
            start_stdin_jsonl_reader(
                request_queue,
                quit_on_eof=_quit_on_stdin_eof(
                    hold=bool(args.hold),
                    tcp_jsonl_port=args.tcp_jsonl_port,
                    telemetry=telemetry,
                ),
            )
        run_interactive_loop(
            runtime,
            telemetry=telemetry,
            request_queue=request_queue,
            telemetry_rate_hz=float(args.telemetry_rate_hz),
        )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
        if telemetry is not None:
            telemetry.close()
        print("TILED_INTERACTIVE_EXIT", flush=True)
