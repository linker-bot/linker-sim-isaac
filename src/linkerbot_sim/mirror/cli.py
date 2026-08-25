"""Mirror 命令行入口；profile 只选择公开的物理引擎组合。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.mirror.app import run_mirror
from linkerbot_sim.mirror.bootstrap import create_mirror_runtime
from linkerbot_sim.mirror.interface.transport import (
    StdinJsonlTransport,
    TcpJsonlTransport,
    WebSocketTransport,
    make_json_handler,
)


def _endpoint(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("endpoint must be written as HOST:PORT")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("PORT must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("PORT must be within [1, 65535]")
    return host, port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LinkerBot Mirror reality simulation")
    parser.add_argument(
        "--profile",
        choices=(
            "physx_cpu",
            "physx_cpu_hybrid",
            "newton_cpu",
            "newton_cuda",
        ),
        default="physx_cpu",
    )
    parser.add_argument(
        "--stdin",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the stdin JSONL switch in the profile",
    )
    parser.add_argument("--tcp-jsonl", type=_endpoint, metavar="HOST:PORT")
    parser.add_argument("--websocket", type=_endpoint, metavar="HOST:PORT")
    parser.add_argument("--response-timeout-s", type=float)
    parser.add_argument("--poll-timeout-s", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_mirror_config(args.profile)
    interface = config.control.interface
    response_timeout_s = (
        interface.response_timeout_s
        if args.response_timeout_s is None
        else float(args.response_timeout_s)
    )
    poll_timeout_s = (
        interface.queue_poll_timeout_s
        if args.poll_timeout_s is None
        else float(args.poll_timeout_s)
    )
    if response_timeout_s <= 0.0 or poll_timeout_s <= 0.0:
        raise ValueError("timeout must be > 0")
    stdin_enabled = interface.stdin_enabled if args.stdin is None else bool(args.stdin)
    runtime = create_mirror_runtime(config)
    try:
        handler = make_json_handler(
            runtime.controller.submit_and_wait,
            timeout_s=response_timeout_s,
        )
        endpoints: list[object] = []
        if stdin_enabled:
            endpoints.append(
                StdinJsonlTransport(
                    handler,
                    eof_requests_quit=(
                        runtime.controller.request_quit
                        if interface.stdin_eof_policy == "exit"
                        else None
                    ),
                    max_message_bytes=interface.max_message_bytes,
                    poll_interval_s=interface.queue_poll_timeout_s,
                    shutdown_timeout_s=interface.shutdown_timeout_s,
                )
            )
        if args.tcp_jsonl is not None:
            endpoints.append(
                TcpJsonlTransport(
                    handler,
                    host=args.tcp_jsonl[0],
                    port=args.tcp_jsonl[1],
                    max_message_bytes=interface.max_message_bytes,
                    max_connections=interface.max_connections,
                    startup_timeout_s=interface.startup_timeout_s,
                    shutdown_timeout_s=interface.shutdown_timeout_s,
                )
            )
        if args.websocket is not None:
            endpoints.append(
                WebSocketTransport(
                    handler,
                    host=args.websocket[0],
                    port=args.websocket[1],
                    max_message_bytes=interface.max_message_bytes,
                    max_connections=interface.max_connections,
                    startup_timeout_s=interface.startup_timeout_s,
                    shutdown_timeout_s=interface.shutdown_timeout_s,
                )
            )
        result = run_mirror(
            runtime,
            endpoints=endpoints,
            poll_timeout_s=poll_timeout_s,
            on_ready=lambda: print("MIRROR_INTERACTIVE_READY", flush=True),
            # GUI Mirror 使用 fast shutdown，native close 可能不把控制流交还给 Python；
            # 因此在全部产品资源停止、session 仍存活的边界输出退出 marker。
            before_session_close=lambda _report: print(
                "MIRROR_INTERACTIVE_EXIT", flush=True
            ),
        )
    except BaseException:
        # endpoint 构造失败发生在 run_mirror 的 finally 之前；这里仍由 owner thread 尝试
        # 完整关闭。close 可重入，因此 run_mirror 已完成关闭时本分支也是幂等的。
        if not runtime.is_closed:
            report = runtime.close()
            if not report.stopped:
                print(
                    "MIRROR_STARTUP_ROLLBACK_INCOMPLETE "
                    f"live_resources={list(report.live_resources)}",
                    flush=True,
                )
        raise
    if result.close_report is not None and not result.close_report.stopped:
        raise RuntimeError(
            "Mirror shutdown did not complete: "
            f"{result.close_report.live_resources} {result.close_report.errors}"
        )
    return 0


__all__ = ["build_parser", "main"]
