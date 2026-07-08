"""CLI entrypoint for the single-arm interactive runtime."""

from __future__ import annotations

import argparse
import sys

from linkerbot_sim.app.interactive.single_arm.runtime import (
    run_interactive_single_arm_motion,
)
from linkerbot_sim.app.interactive.state_stream import InteractiveStateStreamConfig
from linkerbot_sim.app.runtime.single_robot import create_single_robot_runtime


def parse_args() -> argparse.Namespace:
    """解析单臂实时交互 runtime 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="scene1")
    parser.add_argument("--cumotion-profile", default="default")
    parser.add_argument("--logging-profile", default="default_logger")
    parser.add_argument("--control-mode", default="position")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--tcp-jsonl-host", default="127.0.0.1")
    parser.add_argument("--tcp-jsonl-port", type=int, default=None)
    parser.add_argument("--websocket-host", default="127.0.0.1")
    parser.add_argument("--websocket-port", type=int, default=None)
    parser.add_argument(
        "--state-rate-hz",
        type=float,
        default=60.0,
        help="Foxglove 状态采样频率；<=0 时关闭状态采样",
    )
    parser.add_argument(
        "--state-include-efforts",
        action="store_true",
        help="读取并发布 commanded/measured/applied effort",
    )
    parser.add_argument(
        "--state-include-objects",
        action="store_true",
        help="读取并发布 env runtime object 位姿",
    )
    parser.add_argument("--foxglove-live-host", default="127.0.0.1")
    parser.add_argument(
        "--foxglove-live-port",
        type=int,
        default=None,
        help="Foxglove live server port；单臂日常调试建议 8765；不传则不开状态流",
    )
    parser.add_argument("--foxglove-mcap-path", default=None)
    parser.add_argument(
        "--foxglove-joint-effort-field",
        choices=("none", "commanded", "measured", "applied"),
        default="none",
        help="写入 Foxglove /joint_states effort 字段的 effort 语义",
    )
    return parser.parse_args()


def run_interactive_mode(args: argparse.Namespace) -> int:
    """创建单臂 runtime，并启动实时交互式 JSON motion 循环。"""

    runtime = create_single_robot_runtime(
        env=args.env,
        cumotion_profile=args.cumotion_profile,
        logging_profile=args.logging_profile,
        control_mode=args.control_mode,
        gui=args.gui,
        status_prefix="SINGLE_ARM_INTERACTIVE",
    )
    completed = False
    try:
        steps = run_interactive_single_arm_motion(
            runtime,
            stdin_enabled=True,
            tcp_jsonl_host=args.tcp_jsonl_host,
            tcp_jsonl_port=args.tcp_jsonl_port,
            websocket_host=args.websocket_host,
            websocket_port=args.websocket_port,
            state_stream_config=InteractiveStateStreamConfig(
                rate_hz=args.state_rate_hz,
                include_efforts=args.state_include_efforts,
                include_objects=args.state_include_objects,
                foxglove_live_host=args.foxglove_live_host,
                foxglove_live_port=args.foxglove_live_port,
                foxglove_mcap_path=args.foxglove_mcap_path,
                foxglove_joint_effort_field=args.foxglove_joint_effort_field,
            ),
        )
        print(f"SINGLE_ARM_INTERACTIVE_OK steps={steps}", flush=True)
        completed = True
        return steps
    finally:
        if completed:
            runtime.close()


def main() -> None:
    """脚本入口。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    run_interactive_mode(parse_args())
