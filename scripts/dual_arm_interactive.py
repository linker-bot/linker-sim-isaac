#!/usr/bin/env python3
"""运行双 AR5+L6 实时交互式 JSON motion runtime。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 仓库采用 src-layout。直接运行本脚本时，Python 只会自动把 scripts/ 放进 sys.path；
# 因此这里显式把 src/ 加入搜索路径，让导入始终指向当前工作区代码。
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.app.interactive.dual_arm import (  # noqa: E402
    run_interactive_dual_arm_motion,
)
from linkerbot_sim.app.motion.specs import (  # noqa: E402
    CartesianTcpFrameSpec,
    DualArmTcpSpec,
)
from linkerbot_sim.app.runtime.dual_robot import create_dual_robot_runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析实时交互 runtime 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="scene3")
    parser.add_argument("--cumotion-profile", default="default")
    parser.add_argument("--control-mode", default="position")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="无命令时保持当前姿态")
    parser.add_argument("--tcp-jsonl-host", default="127.0.0.1")
    parser.add_argument("--tcp-jsonl-port", type=int, default=None)
    parser.add_argument("--websocket-host", default="127.0.0.1")
    parser.add_argument("--websocket-port", type=int, default=None)
    return parser.parse_args()


def default_dual_arm_tcp() -> DualArmTcpSpec:
    """返回交互 runtime 默认测试 TCP。"""

    return DualArmTcpSpec(
        left=CartesianTcpFrameSpec(
            frame_name="left_demo_tcp",
            xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 0.0),
        ),
        right=CartesianTcpFrameSpec(
            frame_name="right_demo_tcp",
            xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 0.0),
        ),
    )


def run_interactive_mode(args: argparse.Namespace) -> int:
    """创建双臂 runtime，并启动实时交互式 JSON motion 循环。"""

    runtime = create_dual_robot_runtime(
        env=args.env,
        control_mode=args.control_mode,
        gui=args.gui,
        hold_app=args.hold,
        status_prefix="DUAL_ARM_INTERACTIVE",
    )
    completed = False
    try:
        steps = run_interactive_dual_arm_motion(
            runtime,
            tcp=default_dual_arm_tcp(),
            cumotion_profile=args.cumotion_profile,
            stdin_enabled=True,
            tcp_jsonl_host=args.tcp_jsonl_host,
            tcp_jsonl_port=args.tcp_jsonl_port,
            websocket_host=args.websocket_host,
            websocket_port=args.websocket_port,
        )
        print(f"DUAL_ARM_INTERACTIVE_OK steps={steps}", flush=True)
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


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"DUAL_ARM_INTERACTIVE_FAILED {type(exc).__name__}: {exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
