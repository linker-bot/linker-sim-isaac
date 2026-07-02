#!/usr/bin/env python3
"""运行双 AR5+L6 Isaac 运动测试。

该脚本导入左右两个 AR5+L6 articulation，创建两个 JointController，然后在同一个双臂融合
cuMotion C-space 中执行客户脚本直接定义的示例动作。

用途边界：
    * 验证 scene ``robots.dual.left/right`` 能被 Isaac 执行层正确导入为两个机器人。
    * 验证每侧 ``JointController`` 能从 ``controlled_joints=["all"]`` 解析出主动 command-space，
      并自动排除 MJCF mimic/equality follower。
    * 验证 cuMotion 双臂融合模型、左右 TCP 注入、IK 和路径规划结果拆回左右 controller
      command-space 的调用链。

非目标：
    * 不验证 rope/contact 抓取。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 仓库采用 src-layout。直接运行本脚本时，Python 只会自动把 scripts/ 放进 sys.path；
# 因此这里显式把 src/ 加入搜索路径，让导入始终指向当前工作区代码。
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
# 这个脚本常被直接从仓库根目录执行，也会被 pytest 作为普通模块 import。显式补 sys.path，
# 可以避免依赖用户 shell 里是否已经设置 PYTHONPATH，同时又不在包代码里引入 Isaac。
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# 导入分层说明：
# - app.cumotion_motion_specs：客户脚本直接定义 TCP 和临时动作参数；
# - app.dual_robot_runtime：按 profile 创建完整 Isaac 双机器人 runtime；
# - app.dual_arm_cumotion_motion：封装 cuMotion context、规划和同步执行。
from linkerbot_sim.app.motion.specs import (  # noqa: E402
    CartesianTcpFrameSpec,
    CSpaceDeltaPlanMoveSpec,
    DualArmTcpSpec,
    IkOffsetMoveSpec,
    SpecifiedPathMoveSpec,
)
from linkerbot_sim.app.runtime.dual_robot import (  # noqa: E402
    create_dual_robot_runtime,
    load_dual_robot_runtime_config,
)
from linkerbot_sim.app.motion.dual_arm import (  # noqa: E402
    dual_arm_cumotion_summary,
    hold_dual_current_pose,
    run_dual_arm_cumotion_motion_result,
)
from linkerbot_sim.app.interactive.dual_arm import (  # noqa: E402
    run_interactive_dual_arm_motion,
)
from linkerbot_sim.planning.requests import (  # noqa: E402
    TaskSpacePath,
    TcpArcSegment,
    TcpLineSegment,
)


def parse_args() -> argparse.Namespace:
    """解析运行参数。

    这些参数只控制“使用哪套 profile 和 runtime 运行方式”。具体测试动作固定写在本脚本中，
    这样测试行为不会随着外部 trajectory/task 配置漂移。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    # env 是 scene profile 名称，不是直接文件路径。scene 决定机器人实例、对象和世界设置。
    parser.add_argument("--env", default="scene3")
    parser.add_argument("--cumotion-profile", default="default")
    # 目前 motion test 默认只验证 position 控制；保留 control-mode 参数是为了快速对比
    # controller runtime profile 是否也能支持其它模式。
    parser.add_argument("--control-mode", default="position")
    # GUI 只影响可视化和渲染 step；无 GUI 时更适合自动化检查。
    parser.add_argument("--gui", action="store_true")
    # hold 只有配合 --gui 才有意义。main() 会把 hold 参数同时传给 app runtime 和动作序列，
    # 前者决定 SimulationApp 是否保持可用，后者决定动作结束后是否继续 stepping。
    parser.add_argument("--hold", action="store_true", help="最终目标保持到窗口关闭")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启动交互式 JSON motion runtime",
    )
    parser.add_argument("--tcp-jsonl-host", default="127.0.0.1")
    parser.add_argument("--tcp-jsonl-port", type=int, default=None)
    parser.add_argument("--websocket-host", default="127.0.0.1")
    parser.add_argument("--websocket-port", type=int, default=None)
    # dry-run 只校验 profile，不启动 Isaac。它适合在没有 GPU/Kit 的环境里检查配置引用。
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """脚本入口：解析配置、启动 Isaac、导入双机器人并播放同步测试动作。"""

    if hasattr(sys.stdout, "reconfigure"):
        # Isaac/Kit 启动日志很多；行缓冲能让测试输出及时出现在 CI 或终端里。
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    tcp = DualArmTcpSpec(
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

    if args.dry_run:
        load_dual_robot_runtime_config(env=args.env)
        dual_arm_cumotion_summary(
            env=args.env,
            cumotion_profile=args.cumotion_profile,
            tcp=tcp,
        )
        print("DUAL_ARM_MOTION_DRY_RUN_OK", flush=True)
        return

    # create_dual_robot_runtime 负责 Isaac 生命周期、USD stage 导入、world.reset()、
    # JointController 创建和 DualRobotRuntime 组装。动作脚本只消费返回的 execution runtime，
    # 避免把底层导入细节散落在测试动作流程里。
    runtime = create_dual_robot_runtime(
        env=args.env,
        control_mode=args.control_mode,
        gui=args.gui,
        hold_app=args.hold,
        status_prefix="DUAL_ARM_MOTION",
    )
    completed = False
    try:
        if args.interactive:
            steps = run_interactive_dual_arm_motion(
                runtime,
                tcp=tcp,
                cumotion_profile=args.cumotion_profile,
                stdin_enabled=True,
                tcp_jsonl_host=args.tcp_jsonl_host,
                tcp_jsonl_port=args.tcp_jsonl_port,
                websocket_host=args.websocket_host,
                websocket_port=args.websocket_port,
            )
        else:
            result = run_dual_arm_cumotion_motion_result(
                runtime,
                tcp=tcp,
                moves=(
                    IkOffsetMoveSpec(
                        side="left",
                        tcp_frame_name="left_demo_tcp",
                        tcp_offset=(0.2, -0.4, -0.1),
                        duration_s=1.0,
                        phase="left_ik_lift",
                    ),
                    SpecifiedPathMoveSpec(
                        side="left",
                        tcp_frame_name="left_demo_tcp",
                        path=TaskSpacePath(
                            segments=(
                                TcpLineSegment(
                                    target_offset=(0.0, 0.0, 0.1),
                                    orientation_mode="none",
                                ),
                            ),
                        ),
                        duration_s=1.2,
                        phase="left_tcp_line",
                    ),
                    SpecifiedPathMoveSpec(
                        side="right",
                        tcp_frame_name="right_demo_tcp",
                        path=TaskSpacePath(
                            segments=(
                                TcpArcSegment(
                                    target_offset=(0.2, 0.2, 0.1),
                                    intermediate_offset=(0.0, 0.03, 0.02),
                                    arc_mode="three_point",
                                    constant_orientation=True,
                                ),
                            ),
                        ),
                        duration_s=1.6,
                        phase="right_tcp_arc",
                    ),
                    CSpaceDeltaPlanMoveSpec(
                        side="right",
                        tcp_frame_name="right_demo_tcp",
                        joint_deltas=(
                            0.18,
                            -0.14,
                            0.12,
                            -0.1,
                            0.08,
                            -0.06,
                            0.04,
                        ),
                        duration_s=1.6,
                        phase="right_cspace_plan",
                    ),
                ),
                cumotion_profile=args.cumotion_profile,
            )
            steps = result.step
            if not result.success:
                print(
                    "DUAL_ARM_MOTION_TEST_PLAN_FAILED "
                    f"steps={result.step} move={result.failed_move_index} "
                    f"side={result.side} tcp={result.tcp_frame_name} "
                    f"phase={result.phase} status={result.status} "
                    f"message={result.message}",
                    flush=True,
                )
                if args.gui:
                    steps = hold_dual_current_pose(
                        runtime.execution,
                        step=steps,
                        simulation_app=runtime.session.app,
                    )
                completed = True
                return
        if args.hold and args.gui:
            steps = hold_dual_current_pose(
                runtime.execution,
                step=steps,
                simulation_app=runtime.session.app,
            )
        print(
            f"DUAL_ARM_MOTION_TEST_OK steps={steps} cumotion=parameterized_moves",
            flush=True,
        )
        completed = True
    finally:
        # Isaac/Kit shutdown can request process exit in headless mode. Close only after
        # successful completion so a planning/IK failure is not masked as exit code 0.
        if completed:
            runtime.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"DUAL_ARM_MOTION_TEST_FAILED {type(exc).__name__}: {exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
