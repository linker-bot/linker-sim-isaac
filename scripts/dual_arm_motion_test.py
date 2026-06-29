#!/usr/bin/env python3
"""只运行双 AR5+L6 Isaac 运动测试。

该脚本从 ``pinch_grasp.py`` 中拆出双臂执行验证逻辑。它不加载 rope，也不做 cuMotion 规划；
只导入左右两个 AR5+L6 articulation，创建两个 JointController，并按类似单臂 pinch grasp
的阶段同步下发目标：

1. 预成型左右手；
2. 左右臂做小幅多关节 reach；
3. 左右手闭合；
4. 左右臂和手返回初始目标。

用途边界：
    * 验证 ``configs/robots/ar5v2_l6v1_dual.yaml`` 能被 Isaac 执行层正确导入为两个机器人。
    * 验证每侧 ``JointController`` 能从 ``controlled_joints=["all"]`` 解析出主动 command-space，
      并自动排除 MJCF mimic/equality follower。
    * 验证 ``DualCommandPositionTargetStep`` 会在同一个 physics step 前分别下发左右目标，
      而不是先推进左臂、再推进右臂。

非目标：
    * 不验证 cuMotion IK、trajectory optimization 或 TCP frame 注入。
    * 不验证 rope/contact 抓取，只复用 pinch_grasp 中的默认手型常量来构造可见的手部动作。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
# 这个脚本常被直接从仓库根目录执行，也会被 pytest 作为普通模块 import。显式补 sys.path，
# 可以避免依赖用户 shell 里是否已经设置 PYTHONPATH，同时又不在包代码里引入 Isaac。
for path in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from linkerbot_sim.app.runtime_settings import EnvRuntimeSettings  # noqa: E402
from linkerbot_sim.app.simulation_session import create_simulation_session  # noqa: E402
from linkerbot_sim.assets.robot_loader import DualRobotExecutionConfig  # noqa: E402
from linkerbot_sim.controllers.config import (  # noqa: E402
    load_controller_profiles,
)
from linkerbot_sim.envs.scene_objects import (  # noqa: E402
    add_scene_objects,
    scene_objects_from_env_config,
)
from linkerbot_sim.execution.dual_runtime import (  # noqa: E402
    DualRobotRuntime,
    RobotSideRuntime,
)
from linkerbot_sim.execution.dual_steps import (  # noqa: E402
    DualCommandPositionTargetStep,
)
from linkerbot_sim.execution.setup import (  # noqa: E402
    finalize_robot_controller,
    import_execution_robot_to_stage,
)
from linkerbot_sim.robots.joint_groups import target_vector_from_mapping  # noqa: E402
from linkerbot_sim.utils.config import load_yaml  # noqa: E402
from pinch_grasp import (  # noqa: E402
    default_closed_pinch_hand_targets,
    default_pre_pinch_hand_targets,
)


DEFAULT_DUAL_ROBOT_CONFIG = Path("configs/robots/ar5v2_l6v1_dual.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=DEFAULT_DUAL_ROBOT_CONFIG,
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=Path("configs/controllers"),
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/envs/rope_scene.yaml"),
    )
    parser.add_argument("--control-mode", default="position")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="最终目标保持到窗口关闭")
    parser.add_argument("--short-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--physics-frequency", type=float, default=None)
    parser.add_argument("--render-frequency", type=float, default=None)
    parser.add_argument("--gravity-z", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """脚本入口：解析配置、启动 Isaac、导入双机器人并播放同步测试动作。"""

    if hasattr(sys.stdout, "reconfigure"):
        # Isaac/Kit 启动日志很多；行缓冲能让 smoke test 输出及时出现在 CI 或终端里。
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    # robot YAML 在这里仅负责 Isaac 导入和安装位姿。双臂动作语义、cuMotion 算法参数都不在
    # 本脚本使用范围内；这个测试的重点是两个真实 articulation 的执行骨架。
    robot_config = load_yaml(args.robot_config)
    dual_config = DualRobotExecutionConfig.from_mapping(robot_config)
    env_config = load_yaml(args.env_config)
    runtime_settings = EnvRuntimeSettings.from_env_config(
        env_config,
        physics_frequency_override=args.physics_frequency,
        render_frequency_override=args.render_frequency,
        gravity_z_override=args.gravity_z,
    )
    scene_objects = scene_objects_from_env_config(env_config)
    _print_config_summary(dual_config, dry_run=args.dry_run)
    _print_scene_object_config_summary(scene_objects, prefix="DUAL_ARM_MOTION")
    if args.dry_run:
        return

    controller_profiles = load_controller_profiles(args.controller_config)
    session = create_simulation_session(gui=args.gui, settings=runtime_settings)
    try:
        world = session.world
        stage = session.stage
        _print_scene_objects(
            add_scene_objects(stage, scene_objects), prefix="DUAL_ARM_MOTION"
        )

        # 左右两侧是两个独立 articulation，各自有自己的 articulation controller 和 command-space。
        # 这里先只导入资产和写 USD/PhysX 覆盖；实际 JointController 在 world.reset() 后创建。
        imported = {
            "left": import_execution_robot_to_stage(
                world=world,
                stage=stage,
                single_articulation_type=session.single_articulation_type,
                robot_execution=dual_config.left,
                controller_profiles=controller_profiles,
                env_config=env_config,
            ),
            "right": import_execution_robot_to_stage(
                world=world,
                stage=stage,
                single_articulation_type=session.single_articulation_type,
                robot_execution=dual_config.right,
                controller_profiles=controller_profiles,
                env_config=env_config,
            ),
        }

        # reset 会让 Isaac 根据当前 stage 建立 articulation view。之后读取 num_dof、dof_names、
        # 创建 JointController 和配置 runtime gain 才可靠。
        world.reset()
        world.get_physics_context().set_gravity(runtime_settings.gravity_z)

        runtimes: dict[str, RobotSideRuntime] = {}
        for side, imported_side in imported.items():
            prepared = finalize_robot_controller(
                imported=imported_side,
                controller_profiles=controller_profiles,
                control_mode=args.control_mode,
            )
            controller = prepared.joint_controller
            # RobotSideRuntime 是 execution 层的最小单侧句柄：articulation + controller。
            # DualRobotRuntime 再把左右句柄放到同一个 world step 下同步执行。
            runtimes[side] = RobotSideRuntime(
                side=side,
                articulation=prepared.articulation,
                joint_controller=controller,
            )
            print(
                "DUAL_ARM_MOTION_IMPORTED "
                f"side={side} asset={prepared.asset_path} "
                f"prim_path={prepared.articulation.prim_path} "
                f"num_dof={prepared.articulation.num_dof} "
                f"command_joints={list(controller.command_joint_names)}",
                flush=True,
            )

        runtime = DualRobotRuntime(
            left=runtimes["left"],
            right=runtimes["right"],
            simulation_world=world,
            articulation_action_type=session.articulation_action_type,
            simulation_app=session.app if args.hold else None,
            render_enabled=args.gui,
        )
        steps = run_dual_arm_motion_sequence(
            runtime,
            short_smoke=args.short_smoke,
            hold=args.hold and args.gui,
            simulation_app=session.app,
        )
        print(f"DUAL_ARM_MOTION_TEST_OK steps={steps}", flush=True)
    finally:
        session.app.close()


def run_dual_arm_motion_sequence(
    runtime: DualRobotRuntime,
    *,
    short_smoke: bool,
    hold: bool,
    simulation_app,
) -> int:
    """构造并播放一个不依赖 cuMotion 的双臂同步动作序列。

    输入的 ``runtime`` 已包含左右 articulation 和 controller。函数只在 controller
    command-space 中构造目标：机械臂做小幅 scripted reach，手部用 pinch_grasp 的默认
    pre/closed 手型。每个阶段都交给 ``DualCommandPositionTargetStep``，确保左右目标在同一
    个 world step 前下发。
    """

    # 当前 command-space 只包含主动命令关节，不包含 mimic follower。后续所有目标都在这个
    # 空间中构造；follower 展开留给 JointController。
    left_start = current_command(runtime.left)
    right_start = current_command(runtime.right)

    # 阶段目标先全部预计算出来，便于每个 phase 只描述 from/to 和 duration。这样也避免在播放
    # 过程中读取不断变化的实际状态，测试的是确定性的 command interpolation。
    left_pre = hand_target_command(
        left_start,
        runtime.left.joint_controller.command_joint_names,
        "left",
        closed=False,
    )
    right_pre = hand_target_command(
        right_start,
        runtime.right.joint_controller.command_joint_names,
        "right",
        closed=False,
    )
    left_reach = arm_reach_target(
        left_pre, runtime.left.joint_controller.command_joint_names, "left"
    )
    right_reach = arm_reach_target(
        right_pre, runtime.right.joint_controller.command_joint_names, "right"
    )
    left_closed = hand_target_command(
        left_reach,
        runtime.left.joint_controller.command_joint_names,
        "left",
        closed=True,
    )
    right_closed = hand_target_command(
        right_reach,
        runtime.right.joint_controller.command_joint_names,
        "right",
        closed=True,
    )

    durations = _phase_durations(short_smoke)
    step = 0
    # 每个元组都是一个同步阶段。DualCommandPositionTargetStep.run(...) 会在内部根据 physics dt
    # 插值左右 command，并在每一帧先 apply 左右目标，再调用一次 world.step()。
    for phase, left_from, right_from, left_to, right_to, duration in (
        ("pre_shape", left_start, right_start, left_pre, right_pre, durations["hand"]),
        ("arm_reach", left_pre, right_pre, left_reach, right_reach, durations["arm"]),
        (
            "close_hands",
            left_reach,
            right_reach,
            left_closed,
            right_closed,
            durations["hand"],
        ),
        (
            "return_home",
            left_closed,
            right_closed,
            left_start,
            right_start,
            durations["return"],
        ),
    ):
        step = DualCommandPositionTargetStep(
            left_start_command=left_from,
            right_start_command=right_from,
            left_target_command=left_to,
            right_target_command=right_to,
            duration=duration,
            phase=f"dual_{phase}",
        ).run(runtime, step)

    if hold:
        # GUI 调试时可保持窗口打开。这里继续通过 DualCommandPositionTargetStep 执行 hold，
        # 而不是手写 world.step，保持和正常阶段完全相同的左右同步路径。
        while simulation_app.is_running():
            step = DualCommandPositionTargetStep(
                left_start_command=left_start,
                right_start_command=right_start,
                left_target_command=left_start,
                right_target_command=right_start,
                duration=durations["hand"],
                phase="dual_hold",
            ).run(runtime, step)
    return step


def arm_reach_target(
    start_command: np.ndarray, command_joint_names: tuple[str, ...], side: str
) -> np.ndarray:
    """构造类似单臂 scripted reach 的小幅多关节机械臂目标。

    这里故意不调用 IK：motion test 只验证执行骨架和 controller 行为。左右使用相反符号，让
    GUI 下能直观看到两侧都在动，同时避免假设两侧关节轴方向完全一致。
    """

    target = np.asarray(start_command, dtype=float).reshape(-1).copy()
    arm_indices = [
        index for index, name in enumerate(command_joint_names) if "_arm_joint_" in name
    ]
    sign = 1.0 if side == "left" else -1.0
    deltas = (0.08, -0.06, 0.05, -0.04, 0.035, -0.025, 0.02)
    for offset, index in zip(deltas, arm_indices):
        target[int(index)] += sign * float(offset)
    return target


def hand_target_command(
    base_command: np.ndarray,
    command_joint_names: tuple[str, ...],
    side: str,
    *,
    closed: bool,
) -> np.ndarray:
    """把默认 pinch 手型写入当前 command-space 目标。

    ``target_vector_from_mapping`` 只覆盖目标字典中出现的 master joints，其它 command 关节沿用
    ``base_command``。DIP 等 follower 不在 command-space 中，由 JointController 运行时跟随。
    """

    targets = (
        default_closed_pinch_hand_targets(side)
        if closed
        else default_pre_pinch_hand_targets(side)
    )
    return target_vector_from_mapping(
        command_joint_names,
        targets,
        base=np.asarray(base_command, dtype=float),
    )


def current_command(side_runtime: RobotSideRuntime) -> np.ndarray:
    """读取 articulation 当前关节位置，并投影到 controller command-space。"""

    positions = np.asarray(
        side_runtime.articulation.get_joint_positions(), dtype=float
    ).reshape(-1)
    return positions[np.asarray(side_runtime.joint_controller.command_indices, dtype=int)]


def _phase_durations(short_smoke: bool) -> dict[str, float]:
    """返回各阶段持续时间；short smoke 用于 CI/快速 sanity check。"""

    if short_smoke:
        return {"hand": 0.04, "arm": 0.06, "return": 0.04}
    return {"hand": 0.25, "arm": 0.6, "return": 0.45}


def _print_config_summary(
    dual_config: DualRobotExecutionConfig, *, dry_run: bool
) -> None:
    """打印配置摘要，便于 dry-run/CI 确认解析到的是双机器人结构。"""

    print(
        "DUAL_ARM_MOTION_CONFIG "
        f"dry_run={dry_run} "
        f"left_asset={dual_config.left.robot.asset_path} "
        f"right_asset={dual_config.right.robot.asset_path} "
        f"left_joint_selector={list(dual_config.left.controlled_joints)} "
        f"right_joint_selector={list(dual_config.right.controlled_joints)}",
        flush=True,
    )


def _print_scene_objects(scene_objects, *, prefix: str) -> None:
    for scene_object in scene_objects:
        print(
            f"{prefix}_SCENE_OBJECT "
            f"name={scene_object.name} type={scene_object.asset_type} "
            f"asset={scene_object.asset_path} prim_path={scene_object.prim_path} "
            f"imported_path={scene_object.imported_path} "
            f"static={scene_object.static}",
            flush=True,
        )


def _print_scene_object_config_summary(scene_objects, *, prefix: str) -> None:
    print(
        f"{prefix}_SCENE_OBJECT_CONFIG count={len(scene_objects)} "
        f"names={[scene_object.name for scene_object in scene_objects]} "
        f"static={[scene_object.physics.static for scene_object in scene_objects]} "
        f"root_pose={[_scene_object_root_pose_summary(scene_object) for scene_object in scene_objects]}",
        flush=True,
    )


def _scene_object_root_pose_summary(scene_object) -> dict[str, list[float]]:
    return {
        "xyz": list(scene_object.root_pose.xyz),
        "rpy": list(scene_object.root_pose.rpy),
    }

if __name__ == "__main__":
    main()
