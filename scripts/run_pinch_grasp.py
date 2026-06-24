#!/usr/bin/env python3
"""运行机械臂 + 灵巧手的绳端夹捏抓取 demo。

脚本职责按执行顺序分为：

1. 读取机器人、控制器、环境、绳体和抓取轨迹配置。
2. 启动 Isaac Sim/Isaac Lab，并按配置创建 World、地面、重力和渲染参数。
3. 加载 capsule rope 对象，导入 AR5 + L6 组合机器人资产。
4. 写入机器人 USD/PhysX runtime 覆盖，例如 drive、摩擦、solver iteration 和重力设置。
5. 创建 ``ImplicitDriveController``，把主动关节目标扩展为完整 articulation 控制目标。
6. 创建 mimic follower 映射器，使 L6 从动关节按实际主动关节状态跟随。
7. 构造 pinch TCP，并通过 cuMotion 求解 approach / grasp / lift / wiggle 等阶段目标。
8. 按阶段推进仿真，同时记录关节目标和实际状态到 CSV。

该脚本是任务入口，不直接实现抓取规划细节；抓取阶段编排在
``manipulation_project.tasks.pinch_grasp.PinchGraspTask`` 中。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from manipulation_project.app.launch import launch_simulation_app
from manipulation_project.assets.robot_loader import RobotAssetConfig, import_robot_asset
from manipulation_project.assets.solver_overrides import SolverIterationConfig, apply_solver_iteration_overrides
from manipulation_project.assets.usd_overrides import apply_robot_usd_overrides, disable_robot_gravity
from manipulation_project.backends.cumotion.context import CuMotionConfig
from manipulation_project.controllers.config import implicit_drive_settings, load_controller_profiles, physx_override_configs
from manipulation_project.controllers.implicit_drive_controller import ImplicitDriveController
from manipulation_project.envs.scene_builder import build_world, configure_visuals
from manipulation_project.logging.joint_logger import JointTrackingLogger
from manipulation_project.objects.capsule_rope import CapsuleRopeConfig, add_capsule_rope_reference
from manipulation_project.robots.mimic import MimicFollowerTargetMapper, mjcf_equality_follower_joint_names
from manipulation_project.tasks.pinch_grasp import PinchGraspConfig, PinchGraspTask
from manipulation_project.utils.config import load_yaml
from manipulation_project.utils.paths import repo_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    各配置文件默认指向仓库内的标准抓绳 demo：
    - robot config 选择 AR5V2_L + L6V1_L 组合机器人；
    - controller config 目录提供按部件分组的 implicit drive 参数；
    - env config 提供物理步频、重力和 solver iteration；
    - rope config 提供 capsule rope 资产路径和 prim 路径；
    - grasp config 提供抓取阶段时长、目标姿态、手指闭合角度和 wiggle 参数。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/ar5v2_l6v1_l.yaml"))
    parser.add_argument("--controller-config", type=Path, default=Path("configs/controllers"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/envs/rope_scene.yaml"))
    parser.add_argument("--rope-config", type=Path, default=Path("configs/objects/capsule_rope.yaml"))
    parser.add_argument("--grasp-config", type=Path, default=Path("configs/trajectories/pinch_grasp.yaml"))
    parser.add_argument("--log", type=Path, default=Path("logs/joint_tracking/run_pinch_grasp.csv"))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="最终目标保持到窗口关闭")
    parser.add_argument("--no-grasp", action="store_true", help="只导入机器人和绳体，并短暂保持初始姿态")
    parser.add_argument("--short-smoke", action="store_true", help="覆盖阶段时长，用于快速 headless smoke")
    parser.add_argument("--endpoint", choices=("left", "right"), default=None)
    parser.add_argument("--physics-frequency", type=float, default=None)
    parser.add_argument("--render-frequency", type=float, default=None)
    parser.add_argument("--gravity-z", type=float, default=None)
    parser.add_argument("--enable-robot-gravity", action="store_true")
    return parser.parse_args()


def solver_settings(env_config: dict) -> SolverIterationConfig | None:
    """从环境配置构造机器人 solver iteration 覆盖设置。

    返回:
        配置中存在 ``solver`` 时返回 ``SolverIterationConfig``；不存在时返回
        ``None``，表示不主动覆盖 PhysX 默认 solver 设置。
    """

    solver = env_config.get("solver")
    if solver is None:
        return None
    return SolverIterationConfig(
        solver_type=str(solver.get("type", "TGS")),
        arm_position_iterations=int(solver.get("arm_position_iterations", 32)),
        arm_velocity_iterations=int(solver.get("arm_velocity_iterations", 4)),
        hand_position_iterations=int(solver.get("hand_position_iterations", 32)),
        hand_velocity_iterations=int(solver.get("hand_velocity_iterations", 4)),
        apply_scope=str(solver.get("apply_scope", "arm_hand")),
    )


def robot_cumotion_config(robot_config: dict) -> CuMotionConfig:
    """读取 cuMotion 机器人模型配置。

    机器人配置必须通过 ``cumotion`` 段显式描述 cuMotion 资源和默认求解器参数。
    """

    return CuMotionConfig.from_mapping(robot_config)


def short_smoke_config(config: PinchGraspConfig) -> PinchGraspConfig:
    """把抓取配置压缩成快速 headless smoke 配置。

    该模式用于 CI 或快速导入测试：每个阶段只执行极短时间，禁用 wiggle 和后处理关节扫描，
    目的是尽快验证资产导入、IK 初始化、控制器配置和主循环是否能跑通。
    """

    return replace(
        config,
        lift_height=0.05,
        prep_duration=0.02,
        move_duration=0.02,
        approach_duration=0.02,
        close_duration=0.02,
        lift_duration=0.02,
        wiggle_cycles=0,
        wiggle_duration=0.02,
        final_hold_duration=0.02,
        post_joint_sweep_duration=0.02,
        post_joint_sweep_targets=(),
    )


def hold_initial_pose(robot, world, articulation_action_type, controller, simulation_app, render: bool, logger) -> None:
    """保持当前姿态几步，用于 import smoke。

    ``--no-grasp`` 会走这个分支。它不执行抓取任务，只把当前机器人关节位置作为目标反复下发，
    用于确认机器人资产、驱动参数、mimic follower 和日志系统是否能正常初始化。
    如果同时传入 ``--hold`` 和 ``--gui``，会持续保持到 Isaac 窗口关闭。
    """

    full_target = np.asarray(robot.get_joint_positions(), dtype=float)
    full_velocity = np.zeros(robot.num_dof, dtype=float)
    step = 0
    while step < 3 or (simulation_app is not None and simulation_app.is_running()):
        controller.apply(articulation_action_type, full_target, full_velocity)
        world.step(render=render)
        if logger is not None:
            logger.write(
                step=step,
                time_s=(step + 1) * float(world.get_physics_dt()),
                phase="initial_hold",
                drive_update=True,
                desired_position=full_target[controller.driven_indices],
                actual_position=np.asarray(robot.get_joint_positions(), dtype=float)[controller.driven_indices],
                desired_velocity=full_velocity[controller.driven_indices],
                actual_velocity=np.asarray(robot.get_joint_velocities(), dtype=float)[controller.driven_indices],
            )
        step += 1
        if simulation_app is None and step >= 3:
            break


def main() -> None:
    """脚本主入口。"""

    # Isaac/Kit 日志很多，开启行缓冲可以保证 RUN_PINCH_GRASP_* 状态行尽快刷出，
    # 方便 live log、调试脚本和外部监控程序读取。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    # 先加载所有 YAML 配置。这里还没有启动 Isaac Sim，尽量把纯 Python 的配置错误提前暴露，
    # 避免启动 GUI 后才因为路径或字段缺失失败。
    robot_config = load_yaml(args.robot_config)
    controller_profiles = load_controller_profiles(args.controller_config)
    env_config = load_yaml(args.env_config)
    rope_config_data = load_yaml(args.rope_config)
    grasp_config_data = load_yaml(args.grasp_config)

    # RobotAssetConfig 只描述“如何把资产导入 stage”，例如 asset_type、asset_path、prim_path。
    # controlled_joints 则描述控制器主动下发目标的关节集合，mimic follower 会在运行时自动补齐。
    robot_asset = RobotAssetConfig.from_mapping(robot_config)
    controlled_joints = list(robot_config.get("controlled_joints", ["all"]))
    robot_cumotion = robot_cumotion_config(robot_config)

    # 把原始 YAML dict 转成任务使用的 dataclass。命令行参数只覆盖少量常用字段，
    # 复杂参数仍建议写在 YAML 中，保证实验可复现。
    rope_config = CapsuleRopeConfig.from_mapping(rope_config_data)
    grasp_config = PinchGraspConfig.from_mapping(grasp_config_data)
    if args.endpoint is not None:
        grasp_config = replace(grasp_config, endpoint=args.endpoint)
    if args.short_smoke:
        grasp_config = short_smoke_config(grasp_config)

    # 物理步频决定接触稳定性和控制刷新上限；渲染步频只影响 GUI/相机刷新。
    # headless 模式下 rendering_dt 之后会直接跟 physics_dt 对齐，避免无意义的渲染节拍。
    if "env" not in env_config:
        raise ValueError("Environment config must contain top-level env section")
    env = env_config["env"]
    physics_frequency = float(args.physics_frequency if args.physics_frequency is not None else env.get("physics_frequency", 600.0))
    render_frequency = float(args.render_frequency if args.render_frequency is not None else env.get("render_frequency", 100.0))
    gravity_z = float(args.gravity_z if args.gravity_z is not None else env.get("gravity_z", -9.81))
    if physics_frequency <= 0 or render_frequency <= 0:
        raise ValueError("physics and render frequencies must be positive")

    # Isaac Sim 首次启动时需要接受 EULA；这里设置默认值，避免 headless 运行卡在交互确认。
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    simulation_app = launch_simulation_app(gui=args.gui)
    try:
        # Isaac 相关 import 必须放在 SimulationApp 启动之后，否则部分扩展和 USD context 尚未初始化。
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        import omni.usd

        # 创建 World。physics_dt 控制 PhysX step，rendering_dt 控制 GUI 刷新间隔。
        physics_dt = 1.0 / physics_frequency
        rendering_dt = 1.0 / render_frequency if args.gui else physics_dt
        world = build_world(physics_dt=physics_dt, rendering_dt=rendering_dt, gravity_z=gravity_z)
        if args.gui:
            configure_visuals()

        # 绳体对象以 USD reference 的方式挂到 stage 中。这里返回的 rope_model 主要用于打印诊断，
        # 真实碰撞体和关节由 USD/PhysX 在 stage 中维护。
        stage = omni.usd.get_context().get_stage()
        rope_model = add_capsule_rope_reference(stage, rope_config)
        print(
            "RUN_PINCH_GRASP_ROPE "
            f"asset={rope_config.asset_file()} prim_path={rope_config.prim_path} "
            f"segments={rope_config.segments} shape={rope_config.shape} "
            f"bodies={len(rope_model['bodies'])} joints={len(rope_model['joints'])}",
            flush=True,
        )

        # 导入机器人资产。MJCF/URDF 导入后会生成 stage prim；后续控制和 PhysX 覆盖都基于该 prim。
        articulation_path, asset_path, imported_root_path = import_robot_asset(robot_asset)
        mjcf_path = asset_path if robot_asset.asset_type == "mjcf" else None

        # 对刚导入的 USD prim 做运行时覆盖：drive 参数、关节摩擦、最大力、碰撞近似等。
        # 这些覆盖不会修改原始资产文件，只影响当前 stage。
        apply_robot_usd_overrides(
            imported_root_path,
            physx_override_configs(controller_profiles),
            driven_joint_names=controlled_joints,
            mjcf_path=mjcf_path,
        )

        # 根据 env.solver 覆盖 articulation/rigid body 的 PhysX solver iteration。
        # 抓绳接触和灵巧手多关节链都对 solver iteration 较敏感。
        solver_config = solver_settings(env_config)
        solver_counts = (
            apply_solver_iteration_overrides(stage, articulation_path, solver_config)
            if solver_config is not None
            else {"configured": 0}
        )

        # 默认关闭机器人刚体重力，让机器人主要由 position drive 控制。
        # 如果需要测试真实重力下的下垂或力控行为，可以传 --enable-robot-gravity。
        if not args.enable_robot_gravity:
            disabled = disable_robot_gravity(imported_root_path)
            print(f"RUN_PINCH_GRASP_GRAVITY robot_gravity=false disabled_rigid_bodies={len(disabled)}", flush=True)
        else:
            print("RUN_PINCH_GRASP_GRAVITY robot_gravity=true", flush=True)
        print(f"RUN_PINCH_GRASP_SOLVER {solver_counts}", flush=True)

        # 将导入后的 articulation 包装为 Isaac Sim SingleArticulation，并 reset world 以初始化 handles。
        robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name=robot_asset.name))
        world.reset()
        world.get_physics_context().set_gravity(gravity_z)
        if not args.enable_robot_gravity:
            robot.disable_gravity()
        robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=float))

        # implicit drive 控制器负责：
        # - 配置 articulation controller 为 position 模式；
        # - 写入主动关节和 follower 关节的 stiffness/damping/max effort；
        # - 把命令关节目标扩展为完整 DOF 目标。
        controller = ImplicitDriveController(
            robot,
            joint_names=controlled_joints,
            settings=implicit_drive_settings(controller_profiles),
            mjcf_path=mjcf_path,
        )
        controller.configure_runtime()

        # L6 手的 DIP 等 follower 关节由 MJCF equality 描述。运行时根据实际 master 关节状态
        # 更新 follower 目标，避免 follower 跟随“命令目标”而不是“实际主动关节”导致超前。
        follower_mapper = MimicFollowerTargetMapper(list(robot.dof_names), mjcf_path)
        mimic_names = mjcf_equality_follower_joint_names(mjcf_path)

        # 日志只记录实际受驱动的 DOF，即主动关节 + mimic follower。flush_interval_steps 控制
        # CSV 刷盘频率，避免每个 physics step 都 flush 造成 I/O 开销过大。
        driven_joint_names = [list(robot.dof_names)[int(index)] for index in controller.driven_indices]
        flush_interval_steps = max(1, int(round(0.05 / float(world.get_physics_dt()))))
        logger = JointTrackingLogger(repo_path(args.log), driven_joint_names, flush_interval_steps=flush_interval_steps)
        print(
            "RUN_PINCH_GRASP_IMPORTED "
            f"asset={asset_path} prim_path={articulation_path} num_dof={robot.num_dof} "
            f"mimic_joint_names={sorted(mimic_names)} follower_relations={follower_mapper.relations}",
            flush=True,
        )
        print("RUN_PINCH_GRASP_DOF_NAMES " + ", ".join(list(robot.dof_names)), flush=True)

        try:
            if args.no_grasp:
                # 仅做导入和控制器 smoke test，不构造 pinch TCP，也不调用 cuMotion。
                hold_initial_pose(
                    robot,
                    world,
                    ArticulationAction,
                    controller,
                    simulation_app if args.hold else None,
                    args.gui,
                    logger,
                )
                print("RUN_PINCH_GRASP_HOLD_OK", flush=True)
            else:
                # PinchGraspTask 会：
                # - 根据闭合手型计算 pinch TCP 相对法兰的 offset；
                # - 生成临时 URDF，把 pinch TCP 作为 fixed frame 挂到 robot cumotion.flange_frame；
                # - 使用 cuMotion 求解 approach/grasp/lift/wiggle 关键帧；
                # - 按阶段插值并通过 controller/follower_mapper 下发到 Isaac。
                task = PinchGraspTask(
                    config=grasp_config,
                    rope_config=rope_config,
                    mjcf_path=asset_path,
                    cumotion_config=robot_cumotion,
                )
                result = task.run(
                    robot=robot,
                    world=world,
                    articulation_action_type=ArticulationAction,
                    driven_indices=controller.driven_indices,
                    simulation_app=simulation_app,
                    render=args.gui,
                    drive_logger=logger,
                    follower_mapper=follower_mapper,
                )
                print(
                    "RUN_PINCH_GRASP_OK "
                    f"steps={result['steps']} ik={result['ik']} log={repo_path(args.log)}",
                    flush=True,
                )
        finally:
            # 无论任务成功、失败还是用户 Ctrl+C，都尽量关闭 CSV 文件，避免最后几行日志丢失。
            logger.close()
    finally:
        # 必须关闭 SimulationApp，否则 Kit/Isaac 进程和扩展资源可能残留。
        simulation_app.close()


if __name__ == "__main__":
    main()
