#!/usr/bin/env python3
"""运行机械臂 + 灵巧手的绳端夹捏抓取 demo。

脚本会读取 capsule rope USD 资产、导入机器人、配置 implicit drive、构造 pinch TCP，并通过 IK
规划和执行 approach / grasp / lift / wiggle 等阶段。
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
    """解析命令行参数。"""

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


def robot_cumotion_settings(robot_config: dict) -> dict:
    """读取 cuMotion 机器人模型配置。"""

    settings = dict(robot_config.get("cumotion") or {})
    if "ik" in robot_config:
        raise ValueError("Robot config must use cumotion.xrdf_path and cumotion.urdf_path, not ik.*")
    obsolete_keys = sorted(key for key in ("robot_description", "base_urdf") if key in settings)
    if obsolete_keys:
        raise ValueError(f"Robot cuMotion config uses obsolete key(s): {obsolete_keys}; use xrdf_path and urdf_path")
    if not settings:
        raise ValueError("Robot config must provide cumotion.xrdf_path and cumotion.urdf_path")
    missing = [key for key in ("xrdf_path", "urdf_path", "flange_frame") if not settings.get(key)]
    if missing:
        raise ValueError(f"Robot cuMotion config is missing required key(s): {missing}")
    return settings


def short_smoke_config(config: PinchGraspConfig) -> PinchGraspConfig:
    """把抓取配置压缩成快速 headless smoke 配置。"""

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
    """保持当前姿态几步，用于 import smoke。"""

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

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    robot_config = load_yaml(args.robot_config)
    controller_profiles = load_controller_profiles(args.controller_config)
    env_config = load_yaml(args.env_config)
    rope_config_data = load_yaml(args.rope_config)
    grasp_config_data = load_yaml(args.grasp_config)

    robot_asset = RobotAssetConfig.from_mapping(robot_config)
    controlled_joints = list(robot_config.get("controlled_joints", ["all"]))
    robot_cumotion = robot_cumotion_settings(robot_config)

    rope_config = CapsuleRopeConfig.from_mapping(rope_config_data)
    grasp_config = PinchGraspConfig.from_mapping(grasp_config_data)
    if args.endpoint is not None:
        grasp_config = replace(grasp_config, endpoint=args.endpoint)
    if args.short_smoke:
        grasp_config = short_smoke_config(grasp_config)

    if "env" not in env_config:
        raise ValueError("Environment config must contain top-level env section")
    env = env_config["env"]
    physics_frequency = float(args.physics_frequency if args.physics_frequency is not None else env.get("physics_frequency", 600.0))
    render_frequency = float(args.render_frequency if args.render_frequency is not None else env.get("render_frequency", 100.0))
    gravity_z = float(args.gravity_z if args.gravity_z is not None else env.get("gravity_z", -9.81))
    if physics_frequency <= 0 or render_frequency <= 0:
        raise ValueError("physics and render frequencies must be positive")

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    simulation_app = launch_simulation_app(gui=args.gui)
    try:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        import omni.usd

        physics_dt = 1.0 / physics_frequency
        rendering_dt = 1.0 / render_frequency if args.gui else physics_dt
        world = build_world(physics_dt=physics_dt, rendering_dt=rendering_dt, gravity_z=gravity_z)
        if args.gui:
            configure_visuals()

        stage = omni.usd.get_context().get_stage()
        rope_model = add_capsule_rope_reference(stage, rope_config)
        print(
            "RUN_PINCH_GRASP_ROPE "
            f"asset={rope_config.asset_file()} prim_path={rope_config.prim_path} "
            f"segments={rope_config.segments} shape={rope_config.shape} "
            f"bodies={len(rope_model['bodies'])} joints={len(rope_model['joints'])}",
            flush=True,
        )

        articulation_path, asset_path, imported_root_path = import_robot_asset(robot_asset)
        mjcf_path = asset_path if robot_asset.asset_type == "mjcf" else None
        apply_robot_usd_overrides(
            imported_root_path,
            physx_override_configs(controller_profiles),
            driven_joint_names=controlled_joints,
            mjcf_path=mjcf_path,
        )
        solver_config = solver_settings(env_config)
        solver_counts = (
            apply_solver_iteration_overrides(stage, articulation_path, solver_config)
            if solver_config is not None
            else {"configured": 0}
        )
        if not args.enable_robot_gravity:
            disabled = disable_robot_gravity(imported_root_path)
            print(f"RUN_PINCH_GRASP_GRAVITY robot_gravity=false disabled_rigid_bodies={len(disabled)}", flush=True)
        else:
            print("RUN_PINCH_GRASP_GRAVITY robot_gravity=true", flush=True)
        print(f"RUN_PINCH_GRASP_SOLVER {solver_counts}", flush=True)

        robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name=robot_asset.name))
        world.reset()
        world.get_physics_context().set_gravity(gravity_z)
        if not args.enable_robot_gravity:
            robot.disable_gravity()
        robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=float))

        controller = ImplicitDriveController(
            robot,
            joint_names=controlled_joints,
            settings=implicit_drive_settings(controller_profiles),
            mjcf_path=mjcf_path,
        )
        controller.configure_runtime()
        follower_mapper = MimicFollowerTargetMapper(list(robot.dof_names), mjcf_path)
        mimic_names = mjcf_equality_follower_joint_names(mjcf_path)

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
                task = PinchGraspTask(
                    config=grasp_config,
                    rope_config=rope_config,
                    mjcf_path=asset_path,
                    cumotion_xrdf_path=repo_path(robot_cumotion["xrdf_path"]),
                    cumotion_urdf_path=repo_path(robot_cumotion["urdf_path"]),
                    parent_frame=str(robot_cumotion["flange_frame"]),
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
            logger.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
