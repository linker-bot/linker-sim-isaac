#!/usr/bin/env python3
"""运行固定时长的关节目标运动。

该脚本是项目的最小实际运动 smoke/demo 入口：导入机器人资产，构造 implicit
position drive 控制器，从配置文件读取稀疏关节目标，然后在 Isaac Sim 中执行并记录
CSV 跟踪日志。
"""

from __future__ import annotations

import argparse
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
from manipulation_project.tasks.move_joint_targets import (
    MoveJointTargetsConfig,
    build_command_trajectory_from_sparse_targets,
    execute_joint_target_trajectory,
)
from manipulation_project.utils.config import load_yaml
from manipulation_project.utils.paths import repo_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回:
        argparse namespace。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/ar5_l6.yaml"))
    parser.add_argument("--controller-config", type=Path, default=Path("configs/controllers/implicit_position_drive.yaml"))
    parser.add_argument("--trajectory-config", type=Path, default=Path("configs/trajectories/joint_target.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/envs/empty_scene.yaml"))
    parser.add_argument("--log", type=Path, default=Path("logs/joint_tracking/run_joint_target.csv"))
    parser.add_argument("--joint", nargs=2, action="append", metavar=("NAME", "RAD"), help="覆盖或追加目标关节")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--sample-hz", type=float, default=None)
    parser.add_argument("--physics-frequency", type=float, default=None)
    parser.add_argument("--render-frequency", type=float, default=None)
    parser.add_argument("--gravity-z", type=float, default=None)
    parser.add_argument("--asset-type", choices=("mjcf", "urdf"), default=None)
    parser.add_argument("--asset-path", type=Path, default=None)
    parser.add_argument("--asset-prim-path", default=None)
    parser.add_argument("--controlled-joints", nargs="+", default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="运动结束后继续保持窗口")
    parser.add_argument("--enable-robot-gravity", action="store_true")
    return parser.parse_args()


def solver_settings(env_config: dict) -> SolverIterationConfig | None:
    """从环境配置构造机器人 solver iteration 覆盖设置。

    返回:
        配置中存在 ``solver`` 时返回 ``SolverIterationConfig``；不存在时返回
        ``None``，表示保持 Isaac/PhysX 默认 solver 设置。
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


def apply_robot_asset_overrides(asset: RobotAssetConfig, args: argparse.Namespace) -> RobotAssetConfig:
    """按命令行参数覆盖机器人资产配置。"""

    result = asset
    if args.asset_type is not None:
        result = RobotAssetConfig(
            asset_type=args.asset_type,
            asset_path=result.asset_path,
            prim_path=result.prim_path,
            name=result.name,
            urdf_drive_type=result.urdf_drive_type,
        )
    if args.asset_path is not None:
        result = RobotAssetConfig(
            asset_type=result.asset_type,
            asset_path=repo_path(args.asset_path),
            prim_path=result.prim_path,
            name=result.name,
            urdf_drive_type=result.urdf_drive_type,
        )
    if args.asset_prim_path is not None:
        result = RobotAssetConfig(
            asset_type=result.asset_type,
            asset_path=result.asset_path,
            prim_path=args.asset_prim_path,
            name=result.name,
            urdf_drive_type=result.urdf_drive_type,
        )
    return result


def main() -> None:
    """脚本主入口。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if args.duration is not None and args.duration < 0:
        raise ValueError("--duration cannot be negative")
    if args.sample_hz is not None and args.sample_hz <= 0:
        raise ValueError("--sample-hz must be positive")

    robot_config_data = load_yaml(args.robot_config)
    controller_profiles = load_controller_profiles(args.controller_config)
    trajectory_config_data = load_yaml(args.trajectory_config)
    env_config_data = load_yaml(args.env_config)

    robot_asset = apply_robot_asset_overrides(RobotAssetConfig.from_mapping(robot_config_data), args)
    controlled_joints = args.controlled_joints or list(robot_config_data.get("controlled_joints", ["all"]))

    trajectory = trajectory_config_data.get("trajectory", trajectory_config_data)
    targets = {str(name): float(value) for name, value in dict(trajectory.get("targets", {})).items()}
    if args.joint:
        for name, value in args.joint:
            targets[str(name)] = float(value)
    if not targets:
        raise ValueError("No joint targets were provided in config or via --joint")

    duration_s = float(args.duration if args.duration is not None else trajectory.get("duration", 2.0))
    sample_hz = float(args.sample_hz if args.sample_hz is not None else trajectory.get("sample_hz", 200.0))
    task_config = MoveJointTargetsConfig(
        targets=targets,
        duration_s=duration_s,
        sample_hz=sample_hz,
        interpolation=str(trajectory.get("interpolation", "smoothstep")),
    )

    env = env_config_data.get("env", env_config_data)
    physics_frequency = float(args.physics_frequency if args.physics_frequency is not None else env.get("physics_frequency", 400.0))
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
        articulation_path, asset_path, imported_root_path = import_robot_asset(robot_asset)
        mjcf_path = asset_path if robot_asset.asset_type == "mjcf" else None
        apply_robot_usd_overrides(
            imported_root_path,
            physx_override_configs(controller_profiles),
            driven_joint_names=controlled_joints,
            mjcf_path=mjcf_path,
        )
        solver_config = solver_settings(env_config_data)
        solver_counts = (
            apply_solver_iteration_overrides(stage, articulation_path, solver_config)
            if solver_config is not None
            else {"configured": 0}
        )
        if not args.enable_robot_gravity:
            disabled = disable_robot_gravity(imported_root_path)
            print(f"RUN_JOINT_TARGET_GRAVITY robot_gravity=false disabled_rigid_bodies={len(disabled)}", flush=True)
        else:
            print("RUN_JOINT_TARGET_GRAVITY robot_gravity=true", flush=True)
        print(f"RUN_JOINT_TARGET_SOLVER {solver_counts}", flush=True)

        robot = world.scene.add(SingleArticulation(prim_path=articulation_path, name=robot_asset.name))
        world.reset()
        world.get_physics_context().set_gravity(gravity_z)
        if not args.enable_robot_gravity:
            robot.disable_gravity()

        controller = ImplicitDriveController(
            robot,
            joint_names=controlled_joints,
            settings=implicit_drive_settings(controller_profiles),
            mjcf_path=mjcf_path,
        )
        controller.configure_runtime()

        dof_names = list(robot.dof_names)
        current_positions = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)
        command_trajectory = build_command_trajectory_from_sparse_targets(
            dof_names=dof_names,
            command_indices=controller.command_indices,
            current_positions=current_positions,
            config=task_config,
        )

        print(
            "RUN_JOINT_TARGET_IMPORTED "
            f"asset_type={robot_asset.asset_type} asset={asset_path} "
            f"prim_path={articulation_path} num_dof={robot.num_dof}",
            flush=True,
        )
        print("RUN_JOINT_TARGET_DOF_NAMES " + ", ".join(dof_names), flush=True)
        print(
            "RUN_JOINT_TARGET_TRAJECTORY "
            f"duration_s={duration_s:.6g} sample_hz={sample_hz:.6g} "
            f"points={len(command_trajectory)} targets={targets}",
            flush=True,
        )

        execute_joint_target_trajectory(
            robot=robot,
            world=world,
            articulation_action_type=ArticulationAction,
            controller=controller,
            trajectory=command_trajectory,
            log_path=repo_path(args.log),
            render=args.gui,
            simulation_app=simulation_app,
            hold=args.hold,
        )
        print(f"RUN_JOINT_TARGET_OK log={repo_path(args.log)}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
