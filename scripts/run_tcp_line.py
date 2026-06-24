#!/usr/bin/env python3
"""运行 TCP 笛卡尔直线运动 demo。

脚本导入机器人资产，读取 cuMotion FK/IK 模型，从当前 TCP 位姿生成一条 base
坐标系下的直线 TCP 轨迹，再转成 implicit drive 控制器可执行的关节目标轨迹。
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
from manipulation_project.execution.joint_trajectory_executor import execute_joint_trajectory
from manipulation_project.tasks.move_tcp_line import MoveTcpLineConfig, build_tcp_line_command_trajectory
from manipulation_project.utils.config import load_yaml
from manipulation_project.utils.paths import repo_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/ar5v2_l6v1_l.yaml"))
    parser.add_argument("--controller-config", type=Path, default=Path("configs/controllers"))
    parser.add_argument("--trajectory-config", type=Path, default=Path("configs/trajectories/tcp_line.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/envs/empty_scene.yaml"))
    parser.add_argument("--log", type=Path, default=Path("logs/joint_tracking/run_tcp_line.csv"))
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
    """从环境配置构造机器人 solver iteration 覆盖设置。"""

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


def robot_cumotion_settings(robot_config: dict) -> dict:
    """读取 cuMotion 机器人模型配置。"""

    settings = dict(robot_config.get("cumotion") or {})
    if "ik" in robot_config:
        raise ValueError("Robot config must use cumotion.xrdf_path and cumotion.urdf_path, not ik.*")
    obsolete_keys = sorted(key for key in ("robot_description", "base_urdf") if key in settings)
    if obsolete_keys:
        raise ValueError(f"Robot cuMotion config uses obsolete key(s): {obsolete_keys}; use xrdf_path and urdf_path")
    missing = [key for key in ("xrdf_path", "urdf_path", "flange_frame") if not settings.get(key)]
    if missing:
        raise ValueError(f"Robot cuMotion config is missing required key(s): {missing}")
    return settings


def main() -> None:
    """脚本主入口。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    robot_config_data = load_yaml(args.robot_config)
    controller_profiles = load_controller_profiles(args.controller_config)
    trajectory_config_data = load_yaml(args.trajectory_config)
    env_config_data = load_yaml(args.env_config)

    robot_asset = apply_robot_asset_overrides(RobotAssetConfig.from_mapping(robot_config_data), args)
    controlled_joints = args.controlled_joints or list(robot_config_data.get("controlled_joints", ["all"]))
    robot_cumotion = robot_cumotion_settings(robot_config_data)

    tcp_line_config = MoveTcpLineConfig.from_mapping(trajectory_config_data)
    if tcp_line_config.tcp_frame_name is None:
        tcp_line_config = MoveTcpLineConfig(
            tcp_frame_name=str(robot_cumotion["flange_frame"]),
            start_position=tcp_line_config.start_position,
            target_position=tcp_line_config.target_position,
            target_offset=tcp_line_config.target_offset,
            orientation_mode=tcp_line_config.orientation_mode,
            target_orientation=tcp_line_config.target_orientation,
            target_rpy_deg=tcp_line_config.target_rpy_deg,
            duration_s=tcp_line_config.duration_s,
            sample_hz=tcp_line_config.sample_hz,
            ik_position_tolerance=tcp_line_config.ik_position_tolerance,
            ik_orientation_tolerance=tcp_line_config.ik_orientation_tolerance,
            ik_max_iterations=tcp_line_config.ik_max_iterations,
            ik_bfgs_max_iterations=tcp_line_config.ik_bfgs_max_iterations,
            ik_orientation_weight=tcp_line_config.ik_orientation_weight,
            phase=tcp_line_config.phase,
    )
    if args.duration is not None:
        tcp_line_config = replace(tcp_line_config, duration_s=float(args.duration))
    if args.sample_hz is not None:
        tcp_line_config = replace(tcp_line_config, sample_hz=float(args.sample_hz))
    tcp_line_config.validate()

    if "env" not in env_config_data:
        raise ValueError("Environment config must contain top-level env section")
    env = env_config_data["env"]
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
            print(f"RUN_TCP_LINE_GRAVITY robot_gravity=false disabled_rigid_bodies={len(disabled)}", flush=True)
        else:
            print("RUN_TCP_LINE_GRAVITY robot_gravity=true", flush=True)
        print(f"RUN_TCP_LINE_SOLVER {solver_counts}", flush=True)

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

        dof_names = list(robot.dof_names)
        current_positions = np.asarray(robot.get_joint_positions(), dtype=float).reshape(-1)
        cumotion_config = CuMotionConfig(
            xrdf_path=repo_path(robot_cumotion["xrdf_path"]),
            urdf_path=repo_path(robot_cumotion["urdf_path"]),
            flange_frame=str(robot_cumotion["flange_frame"]),
            default_tcp_frame=tcp_line_config.tcp_frame_name,
            ccd_max_iterations=tcp_line_config.ik_max_iterations,
            bfgs_max_iterations=tcp_line_config.ik_bfgs_max_iterations,
            orientation_weight=tcp_line_config.ik_orientation_weight,
            position_tolerance=tcp_line_config.ik_position_tolerance,
            orientation_tolerance=tcp_line_config.ik_orientation_tolerance,
        )
        command_trajectory, diagnostics = build_tcp_line_command_trajectory(
            dof_names=dof_names,
            command_indices=controller.command_indices,
            current_positions=current_positions,
            config=tcp_line_config,
            cumotion_config=cumotion_config,
        )

        print(
            "RUN_TCP_LINE_IMPORTED "
            f"asset_type={robot_asset.asset_type} asset={asset_path} "
            f"prim_path={articulation_path} num_dof={robot.num_dof}",
            flush=True,
        )
        print("RUN_TCP_LINE_DOF_NAMES " + ", ".join(dof_names), flush=True)
        print(
            "RUN_TCP_LINE_TRAJECTORY "
            f"tcp_frame={tcp_line_config.tcp_frame_name} "
            f"start={diagnostics.start_position.tolist()} "
            f"target={diagnostics.target_position.tolist()} "
            f"duration_s={tcp_line_config.duration_s:.6g} sample_hz={tcp_line_config.sample_hz:.6g} "
            f"points={len(command_trajectory)} max_ik_error={diagnostics.max_position_error:.6g}",
            flush=True,
        )

        execute_joint_trajectory(
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
        print(f"RUN_TCP_LINE_OK log={repo_path(args.log)}", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
