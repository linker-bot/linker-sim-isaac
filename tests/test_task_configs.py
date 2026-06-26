from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from manipulation_project.assets.asset_paths import (
    DEFAULT_AR5_RIGHT_URDF,
    DEFAULT_L6_RIGHT_URDF,
)
from manipulation_project.assets.robot_loader import RobotAssetConfig
from manipulation_project.assets.solver_overrides import SolverIterationConfig
from manipulation_project.backends.cumotion.context import CuMotionConfig
from manipulation_project.objects.capsule_rope import CapsuleRopeConfig, endpoint_center
from manipulation_project.tasks.pinch_grasp import (
    PinchGraspConfig,
    grasp_target_position,
)
from manipulation_project.utils.config import load_yaml


def test_default_robot_config_paths_exist() -> None:
    config = load_yaml("configs/robots/ar5v2_l6v1_l.yaml")
    robot = RobotAssetConfig.from_mapping(config)
    assert robot.asset_path.is_file()
    assert config["controlled_joints"]


def test_robot_configs_are_cumotion_only() -> None:
    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        assert "cumotion" in config
        cumotion = CuMotionConfig.from_mapping(config)
        assert Path(cumotion.xrdf_path).is_file()
        assert Path(cumotion.urdf_path).is_file()
        assert cumotion.flange_frame
        assert cumotion.custom_tcp_frame is None or cumotion.custom_tcp_frame
        ik_config = cumotion.kinematics.ik
        assert ik_config.position_tolerance >= 0.0
        assert ik_config.orientation_tolerance >= 0.0
        assert ik_config.ccd_max_iterations > 0
        assert ik_config.bfgs_max_iterations > 0
        assert "robot_description" not in config["cumotion"]
        assert "base_urdf" not in config["cumotion"]
        assert "lula" not in config
        assert "ik" not in config


def test_cumotion_profiles_and_examples_are_valid_defaults() -> None:
    """内置 cuMotion profile/example 应只提供可合并的后端默认值。"""

    for path in sorted(Path("configs/cumotion").glob("*.yaml")):
        config = load_yaml(path)
        assert "cumotion" in config
        assert "motion_planning" not in config
        cumotion = config["cumotion"]
        assert "xrdf_path" not in cumotion
        assert "urdf_path" not in cumotion
        assert "flange_frame" not in cumotion
        assert "kinematics" in cumotion
        assert "ik" in cumotion["kinematics"]
        ik_config = cumotion["kinematics"]["ik"]
        assert ik_config["position_tolerance"] >= 0.0
        assert ik_config["orientation_tolerance"] >= 0.0
        assert ik_config["ccd_max_iterations"] > 0
        assert ik_config["bfgs_max_iterations"] > 0
        assert "motion_planner" in cumotion
        grasp_config = PinchGraspConfig.from_mapping(
            {"grasp": {"motion_planning": cumotion["motion_planner"]}}
        )
        grasp_config.motion_planning.validate()


def test_cumotion_config_parses_grouped_kinematics() -> None:
    """推荐配置结构应把 IK/FK 参数放在 cumotion.kinematics 下。"""

    config = CuMotionConfig.from_mapping(
        {
            "cumotion": {
                "xrdf_path": "robot.xrdf",
                "urdf_path": "robot.urdf",
                "flange_frame": "flange",
                "kinematics": {
                    "ik": {
                        "cspace_seeds": [0.1, 0.2],
                        "position_tolerance": 0.003,
                        "orientation_tolerance": 0.04,
                        "ccd_max_iterations": 11,
                        "bfgs_max_iterations": 12,
                        "orientation_weight": 0.2,
                        "collision_free_params": {"max_iterations": 7},
                    },
                    "fk": {},
                },
            }
        }
    )

    ik_config = config.kinematics.ik
    np.testing.assert_allclose(ik_config.cspace_seeds, [0.1, 0.2])
    assert ik_config.position_tolerance == 0.003
    assert ik_config.orientation_tolerance == 0.04
    assert ik_config.ccd_max_iterations == 11
    assert ik_config.bfgs_max_iterations == 12
    assert ik_config.orientation_weight == 0.2
    assert ik_config.collision_free_params == {"max_iterations": 7}


def test_cumotion_config_accepts_legacy_flat_ik_fields() -> None:
    """旧式 flat IK 字段只作为兼容入口，解析后统一落到 kinematics.ik。"""

    config = CuMotionConfig.from_mapping(
        {
            "cumotion": {
                "xrdf_path": "robot.xrdf",
                "urdf_path": "robot.urdf",
                "flange_frame": "flange",
                "ik_cspace_seeds": [0.0, 0.1],
                "position_tolerance": 0.01,
                "orientation_weight": 0.5,
                "collision_free_ik_params": {"max_iterations": 3},
            }
        }
    )

    ik_config = config.kinematics.ik
    np.testing.assert_allclose(ik_config.cspace_seeds, [0.0, 0.1])
    assert ik_config.position_tolerance == 0.01
    assert ik_config.orientation_weight == 0.5
    assert ik_config.collision_free_params == {"max_iterations": 3}
    assert config.position_tolerance == ik_config.position_tolerance


def test_cumotion_profile_merge_order() -> None:
    """profile 默认值应低于 robot YAML 和 trajectory YAML 的显式配置。"""

    import importlib.util

    script_path = Path("scripts/run_pinch_grasp.py")
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    profile = {
        "cumotion": {
            "kinematics": {
                "ik": {
                    "position_tolerance": 0.01,
                    "orientation_weight": 0.1,
                },
            },
            "motion_planner": {
                "planning_pipeline": "graph_search",
                "graph_search": {"generate_interpolated_path": True},
            },
        },
    }
    robot = {
        "cumotion": {
            "xrdf_path": "robot.xrdf",
            "urdf_path": "robot.urdf",
            "flange_frame": "flange",
            "kinematics": {"ik": {"position_tolerance": 0.002}},
        }
    }
    grasp = {
        "grasp": {
            "endpoint": "left",
            "motion_planning": {
                "planning_pipeline": "trajectory_optimization",
            },
            "pre_pinch_hand_targets": {},
            "closed_pinch_hand_targets": {},
        }
    }

    merged_robot = module.merged_robot_config_with_cumotion_profile(robot, profile)
    merged_ik = merged_robot["cumotion"]["kinematics"]["ik"]
    assert merged_ik["position_tolerance"] == 0.002
    assert merged_ik["orientation_weight"] == 0.1
    assert merged_robot["cumotion"]["flange_frame"] == "flange"

    merged_grasp = module.merged_grasp_config_with_cumotion_profile(grasp, profile)
    motion_planning = merged_grasp["grasp"]["motion_planning"]
    assert motion_planning["planning_pipeline"] == "trajectory_optimization"
    assert motion_planning["graph_search"]["generate_interpolated_path"] is True


def test_right_side_urdf_assets_exist() -> None:
    assert DEFAULT_AR5_RIGHT_URDF.is_file()
    assert DEFAULT_L6_RIGHT_URDF.is_file()


def test_robot_asset_mesh_references_exist() -> None:
    asset_files = [
        *Path("assets/single_system").glob("**/*.urdf"),
        *Path("assets/single_system").glob("**/*.xml"),
        *Path("assets/combined_system").glob("**/*.xml"),
    ]
    assert asset_files

    missing: list[tuple[Path, str]] = []
    for asset_file in asset_files:
        root = ET.parse(asset_file).getroot()
        for element in root.iter():
            reference = element.get("filename") or element.get("file")
            if not reference or not reference.lower().endswith(".stl"):
                continue
            if not (asset_file.parent / reference).resolve().is_file():
                missing.append((asset_file, reference))

    assert missing == []


def test_default_joint_trajectory_config() -> None:
    config = load_yaml("configs/trajectories/joint_target.yaml")
    trajectory = config["trajectory"]
    assert trajectory["type"] == "joint_target"
    assert trajectory["duration"] > 0
    assert trajectory["targets"]


def test_default_rope_and_grasp_configs() -> None:
    rope = CapsuleRopeConfig.from_mapping(
        load_yaml("configs/objects/capsule_rope.yaml")
    )
    rope.validate()
    assert rope.asset_file().is_file()
    assert rope.prim_path == "/World/CapsuleRope"
    assert rope.root_path == "/CapsuleRope"
    assert rope.radius is not None and rope.radius > 0.0
    assert rope.twist_limit is not None and rope.twist_limit > 0.0
    grasp = PinchGraspConfig.from_mapping(
        load_yaml("configs/trajectories/pinch_grasp.yaml")
    )
    grasp.validate()
    left_center = endpoint_center(rope, "left")
    target = grasp_target_position(grasp, rope)
    assert target[1] == left_center[1]
    assert target[2] > left_center[2]


def test_task_configs_reject_obsolete_shapes() -> None:
    try:
        RobotAssetConfig.from_mapping({"asset_path": "assets/example.xml"})
    except ValueError:
        pass
    else:
        raise AssertionError(
            "RobotAssetConfig accepted config without top-level robot section"
        )

    try:
        CapsuleRopeConfig.from_mapping({"segments": 18, "length": 0.75})
    except ValueError:
        pass
    else:
        raise AssertionError(
            "CapsuleRopeConfig accepted config without object/rope sections"
        )

    try:
        PinchGraspConfig.from_mapping({"endpoint": "left"})
    except ValueError:
        pass
    else:
        raise AssertionError(
            "PinchGraspConfig accepted config without top-level grasp section"
        )


def test_env_configs_provide_solver_settings() -> None:
    """所有项目内置场景都应显式提供 PhysX solver 设置。"""

    for path in sorted(Path("configs/envs").glob("*.yaml")):
        config = load_yaml(path)
        assert "solver" in config, f"{path} must provide solver settings"
        solver = config["solver"]
        parsed = SolverIterationConfig(
            solver_type=str(solver["type"]),
            arm_position_iterations=int(solver["arm_position_iterations"]),
            arm_velocity_iterations=int(solver["arm_velocity_iterations"]),
            hand_position_iterations=int(solver["hand_position_iterations"]),
            hand_velocity_iterations=int(solver["hand_velocity_iterations"]),
            apply_scope=str(solver["apply_scope"]),
        )
        assert parsed.solver_type.upper() in {"PGS", "TGS"}
        assert parsed.apply_scope in {"arm", "hand", "arm_hand", "articulation"}


def test_solver_settings_are_optional_in_scripts() -> None:
    """外部自定义场景没写 solver 时，脚本不主动覆盖 PhysX 默认值。"""

    import importlib.util

    for script_path in (Path("scripts/run_pinch_grasp.py"),):
        spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.solver_settings({"env": {"name": "custom_scene"}}) is None
