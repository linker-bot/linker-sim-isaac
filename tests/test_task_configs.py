from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

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
        assert cumotion.position_tolerance >= 0.0
        assert cumotion.orientation_tolerance >= 0.0
        assert cumotion.ccd_max_iterations > 0
        assert cumotion.bfgs_max_iterations > 0
        assert "robot_description" not in config["cumotion"]
        assert "base_urdf" not in config["cumotion"]
        assert "lula" not in config
        assert "ik" not in config


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
