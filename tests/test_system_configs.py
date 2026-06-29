from __future__ import annotations

from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from linkerbot_sim.assets.asset_paths import (
    DEFAULT_AR5_RIGHT_URDF,
    DEFAULT_L6_RIGHT_URDF,
    DEFAULT_WORKSTATION_V1_ARMBASE_URDF,
    DEFAULT_WORKSTATION_V1_TABLEBASE_URDF,
)
from linkerbot_sim.assets.robot_loader import (
    DualRobotExecutionConfig,
    RobotGravityPolicy,
    RobotAssetConfig,
    RobotExecutionConfig,
)
from linkerbot_sim.assets.solver_overrides import SolverIterationConfig
from linkerbot_sim.app.runtime_settings import EnvRuntimeSettings
from linkerbot_sim.backends.cumotion.context import CuMotionConfig
from linkerbot_sim.backends.cumotion.dual_urdf import (
    prepare_cumotion_config_from_robot_config,
)
from linkerbot_sim.backends.cumotion.profile_config import (
    merged_robot_config_with_cumotion_profile,
    motion_planner_config_from_profile,
)
from linkerbot_sim.envs.scene_objects import (
    SceneObjectConfig,
    scene_objects_from_env_config,
)
from linkerbot_sim.objects.capsule_rope import CapsuleRopeConfig, endpoint_center
from pinch_grasp import grasp_target_position
from linkerbot_sim.utils.config import load_yaml


def test_default_robot_config_paths_exist() -> None:
    config = load_yaml("configs/robots/ar5v2_l6v1_l.yaml")
    robot = RobotAssetConfig.from_mapping(config)
    execution = RobotExecutionConfig.from_mapping(config)
    assert robot.asset_path.is_file()
    assert execution.controlled_joints == ("all",)


def test_robot_configs_are_cumotion_only() -> None:
    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        assert "cumotion" in config
        _assert_robot_cumotion_section_contains_only_model_resources(config)
        if "robots" in config:
            dual = DualRobotExecutionConfig.from_mapping(config)
            assert dual.left.robot.asset_path.is_file()
            assert dual.right.robot.asset_path.is_file()
            assert dual.left.robot.import_config.collision_approximation in {
                "convex_decomposition",
                "convex_hull",
            }
            assert dual.right.robot.import_config.collision_approximation in {
                "convex_decomposition",
                "convex_hull",
            }
            assert dual.left.robot.gravity_policy.enabled_for_component("arm") is False
            assert dual.right.robot.gravity_policy.enabled_for_component("hand") is False
            assert len(dual.left.root_pose.xyz) == 3
            assert len(dual.left.root_pose.rpy) == 3
            assert len(dual.right.root_pose.xyz) == 3
            assert len(dual.right.root_pose.rpy) == 3
            assert dual.left.controlled_joints
            assert dual.right.controlled_joints
            assert "robot" not in config
            assert "controlled_joints" not in config
            assert "dual_arm" not in config
            for side in ("left", "right"):
                side_cumotion = config["cumotion"].get(side)
                assert isinstance(side_cumotion, dict)
                assert Path(side_cumotion["xrdf_path"]).is_file()
                assert Path(side_cumotion["urdf_path"]).is_file()
                assert side_cumotion["flange_frame"]
            assert "xrdf_path" not in config["cumotion"]
            assert "urdf_path" not in config["cumotion"]
            assert "flange_frame" not in config["cumotion"]
            assert "default_side" not in config["cumotion"]
            prepared = prepare_cumotion_config_from_robot_config(config)
            cumotion = prepared.backend_config
            assert prepared.generated_assets is True
            assert prepared.urdf_path.is_file()
            assert prepared.xrdf_path.is_file()
            assert cumotion.flange_frame is None
            assert prepared.flange_frames["left"]
            assert prepared.flange_frames["right"]
        else:
            robot = RobotAssetConfig.from_mapping(config)
            assert robot.import_config.collision_approximation in {
                "convex_decomposition",
                "convex_hull",
            }
            assert robot.gravity_policy.enabled_for_component("arm") is False
            cumotion = CuMotionConfig.from_mapping(config)
            assert Path(cumotion.urdf_path).is_file()
        assert Path(cumotion.xrdf_path).is_file()
        if "robots" not in config:
            assert cumotion.flange_frame
            assert "controlled_joints" not in config
            assert "arm_joints" not in config
            assert "hand_master_joints" not in config
            assert "tcp" not in config
        assert cumotion.custom_tcp_frame is None or cumotion.custom_tcp_frame
        assert "robot_description" not in config["cumotion"]
        assert "base_urdf" not in config["cumotion"]
        assert "lula" not in config
        assert "ik" not in config


def _assert_robot_cumotion_section_contains_only_model_resources(config: dict) -> None:
    """Robot YAML 只声明 cuMotion 模型资源；算法参数由 configs/cumotion/*.yaml 提供。"""

    disallowed_keys = {
        "kinematics",
        "motion_planner",
        "position_tolerance",
        "orientation_tolerance",
        "ccd_max_iterations",
        "bfgs_max_iterations",
        "orientation_weight",
        "collision_free_ik_params",
        "collision_free_params",
        "motion_planner_config_path",
        "motion_planner_params",
        "trajectory_limits",
        "trajectory_solver_params",
    }
    cumotion = config["cumotion"]
    assert not (set(cumotion) & disallowed_keys)
    if "robots" in config:
        allowed_top_level = {
            "left",
            "right",
            "output_dir",
            "robot_name",
            "parent_link",
            "left_base_link",
            "right_base_link",
            "left_mount_joint",
            "right_mount_joint",
        }
        assert set(cumotion) <= allowed_top_level
        for side in ("left", "right"):
            assert set(cumotion[side]) <= {"xrdf_path", "urdf_path", "flange_frame"}
    else:
        assert set(cumotion) <= {
            "xrdf_path",
            "urdf_path",
            "flange_frame",
            "custom_tcp_frame",
        }


def test_dual_arm_semantic_configs_are_separate_from_robot_configs() -> None:
    for path in sorted(Path("configs/dual_arm").glob("*.yaml")):
        config = load_yaml(path)
        assert set(config) == {"dual_arm"}
        dual_arm = config["dual_arm"]
        for side in ("left", "right"):
            side_config = dual_arm[side]
            assert side_config["arm_joints"]
            assert side_config["flange_frame"]
            assert side_config["tcp_frame"]
            assert Path(side_config["combined_mjcf_path"]).is_file()
            assert "pre_pinch_hand_targets" not in side_config
            assert "closed_pinch_hand_targets" not in side_config


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
        motion_planner_config_from_profile(config).validate()


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


def test_cumotion_config_rejects_removed_flat_ik_fields() -> None:
    try:
        CuMotionConfig.from_mapping(
            {
                "cumotion": {
                    "xrdf_path": "robot.xrdf",
                    "urdf_path": "robot.urdf",
                    "flange_frame": "flange",
                    "ik_cspace_seeds": [0.0, 0.1],
                }
            }
        )
    except ValueError as exc:
        assert "removed field" in str(exc)
    else:
        raise AssertionError("CuMotionConfig accepted removed flat IK field")


def test_env_runtime_settings_apply_cli_overrides() -> None:
    settings = EnvRuntimeSettings.from_env_config(
        {
            "env": {
                "physics_frequency": 300.0,
                "render_frequency": 60.0,
                "gravity_z": -9.81,
                "add_ground": False,
            }
        },
        physics_frequency_override=240.0,
        gravity_z_override=-3.0,
    )

    assert settings.physics_frequency == 240.0
    assert settings.render_frequency == 60.0
    assert settings.gravity_z == -3.0
    assert settings.add_ground is False
    assert settings.physics_dt == 1.0 / 240.0
    assert settings.rendering_dt(gui=True) == 1.0 / 60.0
    assert settings.rendering_dt(gui=False) == settings.physics_dt


def test_env_runtime_settings_reject_invalid_env_mapping() -> None:
    try:
        EnvRuntimeSettings.from_env_config({"env": {"physics_frequency": 0.0}})
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("EnvRuntimeSettings accepted zero physics frequency")

    try:
        EnvRuntimeSettings.from_env_config({})
    except ValueError as exc:
        assert "top-level env mapping" in str(exc)
    else:
        raise AssertionError("EnvRuntimeSettings accepted missing env mapping")


def test_cumotion_profile_merge_order() -> None:
    """profile 默认值应低于 robot YAML；动作参数直接来自脚本。"""

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
    merged_robot = merged_robot_config_with_cumotion_profile(robot, profile)
    merged_ik = merged_robot["cumotion"]["kinematics"]["ik"]
    assert merged_ik["position_tolerance"] == 0.002
    assert merged_ik["orientation_weight"] == 0.1
    assert merged_robot["cumotion"]["flange_frame"] == "flange"

    backend = motion_planner_config_from_profile(profile)
    assert backend.planning_pipeline == "graph_search"
    assert backend.graph_search.generate_interpolated_path is True


def test_robot_gravity_policy_parses_grouped_mapping() -> None:
    grouped = RobotGravityPolicy.from_mapping(
        {"default": True, "arm": False, "hand": True},
        label="robot.physics.gravity",
    )

    assert grouped.enabled_for_component("default") is True
    assert grouped.enabled_for_component("arm") is False
    assert grouped.enabled_for_component("hand") is True
    assert grouped.enabled_for_name("AR5V2_L_arm_link_1") is False
    assert grouped.enabled_for_name("L6V1_L_hand_base_link") is True
    assert grouped.enabled_for_name("unclassified") is True

    all_enabled = RobotGravityPolicy.from_mapping(
        {"default": True}, label="robot.physics.gravity"
    )
    assert all_enabled.enabled_for_component("arm") is True
    assert all_enabled.enabled_for_component("hand") is True


def test_right_side_urdf_assets_exist() -> None:
    assert DEFAULT_AR5_RIGHT_URDF.is_file()
    assert DEFAULT_L6_RIGHT_URDF.is_file()


def test_workstation_static_urdf_assets_exist() -> None:
    workstation_assets = {
        DEFAULT_WORKSTATION_V1_ARMBASE_URDF: "workstationV1_armbase_frame",
        DEFAULT_WORKSTATION_V1_TABLEBASE_URDF: "workstationV1_tablebase_frame",
    }

    for asset_file, frame_name in workstation_assets.items():
        assert asset_file.is_file()
        root = ET.parse(asset_file).getroot()
        links = root.findall("link")
        assert root.get("name") == frame_name
        assert [link.get("name") for link in links] == [frame_name]
        assert root.findall("joint") == []


def test_workstation_uses_primitive_collisions() -> None:
    armbase_root = ET.parse(DEFAULT_WORKSTATION_V1_ARMBASE_URDF).getroot()
    tablebase_root = ET.parse(DEFAULT_WORKSTATION_V1_TABLEBASE_URDF).getroot()
    armbase_collisions = _workstation_collision_mapping(armbase_root)
    tablebase_collisions = _workstation_collision_mapping(tablebase_root)
    expected_names = ("table_body", "armbase_column", "armbase_top_flange")

    assert tuple(armbase_collisions) == expected_names
    assert tuple(tablebase_collisions) == expected_names

    for collisions in (armbase_collisions, tablebase_collisions):
        assert all(
            collision.find("./geometry/mesh") is None
            for collision in collisions.values()
        )
        assert (
            sum(
                collision.find("./geometry/box") is not None
                for collision in collisions.values()
            )
            == 2
        )
        assert (
            sum(
                collision.find("./geometry/cylinder") is not None
                for collision in collisions.values()
            )
            == 1
        )
        assert (
            collisions["armbase_top_flange"].find("./origin").get("rpy")
            == "1.5708 0 0"
        )

    offset = (-0.03, 0.0, 0.5)
    for name in expected_names:
        armbase_xyz = _origin_xyz(armbase_collisions[name])
        tablebase_xyz = _origin_xyz(tablebase_collisions[name])
        np.testing.assert_allclose(
            tablebase_xyz,
            tuple(value + delta for value, delta in zip(armbase_xyz, offset)),
        )


def _workstation_collision_mapping(root: ET.Element) -> dict[str, ET.Element]:
    collisions = root.findall("./link/collision")
    names = [collision.get("name") for collision in collisions]
    assert len(collisions) == 3
    assert len(set(names)) == len(names)
    return {str(name): collision for name, collision in zip(names, collisions)}


def _origin_xyz(collision: ET.Element) -> tuple[float, float, float]:
    return tuple(float(value) for value in collision.find("./origin").get("xyz").split())


def test_robot_asset_mesh_references_exist() -> None:
    asset_files = [
        *Path("assets/single_system").glob("**/*.urdf"),
        *Path("assets/single_system").glob("**/*.xml"),
        *Path("assets/combined_system").glob("**/*.xml"),
        *Path("assets/static_env_objects").glob("**/*.urdf"),
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


def test_default_rope_and_pinch_grasp_action_constants() -> None:
    rope = CapsuleRopeConfig.from_mapping(
        load_yaml("configs/objects/capsule_rope.yaml")
    )
    rope.validate()
    assert rope.asset_file().is_file()
    assert rope.prim_path == "/World/CapsuleRope"
    assert rope.root_path == "/CapsuleRope"
    assert rope.radius is not None and rope.radius > 0.0
    assert rope.twist_limit is not None and rope.twist_limit > 0.0
    left_center = endpoint_center(rope, "left")
    target = grasp_target_position(rope, endpoint="left")
    assert target[1] == left_center[1]
    assert target[2] > left_center[2]


def test_system_configs_reject_obsolete_shapes() -> None:
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


def test_robot_asset_config_parses_collision_approximation() -> None:
    config = RobotAssetConfig.from_mapping(
        {
            "robot": {
                "asset_type": "urdf",
                "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
                "prim_path": "/World/Robot",
                "import": {"collision_approximation": "convex_hull"},
            }
        }
    )

    assert config.import_config.collision_approximation == "convex_hull"


def test_import_config_rejects_removed_collision_approximation_aliases() -> None:
    try:
        RobotAssetConfig.from_mapping(
            {
                "robot": {
                    "asset_type": "urdf",
                    "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
                    "prim_path": "/World/Robot",
                    "import": {"collision_approximation": "convexHull"},
                }
            }
        )
    except ValueError as exc:
        assert "collision_approximation" in str(exc)
    else:
        raise AssertionError("RobotAssetConfig accepted removed collision alias")


def test_env_configs_provide_solver_settings() -> None:
    """所有项目内置场景都应显式提供 PhysX solver 设置。"""

    for path in sorted(Path("configs/envs").glob("*.yaml")):
        config = load_yaml(path)
        env = config.get("env", {})
        assert isinstance(env, dict)
        if "add_ground" in env:
            assert isinstance(env["add_ground"], bool)
        assert "solver" in config, f"{path} must provide solver settings"
        solver = config["solver"]
        parsed = SolverIterationConfig(
            solver_type=str(solver["type"]) if "type" in solver else None,
            arm_position_iterations=(
                int(solver["arm_position_iterations"])
                if "arm_position_iterations" in solver
                else None
            ),
            arm_velocity_iterations=(
                int(solver["arm_velocity_iterations"])
                if "arm_velocity_iterations" in solver
                else None
            ),
            hand_position_iterations=(
                int(solver["hand_position_iterations"])
                if "hand_position_iterations" in solver
                else None
            ),
            hand_velocity_iterations=(
                int(solver["hand_velocity_iterations"])
                if "hand_velocity_iterations" in solver
                else None
            ),
        )
        assert parsed.solver_type is not None or any(
            getattr(parsed, field_name) is not None
            for field_name in (
                "arm_position_iterations",
                "arm_velocity_iterations",
                "hand_position_iterations",
                "hand_velocity_iterations",
            )
        )
        if parsed.solver_type is not None:
            assert parsed.solver_type.upper() in {"PGS", "TGS"}
        for field_name in (
            "arm_position_iterations",
            "arm_velocity_iterations",
            "hand_position_iterations",
            "hand_velocity_iterations",
        ):
            value = getattr(parsed, field_name)
            assert value is None or value >= 0
        assert "apply_scope" not in solver


def test_env_scene_object_configs_reference_existing_assets() -> None:
    """env.objects 描述已有资产如何进入 stage，资产路径必须可解析。"""

    for path in sorted(Path("configs/envs").glob("*.yaml")):
        config = load_yaml(path)
        scene_objects = scene_objects_from_env_config(config)
        for scene_object in scene_objects:
            assert scene_object.asset_path.is_file()
            assert scene_object.asset_type in {"usd", "urdf"}
            assert scene_object.prim_path.startswith("/")
            assert scene_object.import_config.collision_approximation in {
                "convex_decomposition",
                "convex_hull",
            }
            assert isinstance(scene_object.physics.static, bool)
            assert len(scene_object.root_pose.xyz) == 3
            assert len(scene_object.root_pose.rpy) == 3


def test_scene_object_config_parses_root_pose() -> None:
    config = SceneObjectConfig.from_mapping(
        {
            "name": "fixture",
            "asset_type": "usd",
            "asset_path": (
                "assets/dynamic_env_objects/capsuleropeV1_default/"
                "capsuleropeV1_default.usda"
            ),
            "prim_path": "/World/Fixture",
            "root_pose": {"xyz": [1, 2, 3], "rpy": [0.1, 0.2, 0.3]},
            "import": {"collision_approximation": "convex_hull"},
            "physics": {
                "static": True,
                "material": {
                    "static_friction": 0.8,
                    "dynamic_friction": 0.6,
                    "friction_combine_mode": "average",
                },
            },
        },
        index=0,
    )

    assert config.name == "fixture"
    assert config.asset_type == "usd"
    assert config.prim_path == "/World/Fixture"
    assert config.import_config.collision_approximation == "convex_hull"
    assert config.root_pose.xyz == (1.0, 2.0, 3.0)
    assert config.root_pose.rpy == (0.1, 0.2, 0.3)
    assert config.physics.static is True
    assert config.physics.material is not None
    assert config.physics.material.static_friction == 0.8
    assert config.physics.material.dynamic_friction == 0.6
    assert config.physics.material.restitution is None
    assert config.physics.material.friction_combine_mode == "average"


def test_scene_object_config_rejects_invalid_shapes() -> None:
    for item in (
        {"asset_type": "obj", "asset_path": "x.obj", "prim_path": "/World/Object"},
        {"asset_type": "usd", "prim_path": "/World/Object"},
        {"asset_type": "usd", "asset_path": "x.usd", "prim_path": "World/Object"},
        {
            "asset_type": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "import": {"collision_approximation": "triangle_mesh"},
        },
        {
            "asset_type": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "import": {"collision_approximation": "convex_hull", "density": 1.0},
        },
        {
            "asset_type": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"static": "true"},
        },
        {
            "asset_type": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"static": True, "material": {"static_friction": -0.1}},
        },
        {
            "asset_type": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"material": {"friction_combine_mode": "unknown"}},
        },
        {
            "asset_type": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"material": {"density": 1.0}},
        },
        {
            "asset_type": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"static_friction": 0.8},
        },
    ):
        try:
            SceneObjectConfig.from_mapping(item, index=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"SceneObjectConfig accepted invalid item: {item}")


def test_solver_settings_are_optional_in_scripts() -> None:
    """外部自定义场景没写 solver 时，脚本不主动覆盖 PhysX 默认值。"""

    from linkerbot_sim.assets.solver_overrides import solver_settings

    assert solver_settings({"env": {"name": "custom_scene"}}) is None


def test_solver_settings_only_cover_explicit_fields() -> None:
    """solver 字段不做默认补齐：写哪个 PhysX 属性，就只覆盖哪个属性。"""

    from linkerbot_sim.assets.solver_overrides import (
        solver_iterations_for_prim_name,
        solver_settings,
    )

    hand_only = solver_settings(
        {
            "solver": {
                "type": "TGS",
                "hand_position_iterations": 48,
            }
        }
    )
    assert hand_only is not None
    assert hand_only.solver_type == "TGS"
    assert hand_only.arm_position_iterations is None
    assert hand_only.arm_velocity_iterations is None
    assert hand_only.hand_position_iterations == 48
    assert hand_only.hand_velocity_iterations is None
    assert solver_iterations_for_prim_name("L6V1_L_hand_base_link", hand_only) == (
        48,
        None,
        "hand",
    )
    assert solver_iterations_for_prim_name("AR5V2_L_arm_base", hand_only) is None

    arm_velocity_only = solver_settings(
        {"solver": {"arm_velocity_iterations": 6}}
    )
    assert arm_velocity_only is not None
    assert arm_velocity_only.solver_type is None
    assert solver_iterations_for_prim_name(
        "AR5V2_L_arm_link1", arm_velocity_only
    ) == (None, 6, "arm")
    assert solver_iterations_for_prim_name(
        "L6V1_L_hand_base_link", arm_velocity_only
    ) is None
