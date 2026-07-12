from __future__ import annotations

import ast
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from linkerbot_sim.assets.robot_config import (
    RobotAssetConfig,
    RobotGravityPolicy,
    RobotPhysxOverrides,
)
from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    robot_instances_from_env_config,
)
from linkerbot_sim.assets.solver_overrides import (
    SolverIterationConfig,
    robot_solver_settings,
)
from linkerbot_sim.envs.settings import EnvRuntimeSettings
from linkerbot_sim.backends.curobo.config import (
    CuroboConfig,
    CuroboIkConfig,
    CuroboMotionPlannerConfig,
)
from linkerbot_sim.backends.curobo.profile_merge import (
    merged_robot_config_with_curobo_profile,
    robot_curobo_config,
)
from linkerbot_sim.objects.runtime import runtime_objects_from_env_config
from linkerbot_sim.objects.rigid.config import (
    RigidObjectConfig,
    rigid_objects_from_env_config,
)
from linkerbot_sim.objects.config import (
    ObjectProfileConfig,
    ObjectSceneInstanceConfig,
)
from linkerbot_sim.objects.dynamic_chain.capsule_rope import CapsuleRopeConfig
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.paths import repo_path
from tools.object_assets.flexible.rope.builder import CapsuleRopeAssetConfig
from tools.object_assets.rigid.tblock.builder import TBlockAssetConfig


def _robot_asset_config(data: dict[str, object]) -> RobotAssetConfig:
    return RobotAssetConfig.from_mapping(data, prim_path="/World/Robots/test_robot")


def test_domain_layers_do_not_import_the_app_composition_root() -> None:
    domain_roots = (
        Path("src/linkerbot_sim/envs"),
        Path("src/linkerbot_sim/objects"),
        Path("src/linkerbot_sim/tiled/scene"),
    )
    violations: list[str] = []
    for root in domain_roots:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "linkerbot_sim.app"
                ):
                    violations.append(f"{path}:{node.lineno}:{node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("linkerbot_sim.app"):
                            violations.append(f"{path}:{node.lineno}:{alias.name}")
    assert violations == []


def test_default_robot_config_paths_exist() -> None:
    config = load_yaml("configs/robots/ar5v2_l6v1_l.yaml")
    env_config = load_yaml("configs/envs/scene1.yaml")
    scene_instance = robot_instances_from_env_config(env_config)[0]
    robot = _robot_asset_config(config)
    execution = RobotExecutionConfig.from_mapping(
        config,
        scene_instance=scene_instance,
    )
    assert robot.asset_path.is_file()
    assert execution.controlled_joints == ("all",)


def test_robot_configs_are_curobo_only() -> None:
    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        assert "curobo" in config
        _assert_robot_curobo_section_contains_only_model_resources(config)
        assert "robots" not in config
        robot = _robot_asset_config(config)
        assert robot.import_config.collision_approximation in {
            "convex_decomposition",
            "convex_hull",
        }
        assert robot.gravity_policy.enabled_for_component("arm") is False
        assert robot.physx_overrides.default.contact_static_friction == 0.8
        assert robot.solver_iterations is not None
        if config["robot"]["kind"] in {"arm", "arm_hand"}:
            assert robot.solver_iterations.arm_position_iterations is not None
            assert robot.solver_iterations.arm_velocity_iterations is not None
        if config["robot"]["kind"] in {"hand", "arm_hand"}:
            assert robot.solver_iterations.hand_position_iterations is not None
            assert robot.solver_iterations.hand_velocity_iterations is not None
        assert "controlled_joints" not in config
        assert "arm_joints" not in config
        assert "hand_master_joints" not in config
        assert "tcp" not in config
        assert "lula" not in config
        assert "ik" not in config
        if config["curobo"].get("enabled") is False:
            assert config["robot"]["kind"] == "hand"
            assert "robot" not in config["curobo"]
            continue
        curobo = CuroboConfig.from_mapping(config)
        assert curobo.robot.urdf_path is not None
        assert curobo.robot.urdf_path.is_file()
        assert curobo.robot.flange_frame
        assert curobo.robot.default_tcp_frame is None or curobo.robot.default_tcp_frame
        for tcp in curobo.robot.custom_tcp_frames:
            assert tcp.frame_name
            assert tcp.parent_frame
        assert "robot_description" not in config["curobo"]["robot"]
        assert "base_urdf" not in config["curobo"]["robot"]


def _assert_robot_curobo_section_contains_only_model_resources(config: dict) -> None:
    """Robot YAML 只声明 cuRobo 模型资源；算法参数由 configs/curobo/*.yaml 提供。"""

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
        "device",
    }
    curobo = config["curobo"]
    assert set(curobo) <= {"enabled", "planning_joint_group", "robot"}
    if curobo.get("enabled") is False:
        assert "robot" not in curobo
        return
    robot = curobo["robot"]
    assert not (set(robot) & disallowed_keys)
    assert set(robot) <= {
        "robot_config_path",
        "urdf_path",
        "base_link",
        "flange_frame",
        "tool_frames",
        "default_tcp_frame",
        "custom_tcps",
        "load_collision_spheres",
    }


def test_curobo_profiles_and_examples_are_valid_defaults() -> None:
    """内置 cuRobo profile/example 应只提供可合并的后端默认值。"""

    for path in sorted(Path("configs/curobo").glob("*.yaml")):
        config = load_yaml(path)
        assert "curobo" in config
        assert "motion_planning" not in config
        curobo = config["curobo"]
        assert "robot" not in curobo
        assert "kinematics" in curobo
        assert "ik" in curobo["kinematics"]
        ik_config = CuroboIkConfig.from_mapping(curobo["kinematics"]["ik"])
        assert ik_config.position_tolerance >= 0.0
        assert ik_config.orientation_tolerance >= 0.0
        assert ik_config.num_seeds > 0
        assert ik_config.max_batch_size > 0
        assert "motion_planner" in curobo
        CuroboMotionPlannerConfig.from_mapping(curobo["motion_planner"]).validate()


def test_curobo_config_parses_grouped_kinematics() -> None:
    """推荐配置结构应把 IK 参数放在 curobo.kinematics 下。"""

    config = CuroboConfig.from_mapping(
        {
            "curobo": {
                "enabled": True,
                "planning_joint_group": "arm",
                "robot": {
                    "urdf_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
                    "flange_frame": "flange",
                    "default_tcp_frame": "tool",
                },
                "kinematics": {
                    "ik": {
                        "num_seeds": 16,
                        "max_batch_size": 64,
                        "position_tolerance": 0.003,
                        "orientation_tolerance": 0.04,
                        "collision_cache": {"cuboid": 3},
                    },
                },
            }
        }
    )

    ik_config = config.ik
    assert ik_config.num_seeds == 16
    assert ik_config.max_batch_size == 64
    assert ik_config.position_tolerance == 0.003
    assert ik_config.orientation_tolerance == 0.04
    assert ik_config.collision_cache == {"cuboid": 3}


def test_curobo_config_rejects_invalid_ik_batch_sizes() -> None:
    try:
        CuroboConfig.from_mapping(
            {
                "curobo": {
                    "enabled": True,
                    "planning_joint_group": "arm",
                    "robot": {
                        "urdf_path": ("assets/single_system/arm/AR5V2_L/AR5V2_L.urdf"),
                        "default_tcp_frame": "tool",
                    },
                    "kinematics": {"ik": {"num_seeds": 0}},
                }
            }
        )
    except ValueError as exc:
        assert "num_seeds" in str(exc)
    else:
        raise AssertionError("CuroboConfig accepted invalid IK seed count")


def test_env_runtime_settings_read_env_profile_values() -> None:
    settings = EnvRuntimeSettings.from_env_config(
        {
            "env": {
                "physics_frequency": 300.0,
                "render_frequency": 60.0,
                "gravity_z": -9.81,
                "add_ground": False,
                "ground_height": 0.12,
            },
            "visuals": {
                "viewport": {
                    "enabled": True,
                    "eye": [2.0, -1.0, 1.2],
                    "target": [0.1, 0.0, 0.4],
                    "prim_path": "/OmniverseKit_Persp",
                },
                "lights": {
                    "key": {
                        "enabled": True,
                        "path": "/World/TestKeyLight",
                        "intensity": 900.0,
                        "angle": 0.25,
                        "color": [1.0, 0.95, 0.9],
                        "rotation_rpy": [0.1, 0.2, 0.3],
                    },
                    "fill": {
                        "enabled": False,
                        "path": "/World/TestFillLight",
                        "intensity": 125.0,
                        "color": [0.8, 0.9, 1.0],
                    },
                },
            },
        }
    )

    assert settings.physics_frequency == 300.0
    assert settings.render_frequency == 60.0
    assert settings.gravity_z == -9.81
    assert settings.add_ground is False
    assert settings.ground_height == 0.12
    assert settings.physics_dt == 1.0 / 300.0
    assert settings.rendering_dt(gui=True) == 1.0 / 60.0
    assert settings.rendering_dt(gui=False) == settings.physics_dt
    assert settings.visuals.viewport.eye == (2.0, -1.0, 1.2)
    assert settings.visuals.viewport.target == (0.1, 0.0, 0.4)
    assert settings.visuals.viewport.prim_path == "/OmniverseKit_Persp"
    assert settings.visuals.key_light.path == "/World/TestKeyLight"
    assert settings.visuals.key_light.intensity == 900.0
    assert settings.visuals.key_light.angle == 0.25
    assert settings.visuals.key_light.color == (1.0, 0.95, 0.9)
    assert settings.visuals.key_light.rotation_rpy == (0.1, 0.2, 0.3)
    assert settings.visuals.fill_light.enabled is False
    assert settings.visuals.fill_light.path == "/World/TestFillLight"
    assert settings.visuals.fill_light.intensity == 125.0
    assert settings.visuals.fill_light.color == (0.8, 0.9, 1.0)
    assert settings.sensors.cameras == ()


def test_env_runtime_settings_use_default_visuals() -> None:
    settings = EnvRuntimeSettings.from_env_config({"env": {}})

    assert settings.visuals.viewport.eye == (1.35, -1.65, 1.05)
    assert settings.visuals.viewport.target == (0.0, -0.1, 0.42)
    assert settings.ground_height == 0.0
    assert settings.visuals.key_light.intensity == 1200.0
    assert settings.visuals.key_light.angle == 0.5
    assert settings.visuals.fill_light.intensity == 250.0


def test_env_runtime_settings_use_visual_defaults_for_partial_mapping() -> None:
    settings = EnvRuntimeSettings.from_env_config(
        {
            "env": {},
            "visuals": {
                "viewport": {"enabled": False},
                "lights": {
                    "key": {"intensity": 900.0},
                    "fill": {"enabled": False},
                },
            },
        }
    )

    assert settings.visuals.viewport.enabled is False
    assert settings.visuals.viewport.eye == (1.35, -1.65, 1.05)
    assert settings.visuals.viewport.target == (0.0, -0.1, 0.42)
    assert settings.visuals.viewport.prim_path == "/OmniverseKit_Persp"
    assert settings.visuals.key_light.enabled is True
    assert settings.visuals.key_light.path == "/World/KeyLight"
    assert settings.visuals.key_light.intensity == 900.0
    assert settings.visuals.key_light.angle == 0.5
    assert settings.visuals.key_light.color is None
    assert settings.visuals.key_light.rotation_rpy is None
    assert settings.visuals.fill_light.enabled is False
    assert settings.visuals.fill_light.path == "/World/FillLight"
    assert settings.visuals.fill_light.intensity == 250.0
    assert settings.visuals.fill_light.color is None


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

    try:
        EnvRuntimeSettings.from_env_config(
            {
                "env": {},
                "visuals": {"viewport": {"eye": [1.0, 2.0]}},
            }
        )
    except ValueError as exc:
        assert "visuals.viewport.eye" in str(exc)
    else:
        raise AssertionError("EnvRuntimeSettings accepted invalid viewport eye")


def test_curobo_profile_merge_order() -> None:
    """profile 默认值应低于 robot YAML；动作参数直接来自脚本。"""

    profile = {
        "curobo": {
            "kinematics": {
                "ik": {
                    "position_tolerance": 0.01,
                    "num_seeds": 8,
                },
            },
            "motion_planner": {
                "num_ik_seeds": 8,
                "num_trajopt_seeds": 2,
                "max_batch_size": 128,
            },
        },
    }
    robot = {
        "curobo": {
            "enabled": True,
            "planning_joint_group": "arm",
            "robot": {
                "urdf_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
                "flange_frame": "flange",
                "default_tcp_frame": "tool",
            },
            "kinematics": {"ik": {"position_tolerance": 0.002}},
        }
    }
    merged_robot = merged_robot_config_with_curobo_profile(robot, profile)
    merged_ik = merged_robot["curobo"]["kinematics"]["ik"]
    assert merged_ik["position_tolerance"] == 0.002
    assert merged_ik["num_seeds"] == 8
    assert merged_robot["curobo"]["robot"]["flange_frame"] == "flange"

    backend = CuroboConfig.from_mapping(merged_robot).motion_planner
    assert backend.num_ik_seeds == 8
    assert backend.num_trajopt_seeds == 2


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


def test_robot_physx_overrides_parse_and_apply_grouped_mapping() -> None:
    from linkerbot_sim.assets.usd_overrides import PhysxOverrideConfig

    overrides = RobotPhysxOverrides.from_mapping(
        {
            "material": {
                "contact_static_friction": 0.7,
                "contact_dynamic_friction": 0.4,
                "contact_restitution": 0.0,
                "friction_combine_mode": "max",
            },
            "rigid_body": {"linear_damping": 0.02, "angular_damping": 0.1},
            "hand": {"rigid_body": {"angular_damping": 0.2}},
        },
        label="robot.physics.physx",
    )
    configs = overrides.apply_to_configs(
        {
            "default": PhysxOverrideConfig(joint_friction=0.25),
            "arm": PhysxOverrideConfig(joint_friction=0.5),
            "hand": PhysxOverrideConfig(joint_friction=0.75),
        }
    )

    assert configs["arm"].contact_static_friction == 0.7
    assert configs["arm"].rigid_body_linear_damping == 0.02
    assert configs["hand"].rigid_body_angular_damping == 0.2
    assert configs["hand"].joint_friction == 0.75
    assert configs["arm"].friction_combine_mode == "max"

    preserved = RobotPhysxOverrides.from_mapping(
        {"material": None}, label="robot.physics.physx"
    ).apply_to_configs({"default": PhysxOverrideConfig()})
    assert preserved["default"].contact_material_override is False

    try:
        RobotPhysxOverrides.from_mapping(
            {"material": {"friction_combine_mode": "unknown"}},
            label="robot.physics.physx",
        )
    except ValueError as exc:
        assert "friction_combine_mode" in str(exc)
    else:
        raise AssertionError("robot material accepted invalid combine mode")


def test_robot_solver_iterations_parse_grouped_mapping() -> None:
    config = robot_solver_settings(
        {
            "arm": {"velocity_iterations": 6},
            "hand": {"position_iterations": 48},
        },
        label="robot.physics.solver",
    )

    assert config is not None
    assert config.solver_type is None
    assert config.arm_position_iterations is None
    assert config.arm_velocity_iterations == 6
    assert config.hand_position_iterations == 48
    assert config.hand_velocity_iterations is None


def test_right_side_urdf_assets_exist() -> None:
    arm_urdf = robot_curobo_config(
        load_profile_yaml("robot", "ar5v2_l6v1_r")
    ).robot.urdf_path
    assert arm_urdf is not None and arm_urdf.is_file()

    hand_mjcf = _robot_asset_config(load_profile_yaml("robot", "l6v1_r")).asset_path
    assert hand_mjcf.with_suffix(".urdf").is_file()


def test_workstation_static_urdf_assets_exist() -> None:
    armbase_urdf = repo_path(
        ObjectProfileConfig.from_profile("workstation_armbase").asset_path
    )
    tablebase_urdf = repo_path(
        ObjectProfileConfig.from_profile("workstation_tablebase").asset_path
    )
    workstation_assets = {
        armbase_urdf: "workstationV1_armbase_frame",
        tablebase_urdf: "workstationV1_tablebase_frame",
    }

    for asset_file, frame_name in workstation_assets.items():
        assert asset_file.is_file()
        root = ET.parse(asset_file).getroot()
        links = root.findall("link")
        assert root.get("name") == frame_name
        assert [link.get("name") for link in links] == [frame_name]
        assert root.findall("joint") == []


def test_workstation_uses_primitive_collisions() -> None:
    armbase_path = repo_path(
        ObjectProfileConfig.from_profile("workstation_armbase").asset_path
    )
    tablebase_path = repo_path(
        ObjectProfileConfig.from_profile("workstation_tablebase").asset_path
    )
    armbase_root = ET.parse(armbase_path).getroot()
    tablebase_root = ET.parse(tablebase_path).getroot()
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
            collisions["armbase_top_flange"].find("./origin").get("rpy") == "1.5708 0 0"
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
    return tuple(
        float(value) for value in collision.find("./origin").get("xyz").split()
    )


def test_robot_asset_mesh_references_exist() -> None:
    asset_files = [
        *Path("assets/single_system").glob("**/*.urdf"),
        *Path("assets/single_system").glob("**/*.xml"),
        *Path("assets/combined_system").glob("**/*.xml"),
        *Path("assets/rigid_env_objects").glob("**/*.urdf"),
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


def test_mjcf_fingertip_frames_are_non_physical_sites() -> None:
    asset_files = [
        *Path("assets/single_system/hand").glob("L6V1_*/L6V1_*.xml"),
        *Path("assets/combined_system").glob("AR5V2_L6V1_*/AR5V2_L6V1_*.xml"),
    ]
    assert len(asset_files) == 4

    for asset_file in asset_files:
        root = ET.parse(asset_file).getroot()
        tip_sites = {
            str(site.get("name"))
            for site in root.iter("site")
            if str(site.get("name", "")).endswith("_tip")
        }
        tip_bodies = {
            str(body.get("name"))
            for body in root.iter("body")
            if str(body.get("name", "")).endswith("_tip")
        }
        assert len(tip_sites) == 5, asset_file
        assert tip_bodies == set(), asset_file


def test_mjcf_default_geometries_have_explicit_types() -> None:
    asset_files = [
        *Path("assets/single_system").glob("**/*.xml"),
        *Path("assets/combined_system").glob("**/*.xml"),
    ]
    assert len(asset_files) == 6

    for asset_file in asset_files:
        root = ET.parse(asset_file).getroot()
        default_geometries = root.findall("./default/geom")
        assert default_geometries, asset_file
        assert all(geom.get("type") is not None for geom in default_geometries), (
            asset_file
        )


def test_default_rope_and_pinch_grasp_action_constants() -> None:
    rope = CapsuleRopeConfig.from_mapping(
        load_yaml("configs/objects/capsule_rope.yaml")
    )
    rope.validate()
    assert rope.asset_file().is_file()
    assert rope.prim_path == "/World/CapsuleRope"
    assert rope.root_path == "/CapsuleRope"
    assert rope.physics.material is not None
    assert rope.physics.material.static_friction == 0.7
    assert rope.physics.material.dynamic_friction == 0.5
    assert rope.physics.solver_position_iterations == 48


def test_capsule_rope_runtime_config_does_not_contain_generation_fields() -> None:
    config = load_yaml("configs/objects/capsule_rope.yaml")
    assert "rope" not in config
    object_section = config["object"]
    generation_fields = {
        "segments",
        "length",
        "radius",
        "center",
        "shape",
        "total_mass",
        "endpoint_box_mass",
        "endpoint_box_size",
        "endpoint_linear_damping",
        "endpoint_angular_damping",
        "segment_linear_damping",
        "segment_angular_damping",
        "bend_limit",
        "bend_stiffness",
        "bend_damping",
        "lock_twist",
        "twist_limit",
        "twist_stiffness",
        "twist_damping",
        "disable_adjacent_collisions",
        "endpoint_color",
        "rope_color",
        "env_static_friction",
        "env_dynamic_friction",
        "env_restitution",
    }
    assert set(object_section).isdisjoint(generation_fields)


def test_capsule_rope_asset_generation_config_lives_under_tools() -> None:
    asset = CapsuleRopeAssetConfig.from_mapping(
        load_yaml("tools/object_assets/flexible/rope/config.yaml")
    )
    asset.validate()
    assert asset.segments == 12
    assert asset.length == 0.75
    assert asset.radius is not None and asset.radius > 0.0
    assert asset.twist_limit is not None and asset.twist_limit > 0.0


def test_tblock_asset_generation_config_lives_under_tools() -> None:
    asset = TBlockAssetConfig.from_mapping(
        load_yaml("tools/object_assets/rigid/tblock/config.yaml")
    )
    asset.validate()
    assert asset.asset_path == (
        "assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda"
    )
    assert asset.root_path == "/TBlock"
    assert asset.stem_size == (0.04, 0.08, 0.16)
    assert asset.cap_size == (0.04, 0.2, 0.06)


def test_scene1_places_rope_from_env_root_pose() -> None:
    env_config = load_yaml("configs/envs/scene1.yaml")
    runtime_objects = runtime_objects_from_env_config(env_config)
    rope_object = next(
        item for item in runtime_objects if item.runtime_handle == "rope"
    )

    assert rope_object.prim_path == "/World/CapsuleRope"
    assert rope_object.root_pose.xyz == (0.1, -0.55, -0.4)


def test_system_configs_reject_unknown_shapes() -> None:
    try:
        _robot_asset_config({"asset_path": "assets/example.xml"})
    except ValueError:
        pass
    else:
        raise AssertionError(
            "RobotAssetConfig accepted config without top-level robot section"
        )

    try:
        CapsuleRopeConfig.from_mapping(
            {"object": {"asset_path": "x.usd"}, "rope": {"segments": 18}}
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "CapsuleRopeConfig accepted generation fields in runtime config"
        )


def test_robot_asset_config_parses_collision_approximation() -> None:
    config = _robot_asset_config(
        {
            "robot": {
                "asset_type": "urdf",
                "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
                "import": {"collision_approximation": "convex_hull"},
            }
        }
    )

    assert config.import_config.collision_approximation == "convex_hull"


def test_robot_asset_config_parses_self_collision() -> None:
    config = _robot_asset_config(
        {
            "robot": {
                "asset_type": "mjcf",
                "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.xml",
                "import": {
                    "collision_approximation": "convex_decomposition",
                    "self_collision": True,
                },
            }
        }
    )

    assert config.import_config.self_collision is True


def test_robot_asset_config_defaults_self_collision_to_false() -> None:
    config = _robot_asset_config(
        {
            "robot": {
                "asset_type": "mjcf",
                "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.xml",
                "import": {"collision_approximation": "convex_decomposition"},
            }
        }
    )

    assert config.import_config.self_collision is False


def test_robot_asset_config_parses_named_importer_settings() -> None:
    config = _robot_asset_config(
        {
            "robot": {
                "asset_type": "urdf",
                "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
                "import": {
                    "fix_base": False,
                    "merge_fixed_joints": False,
                    "collision_from_visuals": True,
                    "import_inertia_tensor": False,
                },
            }
        }
    )

    assert config.import_config.fix_base is False
    assert config.import_config.merge_fixed_joints is False
    assert config.import_config.collision_from_visuals is True
    assert config.import_config.import_inertia_tensor is False
    assert config.import_config.import_sites is True

    mjcf = _robot_asset_config(
        {
            "robot": {
                "asset_type": "mjcf",
                "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.xml",
                "import": {"merge_fixed_joints": True, "import_sites": False},
            }
        }
    )
    assert mjcf.import_config.merge_fixed_joints is True
    assert mjcf.import_config.import_sites is False


def test_import_config_rejects_fields_unsupported_by_asset_format() -> None:
    for asset_type, field in (
        ("mjcf", "collision_from_visuals"),
        ("urdf", "import_sites"),
    ):
        asset_path = (
            "assets/single_system/arm/AR5V2_L/AR5V2_L.xml"
            if asset_type == "mjcf"
            else "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf"
        )
        try:
            _robot_asset_config(
                {
                    "robot": {
                        "asset_type": asset_type,
                        "asset_path": asset_path,
                        "import": {field: False},
                    }
                }
            )
        except ValueError as exc:
            assert f"unsupported keys: {field}" in str(exc)
        else:
            raise AssertionError(f"{asset_type} accepted unsupported field {field}")


def test_robot_asset_config_rejects_non_boolean_importer_settings() -> None:
    for field in (
        "fix_base",
        "merge_fixed_joints",
        "collision_from_visuals",
        "import_inertia_tensor",
    ):
        try:
            _robot_asset_config(
                {
                    "robot": {
                        "asset_type": "urdf",
                        "asset_path": ("assets/single_system/arm/AR5V2_L/AR5V2_L.urdf"),
                        "import": {field: "false"},
                    }
                }
            )
        except ValueError as exc:
            assert f"robot.import.{field}" in str(exc)
        else:
            raise AssertionError(f"accepted non-boolean robot.import.{field}")

    try:
        _robot_asset_config(
            {
                "robot": {
                    "asset_type": "urdf",
                    "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
                    "import": {"import_sites": False},
                }
            }
        )
    except ValueError as exc:
        assert "unsupported keys: import_sites" in str(exc)
    else:
        raise AssertionError("URDF robot accepted MJCF-only import_sites")


def test_robot_asset_config_rejects_non_bool_self_collision() -> None:
    try:
        _robot_asset_config(
            {
                "robot": {
                    "asset_type": "mjcf",
                    "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.xml",
                    "import": {"self_collision": "true"},
                }
            }
        )
    except ValueError as exc:
        assert "self_collision" in str(exc)
    else:
        raise AssertionError("RobotAssetConfig accepted non-bool self_collision")


def test_import_config_rejects_unknown_collision_approximation() -> None:
    try:
        _robot_asset_config(
            {
                "robot": {
                    "asset_type": "urdf",
                    "asset_path": "assets/single_system/arm/AR5V2_L/AR5V2_L.urdf",
                    "import": {"collision_approximation": "convexHull"},
                }
            }
        )
    except ValueError as exc:
        assert "collision_approximation" in str(exc)
    else:
        raise AssertionError(
            "RobotAssetConfig accepted unknown collision approximation"
        )


def test_env_configs_provide_solver_settings() -> None:
    """所有项目内置场景都应显式提供 scene 级 PhysX solver type。"""

    for path in sorted(Path("configs/envs").glob("*.yaml")):
        config = load_yaml(path)
        env = config.get("env", {})
        assert isinstance(env, dict)
        if "add_ground" in env:
            assert isinstance(env["add_ground"], bool)
        if "ground_height" in env:
            assert isinstance(env["ground_height"], (int, float))
            assert not isinstance(env["ground_height"], bool)
        assert "solver" in config, f"{path} must provide solver settings"
        solver = config["solver"]
        parsed = SolverIterationConfig(solver_type=str(solver["type"]))
        assert parsed.solver_type is not None
        assert parsed.solver_type.upper() in {"PGS", "TGS"}
        assert set(solver) == {"type"}


def test_robot_configs_provide_solver_iteration_settings() -> None:
    """机器人刚体 solver iteration 应写在 robot.physics.solver。"""

    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        solver = _robot_asset_config(config).solver_iterations
        assert solver is not None, f"{path} must provide robot.physics.solver"
        assert solver.solver_type is None
        for field_name in (
            "arm_position_iterations",
            "arm_velocity_iterations",
            "hand_position_iterations",
            "hand_velocity_iterations",
        ):
            value = getattr(solver, field_name)
            assert value is None or value >= 0
        assert any(
            getattr(solver, field_name) is not None
            for field_name in (
                "arm_position_iterations",
                "arm_velocity_iterations",
                "hand_position_iterations",
                "hand_velocity_iterations",
            )
        )


def test_env_profiles_define_robot_scene_instances() -> None:
    """env.robots 选择 robot profile，并保存 scene 中的安装位姿。"""

    allowed_keys = {
        "label",
        "robot_profile",
        "root_pose",
        "prim_path",
        "controller_profile",
    }
    for path in sorted(Path("configs/envs").glob("*.yaml")):
        config = load_yaml(path)
        settings = EnvRuntimeSettings.from_env_config(config)
        assert settings.visuals.viewport.prim_path.startswith("/")
        assert settings.visuals.key_light.path.startswith("/")
        assert settings.visuals.fill_light.path.startswith("/")
        robots = config.get("robots")
        assert isinstance(robots, list), f"{path} must contain robots list"
        assert robots
        instances = robot_instances_from_env_config(config)
        assert [instance.robot_id for instance in instances] == list(
            range(len(instances))
        )
        for raw, instance in zip(robots, instances, strict=True):
            assert set(raw) <= allowed_keys
            assert raw.get("label") == instance.label
            assert instance.robot_profile
            assert len(instance.root_pose.xyz) == 3
            assert len(instance.root_pose.rpy) == 3
            assert Path(f"configs/robots/{instance.robot_profile}.yaml").is_file()


def test_env_scene_robot_profiles_match_runtime_shapes() -> None:
    for env_name in ("scene1", "scene2"):
        env_config = load_yaml(f"configs/envs/{env_name}.yaml")
        instances = robot_instances_from_env_config(env_config)
        for instance in instances:
            profile = load_yaml(f"configs/robots/{instance.robot_profile}.yaml")
            assert "robot" in profile
            assert "robots" not in profile


def test_robot_profiles_do_not_own_scene_root_pose() -> None:
    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        assert "root_pose" not in config
        assert "robots" not in config


def test_env_profiles_do_not_inline_robot_solver_iterations() -> None:
    for path in sorted(Path("configs/envs").glob("*.yaml")):
        solver = load_yaml(path).get("solver", {})
        assert set(solver).isdisjoint(
            {
                "arm_position_iterations",
                "arm_velocity_iterations",
                "hand_position_iterations",
                "hand_velocity_iterations",
            }
        )


def test_robot_profiles_do_not_inline_scene_solver_type() -> None:
    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        robot = config["robot"]
        solver = robot.get("physics", {}).get("solver", {})
        assert "type" not in solver
        assert set(solver) <= {"arm", "hand"}
        for component in ("arm", "hand"):
            if component in solver:
                assert set(solver[component]) <= {
                    "position_iterations",
                    "velocity_iterations",
                }


def test_env_rigid_object_configs_reference_existing_assets() -> None:
    """env.objects 经 object_profile 合并后必须能解析到已有资产。"""

    for path in sorted(Path("configs/envs").glob("*.yaml")):
        config = load_yaml(path)
        rigid_objects = rigid_objects_from_env_config(config)
        for rigid_object in rigid_objects:
            assert rigid_object.asset_path.is_file()
            assert rigid_object.asset_type in {"usd", "urdf"}
            assert rigid_object.prim_path.startswith("/")
            assert rigid_object.import_config.collision_approximation in {
                "convex_decomposition",
                "convex_hull",
            }
            assert isinstance(rigid_object.physics.static, bool)
            assert len(rigid_object.root_pose.xyz) == 3
            assert len(rigid_object.root_pose.rpy) == 3


def test_env_profiles_do_not_inline_object_asset_paths() -> None:
    allowed_object_instance_keys = {
        "name",
        "object_profile",
        "runtime_handle",
        "prim_path",
        "root_pose",
    }
    disallowed_object_profile_keys = {
        "kind",
        "source",
        "asset_path",
        "root_path",
        "urdf_drive_type",
        "import",
        "physics",
        "planning_collision",
    }
    for path in sorted(Path("configs/envs").glob("*.yaml")):
        config = load_yaml(path)
        for index, item in enumerate(config.get("objects", ()) or ()):
            assert set(item) <= allowed_object_instance_keys
            assert set(item).isdisjoint(disallowed_object_profile_keys)
            assert "object_profile" in item
            assert "prim_path" in item
            assert "root_pose" in item
            instance = ObjectSceneInstanceConfig.from_mapping(item, index=index)
            assert instance.effective_prim_path == item["prim_path"]


def test_object_profiles_define_all_object_runtime_properties() -> None:
    for path in sorted(Path("configs/objects").glob("*.yaml")):
        profile = ObjectProfileConfig.from_mapping(
            load_yaml(path), profile_name=path.stem
        )
        assert profile.kind in {"rigid", "dynamic_chain"}
        assert profile.source in {"usd", "urdf"}
        assert profile.asset_path
        assert profile.root_path is None or profile.root_path.startswith("/")
        assert profile.raw is not None
        assert "prim_path" not in profile.raw["object"]


def test_env_object_scene_instances_reject_object_profile_properties() -> None:
    for key, value in (
        ("kind", "rigid"),
        ("source", "urdf"),
        ("asset_path", "assets/example.urdf"),
        ("import", {}),
        ("physics", {}),
        ("planning_collision", {"shape": "cuboid", "size": [1, 1, 1]}),
    ):
        try:
            ObjectSceneInstanceConfig.from_mapping(
                {
                    "name": "object",
                    "object_profile": "workstation_armbase",
                    "root_pose": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
                    key: value,
                },
                index=0,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Scene instance accepted object profile key: {key}")

    try:
        ObjectSceneInstanceConfig.from_mapping(
            {"name": "object", "object_profile": "workstation_armbase"},
            index=0,
        )
    except ValueError as exc:
        assert "root_pose" in str(exc)
    else:
        raise AssertionError("Scene instance accepted missing root_pose")


def test_rigid_object_config_parses_planning_collision() -> None:
    config = RigidObjectConfig.from_mapping(
        {
            "name": "block",
            "source": "usd",
            "asset_path": "assets/rigid_env_objects/TblockV1_default/TblockV1_default.usda",
            "prim_path": "/World/Block",
            "planning_collision": {
                "shape": "cuboid",
                "size": [0.1, 0.2, 0.3],
                "xyz": [0.0, 0.0, 0.15],
                "padding": 0.01,
            },
        },
        index=0,
    )

    assert config.planning_collision is not None
    assert config.planning_collision.shape == "cuboid"
    assert config.planning_collision.size == (0.1, 0.2, 0.3)
    assert config.planning_collision.xyz == (0.0, 0.0, 0.15)
    assert config.planning_collision.padding == 0.01


def test_rigid_object_config_parses_root_pose() -> None:
    config = RigidObjectConfig.from_mapping(
        {
            "name": "fixture",
            "source": "usd",
            "asset_path": (
                "assets/flexible_env_objects/capsuleropeV1_default/"
                "capsuleropeV1_default.usda"
            ),
            "prim_path": "/World/Fixture",
            "root_pose": {"xyz": [1, 2, 3], "rpy": [0.1, 0.2, 0.3]},
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
    assert config.import_config.collision_approximation == "convex_decomposition"
    assert config.root_pose.xyz == (1.0, 2.0, 3.0)
    assert config.root_pose.rpy == (0.1, 0.2, 0.3)
    assert config.physics.static is True
    assert config.physics.material is not None
    assert config.physics.material.static_friction == 0.8
    assert config.physics.material.dynamic_friction == 0.6
    assert config.physics.material.restitution is None
    assert config.physics.material.friction_combine_mode == "average"


def test_usd_rigid_object_rejects_importer_fields() -> None:
    for import_value in ({"collision_approximation": "convex_hull"}, None):
        try:
            RigidObjectConfig.from_mapping(
                {
                    "name": "fixture",
                    "source": "usd",
                    "asset_path": "fixture.usd",
                    "prim_path": "/World/Fixture",
                    "import": import_value,
                },
                index=0,
            )
        except ValueError as exc:
            assert "not supported for USD assets" in str(exc)
        else:
            raise AssertionError("USD object accepted dead importer fields")


def test_rigid_object_config_rejects_invalid_shapes() -> None:
    for item in (
        {
            "asset_type": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
        },
        {"source": "obj", "asset_path": "x.obj", "prim_path": "/World/Object"},
        {"source": "usd", "prim_path": "/World/Object"},
        {"source": "usd", "asset_path": "x.usd", "prim_path": "World/Object"},
        {
            "source": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "import": {"collision_approximation": "triangle_mesh"},
        },
        {
            "source": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "import": {"collision_approximation": "convex_hull", "density": 1.0},
        },
        {
            "source": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "import": {"self_collision": True},
        },
        {
            "source": "urdf",
            "asset_path": "x.urdf",
            "prim_path": "/World/Object",
            "import": {"fix_base": True},
            "physics": {"static": False},
        },
        {
            "source": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"static": "true"},
        },
        {
            "source": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"static": True, "material": {"static_friction": -0.1}},
        },
        {
            "source": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"material": {"friction_combine_mode": "unknown"}},
        },
        {
            "source": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"material": {"density": 1.0}},
        },
        {
            "source": "usd",
            "asset_path": "x.usd",
            "prim_path": "/World/Object",
            "physics": {"static_friction": 0.8},
        },
    ):
        try:
            RigidObjectConfig.from_mapping(item, index=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"RigidObjectConfig accepted invalid item: {item}")


def test_solver_settings_are_optional_in_scripts() -> None:
    """外部自定义场景没写 solver 时，脚本不主动覆盖 PhysX 默认值。"""

    from linkerbot_sim.assets.solver_overrides import scene_solver_settings

    assert scene_solver_settings({"env": {"name": "custom_scene"}}) is None


def test_solver_settings_keep_scene_and_robot_layers_separate() -> None:
    """env solver 只管 scene type；robot solver 只管刚体 iteration。"""

    from linkerbot_sim.assets.solver_overrides import (
        merge_solver_configs,
        scene_solver_settings,
        solver_iterations_for_prim_name,
    )

    scene_only = scene_solver_settings({"solver": {"type": "TGS"}})
    hand_only = robot_solver_settings(
        {"hand": {"position_iterations": 48}},
        label="robot.physics.solver",
    )
    merged = merge_solver_configs(scene_only, hand_only)
    assert scene_only is not None
    assert hand_only is not None
    assert merged is not None
    assert merged.solver_type == "TGS"
    assert merged.arm_position_iterations is None
    assert merged.arm_velocity_iterations is None
    assert merged.hand_position_iterations == 48
    assert merged.hand_velocity_iterations is None
    assert solver_iterations_for_prim_name("L6V1_L_hand_base_link", merged) == (
        48,
        None,
        "hand",
    )
    assert solver_iterations_for_prim_name("AR5V2_L_arm_base", merged) is None

    arm_velocity_only = robot_solver_settings(
        {"arm": {"velocity_iterations": 6}},
        label="robot.physics.solver",
    )
    assert arm_velocity_only is not None
    assert arm_velocity_only.solver_type is None
    assert solver_iterations_for_prim_name("AR5V2_L_arm_link1", arm_velocity_only) == (
        None,
        6,
        "arm",
    )
    assert (
        solver_iterations_for_prim_name("L6V1_L_hand_base_link", arm_velocity_only)
        is None
    )


def test_nonstandard_rigid_body_groups_drive_gravity_and_solver_selection() -> None:
    from linkerbot_sim.assets.solver_overrides import solver_iterations_for_prim_name
    from linkerbot_sim.robots.classification import RobotComponentMapping

    mapping = RobotComponentMapping.from_profile(
        {"rigid_body_groups": {"arm": ["body_a"], "hand": ["body_b"]}}
    )
    gravity = RobotGravityPolicy.from_mapping(
        {"default": False, "arm": True, "hand": False},
        label="robot.physics.gravity",
    )
    solver = robot_solver_settings(
        {
            "arm": {"position_iterations": 17},
            "hand": {"velocity_iterations": 9},
        },
        label="robot.physics.solver",
    )

    assert solver is not None
    assert gravity.enabled_for_component(mapping.rigid_body_component("body_a"))
    assert not gravity.enabled_for_component(mapping.rigid_body_component("body_b"))
    assert solver_iterations_for_prim_name(
        "body_a", solver, component_mapping=mapping
    ) == (17, None, "arm")
    assert solver_iterations_for_prim_name(
        "body_b", solver, component_mapping=mapping
    ) == (None, 9, "hand")


def test_env_solver_rejects_robot_iteration_fields() -> None:
    from linkerbot_sim.assets.solver_overrides import scene_solver_settings

    try:
        scene_solver_settings(
            {"solver": {"type": "TGS", "hand_position_iterations": 48}}
        )
    except ValueError as exc:
        assert "robot.physics.solver" in str(exc)
    else:
        raise AssertionError("env solver accepted robot iteration field")


def test_robot_solver_rejects_scene_type_field() -> None:
    try:
        robot_solver_settings({"type": "TGS"}, label="robot.physics.solver")
    except ValueError as exc:
        assert "env solver.type" in str(exc)
    else:
        raise AssertionError("robot solver accepted scene type field")


def test_robot_solver_rejects_negative_iterations() -> None:
    try:
        robot_solver_settings(
            {"arm": {"position_iterations": -1}},
            label="robot.physics.solver",
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("robot solver accepted negative iteration count")
