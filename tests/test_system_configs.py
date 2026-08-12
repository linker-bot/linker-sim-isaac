from __future__ import annotations

import ast
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from linkerbot_sim.assets.robot_config import RobotAssetConfig
from linkerbot_sim.configuration.robots import (
    RobotGravityPolicy,
    RobotProfileSettings,
    RobotPhysxOverrides,
)
from linkerbot_sim.assets.solver_overrides import (
    robot_solver_settings,
)
from linkerbot_sim.backends.curobo.profile_merge import (
    curobo_config_from_profiles,
)
from linkerbot_sim.configuration.curobo import CuroboProfileSettings
from linkerbot_sim.configuration.objects import (
    DynamicChainObjectProfileConfig,
    ObjectProfileConfig,
    RigidObjectPhysicsConfig,
    RigidObjectProfileConfig,
    object_profile_from_mapping,
)
from linkerbot_sim.configuration.scenes import ObjectInstanceSettings
from linkerbot_sim.utils.config import load_yaml
from linkerbot_sim.utils.paths import repo_path
from tools.object_assets.flexible.rope.builder import CapsuleRopeAssetConfig
from tools.object_assets.rigid.tblock.builder import TBlockAssetConfig


def _robot_asset_config(data: dict[str, object]) -> RobotAssetConfig:
    robot = data.get("robot")
    if isinstance(robot, dict) and not {"curobo", "joint_groups"} <= set(data):
        kind = str(robot.get("kind", "arm"))
        complete_profile: dict[str, object] = {
            "robot": {"kind": kind, "name": "test_robot", **robot},
            "curobo": {"enabled": False},
            "joint_groups": {
                "arm": [] if kind == "hand" else ["test_arm_joint"],
                "hand": [] if kind == "arm" else ["test_hand_joint"],
            },
        }
    else:
        complete_profile = data
    return RobotAssetConfig.from_profile(
        RobotProfileSettings.from_mapping(complete_profile),
        prim_path="/World/Robots/test_robot",
    )


def load_robot_profile_by_name(name: str) -> RobotProfileSettings:
    path = Path("configs/robots") / f"{name}.yaml"
    return RobotProfileSettings.from_mapping(load_yaml(path), source=str(path))


def _object_profile(name: str) -> ObjectProfileConfig:
    path = Path("configs/objects") / f"{name}.yaml"
    return object_profile_from_mapping(
        load_yaml(path), profile_name=name, source=str(path)
    )


def test_domain_layers_do_not_import_the_app_composition_root() -> None:
    domain_roots = (
        Path("src/linkerbot_sim/assets"),
        Path("src/linkerbot_sim/objects"),
        Path("src/linkerbot_sim/robots"),
        Path("src/linkerbot_sim/planning"),
        Path("src/linkerbot_sim/trajectories"),
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


def test_robot_configs_are_curobo_only() -> None:
    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        profile = RobotProfileSettings.from_mapping(config, source=str(path))
        assert "curobo" in config
        _assert_robot_curobo_section_contains_only_model_resources(config)
        assert "robots" not in config
        robot = _robot_asset_config(config)
        assert robot.import_config.collision_approximation in {
            "convex_decomposition",
            "convex_hull",
        }
        assert robot.gravity_policy.enabled_for_component("arm") is False
        assert robot.contact_material is not None
        assert robot.contact_material.contact_static_friction == 0.8
        assert robot.physx.solver_iterations is not None
        if config["robot"]["kind"] in {"arm", "arm_hand"}:
            assert robot.physx.solver_iterations.arm_position_iterations is not None
            assert robot.physx.solver_iterations.arm_velocity_iterations is not None
        if config["robot"]["kind"] in {"hand", "arm_hand"}:
            assert robot.physx.solver_iterations.hand_position_iterations is not None
            assert robot.physx.solver_iterations.hand_velocity_iterations is not None
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
        curobo = curobo_config_from_profiles(profile, cuda_device=0)
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
    """Robot YAML 只声明 cuRobo 模型资源；算法参数由独立算法 profile 提供。"""

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


def test_curobo_mode_profiles_are_strict_device_free_numerical_settings() -> None:
    """mode profile 只拥有 cuRobo 数值设置，CUDA 编号仍由 mode root 唯一提供。"""

    expected_planner = {
        "kaleidoscope_batch_ik": False,
        "mirror": True,
    }
    paths = sorted(Path("configs/curobo").glob("*.yaml"))
    assert {path.stem for path in paths} == set(expected_planner)
    for path in paths:
        config = load_yaml(path)
        assert set(config) == {"curobo"}
        assert "compute" not in config
        assert "cuda_device" not in repr(config)
        curobo = config["curobo"]
        assert "robot" not in curobo
        assert "device" not in curobo
        assert "task_bundle" not in curobo
        settings = CuroboProfileSettings.from_mapping(curobo)
        assert settings.kinematics.max_batch_size > 0
        assert settings.kinematics.seed_count > 0
        assert settings.kinematics.collision_cache is None
        assert (settings.motion_planner is not None) is expected_planner[path.stem]


def test_curobo_typed_profiles_compose_without_mapping_round_trip() -> None:
    """robot 模型、算法 profile 与根设备直接投影为后端配置。"""

    profile = CuroboProfileSettings.from_mapping(
        {
            "kinematics": {
                "max_batch_size": 8,
                "seed_count": 8,
                "collision_check": False,
                "use_cuda_graph": False,
            },
            "motion_planner": {
                "warmup": False,
                "use_cuda_graph": False,
                "ik_seed_count": 8,
                "trajectory_seed_count": 2,
                "collision_check": False,
                "collision_cache": {"cuboid": 0, "mesh": 0},
            },
        }
    )
    config = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_l"),
        curobo_settings=profile,
        cuda_device=4,
    )

    assert config.device.device == "cuda:4"
    assert config.robot.flange_frame == "AR5V2_L_arm_flan_link"
    assert config.ik.position_tolerance == 0.002
    assert config.ik.num_seeds == 8
    assert config.motion_planner.num_ik_seeds == 8
    assert config.motion_planner.num_trajopt_seeds == 2


def test_curobo_runtime_probe_has_no_implicit_cuda_zero() -> None:
    """诊断脚本的设备相关调用必须消费解析后的索引，而不是默认 GPU 0。"""

    script_path = Path("scripts/check_curobo_runtime.py")
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    assert not any(
        isinstance(node, ast.Constant) and node.value == "cuda:0"
        for node in ast.walk(tree)
    )

    indexed_calls = {
        "get_device_capability",
        "get_device_name",
        "memory_allocated",
        "memory_reserved",
        "set_device",
        "synchronize",
        "synchronize_device",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in indexed_calls:
            continue
        if not node.args or (
            isinstance(node.args[0], ast.Constant) and node.args[0].value == 0
        ):
            violations.append(f"{node.func.attr}:{node.lineno}")
    assert violations == []
    assert '"--cuda-device"' in source
    assert "default=None" in source


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


def test_robot_shared_material_and_physx_overrides_compose_in_order() -> None:
    from linkerbot_sim.assets.usd_overrides import RobotUsdOverrideConfig
    from linkerbot_sim.configuration.robots import RobotContactMaterialSettings

    material = RobotContactMaterialSettings.from_mapping(
        {
            "contact_static_friction": 0.7,
            "contact_dynamic_friction": 0.4,
            "contact_restitution": 0.0,
        },
        label="robot.physics.material",
    )
    overrides = RobotPhysxOverrides.from_mapping(
        {
            "material": {"friction_combine_mode": "max"},
            "rigid_body": {"linear_damping": 0.02, "angular_damping": 0.1},
            "joint": {"friction": 0.25, "follower_friction": 0.2},
            "hand": {
                "rigid_body": {"angular_damping": 0.2},
                "joint": {"friction": 0.75},
            },
        },
        label="robot.physics.physx",
    )
    assert material is not None
    configs = material.apply_to_configs(
        {
            "default": RobotUsdOverrideConfig(),
            "arm": RobotUsdOverrideConfig(),
            "hand": RobotUsdOverrideConfig(),
        }
    )
    configs = overrides.apply_to_configs(configs)

    assert configs["arm"].contact_static_friction == 0.7
    assert configs["arm"].contact_dynamic_friction == 0.4
    assert configs["arm"].contact_material_override is True
    assert configs["arm"].rigid_body_linear_damping == 0.02
    assert configs["hand"].rigid_body_angular_damping == 0.2
    assert configs["hand"].joint_friction == 0.75
    assert configs["hand"].follower_joint_friction == 0.2
    assert configs["arm"].friction_combine_mode == "max"

    preserved = RobotPhysxOverrides.from_mapping(
        {"material": None}, label="robot.physics.physx"
    ).apply_to_configs({"default": RobotUsdOverrideConfig()})
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
        label="robot.physics.physx.solver",
    )

    assert config is not None
    assert config.solver_type is None
    assert config.arm_position_iterations is None
    assert config.arm_velocity_iterations == 6
    assert config.hand_position_iterations == 48
    assert config.hand_velocity_iterations is None


def test_right_side_urdf_assets_exist() -> None:
    arm_urdf = curobo_config_from_profiles(
        load_robot_profile_by_name("ar5v2_l6v1_r"),
        cuda_device=0,
    ).robot.urdf_path
    assert arm_urdf is not None and arm_urdf.is_file()

    hand_mjcf = RobotAssetConfig.from_profile(
        load_robot_profile_by_name("l6v1_r"),
        prim_path="/World/Robots/test_hand",
    ).asset_path
    assert hand_mjcf.with_suffix(".urdf").is_file()


def test_workstation_static_urdf_assets_exist() -> None:
    armbase_urdf = repo_path(_object_profile("workstation_armbase").asset_path)
    tablebase_urdf = repo_path(_object_profile("workstation_tablebase").asset_path)
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
    armbase_path = repo_path(_object_profile("workstation_armbase").asset_path)
    tablebase_path = repo_path(_object_profile("workstation_tablebase").asset_path)
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


def test_industrial_warehouse_separates_visual_floor_and_analytic_collider() -> None:
    """仓库视觉 mesh 不参与接触，CPU/CUDA 后端共用同位解析 Plane。"""

    from linkerbot_sim.configuration import load_mirror_config
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    asset_path = repo_path(_object_profile("industrial_warehouse").asset_path)
    stage = Usd.Stage.Open(str(asset_path))
    assert stage is not None

    root = "/IndustrialWarehouse/CentimeterAsset"
    visual_floor = stage.GetPrimAtPath(f"{root}/SM_Floor_A1")
    collider = stage.GetPrimAtPath(f"{root}/FloorCollider")

    assert visual_floor.IsA(UsdGeom.Mesh)
    assert not visual_floor.HasAPI(UsdPhysics.CollisionAPI)
    assert collider.IsA(UsdGeom.Plane)
    assert collider.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Plane(collider).GetAxisAttr().Get() == UsdGeom.Tokens.z
    assert UsdGeom.Imageable(collider).ComputeVisibility() == UsdGeom.Tokens.invisible
    colliders = tuple(
        str(prim.GetPath())
        for prim in Usd.PrimRange(stage.GetDefaultPrim())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    assert colliders == (f"{root}/FloorCollider",)

    scene = load_mirror_config().scene
    warehouse = next(item for item in scene.objects if item.name == "warehouse")
    assert warehouse.root_pose.rpy == (0.0, 0.0, 0.0)
    composed = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(composed, "/World")
    placed = UsdGeom.Xform.Define(composed, warehouse.prim_path)
    placed.GetPrim().GetReferences().AddReference(str(asset_path))
    placed.AddTranslateOp().Set(Gf.Vec3d(*warehouse.root_pose.xyz))

    cache = UsdGeom.XformCache()
    composed_root = f"{warehouse.prim_path}/CentimeterAsset"
    floor_matrix = cache.GetLocalToWorldTransform(
        composed.GetPrimAtPath(f"{composed_root}/SM_Floor_A1")
    )
    collider_matrix = cache.GetLocalToWorldTransform(
        composed.GetPrimAtPath(f"{composed_root}/FloorCollider")
    )
    floor_origin = np.asarray(floor_matrix.Transform(Gf.Vec3d()), dtype=np.float64)
    collider_origin = np.asarray(
        collider_matrix.Transform(Gf.Vec3d()), dtype=np.float64
    )
    np.testing.assert_allclose(floor_origin, warehouse.root_pose.xyz, atol=1.0e-9)
    np.testing.assert_allclose(collider_origin, floor_origin, atol=1.0e-9)
    normal = np.asarray(
        collider_matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(normal, (0.0, 0.0, 1.0), atol=1.0e-9)


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


def test_mjcf_explicit_inertials_have_positions_required_by_mujoco_3_8() -> None:
    asset_files = [
        *Path("assets/single_system").glob("**/*.xml"),
        *Path("assets/combined_system").glob("**/*.xml"),
    ]
    assert len(asset_files) == 6

    for asset_file in asset_files:
        root = ET.parse(asset_file).getroot()
        assert all(
            inertial.get("pos") is not None for inertial in root.iter("inertial")
        ), asset_file


def test_mjcf_equality_joints_exclude_non_schema_drive_attributes() -> None:
    asset_files = [
        *Path("assets/single_system").glob("**/*.xml"),
        *Path("assets/combined_system").glob("**/*.xml"),
    ]
    assert len(asset_files) == 6

    for asset_file in asset_files:
        root = ET.parse(asset_file).getroot()
        for equality in root.findall("./equality/joint"):
            assert not ({"limited", "damping"} & equality.attrib.keys()), asset_file
            assert len(str(equality.get("polycoef", "0 1 0 0 0")).split()) <= 5, (
                asset_file
            )


def test_mjcf_actuators_exclude_equality_followers_and_have_valid_ranges() -> None:
    asset_files = [
        *Path("assets/single_system").glob("**/*.xml"),
        *Path("assets/combined_system").glob("**/*.xml"),
    ]
    assert len(asset_files) == 6

    for asset_file in asset_files:
        root = ET.parse(asset_file).getroot()
        follower_joints = {
            str(equality.get("joint1"))
            for equality in root.findall("./equality/joint")
            if equality.get("joint2") is not None
        }
        actuators = root.findall("./actuator/*")
        actuator_joints = {str(actuator.get("joint")) for actuator in actuators}
        assert follower_joints.isdisjoint(actuator_joints), asset_file
        for actuator in actuators:
            if actuator.get("ctrllimited") != "true":
                continue
            control_range = tuple(
                float(value) for value in str(actuator.get("ctrlrange", "")).split()
            )
            assert len(control_range) == 2, (asset_file, actuator.get("name"))
            assert control_range[0] < control_range[1], (
                asset_file,
                actuator.get("name"),
                control_range,
            )


def test_default_rope_and_pinch_grasp_action_constants() -> None:
    rope = object_profile_from_mapping(
        load_yaml("configs/objects/capsule_rope.yaml"),
        profile_name="capsule_rope",
    )
    assert isinstance(rope, DynamicChainObjectProfileConfig)
    assert repo_path(rope.asset_path).is_file()
    assert rope.root_path == "/CapsuleRope"
    assert rope.physics.material is not None
    assert rope.physics.material.static_friction == 0.7
    assert rope.physics.material.dynamic_friction == 0.5
    assert rope.physics.physx is not None
    assert rope.physics.physx.solver is not None
    assert rope.physics.physx.solver.position_iterations == 48


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
        object_profile_from_mapping(
            {
                "object": {
                    "kind": "dynamic_chain",
                    "source": "usd",
                    "asset_path": "x.usd",
                    "root_path": "/Rope",
                    "segments": 18,
                    "state_summary": {"reference_body": "segment_0"},
                }
            },
            profile_name="rope",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("object profile accepted an asset-generation field")


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
                },
            }
        }
    )

    assert config.import_config.fix_base is False
    assert config.import_config.merge_fixed_joints is False
    assert config.import_config.collision_from_visuals is True


def test_import_config_rejects_fields_unsupported_by_asset_format() -> None:
    for asset_type, field in (
        ("mjcf", "collision_from_visuals"),
        ("mjcf", "merge_fixed_joints"),
        ("mjcf", "import_sites"),
        ("mjcf", "import_inertia_tensor"),
        ("urdf", "import_sites"),
        ("urdf", "import_inertia_tensor"),
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


def test_robot_configs_provide_solver_iteration_settings() -> None:
    """机器人刚体 solver iteration 应写在 robot.physics.physx.solver。"""

    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        solver = _robot_asset_config(config).physx.solver_iterations
        assert solver is not None, f"{path} must provide robot.physics.physx.solver"
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


def test_robot_profiles_do_not_own_scene_root_pose() -> None:
    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        assert "root_pose" not in config
        assert "robots" not in config


def test_robot_profiles_do_not_inline_scene_solver_type() -> None:
    for path in sorted(Path("configs/robots").glob("*.yaml")):
        config = load_yaml(path)
        robot = config["robot"]
        solver = robot.get("physics", {}).get("physx", {}).get("solver", {})
        assert "type" not in solver
        assert set(solver) <= {"arm", "hand"}
        for component in ("arm", "hand"):
            if component in solver:
                assert set(solver[component]) <= {
                    "position_iterations",
                    "velocity_iterations",
                }


def test_object_profiles_define_all_object_runtime_properties() -> None:
    for path in sorted(Path("configs/objects").glob("*.yaml")):
        profile = object_profile_from_mapping(load_yaml(path), profile_name=path.stem)
        assert profile.asset_path
        if isinstance(profile, RigidObjectProfileConfig):
            assert profile.kind == "rigid"
            assert profile.source in {"usd", "urdf"}
            assert isinstance(profile.physics, RigidObjectPhysicsConfig)
        else:
            assert isinstance(profile, DynamicChainObjectProfileConfig)
            assert profile.kind == "dynamic_chain"
            assert profile.source == "usd"
            assert profile.root_path.startswith("/")
        assert not hasattr(profile, "raw")


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
            ObjectInstanceSettings.from_mapping(
                {
                    "name": "object",
                    "object_profile": "workstation_armbase",
                    "prim_path": "/World/Object",
                    "root_pose": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
                    key: value,
                },
                label="scene.objects[0]",
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Scene instance accepted object profile key: {key}")

    try:
        ObjectInstanceSettings.from_mapping(
            {
                "name": "object",
                "object_profile": "workstation_armbase",
                "prim_path": "/World/Object",
            },
            label="scene.objects[0]",
        )
    except ValueError as exc:
        assert "root_pose" in str(exc)
    else:
        raise AssertionError("Scene instance accepted missing root_pose")


def test_solver_settings_keep_scene_and_robot_layers_separate() -> None:
    """typed scene solver 与 robot iteration 可组合但不混淆所有权。"""

    from linkerbot_sim.assets.solver_overrides import (
        SolverIterationConfig,
        merge_solver_configs,
        solver_iterations_for_prim_name,
    )

    scene_only = SolverIterationConfig(solver_type="TGS")
    hand_only = robot_solver_settings(
        {"hand": {"position_iterations": 48}},
        label="robot.physics.physx.solver",
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
        label="robot.physics.physx.solver",
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
        label="robot.physics.physx.solver",
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


def test_robot_solver_rejects_scene_type_field() -> None:
    try:
        robot_solver_settings({"type": "TGS"}, label="robot.physics.physx.solver")
    except ValueError as exc:
        assert "env solver.type" in str(exc)
    else:
        raise AssertionError("robot solver accepted scene type field")


def test_robot_solver_rejects_negative_iterations() -> None:
    try:
        robot_solver_settings(
            {"arm": {"position_iterations": -1}},
            label="robot.physics.physx.solver",
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("robot solver accepted negative iteration count")
