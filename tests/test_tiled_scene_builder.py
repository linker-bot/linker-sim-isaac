from __future__ import annotations

from pathlib import Path

import numpy as np

from linkerbot_sim.assets.robot_loader import RobotGravityPolicy, RootPoseConfig
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.paths import env_origins
from linkerbot_sim.tiled.scene import (
    _apply_per_env_object_pose_overrides,
    _apply_per_env_robot_root_pose_overrides,
    _clone_config_compatible_with_robots,
    _filter_env_collisions,
    _grid_cloner_default_positions,
    _physics_replication_root_path,
    _robot_world_root_pose,
    _tiled_object_prim_paths,
    ImportedTiledRobot,
    env_local_robot_execution,
    env_local_runtime_object_configs,
    tiled_robot_instances_from_env_config,
)


def _sample_imported_tiled_robot() -> ImportedTiledRobot:
    env_config = load_profile_yaml("env", "scene3")
    scene_instance = tiled_robot_instances_from_env_config(env_config)[0].scene_instance
    profile = load_profile_yaml("robot", scene_instance.robot_profile)
    execution = env_local_robot_execution(
        robot_profile_config=profile,
        scene_instance=scene_instance,
        env_root="/World/envs/env_0",
        robot_name="tiled_left",
    )
    return ImportedTiledRobot(
        name="left",
        profile_name=scene_instance.robot_profile,
        execution=execution,
        articulation_root_suffix="AR5V2_L6V1_L/base",
        imported_root_suffix="AR5V2_L6V1_L",
        articulation_paths=(
            "/World/envs/env_0/AR5V2_L6V1_L/base",
            "/World/envs/env_1/AR5V2_L6V1_L/base",
        ),
        imported_root_paths=(
            "/World/envs/env_0/AR5V2_L6V1_L",
            "/World/envs/env_1/AR5V2_L6V1_L",
        ),
        asset_path=Path("robot.xml"),
        asset_type="mjcf",
        controlled_joints=(),
        gravity_policy=RobotGravityPolicy(),
        gravity_counts={},
        solver_counts={},
    )


def test_tiled_robot_instances_accept_single_robot_env() -> None:
    env_config = {
        "robots": {
            "single": {
                "robot_profile": "ar5v2_l6v1_l",
                "root_pose": {"xyz": [0.0, 0.1, 0.0], "rpy": [0.0, 0.0, 0.0]},
            }
        }
    }

    instances = tiled_robot_instances_from_env_config(env_config)

    assert [instance.name for instance in instances] == ["single"]
    assert instances[0].profile_name == "ar5v2_l6v1_l"


def test_tiled_robot_instances_accept_dual_robot_env() -> None:
    env_config = load_profile_yaml("env", "scene3")

    instances = tiled_robot_instances_from_env_config(env_config)

    assert [instance.name for instance in instances] == ["left", "right"]
    assert [instance.profile_name for instance in instances] == [
        "ar5v2_l6v1_l",
        "ar5v2_l6v1_r",
    ]


def test_env_local_robot_execution_rewrites_only_runtime_prim_path() -> None:
    env_config = load_profile_yaml("env", "scene3")
    scene_instance = tiled_robot_instances_from_env_config(env_config)[0].scene_instance
    profile = load_profile_yaml("robot", scene_instance.robot_profile)

    execution = env_local_robot_execution(
        robot_profile_config=profile,
        scene_instance=scene_instance,
        env_root="/World/envs/env_0",
        robot_name="tiled_left",
    )

    assert execution.robot.name == "tiled_left"
    assert execution.robot.prim_path == "/World/envs/env_0/AR5V2_L6V1_L"
    assert execution.root_pose.xyz == (0.0, 0.09, 0.0)
    assert profile["robot"]["prim_path"] == "/World/AR5V2_L6V1_L"


def test_env_local_runtime_object_configs_rewrite_profile_paths() -> None:
    env_config = load_profile_yaml("env", "scene3")

    objects = env_local_runtime_object_configs(
        env_config,
        env_root="/World/envs/env_0",
    )

    assert [item.name for item in objects] == ["workstation", "Tblock"]
    assert [item.profile.prim_path for item in objects] == [
        "/World/envs/env_0/WorkstationArmBase",
        "/World/envs/env_0/TBlock",
    ]


def test_tiled_object_prim_paths_follow_env_roots() -> None:
    env_config = load_profile_yaml("env", "scene3_tiled")
    tiled_config = TiledEnvConfig.from_env_config(env_config)
    roots = tuple(f"/World/envs/env_{index}" for index in range(tiled_config.num_envs))
    objects = env_local_runtime_object_configs(
        env_config,
        env_root="/World/envs/env_0",
    )

    object_paths = _tiled_object_prim_paths(
        object_configs=objects,
        env_zero="/World/envs/env_0",
        env_roots=roots,
    )

    assert object_paths["Tblock"] == tuple(
        f"/World/envs/env_{index}/TBlock" for index in range(tiled_config.num_envs)
    )


def test_per_env_object_pose_override_requires_base_object(monkeypatch) -> None:
    env_config = {
        "tiled": {
            "enabled": True,
            "num_envs": 1,
            "per_env": [
                {
                    "env_id": 0,
                    "objects": {
                        "missing": {
                            "root_pose": {
                                "xyz": [0.0, 0.0, 0.0],
                                "rpy": [0.0, 0.0, 0.0],
                            }
                        }
                    },
                }
            ],
        }
    }
    tiled_config = TiledEnvConfig.from_env_config(env_config)

    calls: list[tuple[str, tuple[float, float, float]]] = []
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.objects.apply_root_pose_to_prim",
        lambda stage, path, pose: calls.append((path, pose.xyz)),
    )

    try:
        _apply_per_env_object_pose_overrides(
            stage=object(),
            config=tiled_config,
            object_prim_paths={"Tblock": ("/World/envs/env_0/TBlock",)},
            status_prefix=None,
        )
    except ValueError as exc:
        assert "unknown object" in str(exc)
    else:
        raise AssertionError("accepted per-env override for an unknown object")
    assert calls == []


def test_per_env_object_pose_override_applies_env_local_path(monkeypatch) -> None:
    env_config = load_profile_yaml("env", "scene3_tiled")
    env_config["tiled"]["num_envs"] = 2
    env_config["tiled"]["per_env"] = tuple(
        item for item in env_config["tiled"]["per_env"] if item["env_id"] < 2
    )
    tiled_config = TiledEnvConfig.from_env_config(env_config)

    calls: list[tuple[str, tuple[float, float, float]]] = []
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.objects.apply_root_pose_to_prim",
        lambda stage, path, pose: calls.append((path, pose.xyz)),
    )

    applied = _apply_per_env_object_pose_overrides(
        stage=object(),
        config=tiled_config,
        object_prim_paths={
            "Tblock": (
                "/World/envs/env_0/TBlock",
                "/World/envs/env_1/TBlock",
            )
        },
        status_prefix=None,
    )

    assert applied == 2
    assert calls == [
        ("/World/envs/env_0/TBlock", (0.15, 0.0, -0.4)),
        ("/World/envs/env_1/TBlock", (0.12, 0.04, -0.4)),
    ]


def test_robot_world_root_pose_adds_env_origin_without_changing_rpy() -> None:
    pose = RootPoseConfig(xyz=(0.0, 0.09, 0.1), rpy=(-1.5707, 0.0, 0.2))

    world_pose = _robot_world_root_pose(pose, np.asarray([2.0, 4.0, 0.5]))

    assert world_pose.xyz == (2.0, 4.09, 0.6)
    assert world_pose.rpy == pose.rpy


def test_per_env_robot_root_pose_override_writes_mjcf_world_anchors(
    monkeypatch,
) -> None:
    robot = _sample_imported_tiled_robot()

    calls: list[tuple[str, tuple[float, float, float]]] = []
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.root_pose.apply_mjcf_fixed_root_joint_pose",
        lambda stage, path, pose: calls.append((path, pose.xyz)),
    )

    applied = _apply_per_env_robot_root_pose_overrides(
        stage=object(),
        robots={"left": robot},
        env_origins=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        status_prefix=None,
    )

    assert applied == 2
    assert calls == [
        ("/World/envs/env_0/AR5V2_L6V1_L", (0.0, 0.09, 0.0)),
        ("/World/envs/env_1/AR5V2_L6V1_L", (2.0, 0.09, 0.0)),
    ]


def test_mjcf_world_fixed_root_joints_disable_physics_replication(
    monkeypatch,
) -> None:
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "clone": {"replicate_physics": True},
            }
        }
    )
    robot = _sample_imported_tiled_robot()
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.clone.mjcf_fixed_root_joint_paths_without_body0",
        lambda stage, root_path: (f"{root_path}/joints/rootJoint_base",),
    )

    effective = _clone_config_compatible_with_robots(
        stage=object(),
        config=config,
        robots={"left": robot},
        status_prefix=None,
    )

    assert config.clone.replicate_physics is True
    assert effective.clone.replicate_physics is False


def test_grid_cloner_offsets_preserve_project_env_origin_semantics() -> None:
    config = TiledEnvConfig.from_env_config(
        {"tiled": {"enabled": True, "num_envs": 5, "spacing": 2.5}}
    )

    desired = env_origins(config)
    grid_default = _grid_cloner_default_positions(config)
    offsets = desired - grid_default

    assert grid_default.shape == desired.shape
    assert not np.allclose(grid_default, desired)
    np.testing.assert_allclose(grid_default + offsets, desired)


def test_physics_replication_root_path_matches_grid_cloner_prefix() -> None:
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "base_env_path": "/World/envs",
                "env_prefix": "env",
            }
        }
    )

    assert _physics_replication_root_path(config) == "/World/envs/env_"


def test_collision_filter_uses_filtered_pairs_instead_of_collision_groups() -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for env_id in range(2):
        root_path = f"/World/envs/env_{env_id}"
        UsdGeom.Xform.Define(stage, root_path)
        body = UsdGeom.Xform.Define(stage, f"{root_path}/Body").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        collider = UsdGeom.Cube.Define(stage, f"{root_path}/StaticCollider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)

    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "clone": {
                    "filter_collisions": True,
                    "collision_root_path": "/World/collisions",
                },
            }
        }
    )

    applied = _filter_env_collisions(
        stage=stage,
        config=config,
        env_roots=("/World/envs/env_0", "/World/envs/env_1"),
    )

    assert applied is True
    assert not stage.GetPrimAtPath("/World/collisions").IsValid()

    body_api = UsdPhysics.FilteredPairsAPI.Get(
        stage, Sdf.Path("/World/envs/env_0/Body")
    )
    body_targets = body_api.GetFilteredPairsRel().GetTargets()
    assert Sdf.Path("/World/envs/env_1/Body") in body_targets
    assert Sdf.Path("/World/envs/env_1/StaticCollider") in body_targets

    collider_api = UsdPhysics.FilteredPairsAPI.Get(
        stage,
        Sdf.Path("/World/envs/env_0/StaticCollider"),
    )
    collider_targets = collider_api.GetFilteredPairsRel().GetTargets()
    assert Sdf.Path("/World/envs/env_1/Body") in collider_targets
    assert Sdf.Path("/World/envs/env_1/StaticCollider") in collider_targets


def test_collision_filter_disabled_does_not_author_filtered_pairs() -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for env_id in range(2):
        body = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Body").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "clone": {"filter_collisions": False},
            }
        }
    )

    applied = _filter_env_collisions(
        stage=stage,
        config=config,
        env_roots=("/World/envs/env_0", "/World/envs/env_1"),
    )

    assert applied is False
    body_api = UsdPhysics.FilteredPairsAPI.Get(
        stage, Sdf.Path("/World/envs/env_0/Body")
    )
    assert not body_api
