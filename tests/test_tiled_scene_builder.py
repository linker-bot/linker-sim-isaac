from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from linkerbot_sim.assets.robot_config import RobotGravityPolicy
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.configs.profiles import load_profile_yaml
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.scene.builder import build_isaac_tiled_scene
from linkerbot_sim.tiled.scene.paths import env_origins
from linkerbot_sim.tiled.scene.clone import (
    _clone_config_compatible_with_robots,
    _grid_cloner_default_positions,
    _physics_replication_root_path,
)
from linkerbot_sim.tiled.scene.objects import (
    _apply_per_env_object_pose_overrides,
    _tiled_object_prim_paths,
    env_local_runtime_object_configs,
)
from linkerbot_sim.tiled.scene.robots import (
    _import_env_zero_robots,
    env_local_robot_execution,
    tiled_robot_instances_from_env_config,
)
from linkerbot_sim.tiled.scene import robots as tiled_robots
from linkerbot_sim.tiled.scene.collision_filter import (
    _author_collision_group_prims,
    _filter_env_collisions,
    _plan_env_collision_groups,
    _resolve_global_collision_paths,
    _resolve_physics_scene_path,
)
from linkerbot_sim.tiled.scene.root_pose import (
    _apply_per_env_robot_root_pose_overrides,
    _robot_world_root_pose,
)
from linkerbot_sim.tiled.scene.types import ImportedTiledRobot


def _sample_imported_tiled_robot() -> ImportedTiledRobot:
    env_config = load_profile_yaml("env", "scene3")
    scene_instance = tiled_robot_instances_from_env_config(env_config)[0].scene_instance
    profile = load_profile_yaml("robot", scene_instance.robot_profile)
    execution = env_local_robot_execution(
        robot_profile_config=profile,
        scene_instance=scene_instance,
        env_root="/World/envs/env_0",
        robot_name="tiled_robot_0",
    )
    return ImportedTiledRobot(
        name="robot_0",
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


def test_tiled_robot_instances_accept_multi_robot_env() -> None:
    env_config = load_profile_yaml("env", "scene3")

    instances = tiled_robot_instances_from_env_config(env_config)

    assert [instance.label for instance in instances] == [
        str(robot["label"]) for robot in env_config["robots"]
    ]
    assert [instance.profile_name for instance in instances] == [
        "ar5v2_l6v1_l",
        "ar5v2_l6v1_r",
    ]
    assert [instance.scene_instance.robot_id for instance in instances] == [0, 1]
    assert [instance.scene_instance.label for instance in instances] == [
        str(robot["label"]) for robot in env_config["robots"]
    ]


def test_tiled_robot_instances_accept_new_robot_list_and_use_labels() -> None:
    env_config = {
        "robots": [
            {
                "label": "robot_a",
                "robot_profile": "ar5v2_l6v1_l",
                "root_pose": {"xyz": [0.0, 0.1, 0.0], "rpy": [0.0, 0.0, 0.0]},
            },
            {
                "label": "robot_b",
                "robot_profile": "ar5v2_l6v1_l",
                "root_pose": {"xyz": [0.0, -0.1, 0.0], "rpy": [0.0, 0.0, 0.0]},
            },
        ]
    }

    instances = tiled_robot_instances_from_env_config(env_config)

    assert [instance.label for instance in instances] == ["robot_a", "robot_b"]
    assert [instance.scene_instance.robot_id for instance in instances] == [0, 1]
    assert [instance.scene_instance.label for instance in instances] == [
        "robot_a",
        "robot_b",
    ]
    executions = [
        env_local_robot_execution(
            robot_profile_config=load_profile_yaml("robot", instance.profile_name),
            scene_instance=instance.scene_instance,
            env_root="/World/envs/env_0",
            robot_name=f"tiled_{instance.label}",
        )
        for instance in instances
    ]
    assert [execution.robot.prim_path for execution in executions] == [
        "/World/envs/env_0/Robots/robot_a",
        "/World/envs/env_0/Robots/robot_b",
    ]


def test_env_local_robot_execution_rewrites_only_runtime_prim_path() -> None:
    env_config = load_profile_yaml("env", "scene3")
    scene_instance = tiled_robot_instances_from_env_config(env_config)[0].scene_instance
    profile = load_profile_yaml("robot", scene_instance.robot_profile)

    execution = env_local_robot_execution(
        robot_profile_config=profile,
        scene_instance=scene_instance,
        env_root="/World/envs/env_0",
        robot_name="tiled_robot_0",
    )

    assert execution.robot.name == "tiled_robot_0"
    assert execution.robot.prim_path == (
        f"/World/envs/env_0/Robots/{scene_instance.label}"
    )
    assert execution.root_pose.xyz == (0.0, 0.09, 0.0)
    assert "prim_path" not in profile["robot"]


def test_tiled_import_resolves_per_robot_controller_bundles_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_config = {
        "robots": [
            {
                "label": "instance_override",
                "robot_profile": "ar5v2_l6v1_l",
                "controller_profile": "instance_bundle",
                "root_pose": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            },
            {
                "label": "robot_override_a",
                "robot_profile": "ar5v2_l6v1_r",
                "root_pose": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            },
            {
                "label": "robot_override_b",
                "robot_profile": "ar5v2_l6v1_r",
                "root_pose": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            },
        ]
    }
    original_loader = load_profile_yaml

    def fake_profile_loader(group: str, name: str):
        profile = deepcopy(original_loader(group, name))
        if group == "robot" and name == "ar5v2_l6v1_r":
            profile["robot"]["controller_profile"] = "robot_bundle"
        return profile

    bundle_calls: list[str] = []
    bundles = {"instance_bundle": object(), "robot_bundle": object()}
    imported_profiles: dict[str, object] = {}

    def bundle_loader(name: str):
        bundle_calls.append(name)
        return bundles[name]

    def fake_import(*, robot_execution, controller_profiles, **_kwargs):
        name = robot_execution.robot.name
        imported_profiles[name] = controller_profiles
        root = f"/World/envs/env_0/{name}"
        return {
            "articulation_path": f"{root}/base",
            "imported_root_path": root,
            "asset_path": robot_execution.robot.asset_path,
            "gravity_policy": robot_execution.robot.gravity_policy,
            "gravity_counts": {},
            "solver_counts": {},
        }

    monkeypatch.setattr(tiled_robots, "load_profile_yaml", fake_profile_loader)
    monkeypatch.setattr(tiled_robots, "_import_robot_to_env_zero", fake_import)

    imported = _import_env_zero_robots(
        stage=object(),
        env_config=env_config,
        tiled_config=object(),
        controller_bundle="runtime_bundle",
        controller_bundle_loader=bundle_loader,
        env_roots=("/World/envs/env_0",),
        status_prefix=None,
    )

    assert bundle_calls == ["instance_bundle", "robot_bundle"]
    assert imported["instance_override"].controller_profile == "instance_bundle"
    assert imported["robot_override_a"].controller_profile == "robot_bundle"
    assert imported["robot_override_b"].controller_profile == "robot_bundle"
    assert imported_profiles == {
        "tiled_instance_override": bundles["instance_bundle"],
        "tiled_robot_override_a": bundles["robot_bundle"],
        "tiled_robot_override_b": bundles["robot_bundle"],
    }


def test_env_local_runtime_object_configs_namespace_resolved_paths_only() -> None:
    env_config = load_profile_yaml("env", "scene3")

    objects = env_local_runtime_object_configs(
        env_config,
        env_root="/World/envs/env_0",
    )

    assert [item.name for item in objects] == ["warehouse", "workstation", "Tblock"]
    assert [item.prim_path for item in objects] == [
        "/World/envs/env_0/IndustrialWarehouse",
        "/World/envs/env_0/WorkstationArmBase",
        "/World/envs/env_0/TBlock",
    ]
    assert all("prim_path" not in item.profile.raw["object"] for item in objects)


def test_env_local_runtime_object_configs_namespace_repeated_profile_paths() -> None:
    env_config = {
        "objects": [
            {
                "name": name,
                "object_profile": "workstation_armbase",
                "root_pose": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            }
            for name in ("fixture_a", "fixture_b")
        ]
    }

    objects = env_local_runtime_object_configs(
        env_config,
        env_root="/World/envs/env_0",
    )

    assert [item.prim_path for item in objects] == [
        "/World/envs/env_0/Objects/fixture_a",
        "/World/envs/env_0/Objects/fixture_b",
    ]
    assert objects[0].profile is objects[1].profile
    assert "prim_path" not in objects[0].profile.raw["object"]


def test_tiled_builder_rejects_robot_object_prim_tree_overlap_before_stage_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_config = {
        "robots": [
            {
                "label": "robot",
                "robot_profile": "ar5v2_l",
                "prim_path": "/World/Shared",
                "root_pose": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            }
        ],
        "objects": [
            {
                "name": "fixture",
                "object_profile": "TblockV1_default",
                "prim_path": "/World/Shared/fixture",
                "root_pose": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            }
        ],
        "tiled": {"enabled": True, "num_envs": 1},
    }
    stage_mutated = False

    def fail_if_called(*_args, **_kwargs) -> None:
        nonlocal stage_mutated
        stage_mutated = True

    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.builder._define_env_zero", fail_if_called
    )

    with pytest.raises(ValueError, match="prim paths overlap"):
        build_isaac_tiled_scene(
            world=object(),
            stage=object(),
            env_config=env_config,
            tiled_config=TiledEnvConfig.from_env_config(env_config),
        )

    assert stage_mutated is False


def test_env_local_runtime_object_configs_reject_namespace_path_collision() -> None:
    env_config = {
        "objects": [
            {
                "name": "fixture_a",
                "object_profile": "workstation_armbase",
                "prim_path": "/World/envs/env_0/Objects/shared",
                "root_pose": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            },
            {
                "name": "fixture_b",
                "object_profile": "workstation_armbase",
                "prim_path": "/World/Objects/shared",
                "root_pose": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            },
        ]
    }

    with pytest.raises(ValueError, match="Duplicate tiled object prim path"):
        env_local_runtime_object_configs(
            env_config,
            env_root="/World/envs/env_0",
        )


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
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "per_env": [
                    {
                        "env_id": 1,
                        "robots": {
                            "left": {
                                "root_pose": {
                                    "xyz": [0.2, 0.15, 0.05],
                                    "rpy": [-1.2, 0.1, 0.2],
                                }
                            }
                        },
                    }
                ],
            }
        }
    )

    anchor_calls: list[tuple[str, tuple[float, float, float]]] = []
    xform_calls: list[tuple[str, RootPoseConfig]] = []
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.root_pose.apply_mjcf_fixed_root_joint_pose",
        lambda stage, path, pose: anchor_calls.append((path, pose.xyz)),
    )
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.root_pose.apply_root_pose_to_prim",
        lambda stage, path, pose: xform_calls.append((path, pose)),
    )

    applied = _apply_per_env_robot_root_pose_overrides(
        stage=object(),
        config=config,
        robots={"left": robot},
        env_origins=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        status_prefix=None,
    )

    assert applied == 2
    assert [call[1] for call in xform_calls] == [
        robot.execution.root_pose,
        RootPoseConfig(xyz=(0.2, 0.15, 0.05), rpy=(-1.2, 0.1, 0.2)),
    ]
    assert anchor_calls == [
        ("/World/envs/env_0/AR5V2_L6V1_L", (0.0, 0.09, 0.0)),
        ("/World/envs/env_1/AR5V2_L6V1_L", (2.2, 0.15, 0.05)),
    ]


def test_per_env_robot_root_pose_rejects_unknown_robot_before_stage_write(
    monkeypatch,
) -> None:
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "per_env": [
                    {
                        "env_id": 1,
                        "robots": {
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
    )
    writes: list[str] = []
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.root_pose.apply_root_pose_to_prim",
        lambda stage, path, pose: writes.append(path),
    )

    with pytest.raises(ValueError, match="unknown robot label.*missing"):
        _apply_per_env_robot_root_pose_overrides(
            stage=object(),
            config=config,
            robots={"left": _sample_imported_tiled_robot()},
            env_origins=np.zeros((2, 3), dtype=float),
            status_prefix=None,
        )

    assert writes == []


def test_tiled_ik_root_frames_use_resolved_per_env_robot_pose() -> None:
    from types import SimpleNamespace

    from linkerbot_sim.app.interactive.tiled_scene.runtime.ik import (
        _robot_root_world_frames,
    )
    from linkerbot_sim.utils.rotations import (
        rpy_xyz_to_matrix,
        rpy_xyz_to_quat_wxyz,
    )

    robot = _sample_imported_tiled_robot()
    override = RootPoseConfig(xyz=(0.2, 0.15, 0.05), rpy=(-1.2, 0.1, 0.2))
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "per_env": [
                    {
                        "env_id": 1,
                        "robots": {
                            "left": {
                                "root_pose": {
                                    "xyz": list(override.xyz),
                                    "rpy": list(override.rpy),
                                }
                            }
                        },
                    }
                ],
            }
        }
    )
    scene = SimpleNamespace(
        robots={"left": robot},
        config=config,
        env_origins=np.asarray([[10.0, -3.0, 0.75], [12.0, -3.0, 0.75]]),
    )

    positions, rotations, quats = _robot_root_world_frames(scene, "left")

    np.testing.assert_allclose(
        positions,
        [
            np.asarray(robot.execution.root_pose.xyz) + scene.env_origins[0],
            np.asarray(override.xyz) + scene.env_origins[1],
        ],
    )
    np.testing.assert_allclose(
        rotations[0], rpy_xyz_to_matrix(robot.execution.root_pose.rpy)
    )
    np.testing.assert_allclose(rotations[1], rpy_xyz_to_matrix(override.rpy))
    np.testing.assert_allclose(
        quats[0], rpy_xyz_to_quat_wxyz(robot.execution.root_pose.rpy)
    )
    np.testing.assert_allclose(quats[1], rpy_xyz_to_quat_wxyz(override.rpy))


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
        {
            "tiled": {
                "enabled": True,
                "num_envs": 5,
                "spacing": 2.5,
                "layout": {"origin_xyz": [10.0, -3.0, 0.75]},
            }
        }
    )

    desired = env_origins(config)
    grid_default = _grid_cloner_default_positions(config)
    offsets = desired - grid_default

    assert grid_default.shape == desired.shape
    assert not np.allclose(grid_default, desired)
    np.testing.assert_allclose(grid_default + offsets, desired)
    np.testing.assert_allclose(
        desired,
        [
            [10.0, -3.0, 0.75],
            [12.5, -3.0, 0.75],
            [15.0, -3.0, 0.75],
            [10.0, -0.5, 0.75],
            [12.5, -0.5, 0.75],
        ],
    )


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
                    "collision_filter_strategy": "filtered_pairs",
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


def test_physics_scene_auto_rejects_multiple_scenes_and_explicit_path_checks_type() -> (
    None
):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/World/physicsSceneA")
    UsdPhysics.Scene.Define(stage, "/World/physicsSceneB")
    UsdGeom.Xform.Define(stage, "/World/notPhysics")

    with pytest.raises(RuntimeError, match="Multiple UsdPhysics.Scene"):
        _resolve_physics_scene_path(stage, None)
    assert (
        _resolve_physics_scene_path(stage, "/World/physicsSceneB")
        == "/World/physicsSceneB"
    )
    with pytest.raises(RuntimeError, match="not a UsdPhysics.Scene"):
        _resolve_physics_scene_path(stage, "/World/notPhysics")
    with pytest.raises(RuntimeError, match="does not exist"):
        _resolve_physics_scene_path(stage, "/World/missing")


def test_global_collision_paths_resolve_auto_and_extra_with_stable_dedup() -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for env_id in range(2):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
    ground_root = UsdGeom.Xform.Define(stage, "/World/defaultGroundPlane")
    ground = UsdGeom.Cube.Define(stage, f"{ground_root.GetPath()}/Collision").GetPrim()
    UsdPhysics.CollisionAPI.Apply(ground)
    fixture = UsdGeom.Cube.Define(stage, "/World/customFixture").GetPrim()
    UsdPhysics.CollisionAPI.Apply(fixture)
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "num_envs": 2,
                "clone": {
                    "global_collision_paths": "auto",
                    "extra_global_collision_paths": [
                        "/World/customFixture",
                        "/World/defaultGroundPlane",
                    ],
                },
            }
        }
    )

    assert _resolve_global_collision_paths(
        stage=stage,
        config=config,
        env_roots=("/World/envs/env_0", "/World/envs/env_1"),
    ) == ("/World/defaultGroundPlane", "/World/customFixture")


def test_global_collision_paths_reject_missing_non_collider_and_env_overlap() -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    env_root = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    env_collider = UsdGeom.Cube.Define(
        stage, f"{env_root.GetPath()}/Collider"
    ).GetPrim()
    UsdPhysics.CollisionAPI.Apply(env_collider)
    UsdGeom.Xform.Define(stage, "/World/emptyFixture")

    for path, message in (
        ("/World/missing", "does not exist"),
        ("/World/emptyFixture", "contains no UsdPhysics collider"),
        ("/World/envs/env_0/Collider", "overlaps tiled env root"),
    ):
        config = TiledEnvConfig.from_env_config(
            {
                "tiled": {
                    "clone": {
                        "global_collision_paths": [path],
                    }
                }
            }
        )
        with pytest.raises(RuntimeError, match=message):
            _resolve_global_collision_paths(
                stage=stage,
                config=config,
                env_roots=("/World/envs/env_0",),
            )


def test_collision_group_prims_author_invert_whitelist_targets() -> None:
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for env_id in range(2):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
    UsdGeom.Xform.Define(stage, "/World/defaultGroundPlane")

    plan = _plan_env_collision_groups(
        env_roots=("/World/envs/env_0", "/World/envs/env_1"),
        global_paths=("/World/defaultGroundPlane",),
        collision_root_path="/World/collisions",
    )
    authored = _author_collision_group_prims(stage, plan)
    assert authored == plan.total_filtered_group_targets()

    def _filtered(group_path: str) -> set:
        cg = UsdPhysics.CollisionGroup.Get(stage, Sdf.Path(group_path))
        return set(cg.GetFilteredGroupsRel().GetTargets())

    def _includes(group_path: str) -> set:
        coll = Usd.CollectionAPI(stage.GetPrimAtPath(group_path), "colliders")
        return set(coll.GetIncludesRel().GetTargets())

    # env 白名单 = 自身 + global；不含另一个 env → 跨 env 不碰。
    assert _filtered("/World/collisions/env_0") == {
        Sdf.Path("/World/collisions/env_0"),
        Sdf.Path("/World/collisions/global"),
    }
    assert Sdf.Path("/World/collisions/env_1") not in _filtered(
        "/World/collisions/env_0"
    )
    # env collection 只含自己的子树（互斥，避免 collider 落入多个组）。
    assert _includes("/World/collisions/env_0") == {Sdf.Path("/World/envs/env_0")}
    # global 组与每个 env 都碰撞。
    assert _filtered("/World/collisions/global") == {
        Sdf.Path("/World/collisions/env_0"),
        Sdf.Path("/World/collisions/env_1"),
    }
    assert _includes("/World/collisions/global") == {
        Sdf.Path("/World/defaultGroundPlane")
    }


def test_collision_groups_strategy_enables_scene_invert_filter() -> None:
    import pytest

    physx = pytest.importorskip("pxr.PhysxSchema")
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    for env_id in range(2):
        body = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}/Body").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)

    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "clone": {
                    "filter_collisions": True,
                    "collision_filter_strategy": "collision_groups",
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
    scene_api = physx.PhysxSceneAPI.Get(stage, Sdf.Path("/World/physicsScene"))
    assert bool(scene_api.GetInvertCollisionGroupFilterAttr().Get()) is True
    assert stage.GetPrimAtPath("/World/collisions/env_0").IsValid()


def test_collision_groups_whitelist_custom_ground_and_fixture() -> None:
    physx = pytest.importorskip("pxr.PhysxSchema")
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/World/customPhysics")
    for env_id in range(2):
        collider = UsdGeom.Cube.Define(
            stage, f"/World/envs/env_{env_id}/Collider"
        ).GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
    ground = UsdGeom.Cube.Define(stage, "/World/customGround").GetPrim()
    UsdPhysics.CollisionAPI.Apply(ground)
    fixture_root = UsdGeom.Xform.Define(stage, "/World/sharedFixture")
    fixture = UsdGeom.Cube.Define(stage, f"{fixture_root.GetPath()}/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(fixture)
    config = TiledEnvConfig.from_env_config(
        {
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "clone": {
                    "physics_scene_path": "/World/customPhysics",
                    "global_collision_paths": ["/World/customGround"],
                    "extra_global_collision_paths": ["/World/sharedFixture"],
                },
            }
        }
    )

    assert _filter_env_collisions(
        stage=stage,
        config=config,
        env_roots=("/World/envs/env_0", "/World/envs/env_1"),
    )

    scene_api = physx.PhysxSceneAPI.Get(stage, Sdf.Path("/World/customPhysics"))
    assert bool(scene_api.GetInvertCollisionGroupFilterAttr().Get()) is True
    global_collection = Usd.CollectionAPI(
        stage.GetPrimAtPath("/World/collisions/global"), "colliders"
    )
    assert set(global_collection.GetIncludesRel().GetTargets()) == {
        Sdf.Path("/World/customGround"),
        Sdf.Path("/World/sharedFixture"),
    }
    env_zero = UsdPhysics.CollisionGroup.Get(stage, Sdf.Path("/World/collisions/env_0"))
    assert set(env_zero.GetFilteredGroupsRel().GetTargets()) == {
        Sdf.Path("/World/collisions/env_0"),
        Sdf.Path("/World/collisions/global"),
    }
    assert Sdf.Path("/World/collisions/env_1") not in set(
        env_zero.GetFilteredGroupsRel().GetTargets()
    )
