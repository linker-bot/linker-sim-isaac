from __future__ import annotations

from pathlib import Path
import re

import pytest

from linkerbot_sim.assets.robot_config import (
    RobotAssetConfig,
    load_robot_profile,
    validate_robot_profile,
)
from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    RobotSceneInstanceConfig,
)
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.robots.capabilities import (
    PlanningBindingConfig,
    robot_kind_from_profile,
)
from linkerbot_sim.utils.config import load_yaml


def _hand_profile() -> dict[str, object]:
    return {
        "robot": {
            "kind": "hand",
            "name": "unit_hand",
            "asset_type": "mjcf",
            "asset_path": "assets/single_system/hand/L6V1_L/L6V1_L.xml",
        },
        "curobo": {"enabled": False},
        "joint_groups": {"arm": [], "hand": ["hand_joint"]},
    }


def test_all_bundled_robot_yaml_loads_strictly() -> None:
    paths = sorted(Path("configs/robots").glob("*.yaml"))

    assert paths
    for path in paths:
        raw = load_yaml(path)
        canonical = load_robot_profile(path)
        assert canonical == raw
        instance = RobotSceneInstanceConfig(
            robot_profile=path.stem,
            root_pose=RootPoseConfig(),
            label="robot_0",
        )
        asset = RobotAssetConfig.from_mapping(
            canonical,
            prim_path=instance.effective_prim_path,
        )
        execution = RobotExecutionConfig.from_mapping(
            canonical,
            scene_instance=instance,
        )
        kind = robot_kind_from_profile(canonical)
        binding = PlanningBindingConfig.from_profile(canonical, kind=kind)
        assert asset.asset_path == execution.robot.asset_path
        assert execution.controlled_joints == ("all",)
        assert binding.enabled is (kind.value != "hand")


def test_robot_asset_config_requires_resolved_prim_path() -> None:
    with pytest.raises(TypeError, match="prim_path"):
        RobotAssetConfig.from_mapping(_hand_profile())  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("mutate", "path"),
    (
        (lambda profile: profile.update({"robto": {}}), "profile.robto"),
        (
            lambda profile: profile["robot"].update({"asset_pth": "x"}),
            "robot.asset_pth",
        ),
        (
            lambda profile: profile["robot"].update({"import": {"fix_bsae": True}}),
            "robot.import.fix_bsae",
        ),
        (
            lambda profile: profile["robot"].update({"physics": {"gravty": {}}}),
            "robot.physics.gravty",
        ),
        (
            lambda profile: profile["robot"].update(
                {"physics": {"physx": {"material": {"restituton": 0.0}}}}
            ),
            "robot.physics.physx.material.restituton",
        ),
        (
            lambda profile: profile["robot"].update(
                {"physics": {"solver": {"hand": {"position_iteratons": 2}}}}
            ),
            "robot.physics.solver.hand.position_iteratons",
        ),
        (
            lambda profile: profile["robot"].update(
                {
                    "planning_collision": {
                        "spheres": [
                            {
                                "name": "hand",
                                "center": [0.0, 0.0, 0.0],
                                "radius": 0.1,
                                "raduis": 0.2,
                            }
                        ]
                    }
                }
            ),
            "robot.planning_collision.spheres[0].raduis",
        ),
        (
            lambda profile: profile["joint_groups"].update({"hnad": []}),
            "joint_groups.hnad",
        ),
        (
            lambda profile: profile.update({"rigid_body_groups": {"hnad": []}}),
            "rigid_body_groups.hnad",
        ),
        (
            lambda profile: profile["curobo"].update({"motion_planner": {}}),
            "curobo.motion_planner",
        ),
    ),
)
def test_robot_typo_reports_complete_nested_path(mutate, path: str) -> None:
    profile = _hand_profile()
    mutate(profile)

    with pytest.raises(ValueError, match=re.escape(path)):
        validate_robot_profile(profile)


@pytest.mark.parametrize("value", ("false", 0, 1, None, [], {}))
def test_robot_boolean_fields_are_strict(value: object) -> None:
    profile = _hand_profile()
    robot = profile["robot"]
    assert isinstance(robot, dict)
    robot["import"] = {"self_collision": value}

    with pytest.raises(ValueError, match=r"robot\.import\.self_collision"):
        validate_robot_profile(profile)


@pytest.mark.parametrize("value", (True, "32", -1, 1.5, None))
def test_robot_solver_iterations_are_strict_non_negative_integers(
    value: object,
) -> None:
    profile = _hand_profile()
    robot = profile["robot"]
    assert isinstance(robot, dict)
    robot["physics"] = {"solver": {"hand": {"position_iterations": value}}}

    with pytest.raises(
        ValueError,
        match=r"robot\.physics\.solver\.hand\.position_iterations",
    ):
        validate_robot_profile(profile)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("center", [0.0, 0.0], "center"),
        ("center", [0.0, True, 0.0], "center"),
        ("radius", 0.0, "radius"),
        ("radius", -0.1, "radius"),
        ("radius", float("nan"), "radius"),
        ("radius", True, "radius"),
    ),
)
def test_robot_planning_collision_vectors_and_radius_are_strict(
    field: str,
    value: object,
    message: str,
) -> None:
    profile = _hand_profile()
    robot = profile["robot"]
    assert isinstance(robot, dict)
    sphere: dict[str, object] = {
        "name": "hand",
        "center": [0.0, 0.0, 0.0],
        "radius": 0.1,
    }
    sphere[field] = value
    robot["planning_collision"] = {"spheres": [sphere]}

    with pytest.raises(ValueError, match=message):
        validate_robot_profile(profile)


def test_robot_physx_restitution_is_bounded() -> None:
    profile = _hand_profile()
    robot = profile["robot"]
    assert isinstance(robot, dict)
    robot["physics"] = {"physx": {"material": {"contact_restitution": 1.01}}}

    with pytest.raises(ValueError, match="contact_restitution.*between 0 and 1"):
        validate_robot_profile(profile)


def test_robot_controlled_joints_must_use_declared_command_groups() -> None:
    profile = _hand_profile()
    profile["controlled_joints"] = ["missing_joint"]

    with pytest.raises(ValueError, match="outside arm/hand groups"):
        validate_robot_profile(profile)

    profile["controlled_joints"] = ["all", "hand_joint"]
    with pytest.raises(ValueError, match="must be the only selector"):
        validate_robot_profile(profile)


def test_disabled_curobo_binding_rejects_model_fields() -> None:
    profile = _hand_profile()
    profile["curobo"] = {
        "enabled": False,
        "robot": {"urdf_path": "unused.urdf"},
    }

    with pytest.raises(ValueError, match=r"curobo\.robot"):
        validate_robot_profile(profile)


def test_mjcf_robot_rejects_urdf_only_drive_type() -> None:
    profile = _hand_profile()
    robot = profile["robot"]
    assert isinstance(robot, dict)
    robot["urdf_drive_type"] = "position"

    with pytest.raises(ValueError, match="only supported for URDF"):
        validate_robot_profile(profile)


def test_robot_rejects_profile_owned_prim_path() -> None:
    profile = _hand_profile()
    robot = profile["robot"]
    assert isinstance(robot, dict)
    robot["prim_path"] = "/World/Robot"

    with pytest.raises(ValueError, match=r"robot\.prim_path"):
        validate_robot_profile(profile)
