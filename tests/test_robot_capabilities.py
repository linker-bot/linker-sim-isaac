from __future__ import annotations

import pytest

from linkerbot_sim.robots.capabilities import (
    PlanningBindingConfig,
    RobotKind,
    robot_kind_from_profile,
)
from linkerbot_sim.robots.joint_groups import JointGroupLayout
from linkerbot_sim.utils.config import load_yaml


def test_builtin_profiles_declare_kind_binding_and_disjoint_groups() -> None:
    for name, expected_kind in (
        ("ar5v2_l", RobotKind.ARM),
        ("ar5v2_l6v1_l", RobotKind.ARM_HAND),
        ("ar5v2_l6v1_r", RobotKind.ARM_HAND),
        ("l6v1_l", RobotKind.HAND),
        ("l6v1_r", RobotKind.HAND),
    ):
        profile = load_yaml(f"configs/robots/{name}.yaml")
        kind = robot_kind_from_profile(profile)
        binding = PlanningBindingConfig.from_profile(profile, kind=kind)
        groups = profile["joint_groups"]
        command_names = tuple((*groups["arm"], *groups["hand"]))
        layout = JointGroupLayout.resolve(
            kind=kind,
            command_joint_names=command_names,
            joint_groups=groups,
        )
        assert kind is expected_kind
        assert binding.enabled is (expected_kind is not RobotKind.HAND)
        assert not (set(layout.arm) & set(layout.hand))


def test_hand_profile_cannot_enable_planning() -> None:
    profile = {
        "robot": {"kind": "hand"},
        "curobo": {
            "enabled": True,
            "planning_joint_group": "arm",
            "robot": {"urdf_path": "x"},
        },
    }
    with pytest.raises(ValueError, match="hand.*cannot enable"):
        PlanningBindingConfig.from_profile(profile, kind=RobotKind.HAND)


def test_kind_and_curobo_enabled_are_required() -> None:
    with pytest.raises(ValueError, match="robot.kind is required"):
        robot_kind_from_profile({"robot": {}, "curobo": {"enabled": False}})
    with pytest.raises(ValueError, match="curobo.enabled is required"):
        PlanningBindingConfig.from_profile(
            {"robot": {"kind": "arm"}, "curobo": {"robot": {"urdf_path": "x"}}},
            kind=RobotKind.ARM,
        )
    with pytest.raises(ValueError, match="planning_joint_group is required"):
        PlanningBindingConfig.from_profile(
            {
                "robot": {"kind": "arm"},
                "curobo": {"enabled": True, "robot": {"urdf_path": "x"}},
            },
            kind=RobotKind.ARM,
        )


def test_joint_group_layout_rejects_multiple_writers() -> None:
    with pytest.raises(ValueError, match="multiple writers"):
        JointGroupLayout(
            ("j0", "j1"),
            arm=("j0",),
            hand=("j0", "j1"),
        )


def test_joint_group_layout_requires_explicit_groups() -> None:
    with pytest.raises(ValueError, match="joint_groups is required"):
        JointGroupLayout.resolve(
            kind=RobotKind.ARM,
            command_joint_names=("arm_joint",),
            joint_groups=None,
            planning_joint_names=("arm_joint",),
        )
