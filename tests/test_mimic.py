from __future__ import annotations

import numpy as np

from linkerbot_sim.assets.asset_paths import DEFAULT_AR5_L6_MJCF
from linkerbot_sim.robots.mimic import (
    MimicFollowerControl,
    MimicFollowerTargetMapper,
    MjcfJointEquality,
    expand_targets_with_mjcf_equalities,
    mjcf_equality_follower_joint_names,
    parse_mjcf_joint_equalities,
    resolve_mimic_follower_controls,
)


def test_parse_ar5_l6_mjcf_equalities() -> None:
    equalities = parse_mjcf_joint_equalities(DEFAULT_AR5_L6_MJCF)
    names = {equality.dependent_joint for equality in equalities}
    assert "L6V1_L_hand_index_dip" in names
    assert "L6V1_L_hand_thumb_dip" in names
    assert len(equalities) == 5


def test_expand_hand_targets_with_followers() -> None:
    expanded = expand_targets_with_mjcf_equalities(
        {"L6V1_L_hand_index_mcp_pitch": 0.4, "L6V1_L_hand_thumb_cmc_pitch": 0.5},
        DEFAULT_AR5_L6_MJCF,
    )
    assert np.isclose(expanded["L6V1_L_hand_index_dip"], 0.4 * 1.125676)
    assert np.isclose(expanded["L6V1_L_hand_thumb_dip"], 0.5 * 1.226495)


def test_resolve_follower_controls() -> None:
    dof_names = [
        "L6V1_L_hand_index_mcp_pitch",
        "L6V1_L_hand_index_dip",
        "L6V1_L_hand_thumb_cmc_pitch",
        "L6V1_L_hand_thumb_dip",
    ]
    controls = resolve_mimic_follower_controls(dof_names, DEFAULT_AR5_L6_MJCF)
    follower_names = {control.dependent_joint for control in controls}
    assert follower_names == {"L6V1_L_hand_index_dip", "L6V1_L_hand_thumb_dip"}
    assert {control.master_index for control in controls} == {0, 2}


def test_polycoef_mimic_position_and_velocity() -> None:
    equality = MjcfJointEquality(
        name="nonlinear",
        dependent_joint="follower",
        master_joint="master",
        polycoef=(0.2, 1.5, -0.25, 0.1),
    )
    master_position = 0.4
    master_velocity = 0.2
    expected_position = (
        0.2
        + 1.5 * master_position
        - 0.25 * master_position**2
        + 0.1 * master_position**3
    )
    expected_velocity = (
        1.5 - 0.5 * master_position + 0.3 * master_position**2
    ) * master_velocity
    assert np.isclose(equality.evaluate_position(master_position), expected_position)
    assert np.isclose(
        equality.evaluate_velocity(master_position, master_velocity), expected_velocity
    )


def test_follower_mapper_uses_actual_master_state() -> None:
    mapper = MimicFollowerTargetMapper(["master", "follower"], None)
    mapper.controls = [
        MimicFollowerControl(
            dependent_joint="follower",
            master_joint="master",
            dependent_index=1,
            master_index=0,
            polycoef=(0.1, 2.0, 0.5),
        )
    ]
    target_positions = np.asarray([0.9, 0.9])
    target_velocities = np.asarray([0.8, 0.8])
    actual_positions = np.asarray([0.3, 0.0])
    actual_velocities = np.asarray([0.4, 0.0])
    mapper.apply_from_actual(
        target_positions, target_velocities, actual_positions, actual_velocities
    )
    np.testing.assert_allclose(target_positions, [0.9, 0.1 + 2.0 * 0.3 + 0.5 * 0.3**2])
    np.testing.assert_allclose(target_velocities, [0.8, (2.0 + 2.0 * 0.5 * 0.3) * 0.4])


def test_follower_joint_name_set() -> None:
    names = mjcf_equality_follower_joint_names(DEFAULT_AR5_L6_MJCF)
    assert {
        "L6V1_L_hand_index_dip",
        "L6V1_L_hand_middle_dip",
        "L6V1_L_hand_ring_dip",
        "L6V1_L_hand_pinky_dip",
        "L6V1_L_hand_thumb_dip",
    } <= names
