from __future__ import annotations

import numpy as np

from manipulation_project.assets.asset_paths import DEFAULT_AR5_L6_MJCF
from manipulation_project.robots.mimic import (
    MimicFollowerControl,
    MjcfJointEquality,
    expand_targets_with_mjcf_equalities,
    follower_targets_from_masters,
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
    dof_names = ["L6V1_L_hand_index_mcp_pitch", "L6V1_L_hand_index_dip", "L6V1_L_hand_thumb_cmc_pitch", "L6V1_L_hand_thumb_dip"]
    command_indices = np.asarray([0, 2], dtype=int)
    controls = resolve_mimic_follower_controls(dof_names, DEFAULT_AR5_L6_MJCF, command_indices)
    follower_names = {control.dependent_joint for control in controls}
    assert follower_names == {"L6V1_L_hand_index_dip", "L6V1_L_hand_thumb_dip"}
    positions, velocities = follower_targets_from_masters(np.asarray([0.4, 0.5]), np.asarray([0.1, 0.2]), controls)
    assert positions.shape == (2,)
    assert velocities.shape == (2,)


def test_polycoef_mimic_position_and_velocity() -> None:
    equality = MjcfJointEquality(
        name="nonlinear",
        dependent_joint="follower",
        master_joint="master",
        polycoef=(0.2, 1.5, -0.25, 0.1),
    )
    master_position = 0.4
    master_velocity = 0.2
    expected_position = 0.2 + 1.5 * master_position - 0.25 * master_position**2 + 0.1 * master_position**3
    expected_velocity = (1.5 - 0.5 * master_position + 0.3 * master_position**2) * master_velocity
    assert np.isclose(equality.evaluate_position(master_position), expected_position)
    assert np.isclose(equality.evaluate_velocity(master_position, master_velocity), expected_velocity)


def test_follower_targets_support_nonlinear_polycoef() -> None:
    controls = [
        MimicFollowerControl(
            dependent_joint="follower",
            master_joint="master",
            dependent_index=1,
            master_slot=0,
            polycoef=(0.1, 2.0, 0.5),
        )
    ]
    positions, velocities = follower_targets_from_masters(np.asarray([0.3]), np.asarray([0.4]), controls)
    np.testing.assert_allclose(positions, [0.1 + 2.0 * 0.3 + 0.5 * 0.3**2])
    np.testing.assert_allclose(velocities, [(2.0 + 2.0 * 0.5 * 0.3) * 0.4])


def test_follower_joint_name_set() -> None:
    names = mjcf_equality_follower_joint_names(DEFAULT_AR5_L6_MJCF)
    assert {"L6V1_L_hand_index_dip", "L6V1_L_hand_middle_dip", "L6V1_L_hand_ring_dip", "L6V1_L_hand_pinky_dip", "L6V1_L_hand_thumb_dip"} <= names
