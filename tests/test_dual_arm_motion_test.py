from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from dual_arm_motion_test import (  # noqa: E402
    _phase_durations,
    arm_reach_target,
    hand_target_command,
)
from linkerbot_sim.assets.asset_paths import (  # noqa: E402
    DEFAULT_CONTROLLED_AR5_L6_JOINT_NAMES,
)


def _side_command_joints(side: str) -> tuple[str, ...]:
    if side == "left":
        return DEFAULT_CONTROLLED_AR5_L6_JOINT_NAMES
    if side == "right":
        return tuple(
            str(name).replace("AR5V2_L_", "AR5V2_R_").replace("L6V1_L_", "L6V1_R_")
            for name in DEFAULT_CONTROLLED_AR5_L6_JOINT_NAMES
        )
    raise ValueError(f"unsupported side: {side}")


def test_arm_reach_target_moves_arm_joints_side_aware() -> None:
    left_joints = _side_command_joints("left")
    right_joints = _side_command_joints("right")
    left_start = np.linspace(-0.2, 0.2, len(left_joints), dtype=float)
    right_start = np.linspace(0.3, -0.3, len(right_joints), dtype=float)

    left_target = arm_reach_target(left_start, left_joints, "left")
    right_target = arm_reach_target(right_start, right_joints, "right")

    left_arm_indices = [
        index for index, name in enumerate(left_joints) if "_arm_joint_" in name
    ]
    right_arm_indices = [
        index for index, name in enumerate(right_joints) if "_arm_joint_" in name
    ]
    expected_deltas = np.asarray(
        [0.08, -0.06, 0.05, -0.04, 0.035, -0.025, 0.02],
        dtype=float,
    )
    np.testing.assert_allclose(
        left_target[left_arm_indices] - left_start[left_arm_indices], expected_deltas
    )
    np.testing.assert_allclose(
        right_target[right_arm_indices] - right_start[right_arm_indices],
        -expected_deltas,
    )

    left_hand_indices = [
        index for index, name in enumerate(left_joints) if "_arm_joint_" not in name
    ]
    right_hand_indices = [
        index for index, name in enumerate(right_joints) if "_arm_joint_" not in name
    ]
    np.testing.assert_allclose(left_target[left_hand_indices], left_start[left_hand_indices])
    np.testing.assert_allclose(
        right_target[right_hand_indices], right_start[right_hand_indices]
    )


def test_hand_target_command_applies_side_aware_pinch_targets() -> None:
    left_joints = _side_command_joints("left")
    right_joints = _side_command_joints("right")
    left_base = np.linspace(-0.4, 0.4, len(left_joints), dtype=float)
    right_base = np.linspace(0.4, -0.4, len(right_joints), dtype=float)

    left_pre = hand_target_command(left_base, left_joints, "left", closed=False)
    right_closed = hand_target_command(right_base, right_joints, "right", closed=True)

    assert left_pre[left_joints.index("L6V1_L_hand_thumb_cmc_roll")] == 0.95
    assert left_pre[left_joints.index("L6V1_L_hand_index_mcp_pitch")] == 0.25
    assert right_closed[right_joints.index("L6V1_R_hand_thumb_cmc_pitch")] == 0.7
    assert right_closed[right_joints.index("L6V1_R_hand_index_mcp_pitch")] == 0.85

    left_arm_indices = [
        index for index, name in enumerate(left_joints) if "_arm_joint_" in name
    ]
    right_arm_indices = [
        index for index, name in enumerate(right_joints) if "_arm_joint_" in name
    ]
    np.testing.assert_allclose(left_pre[left_arm_indices], left_base[left_arm_indices])
    np.testing.assert_allclose(
        right_closed[right_arm_indices], right_base[right_arm_indices]
    )


def test_phase_durations_short_smoke_are_positive_and_shorter() -> None:
    normal = _phase_durations(False)
    short = _phase_durations(True)

    assert set(normal) == {"hand", "arm", "return"}
    assert set(short) == set(normal)
    for phase, duration in short.items():
        assert 0.0 < duration < normal[phase]
