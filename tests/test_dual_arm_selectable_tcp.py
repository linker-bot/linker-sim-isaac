from __future__ import annotations

import numpy as np

from linkerbot_sim.planning.dual_arm_cspace_partition import (
    DualArmJointPartitions,
    selected_side_goal,
    split_dual_arm_trajectory_to_commands,
)
from linkerbot_sim.trajectories.types import JointTrajectory


def test_selected_side_goal_writes_only_selected_side() -> None:
    partitions = DualArmJointPartitions.from_joint_names(
        ("l1", "l2", "r1", "r2"),
        left_joint_names=("l1", "l2"),
        right_joint_names=("r1", "r2"),
    )
    base = np.asarray([0.0, 0.0, 10.0, 20.0])
    solved = np.asarray([1.0, 2.0, 30.0, 40.0])

    left_goal = selected_side_goal(
        base_q=base,
        solved_q=solved,
        partitions=partitions,
        active_side="left",
    )

    np.testing.assert_allclose(left_goal, [1.0, 2.0, 10.0, 20.0])


def test_selected_side_goal_can_update_right_after_left() -> None:
    partitions = DualArmJointPartitions.from_joint_names(
        ("l1", "l2", "r1", "r2"),
        left_joint_names=("l1", "l2"),
        right_joint_names=("r1", "r2"),
    )
    left_goal = np.asarray([1.0, 2.0, 10.0, 20.0])
    solved = np.asarray([100.0, 200.0, 30.0, 40.0])

    right_goal = selected_side_goal(
        base_q=left_goal,
        solved_q=solved,
        partitions=partitions,
        active_side="right",
    )

    np.testing.assert_allclose(right_goal, [1.0, 2.0, 30.0, 40.0])


def test_split_dual_arm_trajectory_to_left_and_right_commands() -> None:
    joint_names = ("l1", "l2", "r1", "r2")
    partitions = DualArmJointPartitions.from_joint_names(
        joint_names,
        left_joint_names=("l1", "l2"),
        right_joint_names=("r1", "r2"),
    )
    dual = JointTrajectory.from_samples(
        times=np.asarray([0.1, 0.2]),
        positions=np.asarray([[1.0, 2.0, 30.0, 40.0], [3.0, 4.0, 50.0, 60.0]]),
        velocities=np.ones((2, 4)),
        joint_names=joint_names,
        phases=("dual", "dual"),
    )

    left, right = split_dual_arm_trajectory_to_commands(
        dual_arm_trajectory=dual,
        partitions=partitions,
        left_command_joint_names=("l1", "l2", "lh"),
        right_command_joint_names=("r1", "r2", "rh"),
        left_start_command=np.asarray([0.0, 0.0, 0.5]),
        right_start_command=np.asarray([10.0, 20.0, 0.6]),
        left_target_command=np.asarray([3.0, 4.0, 0.8]),
        right_target_command=np.asarray([50.0, 60.0, 0.9]),
        phase="split",
    )

    np.testing.assert_allclose(left.positions[:, :2], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(right.positions[:, :2], [[30.0, 40.0], [50.0, 60.0]])
    np.testing.assert_allclose(left.positions[-1], [3.0, 4.0, 0.8])
    np.testing.assert_allclose(right.positions[-1], [50.0, 60.0, 0.9])
    assert left.joint_names == ("l1", "l2", "lh")
    assert right.joint_names == ("r1", "r2", "rh")
