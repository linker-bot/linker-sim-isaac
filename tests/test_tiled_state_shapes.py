from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.tiled.state import (
    TiledObjectState,
    TiledRobotJointState,
    TiledState,
    broadcast_rows,
)


def test_robot_joint_state_validates_shapes() -> None:
    state = TiledRobotJointState(
        joint_names=("j0", "j1"),
        positions=np.zeros((3, 2)),
        velocities=np.ones((3, 2)),
    )

    assert state.positions.shape == (3, 2)


def test_robot_joint_state_rejects_mismatched_joint_names() -> None:
    with pytest.raises(ValueError, match="joint_names"):
        TiledRobotJointState(
            joint_names=("j0",),
            positions=np.zeros((3, 2)),
            velocities=np.ones((3, 2)),
        )


def test_object_state_from_world_computes_local_positions() -> None:
    state = TiledObjectState.from_world(
        name="block",
        positions_world=np.asarray([[1.0, 2.0, 3.0], [4.0, 2.0, 3.0]]),
        orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        env_origins=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(state.positions_local, [[1.0, 2.0, 3.0]] * 2)


def test_tiled_state_rejects_negative_step() -> None:
    with pytest.raises(ValueError, match="step"):
        TiledState(step=-1, time_s=0.0, robots={})


def test_broadcast_rows_repeats_single_row() -> None:
    np.testing.assert_allclose(
        broadcast_rows(np.asarray([1.0, 2.0]), 3, label="x"),
        [[1.0, 2.0]] * 3,
    )
