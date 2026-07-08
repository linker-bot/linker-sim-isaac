from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.tiled.trajectory import TiledTrajectoryBuffer, TiledTrajectoryOverlay


def test_trajectory_buffer_broadcasts_single_trajectory_to_selected_envs() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=3)

    loaded = buffer.load(
        robot_name="left",
        env_ids=[0, 2],
        times=[0.0, 1.0],
        positions=[[0.0, 0.0], [1.0, 2.0]],
        joint_names=["j1", "j2"],
    )
    result = buffer.step(
        robot_name="left",
        current_positions=np.zeros((3, 2)),
        dt_s=0.5,
    )

    assert loaded == (0, 2)
    np.testing.assert_allclose(
        result.joint_positions,
        [[0.5, 1.0], [0.0, 0.0], [0.5, 1.0]],
    )
    assert result.active_env_ids == (0, 2)
    assert result.idle_env_ids == (1,)


def test_trajectory_buffer_accepts_per_env_trajectories_and_holds_unselected() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=2)
    buffer.load(
        robot_name="right",
        env_ids=[0, 1],
        times=[0.0, 1.0],
        positions=[
            [[0.0], [1.0]],
            [[0.0], [3.0]],
        ],
        joint_names=["j1"],
    )

    result = buffer.step(
        robot_name="right",
        current_positions=np.asarray([[10.0], [20.0]]),
        dt_s=0.25,
        env_ids=[1],
    )

    np.testing.assert_allclose(result.joint_positions, [[10.0], [0.75]])
    assert result.active_env_ids == (1,)
    assert result.idle_env_ids == ()


def test_trajectory_buffer_marks_completed_and_holds_final_sample() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=1)
    buffer.load(
        robot_name="arm",
        env_ids=[0],
        times=[0.0, 0.5],
        positions=[[0.0, 0.0], [1.0, -1.0]],
        joint_names=["j1", "j2"],
    )

    first = buffer.step(
        robot_name="arm",
        current_positions=np.zeros((1, 2)),
        dt_s=1.0,
    )
    second = buffer.step(
        robot_name="arm",
        current_positions=np.asarray([[9.0, 9.0]]),
        dt_s=1.0,
    )

    assert first.completed_env_ids == (0,)
    np.testing.assert_allclose(first.joint_positions, [[1.0, -1.0]])
    assert second.completed_env_ids == (0,)
    np.testing.assert_allclose(second.joint_positions, [[1.0, -1.0]])


def test_trajectory_buffer_status_and_clear_are_robot_scoped() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=2)
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 1.0],
        positions=[[0.0], [1.0]],
        joint_names=["j1"],
        request_id="req-left",
        source="planner",
    )
    buffer.load(
        robot_name="right",
        env_ids=[1],
        times=[0.0, 1.0],
        positions=[[0.0], [2.0]],
        joint_names=["j1"],
    )

    status = buffer.status(robot_name="left")
    cleared = buffer.clear(robot_name="left", env_ids=[0])

    assert status["robots"]["left"]["active_env_ids"] == [0]
    assert status["robots"]["left"]["envs"][0]["request_id"] == "req-left"
    assert status["robots"]["left"]["envs"][0]["source"] == "planner"
    assert cleared == {"left": [0]}
    assert buffer.status()["robots"]["right"]["active_env_ids"] == [1]


def test_trajectory_buffer_rejects_active_replace_when_disabled() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=1)
    buffer.load(
        robot_name="arm",
        env_ids=[0],
        times=[0.0, 1.0],
        positions=[[0.0], [1.0]],
        joint_names=["j1"],
    )

    with pytest.raises(ValueError, match="still active"):
        buffer.load(
            robot_name="arm",
            env_ids=[0],
            times=[0.0, 1.0],
            positions=[[0.0], [2.0]],
            joint_names=["j1"],
            replace=False,
        )


def test_trajectory_buffer_applies_sync_overlay_to_selected_envs() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=2)
    buffer.load(
        robot_name="left",
        env_ids=[1],
        times=[0.0, 1.0],
        positions=[[0.0, 10.0, 20.0], [1.0, 11.0, 21.0]],
        joint_names=["arm", "hand_a", "hand_b"],
        overlays=(
            TiledTrajectoryOverlay(
                joint_indices=(1, 2),
                start_positions=np.asarray([[5.0, 6.0]]),
                target_positions=np.asarray([[7.0, 10.0]]),
            ),
        ),
    )

    result = buffer.step(
        robot_name="left",
        current_positions=np.asarray([[100.0, 100.0, 100.0], [0.0, 5.0, 6.0]]),
        dt_s=0.5,
    )

    np.testing.assert_allclose(result.joint_positions[0], [100.0, 100.0, 100.0])
    np.testing.assert_allclose(result.joint_positions[1], [0.5, 6.0, 8.0])
    status = buffer.status(robot_name="left")
    assert status["robots"]["left"]["envs"][0]["overlay_joint_names"] == [
        "hand_a",
        "hand_b",
    ]


def test_trajectory_buffer_overlay_duration_reaches_target_early() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=1)
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 1.0],
        positions=[[0.0, 5.0], [1.0, 6.0]],
        joint_names=["arm", "hand"],
        overlays=(
            TiledTrajectoryOverlay(
                joint_indices=(1,),
                start_positions=np.asarray([[2.0]]),
                target_positions=np.asarray([[8.0]]),
                duration_s=0.25,
            ),
        ),
    )

    result = buffer.step(
        robot_name="left",
        current_positions=np.asarray([[0.0, 2.0]]),
        dt_s=0.5,
    )

    np.testing.assert_allclose(result.joint_positions, [[0.5, 8.0]])


def test_trajectory_buffer_runs_before_sync_after_overlay_sequence() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=1)
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 0.1],
        positions=[[0.0, 0.0], [1.0, 0.0]],
        joint_names=["arm", "hand"],
        overlays=(
            TiledTrajectoryOverlay(
                joint_indices=(1,),
                start_positions=np.asarray([[0.0]]),
                target_positions=np.asarray([[0.2]]),
                duration_s=0.1,
                timing="before",
            ),
            TiledTrajectoryOverlay(
                joint_indices=(1,),
                start_positions=np.asarray([[0.0]]),
                target_positions=np.asarray([[0.8]]),
                duration_s=0.1,
                timing="sync",
            ),
            TiledTrajectoryOverlay(
                joint_indices=(1,),
                start_positions=np.asarray([[0.0]]),
                target_positions=np.asarray([[0.0]]),
                duration_s=0.1,
                timing="after",
            ),
        ),
    )

    before = buffer.step(
        robot_name="left",
        current_positions=np.asarray([[0.0, 0.0]]),
        dt_s=0.1,
    )
    main = buffer.step(
        robot_name="left",
        current_positions=before.joint_positions,
        dt_s=0.1,
    )
    after = buffer.step(
        robot_name="left",
        current_positions=main.joint_positions,
        dt_s=0.1,
    )

    np.testing.assert_allclose(before.joint_positions, [[0.0, 0.2]])
    np.testing.assert_allclose(main.joint_positions, [[1.0, 0.8]])
    np.testing.assert_allclose(after.joint_positions, [[1.0, 0.0]])


def test_trajectory_buffer_appends_hand_only_playback() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=1)
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 0.1],
        positions=[[0.0, 0.2], [1.0, 0.2]],
        joint_names=["arm", "hand"],
    )
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 0.1],
        positions=[[1.0, 0.2], [1.0, 0.2]],
        joint_names=["arm", "hand"],
        overlays=(
            TiledTrajectoryOverlay(
                joint_indices=(1,),
                start_positions=np.asarray([[0.2]]),
                target_positions=np.asarray([[0.8]]),
                duration_s=0.1,
            ),
        ),
        replace=False,
        append=True,
    )

    main = buffer.step(
        robot_name="left",
        current_positions=np.asarray([[0.0, 0.2]]),
        dt_s=0.1,
    )
    hand = buffer.step(
        robot_name="left",
        current_positions=main.joint_positions,
        dt_s=0.1,
    )

    np.testing.assert_allclose(main.joint_positions, [[1.0, 0.2]])
    np.testing.assert_allclose(hand.joint_positions, [[1.0, 0.8]])
