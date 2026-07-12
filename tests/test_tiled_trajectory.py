from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.tiled.playback.buffer import TiledTrajectoryBuffer
from linkerbot_sim.tiled.playback.models import PlaybackJointTrack


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


def test_trajectory_buffer_applies_sync_joint_track_to_selected_envs() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=2)
    buffer.load(
        robot_name="left",
        env_ids=[1],
        times=[0.0, 1.0],
        positions=[[0.0, 10.0, 20.0], [1.0, 11.0, 21.0]],
        joint_names=["arm", "hand_a", "hand_b"],
        joint_tracks=(
            PlaybackJointTrack(
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
    assert status["robots"]["left"]["envs"][0]["joint_track_names"] == [
        "hand_a",
        "hand_b",
    ]


def test_trajectory_buffer_joint_track_duration_reaches_target_early() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=1)
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 1.0],
        positions=[[0.0, 5.0], [1.0, 6.0]],
        joint_names=["arm", "hand"],
        joint_tracks=(
            PlaybackJointTrack(
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


def test_trajectory_buffer_runs_before_sync_after_joint_track_sequence() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=1)
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 0.1],
        positions=[[0.0, 0.0], [1.0, 0.0]],
        joint_names=["arm", "hand"],
        joint_tracks=(
            PlaybackJointTrack(
                joint_indices=(1,),
                start_positions=np.asarray([[0.0]]),
                target_positions=np.asarray([[0.2]]),
                duration_s=0.1,
                timing="before",
            ),
            PlaybackJointTrack(
                joint_indices=(1,),
                start_positions=np.asarray([[0.0]]),
                target_positions=np.asarray([[0.8]]),
                duration_s=0.1,
                timing="sync",
            ),
            PlaybackJointTrack(
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
        joint_tracks=(
            PlaybackJointTrack(
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


def test_trajectory_buffer_rejects_append_over_depth_without_mutating_queue() -> None:
    buffer = TiledTrajectoryBuffer(
        num_envs=1,
        max_queue_depth_per_env=1,
    )
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 0.1],
        positions=[[0.0], [1.0]],
        joint_names=["j1"],
        request_id="existing",
    )

    with pytest.raises(ValueError, match="trajectories=2>1"):
        buffer.load(
            robot_name="left",
            env_ids=[0],
            times=[0.0, 0.1],
            positions=[[1.0], [2.0]],
            joint_names=["j1"],
            request_id="rejected",
            append=True,
        )

    status = buffer.status(robot_name="left")
    assert status["rejected_loads"] == 1
    assert status["queued_trajectories"] == 1
    assert status["queued_samples"] == 2
    assert status["queued_duration_s"] == pytest.approx(0.1)
    assert status["limits"] == {
        "max_queue_depth_per_env": 1,
        "max_samples_per_env": 100_000,
        "max_duration_s_per_env": 3600.0,
        "overflow_policy": "reject",
    }
    env_status = status["robots"]["left"]["envs"][0]
    assert env_status["request_id"] == "existing"
    assert env_status["queued_trajectories"] == 1


def test_trajectory_buffer_sample_rejection_is_atomic_across_selected_envs() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=2, max_samples_per_env=2)
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 0.1],
        positions=[[0.0], [1.0]],
        joint_names=["j1"],
        request_id="existing",
    )

    with pytest.raises(ValueError, match="samples=3>2"):
        buffer.load(
            robot_name="left",
            env_ids=[0, 1],
            times=[0.0, 0.1, 0.2],
            positions=[[0.0], [1.0], [2.0]],
            joint_names=["j1"],
            request_id="rejected",
        )

    status = buffer.status(robot_name="left")
    assert status["rejected_loads"] == 1
    assert status["robots"]["left"]["count"] == 1
    assert status["robots"]["left"]["envs"][0]["request_id"] == "existing"


def test_trajectory_buffer_enforces_total_queued_duration() -> None:
    buffer = TiledTrajectoryBuffer(
        num_envs=1,
        max_queue_depth_per_env=3,
        max_duration_s_per_env=0.2,
    )
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 0.1],
        positions=[[0.0], [1.0]],
        joint_names=["j1"],
    )
    buffer.load(
        robot_name="left",
        env_ids=[0],
        times=[0.0, 0.1],
        positions=[[1.0], [2.0]],
        joint_names=["j1"],
        append=True,
    )

    with pytest.raises(ValueError, match="duration_s=0.3>0.2"):
        buffer.load(
            robot_name="left",
            env_ids=[0],
            times=[0.0, 0.1],
            positions=[[2.0], [3.0]],
            joint_names=["j1"],
            append=True,
        )

    status = buffer.status(robot_name="left")
    assert status["queued_trajectories"] == 2
    assert status["queued_samples"] == 4
    assert status["queued_duration_s"] == pytest.approx(0.2)
    assert status["robots"]["left"]["rejected_loads"] == 1


def test_trajectory_buffer_rejected_loads_are_robot_scoped() -> None:
    buffer = TiledTrajectoryBuffer(num_envs=1, max_queue_depth_per_env=1)
    for robot in ("left", "right"):
        buffer.load(
            robot_name=robot,
            env_ids=[0],
            times=[0.0, 0.1],
            positions=[[0.0], [1.0]],
            joint_names=["j1"],
        )
    with pytest.raises(ValueError, match="capacity exceeded"):
        buffer.load(
            robot_name="right",
            env_ids=[0],
            times=[0.0, 0.1],
            positions=[[1.0], [2.0]],
            joint_names=["j1"],
            append=True,
        )

    left = buffer.status(robot_name="left")
    right = buffer.status(robot_name="right")
    combined = buffer.status()

    assert left["rejected_loads"] == 0
    assert left["rejected_loads_scope"] == "robot"
    assert left["robots"]["left"]["rejected_loads"] == 0
    assert right["rejected_loads"] == 1
    assert right["robots"]["right"]["rejected_loads"] == 1
    assert combined["rejected_loads"] == 1
    assert combined["rejected_loads_scope"] == "buffer"


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_playback_rejects_non_finite_command_inputs(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        PlaybackJointTrack(
            joint_indices=(0,),
            start_positions=np.asarray([0.0]),
            target_positions=np.asarray([1.0]),
            duration_s=value,
        )

    buffer = TiledTrajectoryBuffer(num_envs=1)
    with pytest.raises(ValueError, match="finite"):
        buffer.step(
            robot_name="robot",
            current_positions=np.asarray([[0.0]]),
            dt_s=value,
        )
    with pytest.raises(ValueError, match="finite"):
        buffer.step(
            robot_name="robot",
            current_positions=np.asarray([[value]]),
            dt_s=0.1,
        )
