from __future__ import annotations

import numpy as np

from linkerbot_sim.tiled.planning.batching import (
    planning_batch_key,
    planning_batch_layout,
    request_problem_count,
)
from linkerbot_sim.tiled.planning.types import (
    TiledPlanningRequest,
    TiledPlanningSegment,
)


def _joint_request(
    *,
    request_id: str,
    env_ids: tuple[int, ...] = (0, 1),
    avoid_collisions: bool = False,
) -> TiledPlanningRequest:
    rows = len(env_ids)
    return TiledPlanningRequest(
        request_id=request_id,
        robot_name="arm",
        env_ids=env_ids,
        current_positions=np.zeros((rows, 2)),
        goal_positions=np.ones((rows, 2)),
        joint_names=("j0", "j1"),
        duration_s=1.0,
        sample_dt_s=0.05,
        avoid_collisions=avoid_collisions,
    )


def test_problem_count_uses_request_env_rows() -> None:
    assert request_problem_count(_joint_request(request_id="a")) == 2


def test_joint_requests_with_same_shape_share_batch_key() -> None:
    first = _joint_request(request_id="a", env_ids=(0,))
    second = _joint_request(request_id="b", env_ids=(3, 4, 5))

    assert planning_batch_key(first) == planning_batch_key(second)


def test_batch_layout_preserves_fifo_requests_and_contiguous_row_slices() -> None:
    first = _joint_request(request_id="a", env_ids=(10,))
    second = _joint_request(request_id="b", env_ids=(20, 21, 22))

    layout = planning_batch_layout((first, second))

    assert layout is not None
    assert layout.requests == (first, second)
    assert layout.row_slices == (slice(0, 1), slice(1, 4))
    assert layout.problem_count == 4


def test_collision_policy_is_part_of_batch_key() -> None:
    plain = _joint_request(request_id="plain")
    collision = _joint_request(request_id="collision", avoid_collisions=True)

    assert planning_batch_key(plain) != planning_batch_key(collision)


def test_task_space_segment_is_not_manager_batched() -> None:
    request = TiledPlanningRequest(
        request_id="path",
        robot_name="arm",
        env_ids=(0,),
        current_positions=np.zeros((1, 2)),
        joint_names=("j0", "j1"),
        sample_dt_s=0.02,
        segments=(TiledPlanningSegment(kind="linear_pose_path", path=object()),),
    )

    assert planning_batch_key(request) is None
    assert planning_batch_layout((request,)) is None
