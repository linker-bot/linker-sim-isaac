from __future__ import annotations

import time

import numpy as np
import pytest

from linkerbot_sim.app.interactive.tiled_scene.plan_messages import (
    planning_request_from_message,
)
from linkerbot_sim.configs.runtime import (
    PlannerRequestDefaults,
    RuntimeCommandDefaults,
)
from linkerbot_sim.planning.requests import (
    LinearPosePathRequest,
    TaskSpacePath,
    TcpLineSegment,
)
from linkerbot_sim.planning.results import MotionResult
from linkerbot_sim.tiled.planning.linear_backend import LinearJointPlannerBackend
from linkerbot_sim.tiled.planning.manager import TiledPlannerManager
from linkerbot_sim.tiled.planning.backends.curobo import (
    TiledCuroboPlanningBackend,
)
from linkerbot_sim.tiled.planning.types import (
    TiledPlanningSegment,
    TiledPlanningRequest,
    TiledPlanningResult,
)
from linkerbot_sim.trajectories.types import JointTrajectory


def _request(
    request_id: str = "req-1", robot_name: str = "left"
) -> TiledPlanningRequest:
    return TiledPlanningRequest(
        request_id=request_id,
        robot_name=robot_name,
        env_ids=(0, 1),
        current_positions=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        goal_positions=np.asarray([[1.0, 2.0], [3.0, 5.0]]),
        joint_names=("j1", "j2"),
        duration_s=0.1,
        sample_dt_s=0.05,
    )


def test_linear_joint_planner_returns_batched_trajectory() -> None:
    result = LinearJointPlannerBackend().plan(_request())

    assert result.success is True
    assert result.status == "SUCCESS"
    np.testing.assert_allclose(result.times, [0.0, 0.05, 0.1])
    assert result.positions.shape == (2, 3, 2)
    np.testing.assert_allclose(result.positions[0, -1], [1.0, 2.0])
    np.testing.assert_allclose(result.positions[1, -1], [3.0, 5.0])


def test_linear_joint_planner_uses_canonical_partial_sample_grid() -> None:
    request = _request()
    request = TiledPlanningRequest(
        request_id=request.request_id,
        robot_name=request.robot_name,
        env_ids=request.env_ids,
        current_positions=request.current_positions,
        goal_positions=request.goal_positions,
        joint_names=request.joint_names,
        duration_s=0.11,
        sample_dt_s=0.05,
    )

    result = LinearJointPlannerBackend().plan(request)

    np.testing.assert_allclose(result.times, [0.0, 0.05, 0.1, 0.11])


def test_linear_joint_planner_concatenates_joint_segments() -> None:
    request = TiledPlanningRequest(
        request_id="queue",
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[0.0, 0.0]]),
        joint_names=("j1", "j2"),
        duration_s=0.1,
        sample_dt_s=0.05,
        segments=(
            TiledPlanningSegment(
                kind="joint_delta_pos",
                goal_positions=np.asarray([[1.0, 0.0]]),
                duration_s=0.1,
                sample_dt_s=0.05,
            ),
            TiledPlanningSegment(
                kind="joint_position_target",
                goal_positions=np.asarray([[1.0, 2.0]]),
                duration_s=0.05,
                sample_dt_s=0.05,
            ),
        ),
    )

    result = LinearJointPlannerBackend().plan(request)

    assert result.success is True
    np.testing.assert_allclose(result.times, [0.0, 0.05, 0.1, 0.15])
    np.testing.assert_allclose(result.positions[0, -1], [1.0, 2.0])


def test_planner_manager_collects_ready_results() -> None:
    manager = TiledPlannerManager(max_workers=1)
    try:
        request_id = manager.submit(_request())
        ready = manager.collect_ready(timeout_s=1.0)

        assert request_id == "req-1"
        assert len(ready) == 1
        assert ready[0].request_id == "req-1"
        assert ready[0].success is True
        assert manager.status()["completed"][0]["request_id"] == "req-1"
    finally:
        manager.shutdown()


def test_planner_manager_status_does_not_dispatch_or_consume_results() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.request_ids: list[str] = []

        def plan(self, request):
            self.request_ids.append(request.request_id)
            return LinearJointPlannerBackend().plan(request)

    backend = RecordingBackend()
    manager = TiledPlannerManager(backend=backend, max_workers=1)
    try:
        manager.submit(_request())

        status = manager.status()

        assert status["queued_request_ids"] == ["req-1"]
        assert status["completed"] == []
        assert backend.request_ids == []

        ready = manager.collect_ready(timeout_s=1.0)
        assert [result.request_id for result in ready] == ["req-1"]
        assert backend.request_ids == ["req-1"]
    finally:
        manager.shutdown()


def test_planner_manager_uses_backend_plan_many_for_compatible_requests() -> None:
    class BatchBackend:
        def __init__(self) -> None:
            self.batches = []

        def plan_many(self, requests):
            self.batches.append(tuple(request.request_id for request in requests))
            return tuple(
                LinearJointPlannerBackend().plan(request) for request in requests
            )

        def plan(self, request):  # pragma: no cover - 本测试必须走 plan_many
            raise AssertionError("compatible requests should use plan_many")

    backend = BatchBackend()
    manager = TiledPlannerManager(
        backend=backend,
        max_workers=1,
        max_batch_problems=8,
    )
    try:
        manager.submit(_request("first"))
        manager.submit(_request("second"))
        manager.submit(_request("third"))

        ready = manager.collect_ready(timeout_s=1.0)

        assert [result.request_id for result in ready] == [
            "first",
            "second",
            "third",
        ]
        assert backend.batches == [("first", "second", "third")]
    finally:
        manager.shutdown()


def test_planner_manager_chunks_plan_many_by_problem_count() -> None:
    class BatchBackend:
        def __init__(self) -> None:
            self.batches = []

        def plan_many(self, requests):
            self.batches.append(tuple(request.request_id for request in requests))
            return tuple(
                LinearJointPlannerBackend().plan(request) for request in requests
            )

        def plan(self, request):  # pragma: no cover - 本测试必须走 plan_many
            raise AssertionError("compatible requests should use plan_many")

    backend = BatchBackend()
    manager = TiledPlannerManager(
        backend=backend,
        max_workers=1,
        max_batch_problems=4,
    )
    try:
        manager.submit(_request("first"))
        manager.submit(_request("second"))
        manager.submit(_request("third"))

        ready = []
        while len(ready) < 3:
            ready.extend(manager.collect_ready(timeout_s=1.0))

        assert [result.request_id for result in ready] == [
            "first",
            "second",
            "third",
        ]
        assert backend.batches == [("first", "second"), ("third",)]
    finally:
        manager.shutdown()


def test_planner_manager_splits_single_oversized_request_without_backend_bypass() -> (
    None
):
    class BoundedBackend:
        def __init__(self) -> None:
            self.problem_counts: list[int] = []

        def plan_many(self, requests):
            count = sum(len(request.env_ids) for request in requests)
            self.problem_counts.append(count)
            assert count <= 2
            return tuple(
                LinearJointPlannerBackend().plan(request) for request in requests
            )

    current = np.arange(10, dtype=float).reshape(5, 2)
    goal = current + 1.0
    request = TiledPlanningRequest(
        request_id="oversized",
        robot_name="left",
        env_ids=(0, 1, 2, 3, 4),
        current_positions=current,
        joint_names=("j1", "j2"),
        segments=(
            TiledPlanningSegment(
                kind="joint_position_target",
                duration_s=0.1,
                sample_dt_s=0.05,
                goal_positions=goal,
            ),
        ),
        duration_s=0.1,
        sample_dt_s=0.05,
    )
    backend = BoundedBackend()
    manager = TiledPlannerManager(
        backend=backend,
        max_workers=1,
        max_batch_problems=2,
        oversize_request_policy="split",
    )
    try:
        manager.submit(request)
        ready = manager.collect_ready(timeout_s=1.0)

        assert backend.problem_counts == [2, 2, 1]
        assert len(ready) == 1
        assert ready[0].request_id == "oversized"
        assert ready[0].env_ids == (0, 1, 2, 3, 4)
        np.testing.assert_allclose(ready[0].positions[:, -1, :], goal)
        assert manager.status()["split_requests"] == 1
    finally:
        manager.shutdown()


def test_planner_manager_split_failure_is_request_atomic() -> None:
    class FailingChunkBackend:
        def plan_many(self, requests):
            request = requests[0]
            if request.env_ids == (2, 3):
                return (
                    TiledPlanningResult.failed(
                        request, status="NO_PATH", message="chunk failed"
                    ),
                )
            return (LinearJointPlannerBackend().plan(request),)

    current = np.zeros((5, 2), dtype=float)
    request = TiledPlanningRequest(
        request_id="atomic",
        robot_name="left",
        env_ids=(0, 1, 2, 3, 4),
        current_positions=current,
        goal_positions=np.ones_like(current),
        joint_names=("j1", "j2"),
        sample_dt_s=0.02,
    )
    manager = TiledPlannerManager(
        backend=FailingChunkBackend(),
        max_workers=1,
        max_batch_problems=2,
        oversize_request_policy="split",
    )
    try:
        manager.submit(request)
        ready = manager.collect_ready(timeout_s=1.0)

        assert len(ready) == 1
        assert ready[0].request_id == "atomic"
        assert ready[0].env_ids == request.env_ids
        assert ready[0].success is False
        assert ready[0].status == "NO_PATH"
        assert ready[0].load_on_success is False
    finally:
        manager.shutdown()


def test_planner_manager_rejects_single_oversized_request_before_queueing() -> None:
    manager = TiledPlannerManager(
        max_workers=1,
        max_batch_problems=1,
        oversize_request_policy="reject",
    )
    try:
        with pytest.raises(ValueError, match="exceeding max_batch_problems"):
            manager.submit(_request("oversized"))

        status = manager.status()
        assert status["pending_count"] == 0
        assert status["rejected_requests"] == 1
        assert status["oversize_request_policy"] == "reject"
    finally:
        manager.shutdown()


def test_planner_manager_keeps_fifo_when_batch_keys_differ() -> None:
    class BatchBackend:
        def __init__(self) -> None:
            self.batches = []

        def plan_many(self, requests):
            self.batches.append(tuple(request.request_id for request in requests))
            return tuple(
                LinearJointPlannerBackend().plan(request) for request in requests
            )

        def plan(self, request):  # pragma: no cover - 本测试必须走 plan_many
            raise AssertionError("manager should call plan_many for batch backend")

    backend = BatchBackend()
    manager = TiledPlannerManager(
        backend=backend,
        max_workers=1,
        max_batch_problems=8,
    )
    try:
        manager.submit(_request("first", robot_name="left"))
        manager.submit(_request("second", robot_name="right"))
        manager.submit(_request("third", robot_name="left"))

        ready = []
        while len(ready) < 3:
            ready.extend(manager.collect_ready(timeout_s=1.0))

        assert [result.request_id for result in ready] == [
            "first",
            "second",
            "third",
        ]
        assert backend.batches == [("first",), ("second",), ("third",)]
    finally:
        manager.shutdown()


def test_planner_manager_rejects_when_pending_limit_is_full() -> None:
    class SlowBackend:
        def plan(self, request):
            time.sleep(0.05)
            return LinearJointPlannerBackend().plan(request)

    manager = TiledPlannerManager(
        backend=SlowBackend(),
        max_workers=1,
        max_pending_requests=1,
    )
    try:
        manager.submit(_request("first"))

        try:
            manager.submit(_request("second"))
        except RuntimeError as exc:
            assert "too many pending" in str(exc)
        else:  # pragma: no cover - 失败路径让断言更清楚
            raise AssertionError("pending limit did not reject second request")
    finally:
        manager.shutdown()


def test_planner_manager_limits_and_clears_completed_results() -> None:
    manager = TiledPlannerManager(max_workers=1, max_completed_results=1)
    try:
        manager.submit(_request("old"))
        manager.collect_ready(timeout_s=1.0)
        manager.submit(_request("new"))
        manager.collect_ready(timeout_s=1.0)

        status = manager.status()
        assert [item["request_id"] for item in status["completed"]] == ["new"]

        missing = manager.clear_completed("old")
        cleared = manager.clear_completed(["new"])
        assert missing == {"cleared": [], "missing": ["old"], "count": 0}
        assert cleared == {"cleared": ["new"], "missing": [], "count": 1}
        assert manager.status()["completed"] == []
    finally:
        manager.shutdown()


def test_planner_manager_cancel_matching_marks_stale_results_cancelled() -> None:
    class SlowBackend:
        def plan(self, request):
            time.sleep(0.05)
            return LinearJointPlannerBackend().plan(request)

    manager = TiledPlannerManager(backend=SlowBackend(), max_workers=1)
    try:
        manager.submit(_request("slow"))
        cancelled = manager.cancel_matching(env_ids=[1])
        ready = manager.collect_ready(timeout_s=1.0)

        assert cancelled[0]["request_id"] == "slow"
        assert ready[0].request_id == "slow"
        assert ready[0].success is False
        assert ready[0].status == "CANCELLED"
    finally:
        manager.shutdown()


def test_tiled_planning_message_parses_avoid_collisions_flag() -> None:
    request = planning_request_from_message(
        {
            "type": "plan",
            "kind": "joint_position_target",
            "joint_positions": [1.0, 2.0],
            "avoid_collisions": True,
        },
        robot_name="left",
        env_ids=np.asarray([0]),
        current_positions=np.asarray([[0.0, 0.0]]),
        command_joint_names=("j1", "j2"),
        default_sample_dt_s=0.05,
    )

    assert request.avoid_collisions is True
    assert request.metadata["avoid_collisions"] is True


def test_tiled_plan_request_explicit_values_override_runtime_defaults() -> None:
    defaults = PlannerRequestDefaults(
        duration_s=2.0,
        avoid_collisions=True,
        load_on_success=False,
        replace=False,
    )
    inherited = planning_request_from_message(
        {
            "type": "plan",
            "kind": "linear_pose_path",
            "target_offset": [0.0, 0.0, 0.1],
        },
        robot_name="left",
        env_ids=np.asarray([0]),
        current_positions=np.asarray([[0.0, 0.0]]),
        command_joint_names=("j1", "j2"),
        default_sample_dt_s=0.01,
        request_defaults=defaults,
        command_defaults=RuntimeCommandDefaults(orientation_mode="current"),
    )
    explicit = planning_request_from_message(
        {
            "type": "plan",
            "kind": "linear_pose_path",
            "target_offset": [0.0, 0.0, 0.1],
            "duration_s": 0.4,
            "sample_dt_s": 0.02,
            "orientation_mode": "free",
            "avoid_collisions": False,
            "load_on_success": True,
            "replace": True,
        },
        robot_name="left",
        env_ids=np.asarray([0]),
        current_positions=np.asarray([[0.0, 0.0]]),
        command_joint_names=("j1", "j2"),
        default_sample_dt_s=0.01,
        request_defaults=defaults,
        command_defaults=RuntimeCommandDefaults(orientation_mode="current"),
    )

    inherited_line = inherited.segments[0].path.segments[0]
    explicit_line = explicit.segments[0].path.segments[0]
    assert inherited.duration_s == pytest.approx(2.0)
    assert inherited.sample_dt_s == pytest.approx(0.01)
    assert inherited.avoid_collisions is True
    assert inherited.load_on_success is False
    assert inherited.replace is False
    assert inherited_line.orientation_mode == "current"
    assert explicit.duration_s == pytest.approx(0.4)
    assert explicit.sample_dt_s == pytest.approx(0.02)
    assert explicit.avoid_collisions is False
    assert explicit.load_on_success is True
    assert explicit.replace is True
    assert explicit_line.orientation_mode == "free"


def test_tiled_async_explicit_target_orientation_implies_target_mode() -> None:
    request = planning_request_from_message(
        {
            "type": "plan",
            "kind": "linear_pose_path",
            "target_offset": [0.0, 0.0, 0.1],
            "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        robot_name="left",
        env_ids=np.asarray([0]),
        current_positions=np.asarray([[0.0, 0.0]]),
        command_joint_names=("j1", "j2"),
        default_sample_dt_s=0.01,
        command_defaults=RuntimeCommandDefaults(orientation_mode="current"),
    )

    line = request.segments[0].path.segments[0]
    assert line.orientation_mode == "target"
    np.testing.assert_allclose(line.target_orientation, [1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("orientation_mode", ("free", "current"))
def test_tiled_async_rejects_non_target_mode_with_target_orientation(
    orientation_mode: str,
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        planning_request_from_message(
            {
                "type": "plan",
                "kind": "linear_pose_path",
                "target_offset": [0.0, 0.0, 0.1],
                "orientation_mode": orientation_mode,
                "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            robot_name="left",
            env_ids=np.asarray([0]),
            current_positions=np.asarray([[0.0, 0.0]]),
            command_joint_names=("j1", "j2"),
            default_sample_dt_s=0.01,
            command_defaults=RuntimeCommandDefaults(orientation_mode="target"),
        )


def test_tiled_plan_request_rejects_truthy_boolean_strings() -> None:
    with pytest.raises(ValueError, match="plan.avoid_collisions must be a boolean"):
        planning_request_from_message(
            {
                "type": "plan",
                "kind": "joint_position_target",
                "joint_positions": [1.0, 2.0],
                "avoid_collisions": "false",
            },
            robot_name="left",
            env_ids=np.asarray([0]),
            current_positions=np.asarray([[0.0, 0.0]]),
            command_joint_names=("j1", "j2"),
            default_sample_dt_s=0.01,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("coordination", "static_others"),
        ("force_collision_refresh", True),
        ("pose_reference_frame", "world"),
        ("duration", 1.0),
    ),
)
def test_tiled_plan_rejects_unknown_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        planning_request_from_message(
            {
                "type": "plan",
                "kind": "joint_position_target",
                "joint_positions": [1.0, 2.0],
                field: value,
            },
            robot_name="left",
            env_ids=np.asarray([0]),
            current_positions=np.asarray([[0.0, 0.0]]),
            command_joint_names=("j1", "j2"),
            default_sample_dt_s=0.01,
        )


def test_curobo_tiled_backend_accepts_linear_pose_path_segments() -> None:
    calls = []

    class FakePlanner:
        def joint_names(self):
            return ["j1", "j2"]

        def plan(self, request):
            calls.append(request)
            current = np.asarray(request.current_q, dtype=float).reshape(-1)
            goal = current + np.asarray([0.5, 0.25], dtype=float)
            trajectory = JointTrajectory.from_samples(
                times=np.asarray([0.0, 0.05, 0.1]),
                positions=np.asarray([current, (current + goal) * 0.5, goal]),
                joint_names=("j1", "j2"),
            )
            return MotionResult(
                path=None,
                trajectory=trajectory,
                success=True,
                status="SUCCESS",
            )

    request = TiledPlanningRequest(
        request_id="path",
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[0.0, 0.0]]),
        joint_names=("j1", "j2"),
        duration_s=0.1,
        sample_dt_s=0.05,
        segments=(
            TiledPlanningSegment(
                kind="linear_pose_path",
                path=TaskSpacePath(
                    segments=(
                        TcpLineSegment(target_offset=np.asarray([0.0, 0.0, 0.1])),
                    )
                ),
                tcp_frame_name="tool",
                duration_s=0.1,
                sample_dt_s=0.05,
            ),
        ),
        avoid_collisions=True,
    )

    result = TiledCuroboPlanningBackend(lambda robot: FakePlanner()).plan(request)

    assert result.success is True
    assert isinstance(calls[0], LinearPosePathRequest)
    assert calls[0].tcp_frame_name == "tool"
    assert calls[0].avoid_collisions is True
    np.testing.assert_allclose(result.times, [0.0, 0.05, 0.1])
    np.testing.assert_allclose(result.positions[0, -1], [0.5, 0.25])


def test_tiled_curobo_backend_lives_under_tiled_planning() -> None:
    assert TiledCuroboPlanningBackend.__module__ == (
        "linkerbot_sim.tiled.planning.backends.curobo"
    )
