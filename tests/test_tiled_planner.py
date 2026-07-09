from __future__ import annotations

import importlib
import time

import numpy as np

import linkerbot_sim.tiled as tiled
from linkerbot_sim.backends.cumotion import CuMotionJointPlannerBackend
from linkerbot_sim.planning.requests import (
    SpecifiedPathRequest,
    TaskSpacePath,
    TcpLineSegment,
)
from linkerbot_sim.planning.results import MotionResult
from linkerbot_sim.tiled.planner_manager import (
    LinearJointPlannerBackend,
    TiledPlannerManager,
    TiledPlanningSegment,
    TiledPlanningRequest,
)
from linkerbot_sim.trajectories.types import JointTrajectory


def _request(request_id: str = "req-1") -> TiledPlanningRequest:
    return TiledPlanningRequest(
        request_id=request_id,
        robot_name="left",
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


def test_cumotion_tiled_backend_accepts_specified_path_segments() -> None:
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
                kind="task_space_line",
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
    )

    result = CuMotionJointPlannerBackend(lambda robot: FakePlanner()).plan(request)

    assert result.success is True
    assert isinstance(calls[0], SpecifiedPathRequest)
    assert calls[0].tcp_frame_name == "tool"
    np.testing.assert_allclose(result.times, [0.0, 0.05, 0.1])
    np.testing.assert_allclose(result.positions[0, -1], [0.5, 0.25])


def test_cumotion_planner_backend_lives_under_cumotion_backend() -> None:
    assert CuMotionJointPlannerBackend.__module__ == (
        "linkerbot_sim.backends.cumotion.tiled_planner"
    )
    assert not hasattr(tiled, "CuMotionJointPlannerBackend")


def test_removed_tiled_interactive_planning_import_path_is_not_available() -> None:
    try:
        importlib.import_module("linkerbot_sim.tiled.interactive_planning")
    except ModuleNotFoundError:
        return
    raise AssertionError(
        "old tiled.interactive_planning import path is still available"
    )
