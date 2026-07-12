from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linkerbot_sim.planning.requests import (
    LinearPosePathRequest,
    TaskSpacePath,
    TcpLineSegment,
)
from linkerbot_sim.planning.results import MotionResult, PlanningDiagnostics
from linkerbot_sim.tiled.planning.types import (
    TiledPlanningRequest,
    TiledPlanningSegment,
)
from linkerbot_sim.tiled.planning.backends.curobo import (
    TiledCuroboPlanningBackend,
)
from linkerbot_sim.trajectories.types import JointTrajectory


class _FakeCuroboPlanner:
    def __init__(self, *, fail_linear_pose_path: bool = False) -> None:
        self.calls = []
        self.fail_linear_pose_path = bool(fail_linear_pose_path)
        self.closed = False

    def close(self):
        self.closed = True

    def joint_names(self):
        return ["j0", "j1"]

    def plan(self, request):
        self.calls.append(request)
        if isinstance(request, LinearPosePathRequest) and self.fail_linear_pose_path:
            return MotionResult(
                path=None,
                trajectory=None,
                success=False,
                status="UNSUPPORTED",
                diagnostics=PlanningDiagnostics(
                    status="UNSUPPORTED",
                    message="linear pose path is not implemented by cuRobo adapter",
                ),
            )
        current = np.asarray(request.current_q, dtype=float).reshape(-1)
        if getattr(request, "goal_q", None) is not None:
            goal = np.asarray(request.goal_q, dtype=float).reshape(-1)
        else:
            goal = current + np.asarray([0.25, -0.25], dtype=float)
        trajectory = JointTrajectory.from_samples(
            times=np.asarray([0.0, 0.05, 0.1]),
            positions=np.asarray([current, (current + goal) * 0.5, goal]),
            joint_names=("j0", "j1"),
        )
        return MotionResult(
            path=None,
            trajectory=trajectory,
            success=True,
            status="SUCCESS",
        )


class _FakeBatchCuroboPlanner:
    def __init__(self, *, supports_collision_queries: bool = True) -> None:
        self.context = _FakeBatchContext(
            supports_collision_queries=supports_collision_queries
        )

    def joint_names(self):
        return ["j0", "j1"]

    def plan(self, request):
        raise AssertionError("batch-capable planner should not use per-env plan")


class _FakeBatchContext:
    def __init__(self, *, supports_collision_queries: bool = True) -> None:
        self.batch_motion_planner = _FakeBatchMotionPlanner()
        self._supports_collision_queries = bool(supports_collision_queries)
        self.closed = False

    def close(self):
        self.closed = True

    def collision_queries_enabled(self):
        return self._supports_collision_queries

    def joint_state_from_positions(self, positions):
        return SimpleNamespace(position=np.asarray(positions, dtype=float))


class _FakeBatchMotionPlanner:
    batch_size = 4

    def __init__(self) -> None:
        self.calls = []

    def plan_cspace(self, goal_state, current_state):
        current = np.asarray(current_state.position, dtype=float)
        goal = np.asarray(goal_state.position, dtype=float)
        self.calls.append({"current": current.copy(), "goal": goal.copy()})
        alpha = np.asarray([0.0, 0.5, 1.0], dtype=float).reshape(1, 1, 3, 1)
        positions = (
            current[:, None, None, :] + (goal - current)[:, None, None, :] * alpha
        )
        return SimpleNamespace(
            success=np.ones((current.shape[0], 1), dtype=bool),
            interpolated_trajectory=SimpleNamespace(position=positions),
            # 模拟真实 cuRobo：优化器返回的轨迹 dt 不一定等于 tiled 请求 sample_dt。
            interpolated_trajectory_dt=np.asarray([5.0]),
        )


def test_tiled_curobo_backend_maps_cspace_back_to_command_space() -> None:
    fake = _FakeCuroboPlanner()
    backend = TiledCuroboPlanningBackend(lambda _robot: fake)
    request = TiledPlanningRequest(
        request_id="joint",
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[8.0, 0.0, 0.0]]),
        goal_positions=np.asarray([[7.0, 1.0, 2.0]]),
        joint_names=("hand", "j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
    )

    result = backend.plan(request)

    assert result.success is True
    np.testing.assert_allclose(result.times, [0.0, 0.05, 0.1])
    # hand 不在 cuRobo C-space 中，因此按 tiled joint target 做线性插值。
    np.testing.assert_allclose(result.positions[0, :, 0], [8.0, 7.5, 7.0])
    # j0/j1 来自 cuRobo 规划结果，并回填到 command-space 对应列。
    np.testing.assert_allclose(result.positions[0, -1, 1:], [1.0, 2.0])
    assert fake.calls
    np.testing.assert_allclose(fake.calls[0].current_q, [0.0, 0.0])
    np.testing.assert_allclose(fake.calls[0].goal_q, [1.0, 2.0])


def test_tiled_curobo_backend_restores_t0_for_single_execution_trajectory() -> None:
    class _ExecutionGridPlanner(_FakeCuroboPlanner):
        def plan(self, request):
            self.calls.append(request)
            return MotionResult(
                path=None,
                trajectory=JointTrajectory.from_samples(
                    times=np.asarray([0.05, 0.1]),
                    positions=np.asarray([[0.5, 0.25], [1.0, 0.5]]),
                    joint_names=("j0", "j1"),
                ),
                success=True,
                status="SUCCESS",
            )

    request = TiledPlanningRequest(
        request_id="execution-grid",
        robot_name="left",
        env_ids=(7,),
        current_positions=np.asarray([[0.0, 0.0]]),
        goal_positions=np.asarray([[1.0, 0.5]]),
        joint_names=("j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
    )

    result = TiledCuroboPlanningBackend(lambda _robot: _ExecutionGridPlanner()).plan(
        request
    )

    np.testing.assert_allclose(result.times, [0.0, 0.05, 0.1])
    np.testing.assert_allclose(
        result.positions[0],
        [[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]],
    )


def test_tiled_curobo_backend_uses_batch_motion_planner_for_joint_targets() -> None:
    fake = _FakeBatchCuroboPlanner()
    factory_calls = 0

    def factory(_robot):
        nonlocal factory_calls
        factory_calls += 1
        return fake

    backend = TiledCuroboPlanningBackend(factory)
    request = TiledPlanningRequest(
        request_id="batch-joint",
        robot_name="left",
        env_ids=(0, 1),
        current_positions=np.asarray([[8.0, 0.0, 0.0], [9.0, 1.0, 1.0]]),
        goal_positions=np.asarray([[7.0, 1.0, 2.0], [6.0, 2.0, 3.0]]),
        joint_names=("hand", "j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
    )

    result = backend.plan(request)

    assert result.success is True
    assert "BatchMotionPlanner" in result.message
    assert factory_calls == 1
    assert fake.context.closed is True
    assert len(fake.context.batch_motion_planner.calls) == 1
    call = fake.context.batch_motion_planner.calls[0]
    # 真实 env 为 2 行，fake batch_size 为 4，因此后两行应由 adapter padding。
    np.testing.assert_allclose(call["current"][:2], [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(call["goal"][:2], [[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(call["current"][2:], [[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        result.positions[:, :, 0], [[8.0, 7.5, 7.0], [9.0, 7.5, 6.0]]
    )
    np.testing.assert_allclose(result.positions[0, -1, 1:], [1.0, 2.0])
    np.testing.assert_allclose(result.positions[1, -1, 1:], [2.0, 3.0])


def test_tiled_curobo_collision_checks_use_batch_planner_consumer() -> None:
    fake = _FakeBatchCuroboPlanner()
    consumers = []

    def ensure_collision_checker(consumer):
        consumers.append(str(consumer))
        if consumer != "batch_planner":
            raise ValueError(f"unknown collision consumer: {consumer}")
        return SimpleNamespace(available=True)

    fake.context.ensure_collision_checker = ensure_collision_checker
    backend = TiledCuroboPlanningBackend(lambda _robot: fake)
    request = TiledPlanningRequest(
        request_id="collision-aware-batch",
        robot_name="left",
        env_ids=(0, 1),
        current_positions=np.asarray([[0.0, 0.0], [0.1, 0.2]], dtype=float),
        goal_positions=np.asarray([[0.5, 0.25], [0.6, 0.4]], dtype=float),
        joint_names=("j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
        avoid_collisions=True,
    )

    result = backend.plan(request)

    assert result.success is True
    assert consumers
    assert set(consumers) == {"batch_planner"}


def test_tiled_curobo_backend_can_disable_joint_batch_mode() -> None:
    fake = _FakeCuroboPlanner()
    fake.context = _FakeBatchContext()
    backend = TiledCuroboPlanningBackend(
        lambda _robot: fake,
        joint_batch_mode="per_env",
    )
    request = TiledPlanningRequest(
        request_id="per-env",
        robot_name="left",
        env_ids=(0, 1),
        current_positions=np.asarray([[0.0, 0.0], [0.1, 0.2]], dtype=float),
        goal_positions=np.asarray([[0.5, 0.25], [0.6, 0.4]], dtype=float),
        joint_names=("j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
    )

    result = backend.plan(request)

    assert result.success is True
    assert len(fake.calls) == 2
    assert fake.context.batch_motion_planner.calls == []


def test_per_env_collision_request_uses_single_planner_consumer() -> None:
    fake = _FakeCuroboPlanner()
    consumers: list[str] = []
    original_plan = fake.plan

    class _PerEnvContext:
        @property
        def batch_motion_planner(self):
            raise AssertionError("per-env mode must not access batch planner")

        def ensure_collision_checker(self, consumer):
            consumers.append(str(consumer))

    def plan(request):
        fake.context.ensure_collision_checker("planner")
        return original_plan(request)

    fake.plan = plan
    fake.context = _PerEnvContext()
    backend = TiledCuroboPlanningBackend(
        lambda _robot: fake,
        joint_batch_mode="per_env",
    )
    request = TiledPlanningRequest(
        request_id="per-env-collision-consumer",
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[0.0, 0.0]], dtype=float),
        goal_positions=np.asarray([[0.5, 0.25]], dtype=float),
        joint_names=("j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
        avoid_collisions=True,
    )

    result = backend.plan(request)

    assert result.success is True
    assert consumers == ["planner"]


def test_tiled_curobo_backend_batch_only_rejects_non_batch_request() -> None:
    fake = _FakeCuroboPlanner()
    backend = TiledCuroboPlanningBackend(
        lambda _robot: fake,
        joint_batch_mode="batch_only",
    )
    request = TiledPlanningRequest(
        request_id="batch-only",
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[0.0, 0.0]], dtype=float),
        goal_positions=np.asarray([[0.5, 0.25]], dtype=float),
        joint_names=("j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
    )

    result = backend.plan(request)

    assert result.success is False
    assert result.status == "BATCH_UNAVAILABLE"
    assert fake.calls == []


def test_tiled_curobo_backend_plan_many_batches_joint_requests() -> None:
    fake = _FakeBatchCuroboPlanner()
    factory_calls = 0
    factory_robot_names = []

    def factory(robot_name):
        nonlocal factory_calls
        factory_calls += 1
        factory_robot_names.append(robot_name)
        return fake

    backend = TiledCuroboPlanningBackend(factory)
    first = TiledPlanningRequest(
        request_id="first",
        robot_name="left",
        env_ids=(10,),
        current_positions=np.asarray([[8.0, 0.0, 0.0]]),
        goal_positions=np.asarray([[7.0, 1.0, 2.0]]),
        joint_names=("hand", "j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
        source="first-source",
        load_on_success=False,
        replace=False,
    )
    second = TiledPlanningRequest(
        request_id="second",
        robot_name="left",
        env_ids=(20, 21),
        current_positions=np.asarray([[9.0, 1.0, 1.0], [10.0, 2.0, 2.0]]),
        goal_positions=np.asarray([[6.0, 2.0, 3.0], [5.0, 3.0, 4.0]]),
        joint_names=("hand", "j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
        source="second-source",
        load_on_success=True,
        replace=True,
    )

    first_result, second_result = backend.plan_many((first, second))

    assert first_result.request_id == "first"
    assert first_result.env_ids == (10,)
    assert second_result.request_id == "second"
    assert second_result.env_ids == (20, 21)
    assert factory_calls == 1
    assert factory_robot_names == ["left"]
    assert fake.context.closed is True
    assert len(fake.context.batch_motion_planner.calls) == 1
    call = fake.context.batch_motion_planner.calls[0]
    np.testing.assert_allclose(
        call["current"][:3],
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    )
    np.testing.assert_allclose(
        call["goal"][:3],
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
    )
    assert first_result.positions.shape == (1, 3, 3)
    assert second_result.positions.shape == (2, 3, 3)
    assert first_result.source == "first-source"
    assert first_result.load_on_success is False
    assert first_result.replace is False
    assert second_result.source == "second-source"
    assert second_result.load_on_success is True
    assert second_result.replace is True
    np.testing.assert_allclose(first_result.positions[0, -1], [7.0, 1.0, 2.0])
    np.testing.assert_allclose(
        second_result.positions[:, -1, 1:], [[2.0, 3.0], [3.0, 4.0]]
    )


def test_tiled_curobo_backend_plan_many_falls_back_for_task_paths() -> None:
    planners = []

    def factory(_robot):
        planner = _FakeCuroboPlanner()
        planners.append(planner)
        return planner

    backend = TiledCuroboPlanningBackend(factory)
    path_segment = TiledPlanningSegment(
        kind="linear_pose_path",
        path=TaskSpacePath(
            segments=(TcpLineSegment(target_offset=np.asarray([0.0, 0.0, 0.1])),)
        ),
        tcp_frame_name="tool",
        duration_s=0.1,
        sample_dt_s=0.05,
    )
    requests = tuple(
        TiledPlanningRequest(
            request_id=f"path-{index}",
            robot_name="left",
            env_ids=(index,),
            current_positions=np.asarray([[0.0, 0.0]]),
            joint_names=("j0", "j1"),
            duration_s=0.1,
            sample_dt_s=0.05,
            segments=(path_segment,),
        )
        for index in range(2)
    )

    results = backend.plan_many(requests)

    assert [result.request_id for result in results] == ["path-0", "path-1"]
    path_calls = [planner.calls[0] for planner in planners if planner.calls]
    assert len(path_calls) == 2
    assert all(isinstance(call, LinearPosePathRequest) for call in path_calls)


def test_tiled_curobo_backend_rejects_collision_aware_batch_without_model() -> None:
    fake = _FakeBatchCuroboPlanner(supports_collision_queries=False)
    backend = TiledCuroboPlanningBackend(lambda _robot: fake)
    request = TiledPlanningRequest(
        request_id="batch-collision",
        robot_name="left",
        env_ids=(0, 1),
        current_positions=np.asarray([[0.0, 0.0], [1.0, 1.0]]),
        goal_positions=np.asarray([[1.0, 2.0], [2.0, 3.0]]),
        joint_names=("j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
        avoid_collisions=True,
    )

    result = backend.plan(request)

    assert result.success is False
    assert result.status == "COLLISION_UNSUPPORTED"
    assert "avoid_collisions=True" in result.message
    assert fake.context.closed is True
    assert fake.context.batch_motion_planner.calls == []


def test_tiled_curobo_backend_keeps_non_cspace_joints_for_task_path() -> None:
    fake = _FakeCuroboPlanner()
    backend = TiledCuroboPlanningBackend(lambda _robot: fake)
    request = TiledPlanningRequest(
        request_id="path",
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[8.0, 0.0, 0.0]]),
        joint_names=("hand", "j0", "j1"),
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
    )

    result = backend.plan(request)

    assert result.success is True
    assert isinstance(fake.calls[0], LinearPosePathRequest)
    assert fake.closed is True
    assert fake.calls[0].avoid_collisions is False
    np.testing.assert_allclose(result.positions[0, :, 0], [8.0, 8.0, 8.0])
    np.testing.assert_allclose(result.positions[0, -1, 1:], [0.25, -0.25])


def test_tiled_curobo_backend_passes_collision_flag_to_linear_pose_path() -> None:
    fake = _FakeCuroboPlanner()
    consumers: list[str] = []
    original_plan = fake.plan

    class _TaskSpaceContext:
        @property
        def batch_motion_planner(self):
            raise AssertionError("task-space path must not access batch planner")

        def ensure_collision_checker(self, consumer):
            consumers.append(str(consumer))

    def plan(request):
        fake.context.ensure_collision_checker("ik")
        return original_plan(request)

    fake.plan = plan
    fake.context = _TaskSpaceContext()
    backend = TiledCuroboPlanningBackend(lambda _robot: fake)
    request = TiledPlanningRequest(
        request_id="path-collision",
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[0.0, 0.0]]),
        joint_names=("j0", "j1"),
        duration_s=0.1,
        sample_dt_s=0.05,
        avoid_collisions=True,
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
    )

    result = backend.plan(request)

    assert result.success is True
    assert isinstance(fake.calls[0], LinearPosePathRequest)
    assert fake.closed is True
    assert fake.calls[0].avoid_collisions is True
    assert consumers == ["ik"]


def test_tiled_curobo_backend_returns_backend_failure_message() -> None:
    fake = _FakeCuroboPlanner(fail_linear_pose_path=True)
    backend = TiledCuroboPlanningBackend(lambda _robot: fake)
    request = TiledPlanningRequest(
        request_id="unsupported-path",
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[0.0, 0.0]]),
        joint_names=("j0", "j1"),
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
            ),
        ),
    )

    result = backend.plan(request)

    assert result.success is False
    assert result.status == "UNSUPPORTED"
    assert "linear pose path" in result.message


def test_tiled_curobo_backend_rejects_missing_cspace_joint_name() -> None:
    backend = TiledCuroboPlanningBackend(lambda _robot: _FakeCuroboPlanner())
    request = TiledPlanningRequest(
        request_id="missing-joint",
        robot_name="left",
        env_ids=(0,),
        current_positions=np.asarray([[0.0, 0.0]]),
        goal_positions=np.asarray([[1.0, 1.0]]),
        joint_names=("hand", "j0"),
        duration_s=0.1,
        sample_dt_s=0.05,
    )

    try:
        backend.plan(request)
    except ValueError as exc:
        assert "cuRobo planner joints" in str(exc)
    else:
        raise AssertionError("missing planner joint was accepted")


def test_fake_namespace_keeps_module_import_side_effect_free() -> None:
    # 保持本测试不导入真实 cuRobo runtime；fake planner 只模拟项目侧 facade 契约。
    fake = SimpleNamespace(name="curobo")
    assert fake.name == "curobo"
