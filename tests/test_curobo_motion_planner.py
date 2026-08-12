from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.backends.curobo.motion_planner import CuroboMotionPlanner
from linkerbot_sim.planning.requests import (
    LinearPosePathRequest,
    MotionRequest,
    PoseTarget,
    TaskSpacePath,
    TcpLineSegment,
    TcpPoseSequenceSegment,
)


class _FakeCuroboPlanner:
    def __init__(self) -> None:
        self.cspace_calls = []
        self.pose_calls = []
        self.criteria_updates = []

    def update_tool_pose_criteria(self, criteria):
        self.criteria_updates.append(dict(criteria))

    def plan_cspace(self, goal_state, current_state):
        self.cspace_calls.append((goal_state, current_state))
        return _fake_success_result()

    def plan_pose(self, goal_tool_poses, current_state):
        self.pose_calls.append((goal_tool_poses, current_state))
        return _fake_success_result()


class _FakeCuroboContext:
    default_tcp_frame = "tool"

    def __init__(
        self,
        *,
        supports_collision_queries: bool = True,
        tcp_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        frame_names=("tool",),
    ) -> None:
        self._frame_names = tuple(str(name) for name in frame_names)
        self.default_tcp_frame = self._frame_names[0]
        self.motion_planner = _FakeCuroboPlanner()
        self.ik_solver = _FakeCuroboIkSolver()
        self.types = SimpleNamespace(ToolPoseCriteria=_FakeToolPoseCriteria)
        self._supports_collision_queries = bool(supports_collision_queries)
        self.collision_consumers = []
        self._tcp_orientation_wxyz = np.asarray(tcp_orientation_wxyz, dtype=float)

    def joint_names(self):
        return ["j0", "j1"]

    def collision_queries_enabled(self):
        return self._supports_collision_queries

    def ensure_collision_checker(self, consumer):
        self.collision_consumers.append(str(consumer))
        return SimpleNamespace(available=self._supports_collision_queries)

    def frame_names(self):
        return list(self._frame_names)

    def joint_state_from_positions(self, positions):
        # fake context 只记录适配层传入的矩阵，避免测试依赖真实 cuRobo JointState。
        return {"joint_names": self.joint_names(), "position": np.asarray(positions)}

    def goal_tool_pose_from_arrays(
        self,
        *,
        positions,
        orientations_wxyz,
        tool_frames,
    ):
        # fake goal object 保留数组 shape，便于断言 task-space 请求是否被正确转发。
        return {
            "positions": np.asarray(positions),
            "orientations_wxyz": (
                None if orientations_wxyz is None else np.asarray(orientations_wxyz)
            ),
            "tool_frames": tuple(tool_frames),
        }

    def compute_tcp_poses(self, joint_positions, *, tcp_frame_name=None):
        frame_name = str(tcp_frame_name or self.default_tcp_frame)
        frame_index = self._frame_names.index(frame_name)
        joint_positions = np.asarray(joint_positions, dtype=float)
        return (
            np.tile(
                np.asarray(
                    [float(frame_index), float(frame_index), 0.0],
                    dtype=float,
                ).reshape(1, 3),
                (joint_positions.shape[0], 1),
            ),
            np.tile(
                self._tcp_orientation_wxyz.reshape(1, 4),
                (joint_positions.shape[0], 1),
            ),
        )


class _FakeCuroboIkSolver:
    def __init__(self) -> None:
        self.calls = []
        self.criteria_updates = []

    def update_tool_pose_criteria(self, criteria):
        self.criteria_updates.append(dict(criteria))

    def solve_pose(self, goal_tool_poses, *, current_state=None, seed_config=None):
        seeds = np.asarray(seed_config, dtype=float)
        positions = np.asarray(goal_tool_poses["positions"], dtype=float)
        seed_positions = seeds[:, 0, :] if seeds.ndim == 3 else seeds
        self.calls.append(
            {
                "goal": goal_tool_poses,
                "current_state": current_state,
                "seed_config": seeds.copy(),
            }
        )
        solution = seed_positions.copy()
        target_positions = positions[:, 0, :] if positions.ndim == 3 else positions
        solution[:, 0] = target_positions[:, 0]
        solution[:, 1] = target_positions[:, 1]
        return SimpleNamespace(
            success=np.ones(seeds.shape[0], dtype=bool),
            solution=solution,
            position_error=np.zeros(seeds.shape[0], dtype=float),
        )


class _FakeToolPoseCriteria:
    @staticmethod
    def track_position():
        return "position"

    @staticmethod
    def track_position_and_orientation():
        return "pose"


class _LazyMotionPlannerContext:
    default_tcp_frame = "tool"

    def __init__(self) -> None:
        self._motion_planner = _FakeCuroboPlanner()
        self.motion_planner_created = False

    @property
    def motion_planner(self):
        self.motion_planner_created = True
        return self._motion_planner

    def joint_names(self):
        return ["j0", "j1"]

    def collision_queries_enabled(self):
        return True

    def joint_state_from_positions(self, positions):
        return {"joint_names": self.joint_names(), "position": np.asarray(positions)}


def test_curobo_motion_planner_defers_single_planner_creation() -> None:
    context = _LazyMotionPlannerContext()
    planner = CuroboMotionPlanner(context)

    assert context.motion_planner_created is False
    assert planner.joint_names() == ["j0", "j1"]
    assert context.motion_planner_created is False

    planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.1]),
            goal_q=np.asarray([1.0, 1.1]),
        )
    )

    assert context.motion_planner_created is True


def test_curobo_motion_planner_routes_joint_goal_to_plan_cspace() -> None:
    context = _FakeCuroboContext()
    planner = CuroboMotionPlanner(context)

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.1]),
            goal_q=np.asarray([1.0, 1.1]),
        )
    )

    assert result.success is True
    assert result.status == "SUCCESS"
    assert len(context.motion_planner.cspace_calls) == 1
    goal_state, current_state = context.motion_planner.cspace_calls[0]
    np.testing.assert_allclose(goal_state["position"], [[1.0, 1.1]])
    np.testing.assert_allclose(current_state["position"], [[0.0, 0.1]])
    np.testing.assert_allclose(result.path[-1], [1.0, 1.0])
    assert result.diagnostics.metrics["num_waypoints"] == 3.0
    assert result.diagnostics.metrics["trajectory_samples"] == 3.0
    assert np.isclose(result.diagnostics.metrics["path_length"], np.sqrt(2.0))


def test_curobo_motion_planner_uses_planner_collision_consumer() -> None:
    context = _FakeCuroboContext()
    planner = CuroboMotionPlanner(context)

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.1]),
            goal_q=np.asarray([1.0, 1.1]),
            avoid_collisions=True,
        )
    )

    assert result.success is True
    assert context.collision_consumers == ["planner"]


def test_curobo_motion_planner_retimes_joint_plan_to_request_grid() -> None:
    context = _FakeCuroboContext()
    planner = CuroboMotionPlanner(context)

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.1]),
            goal_q=np.asarray([1.0, 1.1]),
            duration_s=0.2,
            sample_dt_s=0.05,
        )
    )

    assert result.success is True
    assert result.trajectory is not None
    np.testing.assert_allclose(result.trajectory.times, [0.05, 0.1, 0.15, 0.2])
    np.testing.assert_allclose(result.path[-1], [1.0, 1.0])
    assert result.diagnostics.metrics["trajectory_samples"] == 4.0


def test_curobo_motion_planner_uses_request_dt_when_result_has_no_dt() -> None:
    context = _FakeCuroboContext()
    context.motion_planner.plan_cspace = lambda _goal, _current: SimpleNamespace(
        success=np.asarray([True]),
        status="SUCCESS",
        interpolated_trajectory=SimpleNamespace(
            position=np.asarray([[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]]),
        ),
    )
    planner = CuroboMotionPlanner(context)

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
            duration_s=0.1,
            sample_dt_s=0.05,
        )
    )

    assert result.success is True
    assert result.trajectory is not None
    np.testing.assert_allclose(result.trajectory.times, [0.05, 0.1])


def test_curobo_motion_planner_routes_pose_goal_to_plan_pose() -> None:
    context = _FakeCuroboContext()
    planner = CuroboMotionPlanner(context, tcp_frame_name="tool")

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_pose=PoseTarget(
                position=np.asarray([0.1, 0.2, 0.3]),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
            ),
            tcp_frame_name="tool",
        )
    )

    assert result.success is True
    assert len(context.motion_planner.pose_calls) == 1
    goal, current_state = context.motion_planner.pose_calls[0]
    np.testing.assert_allclose(goal["positions"], [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(goal["orientations_wxyz"], [[1.0, 0.0, 0.0, 0.0]])
    assert goal["tool_frames"] == ("tool",)
    np.testing.assert_allclose(current_state["position"], [[0.0, 0.0]])
    assert context.motion_planner.criteria_updates[0] == {"tool": "pose"}


def test_curobo_motion_planner_pose_goal_none_uses_position_only_criteria() -> None:
    context = _FakeCuroboContext(tcp_orientation_wxyz=(0.0, 1.0, 0.0, 0.0))
    planner = CuroboMotionPlanner(context, tcp_frame_name="tool")

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_pose=PoseTarget(position=np.asarray([0.1, 0.2, 0.3])),
            tcp_frame_name="tool",
        )
    )

    assert result.success is True
    goal, _current_state = context.motion_planner.pose_calls[0]
    np.testing.assert_allclose(goal["positions"], [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(goal["orientations_wxyz"], [[0.0, 1.0, 0.0, 0.0]])
    assert context.motion_planner.criteria_updates[0] == {"tool": "position"}


def test_curobo_motion_planner_solves_linear_pose_path_request_with_sequential_ik() -> (
    None
):
    context = _FakeCuroboContext()
    planner = CuroboMotionPlanner(context, tcp_frame_name="tool")

    result = planner.plan(
        LinearPosePathRequest(
            current_q=np.asarray([0.0, 0.0]),
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_position=np.asarray([0.2, 0.3, 0.4]),
                        orientation_mode="target",
                        target_orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
                    ),
                )
            ),
            tcp_frame_name="tool",
            duration_s=0.04,
            sample_dt_s=0.02,
        )
    )

    assert result.success is True
    assert result.status == "SUCCESS"
    assert result.trajectory is not None
    assert context.ik_solver.calls
    np.testing.assert_allclose(result.path[0], [0.0, 0.0])
    np.testing.assert_allclose(result.path[-1], [0.2, 0.3])
    assert len(context.ik_solver.calls) == 2
    np.testing.assert_allclose(
        context.ik_solver.calls[0]["seed_config"],
        [[[0.0, 0.0]]],
    )
    np.testing.assert_allclose(
        context.ik_solver.calls[1]["seed_config"],
        [[[0.1, 0.15]]],
    )
    assert context.ik_solver.criteria_updates == [{"tool": "pose"}, {"tool": "pose"}]
    assert "linear_pose_path" in result.diagnostics.message


def test_curobo_linear_pose_path_rejects_discontinuous_start_position() -> None:
    context = _FakeCuroboContext()
    planner = CuroboMotionPlanner(context, tcp_frame_name="tool")
    request = LinearPosePathRequest(
        current_q=np.asarray([0.0, 0.0]),
        path=TaskSpacePath(
            segments=(
                TcpLineSegment(
                    start_position=np.asarray([9.0, 8.0, 7.0]),
                    target_position=np.asarray([0.2, 0.3, 0.4]),
                ),
            )
        ),
        tcp_frame_name="tool",
        duration_s=0.04,
        sample_dt_s=0.02,
    )

    result = planner.plan(request)

    assert result.success is False
    assert "start_position" in result.diagnostics.message


def test_curobo_linear_pose_path_requires_runtime_sample_dt() -> None:
    planner = CuroboMotionPlanner(_FakeCuroboContext(), tcp_frame_name="tool")

    result = planner.plan(
        LinearPosePathRequest(
            current_q=np.asarray([0.0, 0.0]),
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_position=np.asarray([0.2, 0.3, 0.4]),
                    ),
                )
            ),
            tcp_frame_name="tool",
            duration_s=0.04,
        )
    )

    assert result.success is False
    assert result.status == "UNSUPPORTED"
    assert "inject the runtime physics dt" in result.diagnostics.message


def test_linear_pose_path_rejects_unimplemented_blend_radius() -> None:
    request = LinearPosePathRequest(
        current_q=np.asarray([0.0, 0.0]),
        path=TaskSpacePath(
            segments=(
                TcpPoseSequenceSegment(
                    poses=(
                        PoseTarget(
                            position=np.asarray([0.2, 0.3, 0.4]),
                            orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
                        ),
                    ),
                    blend_radius=0.01,
                ),
            )
        ),
    )

    with pytest.raises(
        (NotImplementedError, ValueError),
        match="(?i)blend_radius",
    ):
        request.validate_structure()


def test_curobo_linear_pose_path_accepts_matching_start_position() -> None:
    context = _FakeCuroboContext()
    planner = CuroboMotionPlanner(context, tcp_frame_name="tool")

    result = planner.plan(
        LinearPosePathRequest(
            current_q=np.asarray([0.0, 0.0]),
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        start_position=np.asarray([0.0, 0.0, 0.0]),
                        target_position=np.asarray([0.2, 0.3, 0.4]),
                    ),
                )
            ),
            tcp_frame_name="tool",
            duration_s=0.04,
            sample_dt_s=0.02,
        )
    )

    assert result.success is True


def test_curobo_motion_planner_linear_pose_path_free_uses_position_only_criteria() -> (
    None
):
    context = _FakeCuroboContext(tcp_orientation_wxyz=(0.0, 1.0, 0.0, 0.0))
    planner = CuroboMotionPlanner(context, tcp_frame_name="tool")

    result = planner.plan(
        LinearPosePathRequest(
            current_q=np.asarray([0.0, 0.0]),
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_position=np.asarray([0.2, 0.3, 0.4]),
                        orientation_mode="free",
                    ),
                )
            ),
            tcp_frame_name="tool",
            duration_s=0.04,
            sample_dt_s=0.02,
        )
    )

    assert result.success is True
    orientations = np.vstack(
        [call["goal"]["orientations_wxyz"] for call in context.ik_solver.calls]
    )
    np.testing.assert_allclose(
        orientations,
        np.asarray([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    )
    assert context.ik_solver.criteria_updates == [
        {"tool": "position"},
        {"tool": "position"},
    ]


def test_linear_pose_path_rejects_noncanonical_orientation_mode() -> None:
    request = LinearPosePathRequest(
        current_q=np.asarray([0.0, 0.0]),
        path=TaskSpacePath(
            segments=(
                TcpLineSegment(
                    target_position=np.asarray([0.2, 0.3, 0.4]),
                    orientation_mode="none",
                ),
            )
        ),
    )

    with pytest.raises(ValueError, match="free, current, target"):
        request.validate_structure()


def test_curobo_motion_planner_linear_pose_path_fills_inactive_tcp_goals() -> None:
    context = _FakeCuroboContext(frame_names=("left_tcp", "right_tcp"))
    planner = CuroboMotionPlanner(context, tcp_frame_name="left_tcp")

    result = planner.plan(
        LinearPosePathRequest(
            current_q=np.asarray([0.0, 0.0]),
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_position=np.asarray([0.2, 0.3, 0.4]),
                        orientation_mode="target",
                        target_orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
                    ),
                )
            ),
            tcp_frame_name="left_tcp",
            duration_s=0.02,
            sample_dt_s=0.02,
        )
    )

    assert result.success is True
    goal = context.ik_solver.calls[0]["goal"]
    assert goal["tool_frames"] == ("left_tcp", "right_tcp")
    assert goal["positions"].shape == (1, 2, 3)
    np.testing.assert_allclose(goal["positions"][0, 1], [1.0, 1.0, 0.0])


def test_curobo_motion_planner_rejects_collision_aware_motion_without_model() -> None:
    planner = CuroboMotionPlanner(_FakeCuroboContext(supports_collision_queries=False))

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
            avoid_collisions=True,
        )
    )

    assert result.success is False
    assert result.status == "COLLISION_UNSUPPORTED"
    assert "avoid_collisions=True" in result.diagnostics.message


def test_curobo_motion_planner_rejects_collision_aware_linear_pose_without_model() -> (
    None
):
    planner = CuroboMotionPlanner(
        _FakeCuroboContext(supports_collision_queries=False),
        tcp_frame_name="tool",
    )

    result = planner.plan(
        LinearPosePathRequest(
            current_q=np.asarray([0.0, 0.0]),
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_position=np.asarray([0.2, 0.3, 0.4]),
                        orientation_mode="target",
                        target_orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
                    ),
                )
            ),
            tcp_frame_name="tool",
            duration_s=0.04,
            avoid_collisions=True,
        )
    )

    assert result.success is False
    assert result.status == "COLLISION_UNSUPPORTED"
    assert "avoid_collisions=True" in result.diagnostics.message


def test_curobo_motion_planner_converts_failed_result_without_trajectory() -> None:
    context = _FakeCuroboContext()
    context.motion_planner.plan_cspace = lambda _goal, _current: SimpleNamespace(
        success=np.asarray([False]),
        status="IK_FAIL",
        total_time=0.25,
    )
    planner = CuroboMotionPlanner(context)

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
        )
    )

    assert result.success is False
    assert result.path is None
    assert result.diagnostics.message == "IK_FAIL"
    assert result.diagnostics.metrics["total_time"] == 0.25


def _fake_success_result():
    trajectory = SimpleNamespace(
        position=np.asarray([[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]]),
    )
    return SimpleNamespace(
        success=np.asarray([True]),
        status="SUCCESS",
        interpolated_trajectory=trajectory,
        interpolated_trajectory_dt=np.asarray([0.1]),
        total_time=0.05,
    )
