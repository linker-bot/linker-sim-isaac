from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from manipulation_project.backends.cumotion.motion_planner import CuMotionMotionPlanner
from manipulation_project.backends.cumotion.motion_planner_config import (
    GraphSearchConfig,
    MotionPlannerBackendConfig,
    SpecifiedPathConfig,
    TrajectoryGenerationConfig,
    TrajectoryOptimizationConfig,
)
from manipulation_project.planning.collision_objects import CollisionObject
from manipulation_project.planning.requests import (
    CompositePath,
    CompositePathPart,
    CSpaceWaypointPath,
    MotionRequest,
    PoseTarget,
    SpecifiedPathRequest,
    TaskSpacePath,
    TcpArcSegment,
    TcpLineSegment,
    TcpPoseSequenceSegment,
    TcpRotationSegment,
)


class _FakeGraphResults:
    def __init__(
        self,
        *,
        path_found: bool = True,
        path=None,
        interpolated_path=None,
    ) -> None:
        self.path_found = path_found
        self.path = path or []
        self.interpolated_path = interpolated_path or []


class _FakeGraphPlanner:
    def __init__(self, results: _FakeGraphResults) -> None:
        self.results = results
        self.cspace_calls = []
        self.translation_calls = []
        self.pose_calls = []

    def plan_to_cspace_target(self, current, goal, generate_interpolated_path):
        self.cspace_calls.append((current, goal, generate_interpolated_path))
        return self.results

    def plan_to_translation_target(
        self, current, translation, generate_interpolated_path
    ):
        self.translation_calls.append(
            (current, translation, generate_interpolated_path)
        )
        return self.results

    def plan_to_pose_target(self, current, pose_target, generate_interpolated_path):
        self.pose_calls.append((current, pose_target, generate_interpolated_path))
        return self.results


class _FakeTrajectory:
    def __init__(self, payload) -> None:
        self.payload = payload


class _FakeOptimizerResults:
    def __init__(self, *, status="SUCCESS", trajectory=None, target_index=0) -> None:
        self._status = status
        self._trajectory = trajectory if trajectory is not None else _FakeTrajectory(
            "optimizer"
        )
        self._target_index = target_index

    def status(self):
        return self._status

    def trajectory(self):
        return self._trajectory

    def target_index(self):
        return self._target_index


class _FakeTrajectoryGenerator:
    def __init__(self, cumotion) -> None:
        self.cumotion = cumotion
        self.position_limits = None
        self.velocity_limits = None
        self.acceleration_limits = None
        self.jerk_limits = None
        self.solver_params = []

    def generate_trajectory(self, waypoints):
        self.cumotion.generated_waypoints = waypoints
        return _FakeTrajectory(waypoints)

    def generate_time_stamped_trajectory(self, waypoints, times, interpolation_mode):
        self.cumotion.generated_waypoints = waypoints
        self.cumotion.generated_times = times
        self.cumotion.generated_interpolation_mode = interpolation_mode
        return _FakeTrajectory((waypoints, times, interpolation_mode))

    def set_position_limits(self, minimum, maximum) -> None:
        self.position_limits = (minimum, maximum)
        self.cumotion.trajectory_position_limits = self.position_limits

    def set_velocity_limits(self, values) -> None:
        self.velocity_limits = values
        self.cumotion.trajectory_velocity_limits = values

    def set_acceleration_limits(self, values) -> None:
        self.acceleration_limits = values
        self.cumotion.trajectory_acceleration_limits = values

    def set_jerk_limits(self, values) -> None:
        self.jerk_limits = values
        self.cumotion.trajectory_jerk_limits = values

    def set_solver_param(self, name, value) -> bool:
        self.solver_params.append((name, value.value))
        self.cumotion.trajectory_solver_params = self.solver_params
        return True


class _FakeParamValue:
    def __init__(self, value) -> None:
        self.value = value


class _FakeRotation3:
    def __init__(self, w, x, y, z) -> None:
        self.quaternion_wxyz = np.asarray([w, x, y, z], dtype=float)

    @staticmethod
    def from_matrix(matrix):
        return SimpleNamespace(matrix=np.asarray(matrix, dtype=float))


class _FakePose3:
    def __init__(self, rotation, translation) -> None:
        self.rotation = rotation
        self.translation = np.asarray(translation, dtype=float)

    @staticmethod
    def from_translation(translation):
        return _FakePose3(None, translation)


class _FakeCSpacePathSpec:
    """记录 CSpacePathSpec 调用，验证 specified_path 使用官方 C-space API。"""

    def __init__(self, initial_cspace_position) -> None:
        self.waypoints = [np.asarray(initial_cspace_position, dtype=float)]
        self.calls = []

    def add_cspace_waypoint(self, waypoint) -> bool:
        waypoint = np.asarray(waypoint, dtype=float)
        self.calls.append(("add_cspace_waypoint", waypoint))
        self.waypoints.append(waypoint)
        return True


class _FakeLinearCSpacePath:
    """最小 LinearCSpacePath 替身，只暴露 adapter 需要读取的 waypoints。"""

    def __init__(self, waypoints) -> None:
        self._waypoints = [np.asarray(waypoint, dtype=float) for waypoint in waypoints]

    def waypoints(self):
        return self._waypoints


class _FakeTaskSpacePathSpec:
    """记录 TaskSpacePathSpec 追加段的官方 API 名称和参数。

    fake 不尝试模拟 cuMotion 的 task-space 几何，只把 adapter 是否调用了正确 ``add_*`` 方法
    暴露给测试断言。
    """

    def __init__(self, initial_pose) -> None:
        self.initial_pose = initial_pose
        self.calls = []

    def add_translation(self, target_position, blend_radius=0.0) -> bool:
        self.calls.append(
            ("add_translation", np.asarray(target_position, dtype=float), blend_radius)
        )
        return True

    def add_linear_path(self, target_pose, blend_radius=0.0) -> bool:
        self.calls.append(("add_linear_path", target_pose, blend_radius))
        return True

    def add_rotation(self, target_rotation) -> bool:
        self.calls.append(("add_rotation", target_rotation))
        return True

    def add_tangent_arc(self, target_position, constant_orientation=True) -> bool:
        self.calls.append(
            (
                "add_tangent_arc",
                np.asarray(target_position, dtype=float),
                constant_orientation,
            )
        )
        return True

    def add_tangent_arc_with_orientation_target(self, target_pose) -> bool:
        self.calls.append(("add_tangent_arc_with_orientation_target", target_pose))
        return True

    def add_three_point_arc(
        self, target_position, intermediate_position, constant_orientation=True
    ) -> bool:
        self.calls.append(
            (
                "add_three_point_arc",
                np.asarray(target_position, dtype=float),
                np.asarray(intermediate_position, dtype=float),
                constant_orientation,
            )
        )
        return True

    def add_three_point_arc_with_orientation_target(
        self, target_pose, intermediate_position
    ) -> bool:
        self.calls.append(
            (
                "add_three_point_arc_with_orientation_target",
                target_pose,
                np.asarray(intermediate_position, dtype=float),
            )
        )
        return True


class _FakeTaskSpacePathConversionConfig:
    pass


class _FakeIkConfig:
    pass


class _FakeCompositePathSpec:
    """记录 CompositePathSpec 子路径和 transition mode 的调用顺序。"""

    class TransitionMode:
        SKIP = "skip"
        FREE = "free"
        LINEAR_TASK_SPACE = "linear_task_space"

    def __init__(self, initial_cspace_position) -> None:
        self.initial_cspace_position = np.asarray(initial_cspace_position, dtype=float)
        self.calls = []

    def add_cspace_path_spec(self, path_spec, transition_mode) -> bool:
        self.calls.append(("add_cspace_path_spec", path_spec, transition_mode))
        return True

    def add_task_space_path_spec(self, path_spec, transition_mode) -> bool:
        self.calls.append(("add_task_space_path_spec", path_spec, transition_mode))
        return True


class _FakeKinematics:
    def __init__(self) -> None:
        self.pose_calls = []

    def pose(self, current_q, frame_name):
        self.pose_calls.append((np.asarray(current_q, dtype=float), frame_name))
        return _FakePose3(_FakeRotation3(1.0, 0.0, 0.0, 0.0), [0.0, 0.0, 0.0])


class _FakeOptimizerType:
    class CSpaceTarget:
        class TranslationPathConstraint:
            @staticmethod
            def none():
                return SimpleNamespace(kind="translation_path_none")

        class OrientationPathConstraint:
            @staticmethod
            def none():
                return SimpleNamespace(kind="orientation_path_none")

        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class TranslationConstraint:
        @staticmethod
        def target(translation_target, terminal_deviation_limit=None):
            return SimpleNamespace(
                kind="translation_target",
                target=np.asarray(translation_target, dtype=float),
                terminal_deviation_limit=terminal_deviation_limit,
            )

    class OrientationConstraint:
        @staticmethod
        def none():
            return SimpleNamespace(kind="orientation_none")

        @staticmethod
        def terminal_target(orientation_target, terminal_deviation_limit=None):
            return SimpleNamespace(
                kind="orientation_terminal_target",
                target=orientation_target,
                terminal_deviation_limit=terminal_deviation_limit,
            )

    class TaskSpaceTarget:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs


class _FakeGraphConfig:
    def __init__(self, *, frame_name, world_view) -> None:
        self.frame_name = frame_name
        self.world_view = world_view
        self.params = []

    def set_param(self, name, value) -> bool:
        self.params.append((name, value.value))
        return True


class _FakeOptimizerConfig(_FakeGraphConfig):
    pass


class _FakeOptimizer:
    def __init__(self, results: _FakeOptimizerResults) -> None:
        self.results = results
        self.cspace_calls = []
        self.task_space_calls = []

    def plan_to_cspace_target(self, current, target):
        self.cspace_calls.append((current, target))
        return self.results

    def plan_to_task_space_target(self, current, target):
        self.task_space_calls.append((current, target))
        return self.results


class _FakeCumotion:
    Rotation3 = _FakeRotation3
    Pose3 = _FakePose3
    MotionPlannerConfig = SimpleNamespace(ParamValue=_FakeParamValue)
    TrajectoryOptimizerConfig = SimpleNamespace(ParamValue=_FakeParamValue)
    TrajectoryOptimizer = _FakeOptimizerType
    CompositePathSpec = _FakeCompositePathSpec
    TaskSpacePathConversionConfig = _FakeTaskSpacePathConversionConfig
    IkConfig = _FakeIkConfig
    CSpaceTrajectoryGenerator = SimpleNamespace(
        SolverParamValue=_FakeParamValue,
        InterpolationMode=SimpleNamespace(LINEAR="linear", CUBIC_SPLINE="cubic"),
    )

    def __init__(
        self,
        graph_planner: _FakeGraphPlanner | None = None,
        optimizer: _FakeOptimizer | None = None,
    ) -> None:
        self.graph_planner = graph_planner or _FakeGraphPlanner(
            _FakeGraphResults(path=[[0.0, 0.0], [1.0, 1.0]])
        )
        self.optimizer = optimizer or _FakeOptimizer(_FakeOptimizerResults())
        self.graph_config_calls = []
        self.graph_config_file_calls = []
        self.optimizer_config_calls = []
        self.optimizer_config_file_calls = []
        self.generated_waypoints = None
        self.generated_times = None
        self.generated_interpolation_mode = None
        self.trajectory_position_limits = None
        self.trajectory_velocity_limits = None
        self.trajectory_acceleration_limits = None
        self.trajectory_jerk_limits = None
        self.trajectory_solver_params = []
        self.cspace_path_specs = []
        self.linear_cspace_paths = []
        self.task_space_path_specs = []
        self.task_space_conversion_calls = []
        self.composite_path_specs = []
        self.composite_conversion_calls = []

    def create_default_motion_planner_config(
        self, robot_description, frame_name, world_view
    ):
        self.graph_config_calls.append((robot_description, frame_name, world_view))
        return _FakeGraphConfig(frame_name=frame_name, world_view=world_view)

    def create_motion_planner_config_from_file(
        self, path, robot_description, frame_name, world_view
    ):
        self.graph_config_file_calls.append(
            (path, robot_description, frame_name, world_view)
        )
        return _FakeGraphConfig(frame_name=frame_name, world_view=world_view)

    def create_motion_planner(self, config):
        self.graph_planner.config = config
        return self.graph_planner

    def create_default_trajectory_optimizer_config(
        self, robot_description, frame_name, world_view
    ):
        self.optimizer_config_calls.append(
            (robot_description, frame_name, world_view)
        )
        return _FakeOptimizerConfig(frame_name=frame_name, world_view=world_view)

    def create_trajectory_optimizer_config_from_file(
        self, path, robot_description, frame_name, world_view
    ):
        self.optimizer_config_file_calls.append(
            (path, robot_description, frame_name, world_view)
        )
        return _FakeOptimizerConfig(frame_name=frame_name, world_view=world_view)

    def create_trajectory_optimizer(self, config):
        self.optimizer.config = config
        return self.optimizer

    def create_cspace_trajectory_generator(self, kinematics):
        self.trajectory_generator_kinematics = kinematics
        return _FakeTrajectoryGenerator(self)

    def create_cspace_path_spec(self, initial_cspace_position):
        path_spec = _FakeCSpacePathSpec(initial_cspace_position)
        self.cspace_path_specs.append(path_spec)
        return path_spec

    def create_linear_cspace_path(self, cspace_path_spec):
        linear_path = _FakeLinearCSpacePath(cspace_path_spec.waypoints)
        self.linear_cspace_paths.append(linear_path)
        return linear_path

    def create_task_space_path_spec(self, initial_pose):
        path_spec = _FakeTaskSpacePathSpec(initial_pose)
        self.task_space_path_specs.append(path_spec)
        return path_spec

    def convert_task_space_path_spec_to_cspace(
        self, path_spec, kinematics, control_frame, conversion_config, ik_config
    ):
        self.task_space_conversion_calls.append(
            (path_spec, kinematics, control_frame, conversion_config, ik_config)
        )
        return _FakeLinearCSpacePath([[0.0, 0.0], [0.2, 0.4], [1.0, 1.0]])

    def create_composite_path_spec(self, initial_cspace_position):
        path_spec = _FakeCompositePathSpec(initial_cspace_position)
        self.composite_path_specs.append(path_spec)
        return path_spec

    def convert_composite_path_spec_to_cspace(
        self, path_spec, kinematics, control_frame, conversion_config, ik_config
    ):
        self.composite_conversion_calls.append(
            (path_spec, kinematics, control_frame, conversion_config, ik_config)
        )
        return _FakeLinearCSpacePath([[0.0, 0.0], [0.3, 0.5], [1.0, 1.0]])


class _FakeContext:
    def __init__(self, cumotion) -> None:
        self.cumotion = cumotion
        self.config = SimpleNamespace(
            custom_tcp_frame="tool",
            flange_frame="flange",
            motion_planner_config_path=None,
            motion_planner_params={},
            trajectory_limits={},
            trajectory_solver_params={},
            motion_planner=None,
        )
        self.robot_description = "robot_description"
        self.kinematics = _FakeKinematics()
        self.expected_cspace_width = 2
        self.collision_world_calls = []
        self.empty_collision_world_calls = 0
        self.current_collision_handles = {}

    def joint_names(self) -> list[str]:
        return ["j0", "j1"]

    def collision_world(self):
        self.collision_world_calls.append(())
        return SimpleNamespace(
            world_view="context_world_view",
            handles=dict(self.current_collision_handles),
        )

    def empty_collision_world(self):
        self.empty_collision_world_calls += 1
        return SimpleNamespace(world_view="empty_world_view", handles={})

    def has_frame(self, frame_name) -> bool:
        return frame_name in {"tool", "flange", "pinch_tcp"}


def _collision_object() -> CollisionObject:
    return CollisionObject(
        name="table",
        shape="cuboid",
        pose=np.eye(4),
        size=(1.0, 1.0, 0.1),
    )


def _graph_config(**kwargs) -> MotionPlannerBackendConfig:
    return MotionPlannerBackendConfig(
        planning_pipeline="graph_search",
        graph_search=kwargs.get("graph_search", GraphSearchConfig()),
        trajectory_generation=kwargs.get(
            "trajectory_generation", TrajectoryGenerationConfig()
        ),
        trajectory_optimization=kwargs.get(
            "trajectory_optimization", TrajectoryOptimizationConfig()
        ),
        specified_path=kwargs.get("specified_path", SpecifiedPathConfig()),
    )


def test_motion_planner_config_defaults_to_trajectory_optimization() -> None:
    assert MotionPlannerBackendConfig().planning_pipeline == "trajectory_optimization"


def test_motion_request_rejects_mode_keyword() -> None:
    try:
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
            **{"mode": "collision_aware"},
        )
    except TypeError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("expected MotionRequest.mode keyword to be rejected")


def test_facade_dispatches_to_trajectory_optimizer_by_default() -> None:
    optimizer = _FakeOptimizer(_FakeOptimizerResults(trajectory=_FakeTrajectory("ok")))
    fake_cumotion = _FakeCumotion(optimizer=optimizer)
    context = _FakeContext(fake_cumotion)
    obstacle = _collision_object()
    context.current_collision_handles = {
        obstacle.name: SimpleNamespace(name=obstacle.name)
    }

    planner = CuMotionMotionPlanner(context)
    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
        )
    )

    assert result.success
    assert result.joint_path is None
    assert isinstance(result.trajectory, _FakeTrajectory)
    assert fake_cumotion.optimizer_config_calls == [
        ("robot_description", "tool", "context_world_view")
    ]
    assert fake_cumotion.graph_config_calls == []
    assert context.collision_world_calls == [()]
    assert context.empty_collision_world_calls == 0
    current, target = optimizer.cspace_calls[0]
    np.testing.assert_allclose(current, [0.0, 0.0])
    np.testing.assert_allclose(target.args[0], [1.0, 1.0])
    assert result.diagnostics.metrics["num_collision_objects"] == 1.0


def test_trajectory_optimizer_can_ignore_environment_obstacles() -> None:
    config = MotionPlannerBackendConfig(
        trajectory_optimization=TrajectoryOptimizationConfig(
            use_environment_obstacles=False
        )
    )
    fake_cumotion = _FakeCumotion()
    context = _FakeContext(fake_cumotion)

    planner = CuMotionMotionPlanner(context, config=config)
    planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
        )
    )

    assert context.collision_world_calls == []
    assert context.empty_collision_world_calls == 1
    assert fake_cumotion.optimizer_config_calls == [
        ("robot_description", "tool", "empty_world_view")
    ]


def test_trajectory_optimizer_plans_to_translation_and_pose_targets() -> None:
    optimizer = _FakeOptimizer(_FakeOptimizerResults())
    context = _FakeContext(_FakeCumotion(optimizer=optimizer))
    planner = CuMotionMotionPlanner(context)

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_pose=PoseTarget(position=np.asarray([0.1, 0.2, 0.3])),
            tcp_frame_name="pinch_tcp",
        )
    )
    assert result.success
    _current, target = optimizer.task_space_calls[-1]
    translation_constraint, orientation_constraint = target.args
    np.testing.assert_allclose(translation_constraint.target, [0.1, 0.2, 0.3])
    assert orientation_constraint.kind == "orientation_none"

    planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_pose=PoseTarget(
                position=np.asarray([0.1, 0.2, 0.3]),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
            ),
        )
    )
    _current, target = optimizer.task_space_calls[-1]
    _translation_constraint, orientation_constraint = target.args
    assert orientation_constraint.kind == "orientation_terminal_target"
    np.testing.assert_allclose(
        orientation_constraint.target.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0]
    )


def test_trajectory_optimizer_failure_returns_failure_directly() -> None:
    optimizer = _FakeOptimizer(
        _FakeOptimizerResults(status="TRAJECTORY_OPTIMIZATION_FAILURE")
    )
    graph_planner = _FakeGraphPlanner(
        _FakeGraphResults(path=[[0.0, 0.0], [1.0, 1.0]])
    )
    context = _FakeContext(_FakeCumotion(graph_planner=graph_planner, optimizer=optimizer))

    planner = CuMotionMotionPlanner(context)
    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
        )
    )

    assert not result.success
    assert result.status == "TRAJECTORY_OPTIMIZATION_FAILURE"
    assert graph_planner.cspace_calls == []


def test_trajectory_optimizer_config_rejects_fallback_pipeline() -> None:
    try:
        MotionPlannerBackendConfig.from_mapping(
            {
                "trajectory_optimization": {
                    "fallback_pipeline": "graph_search",
                }
            }
        )
    except ValueError as exc:
        assert "fallback_pipeline is not supported" in str(exc)
    else:
        raise AssertionError("expected fallback_pipeline to be rejected")


def test_graph_search_uses_graph_config_and_trajectory_generation() -> None:
    graph_results = _FakeGraphResults(
        path=[[0.0, 0.0], [1.0, 1.0]],
        interpolated_path=[[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]],
    )
    fake_planner = _FakeGraphPlanner(graph_results)
    fake_cumotion = _FakeCumotion(graph_planner=fake_planner)
    context = _FakeContext(fake_cumotion)
    obstacle = _collision_object()
    context.current_collision_handles = {
        obstacle.name: SimpleNamespace(name=obstacle.name)
    }
    config = _graph_config(
        graph_search=GraphSearchConfig(
            generate_interpolated_path=True,
            motion_planner_config_path=Path("planner.yaml"),
            motion_planner_params={"step_size": 0.05},
        ),
        trajectory_generation=TrajectoryGenerationConfig(
            mode="time_stamped",
            interpolation_mode="cubic_spline",
            limits={
                "position_min": [-1.0, -2.0],
                "position_max": [1.0, 2.0],
                "velocity": [0.5, 0.6],
                "acceleration": [1.5, 1.6],
                "jerk": [3.0, 3.1],
            },
            solver_params={"max_iterations": 20},
        ),
    )

    planner = CuMotionMotionPlanner(context, config=config)
    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
            duration_s=2.0,
        )
    )

    assert result.success
    np.testing.assert_allclose(
        result.joint_path,
        [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]],
    )
    assert fake_cumotion.graph_config_file_calls
    assert fake_planner.config.params == [("step_size", 0.05)]
    np.testing.assert_allclose(fake_cumotion.generated_waypoints[-1], [1.0, 1.0])
    assert fake_cumotion.generated_interpolation_mode == "cubic"
    np.testing.assert_allclose(fake_cumotion.trajectory_position_limits[0], [-1.0, -2.0])
    np.testing.assert_allclose(fake_cumotion.trajectory_position_limits[1], [1.0, 2.0])
    np.testing.assert_allclose(fake_cumotion.trajectory_velocity_limits, [0.5, 0.6])
    np.testing.assert_allclose(fake_cumotion.trajectory_acceleration_limits, [1.5, 1.6])
    np.testing.assert_allclose(fake_cumotion.trajectory_jerk_limits, [3.0, 3.1])
    assert fake_cumotion.trajectory_solver_params == [("max_iterations", 20)]
    current, goal, generate_interpolated_path = fake_planner.cspace_calls[0]
    np.testing.assert_allclose(current, [0.0, 0.0])
    np.testing.assert_allclose(goal, [1.0, 1.0])
    assert generate_interpolated_path is True
    assert result.diagnostics.metrics["num_waypoints"] == 3.0
    assert result.diagnostics.metrics["num_collision_objects"] == 1.0


def test_graph_search_can_ignore_environment_obstacles() -> None:
    fake_planner = _FakeGraphPlanner(
        _FakeGraphResults(path=[[0.0, 0.0], [1.0, 1.0]])
    )
    fake_cumotion = _FakeCumotion(graph_planner=fake_planner)
    context = _FakeContext(fake_cumotion)
    config = _graph_config(
        graph_search=GraphSearchConfig(use_environment_obstacles=False),
        trajectory_generation=TrajectoryGenerationConfig(enabled=False),
    )

    planner = CuMotionMotionPlanner(context, config=config)
    planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
        )
    )

    assert context.collision_world_calls == []
    assert context.empty_collision_world_calls == 1
    assert fake_cumotion.graph_config_calls == [
        ("robot_description", "tool", "empty_world_view")
    ]


def test_graph_search_plans_to_translation_and_pose_targets() -> None:
    fake_planner = _FakeGraphPlanner(
        _FakeGraphResults(path=[[0.0, 0.0], [0.1, 0.2]])
    )
    context = _FakeContext(_FakeCumotion(graph_planner=fake_planner))
    planner = CuMotionMotionPlanner(
        context,
        config=_graph_config(
            trajectory_generation=TrajectoryGenerationConfig(enabled=False)
        ),
    )

    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_pose=PoseTarget(position=np.asarray([0.1, 0.2, 0.3])),
            tcp_frame_name="pinch_tcp",
        )
    )

    assert result.success
    current, translation, generate_interpolated_path = (
        fake_planner.translation_calls[0]
    )
    np.testing.assert_allclose(current, [0.0, 0.0])
    np.testing.assert_allclose(translation, [0.1, 0.2, 0.3])
    assert generate_interpolated_path is True

    planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_pose=PoseTarget(
                position=np.asarray([0.1, 0.2, 0.3]),
                orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
            ),
        )
    )
    _current, pose_target, _generate_interpolated_path = fake_planner.pose_calls[0]
    np.testing.assert_allclose(pose_target.translation, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(
        pose_target.rotation.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0]
    )


def test_graph_search_failure_returns_failed_motion_result() -> None:
    fake_planner = _FakeGraphPlanner(_FakeGraphResults(path_found=False))
    context = _FakeContext(_FakeCumotion(graph_planner=fake_planner))

    planner = CuMotionMotionPlanner(context, config=_graph_config())
    result = planner.plan(
        MotionRequest(
            current_q=np.asarray([0.0, 0.0]),
            goal_q=np.asarray([1.0, 1.0]),
        )
    )

    assert not result.success
    assert result.status == "FAILED"
    assert result.joint_path is None
    assert result.trajectory is None


def test_trajectory_generation_rejects_unknown_limit_keys() -> None:
    fake_planner = _FakeGraphPlanner(
        _FakeGraphResults(path=[[0.0, 0.0], [1.0, 1.0]])
    )
    context = _FakeContext(_FakeCumotion(graph_planner=fake_planner))
    config = _graph_config(
        trajectory_generation=TrajectoryGenerationConfig(
            limits={"max_velocity": [0.5, 0.6]}
        )
    )
    planner = CuMotionMotionPlanner(context, config=config)

    try:
        planner.plan(
            MotionRequest(
                current_q=np.asarray([0.0, 0.0]),
                goal_q=np.asarray([1.0, 1.0]),
            )
        )
    except ValueError as exc:
        assert "Unsupported trajectory_generation.limits key" in str(exc)
        assert "max_velocity" in str(exc)
    else:
        raise AssertionError("expected trajectory limit key validation")


def test_specified_path_conversion_config_rejects_unknown_keys() -> None:
    try:
        MotionPlannerBackendConfig(
            specified_path=SpecifiedPathConfig(
                task_space_segments={"conversion": {"bad_key": 1.0}}
            )
        ).validate()
    except ValueError as exc:
        assert "conversion key" in str(exc)
        assert "bad_key" in str(exc)
    else:
        raise AssertionError("expected conversion key validation")


def test_specified_path_conversion_config_validates_numeric_ranges() -> None:
    try:
        MotionPlannerBackendConfig(
            specified_path=SpecifiedPathConfig(
                task_space_segments={
                    "conversion": {
                        "min_position_deviation": 0.01,
                        "max_position_deviation": 0.001,
                    }
                }
            )
        ).validate()
    except ValueError as exc:
        assert "min_position_deviation" in str(exc)
    else:
        raise AssertionError("expected conversion numeric validation")


def test_tcp_line_segment_requires_target_for_target_orientation_mode() -> None:
    request = SpecifiedPathRequest(
        current_q=np.asarray([0.0, 0.0]),
        path=TaskSpacePath(
            segments=(
                TcpLineSegment(
                    target_position=np.asarray([0.1, 0.2, 0.3]),
                    orientation_mode="target",
                ),
            )
        ),
    )
    try:
        request.validate_structure()
    except ValueError as exc:
        assert "target_orientation" in str(exc)
    else:
        raise AssertionError("expected target_orientation validation")


def test_tcp_arc_segment_requires_intermediate_for_three_point_arc() -> None:
    request = SpecifiedPathRequest(
        current_q=np.asarray([0.0, 0.0]),
        path=TaskSpacePath(
            segments=(
                TcpArcSegment(
                    target_position=np.asarray([0.1, 0.2, 0.3]),
                    arc_mode="three_point",
                ),
            )
        ),
    )
    try:
        request.validate_structure()
    except ValueError as exc:
        assert "intermediate_position" in str(exc)
    else:
        raise AssertionError("expected arc intermediate validation")


def test_tcp_pose_sequence_requires_orientations() -> None:
    request = SpecifiedPathRequest(
        current_q=np.asarray([0.0, 0.0]),
        path=TaskSpacePath(
            segments=(
                TcpPoseSequenceSegment(
                    poses=(PoseTarget(position=np.asarray([0.1, 0.2, 0.3])),)
                ),
            )
        ),
    )
    try:
        request.validate_structure()
    except ValueError as exc:
        assert "orientation" in str(exc)
    else:
        raise AssertionError("expected pose orientation validation")


def test_composite_path_part_validates_transition_mode() -> None:
    request = SpecifiedPathRequest(
        current_q=np.asarray([0.0, 0.0]),
        path=CompositePath(
            parts=(
                CompositePathPart(
                    path=CSpaceWaypointPath(
                        waypoints=(
                            np.asarray([0.0, 0.0]),
                            np.asarray([1.0, 1.0]),
                        )
                    ),
                    transition_mode="bad",  # type: ignore[arg-type]
                ),
            )
        ),
    )
    try:
        request.validate_structure()
    except ValueError as exc:
        assert "transition_mode" in str(exc)
    else:
        raise AssertionError("expected transition mode validation")


def test_specified_path_cspace_waypoints_generates_joint_path() -> None:
    fake_cumotion = _FakeCumotion()
    context = _FakeContext(fake_cumotion)
    config = MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        specified_path=SpecifiedPathConfig(family="cspace_waypoints"),
        trajectory_generation=TrajectoryGenerationConfig(enabled=True),
    )

    planner = CuMotionMotionPlanner(context, config=config)
    result = planner.plan(
        SpecifiedPathRequest(
            current_q=np.asarray([0.0, 0.0]),
            path=CSpaceWaypointPath(
                waypoints=(
                    np.asarray([0.0, 0.0]),
                    np.asarray([0.5, 0.25]),
                    np.asarray([1.0, 1.0]),
                )
            ),
        )
    )

    assert result.success
    assert len(fake_cumotion.cspace_path_specs) == 1
    assert len(fake_cumotion.linear_cspace_paths) == 1
    np.testing.assert_allclose(
        result.joint_path,
        [[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]],
    )
    assert isinstance(result.trajectory, _FakeTrajectory)
    assert "pipeline=specified_path" in result.diagnostics.message
    assert "path_conversion=official" in result.diagnostics.message


def test_specified_path_cspace_requires_start_match() -> None:
    context = _FakeContext(_FakeCumotion())
    config = MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        specified_path=SpecifiedPathConfig(family="cspace_waypoints"),
    )
    planner = CuMotionMotionPlanner(context, config=config)

    try:
        planner.plan(
            SpecifiedPathRequest(
                current_q=np.asarray([0.0, 0.0]),
                path=CSpaceWaypointPath(
                    waypoints=(
                        np.asarray([0.1, 0.0]),
                        np.asarray([1.0, 1.0]),
                    )
                ),
            )
        )
    except ValueError as exc:
        assert "first waypoint" in str(exc)
        assert "current_q" in str(exc)
    else:
        raise AssertionError("expected first waypoint mismatch")


def test_specified_path_tcp_line_none_orientation_uses_add_translation() -> None:
    fake_cumotion = _FakeCumotion()
    context = _FakeContext(fake_cumotion)
    config = MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        specified_path=SpecifiedPathConfig(family="task_space_segments"),
        trajectory_generation=TrajectoryGenerationConfig(enabled=False),
    )
    planner = CuMotionMotionPlanner(context, config=config)

    result = planner.plan(
        SpecifiedPathRequest(
            current_q=np.asarray([0.0, 0.0]),
            tcp_frame_name="pinch_tcp",
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_position=np.asarray([0.1, 0.2, 0.3]),
                        orientation_mode="none",
                    ),
                )
            ),
        )
    )

    assert result.success
    calls = fake_cumotion.task_space_path_specs[0].calls
    assert calls[0][0] == "add_translation"
    np.testing.assert_allclose(calls[0][1], [0.1, 0.2, 0.3])
    assert fake_cumotion.task_space_conversion_calls
    assert "family=task_space_segments" in result.diagnostics.message


def test_specified_path_tcp_line_target_orientation_uses_add_linear_path() -> None:
    fake_cumotion = _FakeCumotion()
    context = _FakeContext(fake_cumotion)
    config = MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        specified_path=SpecifiedPathConfig(family="task_space_segments"),
        trajectory_generation=TrajectoryGenerationConfig(enabled=False),
    )
    planner = CuMotionMotionPlanner(context, config=config)

    planner.plan(
        SpecifiedPathRequest(
            current_q=np.asarray([0.0, 0.0]),
            tcp_frame_name="pinch_tcp",
            path=TaskSpacePath(
                segments=(
                    TcpLineSegment(
                        target_position=np.asarray([0.1, 0.2, 0.3]),
                        orientation_mode="target",
                        target_orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
                    ),
                )
            ),
        )
    )

    call = fake_cumotion.task_space_path_specs[0].calls[0]
    assert call[0] == "add_linear_path"
    pose = call[1]
    np.testing.assert_allclose(pose.translation, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose.rotation.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])


def test_specified_path_tcp_rotation_uses_add_rotation() -> None:
    fake_cumotion = _FakeCumotion()
    context = _FakeContext(fake_cumotion)
    config = MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        specified_path=SpecifiedPathConfig(family="task_space_segments"),
        trajectory_generation=TrajectoryGenerationConfig(enabled=False),
    )
    planner = CuMotionMotionPlanner(context, config=config)

    planner.plan(
        SpecifiedPathRequest(
            current_q=np.asarray([0.0, 0.0]),
            tcp_frame_name="pinch_tcp",
            path=TaskSpacePath(
                segments=(TcpRotationSegment(np.asarray([1.0, 0.0, 0.0, 0.0])),)
            ),
        )
    )

    call = fake_cumotion.task_space_path_specs[0].calls[0]
    assert call[0] == "add_rotation"
    np.testing.assert_allclose(call[1].quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])


def test_specified_path_three_point_arc_uses_official_arc_api() -> None:
    fake_cumotion = _FakeCumotion()
    context = _FakeContext(fake_cumotion)
    config = MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        specified_path=SpecifiedPathConfig(family="task_space_segments"),
        trajectory_generation=TrajectoryGenerationConfig(enabled=False),
    )
    planner = CuMotionMotionPlanner(context, config=config)

    planner.plan(
        SpecifiedPathRequest(
            current_q=np.asarray([0.0, 0.0]),
            tcp_frame_name="pinch_tcp",
            path=TaskSpacePath(
                segments=(
                    TcpArcSegment(
                        target_position=np.asarray([0.2, 0.0, 0.1]),
                        intermediate_position=np.asarray([0.1, 0.0, 0.1]),
                        arc_mode="three_point",
                    ),
                )
            ),
        )
    )

    call = fake_cumotion.task_space_path_specs[0].calls[0]
    assert call[0] == "add_three_point_arc"
    np.testing.assert_allclose(call[1], [0.2, 0.0, 0.1])
    np.testing.assert_allclose(call[2], [0.1, 0.0, 0.1])


def test_specified_path_pose_sequence_uses_linear_path_segments() -> None:
    fake_cumotion = _FakeCumotion()
    context = _FakeContext(fake_cumotion)
    config = MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        specified_path=SpecifiedPathConfig(family="task_space_segments"),
        trajectory_generation=TrajectoryGenerationConfig(enabled=False),
    )
    planner = CuMotionMotionPlanner(context, config=config)

    planner.plan(
        SpecifiedPathRequest(
            current_q=np.asarray([0.0, 0.0]),
            tcp_frame_name="pinch_tcp",
            path=TaskSpacePath(
                segments=(
                    TcpPoseSequenceSegment(
                        poses=(
                            PoseTarget(
                                position=np.asarray([0.1, 0.0, 0.0]),
                                orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
                            ),
                            PoseTarget(
                                position=np.asarray([0.2, 0.0, 0.0]),
                                orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
                            ),
                        )
                    ),
                )
            ),
        )
    )

    calls = fake_cumotion.task_space_path_specs[0].calls
    assert [call[0] for call in calls] == ["add_linear_path", "add_linear_path"]


def test_specified_path_composite_converts_to_cspace() -> None:
    fake_cumotion = _FakeCumotion()
    context = _FakeContext(fake_cumotion)
    config = MotionPlannerBackendConfig(
        planning_pipeline="specified_path",
        specified_path=SpecifiedPathConfig(
            family="composite",
            composite={"default_transition_mode": "free"},
        ),
        trajectory_generation=TrajectoryGenerationConfig(enabled=False),
    )
    planner = CuMotionMotionPlanner(context, config=config)

    result = planner.plan(
        SpecifiedPathRequest(
            current_q=np.asarray([0.0, 0.0]),
            tcp_frame_name="pinch_tcp",
            path=CompositePath(
                parts=(
                    CompositePathPart(
                        path=CSpaceWaypointPath(
                            waypoints=(
                                np.asarray([0.0, 0.0]),
                                np.asarray([0.4, 0.4]),
                            )
                        ),
                        transition_mode="skip",
                    ),
                    TaskSpacePath(
                        segments=(
                            TcpLineSegment(
                                target_position=np.asarray([0.1, 0.0, 0.0]),
                                orientation_mode="none",
                            ),
                        )
                    ),
                )
            ),
        )
    )

    assert result.success
    composite_spec = fake_cumotion.composite_path_specs[0]
    assert [call[0] for call in composite_spec.calls] == [
        "add_cspace_path_spec",
        "add_task_space_path_spec",
    ]
    assert composite_spec.calls[0][2] == "skip"
    assert composite_spec.calls[1][2] == "free"
    assert fake_cumotion.composite_conversion_calls
    assert "family=composite" in result.diagnostics.message
