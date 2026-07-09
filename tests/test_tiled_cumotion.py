from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.backends.cumotion.tiled_ik import (
    BatchedCuMotionIKSolver,
    CuMotionJointMapping,
)


class _FakeRotation3:
    def __init__(self, w, x, y, z) -> None:
        self.quaternion_wxyz = np.asarray([w, x, y, z], dtype=float)

    @staticmethod
    def distance(left, right) -> float:
        return float(
            np.linalg.norm(
                np.asarray(left.quaternion_wxyz) - np.asarray(right.quaternion_wxyz)
            )
        )


class _FakePose3:
    def __init__(self, rotation, translation) -> None:
        self.rotation = rotation
        self.translation = np.asarray(translation, dtype=float)

    @staticmethod
    def from_translation(translation):
        return _FakePose3(None, translation)


class _FakeCumotionWithoutBatch:
    Rotation3 = _FakeRotation3
    Pose3 = _FakePose3

    def __init__(self, *, fail_rows: set[int] | None = None) -> None:
        self.solve_calls = []
        self.fail_rows = fail_rows or set()


class _FakeBatchConfig:
    def __init__(self) -> None:
        self.params = {}

    def set_param(self, name, value):
        self.params[name] = value
        return True


class _FakeTranslationConstraintArray:
    def __init__(self, targets, deviation_limit) -> None:
        self.targets = targets
        self.deviation_limit = deviation_limit

    @staticmethod
    def target(translation_targets, deviation_limit=None):
        return _FakeTranslationConstraintArray(translation_targets, deviation_limit)


class _FakeOrientationConstraintArray:
    def __init__(self, targets=None, deviation_limit=None) -> None:
        self.targets = targets
        self.deviation_limit = deviation_limit

    @staticmethod
    def none():
        return _FakeOrientationConstraintArray()

    @staticmethod
    def target(orientation_targets, deviation_limit=None):
        return _FakeOrientationConstraintArray(orientation_targets, deviation_limit)


class _FakeTaskSpaceTargetArray:
    def __init__(self, translation_constraints, orientation_constraints) -> None:
        self.translation_constraints = translation_constraints
        self.orientation_constraints = orientation_constraints

    def num_problems(self):
        return len(self.translation_constraints.targets)


class _FakeBatchProblemResult:
    def __init__(self, status, positions) -> None:
        self._status = status
        self._positions = positions

    def status(self):
        return self._status

    def cspace_positions(self):
        return list(self._positions)

    def target_indices(self):
        return [0 for _ in self._positions]


class _FakeBatchResultsArray:
    def __init__(self, problems) -> None:
        self._problems = problems

    def num_problems(self):
        return len(self._problems)

    def num_successes(self):
        return sum(problem.status() == "SUCCESS" for problem in self._problems)

    def problem(self, problem_index):
        return self._problems[int(problem_index)]


class _FakeCollisionFreeIkSolverType:
    TranslationConstraintArray = _FakeTranslationConstraintArray
    OrientationConstraintArray = _FakeOrientationConstraintArray
    TaskSpaceTargetArray = _FakeTaskSpaceTargetArray

    class Results:
        class Status:
            SUCCESS = "SUCCESS"
            INVERSE_KINEMATICS_FAILURE = "INVERSE_KINEMATICS_FAILURE"


class _FakeBatchSolver:
    def __init__(self, cumotion) -> None:
        self.cumotion = cumotion
        self.solve_array_calls = []

    def solve_array(self, task_space_target_array, cspace_seeds):
        seeds = [np.asarray(seed, dtype=float) for seed in cspace_seeds]
        self.solve_array_calls.append(
            {
                "target_array": task_space_target_array,
                "seeds": [seed.copy() for seed in seeds],
            }
        )
        problems = []
        for env_index, targets in enumerate(
            task_space_target_array.translation_constraints.targets
        ):
            seed = seeds[env_index]
            target = np.asarray(targets[0], dtype=float)
            success = env_index not in self.cumotion.fail_rows
            if success:
                q = seed + target[: seed.size]
                problems.append(_FakeBatchProblemResult("SUCCESS", [q, q + 100.0]))
            else:
                problems.append(
                    _FakeBatchProblemResult("INVERSE_KINEMATICS_FAILURE", [])
                )
        return _FakeBatchResultsArray(problems)


class _FakeBatchCumotion(_FakeCumotionWithoutBatch):
    CollisionFreeIkSolver = _FakeCollisionFreeIkSolverType

    class CollisionFreeIkSolverConfig:
        class ParamValue:
            def __init__(self, value) -> None:
                self.value = value

    def __init__(self, *, fail_rows: set[int] | None = None) -> None:
        super().__init__(fail_rows=fail_rows)
        self.batch_solver = _FakeBatchSolver(self)
        self.batch_configs = []

    def create_default_collision_free_ik_solver_config(
        self, robot_description, tool_frame_name, world_view
    ):
        config = _FakeBatchConfig()
        config.robot_description = robot_description
        config.tool_frame_name = tool_frame_name
        config.world_view = world_view
        self.batch_configs.append(config)
        return config

    def create_collision_free_ik_solver(self, config):
        self.batch_config = config
        return self.batch_solver


class _FakeKinematics:
    def position(self, cspace_position, frame_name):
        q = np.asarray(cspace_position, dtype=float)
        return np.asarray([q[0], q[1], 0.0], dtype=float)

    def orientation(self, cspace_position, frame_name):
        return _FakeRotation3(1.0, 0.0, 0.0, 0.0)


class _FakeCollisionWorld:
    world_view = object()


class _FakeContext:
    def __init__(
        self,
        *,
        fail_rows: set[int] | None = None,
        batch_api: bool = True,
    ) -> None:
        self.cumotion = (
            _FakeBatchCumotion(fail_rows=fail_rows)
            if batch_api
            else _FakeCumotionWithoutBatch(fail_rows=fail_rows)
        )
        self.kinematics = _FakeKinematics()
        self.robot_description = object()
        self.expected_cspace_width = 2
        self.config = SimpleNamespace(
            default_tcp_frame="tool",
            flange_frame="tool",
            kinematics=SimpleNamespace(
                ik=SimpleNamespace(
                    position_tolerance=0.01,
                    orientation_tolerance=0.02,
                    orientation_weight=0.5,
                    ccd_max_iterations=11,
                    bfgs_max_iterations=22,
                )
            ),
        )

    def joint_names(self):
        return ["arm0", "arm1"]

    def has_frame(self, frame_name):
        return frame_name == "tool"

    def empty_collision_world(self):
        return _FakeCollisionWorld()


def test_batched_cumotion_ik_uses_one_explicit_seed_per_env() -> None:
    context = _FakeContext()
    solver = BatchedCuMotionIKSolver(context, tcp_frame_name="tool")

    result = solver.solve(
        target_positions=np.asarray([[0.1, 0.2, 0.0], [1.0, 2.0, 0.0]]),
        target_orientations_wxyz=None,
        seeds=np.asarray([[10.0, 20.0], [30.0, 40.0]]),
        tcp_frame_name="tool",
    )

    assert result.success.tolist() == [True, True]
    np.testing.assert_allclose(result.joint_positions, [[10.1, 20.2], [31.0, 42.0]])
    assert solver.last_backend == "collision_free_solve_array"
    assert len(context.cumotion.batch_solver.solve_array_calls) == 1
    assert context.cumotion.solve_calls == []
    seeds = context.cumotion.batch_solver.solve_array_calls[0]["seeds"]
    np.testing.assert_allclose(seeds, [[10.0, 20.0], [30.0, 40.0]])


def test_batched_cumotion_ik_keeps_orientation_constraints_when_given() -> None:
    context = _FakeContext()
    solver = BatchedCuMotionIKSolver(context, tcp_frame_name="tool")

    result = solver.solve(
        target_positions=np.asarray([[0.1, 0.2, 0.0]]),
        target_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        seeds=np.asarray([[10.0, 20.0]]),
        tcp_frame_name="tool",
    )

    assert result.orientation_error is not None
    np.testing.assert_allclose(result.orientation_error, [0.0])
    target_array = context.cumotion.batch_solver.solve_array_calls[0]["target_array"]
    orientation = target_array.orientation_constraints
    assert orientation.deviation_limit == 0.02
    np.testing.assert_allclose(
        orientation.targets[0][0].quaternion_wxyz,
        [1.0, 0.0, 0.0, 0.0],
    )


def test_batched_cumotion_ik_scatter_cspace_solution_to_command_space() -> None:
    context = _FakeContext()
    solver = BatchedCuMotionIKSolver(
        context,
        tcp_frame_name="tool",
        command_joint_names=("hand", "arm1", "arm0"),
    )

    result = solver.solve(
        target_positions=np.asarray([[0.5, 1.0, 0.0]]),
        target_orientations_wxyz=None,
        seeds=np.asarray([[99.0, 20.0, 10.0]]),
        tcp_frame_name="tool",
    )

    # C-space order is arm0, arm1; command order is hand, arm1, arm0.
    seeds = context.cumotion.batch_solver.solve_array_calls[0]["seeds"]
    np.testing.assert_allclose(seeds, [[10.0, 20.0]])
    np.testing.assert_allclose(result.joint_positions, [[99.0, 21.0, 10.5]])


def test_batched_cumotion_ik_computes_tcp_poses_with_command_mapping() -> None:
    context = _FakeContext()
    solver = BatchedCuMotionIKSolver(
        context,
        tcp_frame_name="tool",
        command_joint_names=("hand", "arm1", "arm0"),
    )

    positions, orientations = solver.compute_tcp_poses(np.asarray([[99.0, 20.0, 10.0]]))

    np.testing.assert_allclose(positions, [[10.0, 20.0, 0.0]])
    np.testing.assert_allclose(orientations, [[1.0, 0.0, 0.0, 0.0]])


def test_batched_cumotion_ik_marks_failed_rows_without_throwing() -> None:
    context = _FakeContext(fail_rows={1})
    solver = BatchedCuMotionIKSolver(context, tcp_frame_name="tool")

    result = solver.solve(
        target_positions=np.asarray([[0.1, 0.2, 0.0], [1.0, 2.0, 0.0]]),
        target_orientations_wxyz=None,
        seeds=np.asarray([[10.0, 20.0], [30.0, 40.0]]),
        tcp_frame_name="tool",
    )

    assert result.success.tolist() == [True, False]
    assert result.status == ("SUCCESS", "INVERSE_KINEMATICS_FAILURE")
    assert np.isfinite(result.position_error[0])
    assert np.isinf(result.position_error[1])


def test_batched_cumotion_ik_rejects_wrong_seed_width() -> None:
    solver = BatchedCuMotionIKSolver(_FakeContext(), tcp_frame_name="tool")

    with pytest.raises(ValueError, match="expected_cspace_width"):
        solver.solve(
            target_positions=np.asarray([[0.1, 0.2, 0.0]]),
            target_orientations_wxyz=None,
            seeds=np.asarray([[1.0, 2.0, 3.0]]),
            tcp_frame_name="tool",
        )


def test_batched_cumotion_ik_requires_batch_api() -> None:
    solver = BatchedCuMotionIKSolver(
        _FakeContext(batch_api=False), tcp_frame_name="tool"
    )

    with pytest.raises(RuntimeError, match="batch IK API"):
        solver.solve(
            target_positions=np.asarray([[0.1, 0.2, 0.0]]),
            target_orientations_wxyz=None,
            seeds=np.asarray([[1.0, 2.0]]),
            tcp_frame_name="tool",
        )


def test_old_tiled_cumotion_import_paths_are_removed() -> None:
    import linkerbot_sim.tiled as tiled

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("linkerbot_sim.tiled.cumotion")
    with pytest.raises(AttributeError):
        getattr(tiled, "BatchedCuMotionIKSolver")
    with pytest.raises(AttributeError):
        getattr(tiled, "CuMotionJointMapping")


def test_joint_mapping_rejects_missing_cspace_joint() -> None:
    with pytest.raises(ValueError, match="missing"):
        CuMotionJointMapping.from_joint_names(
            cspace_joint_names=("arm0", "arm1"),
            command_joint_names=("arm0",),
        )
