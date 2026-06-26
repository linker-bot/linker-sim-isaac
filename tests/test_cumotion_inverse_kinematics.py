from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from manipulation_project.backends.cumotion import inverse_kinematics as ik_module
from manipulation_project.backends.cumotion.inverse_kinematics import (
    CuMotionInverseKinematics,
)
from manipulation_project.planning.requests import IKRequest


class _FakeRotation:
    def __init__(self, matrix=None) -> None:
        self._matrix = np.eye(3) if matrix is None else np.asarray(matrix, dtype=float)

    def matrix(self):
        return self._matrix


class _FakePose:
    def __init__(self, translation=None, rotation=None) -> None:
        self.translation = np.zeros(3) if translation is None else translation
        self.rotation = _FakeRotation(rotation)


class _FakeKinematics:
    def __init__(self) -> None:
        self.pose_calls = []

    def pose(self, joint_positions, frame_name):
        self.pose_calls.append((joint_positions, frame_name))
        return _FakePose(translation=np.asarray([0.1, 0.0, 0.0], dtype=float))


class _FakeParamValue:
    def __init__(self, value) -> None:
        self.value = value


class _FakeSolverConfig:
    def __init__(self) -> None:
        self.params = []

    def set_param(self, name, value):
        self.params.append((name, value.value))
        return True


class _FakeTranslationConstraint:
    calls = []

    @staticmethod
    def target(translation_target, deviation_limit=None):
        _FakeTranslationConstraint.calls.append((translation_target, deviation_limit))
        return ("translation", np.asarray(translation_target), deviation_limit)


class _FakeOrientationConstraint:
    @staticmethod
    def none():
        return ("orientation_none",)

    @staticmethod
    def target(orientation_target, deviation_limit=None):
        return ("orientation", orientation_target, deviation_limit)


class _FakeTaskSpaceTarget:
    def __init__(self, translation, orientation) -> None:
        self.translation = translation
        self.orientation = orientation


class _FakeResultsStatus:
    SUCCESS = "SUCCESS"


class _FakeResults:
    class Status(_FakeResultsStatus):
        pass

    def status(self):
        return self.Status.SUCCESS

    def cspace_positions(self):
        return [np.asarray([0.2, 0.4], dtype=float)]

    def target_indices(self):
        return [0]


class _FakeSolver:
    def __init__(self, cumotion) -> None:
        self.cumotion = cumotion

    def solve(self, target, seeds):
        self.cumotion.solve_calls.append((target, seeds))
        return _FakeResults()


class _FakeRotation3:
    @staticmethod
    def from_matrix(matrix):
        return _FakeRotation(matrix)


class _FakeCumotion:
    CollisionFreeIkSolverConfig = SimpleNamespace(ParamValue=_FakeParamValue)
    CollisionFreeIkSolver = SimpleNamespace(
        TranslationConstraint=_FakeTranslationConstraint,
        OrientationConstraint=_FakeOrientationConstraint,
        TaskSpaceTarget=_FakeTaskSpaceTarget,
        Results=_FakeResults,
    )
    Rotation3 = _FakeRotation3

    def __init__(self) -> None:
        self.config = None
        self.solve_calls = []

    def IkConfig(self):
        return SimpleNamespace()

    def create_default_collision_free_ik_solver_config(
        self, robot_description, frame_name, world_view
    ):
        self.config = _FakeSolverConfig()
        self.config_call = (robot_description, frame_name, world_view)
        return self.config

    def create_collision_free_ik_solver(self, config):
        self.solver_config = config
        return _FakeSolver(self)


class _FakeContext:
    def __init__(self) -> None:
        self.cumotion = _FakeCumotion()
        self.kinematics = _FakeKinematics()
        self.robot_description = "robot"
        self.config = SimpleNamespace(
            ik_cspace_seeds=np.asarray([0.0, 0.0]),
            ccd_max_iterations=10,
            bfgs_max_iterations=20,
            orientation_weight=0.5,
            collision_free_ik_params={"max_iterations": 7},
        )

    def joint_names(self):
        return ["j0", "j1"]

    def frame_names(self):
        return ["tool"]

    def has_frame(self, frame_name):
        return frame_name == "tool"


def test_collision_free_ik_uses_tolerances_params_and_recomputes_errors(
    monkeypatch,
) -> None:
    context = _FakeContext()

    def fake_make_collision_world(_context, collision_objects):
        assert _context is context
        assert collision_objects == ()
        return SimpleNamespace(world_view="world")

    monkeypatch.setattr(ik_module, "make_collision_world", fake_make_collision_world)
    _FakeTranslationConstraint.calls = []
    solver = CuMotionInverseKinematics(context, tcp_frame_name="tool")

    result = solver.solve(
        IKRequest(
            target_position=np.asarray([0.0, 0.0, 0.0]),
            target_orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
            warm_start=np.asarray([0.1, 0.2]),
            position_tolerance=0.012,
            orientation_tolerance=0.034,
            avoid_collisions=True,
        )
    )

    assert result.success
    np.testing.assert_allclose(result.joint_positions, [0.2, 0.4])
    assert result.position_error == 0.1
    assert result.orientation_error == 0.0
    assert context.cumotion.config.params == [("max_iterations", 7)]
    translation_target, deviation_limit = _FakeTranslationConstraint.calls[0]
    np.testing.assert_allclose(translation_target, [0.0, 0.0, 0.0])
    assert deviation_limit == 0.012
    _target, seeds = context.cumotion.solve_calls[0]
    np.testing.assert_allclose(seeds[0], [0.1, 0.2])


def test_ik_request_rejects_wrong_warm_start_length() -> None:
    context = _FakeContext()
    solver = CuMotionInverseKinematics(context, tcp_frame_name="tool")

    try:
        solver.solve(
            IKRequest(
                target_position=np.zeros(3),
                warm_start=np.asarray([0.0]),
            )
        )
    except ValueError as exc:
        assert "warm_start expected 2 values" in str(exc)
    else:
        raise AssertionError("expected warm_start length validation")
