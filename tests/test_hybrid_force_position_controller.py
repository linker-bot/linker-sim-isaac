from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.configuration.control import HybridForcePositionSettings
from linkerbot_sim.controllers.hybrid_force_position import (
    HybridControlError,
    HybridControlParameters,
    HybridControlTarget,
    HybridForcePositionController,
    HybridSingularityError,
    TaskSpaceObservation,
    rotation_vector_error_wxyz,
)
from linkerbot_sim.utils.config import load_yaml


def _settings() -> HybridForcePositionSettings:
    document = load_yaml("configs/control/hybrid_force_position.yaml")
    return HybridForcePositionSettings.from_mapping(document["hybrid_force_position"])


def _parameters(settings: HybridForcePositionSettings) -> HybridControlParameters:
    return HybridControlParameters(
        motion_stiffness=settings.motion.stiffness,
        motion_damping=settings.motion.damping,
        force_proportional=settings.force.proportional,
        force_integral=settings.force.integral,
        posture_stiffness=settings.posture.stiffness,
        posture_damping=settings.posture.damping,
    )


def _target(*, target_x: float = 0.0, target_fz: float = -8.0):
    return HybridControlTarget(
        position=np.asarray([target_x, 0.0, 0.0]),
        orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        force_axes=np.asarray([False, False, True, False, False, False]),
        wrench_tool_on_environment=np.asarray([0.0, 0.0, target_fz, 0.0, 0.0, 0.0]),
    )


def _observation(
    *,
    sequence: int,
    external_fz: float = 8.0,
    jacobian: np.ndarray | None = None,
) -> TaskSpaceObservation:
    return TaskSpaceObservation(
        position=np.zeros(3),
        orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        twist=np.zeros(6),
        jacobian=np.eye(6) if jacobian is None else jacobian,
        joint_positions=np.zeros(6),
        joint_velocities=np.zeros(6),
        external_wrench_environment_on_tool=np.asarray(
            [0.0, 0.0, external_fz, 0.0, 0.0, 0.0]
        ),
        sequence=sequence,
    )


def _controller(
    *,
    settings: HybridForcePositionSettings | None = None,
    target: HybridControlTarget | None = None,
) -> HybridForcePositionController:
    selected = _settings() if settings is None else settings
    return HybridForcePositionController(
        settings=selected,
        parameters=_parameters(selected),
        target=_target() if target is None else target,
        tare_external_wrench=np.zeros(6),
        nominal_joint_positions=np.zeros(6),
        initial_position=np.zeros(3),
        initial_orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        joint_effort_limits=np.full(6, 100.0),
    )


def test_force_feedback_uses_tool_on_environment_sign_and_axis_selection() -> None:
    controller = _controller()

    output = controller.step(_observation(sequence=0), dt=0.01)

    assert output.measured_wrench_tool_on_environment[2] == pytest.approx(-8.0)
    assert output.force_wrench[2] == pytest.approx(-8.0)
    np.testing.assert_allclose(output.force_wrench[[0, 1, 3, 4, 5]], 0.0)
    assert output.joint_efforts[2] == pytest.approx(-2.0)  # effort-rate limited


def test_motion_impedance_only_acts_on_motion_axes() -> None:
    target = _target(target_x=0.01, target_fz=0.0)
    controller = _controller(target=target)

    output = controller.step(_observation(sequence=0, external_fz=0.0), dt=0.01)

    assert output.motion_wrench[0] == pytest.approx(2.0)
    assert output.motion_wrench[2] == 0.0
    assert output.commanded_wrench[0] == pytest.approx(2.0)


def test_contact_hysteresis_and_stale_observation_are_stateful() -> None:
    controller = _controller()

    for sequence in range(3):
        output = controller.step(_observation(sequence=sequence), dt=0.01)
    assert output.contact_axes[2]

    with pytest.raises(HybridControlError, match="stale"):
        controller.step(_observation(sequence=2), dt=0.01)


def test_singular_jacobian_fails_closed() -> None:
    controller = _controller()
    jacobian = np.eye(6)
    jacobian[-1] = 0.0

    with pytest.raises(HybridSingularityError):
        controller.step(_observation(sequence=0, jacobian=jacobian), dt=0.01)


def test_quaternion_error_uses_shortest_arc_and_sign_equivalence() -> None:
    np.testing.assert_allclose(
        rotation_vector_error_wxyz(
            np.asarray([-1.0, 0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        ),
        0.0,
        atol=1.0e-12,
    )
    angle = 0.1
    error = rotation_vector_error_wxyz(
        np.asarray([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]),
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(error, [0.0, 0.0, angle], atol=1.0e-12)
