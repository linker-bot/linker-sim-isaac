from __future__ import annotations

import pytest

from linkerbot_sim.configuration.catalog import load_mirror_config
from linkerbot_sim.mirror.hybrid_parameters import (
    HybridNotConfiguredError,
    HybridParameterGenerationConflict,
    HybridParameterOutOfRange,
    HybridParameterService,
)


def _service() -> HybridParameterService:
    settings = load_mirror_config("physx_cpu_hybrid").hybrid_control
    assert settings is not None
    return HybridParameterService(settings)


def test_initial_state_comes_from_strict_profile_and_exposes_bounds() -> None:
    state = _service().get_state()

    assert state["generation"] == 0
    parameters = state["parameters"]
    limits = state["tuning_limits"]
    assert isinstance(parameters, dict)
    assert isinstance(limits, dict)
    assert parameters["motion_stiffness"] == [
        200.0,
        200.0,
        200.0,
        10.0,
        10.0,
        10.0,
    ]
    assert limits["motion_stiffness"] == [
        500.0,
        500.0,
        500.0,
        30.0,
        30.0,
        30.0,
    ]


def test_partial_update_is_atomic_and_generation_checked() -> None:
    service = _service()
    before = service.snapshot()
    changed = service.set_parameters(
        {
            "motion_stiffness": [100, 101, 102, 3, 4, 5],
            "force_integral": [0.1, 0.2, 0.3, 0.01, 0.02, 0.03],
        },
        expected_generation=0,
    )

    assert changed.changed is True
    assert changed.previous_generation == 0
    assert changed.generation == 1
    assert before.generation == 0
    assert before.values.motion_stiffness == (
        200.0,
        200.0,
        200.0,
        10.0,
        10.0,
        10.0,
    )
    assert service.snapshot().values.motion_stiffness == (
        100.0,
        101.0,
        102.0,
        3.0,
        4.0,
        5.0,
    )
    assert service.snapshot().values.motion_damping == before.values.motion_damping

    with pytest.raises(HybridParameterGenerationConflict) as exc_info:
        service.set_parameters({"posture_damping": 2.0}, expected_generation=0)
    assert (exc_info.value.expected, exc_info.value.actual) == (0, 1)


def test_identical_update_does_not_advance_generation() -> None:
    service = _service()
    change = service.set_parameters({"posture_stiffness": 5.0})

    assert change.changed is False
    assert change.generation == 0


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({}, "at least one"),
        ({"motion_stiffness": [1, 2]}, "exactly six"),
        ({"motion_stiffness": [1, 2, 3, 4, 5, True]}, "finite"),
        ({"posture_stiffness": -1}, "non-negative"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_invalid_updates_do_not_mutate_state(
    updates: dict[str, object], message: str
) -> None:
    service = _service()
    before = service.snapshot()

    with pytest.raises(ValueError, match=message):
        service.set_parameters(updates)

    assert service.snapshot() == before


@pytest.mark.parametrize(
    "updates",
    [
        {"motion_stiffness": [501, 0, 0, 0, 0, 0]},
        {"force_integral": [0, 0, 0, 0, 0, 0.51]},
        {"posture_damping": 5.01},
    ],
)
def test_yaml_bounds_reject_entire_update(updates: dict[str, object]) -> None:
    service = _service()
    before = service.snapshot()

    with pytest.raises(HybridParameterOutOfRange):
        service.set_parameters(updates)

    assert service.snapshot() == before


def test_unconfigured_service_fails_closed() -> None:
    service = HybridParameterService(None)

    assert service.configured is False
    with pytest.raises(HybridNotConfiguredError):
        service.get_state()
    with pytest.raises(HybridNotConfiguredError):
        service.set_parameters({"posture_stiffness": 1.0})
