from __future__ import annotations

import pytest

from linkerbot_sim.mirror.timing import WallClockStepSynchronizer


class _FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration_s: float) -> None:
        self.sleeps.append(duration_s)
        self.now += duration_s


def test_wall_clock_synchronizer_sleeps_only_for_remaining_tick_time() -> None:
    clock = _FakeClock(now=10.0)
    synchronizer = WallClockStepSynchronizer(
        enabled=True,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    synchronizer.before_step(0.1)
    clock.now += 0.04
    synchronizer.before_step(0.1)
    clock.now += 0.07
    synchronizer.before_step(0.1)

    assert clock.sleeps == pytest.approx([0.06, 0.03])


def test_wall_clock_synchronizer_disabled_never_sleeps() -> None:
    clock = _FakeClock()
    synchronizer = WallClockStepSynchronizer(
        enabled=False,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    synchronizer.before_step(0.1)
    clock.now += 0.01
    synchronizer.before_step(0.1)

    assert clock.sleeps == []


def test_wall_clock_synchronizer_rebases_after_lag_without_burst_catchup() -> None:
    clock = _FakeClock()
    synchronizer = WallClockStepSynchronizer(
        enabled=True,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    synchronizer.before_step(0.1)
    clock.now = 0.35
    synchronizer.before_step(0.1)
    clock.now = 0.36
    synchronizer.before_step(0.1)

    assert clock.sleeps == pytest.approx([0.09])


def test_wall_clock_synchronizer_explicit_rebase_makes_next_step_immediate() -> None:
    clock = _FakeClock()
    synchronizer = WallClockStepSynchronizer(
        enabled=True,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    synchronizer.before_step(0.1)
    clock.now = 0.02
    synchronizer.rebase()
    synchronizer.before_step(0.1)

    assert clock.sleeps == []


@pytest.mark.parametrize("physics_dt_s", (0.0, -0.1, float("nan")))
def test_wall_clock_synchronizer_rejects_invalid_physics_dt(
    physics_dt_s: float,
) -> None:
    synchronizer = WallClockStepSynchronizer(enabled=False)

    with pytest.raises(ValueError, match="physics_dt_s"):
        synchronizer.before_step(physics_dt_s)
