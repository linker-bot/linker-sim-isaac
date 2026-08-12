from __future__ import annotations

import numpy as np
import pytest

from linkerbot_sim.configuration.controllers import (
    ControllerProfile,
    ControllerProfiles,
)
from linkerbot_sim.controllers.control_mode import (
    ControlModeGenerationConflict,
    ControlModeRollbackError,
    ControlModeSwitchError,
)
from linkerbot_sim.controllers.projection import joint_control_settings
from linkerbot_sim.controllers.types import ComponentControlSettings, ControlTargets
from linkerbot_sim.mirror.control_mode import (
    MirrorControlBinding,
    MirrorControlModeService,
)


def _profiles() -> ControllerProfiles:
    profile = ControllerProfile(
        name="default",
        position_control=ComponentControlSettings(
            mode="position", method="implicit", max_force=10.0
        ),
        velocity_control=ComponentControlSettings(
            mode="velocity", method="explicit", damping=2.0, max_force=10.0
        ),
        effort_control=ComponentControlSettings(
            mode="effort", method="direct", effort_limit=10.0
        ),
    )
    return ControllerProfiles(arm=profile, hand=profile, default=profile)


class _Robot:
    num_dof = 2

    def get_joint_positions(self):
        return np.asarray([0.2, -0.3], dtype=float)


class _Controller:
    def __init__(
        self,
        label: str,
        events: list[str],
        profiles: ControllerProfiles,
    ) -> None:
        self.label = label
        self.events = events
        self.profiles = profiles
        self.settings = joint_control_settings(profiles, mode="position")
        self.robot = _Robot()
        self.command_indices = np.asarray([0, 1], dtype=int)
        self.last_commanded_efforts = np.asarray([np.nan, np.nan])
        self._cache: ControlTargets | None = None
        self.fail_mode: str | None = None
        self.fail_rollback = False

    def prepare_runtime(self, settings=None):
        selected = self.settings if settings is None else settings
        self.events.append(f"{self.label}:prepare:{selected.default.mode}")
        return selected

    def apply_prepared_runtime(self, prepared, *, clear_target_cache: bool) -> None:
        mode = prepared.default.mode
        self.events.append(f"{self.label}:mode:{mode}")
        if mode == self.fail_mode or (mode == "position" and self.fail_rollback):
            raise RuntimeError(f"{self.label} rejected {mode}")
        self.settings = prepared
        if clear_target_cache:
            self._cache = None

    def build_control_targets(self, **_kwargs) -> ControlTargets:
        return ControlTargets(
            positions=np.asarray([0.2, -0.3]),
            velocities=np.zeros(2),
            efforts=np.zeros(2),
        )

    def apply_targets(self, _action_type, targets: ControlTargets) -> None:
        self.events.append(f"{self.label}:target:{self.settings.default.mode}")
        self._cache = ControlTargets(
            targets.positions,
            targets.velocities,
            targets.efforts,
        )

    def snapshot_control_targets_cache(self):
        return self._cache

    def restore_control_targets_cache(self, targets) -> None:
        self._cache = targets


class _Runtime:
    def __init__(self) -> None:
        self.fatal_error = None
        self.quit_requested = False

    def request_quit(self) -> None:
        self.quit_requested = True


def _service(*controllers: _Controller):
    runtime = _Runtime()
    service = MirrorControlModeService(
        initial_mode="position",
        bindings=tuple(
            MirrorControlBinding(
                label=controller.label,
                controller=controller,
                controller_profiles=controller.profiles,
                articulation_action_type=object,
            )
            for controller in controllers
        ),
    )
    service.bind_runtime(runtime)
    return service, runtime


def test_mirror_mode_switch_is_global_neutral_and_generation_tracked() -> None:
    events: list[str] = []
    profiles = _profiles()
    left = _Controller("left", events, profiles)
    right = _Controller("right", events, profiles)
    service, _runtime = _service(left, right)

    change = service.set_mode("velocity", expected_generation=0)

    assert change.as_dict() == {
        "previous_mode": "position",
        "active_mode": "velocity",
        "generation": 1,
        "changed": True,
    }
    assert left.settings.default.mode == "velocity"
    assert right.settings.default.mode == "velocity"
    assert events.index("left:target:position") < events.index("left:mode:velocity")
    assert events.index("left:mode:velocity") < events.index("left:target:velocity")
    assert service.get_mode().active_mode == "velocity"


def test_mirror_mode_switch_is_idempotent_and_checks_generation_first() -> None:
    events: list[str] = []
    controller = _Controller("left", events, _profiles())
    service, _runtime = _service(controller)

    unchanged = service.set_mode("position", expected_generation=0)

    assert unchanged.changed is False
    assert unchanged.generation == 0
    assert events == []
    with pytest.raises(ControlModeGenerationConflict):
        service.set_mode("position", expected_generation=2)


def test_mirror_forward_failure_rolls_back_all_written_robots() -> None:
    events: list[str] = []
    profiles = _profiles()
    left = _Controller("left", events, profiles)
    right = _Controller("right", events, profiles)
    right.fail_mode = "velocity"
    service, runtime = _service(left, right)

    with pytest.raises(ControlModeSwitchError, match="right rejected velocity"):
        service.set_mode("velocity")

    assert service.get_mode().active_mode == "position"
    assert service.get_mode().generation == 0
    assert left.settings.default.mode == "position"
    assert right.settings.default.mode == "position"
    assert runtime.fatal_error is None
    right_rollback = len(events) - 1 - events[::-1].index("right:mode:position")
    left_rollback = len(events) - 1 - events[::-1].index("left:mode:position")
    assert right_rollback < left_rollback


def test_mirror_rollback_failure_marks_runtime_fatal() -> None:
    events: list[str] = []
    profiles = _profiles()
    left = _Controller("left", events, profiles)
    right = _Controller("right", events, profiles)
    right.fail_mode = "velocity"
    left.fail_rollback = True
    service, runtime = _service(left, right)

    with pytest.raises(ControlModeRollbackError):
        service.set_mode("velocity")

    assert runtime.fatal_error is not None
    assert runtime.quit_requested is True
