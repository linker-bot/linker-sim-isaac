from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from linkerbot_sim.configuration import load_kaleidoscope_config  # noqa: E402
from linkerbot_sim.controllers.control_mode import (  # noqa: E402
    ControlModeGenerationConflict,
    ControlModeIncompatibleError,
    ControlModeLockedError,
    ControlModeRollbackError,
    ControlModeSwitchError,
)
from linkerbot_sim.kaleidoscope.control_mode import (  # noqa: E402
    KaleidoscopeControlBinding,
    KaleidoscopeControlModeCoordinator,
)
from linkerbot_sim.kaleidoscope.runtime import (  # noqa: E402
    KaleidoscopeRuntime,
    SameStepToken,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Kaleidoscope control mode",
)


class _Port:
    def __init__(self, label: str, q: torch.Tensor) -> None:
        self.label = label
        self.device = q.device
        self.command_dim = q.shape[1]
        self.q = q.clone()
        self.active = "position"
        self.targets = {
            mode: torch.zeros_like(q) for mode in ("position", "velocity", "effort")
        }
        self.events: list[tuple[str, str]] = []
        self.fail_apply: set[str] = set()

    def read_joint_positions(self, ids):
        self.events.append(("read", self.active))
        return self.q.index_select(0, ids)

    def prepare_control_runtime(self, projection):
        return SimpleNamespace(
            common=SimpleNamespace(
                mode=projection.modes[0],
                effort_limits=torch.tensor(
                    projection.effort_limits.tolist(),
                    device=self.device,
                    dtype=torch.float32,
                ),
            ),
            projection=projection,
        )

    def validate_prepared_control_runtime(self, prepared):
        assert prepared.common.mode in {"position", "velocity", "effort"}

    def apply_prepared_control_runtime(self, prepared):
        mode = prepared.common.mode
        self.events.append(("apply", mode))
        if mode in self.fail_apply:
            raise RuntimeError(f"apply {self.label} {mode} failed")
        self.active = mode

    def write_joint_position_targets(self, ids, values):
        self.events.append(("target", "position"))
        self.targets["position"].index_copy_(0, ids, values)

    def write_joint_velocity_targets(self, ids, values):
        self.events.append(("target", "velocity"))
        self.targets["velocity"].index_copy_(0, ids, values)

    def write_joint_effort_targets(self, ids, values):
        self.events.append(("target", "effort"))
        self.targets["effort"].index_copy_(0, ids, values)

    def synchronize_control_writes(self):
        self.events.append(("sync", self.active))


class _Views:
    def __init__(self, ports: tuple[_Port, ...]) -> None:
        self.robot_ports = ports
        self.num_envs = ports[0].q.shape[0]
        self.device = ports[0].device
        self.command_dim = sum(port.command_dim for port in ports)
        self.position_references = torch.zeros(
            (self.num_envs, self.command_dim), device=self.device
        )
        self.control_targets = torch.zeros_like(self.position_references)
        self.provider = lambda: "position"

    def bind_control_mode_provider(self, provider):
        self.provider = provider


class _RuntimeOwner:
    def __init__(self) -> None:
        self.fatal_error = None

    def mark_control_mode_fatal(self, message: str) -> None:
        self.fatal_error = message


def _coordinator_inputs():
    config = load_kaleidoscope_config()
    profiles = config.controller_bundles[config.default_controller_bundle]
    names = config.scene.robots[0].resolved_profile.joint_groups.arm[:2]
    q0 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device="cuda")
    q1 = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]], device="cuda")
    ports = (_Port("left", q0), _Port("right", q1))
    views = _Views(ports)
    bindings = tuple(
        KaleidoscopeControlBinding(
            label=port.label,
            port=port,
            controller_profiles=profiles,
            command_joint_names=tuple(names),
            components=("arm", "arm"),
        )
        for port in ports
    )
    return views, ports, bindings


def _coordinator():
    views, ports, bindings = _coordinator_inputs()
    coordinator = KaleidoscopeControlModeCoordinator(
        views=views,
        bindings=bindings,
        supported_modes=("position", "velocity", "effort"),
    )
    owner = _RuntimeOwner()
    coordinator.bind_runtime(owner)
    for port in ports:
        port.events.clear()
    return coordinator, owner, views, ports


@pytest.mark.parametrize("invalid", ("shape", "dtype", "alias"))
def test_coordinator_rejects_invalid_canonical_buffers_before_engine_write(
    invalid: str,
) -> None:
    views, ports, bindings = _coordinator_inputs()
    if invalid == "shape":
        views.position_references = views.position_references[:, :-1]
    elif invalid == "dtype":
        views.control_targets = views.control_targets.to(dtype=torch.float64)
    else:
        views.control_targets = views.position_references

    with pytest.raises((TypeError, ValueError)):
        KaleidoscopeControlModeCoordinator(
            views=views,
            bindings=bindings,
            supported_modes=("position", "velocity", "effort"),
        )

    assert all(port.events == [] for port in ports)


def test_kaleidoscope_switch_is_global_neutral_and_generation_guarded() -> None:
    coordinator, _owner, views, ports = _coordinator()

    first = coordinator.set_mode("velocity", expected_generation=0)

    assert first.changed and first.generation == 1
    assert coordinator.get_mode().active_mode == "velocity"
    assert views.provider() == "velocity"
    torch.testing.assert_close(
        views.control_targets, torch.zeros_like(views.control_targets)
    )
    torch.testing.assert_close(
        views.position_references,
        torch.cat([port.q for port in ports], dim=1),
    )
    assert all(port.active == "velocity" for port in ports)
    events_before = tuple(tuple(port.events) for port in ports)
    same = coordinator.set_mode("velocity", expected_generation=1)
    assert not same.changed and same.generation == 1
    assert tuple(tuple(port.events) for port in ports) == events_before

    with pytest.raises(ControlModeGenerationConflict):
        coordinator.set_mode("effort", expected_generation=0)


def test_kaleidoscope_forward_failure_rolls_back_in_reverse_and_stays_usable() -> None:
    coordinator, owner, views, ports = _coordinator()
    original_target = views.control_targets.clone()
    ports[1].fail_apply.add("velocity")

    with pytest.raises(ControlModeSwitchError, match="right velocity"):
        coordinator.set_mode("velocity")

    assert coordinator.get_mode().active_mode == "position"
    assert coordinator.get_mode().generation == 0
    assert owner.fatal_error is None
    assert all(port.active == "position" for port in ports)
    torch.testing.assert_close(views.control_targets, original_target)
    assert ports[1].events[-4:] == [
        ("apply", "position"),
        ("target", "velocity"),
        ("target", "position"),
        ("sync", "position"),
    ]

    ports[1].fail_apply.clear()
    assert coordinator.set_mode("effort").changed


def test_kaleidoscope_rollback_failure_marks_runtime_fatal() -> None:
    coordinator, owner, _views, ports = _coordinator()
    ports[1].fail_apply.update({"velocity", "position"})

    with pytest.raises(ControlModeRollbackError):
        coordinator.set_mode("velocity")

    assert owner.fatal_error is not None
    assert coordinator.get_mode().active_mode == "position"
    assert coordinator.get_mode().generation == 0


def test_position_mode_always_commits_feedforward_before_position_target() -> None:
    coordinator, _owner, _views, ports = _coordinator()
    coordinator.set_mode("velocity")
    coordinator.set_mode("effort")
    for port in ports:
        port.events.clear()

    coordinator.set_mode("position")

    for port in ports:
        assert port.events == [
            ("read", "effort"),
            ("target", "effort"),
            ("apply", "position"),
            ("target", "velocity"),
            ("target", "position"),
            ("sync", "position"),
        ]


def test_runtime_rejects_mode_switch_during_same_step_and_non_idle_phase() -> None:
    runtime = object.__new__(KaleidoscopeRuntime)
    runtime._closed = False
    runtime._closing_started = False
    runtime._failed = False
    runtime._fatal_error = None
    runtime._phase = "idle"
    runtime._outstanding_token = SameStepToken(0)
    runtime.state_api = SimpleNamespace(poisoned=False)
    runtime.control_mode = SimpleNamespace(
        set_mode=lambda *_args, **_kwargs: pytest.fail("switch must not be delegated")
    )

    with pytest.raises(RuntimeError, match="SAME_STEP"):
        runtime.set_control_mode("velocity")

    runtime._outstanding_token = None
    runtime._phase = "step"
    with pytest.raises(ControlModeLockedError, match="phase"):
        runtime.set_control_mode("velocity")


def test_runtime_rejects_incompatible_action_before_action_or_physics_write() -> None:
    runtime = object.__new__(KaleidoscopeRuntime)
    runtime._closed = False
    runtime._closing_started = False
    runtime._failed = False
    runtime._fatal_error = None
    runtime.state_api = SimpleNamespace(poisoned=False)
    runtime.views = SimpleNamespace(refresh=lambda: SimpleNamespace())
    runtime.control_mode = SimpleNamespace(active_mode="effort")
    runtime.action_term = SimpleNamespace(
        supported_control_modes=("position", "velocity"),
        apply=lambda *_args: pytest.fail("incompatible action must not be evaluated"),
    )
    runtime.session = SimpleNamespace(
        physics_runtime=SimpleNamespace(
            step=lambda **_kwargs: pytest.fail("physics must not advance")
        )
    )

    with pytest.raises(ControlModeIncompatibleError):
        runtime._step_core(torch.zeros((1, 1), device="cuda"))
