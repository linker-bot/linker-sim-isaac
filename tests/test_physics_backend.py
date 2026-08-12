from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest

from linkerbot_sim.assets.solver_overrides import (
    SolverIterationConfig,
    apply_solver_iteration_overrides,
)
from linkerbot_sim.isaac.physics.backend import (
    active_physics_backend,
    normalize_physics_backend,
    require_physics_backend,
)


def test_normalize_physics_backend_is_strict() -> None:
    assert normalize_physics_backend(" PhysX ") == "physx"
    assert normalize_physics_backend("NEWTON") == "newton"
    with pytest.raises(ValueError, match="unsupported physics backend"):
        normalize_physics_backend("auto")


def test_active_physics_backend_uses_simulation_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("isaacsim.core.simulation_manager")
    module.SimulationManager = SimpleNamespace(  # type: ignore[attr-defined]
        get_active_physics_engine=lambda: "newton"
    )
    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(sys.modules, "isaacsim.core.simulation_manager", module)

    assert active_physics_backend() == "newton"


def test_active_physics_backend_does_not_hide_invalid_runtime_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("isaacsim.core.simulation_manager")
    module.SimulationManager = SimpleNamespace(  # type: ignore[attr-defined]
        get_active_physics_engine=lambda: "remotesim"
    )
    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(sys.modules, "isaacsim.core.simulation_manager", module)

    with pytest.raises(ValueError, match="unsupported physics backend"):
        active_physics_backend(fallback="physx")


def test_active_physics_backend_propagates_runtime_getter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> str:
        raise RuntimeError("engine registry unavailable")

    module = ModuleType("isaacsim.core.simulation_manager")
    module.SimulationManager = SimpleNamespace(  # type: ignore[attr-defined]
        get_active_physics_engine=fail
    )
    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(sys.modules, "isaacsim.core.simulation_manager", module)

    with pytest.raises(RuntimeError, match="engine registry unavailable"):
        active_physics_backend(fallback="physx")


def test_require_physics_backend_fails_closed() -> None:
    assert (
        require_physics_backend("physx", feature="solver", backend="physx") == "physx"
    )
    with pytest.raises(RuntimeError, match="solver requires physics backend"):
        require_physics_backend("physx", feature="solver", backend="newton")


def test_tgs_solver_rejects_newton_before_importing_isaac() -> None:
    with pytest.raises(
        RuntimeError, match="TGS.*no behavior-preserving Newton mapping"
    ):
        apply_solver_iteration_overrides(
            object(),
            "/World/Robot",
            SolverIterationConfig(solver_type="TGS"),
            physics_backend="newton",
        )
