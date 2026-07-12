from __future__ import annotations

import pytest

from linkerbot_sim.app.interactive.policies import (
    InteractiveRuntimePolicy,
    resolve_interactive_runtime_policy,
)


def test_interactive_policy_keeps_liveness_and_idle_physics_independent() -> None:
    policy = resolve_interactive_runtime_policy(
        stdin_eof_policy="keep_alive",
        idle_physics_policy="pause",
    )

    assert policy == InteractiveRuntimePolicy(
        stdin_eof_policy="keep_alive",
        idle_physics_policy="pause",
    )
    assert policy.keeps_alive_on_stdin_eof is True
    assert policy.steps_while_idle is False


def test_interactive_policy_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="stdin_eof_policy"):
        resolve_interactive_runtime_policy(stdin_eof_policy="unknown")
    with pytest.raises(ValueError, match="idle_physics_policy"):
        resolve_interactive_runtime_policy(idle_physics_policy="unknown")
