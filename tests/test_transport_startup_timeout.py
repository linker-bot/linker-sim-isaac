from __future__ import annotations

from types import SimpleNamespace

import pytest

from linkerbot_sim.app.interactive.policies import InteractiveRuntimePolicy
from linkerbot_sim.app.interactive.single_scene import runtime as single_scene_runtime
from linkerbot_sim.configs.runtime import (
    InteractiveRuntimeSettings,
    RuntimeProfileConfig,
    RuntimeTransportSettings,
)


def test_runtime_profile_parses_transport_startup_timeout() -> None:
    profile = RuntimeProfileConfig.from_mapping(
        {
            "runtime": {"interactive": {"transport": {"startup_timeout_s": 0.25}}},
        }
    )

    assert profile.interactive.transport.startup_timeout_s == pytest.approx(0.25)


@pytest.mark.parametrize("value", (0, -1, True, "5"))
def test_runtime_profile_rejects_invalid_transport_startup_timeout(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"runtime\.interactive\.transport\.startup_timeout_s",
    ):
        RuntimeProfileConfig.from_mapping(
            {
                "runtime": {"interactive": {"transport": {"startup_timeout_s": value}}},
            }
        )


def test_single_scene_runtime_forwards_transport_startup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    transports = SimpleNamespace(
        stop=lambda **_kwargs: SimpleNamespace(stopped=True, live_resources=())
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "TimelinePlanningSession",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        single_scene_runtime,
        "start_interactive_state_stream",
        lambda *_args, **_kwargs: None,
    )

    def fake_start_interactive_transports(**kwargs: object):
        captured.update(kwargs)
        return transports

    monkeypatch.setattr(
        single_scene_runtime,
        "start_interactive_transports",
        fake_start_interactive_transports,
    )
    runtime = SimpleNamespace(
        session=SimpleNamespace(app=SimpleNamespace(is_running=lambda: False)),
        status=lambda: {},
        status_prefix="TEST_SCENE",
        camera_output=None,
    )

    single_scene_runtime.run_single_scene_interactive_motion(
        runtime,
        stdin_enabled=False,
        policy=InteractiveRuntimePolicy(
            stdin_eof_policy="exit",
            idle_physics_policy="pause",
        ),
        interactive_settings=InteractiveRuntimeSettings(
            transport=RuntimeTransportSettings(startup_timeout_s=0.375)
        ),
    )

    assert captured["startup_timeout_s"] == pytest.approx(0.375)
