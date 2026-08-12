from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scripts import kaleidoscope_viewer as viewer


def _newton_runtime_render_runtime(
    *,
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    viewport_reconfigure: object,
    fail_at: str | None = None,
):
    from linkerbot_sim.kaleidoscope.runtime import KaleidoscopeRuntime

    class _Physics:
        kind = "newton_cuda"
        capabilities = SimpleNamespace(rendering=True)

        def render(self) -> None:
            events.append("physics_snapshot_and_update")
            if fail_at == "render":
                raise RuntimeError("render failed")

        def render_update(self) -> None:
            events.append("renderer_update")
            if fail_at == "render_update":
                raise RuntimeError("render_update failed")

    # Runtime 构造器只创建 SAME_STEP buffer 与复用的全环境 selector；渲染编排不依赖 PyTorch。
    # 注入最小替身可让该合同留在纯 CPU/Python 门禁中，并继续覆盖真实构造器 wiring。
    fake_torch = SimpleNamespace(
        bool=object(),
        int64=object(),
        zeros=lambda *_args, **_kwargs: object(),
        arange=lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    device = object()
    return KaleidoscopeRuntime(
        session=SimpleNamespace(physics_runtime=_Physics()),
        views=SimpleNamespace(num_envs=1, device=device),
        action_term=SimpleNamespace(
            action_dim=1,
            action_low=-1.0,
            action_high=1.0,
            physics_ticks_per_action=1,
        ),
        task=SimpleNamespace(
            num_envs=1,
            device=device,
            action_dim=1,
            observation_dim=1,
            settings=SimpleNamespace(physics_ticks_per_action=1),
        ),
        state_api=SimpleNamespace(num_envs=1, device=device, poisoned=False),
        viewport=SimpleNamespace(render_every_n_steps=1),
        viewport_reconfigure=viewport_reconfigure,
    )


def test_newton_runtime_viewport_requires_camera_reconfigure_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="requires viewport_reconfigure"):
        _newton_runtime_render_runtime(
            monkeypatch=monkeypatch,
            events=[],
            viewport_reconfigure=None,
        )


def test_first_newton_runtime_viewport_render_stabilizes_camera_without_resync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runtime = _newton_runtime_render_runtime(
        monkeypatch=monkeypatch,
        events=events,
        viewport_reconfigure=lambda: events.append("camera_reconfigure"),
    )

    runtime.render()
    runtime.render()

    # 第一次附加 update 不得再次调用 physics render，否则会重复 body_q D2H。
    assert events == [
        "physics_snapshot_and_update",
        "camera_reconfigure",
        "renderer_update",
        "physics_snapshot_and_update",
    ]
    assert runtime._viewport_reconfigure_pending is False


@pytest.mark.parametrize("fail_at", ("render", "camera_reconfigure", "render_update"))
def test_newton_runtime_viewport_render_failure_is_fail_stop(
    fail_at: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def reconfigure() -> None:
        events.append("camera_reconfigure")
        if fail_at == "camera_reconfigure":
            raise RuntimeError("camera_reconfigure failed")

    runtime = _newton_runtime_render_runtime(
        monkeypatch=monkeypatch,
        events=events,
        viewport_reconfigure=reconfigure,
        fail_at=fail_at,
    )

    with pytest.raises(RuntimeError, match=f"{fail_at} failed"):
        runtime.render()
    assert runtime._failed is True
    with pytest.raises(RuntimeError, match="fail-stop"):
        runtime.render()


def test_viewer_cli_validates_selected_environment_and_step_count() -> None:
    args = viewer.parse_args(
        [
            "--profile",
            "newton_cuda",
            "--num-envs",
            "4",
            "--selected-env",
            "3",
            "--steps",
            "2",
        ]
    )
    assert args.profile == "newton_cuda"
    assert args.viewport_profile == "kaleidoscope"
    assert (args.num_envs, args.selected_env, args.steps) == (4, 3, 2)

    with pytest.raises(SystemExit):
        viewer.parse_args(["--num-envs", "2", "--selected-env", "2"])
    with pytest.raises(SystemExit):
        viewer.parse_args(["--steps", "-1"])
    with pytest.raises(SystemExit):
        viewer.parse_args(["--visualization-profile", "kaleidoscope"])


def test_native_viewport_factory_keeps_launch_settings_outside_training_config() -> (
    None
):
    from linkerbot_sim.configuration import (
        load_kaleidoscope_config,
        load_kaleidoscope_viewport_config,
    )
    from linkerbot_sim.kaleidoscope.bootstrap import make_viewport_env

    config = load_kaleidoscope_config()
    viewport = load_kaleidoscope_viewport_config()
    calls: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        num_envs=2,
        device="cuda:0",
        action_dim=3,
        action_low=-1.0,
        action_high=1.0,
        observation_dim=7,
        viewport_enabled=True,
        render_every_n_steps=1,
    )

    env = make_viewport_env(
        config=config,
        viewport=viewport,
        num_envs=2,
        runtime_factory=lambda **kwargs: calls.append(dict(kwargs)) or runtime,
    )

    assert env.runtime is runtime
    assert calls == [
        {
            "config": config,
            "num_envs": 2,
            "viewport": viewport,
        }
    ]


def test_viewer_main_uses_explicit_render_cadence_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import linkerbot_sim.configuration as configuration
    import linkerbot_sim.kaleidoscope as kaleidoscope

    events: list[str] = []

    class _Env:
        device = "cpu"
        num_envs = 2
        action_dim = 3
        render_every_n_steps = 2

        def reset(self, *, seed: int) -> None:
            events.append(f"reset:{seed}")

        def render(self) -> None:
            events.append("render")

        def is_running(self) -> bool:
            return True

        def begin_same_step(self) -> object:
            events.append("begin")
            return object()

        def step_same_step(self, token: object, actions: object) -> None:
            del token
            assert tuple(actions.shape) == (2, 3)
            events.append("step")

        def complete_same_step(self, token: object) -> None:
            del token
            events.append("complete")

        def close(self) -> None:
            events.append("close")

    config = SimpleNamespace(physics=SimpleNamespace(engine="physx", execution="cuda"))
    fake_torch = SimpleNamespace(
        float32=object(),
        Generator=lambda **_kwargs: SimpleNamespace(manual_seed=lambda _seed: None),
        zeros=lambda shape, **_kwargs: SimpleNamespace(shape=shape),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(configuration, "load_kaleidoscope_config", lambda _name: config)
    monkeypatch.setattr(
        kaleidoscope,
        "make_viewport_env",
        lambda **_kwargs: _Env(),
    )

    assert (
        viewer.main(
            [
                "--num-envs",
                "2",
                "--selected-env",
                "1",
                "--steps",
                "3",
            ]
        )
        == 0
    )

    assert events == [
        "reset:0",
        "render",
        "begin",
        "step",
        "complete",
        "begin",
        "step",
        "render",
        "complete",
        "begin",
        "step",
        "complete",
        "close",
    ]
    assert "LINKERBOT_KALEIDOSCOPE_VIEWPORT_VALID" in capsys.readouterr().out
