from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from linkerbot_sim.app.interactive.single_scene import cli as single_scene_cli
from linkerbot_sim.app.interactive.tiled_scene import cli as tiled_scene_cli
from linkerbot_sim.configs.runtime import (
    CameraOutputRuntimeSettings,
    PlannerRequestDefaults,
    PlaybackResourceSettings,
    RuntimeCommandDefaults,
    ShutdownSettings,
    SimulationAppSettings,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_bundled_tiled_scene(args):
    """Resolve the canonical bundled Tiled Scene graph."""

    return tiled_scene_cli.resolve_entry_config(args)


def test_entry_parsers_leave_runtime_overrides_unset() -> None:
    single_scene = single_scene_cli.parse_args([])
    tiled_scene = tiled_scene_cli.parse_args([])

    assert single_scene.runtime_profile == "default_single_scene"
    assert single_scene.env is None
    assert single_scene.gui is None
    assert single_scene.planner_backend is None
    assert single_scene.state_include_efforts is None
    assert single_scene.state_include_objects is None

    assert tiled_scene.runtime_profile == "default_tiled_scene"
    assert tiled_scene.env is None
    assert tiled_scene.gui is None
    assert tiled_scene.stdin_enabled is None
    assert tiled_scene.default_decimation is None
    assert tiled_scene.planner_workers is None
    assert tiled_scene.telemetry_full_batch_json is None
    assert tiled_scene.telemetry_joint_states is None


def test_single_scene_cli_overrides_runtime_profile_and_records_provenance() -> None:
    args = single_scene_cli.parse_args(
        [
            "--gui",
            "--planner-backend",
            "linear",
            "--tcp-jsonl-port",
            "7231",
            "--state-rate-hz",
            "25",
            "--state-include-objects",
        ]
    )

    resolved, _env_config = single_scene_cli.resolve_entry_config(args)

    assert resolved.mode == "single_scene"
    assert resolved.simulation_app.gui is True
    assert resolved.planner.backend == "linear"
    assert resolved.interactive.transport.tcp_jsonl.enabled is True
    assert resolved.interactive.transport.tcp_jsonl.port == 7231
    assert resolved.telemetry.rate_hz == 25.0
    assert resolved.telemetry.include_objects is True
    assert resolved.sources["runtime.planner.backend"] == "cli"


def test_tiled_scene_cli_overrides_runtime_profile_and_records_provenance() -> None:
    args = tiled_scene_cli.parse_args(
        [
            "--no-stdin",
            "--default-decimation",
            "4",
            "--planner-workers",
            "3",
            "--websocket-port",
            "7232",
            "--telemetry-primary-env-id",
            "2",
            "--telemetry-env-ids",
            "2,3",
            "--telemetry-decimation",
            "5",
            "--no-telemetry-full-batch-json",
        ]
    )

    resolved, _env_config = _resolve_bundled_tiled_scene(args)

    assert resolved.mode == "tiled_scene"
    assert resolved.interactive.stdin_enabled is False
    assert resolved.execution.default_decimation == 4
    assert resolved.planner.resources.max_workers == 3
    assert resolved.interactive.transport.websocket.enabled is True
    assert resolved.interactive.transport.websocket.port == 7232
    assert resolved.telemetry.primary_env_id == 2
    assert resolved.telemetry.selected_env_ids == (2, 3)
    assert resolved.telemetry.publish_decimation == 5
    assert resolved.telemetry.include_state_json is False
    assert resolved.sources["runtime.execution.default_decimation"] == "cli"
    assert resolved.sources["runtime.planner.backend"] == "runtime:default_tiled_scene"
    assert resolved.sources["runtime.telemetry.selected_env_ids"] == "cli"
    assert resolved.sources["runtime.telemetry.publish_decimation"] == "cli"


def test_tiled_scene_primary_env_cli_override_requires_matching_selection() -> None:
    args = tiled_scene_cli.parse_args(["--telemetry-primary-env-id", "2"])

    with pytest.raises(ValueError, match="included in selected_env_ids"):
        _resolve_bundled_tiled_scene(args)


def test_tiled_scene_selected_env_cli_override_requires_matching_primary() -> None:
    args = tiled_scene_cli.parse_args(["--telemetry-env-ids", "3,4"])

    with pytest.raises(ValueError, match="included in selected_env_ids"):
        _resolve_bundled_tiled_scene(args)


def test_tiled_scene_runtime_facade_forwards_max_batch_problems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linkerbot_sim.app.interactive.tiled_scene.runtime import core

    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_create(_runtime_type, **kwargs: object):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(core, "create_tiled_scene_runtime", fake_create)

    result = core.TiledSceneRuntime.create(
        env_name="unit",
        env_config={},
        simulation_app=SimulationAppSettings(),
        camera_output_settings=CameraOutputRuntimeSettings(),
        shutdown_settings=ShutdownSettings(),
        default_decimation=2,
        planner_backend="linear",
        curobo_profile="unit_curobo",
        joint_batch_mode="per_env",
        max_batch_problems=17,
        oversize_request_policy="reject",
        failure_policy="reject_request",
        cache_root="relative-cache",
        planner_request_defaults=PlannerRequestDefaults(duration_s=2.0),
        command_defaults=RuntimeCommandDefaults(joint_interpolation="linear"),
        playback_settings=PlaybackResourceSettings(
            max_queue_depth_per_env=3,
            max_samples_per_env=20,
            max_duration_s_per_env=2.0,
        ),
    )

    assert result is sentinel
    assert calls[0]["max_batch_problems"] == 17
    assert calls[0]["planner_backend"] == "linear"
    assert calls[0]["curobo_profile"] == "unit_curobo"
    assert calls[0]["joint_batch_mode"] == "per_env"
    assert calls[0]["oversize_request_policy"] == "reject"
    assert calls[0]["failure_policy"] == "reject_request"
    assert calls[0]["cache_root"] == "relative-cache"
    assert calls[0]["planner_request_defaults"].duration_s == 2.0
    assert calls[0]["command_defaults"].joint_interpolation == "linear"
    assert calls[0]["playback_settings"].max_duration_s_per_env == 2.0


def test_tiled_scene_factory_attaches_request_defaults_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linkerbot_sim.app.interactive.tiled_scene.runtime import factory

    class FakeWorld:
        def reset(self) -> None:
            pass

        def get_physics_context(self) -> SimpleNamespace:
            return SimpleNamespace(set_gravity=lambda _value: None)

    session = SimpleNamespace(world=FakeWorld(), stage=object(), app=None)
    scene = SimpleNamespace(
        config=SimpleNamespace(num_envs=2),
        articulation_views={},
        object_prim_paths={},
        object_handles=(),
        env_origins=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    prepared_camera_output = SimpleNamespace(path_plans=())
    snapshot = {"object": {"env_ids": [0, 1]}}
    snapshot_kwargs: dict[str, object] = {}
    runtime_kwargs: dict[str, object] = {}
    planner_request_defaults = PlannerRequestDefaults(duration_s=2.0)
    command_defaults = RuntimeCommandDefaults(joint_interpolation="linear")
    sentinel = object()

    monkeypatch.setattr(
        "linkerbot_sim.app.runtime.simulation_session.create_simulation_session",
        lambda **_kwargs: session,
    )
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.builder.build_isaac_tiled_scene",
        lambda **_kwargs: scene,
    )
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.cameras.tiled_sensor_camera_settings",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        "linkerbot_sim.sensors.camera.runtime.create_sensor_camera_runtimes",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        "linkerbot_sim.sensors.camera.runtime.initialize_sensor_camera_runtimes",
        lambda _cameras: None,
    )
    monkeypatch.setattr(
        "linkerbot_sim.sensors.camera.observer.prepare_camera_output",
        lambda *_args, **_kwargs: prepared_camera_output,
    )
    monkeypatch.setattr(
        "linkerbot_sim.sensors.camera.observer.open_prepared_camera_output",
        lambda _prepared: None,
    )
    monkeypatch.setattr(
        "linkerbot_sim.utils.output_paths.apply_output_path_plans",
        lambda _plans: None,
    )
    monkeypatch.setattr(
        "linkerbot_sim.tiled.scene.views.finalize_tiled_articulation_views",
        lambda value: value,
    )
    monkeypatch.setattr(
        "linkerbot_sim.configs.profiles.load_profile_yaml",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        factory, "_create_isaac_ik_solvers", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(factory, "_create_tiled_object_pose_views", lambda _scene: {})
    monkeypatch.setattr(
        factory,
        "_create_tiled_planner_backend",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        factory,
        "TiledPlannerManager",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        factory,
        "TiledTrajectoryBuffer",
        lambda **_kwargs: SimpleNamespace(),
    )

    def fake_capture_snapshot(
        *,
        stage: object,
        object_prim_paths: object,
        env_origins: object,
        env_ids: object,
        object_pose_views: object,
    ) -> dict[str, dict[str, object]]:
        snapshot_kwargs.update(
            stage=stage,
            object_prim_paths=object_prim_paths,
            env_origins=env_origins,
            env_ids=env_ids,
            object_pose_views=object_pose_views,
        )
        return snapshot

    def fake_runtime_type(**kwargs: object) -> object:
        runtime_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        factory, "capture_tiled_object_pose_snapshot", fake_capture_snapshot
    )

    result = factory.create_tiled_scene_runtime(
        fake_runtime_type,
        env_name="unit",
        env_config={
            "env": {"physics_frequency": 100.0},
            "tiled": {"enabled": True, "num_envs": 2, "spacing": 2.0},
        },
        simulation_app=SimulationAppSettings(),
        camera_output_settings=CameraOutputRuntimeSettings(),
        shutdown_settings=ShutdownSettings(),
        default_decimation=2,
        planner_request_defaults=planner_request_defaults,
        command_defaults=command_defaults,
    )

    assert result is sentinel
    assert set(snapshot_kwargs) == {
        "stage",
        "object_prim_paths",
        "env_origins",
        "env_ids",
        "object_pose_views",
    }
    assert runtime_kwargs["initial_object_states"] is snapshot
    assert runtime_kwargs["planner_request_defaults"] is planner_request_defaults
    assert runtime_kwargs["command_defaults"] is command_defaults


@pytest.mark.parametrize(
    ("package_name", "runtime_profile", "heavy_modules"),
    (
        (
            "single_scene",
            "default_single_scene",
            (
                "linkerbot_sim.app.interactive.single_scene.runtime",
                "linkerbot_sim.app.runtime.single_scene_runtime",
            ),
        ),
        (
            "tiled_scene",
            "default_tiled_scene",
            (
                "linkerbot_sim.app.interactive.tiled_scene.runtime",
                "linkerbot_sim.app.interactive.tiled_scene.runtime.core",
            ),
        ),
    ),
)
def test_dump_effective_config_does_not_import_or_create_runtime(
    package_name: str,
    runtime_profile: str,
    heavy_modules: tuple[str, ...],
) -> None:
    code = f"""
import sys
from linkerbot_sim.app.interactive import {package_name} as entry
entry.main(["--runtime-profile", "{runtime_profile}", "--dump-effective-config"])
unexpected = [name for name in {heavy_modules!r} if name in sys.modules]
if unexpected:
    raise SystemExit("runtime modules imported: " + ",".join(unexpected))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f'"runtime_profile": "{runtime_profile}"' in completed.stdout
    assert '"fingerprint":' in completed.stdout


@pytest.mark.parametrize(
    ("package_name", "runtime_profile", "expected_mode"),
    (
        ("single_scene", "default_tiled_scene", "single_scene"),
        ("tiled_scene", "default_single_scene", "tiled_scene"),
    ),
)
def test_entry_rejects_wrong_runtime_mode_before_runtime_import(
    package_name: str,
    runtime_profile: str,
    expected_mode: str,
) -> None:
    code = f"""
import sys
from linkerbot_sim.app.interactive import {package_name} as entry
try:
    entry.main(["--runtime-profile", "{runtime_profile}", "--dump-effective-config"])
except ValueError as exc:
    if "incompatible with '{expected_mode}' entrypoint" not in str(exc):
        raise
else:
    raise SystemExit("wrong-mode profile was accepted")
heavy = [name for name in sys.modules if name.startswith(
    "linkerbot_sim.app.interactive.{package_name}.runtime"
)]
if heavy:
    raise SystemExit("runtime modules imported: " + ",".join(heavy))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("entry", "prefix"),
    (
        (single_scene_cli, "SINGLE_SCENE_INTERACTIVE_CONFIG"),
        (tiled_scene_cli, "TILED_SCENE_INTERACTIVE_CONFIG"),
    ),
)
def test_startup_log_contains_only_profile_and_fingerprint(
    entry,
    prefix: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    resolved = SimpleNamespace(fingerprint="b" * 64)
    monkeypatch.setattr(
        entry,
        "resolve_entry_config",
        lambda _args: (resolved, {"private_path": "/sensitive/cache"}),
    )
    monkeypatch.setattr(entry, "run_interactive_mode", lambda *_args, **_kwargs: 7)

    entry.main(["--runtime-profile", "unit"])

    output = capsys.readouterr().out
    assert f"{prefix} runtime_profile=unit fingerprint={'b' * 64}" in output
    assert "/sensitive/cache" not in output


def test_tiled_scene_resolved_settings_are_forwarded_to_runtime_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = tiled_scene_cli.parse_args(
        [
            "--no-stdin",
            "--planner-workers",
            "3",
            "--websocket-port",
            "7232",
            "--telemetry-primary-env-id",
            "1",
            "--telemetry-env-ids",
            "1,2",
            "--telemetry-decimation",
            "3",
        ]
    )
    resolved, env_config = _resolve_bundled_tiled_scene(args)
    runtime = SimpleNamespace(
        camera_output=None,
        close=lambda: None,
        quit_event=SimpleNamespace(is_set=lambda: False, set=lambda: None),
    )
    factory_kwargs: dict[str, object] = {}
    loop_kwargs: dict[str, object] = {}
    telemetry_args: dict[str, object] = {}
    websocket_kwargs: dict[str, object] = {}

    class FakeWebSocketServer:
        bound_port = 7232

        @staticmethod
        def publish_event(_event) -> bool:
            return True

        @staticmethod
        def stop(*, timeout_s: float) -> dict[str, object]:
            assert timeout_s == pytest.approx(2.0)
            return {"thread_alive": False}

    def fake_create_tiled_scene_runtime(**kwargs: object):
        factory_kwargs.update(kwargs)
        return runtime

    monkeypatch.setattr(
        tiled_scene_cli, "create_tiled_scene_runtime", fake_create_tiled_scene_runtime
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.telemetry_publish._runtime_num_envs",
        lambda _runtime: 64,
    )

    def fake_create_telemetry(config, **kwargs: object):
        telemetry_args["config"] = config
        telemetry_args.update(kwargs)
        num_envs = kwargs["num_envs"]
        assert num_envs == 64
        return None

    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.telemetry_publish._create_telemetry",
        fake_create_telemetry,
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.transport.run_interactive_loop",
        lambda _runtime, **kwargs: loop_kwargs.update(kwargs),
    )

    def fake_start_websocket_server(*_args, **kwargs):
        websocket_kwargs.update(kwargs)
        return FakeWebSocketServer()

    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.transport.start_websocket_server",
        fake_start_websocket_server,
    )

    tiled_scene_cli.run_interactive_mode(
        args,
        resolved=resolved,
        env_config=env_config,
    )

    assert factory_kwargs["env_name"] == "scene3_tiled"
    assert factory_kwargs["env_config"] is env_config
    assert factory_kwargs["simulation_app"] is resolved.simulation_app
    assert factory_kwargs["camera_output_settings"] is resolved.camera_output
    assert factory_kwargs["shutdown_settings"] is resolved.shutdown
    assert factory_kwargs["controller_bundle"] == resolved.profiles.controller_bundle
    assert factory_kwargs["planner_workers"] == 3
    assert factory_kwargs["max_batch_problems"] == 64
    assert factory_kwargs["oversize_request_policy"] == "split"
    assert factory_kwargs["failure_policy"] == "hold_failed_env"
    assert factory_kwargs["playback_settings"] is resolved.playback
    assert factory_kwargs["planner_shutdown_timeout_s"] == 30.0
    assert factory_kwargs["planner_backend"] == resolved.planner.backend
    assert factory_kwargs["curobo_profile"] == resolved.profiles.curobo
    assert factory_kwargs["joint_batch_mode"] == resolved.planner.joint_batch_mode
    assert loop_kwargs["telemetry_rate_hz"] == 10.0
    assert loop_kwargs["idle_physics_policy"] == "pause"
    assert loop_kwargs["idle_step_duration_s"] == pytest.approx(0.05)
    assert websocket_kwargs["startup_timeout_s"] == pytest.approx(5.0)
    telemetry_config = telemetry_args["config"]
    assert telemetry_config.selected_env_ids == (1, 2)
    assert telemetry_config.primary_env_id == 1
    assert telemetry_config.publish_decimation == 3
    assert telemetry_config.buffer_size == 1
    assert telemetry_config.drop_policy == "latest"
    assert telemetry_config.on_error == "stop"
    assert telemetry_config.include_scene_markers is True
    assert telemetry_config.include_efforts is False
    assert telemetry_config.include_objects is False
    topics = telemetry_config.topics
    assert topics.joint_states == "/joint_states"
    assert topics.scene == "/scene"
    assert topics.state == "/linkerbot/state"
    assert telemetry_config.mcap_existing_file_policy == "error"
    assert telemetry_args["live_host"] == "127.0.0.1"
    assert telemetry_args["live_port"] is None
    assert telemetry_args["mcap_path"] is None
    assert telemetry_args["mcap_output_plan"] is None
    assert telemetry_args["output_paths_applied"] is False


def test_tiled_scene_zero_telemetry_rate_skips_mcap_preflight_and_runtime_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "existing.mcap"
    target.write_bytes(b"existing")
    args = tiled_scene_cli.parse_args(
        [
            "--no-stdin",
            "--telemetry-rate-hz",
            "0",
            "--foxglove-mcap-path",
            str(target),
        ]
    )
    resolved, env_config = _resolve_bundled_tiled_scene(args)
    factory_kwargs: dict[str, object] = {}
    runtime = SimpleNamespace(
        config=SimpleNamespace(num_envs=64),
        camera_output=None,
        close=lambda: None,
        quit_event=SimpleNamespace(is_set=lambda: True, set=lambda: None),
    )

    def fake_create_tiled_scene_runtime(**kwargs: object):
        factory_kwargs.update(kwargs)
        return runtime

    monkeypatch.setattr(
        tiled_scene_cli, "create_tiled_scene_runtime", fake_create_tiled_scene_runtime
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.transport.run_interactive_loop",
        lambda *_args, **_kwargs: None,
    )

    tiled_scene_cli.run_interactive_mode(
        args,
        resolved=resolved,
        env_config=env_config,
    )

    assert factory_kwargs["additional_output_path_plans"] == ()
    assert runtime.telemetry_status_provider is None
    assert target.read_bytes() == b"existing"


def test_tiled_scene_telemetry_startup_failure_closes_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = tiled_scene_cli.parse_args(["--no-stdin"])
    resolved, env_config = _resolve_bundled_tiled_scene(args)
    close_calls: list[str] = []
    runtime = SimpleNamespace(
        camera_output=None,
        close=lambda: close_calls.append("runtime"),
        quit_event=SimpleNamespace(is_set=lambda: False, set=lambda: None),
    )

    monkeypatch.setattr(
        tiled_scene_cli,
        "create_tiled_scene_runtime",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.telemetry_publish._runtime_num_envs",
        lambda _runtime: 64,
    )

    def fail_telemetry(_config, **kwargs: object) -> None:
        assert kwargs["num_envs"] == 64
        raise RuntimeError("telemetry startup failed")

    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.telemetry_publish._create_telemetry",
        fail_telemetry,
    )

    with pytest.raises(RuntimeError, match="telemetry startup failed"):
        tiled_scene_cli.run_interactive_mode(
            args,
            resolved=resolved,
            env_config=env_config,
        )

    assert close_calls == ["runtime"]


def test_tiled_scene_telemetry_close_failure_still_closes_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = tiled_scene_cli.parse_args(["--no-stdin"])
    resolved, env_config = _resolve_bundled_tiled_scene(args)
    close_calls: list[str] = []
    runtime = SimpleNamespace(
        camera_output=None,
        close=lambda: close_calls.append("runtime"),
        quit_event=SimpleNamespace(is_set=lambda: False, set=lambda: None),
    )

    class FailingTelemetry:
        def close(self) -> bool:
            close_calls.append("telemetry")
            raise RuntimeError("telemetry close failed")

        def status(self) -> dict[str, object]:
            return {"sink_closed": False}

    monkeypatch.setattr(
        tiled_scene_cli,
        "create_tiled_scene_runtime",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.telemetry_publish._runtime_num_envs",
        lambda _runtime: 64,
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.telemetry_publish._create_telemetry",
        lambda _config, **_kwargs: FailingTelemetry(),
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.transport.run_interactive_loop",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="telemetry close failed"):
        tiled_scene_cli.run_interactive_mode(
            args,
            resolved=resolved,
            env_config=env_config,
        )

    assert close_calls == ["telemetry", "runtime"]


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    source_root = str(REPO_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_root if not current else f"{source_root}:{current}"
    return env
