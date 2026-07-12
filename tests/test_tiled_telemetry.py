from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.app.interactive.tiled_scene.telemetry_publish import (
    _create_telemetry,
    _publish_state_telemetry,
)
from linkerbot_sim.app.interactive.tiled_scene.transport import _quit_on_stdin_eof
from linkerbot_sim.telemetry.foxglove import FoxgloveTopicConfig
from linkerbot_sim.telemetry.tiled.config import TiledTelemetryConfig, parse_env_ids
from linkerbot_sim.telemetry.tiled.sink import TiledInteractiveTelemetrySink


class _FakeFoxgloveLogger:
    def __init__(self) -> None:
        self.state_json = []
        self.joint_states = []
        self.scene_spheres = []
        self.closed = False

    def log_state_json(self, state, *, time_s=None) -> None:
        self.state_json.append((state, time_s))

    def log_joint_state(self, **kwargs) -> None:
        self.joint_states.append(kwargs)

    def log_scene_spheres(
        self,
        *,
        entity_id,
        positions,
        frame_id="world",
        radius=0.02,
        color=(0.1, 0.45, 1.0, 1.0),
        time_s=None,
    ) -> None:
        self.scene_spheres.append(
            {
                "entity_id": entity_id,
                "positions": positions,
                "frame_id": frame_id,
                "radius": radius,
                "color": color,
                "time_s": time_s,
            }
        )

    def close(self) -> None:
        self.closed = True


def test_tiled_telemetry_config_uses_parsed_env_ids() -> None:
    topics = FoxgloveTopicConfig(
        joint_states="/debug/joint_states",
        scene="/debug/scene",
        state="/debug/state",
    )
    config = TiledTelemetryConfig(
        selected_env_ids=parse_env_ids("1,3"),
        primary_env_id=3,
        publish_decimation=4,
        topics=topics,
    )

    assert config.selected_env_ids == (1, 3)
    assert config.primary_env_id == 3
    assert config.publish_decimation == 4
    assert config.topics is topics


def test_tiled_telemetry_publishes_selected_robot_joint_state() -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(1,), primary_env_id=1, publish_decimation=2
        ),
    )

    published = sink.publish_interactive_state(
        {
            "event": "state",
            "step": 4,
            "time_s": 0.2,
            "env_ids": [0, 1],
            "state": {
                "robots": {
                    "left": {
                        "joint_names": ["j0", "j1"],
                        "joint_positions": [[0.1, 0.2], [0.3, 0.4]],
                        "joint_velocities": [[1.0, 2.0], [3.0, 4.0]],
                    }
                },
                "episode_steps": [4, 4],
            },
        },
        event="step",
        trigger_response={"event": "step", "kind": "joint_delta_pos", "step": 4},
    )

    assert published is True
    payload, time_s = logger.state_json[0]
    assert payload["event"] == "step"
    assert payload["trigger"]["kind"] == "joint_delta_pos"
    assert time_s == 0.2
    joint_state = logger.joint_states[0]
    assert joint_state["joint_names"] == ["left/j0", "left/j1"]
    np.testing.assert_allclose(joint_state["positions"], [0.3, 0.4])
    np.testing.assert_allclose(joint_state["velocities"], [3.0, 4.0])


def test_tiled_telemetry_requires_primary_inside_explicit_selection() -> None:
    with pytest.raises(ValueError, match="included in selected_env_ids"):
        TiledTelemetryConfig(selected_env_ids=(3, 1), primary_env_id=0)


def test_tiled_telemetry_honors_explicit_primary_env() -> None:
    logger = _FakeFoxgloveLogger()
    config = TiledTelemetryConfig(
        selected_env_ids=(1, 3),
        primary_env_id=3,
    )
    sink = TiledInteractiveTelemetrySink([logger], config=config)

    sink.publish_interactive_state(
        {
            "event": "state",
            "step": 1,
            "time_s": 0.05,
            "env_ids": [1, 3],
            "state": {
                "robots": {
                    "left": {
                        "joint_names": ["j0"],
                        "joint_positions": [[1.0], [3.0]],
                    }
                }
            },
        },
        event="state",
    )

    assert config.primary_env_id == 3
    np.testing.assert_allclose(logger.joint_states[0]["positions"], [3.0])


def test_tiled_telemetry_rejects_primary_env_outside_selection() -> None:
    with pytest.raises(ValueError, match="included in selected_env_ids"):
        TiledTelemetryConfig(selected_env_ids=(1, 3), primary_env_id=2)


def test_tiled_telemetry_publishes_selected_object_and_tcp_markers() -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(1,), primary_env_id=1, publish_decimation=1
        ),
    )

    published = sink.publish_interactive_state(
        {
            "event": "state",
            "step": 1,
            "time_s": 0.05,
            "env_ids": [0, 1],
            "state": {
                "robots": {
                    "left": {
                        "joint_names": ["j0"],
                        "joint_positions": [[0.0], [0.1]],
                        "tcp_positions_world": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
                    }
                },
                "objects": {
                    "Tblock": {
                        "positions_world": [[0.2, 0.0, -0.4], [0.3, 0.1, -0.4]],
                    }
                },
            },
        },
        event="state",
    )

    assert published is True
    assert [item["entity_id"] for item in logger.scene_spheres] == [
        "env_001/object/Tblock",
        "env_001/tcp/left",
    ]
    np.testing.assert_allclose(logger.scene_spheres[0]["positions"], [[0.3, 0.1, -0.4]])
    np.testing.assert_allclose(logger.scene_spheres[1]["positions"], [[1.0, 2.0, 3.0]])


def test_tiled_telemetry_skips_standard_topics_when_selected_env_missing() -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(0,), primary_env_id=0, publish_decimation=1
        ),
    )

    published = sink.publish_interactive_state(
        {
            "event": "state",
            "step": 1,
            "time_s": 0.05,
            "env_ids": [1],
            "state": {
                "robots": {
                    "left": {
                        "joint_names": ["j0"],
                        "joint_positions": [[9.0]],
                        "tcp_positions_world": [[9.0, 0.0, 0.0]],
                    }
                }
            },
        },
        event="state",
    )

    assert published is True
    assert logger.state_json
    assert logger.joint_states == []
    assert logger.scene_spheres == []


def test_tiled_telemetry_object_markers_use_object_env_ids() -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(1,), primary_env_id=1, publish_decimation=1
        ),
    )

    sink.publish_interactive_state(
        {
            "event": "state",
            "step": 1,
            "time_s": 0.05,
            "env_ids": [0, 1],
            "state": {
                "objects": {
                    "Tblock": {
                        "env_ids": [1],
                        "positions_world": [[0.3, 0.1, -0.4]],
                    }
                }
            },
        },
        event="state",
    )

    assert [item["entity_id"] for item in logger.scene_spheres] == [
        "env_001/object/Tblock"
    ]
    np.testing.assert_allclose(logger.scene_spheres[0]["positions"], [[0.3, 0.1, -0.4]])


def test_tiled_telemetry_skips_object_marker_when_local_env_missing() -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(0,), primary_env_id=0, publish_decimation=1
        ),
    )

    sink.publish_interactive_state(
        {
            "event": "state",
            "step": 1,
            "time_s": 0.05,
            "env_ids": [0, 1],
            "state": {
                "objects": {
                    "Tblock": {
                        "env_ids": [1],
                        "positions_world": [[0.3, 0.1, -0.4]],
                    }
                }
            },
        },
        event="state",
    )

    assert logger.scene_spheres == []


def test_tiled_telemetry_decimation_skips_non_reset_events() -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(0,), primary_env_id=0, publish_decimation=3
        ),
    )

    published = sink.publish_interactive_state(
        {
            "step": 4,
            "time_s": 0.2,
            "env_ids": [0],
            "state": {"joint_positions": [[0.1, 0.2]]},
        },
        event="step",
    )

    assert published is False
    assert logger.state_json == []
    assert logger.joint_states == []

    reset_published = sink.publish_interactive_state(
        {
            "step": 4,
            "time_s": 0.2,
            "env_ids": [0],
            "state": {"joint_positions": [[0.1, 0.2]]},
        },
        event="reset",
    )

    assert reset_published is True
    assert logger.joint_states[0]["joint_names"] == ["command_0", "command_1"]


def test_tiled_telemetry_close_closes_all_loggers() -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(selected_env_ids=(0,), primary_env_id=0),
    )

    sink.close()

    assert logger.closed is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("buffer_size", True, "buffer_size"),
        ("buffer_size", 1.5, "buffer_size"),
        ("buffer_size", "2", "buffer_size"),
        ("shutdown_timeout_s", True, "shutdown_timeout_s"),
        ("shutdown_timeout_s", float("nan"), "shutdown_timeout_s"),
        ("shutdown_timeout_s", float("inf"), "shutdown_timeout_s"),
        ("mcap_existing_file_policy", "overwrite", "mcap_existing_file_policy"),
        ("include_objects", "false", "include_objects"),
    ),
)
def test_tiled_telemetry_config_rejects_coerced_or_invalid_values(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        TiledTelemetryConfig(
            selected_env_ids=(0,),
            primary_env_id=0,
            **kwargs,
        )


def test_tiled_telemetry_honors_modalities_efforts_and_exact_topics() -> None:
    logger = _FakeFoxgloveLogger()
    topics = FoxgloveTopicConfig(
        joint_states="/runtime/joints",
        scene="/runtime/markers",
        state="/runtime/batch",
    )
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(0, 1),
            primary_env_id=1,
            topics=topics,
            include_efforts=True,
            include_objects=False,
        ),
    )

    sink.publish_interactive_state(
        {
            "step": 2,
            "time_s": 0.1,
            "env_ids": [0, 1],
            "state": {
                "robots": {
                    "left": {
                        "joint_names": ["j0"],
                        "joint_positions": [[0.1], [0.2]],
                        "joint_velocities": [[1.0], [2.0]],
                        "measured_efforts": [[3.0], [4.0]],
                        "tcp_positions_world": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
                    }
                },
                "objects": {"box": {"positions_world": [[0, 0, 0], [1, 1, 1]]}},
            },
        },
        event="state",
    )

    assert "objects" not in logger.state_json[0][0]["state"]
    np.testing.assert_allclose(logger.joint_states[0]["positions"], [0.2])
    np.testing.assert_allclose(logger.joint_states[0]["efforts"], [4.0])
    assert [marker["entity_id"] for marker in logger.scene_spheres] == [
        "env_001/tcp/left"
    ]
    assert sink.config.topics is topics


def test_tiled_mcap_preflight_fails_before_live_sink_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.mcap"
    target.write_bytes(b"existing")
    opened: list[str] = []
    monkeypatch.setattr(
        "linkerbot_sim.telemetry.tiled.sink.FoxgloveLogger.open_live_server",
        lambda **_kwargs: opened.append("live"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        TiledInteractiveTelemetrySink.open(
            config=TiledTelemetryConfig(
                selected_env_ids=(0,),
                primary_env_id=0,
                mcap_existing_file_policy="error",
            ),
            live_port=8767,
            mcap_path=target,
        )

    assert opened == []
    assert target.read_bytes() == b"existing"


def test_tiled_async_telemetry_drop_newest_reports_status() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingLogger(_FakeFoxgloveLogger):
        def log_state_json(self, state, *, time_s=None) -> None:
            entered.set()
            release.wait(timeout=2.0)
            super().log_state_json(state, time_s=time_s)

    logger = BlockingLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(0,),
            primary_env_id=0,
            buffer_size=1,
            drop_policy="drop_newest",
            include_standard_joint_states=False,
            include_scene_markers=False,
        ),
        asynchronous=True,
    )
    state = {"time_s": 0.0, "env_ids": [0], "state": {}}

    assert sink.publish_interactive_state({**state, "step": 1}, event="state")
    assert entered.wait(timeout=1.0)
    assert sink.publish_interactive_state({**state, "step": 2}, event="state")
    assert not sink.publish_interactive_state({**state, "step": 3}, event="state")
    assert sink.status()["dropped_snapshots"] == 1
    release.set()
    assert sink.close() is True
    assert sink.status()["last_published_sequence"] == 2


@pytest.mark.parametrize(
    ("on_error", "stopped", "can_publish_after_error"),
    (("stop", True, False), ("continue", False, True)),
)
def test_tiled_state_sampling_failure_honors_error_policy(
    on_error: str,
    stopped: bool,
    can_publish_after_error: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(0,),
            primary_env_id=0,
            on_error=on_error,
        ),
    )

    def fail_get_state(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("state sampling failed")

    _publish_state_telemetry(
        sink,
        SimpleNamespace(get_state=fail_get_state),
        event="state",
    )

    status = sink.status()
    assert status["error_count"] == 1
    assert status["stopped_on_error"] is stopped
    assert status["last_error"] == "RuntimeError: state sampling failed"
    assert (
        "TILED_SCENE_INTERACTIVE_TELEMETRY_FAILED RuntimeError"
        in capsys.readouterr().err
    )
    assert (
        sink.publish_interactive_state(
            {"step": 1, "time_s": 0.1, "env_ids": [0], "state": {}},
            event="state",
        )
        is can_publish_after_error
    )
    assert sink.close() is True


def test_tiled_telemetry_close_records_error_and_retries_only_failed_logger() -> None:
    close_calls: list[str] = []

    class RetriableLogger(_FakeFoxgloveLogger):
        def __init__(self, name: str, *, fail_once: bool) -> None:
            super().__init__()
            self.name = name
            self.fail_once = fail_once
            self.attempts = 0

        def close(self) -> None:
            self.attempts += 1
            close_calls.append(self.name)
            if self.fail_once and self.attempts == 1:
                raise RuntimeError(f"{self.name} close failed")
            self.closed = True

    failing = RetriableLogger("failing", fail_once=True)
    stable = RetriableLogger("stable", fail_once=False)
    sink = TiledInteractiveTelemetrySink(
        [failing, stable],
        config=TiledTelemetryConfig(selected_env_ids=(0,), primary_env_id=0),
    )

    assert sink.close() is False
    assert close_calls == ["failing", "stable"]
    assert stable.closed is True
    status = sink.status()
    assert status["error_count"] == 1
    assert status["last_error"] == "RuntimeError: failing close failed"
    assert status["sink_closed"] is False

    assert sink.close() is True
    assert close_calls == ["failing", "stable", "failing"]
    assert sink.status()["sink_closed"] is True


def test_tiled_telemetry_start_failure_closes_every_logger_and_keeps_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls: list[str] = []
    observed_thread_refs: list[object | None] = []

    class Logger(_FakeFoxgloveLogger):
        def __init__(self, name: str, *, fail_close: bool) -> None:
            super().__init__()
            self.name = name
            self.fail_close = fail_close

        def close(self) -> None:
            close_calls.append(self.name)
            if self.fail_close:
                raise RuntimeError("logger cleanup failed")
            super().close()

    class FailingThread:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("tiled thread start failed")

    original_close = TiledInteractiveTelemetrySink.close

    def observing_close(self: TiledInteractiveTelemetrySink) -> bool:
        observed_thread_refs.append(self._thread)
        return original_close(self)

    monkeypatch.setattr(
        "linkerbot_sim.telemetry.tiled.sink.Thread",
        FailingThread,
    )
    monkeypatch.setattr(TiledInteractiveTelemetrySink, "close", observing_close)

    with pytest.raises(RuntimeError, match="tiled thread start failed"):
        TiledInteractiveTelemetrySink(
            [
                Logger("failing", fail_close=True),
                Logger("stable", fail_close=False),
            ],
            config=TiledTelemetryConfig(selected_env_ids=(0,), primary_env_id=0),
            asynchronous=True,
        )

    assert observed_thread_refs == [None]
    assert close_calls == ["failing", "stable"]


def test_disabled_tiled_telemetry_skips_sink_creation_and_allows_stdin_eof_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[object] = []
    monkeypatch.setattr(
        TiledInteractiveTelemetrySink,
        "open",
        lambda **kwargs: opened.append(kwargs),
    )

    telemetry = _create_telemetry(
        None,
        num_envs=1,
        live_host="127.0.0.1",
        live_port=8767,
        mcap_path="state.mcap",
        mcap_output_plan=None,
        output_paths_applied=False,
    )

    assert telemetry is None
    assert opened == []
    assert _quit_on_stdin_eof(
        tcp_jsonl_port=None,
        telemetry=telemetry,
    )


def test_tiled_telemetry_timeout_status_clears_after_successful_retry() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingLogger(_FakeFoxgloveLogger):
        def log_state_json(self, state, *, time_s=None) -> None:
            entered.set()
            release.wait(timeout=2.0)
            super().log_state_json(state, time_s=time_s)

    logger = BlockingLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(
            selected_env_ids=(0,),
            primary_env_id=0,
            shutdown_timeout_s=0.02,
        ),
        asynchronous=True,
    )
    assert sink.publish_interactive_state(
        {"step": 1, "time_s": 0.1, "env_ids": [0], "state": {}},
        event="state",
    )
    assert entered.wait(timeout=1.0)

    assert sink.close() is False
    assert sink.status()["shutdown_timed_out"] is True
    assert logger.closed is False

    release.set()
    assert sink.close() is True
    status = sink.status()
    assert status["shutdown_timed_out"] is False
    assert status["thread_alive"] is False
    assert status["sink_closed"] is True
