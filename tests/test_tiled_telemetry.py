from __future__ import annotations

import importlib

import numpy as np
import pytest

from linkerbot_sim.telemetry.tiled import (
    TiledInteractiveTelemetrySink,
    TiledTelemetryConfig,
)


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


def test_tiled_telemetry_config_parses_env_ids() -> None:
    config = TiledTelemetryConfig.from_env_ids(
        "1,3",
        publish_decimation=4,
        topic_prefix="/debug/tiled/",
    )

    assert config.selected_env_ids == (1, 3)
    assert config.publish_decimation == 4
    assert config.topic_prefix == "/debug/tiled"


def test_tiled_telemetry_publishes_selected_robot_joint_state() -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(selected_env_ids=(1,), publish_decimation=2),
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


def test_tiled_telemetry_publishes_selected_object_and_tcp_markers() -> None:
    logger = _FakeFoxgloveLogger()
    sink = TiledInteractiveTelemetrySink(
        [logger],
        config=TiledTelemetryConfig(selected_env_ids=(1,), publish_decimation=1),
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
        config=TiledTelemetryConfig(selected_env_ids=(0,), publish_decimation=1),
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
        config=TiledTelemetryConfig(selected_env_ids=(1,), publish_decimation=1),
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
        config=TiledTelemetryConfig(selected_env_ids=(0,), publish_decimation=1),
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
        config=TiledTelemetryConfig(selected_env_ids=(0,), publish_decimation=3),
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
        config=TiledTelemetryConfig(selected_env_ids=(0,)),
    )

    sink.close()

    assert logger.closed is True


def test_old_tiled_telemetry_import_paths_are_removed() -> None:
    import linkerbot_sim.tiled as tiled

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("linkerbot_sim.tiled.telemetry")
    with pytest.raises(AttributeError):
        getattr(tiled, "TiledInteractiveTelemetrySink")
    with pytest.raises(AttributeError):
        getattr(tiled, "TiledTelemetryConfig")
