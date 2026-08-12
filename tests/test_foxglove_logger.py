from __future__ import annotations

import sys
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.sensors.camera.foxglove import FoxgloveCameraFrameSink
from linkerbot_sim.telemetry import foxglove


def test_foxglove_time_helpers() -> None:
    assert foxglove._ns_time(None) == 0
    assert foxglove._ns_time(1.25) == 1_250_000_000


def test_foxglove_optional_dependency_error() -> None:
    original_module = sys.modules.get("foxglove")
    sys.modules["foxglove"] = None
    try:
        foxglove._load_foxglove()
    except ImportError as exc:
        assert "foxglove-sdk" in str(exc)
    else:
        raise AssertionError("expected ImportError when foxglove-sdk is unavailable")
    finally:
        if original_module is None:
            sys.modules.pop("foxglove", None)
        else:
            sys.modules["foxglove"] = original_module


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.10"])
def test_foxglove_live_factories_reject_non_loopback_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        foxglove.FoxgloveLogger.open_live_server(host=host)
    with pytest.raises(ValueError, match="loopback"):
        FoxgloveCameraFrameSink.open_live(
            host=host,
            port=8765,
            topic_prefix_by_camera={},
        )


def test_foxglove_vector_shape_validation() -> None:
    class Messages:
        class Vector3:
            def __init__(self, *, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

    vec = foxglove._vector3(np.asarray([1.0, 2.0, 3.0]), Messages)
    assert (vec.x, vec.y, vec.z) == (1.0, 2.0, 3.0)


def test_foxglove_logger_uses_typed_channels_and_json_state(monkeypatch) -> None:
    raw_channels = []
    typed_channels = []

    class RawChannel:
        def __init__(self, topic, **kwargs) -> None:
            self.topic = topic
            self.kwargs = kwargs
            self.logged = []
            raw_channels.append(self)

        def log(self, message, *, log_time=None) -> None:
            self.logged.append((message, log_time))

    class TypedChannel:
        def __init__(self, kind: str, topic: str) -> None:
            self.kind = kind
            self.topic = topic
            typed_channels.append(self)

    fake_foxglove = SimpleNamespace(Channel=RawChannel)
    fake_channels = SimpleNamespace(
        JointStatesChannel=lambda topic: TypedChannel("joint_states", topic),
        SceneUpdateChannel=lambda topic: TypedChannel("scene_update", topic),
    )

    monkeypatch.setattr(
        foxglove,
        "_load_foxglove",
        lambda: (fake_foxglove, SimpleNamespace()),
    )
    monkeypatch.setattr(
        foxglove,
        "_load_foxglove_channels",
        lambda: fake_channels,
    )

    logger = foxglove.FoxgloveLogger(
        object(),
        topics=foxglove.FoxgloveTopicConfig(
            joint_states="/test/joints",
            scene="/test/scene",
            state="/test/state",
            hybrid_control="/test/hybrid",
        ),
    )

    assert logger.joint_channel is typed_channels[0]
    assert logger.scene_channel is typed_channels[1]
    assert [(channel.kind, channel.topic) for channel in typed_channels] == [
        ("joint_states", "/test/joints"),
        ("scene_update", "/test/scene"),
    ]
    assert len(raw_channels) == 1
    assert raw_channels[0].topic == "/test/state"
    assert raw_channels[0].kwargs == {"message_encoding": "json"}

    logger.log_hybrid_control_json({"active": False}, time_s=1.25)

    assert len(raw_channels) == 2
    assert raw_channels[1].topic == "/test/hybrid"
    assert raw_channels[1].kwargs == {"message_encoding": "json"}
    assert raw_channels[1].logged == [({"active": False}, 1_250_000_000)]


def test_foxglove_logger_logs_with_installed_sdk() -> None:
    try:
        foxglove._load_foxglove()
        foxglove._load_foxglove_channels()
    except ImportError:
        pytest.skip("foxglove-sdk is not installed")

    suffix = uuid.uuid4().hex
    logger = foxglove.FoxgloveLogger(
        object(),
        topics=foxglove.FoxgloveTopicConfig(
            joint_states=f"/test/{suffix}/joint_states",
            scene=f"/test/{suffix}/scene",
            state=f"/test/{suffix}/state",
        ),
    )

    logger.log_joint_state(
        joint_names=["joint_a"],
        positions=[0.1],
        velocities=[0.2],
        efforts=[0.3],
        time_s=1.0,
    )
    logger.log_state_json({"ok": True}, time_s=1.0)
    logger.log_scene_spheres(
        entity_id="object_a",
        positions=np.asarray([[0.0, 0.1, 0.2]]),
        time_s=1.0,
    )
