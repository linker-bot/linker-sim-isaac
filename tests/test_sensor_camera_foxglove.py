from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linkerbot_sim.sensors.camera_foxglove import FoxgloveCameraFrameSink
from linkerbot_sim.sensors.camera_frame import CameraFrame


def test_foxglove_camera_sink_logs_raw_image_and_info(monkeypatch) -> None:
    raw_messages = []
    json_messages = []
    servers = []

    class RawImage:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class Timestamp:
        def __init__(self, *, sec: int, nsec: int) -> None:
            self.sec = sec
            self.nsec = nsec

    class RawImageChannel:
        def __init__(self, topic: str) -> None:
            self.topic = topic

        def log(self, msg, *, log_time=None) -> None:
            raw_messages.append((self.topic, msg, log_time))

    class JsonChannel:
        def __init__(self, topic: str, **kwargs) -> None:
            self.topic = topic
            self.kwargs = kwargs

        def log(self, msg, *, log_time=None) -> None:
            json_messages.append((self.topic, msg, log_time))

    fake_foxglove = SimpleNamespace(
        Channel=JsonChannel,
        start_server=lambda **kwargs: servers.append(kwargs) or object(),
    )
    fake_messages = SimpleNamespace(RawImage=RawImage, Timestamp=Timestamp)
    fake_channels = SimpleNamespace(RawImageChannel=RawImageChannel)

    import linkerbot_sim.sensors.camera_foxglove as camera_foxglove

    monkeypatch.setattr(
        camera_foxglove,
        "_load_foxglove",
        lambda: (fake_foxglove, fake_messages),
    )
    monkeypatch.setattr(
        camera_foxglove,
        "_load_foxglove_channels",
        lambda: fake_channels,
    )

    sink = FoxgloveCameraFrameSink.open_live(
        host="127.0.0.1",
        port=8770,
        topic_prefix_by_camera={"wrist_rgbd": "/cameras/wrist_rgbd"},
    )
    frame = CameraFrame(
        camera_name="wrist_rgbd",
        modality="rgb",
        frame_index=0,
        simulation_step=2,
        time_s=0.3,
        data=np.zeros((2, 3, 3), dtype=np.uint8),
    )

    sink.publish(frame)

    assert servers == [
        {"name": "linkerbot-sim-camera", "host": "127.0.0.1", "port": 8770}
    ]
    assert raw_messages[0][0] == "/cameras/wrist_rgbd/rgb"
    raw_image = raw_messages[0][1]
    assert raw_image.kwargs["width"] == 3
    assert raw_image.kwargs["height"] == 2
    assert raw_image.kwargs["encoding"] == "rgb8"
    assert raw_image.kwargs["step"] == 9
    assert len(raw_image.kwargs["data"]) == 18
    assert raw_messages[0][2] == 300_000_000
    assert json_messages[0][0] == "/cameras/wrist_rgbd/info"
    assert json_messages[0][1]["camera_name"] == "wrist_rgbd"
