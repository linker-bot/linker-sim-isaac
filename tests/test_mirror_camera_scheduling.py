"""Observable sampling, freshness and GUI fairness at the owner boundary."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.mirror.rendering import CameraBundle, RenderCoordinator
from linkerbot_sim.sensors.camera.observer import CameraFrameObserver
from linkerbot_sim.sensors.camera.runtime import SensorCameraRuntime
from linkerbot_sim.sensors.camera.config import SensorCameraSettings


class _Camera(SensorCameraRuntime):
    def __init__(self, name, frequency):
        settings = SensorCameraSettings(
            name=name,
            prim_path=f"/World/{name}",
            frequency=frequency,
            resolution=(3, 2),
            modalities=("rgb",),
        )
        super().__init__(settings=settings, camera=object())
        self.active = False
        self.frame_id = 0
        self.readbacks = 0

    def set_render_active(self, active):
        self.active = active

    def finish_render_capture(self, **kwargs):
        self._capture_metadata = {"native_frame_id": self.frame_id, **kwargs}

    def capture_matches_time(self, time_s):
        return self._capture_metadata.get("physics_time_s") == time_s

    def get_modality_data(self, modality, *, device=None, clone=False):
        self.readbacks += 1
        return np.full((2, 3, 3), self.frame_id % 255, dtype=np.uint8)


class _Physics:
    def __init__(self, cameras):
        self.cameras = cameras
        self.simulation_time = 0.0
        self.renders = 0
        self.selected = []
        self.fail = False

    def render(self):
        if self.fail:
            raise RuntimeError("native render failed")
        self.renders += 1
        self.selected.append(tuple(c.name for c in self.cameras if c.active))
        for camera in self.cameras:
            if camera.active:
                camera.frame_id += 1


def _setup(frequencies=(60, 30, 10), *, gui_hz=None):
    cameras = tuple(_Camera(f"camera_{i}", hz) for i, hz in enumerate(frequencies))
    frames = []
    observer = CameraFrameObserver(
        cameras=cameras, publisher=SimpleNamespace(publish=frames.append)
    )
    physics = _Physics(cameras)
    coordinator = RenderCoordinator(
        physics_runtime=physics,
        cameras=CameraBundle(
            cameras=cameras, output=SimpleNamespace(observer=observer)
        ),
        gui_frequency_hz=gui_hz,
    )
    return coordinator, physics, cameras, observer, frames


def test_one_simulation_second_samples_mixed_rates_without_duplicate_readback():
    coordinator, physics, cameras, observer, frames = _setup()
    for step in range(120):
        physics.simulation_time = (step + 1) / 120
        coordinator.after_physics_step()
        observer.observe(physics, step=step)
    assert physics.renders == 60
    for camera, expected in zip(cameras, (60, 30, 10), strict=True):
        recorded = [frame for frame in frames if frame.camera_name == camera.name]
        assert len(recorded) == expected
        assert camera.readbacks == expected
        assert [f.frame_index for f in recorded] == list(range(expected))
        assert np.diff([f.time_s for f in recorded]) == pytest.approx(1 / expected)
        assert all(f.capture_metadata["physics_time_s"] == f.time_s for f in recorded)
    assert physics.selected[0] == tuple(c.name for c in cameras)
    assert physics.selected[1] == (cameras[0].name,)
    assert not any(camera.active for camera in cameras)


def test_non_divisor_frequency_is_quantized_without_long_term_drift_or_catchup():
    coordinator, physics, _, observer, frames = _setup((50,))
    for step in range(120):
        physics.simulation_time = (step + 1) / 120
        coordinator.after_physics_step()
        observer.observe(physics, step=step)
    assert len(frames) == 50
    assert set(round(x * 120) for x in np.diff([f.time_s for f in frames])) == {2, 3}
    physics.simulation_time = 5.0
    coordinator.after_physics_step()
    observer.observe(physics, step=600)
    coordinator.after_physics_step()
    observer.observe(physics, step=600)
    assert len(frames) == 51


def test_explicit_capture_does_not_consume_recording_schedule_and_owns_pixels():
    coordinator, physics, cameras, observer, frames = _setup((60,))
    initial = coordinator.render_frame()
    frozen = initial[cameras[0].name]["rgb"].copy()
    coordinator.render_frame()
    assert frames == []
    assert observer.due_camera_ids(0.0) == (cameras[0].name,)
    coordinator.after_physics_step()
    observer.observe(physics, step=0)
    assert len(frames) == 1 and physics.renders == 3
    np.testing.assert_array_equal(initial[cameras[0].name]["rgb"], frozen)
    observer.observe(physics, step=0)
    assert len(frames) == 1


def test_render_failure_can_retry_same_due_snapshot_and_reset_invalidates_capture():
    coordinator, physics, cameras, observer, frames = _setup((60,))
    physics.fail = True
    with pytest.raises(RuntimeError, match="native render failed"):
        coordinator.after_physics_step()
    assert observer.due_camera_ids(0.0) == (cameras[0].name,)
    assert frames == [] and not cameras[0].active
    physics.fail = False
    coordinator.after_physics_step()
    observer.observe(physics, step=0)
    assert len(frames) == 1
    observer.reset()
    coordinator.invalidate_captures()
    observer.observe(physics, step=0)
    assert len(frames) == 1  # Editing at the same time cannot relabel old pixels.
    coordinator.after_physics_step()
    observer.observe(physics, step=0)
    assert len(frames) == 2 and frames[-1].frame_index == 0


def test_no_consumer_headless_steps_and_gui_ticks_do_not_capture(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        "linkerbot_sim.mirror.rendering.time", SimpleNamespace(monotonic=lambda: now[0])
    )
    coordinator, physics, cameras, _, _ = _setup(gui_hz=60)
    coordinator.cameras.output = None
    coordinator.after_physics_step()
    assert physics.renders == 0
    assert coordinator.gui_wait_timeout(1) == pytest.approx(1 / 60)
    now[0] = 1 / 60
    coordinator.service_gui()
    assert physics.selected == [()]
    assert cameras[0].get_capture_metadata() == {}
    coordinator.render_frame((cameras[0].name,))
    count = physics.renders
    coordinator.service_gui()
    assert physics.renders == count
    assert coordinator.gui_wait_timeout(1) == pytest.approx(1 / 60)
    coordinator.gui_frequency_hz = None
    now[0] = 10
    coordinator.after_physics_step()
    assert physics.renders == count
    assert coordinator.gui_wait_timeout(1) == 1


@pytest.mark.parametrize("scenario", ("pause", "estop", "queries"))
def test_owner_loop_services_gui_while_physics_is_frozen(monkeypatch, scenario):
    from linkerbot_sim.mirror.app import run_mirror
    from linkerbot_sim.mirror.bootstrap import create_mirror_runtime
    from test_mirror_runtime import _assembly

    now = [0.0]
    monkeypatch.setattr(
        "linkerbot_sim.mirror.rendering.time", SimpleNamespace(monotonic=lambda: now[0])
    )
    config = load_mirror_config()
    config = replace(
        config,
        control=replace(
            config.control,
            idle_physics_policy="pause",
            sync_simulation_to_wall_clock=False,
        ),
        outputs=replace(
            config.outputs,
            render=replace(config.outputs.render, enabled=True, gui=True),
        ),
    )
    events = []
    runtime = create_mirror_runtime(
        config, assembly_factory=lambda cfg: _assembly(cfg, events=events)
    )
    monkeypatch.setattr(
        runtime.controller.admission,
        "status",
        lambda: SimpleNamespace(estopped=scenario == "estop"),
    )
    waits = []

    def process_next(*, timeout_s):
        waits.append(timeout_s)
        now[0] += 1 / config.scene.render_frequency_hz
        return object() if scenario == "queries" else None

    monkeypatch.setattr(runtime.controller, "process_next", process_next)
    run_mirror(runtime, max_iterations=4, close_on_exit=False)
    assert events == ["render"] * 4
    assert all(
        0 <= wait <= 1 / config.scene.render_frequency_hz + 1e-9 for wait in waits
    )
    runtime.close()
