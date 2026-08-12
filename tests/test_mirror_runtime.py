from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.mirror.app import run_mirror
from linkerbot_sim.mirror.bootstrap import MirrorAssembly, create_mirror_runtime
from linkerbot_sim.mirror.interface.protocol import MIRROR_PROTOCOL, MirrorRequest
from linkerbot_sim.mirror.lifecycle import close_result_stopped
from linkerbot_sim.mirror.rendering import CameraBundle, RenderCoordinator
from linkerbot_sim.mirror.scene_assembly import (
    MirrorSceneResources,
    _camera_output_settings,
    _state_stream_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("result", "expected"),
    (
        (None, True),
        (True, True),
        (False, False),
        (SimpleNamespace(stopped=True), True),
        (SimpleNamespace(stopped=False), False),
        ({"stopped": False}, False),
        ({"shutdown_timed_out": True}, False),
        ({"shutdown_timed_out": False}, True),
        ({}, True),
        (object(), True),
    ),
)
def test_close_result_stopped_normalizes_supported_close_contracts(
    result: object,
    expected: bool,
) -> None:
    assert close_result_stopped(result) is expected


class _Resource:
    def __init__(
        self, name: str, events: list[str], results: list[bool] | None = None
    ) -> None:
        self.resource_name = name
        self.events = events
        self.results = results or [True]
        self.calls = 0

    def close(self) -> bool:
        self.events.append(self.resource_name)
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


class _BindableMotion(_Resource):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__(name, events)
        self.render_frame = None
        self.step_synchronizer = None

    def bind_render_frame(self, callback) -> None:
        assert self.render_frame is None
        self.render_frame = callback

    def bind_step_synchronizer(self, synchronizer) -> None:
        assert self.step_synchronizer is None
        self.step_synchronizer = synchronizer


class _Physics:
    backend = "physx"
    kind = "physx_cpu"
    execution = "cpu"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def step(self, *, render: bool = False) -> None:
        self.events.append(f"physics_step:{render}")

    def pre_render(self) -> None:
        self.events.append("pre_render")

    def render(self) -> None:
        self.events.append("render")


class _DirectPhysics(_Physics):
    backend = "newton"
    kind = "newton_cuda"
    execution = "cuda"

    def __init__(self, events: list[str], *, fail_assert: bool = False) -> None:
        super().__init__(events)
        self.fail_assert = fail_assert

    def assert_single_world(self, *, consumer: str) -> None:
        self.events.append(f"assert_single_world:{consumer}")
        if self.fail_assert:
            raise RuntimeError("actual=2")


class _Session:
    def __init__(self, physics: object, events: list[str]) -> None:
        self.physics_runtime = physics
        self.app = object()
        self.stage = object()
        self.events = events
        self.close_calls = 0

    def close(self, *, exit_code: int = 0) -> None:
        self.close_calls += 1
        self.events.append(f"session:{exit_code}")


def _assembly(
    config: object,
    *,
    events: list[str],
    physics: object | None = None,
) -> MirrorAssembly:
    del config
    physics = physics or _Physics(events)
    session = _Session(physics, events)
    planner = _Resource("planner", events)
    output = _Resource("output", events)
    controller = _Resource("controller", events)
    view = _Resource("view", events)
    collision = _Resource("collision", events)
    assembly = MirrorAssembly(
        session=session,
        state_getter=lambda: {"state": 1},
        state_setter=lambda value, strict=True: {"strict": strict, **value},
        snapshot_capture=lambda: {"schema": "mirror-test", "state": 1},
        snapshot_restore=lambda value, **_kwargs: {"restored": value["state"]},
        resetter=lambda hold_after_reset=True: {"hold": hold_after_reset},
        motion_backend=planner,
        collision_registry=collision,
        outputs=(output,),
        controllers=(controller,),
        views=(view,),
    )
    return assembly


def test_runtime_owns_session_and_closes_in_strict_phase_order() -> None:
    events: list[str] = []
    config = load_mirror_config()
    runtime = create_mirror_runtime(
        config,
        assembly_factory=lambda cfg: _assembly(cfg, events=events),
    )
    ingress = _Resource("ingress", events)
    runtime.attach_ingress(ingress)

    assert runtime.get_state() == {"state": 1}
    assert runtime.set_state({"state": 2}, strict=True) == {
        "strict": True,
        "state": 2,
    }
    assert runtime.capture_snapshot() == {"schema": "mirror-test", "state": 1}
    assert runtime.restore_snapshot({"schema": "mirror-test", "state": 1}) == {
        "restored": 1
    }
    assert runtime.reset(hold_after_reset=False) == {"hold": False}
    runtime.step(render=True)
    report = runtime.close()
    second = runtime.close()

    assert not hasattr(runtime, "world")
    assert events[:2] == ["physics_step:False", "render"]
    assert "pre_render" not in events
    names = (
        "ingress",
        "output",
        "planner",
        "collision",
        "controller",
        "view",
        "session:0",
    )
    order = {name: events.index(name) for name in names}
    assert order["ingress"] < order["output"] < order["controller"] < order["session:0"]
    assert order["planner"] < order["controller"]
    assert order["collision"] < order["controller"]
    assert order["view"] < order["session:0"]
    assert report.stopped is True
    assert second.stopped is True
    assert runtime.session.close_calls == 1


def test_runtime_binds_its_single_render_coordinator_to_motion_backend() -> None:
    events: list[str] = []
    config = load_mirror_config()
    motion = _BindableMotion("motion", events)

    def assemble(cfg):
        result = _assembly(cfg, events=events)
        result.motion_backend = motion
        return result

    runtime = create_mirror_runtime(config, assembly_factory=assemble)

    assert callable(motion.render_frame)
    assert motion.step_synchronizer is runtime.step_synchronizer
    assert runtime.step_synchronizer.enabled is True
    motion.render_frame()
    assert events == ["render"]
    runtime.close()


def test_render_coordinator_honors_direct_camera_update_budget() -> None:
    events: list[str] = []
    physics = _DirectPhysics(events)
    camera = SimpleNamespace(
        name="direct",
        camera=SimpleNamespace(render_update_count=4),
        get_current_frame=lambda clone: {"clone": clone},
    )
    coordinator = RenderCoordinator(
        physics_runtime=physics,
        cameras=CameraBundle(cameras=(camera,)),
    )

    frames = coordinator.render_frame()

    assert events == ["render"] * 4
    assert frames == {"direct": {"clone": True}}


def test_render_coordinator_publishes_one_newton_snapshot_for_many_render_updates() -> (
    None
):
    events: list[str] = []

    class PipelinedDirectPhysics(_DirectPhysics):
        def render_update(self) -> None:
            events.append("render_update")

    coordinator = RenderCoordinator(
        physics_runtime=PipelinedDirectPhysics(events),
        cameras=CameraBundle(
            cameras=(
                SimpleNamespace(
                    camera=SimpleNamespace(render_update_count=4),
                ),
            ),
        ),
    )

    coordinator.render_frame()

    assert events == ["pre_render", *("render_update" for _ in range(4))]


def test_render_coordinator_updates_multiple_physx_cameras_together() -> None:
    events: list[str] = []
    coordinator = RenderCoordinator(
        physics_runtime=_Physics(events),
        cameras=CameraBundle(
            cameras=(
                SimpleNamespace(name="first"),
                SimpleNamespace(name="second"),
            ),
            capture_hook=lambda cameras: events.append(f"capture:{len(cameras)}"),
        ),
    )

    coordinator.render_frame()

    assert events == ["render", "capture:2"]


def test_render_only_leaves_camera_sampling_to_the_post_step_observer() -> None:
    events: list[str] = []
    coordinator = RenderCoordinator(
        physics_runtime=_Physics(events),
        cameras=CameraBundle(
            cameras=(SimpleNamespace(name="camera"),),
            capture_hook=lambda _cameras: events.append("capture"),
        ),
    )

    coordinator.render_only()

    assert events == ["render"]


def test_render_coordinator_rotates_multiple_direct_cameras_before_capture() -> None:
    events: list[str] = []
    physics = _DirectPhysics(events)

    class DirectCamera:
        def __init__(self, name: str, count: int) -> None:
            self.name = name
            self.render_update_count = count

        def set_render_active(self, active: bool) -> None:
            events.append(f"active:{self.name}:{active}")

    first = DirectCamera("first", 2)
    second = DirectCamera("second", 4)
    bundle = CameraBundle(
        cameras=(
            SimpleNamespace(name="first", camera=first),
            SimpleNamespace(name="second", camera=second),
        ),
        capture_hook=lambda _cameras: events.append("capture") or {"ready": True},
    )
    coordinator = RenderCoordinator(physics_runtime=physics, cameras=bundle)

    assert coordinator.render_frame() == {"ready": True}
    assert events == [
        "active:first:True",
        "active:second:False",
        "render",
        "render",
        "active:first:False",
        "active:second:True",
        "render",
        "render",
        "render",
        "render",
        "active:first:True",
        "active:second:True",
        "capture",
    ]


def test_render_coordinator_restores_all_direct_cameras_after_render_failure() -> None:
    events: list[str] = []

    class FailingPhysics(_DirectPhysics):
        def render(self) -> None:
            super().render()
            raise RuntimeError("render failed")

    class DirectCamera:
        render_update_count = 2

        def __init__(self, name: str) -> None:
            self.name = name

        def set_render_active(self, active: bool) -> None:
            events.append(f"active:{self.name}:{active}")

    coordinator = RenderCoordinator(
        physics_runtime=FailingPhysics(events),
        cameras=CameraBundle(
            cameras=(DirectCamera("first"), DirectCamera("second")),
            capture_hook=lambda _cameras: events.append("capture"),
        ),
    )

    with pytest.raises(RuntimeError, match="render failed"):
        coordinator.render_frame()

    assert events[-2:] == ["active:first:True", "active:second:True"]
    assert "capture" not in events


def test_runtime_admission_capacities_come_from_strict_interface_profile() -> None:
    events: list[str] = []
    config = load_mirror_config()
    runtime = create_mirror_runtime(
        config,
        assembly_factory=lambda cfg: _assembly(cfg, events=events),
    )

    assert runtime.controller.admission.capacity == (
        config.control.interface.admission_capacity
    )
    assert runtime.controller.admission.terminal_capacity == (
        config.control.interface.terminal_history_capacity
    )
    runtime.close()


def test_shutdown_timeout_retains_owner_and_does_not_close_session_early() -> None:
    events: list[str] = []
    config = load_mirror_config()
    runtime = create_mirror_runtime(
        config,
        assembly_factory=lambda cfg: _assembly(cfg, events=events),
    )
    ingress = _Resource("slow_ingress", events, results=[False, True])
    runtime.attach_ingress(ingress)

    first = runtime.close()
    assert first.stopped is False
    assert first.live_resources == ("slow_ingress",)
    assert runtime.session.close_calls == 0
    assert "output" not in events

    second = runtime.close()
    assert second.stopped is True
    assert ingress.calls == 2
    assert runtime.session.close_calls == 1


def test_newton_runtime_composition_asserts_one_world_after_assembly() -> None:
    events: list[str] = []
    config = load_mirror_config("newton_cuda")
    physics = _DirectPhysics(events)
    runtime = create_mirror_runtime(
        config,
        assembly_factory=lambda cfg: _assembly(cfg, events=events, physics=physics),
    )

    assert events == ["assert_single_world:Mirror"]
    runtime.close()


def test_newton_runtime_assertion_failure_rolls_back_session() -> None:
    events: list[str] = []
    config = load_mirror_config("newton_cuda")
    physics = _DirectPhysics(events, fail_assert=True)

    with pytest.raises(RuntimeError, match="actual=2"):
        create_mirror_runtime(
            config,
            assembly_factory=lambda cfg: _assembly(
                cfg,
                events=events,
                physics=physics,
            ),
        )

    assert "assert_single_world:Mirror" in events
    assert "session:1" in events


def test_runtime_rejects_engine_access_from_non_owner_thread() -> None:
    events: list[str] = []
    runtime = create_mirror_runtime(
        load_mirror_config(),
        assembly_factory=lambda cfg: _assembly(cfg, events=events),
    )
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            runtime.step()
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=worker)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert "owner thread" in str(errors[0])
    runtime.close()


def test_newton_camera_ownership_does_not_return_to_physics_runtime() -> None:
    camera_source = (
        REPO_ROOT / "src/linkerbot_sim/sensors/camera/runtime.py"
    ).read_text(encoding="utf-8")
    manager_source = (
        REPO_ROOT / "src/linkerbot_sim/isaac/physics/newton/manager.py"
    ).read_text(encoding="utf-8")

    assert "register_render_resource" not in camera_source
    assert "register_render_resource" not in manager_source
    assert "_render_resources" not in manager_source


def test_run_loop_advances_idle_hold_but_freezes_after_estop() -> None:
    events: list[str] = []
    runtime = create_mirror_runtime(
        load_mirror_config(),
        assembly_factory=lambda cfg: _assembly(cfg, events=events),
    )

    first = run_mirror(
        runtime,
        poll_timeout_s=0.001,
        max_iterations=2,
        close_on_exit=False,
    )
    estop = runtime.controller.submit_and_wait(
        MirrorRequest(
            protocol=MIRROR_PROTOCOL,
            request_id="estop-loop",
            operation="runtime.estop",
            arguments={},
        ),
        timeout_s=0.1,
    )
    before = events.count("physics_step:False")
    second = run_mirror(
        runtime,
        poll_timeout_s=0.001,
        max_iterations=2,
        close_on_exit=False,
    )

    assert first.physics_steps == 2
    assert estop.ok is True
    assert second.physics_steps == 0
    assert events.count("physics_step:False") == before
    runtime.close()


@pytest.mark.parametrize(
    ("sync_enabled", "expected_timeout_s"),
    ((True, 1.0 / 60.0), (False, 0.05)),
)
def test_run_loop_caps_queue_poll_to_physics_dt_only_when_synchronized(
    sync_enabled: bool,
    expected_timeout_s: float,
) -> None:
    events: list[str] = []
    config = load_mirror_config()
    config = replace(
        config,
        control=replace(
            config.control,
            sync_simulation_to_wall_clock=sync_enabled,
        ),
    )
    runtime = create_mirror_runtime(
        config,
        assembly_factory=lambda cfg: _assembly(cfg, events=events),
    )
    observed_timeouts: list[float] = []
    original_process_next = runtime.controller.process_next

    def process_next(*, timeout_s: float):
        observed_timeouts.append(timeout_s)
        return original_process_next(timeout_s=0.0001)

    runtime.controller.process_next = process_next  # type: ignore[method-assign]

    run_mirror(
        runtime,
        poll_timeout_s=0.05,
        max_iterations=1,
        close_on_exit=False,
    )

    assert observed_timeouts == pytest.approx([expected_timeout_s])
    runtime.close()


def test_idle_step_samples_scene_outputs_after_physics() -> None:
    events: list[str] = []
    runtime = create_mirror_runtime(
        load_mirror_config(),
        assembly_factory=lambda cfg: _assembly(cfg, events=events),
    )
    runtime.scene_resources = SimpleNamespace(
        observe_after_step=lambda *, phase: events.append(f"observe:{phase}")
    )

    runtime.step()

    assert events[:2] == ["physics_step:False", "observe:idle"]
    runtime.close()


def test_scene_outputs_share_one_global_step_and_idle_log_is_not_duplicated() -> None:
    events: list[tuple[str, int]] = []

    class Observer:
        def __init__(self, name: str) -> None:
            self.name = name

        def observe(self, _runtime: object, *, step: int, phase: str) -> None:
            del phase
            events.append((self.name, step))

    class Logger:
        def should_write(self, _step: int) -> bool:
            return True

        def collect_step_values(self, *_args: object) -> dict[str, object]:
            return {}

        def write(self, *, step: int, **_kwargs: object) -> None:
            events.append(("log", step))

    controller = SimpleNamespace(
        last_control_targets=object(),
        driven_indices=(0,),
    )
    execution = SimpleNamespace(
        drive_logger=Logger(),
        joint_controller=controller,
        articulation=object(),
    )
    resources = object.__new__(MirrorSceneResources)
    resources._completed_physics_steps = 0
    resources.physics = SimpleNamespace(get_physics_dt=lambda: 0.01)
    resources.robot_registry = SimpleNamespace(
        robots_by_id={0: SimpleNamespace(execution=execution)}
    )
    resources.state_observer = Observer("state")
    resources.camera_observer = Observer("camera")

    resources.observe_after_step(phase="idle")
    motion_step = resources.claim_completed_step()
    resources.observe_after_step(
        step=motion_step,
        phase="motion",
        write_idle_logs=False,
    )

    assert events == [
        ("log", 0),
        ("state", 0),
        ("camera", 0),
        ("state", 1),
        ("camera", 1),
    ]
    resources.reset_observation_clock()
    assert resources.claim_completed_step() == 0


def test_transport_start_failure_still_closes_runtime_owner_graph() -> None:
    events: list[str] = []
    runtime = create_mirror_runtime(
        load_mirror_config(),
        assembly_factory=lambda cfg: _assembly(cfg, events=events),
    )

    class FailingEndpoint:
        def start(self) -> None:
            raise RuntimeError("bind failed")

        def close(self) -> bool:
            events.append("failed_endpoint_closed")
            return True

    with pytest.raises(RuntimeError, match="bind failed"):
        run_mirror(runtime, endpoints=(FailingEndpoint(),), poll_timeout_s=0.001)

    assert runtime.is_closed
    assert "failed_endpoint_closed" in events
    assert runtime.session.close_calls == 1


def test_strict_outputs_project_to_camera_and_state_stream_without_defaults() -> None:
    outputs = load_mirror_config().outputs
    camera = _camera_output_settings("world_rgbd", outputs.camera)
    telemetry = _state_stream_config(outputs.telemetry, mcap_plan=None)

    assert camera.save_dir == "logs/cameras/world_rgbd"
    assert camera.foxglove_topic_prefix == "/cameras/world_rgbd"
    assert outputs.camera.rgb_format == "png"
    assert outputs.camera.depth_format == "npz"
    assert outputs.camera.max_bytes_per_camera == 10 * 1024**3
    assert telemetry is not None
    assert telemetry.rate_hz == outputs.telemetry.rate_hz
    assert telemetry.foxglove_mcap_path == "logs/telemetry/mirror.mcap"
    assert telemetry.topics.state == "/linkerbot/mirror/state"
    assert telemetry.include_hybrid_control is True
    assert telemetry.topics.hybrid_control == "/linkerbot/mirror/hybrid_control"


def test_disabled_outputs_ignore_retained_consumer_settings() -> None:
    outputs = load_mirror_config().outputs
    camera_settings = replace(outputs.camera, enabled=False)
    telemetry_settings = replace(outputs.telemetry, enabled=False)

    camera = _camera_output_settings("world_rgbd", camera_settings)
    telemetry = _state_stream_config(telemetry_settings, mcap_plan=None)

    assert camera_settings.save_root == "logs/cameras"
    assert camera_settings.foxglove_live_port == 8849
    assert not camera.has_consumer
    assert telemetry_settings.mcap_path == "logs/telemetry/mirror.mcap"
    assert telemetry is None


def test_disabled_effort_output_ignores_retained_source() -> None:
    telemetry_settings = replace(
        load_mirror_config().outputs.telemetry,
        include_efforts=False,
        joint_effort_field="measured",
    )

    telemetry = _state_stream_config(telemetry_settings, mcap_plan=None)

    assert telemetry is not None
    assert telemetry.include_efforts is False
    assert telemetry.foxglove_joint_effort_field == "none"
