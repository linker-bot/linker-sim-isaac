"""Mirror Newton 单世界装配合同。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import linkerbot_sim.mirror.scene_assembly as mirror_assembly
from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.isaac.spec import (
    IsaacComputeSpec,
    IsaacNewtonCpuSpec,
    IsaacNewtonCudaSpec,
    IsaacRenderSpec,
    IsaacSessionSpec,
)


def _newton_session(manager: object) -> SimpleNamespace:
    return SimpleNamespace(
        physics_runtime=manager,
        single_articulation_type=object(),
    )


def test_newton_single_world_initializes_before_binding_articulations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class Manager:
        world_count = 0

        def initialize_worlds(self, **kwargs: object) -> None:
            events.append(("initialize", kwargs))
            self.world_count = 1

    manager = Manager()
    session = _newton_session(manager)
    instances = (SimpleNamespace(label="left"), SimpleNamespace(label="right"))
    imported = (object(), object())
    execution_configs = (
        SimpleNamespace(robot=SimpleNamespace(name="left_model")),
        SimpleNamespace(robot=SimpleNamespace(name="right_model")),
    )

    def bind(imported_robot: object, **kwargs: object) -> object:
        assert manager.world_count == 1
        events.append(("bind", imported_robot, kwargs))
        return SimpleNamespace(source=imported_robot, name=kwargs["name"])

    monkeypatch.setattr(
        mirror_assembly,
        "bind_imported_robot_articulation",
        bind,
    )
    handles = (SimpleNamespace(name="workstation"), SimpleNamespace(name="TBlock"))

    physics = mirror_assembly.MirrorPhysicsAdapter(manager)
    bound = mirror_assembly._initialize_newton_runtime_mirror(
        session=session,
        physics=physics,
        instances=instances,
        imported=imported,
        execution_configs=execution_configs,
        object_handles=handles,
    )

    assert [event[0] for event in events] == ["initialize", "bind", "bind"]
    initialize = events[0][1]
    assert initialize["env_root_paths"] == ("/World",)
    assert initialize["env_origins"] == ((0.0, 0.0, 0.0),)
    assert initialize["robots"] == {"left": imported[0], "right": imported[1]}
    assert initialize["object_handles"] is handles
    assert [item.name for item in bound] == ["left_model", "right_model"]


def test_newton_single_world_rejects_wrong_count_before_view_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SimpleNamespace(
        world_count=2,
        initialize_worlds=lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        mirror_assembly,
        "bind_imported_robot_articulation",
        lambda *_args, **_kwargs: pytest.fail("view binding must not run"),
    )

    with pytest.raises(RuntimeError, match="exactly one world"):
        mirror_assembly._initialize_newton_runtime_mirror(
            session=_newton_session(manager),
            physics=mirror_assembly.MirrorPhysicsAdapter(manager),
            instances=(SimpleNamespace(label="left"),),
            imported=(object(),),
            execution_configs=(
                SimpleNamespace(robot=SimpleNamespace(name="left_model")),
            ),
            object_handles=(),
        )


def test_newton_mirror_reserves_viewport_before_assets_and_initializes_after_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class Manager:
        backend = "newton"
        kind = "newton_cuda"
        execution = "cuda"
        world_count = 0
        scene = object()

        def initialize_worlds(self, **_kwargs: object) -> None:
            events.append("manager_initialize")
            self.world_count = 1

        def reset(self) -> None:
            events.append("physics_reset")

        def get_physics_dt(self) -> float:
            return 1.0 / 60.0

        def get_rendering_dt(self) -> float:
            return 1.0 / 60.0

    manager = Manager()
    session = SimpleNamespace(
        physics_runtime=manager,
        stage=object(),
        single_articulation_type=object(),
        articulation_action_type=object(),
        app=object(),
        close=lambda **_kwargs: events.append("app_closed"),
    )
    handles = (SimpleNamespace(name="rope"),)
    imported = SimpleNamespace()
    monkeypatch.setattr(
        mirror_assembly,
        "add_runtime_objects",
        lambda *_args, **_kwargs: events.append("objects_imported") or handles,
    )
    monkeypatch.setattr(
        mirror_assembly,
        "configure_visuals",
        lambda _settings, *, configure_viewport: events.append(
            ("visuals_configured", configure_viewport)
        ),
    )
    monkeypatch.setattr(
        mirror_assembly,
        "runtime_object_handles_by_name",
        lambda _handles: {},
    )
    monkeypatch.setattr(
        mirror_assembly,
        "import_execution_robot_to_stage",
        lambda **_kwargs: events.append("robot_imported") or imported,
    )
    monkeypatch.setattr(
        mirror_assembly,
        "bind_imported_robot_articulation",
        lambda value, **_kwargs: events.append("articulation_bound") or value,
    )
    monkeypatch.setattr(
        mirror_assembly,
        "set_physics_gravity",
        lambda *_args, **_kwargs: events.append("gravity_restored"),
    )
    monkeypatch.setattr(
        mirror_assembly,
        "create_scene_object_state_views",
        lambda *_args, **_kwargs: events.append("rigid_views_bound") or {},
    )

    def create_cameras(**_kwargs: object) -> tuple[object, ...]:
        events.append("camera_viewport_reserved")
        return (SimpleNamespace(close=lambda: events.append("camera_closed")),)

    monkeypatch.setattr(
        mirror_assembly,
        "create_sensor_camera_runtimes",
        create_cameras,
    )

    def initialize_cameras(_cameras: object) -> None:
        events.append("camera_sensor_initialized")
        raise RuntimeError("camera sentinel")

    monkeypatch.setattr(
        mirror_assembly,
        "initialize_sensor_camera_runtimes",
        initialize_cameras,
    )
    monkeypatch.setattr(
        mirror_assembly,
        "prepare_camera_output",
        lambda *_args, **_kwargs: pytest.fail("camera output must not open"),
    )
    session_spec = IsaacSessionSpec(
        experience_family="mirror",
        compute=IsaacComputeSpec(cuda_device=0),
        physics=IsaacNewtonCudaSpec(),
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
        render=IsaacRenderSpec(enabled=True),
    )
    config = load_mirror_config("newton_cuda")

    with pytest.raises(RuntimeError, match="camera sentinel"):
        mirror_assembly.create_mirror_scene_resources(
            scene=config.scene,
            session_spec=session_spec,
            output_settings=config.outputs,
            curobo_settings=config.curobo,
            controller_bundles=config.controller_bundles,
            controller_bundle=config.default_controller_bundle,
            control_mode=config.control.mode,
            session_factory=lambda **_kwargs: session,
        )

    assert events == [
        ("visuals_configured", False),
        "camera_viewport_reserved",
        "objects_imported",
        "robot_imported",
        "robot_imported",
        "manager_initialize",
        "articulation_bound",
        "articulation_bound",
        "physics_reset",
        "gravity_restored",
        "rigid_views_bound",
        "camera_sensor_initialized",
        "camera_closed",
        "app_closed",
    ]


@pytest.mark.parametrize(
    ("profile", "spec_type", "physics_device"),
    (
        ("newton_cpu", IsaacNewtonCpuSpec, "cpu"),
        ("newton_cuda", IsaacNewtonCudaSpec, "cuda:0"),
    ),
)
def test_mirror_newton_profiles_project_execution_without_duplicate_device(
    profile: str,
    spec_type: type,
    physics_device: str,
) -> None:
    config = load_mirror_config(profile)

    spec = mirror_assembly._mirror_session_spec(config)

    assert isinstance(spec.physics, spec_type)
    assert spec.physics.world_count == 1
    assert spec.physics_device == physics_device
    assert spec.compute_device == "cuda:0"
    assert spec.physics_execution == config.physics.execution
