from __future__ import annotations

import json

import pytest

from linkerbot_sim.isaac.physics.physx_pipeline import (
    build_physx_world_kwargs,
    configure_physx_fabric_outputs,
    physx_fabric_output_policy,
    probe_physx_tensor_pipeline,
)
from linkerbot_sim.isaac.spec import (
    IsaacComputeSpec,
    IsaacPhysxCpuSpec,
    IsaacPhysxCudaSpec,
    IsaacSessionSpec,
)


_GETTERS = {
    "get_gpu_max_rigid_contact_count": "max_rigid_contact_count",
    "get_gpu_max_rigid_patch_count": "max_rigid_patch_count",
    "get_gpu_found_lost_pairs_capacity": "found_lost_pairs_capacity",
    "get_gpu_found_lost_aggregate_pairs_capacity": (
        "found_lost_aggregate_pairs_capacity"
    ),
    "get_gpu_total_aggregate_pairs_capacity": "total_aggregate_pairs_capacity",
    "get_gpu_collision_stack_size": "collision_stack_size",
    "get_gpu_heap_capacity": "heap_capacity",
    "get_gpu_temp_buffer_capacity": "temp_buffer_capacity",
    "get_gpu_max_num_partitions": "max_num_partitions",
    "get_gpu_max_soft_body_contacts": "max_soft_body_contacts",
    "get_gpu_max_particle_contacts": "max_particle_contacts",
}

_ENGINE_GPU_BUFFER_VALUES = {
    name: index * 1_000 for index, name in enumerate(_GETTERS.values(), start=1)
}

_FABRIC_OUTPUT_PATHS = {
    "transformations": "/physics/fabricUpdateTransformations",
    "velocities": "/physics/fabricUpdateVelocities",
    "force_sensors": "/physics/fabricUpdateForceSensors",
    "joint_states": "/physics/fabricUpdateJointStates",
    "points": "/physics/fabricUpdatePoints",
}


def _cpu_spec(*, cuda_device: int = 2) -> IsaacSessionSpec:
    return IsaacSessionSpec(
        experience_family="mirror",
        compute=IsaacComputeSpec(cuda_device=cuda_device),
        physics=IsaacPhysxCpuSpec(),
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
    )


def _cuda_spec(*, cuda_device: int = 2) -> IsaacSessionSpec:
    return IsaacSessionSpec(
        experience_family="kaleidoscope",
        compute=IsaacComputeSpec(cuda_device=cuda_device),
        physics=IsaacPhysxCudaSpec(),
        physics_dt=1.0 / 240.0,
        rendering_dt=1.0 / 60.0,
        gravity_z=-9.81,
    )


class _PhysicsContext:
    use_fabric = True
    use_gpu_pipeline = True

    def __init__(
        self,
        *,
        gpu_buffers: dict[str, int] | None = None,
        suppress_readback: bool = True,
        fabric_gpu_interop: bool = True,
        scene_query_support: bool = False,
        fabric_outputs: dict[str, bool] | None = None,
    ) -> None:
        self._buffers = dict(
            _ENGINE_GPU_BUFFER_VALUES if gpu_buffers is None else gpu_buffers
        )
        self._scene_query_support = bool(scene_query_support)
        self._carb_settings = _CarbSettings(
            suppress_readback=suppress_readback,
            fabric_gpu_interop=fabric_gpu_interop,
            fabric_outputs=fabric_outputs,
        )

    def get_broadphase_type(self) -> str:
        return "GPU"

    def is_gpu_dynamics_enabled(self) -> bool:
        return True

    def get_enable_scene_query_support(self) -> bool:
        return self._scene_query_support

    def __getattr__(self, name: str):
        if name not in _GETTERS:
            raise AttributeError(name)
        return lambda: self._buffers[_GETTERS[name]]


class _CarbSettings:
    def __init__(
        self,
        *,
        suppress_readback: bool,
        fabric_gpu_interop: bool,
        fabric_outputs: dict[str, bool] | None = None,
    ) -> None:
        self.set_calls: list[tuple[str, bool]] = []
        self.values = {
            "/physics/fabricUseGPUInterop": bool(fabric_gpu_interop),
            "/physics/suppressReadback": bool(suppress_readback),
            **{
                path: bool((fabric_outputs or {}).get(name, False))
                for name, path in _FABRIC_OUTPUT_PATHS.items()
            },
        }

    def set_bool(self, path: str, value: bool) -> None:
        self.set_calls.append((path, bool(value)))
        self.values[path] = bool(value)

    def get_as_bool(self, path: str) -> bool:
        return self.values[path]


class _World:
    backend = "torch"
    device = "cuda:2"

    def __init__(self, context: object) -> None:
        self._context = context

    def get_physics_context(self) -> object:
        return self._context


def test_cpu_pipeline_has_no_gpu_world_kwargs_or_probe_output(
    capsys,
) -> None:
    settings = _cpu_spec(cuda_device=7)

    assert build_physx_world_kwargs(settings) == {}
    assert probe_physx_tensor_pipeline(object(), settings) is None
    assert capsys.readouterr().out == ""


def test_cuda_pipeline_leaves_gpu_buffer_capacities_at_engine_defaults() -> None:
    settings = _cuda_spec()

    kwargs = build_physx_world_kwargs(settings)

    assert kwargs["backend"] == "torch"
    assert kwargs["device"] == "cuda:2"
    sim_params = kwargs["sim_params"]
    assert isinstance(sim_params, dict)
    assert sim_params == {
        "enable_scene_query_support": False,
        "use_gpu_pipeline": True,
        "use_fabric": True,
    }
    assert not any(name.startswith("gpu_") for name in sim_params)


@pytest.mark.parametrize(
    ("rendering_required", "expected"),
    (
        (
            False,
            {
                "transformations": False,
                "velocities": False,
                "force_sensors": False,
                "joint_states": False,
                "points": False,
            },
        ),
        (
            True,
            {
                "transformations": True,
                "velocities": False,
                "force_sensors": False,
                "joint_states": False,
                "points": False,
            },
        ),
    ),
)
def test_fabric_output_policy_only_enables_render_transformations(
    rendering_required: bool,
    expected: dict[str, bool],
) -> None:
    assert physx_fabric_output_policy(rendering_required=rendering_required) == expected


@pytest.mark.parametrize("value", (0, 1, None, "false"))
def test_fabric_output_policy_rejects_non_boolean_rendering_requirement(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="rendering_required must be a boolean"):
        physx_fabric_output_policy(
            rendering_required=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("rendering_required", (False, True))
def test_configure_fabric_outputs_authors_and_reads_back_all_five_settings(
    rendering_required: bool,
    capsys,
) -> None:
    settings_interface = _CarbSettings(
        suppress_readback=True,
        fabric_gpu_interop=True,
        fabric_outputs={name: not rendering_required for name in _FABRIC_OUTPUT_PATHS},
    )
    expected = physx_fabric_output_policy(rendering_required=rendering_required)

    actual = configure_physx_fabric_outputs(
        rendering_required=rendering_required,
        settings_interface=settings_interface,
    )

    assert actual == expected
    assert {
        name: settings_interface.get_as_bool(path)
        for name, path in _FABRIC_OUTPUT_PATHS.items()
    } == expected
    assert settings_interface.set_calls == [
        (path, expected[name]) for name, path in _FABRIC_OUTPUT_PATHS.items()
    ]
    line = capsys.readouterr().out.strip()
    assert json.loads(line.removeprefix("PHYSX_FABRIC_OUTPUT_POLICY ")) == expected


def test_configure_fabric_outputs_fails_closed_on_readback_mismatch(capsys) -> None:
    class _MismatchedSettings(_CarbSettings):
        def set_bool(self, path: str, value: bool) -> None:
            if path != _FABRIC_OUTPUT_PATHS["points"]:
                super().set_bool(path, value)

    settings_interface = _MismatchedSettings(
        suppress_readback=True,
        fabric_gpu_interop=True,
        fabric_outputs={"points": True},
    )

    with pytest.raises(RuntimeError, match="Fabric output policy mismatch"):
        configure_physx_fabric_outputs(
            rendering_required=False,
            settings_interface=settings_interface,
        )

    assert capsys.readouterr().out == ""


def test_cuda_probe_reads_all_state_and_emits_stable_json(capsys) -> None:
    settings = _cuda_spec()
    world = _World(_PhysicsContext())
    fabric_outputs = physx_fabric_output_policy(rendering_required=False)

    diagnostic = probe_physx_tensor_pipeline(
        world,
        settings,
        fabric_outputs=fabric_outputs,
    )

    line = capsys.readouterr().out.strip()
    assert line.startswith("PHYSX_TENSOR_PIPELINE ")
    assert json.loads(line.removeprefix("PHYSX_TENSOR_PIPELINE ")) == diagnostic
    assert diagnostic is not None
    assert diagnostic["backend"] == "torch"
    assert diagnostic["device"] == "cuda:2"
    assert diagnostic["fabric_gpu_interop"] is True
    assert diagnostic["fabric_outputs"] == fabric_outputs
    assert diagnostic["scene_query_support"] is False
    assert diagnostic["suppress_readback"] is True
    assert diagnostic["gpu_buffers"] == _ENGINE_GPU_BUFFER_VALUES


def test_cuda_probe_fails_closed_on_runtime_mismatch(capsys) -> None:
    settings = _cuda_spec()
    world = _World(_PhysicsContext())
    world.backend = "numpy"

    with pytest.raises(RuntimeError, match=r"mismatch: backend"):
        probe_physx_tensor_pipeline(world, settings)

    assert capsys.readouterr().out == ""


def test_cuda_probe_fails_closed_when_readback_is_not_suppressed(capsys) -> None:
    settings = _cuda_spec()
    world = _World(_PhysicsContext(suppress_readback=False))

    with pytest.raises(RuntimeError, match=r"mismatch: suppress_readback"):
        probe_physx_tensor_pipeline(world, settings)

    assert capsys.readouterr().out == ""


def test_cuda_probe_fails_closed_when_fabric_gpu_interop_is_disabled(capsys) -> None:
    settings = _cuda_spec()
    world = _World(_PhysicsContext(fabric_gpu_interop=False))

    with pytest.raises(RuntimeError, match=r"mismatch: fabric_gpu_interop"):
        probe_physx_tensor_pipeline(world, settings)

    assert capsys.readouterr().out == ""


def test_cuda_probe_fails_closed_on_scene_query_mismatch(capsys) -> None:
    settings = _cuda_spec()
    world = _World(_PhysicsContext(scene_query_support=True))

    with pytest.raises(RuntimeError, match=r"mismatch: scene_query_support"):
        probe_physx_tensor_pipeline(world, settings)

    assert capsys.readouterr().out == ""


def test_cuda_probe_fails_closed_on_fabric_output_mismatch(capsys) -> None:
    settings = _cuda_spec()
    expected_outputs = physx_fabric_output_policy(rendering_required=False)
    active_outputs = dict(expected_outputs)
    active_outputs["transformations"] = True
    world = _World(_PhysicsContext(fabric_outputs=active_outputs))

    with pytest.raises(RuntimeError, match=r"mismatch: fabric_outputs"):
        probe_physx_tensor_pipeline(
            world,
            settings,
            fabric_outputs=expected_outputs,
        )

    assert capsys.readouterr().out == ""


def test_cuda_probe_fails_closed_when_runtime_state_cannot_be_read(capsys) -> None:
    settings = _cuda_spec()

    with pytest.raises(RuntimeError, match="could not read runtime state"):
        probe_physx_tensor_pipeline(_World(object()), settings)

    assert capsys.readouterr().out == ""


def test_cuda_probe_reports_engine_buffer_defaults_without_enforcing_values(
    capsys,
) -> None:
    settings = _cuda_spec()
    active_buffers = dict(_ENGINE_GPU_BUFFER_VALUES)
    active_buffers["max_rigid_contact_count"] = 1

    diagnostic = probe_physx_tensor_pipeline(
        _World(_PhysicsContext(gpu_buffers=active_buffers)), settings
    )

    assert diagnostic is not None
    assert diagnostic["gpu_buffers"] == active_buffers
    assert capsys.readouterr().out.startswith("PHYSX_TENSOR_PIPELINE ")
