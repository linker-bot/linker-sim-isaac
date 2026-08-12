from __future__ import annotations

import pytest

from linkerbot_sim.configuration.common import ConfigurationError
from linkerbot_sim.configuration.physics import (
    PhysxCudaSettings,
    physics_settings_from_mapping,
)


def _physx_cuda_mapping() -> dict[str, object]:
    return {
        "engine": "physx",
        "execution": "cuda",
        "solver_type": "PGS",
        "use_fabric": True,
        "enable_scene_query_support": False,
        "memory": {
            "max_simulator_process_mib": 16_384,
            "min_free_floor_mib": 4_096,
            "min_free_fraction_after_warmup": 0.2,
            "max_steady_growth_mib": 128,
        },
    }


def test_physx_cuda_configuration_uses_engine_gpu_buffer_defaults() -> None:
    settings = physics_settings_from_mapping(_physx_cuda_mapping())

    assert isinstance(settings, PhysxCudaSettings)
    assert not hasattr(settings, "gpu_buffers")


def test_physx_cuda_configuration_rejects_removed_gpu_buffer_overrides() -> None:
    mapping = _physx_cuda_mapping()
    mapping["gpu_buffers"] = {"max_rigid_contact_count": 1}

    with pytest.raises(ConfigurationError, match=r"未知字段: gpu_buffers"):
        physics_settings_from_mapping(mapping)
