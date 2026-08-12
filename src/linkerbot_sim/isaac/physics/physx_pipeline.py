"""PhysX CPU/GPU tensor pipeline 的 World 参数和运行时验收。"""

from __future__ import annotations

import json
from collections.abc import Mapping

from linkerbot_sim.isaac.spec import (
    IsaacPhysxCpuSpec,
    IsaacPhysxCudaSpec,
    IsaacSessionSpec,
)


_GPU_BUFFER_BINDINGS = (
    (
        "max_rigid_contact_count",
        "get_gpu_max_rigid_contact_count",
    ),
    (
        "max_rigid_patch_count",
        "get_gpu_max_rigid_patch_count",
    ),
    (
        "found_lost_pairs_capacity",
        "get_gpu_found_lost_pairs_capacity",
    ),
    (
        "found_lost_aggregate_pairs_capacity",
        "get_gpu_found_lost_aggregate_pairs_capacity",
    ),
    (
        "total_aggregate_pairs_capacity",
        "get_gpu_total_aggregate_pairs_capacity",
    ),
    (
        "collision_stack_size",
        "get_gpu_collision_stack_size",
    ),
    ("heap_capacity", "get_gpu_heap_capacity"),
    (
        "temp_buffer_capacity",
        "get_gpu_temp_buffer_capacity",
    ),
    (
        "max_num_partitions",
        "get_gpu_max_num_partitions",
    ),
    (
        "max_soft_body_contacts",
        "get_gpu_max_soft_body_contacts",
    ),
    (
        "max_particle_contacts",
        "get_gpu_max_particle_contacts",
    ),
)

_FABRIC_OUTPUT_SETTING_PATHS = {
    "transformations": "/physics/fabricUpdateTransformations",
    "velocities": "/physics/fabricUpdateVelocities",
    "force_sensors": "/physics/fabricUpdateForceSensors",
    "joint_states": "/physics/fabricUpdateJointStates",
    "points": "/physics/fabricUpdatePoints",
}


def physx_fabric_output_policy(*, rendering_required: bool) -> dict[str, bool]:
    """解析当前 session 唯一的进程级 Fabric 输出策略。

    Kaleidoscope 没有渲染消费者，因此不维护重复的 USD/Fabric 输出状态；Mirror 只有在
    render/camera 确实需要 transform 时才开启对应同步。
    """

    if type(rendering_required) is not bool:
        raise TypeError("rendering_required must be a boolean")
    return {
        "transformations": rendering_required,
        "velocities": False,
        "force_sensors": False,
        "joint_states": False,
        "points": False,
    }


def configure_physx_fabric_outputs(
    *,
    rendering_required: bool,
    settings_interface: object | None = None,
) -> Mapping[str, bool]:
    """在 World/PhysicsScene 创建前写入并回读验证 Fabric 输出。

    这些 setting 是进程级事实，晚于 World 创建再修改无法证明 tensor pipeline 与 active
    scene 一致，因此配置失败必须在取得 physics owner 前终止。
    """

    policy = physx_fabric_output_policy(rendering_required=rendering_required)
    if settings_interface is None:
        import carb

        settings_interface = carb.settings.get_settings()
    set_bool = getattr(settings_interface, "set_bool", None)
    get_as_bool = getattr(settings_interface, "get_as_bool", None)
    if not callable(set_bool) or not callable(get_as_bool):
        raise RuntimeError("Carb settings interface cannot configure Fabric outputs")
    for name, expected in policy.items():
        set_bool(_FABRIC_OUTPUT_SETTING_PATHS[name], expected)
    actual = {
        name: bool(get_as_bool(path))
        for name, path in _FABRIC_OUTPUT_SETTING_PATHS.items()
    }
    if actual != policy:
        raise RuntimeError(
            f"PhysX Fabric output policy mismatch: expected={policy!r}, "
            f"active={actual!r}"
        )
    print(
        "PHYSX_FABRIC_OUTPUT_POLICY "
        + json.dumps(
            actual,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return actual


def resolved_physx_device(
    spec: IsaacSessionSpec,
) -> str:
    """返回完整 session 已解析的 PhysX 设备事实。"""

    if not isinstance(spec, IsaacSessionSpec):
        raise TypeError("spec must be IsaacSessionSpec")
    physics = spec.physics
    if not isinstance(physics, (IsaacPhysxCpuSpec, IsaacPhysxCudaSpec)):
        raise TypeError("spec.physics must be an Isaac PhysX specification")
    return spec.physics_device


def build_physx_world_kwargs(
    spec: IsaacSessionSpec,
) -> dict[str, object]:
    """生成 PhysX CUDA ``World`` 参数；CPU 明确返回空字典。"""

    device = resolved_physx_device(spec)
    physics = spec.physics
    assert isinstance(physics, (IsaacPhysxCpuSpec, IsaacPhysxCudaSpec))
    if isinstance(physics, IsaacPhysxCpuSpec):
        return {}

    sim_params: dict[str, object] = {
        "enable_scene_query_support": physics.enable_scene_query_support,
        "use_gpu_pipeline": True,
        "use_fabric": True,
    }
    return {
        "backend": "torch",
        "device": device,
        "sim_params": sim_params,
    }


def probe_physx_tensor_pipeline(
    world: object,
    spec: IsaacSessionSpec,
    *,
    fabric_outputs: Mapping[str, bool] | None = None,
) -> Mapping[str, object] | None:
    """回读并验收 GPU PhysX 状态，成功时输出稳定的一行 JSON 诊断。"""

    expected_device = resolved_physx_device(spec)
    physics = spec.physics
    assert isinstance(physics, (IsaacPhysxCpuSpec, IsaacPhysxCudaSpec))
    if isinstance(physics, IsaacPhysxCpuSpec):
        return None

    try:
        physics_context = world.get_physics_context()
        carb_settings = getattr(physics_context, "_carb_settings", None)
        get_as_bool = getattr(carb_settings, "get_as_bool", None)
        if not callable(get_as_bool):
            raise RuntimeError("PhysicsContext carb settings are unavailable")
        actual_buffers = {
            config_name: int(getattr(physics_context, getter_name)())
            for config_name, getter_name in _GPU_BUFFER_BINDINGS
        }
        diagnostic: dict[str, object] = {
            "backend": str(world.backend),
            "broadphase": str(physics_context.get_broadphase_type()),
            "device": str(world.device),
            "fabric": bool(physics_context.use_fabric),
            "fabric_gpu_interop": bool(get_as_bool("/physics/fabricUseGPUInterop")),
            "fabric_outputs": {
                name: bool(get_as_bool(path))
                for name, path in _FABRIC_OUTPUT_SETTING_PATHS.items()
            },
            "gpu_buffers": actual_buffers,
            "gpu_dynamics": bool(physics_context.is_gpu_dynamics_enabled()),
            "gpu_pipeline": bool(physics_context.use_gpu_pipeline),
            "scene_query_support": bool(
                physics_context.get_enable_scene_query_support()
            ),
            "suppress_readback": bool(get_as_bool("/physics/suppressReadback")),
        }
    except Exception as exc:
        raise RuntimeError(
            "PhysX CUDA tensor pipeline probe could not read runtime state"
        ) from exc

    expected: dict[str, object] = {
        "backend": "torch",
        "broadphase": "GPU",
        "device": expected_device,
        "fabric": True,
        "fabric_gpu_interop": True,
        "gpu_dynamics": True,
        "gpu_pipeline": True,
        "scene_query_support": physics.enable_scene_query_support,
        "suppress_readback": True,
    }
    if fabric_outputs is not None:
        expected_fabric_outputs = {
            name: bool(fabric_outputs[name]) for name in _FABRIC_OUTPUT_SETTING_PATHS
        }
        expected["fabric_outputs"] = expected_fabric_outputs
    mismatches = [
        key
        for key, expected_value in expected.items()
        if diagnostic[key] != expected_value
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: configured={expected[key]!r}, active={diagnostic[key]!r}"
            for key in mismatches
        )
        raise RuntimeError(f"PhysX CUDA tensor pipeline mismatch: {details}")

    print(
        "PHYSX_TENSOR_PIPELINE "
        + json.dumps(
            diagnostic,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return diagnostic


__all__ = [
    "build_physx_world_kwargs",
    "configure_physx_fabric_outputs",
    "physx_fabric_output_policy",
    "probe_physx_tensor_pipeline",
    "resolved_physx_device",
]
