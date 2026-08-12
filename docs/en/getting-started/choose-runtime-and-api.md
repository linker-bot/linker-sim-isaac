# Choose A Mode And API

Language: [English](choose-runtime-and-api.md) | [中文](../../zh-CN/getting-started/choose-runtime-and-api.md)

Choose the product first, then choose the narrowest interface that matches the
consumer.

## Product Decision

| Requirement | Mirror | Kaleidoscope |
| --- | --- | --- |
| Reality-mapped workspace | Yes | No |
| More than one robot in a world | Yes | Homogeneous robots per replicated environment |
| Full planning and avoidance | Yes | No |
| Cameras, telemetry, persistent outputs | Yes | No |
| JSON process boundary | Yes | No |
| PhysX CUDA vector execution | No | Yes, `physx_cuda` |
| Newton | Yes, one world | Yes, project-owned multi-world runtime |
| GPU state/snapshot/clone | No | Yes |
| Batched IK and synchronous linear EE actions | Low-batch interactive operations | Yes |
| Gymnasium or skrl | No | Yes |

Use Mirror when the simulation mirrors one physical cell or when a person/service
needs interactive planning, inspection, cameras, or remote JSON control. Use
Kaleidoscope when a trainer needs hundreds of homogeneous environments and can accept
a headless, fixed-shape action/observation contract.

## Mirror Interfaces

### JSON

Use JSON for process isolation or language-neutral clients. Every request contains
exactly `protocol`, `request_id`, `operation`, and `arguments`. The supported ingress
mechanisms are stdin, loopback TCP JSONL, and loopback WebSocket text frames.

JSON is asynchronous at ingress but serialized at the Isaac owner thread. Queue and
motion cancellation remain bounded. See [Mirror JSON](../reference/mirror-json.md).

### Python

Use the Python facade for embedding, direct state access, custom output integration,
or deterministic lifecycle control:

```python
from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.mirror import create_mirror_runtime

config = load_mirror_config("physx_cpu")
runtime = create_mirror_runtime(config)
try:
    state = runtime.get_state()
    runtime.step(render=False)
finally:
    runtime.close()
```

The caller that creates the runtime must remain on its owner thread.

## Kaleidoscope Interfaces

The public tensor contract is backend-neutral. The `physx_cuda` profile selects PhysX
CUDA/Fabric; `newton_cuda` selects the project's multi-world Newton
owner, not the Isaac Newton extension. Both training compositions are headless and
GPU-native and derive physics, Torch, cuRobo, and trainer placement from the same
`compute.cuda_device`. `make_viewport_env()` can explicitly display one selected
environment for either backend.

### Native Torch

This is the preferred environment and debugging API. Observations, actions, rewards,
done flags, state, and snapshots remain CUDA tensors. It also exposes partial reset
and GPU cloning. `get_state`, `set_state`, `snapshot`, `restore_snapshot`, and
`clone_state` keep registered physics, task/history/counter, and RNG fields on device;
only an explicit persistent checkpoint crosses the host boundary.

### skrl

Use `linkerbot_sim.training.skrl.SkrlTorchAdapter` for the CUDA-native training path.
Its generation token guarantees that terminal transitions are copied before the done
rows are reset in the same decision. The paired rollout memory and PPO implementation
avoid host-side selector construction and preserve the final observation used for
time-limit bootstrap.

### Gymnasium

Use `GymnasiumKaleidoscopeAdapter` only for ecosystem compatibility. It is a deliberate
NumPy boundary: actions move host-to-device and results move device-to-host on every
step. Supported autoreset modes are `disabled` and `same_step`; `next_step` is not
implemented.

### Human Viewport

`make_viewport_env()` reads a separate launch-only profile and projects only
`selected_env` into renderer-facing USD. Training steps remain `render=False`; callers
invoke `env.render()` explicitly. This boundary adds no camera, SyntheticData,
Replicator, recording, or image observation and does not alter snapshot/clone
fingerprints.

### No JSON Service

Kaleidoscope has no CLI server or RPC transport. Adding one to the hot runtime would
introduce serialization, host synchronization, queueing, and ambiguous ownership.
Build any monitoring process outside the environment boundary and sample explicitly.

## Recommended Starting Points

- Reality integration: [Mirror quickstart](mirror-quickstart.md)
- Native or framework training: [Kaleidoscope quickstart](kaleidoscope-quickstart.md)
- Exact surfaces: [Python API](../reference/python-api.md)
