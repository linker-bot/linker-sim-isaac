# Troubleshooting

Language: [English](troubleshooting.md) | [中文](../../zh-CN/operations/troubleshooting.md)

Start with the failing boundary: configuration, Kit/physics construction, product
composition, request/action validation, or shutdown.

## Startup

| Symptom | Check |
| --- | --- |
| EULA error | Export `OMNI_KIT_ACCEPT_EULA=Y` in the process environment. |
| `pxr` or extension mismatch | Remove the CPU `dev` extra from the Isaac environment and recreate it with `simulation`. |
| Missing or blank warehouse visuals | Verify the licensed NVIDIA payload at `usd-material/extracted/Industrial_NVD_10012/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd`; configuration validation does not fetch it. |
| Mode/profile error | Run `scripts/validate_mode_config.py` with the exact mode and profile. |
| Unknown configuration field | Move the fact to its owning leaf or remove an unsupported capability; do not bypass strict validation. |
| Unsupported physics selection | Mirror accepts PhysX/CPU and Newton/CPU or CUDA; Kaleidoscope accepts PhysX/CUDA or Newton/CUDA. |
| CUDA device mismatch | Keep the index only at `mode.compute.cuda_device` and rebuild the graph. |
| PhysX allocation failure | Reduce environment/contact load, then inspect the runtime pipeline diagnostics and process-memory gates. |
| Newton world construction failure | Mirror uses `newton_cpu` or `newton_cuda` with one derived world; Kaleidoscope uses `newton_cuda` and derives `world_count` from final `environments.num_envs`. |
| Newton robot drifts against a small position target | Verify that the robot profile's disabled gravity was authored as `mjc:gravcomp=1` before model finalization. Runtime per-link gravity setters are intentionally unsupported. |
| Isaac Newton extension appears | The wrong Kit closure loaded. Project-owned Newton excludes `isaacsim.physics.newton` and its tensor extension. |

For a backend-specific startup failure, exercise each production closure separately:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile physx_cuda --num-envs 2 --steps 2
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 2 --steps 2
```

The first command must select `linkerbot_sim.kaleidoscope.physx_cuda.python.kit`;
the second must select `linkerbot_sim.kaleidoscope.newton.python.kit`. Do not
infer one backend's status from the other backend's failure.

The explicit viewer selects the matching `physx_cuda_viewport` or
`newton_cuda_viewport` Kit. Do not assemble a viewport closure manually.

## Mirror Service

| Symptom | Check |
| --- | --- |
| No ready marker | Inspect the traceback before `MIRROR_INTERACTIVE_FAILED`; construction did not complete. |
| JSON rejected | Envelope must contain exactly protocol, request ID, operation, and arguments; reject duplicate keys and non-finite values. |
| Duplicate ID | Generate a new request ID; retained terminal IDs are intentionally not reusable. |
| Queue full | Reduce concurrent submissions or increase admission capacity at embedded construction after a memory/latency review. |
| Motion rejected after stop | Call `runtime.status`, then a successful `runtime.reset` to clear the emergency-stop latch. |
| Cancel appears late | Cancellation is cooperative; inspect the active operation and its execution boundary. |
| Response timeout | Do not assume the operation was rolled back. Query status with a new ID before retrying. |
| TCP/WebSocket host rejected | Use `localhost` or a numeric loopback address. |

## Mirror Motion

| Symptom | Check |
| --- | --- |
| Robot mismatch | Use the session-local `robot_id`; if supplied, `robot_label` must agree. |
| Joint target rejected | Use discovered names, finite values, and the correct group. |
| Task-space goal wrong | Confirm metres, `wxyz`, and the selected world/env/base/TCP frame. |
| Planner cannot find a path | Inspect collision freshness and approximation separately from physical contacts. |
| Timeline partially expected | Compilation is atomic; execution cancellation is not rollback. Capture/restore a snapshot when rollback is required. |

## Kaleidoscope Native API

| Symptom | Check |
| --- | --- |
| Tensor must live on CUDA | Construct the tensor directly on `env.device`; the API does not perform hidden copies. |
| Wrong selector | Use one-dimensional CUDA `int64`, unique IDs, and valid range. |
| Wrong action shape/dtype | Use `(env.num_envs, env.action_dim)` and CUDA `float32`. |
| Next step refuses | Reset every terminated/truncated row or use the skrl same-decision adapter. |
| State API poisoned | An engine writer failed. Stop using the runtime, close it, and create a new one. |
| IK action truncates rows | Inspect the dense failure mask and task policy; there is no avoidance fallback. |
| Rendering disabled | `make_torch_env()` is intentionally headless; construct with `make_viewport_env()` or Gymnasium `render_mode="human"`. |
| Blank or stale viewport | Verify `selected_env < num_envs` and call `env.render()` explicitly according to `render_every_n_steps`; `step()` never renders implicitly. |
| Camera/SyntheticData unavailable | Expected: the human viewport does not add camera observations, Replicator, SyntheticData, or recording. |

## Gymnasium And skrl

| Symptom | Check |
| --- | --- |
| Gymnasium dependency error | Install the `training` extra. |
| Poor Gymnasium throughput | Expected full NumPy transfer; use native Torch or skrl for device residency. |
| Unsupported autoreset | Select `disabled` or `same_step`. |
| skrl version/source error | The integration is pinned to 2.1.0; audit upstream changes before updating fingerprints. |
| skrl actions rejected | Keep CUDA `float32`, correct batch width, and the environment device. |

## Shutdown

If close reports live resources, preserve the process logs and resource names. Retry
the idempotent close after the worker finishes. Do not destroy the session manually or
force-close the stage underneath camera, planner, tensor-view, or output owners.
