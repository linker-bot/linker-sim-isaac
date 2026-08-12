# Runtime Constraints

Language: [English](constraints.md) | [中文](../../zh-CN/operations/constraints.md)

These constraints are product contracts, not temporary tuning advice.

## Shared Constraints

- Run from the checkout with Python 3.12 and the pinned dependency graph.
- Accept the Isaac EULA before process startup.
- Keep `usd-core` out of the Isaac environment so it cannot shadow Kit's `pxr`.
- Validate the complete mode graph before creating Kit, physics, or CUDA resources.
- An `IsaacSession` is the only owner allowed to close its app, stage, and concrete
  physics runtime.
- Lengths use metres, angles use radians, and public quaternions use `wxyz`.
- Joint column order comes from explicit names/discovery, never asset-file accident.

## Mirror Constraints

- One `MirrorRuntime` owns one reality-mapped world and one session. The world may
  contain several robots and objects.
- Supported physics variants are PhysX CPU, Newton CPU, and Newton CUDA. Both Newton
  compositions derive exactly one world; CPU physics still retains the root CUDA device
  for cuRobo and RTX.
- Isaac, USD, physics, camera, and planner access stays on the owner thread.
- Ingress threads may parse JSON, enqueue work, and wait for owned responses only.
- Admission is bounded and request IDs remain unique while retained in terminal
  history.
- Motion cancellation is cooperative and does not imply rollback.
- Emergency stop freezes idle physics and blocks new motion until reset.
- Runtime control mode switches apply to every robot and are legal only between complete
  motion requests. Mirror v1 has no mode operations; v2 effort motion is direct and
  profile-limited.
- A mode-switch rollback failure is permanent fail-stop state. Reset cannot clear it;
  close and recreate the runtime.
- Hybrid force/position control is limited to Mirror PhysX CPU at the configured
  minimum 200 Hz (the maintained profile uses 240 Hz), one selected robot per request,
  a physical TCP binding, and `reference_frame: world`. The selected arm uses explicit
  effort control on every joint; its position axes are explicit Cartesian impedance,
  not implicit per-axis joint drives. Hand position drives and other robots are not
  overridden.
- Hybrid gains may change only as a separate owner-queued operation between complete
  motions. Every motion freezes one `hybrid_parameter_generation`; `force_axes` is
  selected independently by each motion. Filter and safety limits are not wire-tunable.
- Hybrid motion requires a current tare generation. Reset invalidates tare. A controller
  restore failure is permanent fail-stop state and requests runtime shutdown.
- Output/camera/planner resources must stop before controllers/views and before the
  session. A timeout preserves the live owner.
- A Newton robot profile maps per-component `gravity: false` to body-frequency
  `mjc:gravcomp=1` before model finalization. Runtime per-link gravity switching is
  unsupported; changing that policy requires rebuilding the runtime.

## Kaleidoscope Constraints

- Accepted backends are PhysX CUDA/Fabric and the project's multi-world Newton
  runtime. Both training compositions are headless and GPU-native; either backend may
  use the explicit single-environment debug viewport.
- PhysX scene-query support is disabled; physical contact remains enabled. Newton
  uses one isolated world per environment.
- `mode.compute.cuda_device` is the only GPU-index fact.
- State, actions, observations, rewards, done flags, reset buffers, snapshots, and
  skrl rollout data remain on that CUDA device in the native path.
- Explicit selectors are unique, in-range CUDA `int64` tensors on the same device.
- Actions are finite CUDA `float32` with shape `(num_envs, action_dim)`.
- A fixed task profile selects one action variant and one physics tick count.
- The default training closure has no renderer. The explicit viewport closure still
  has no camera, SyntheticData, Replicator, recording, transport, telemetry, batch
  trajectory planner, planning collision world, avoidance service, or playback queue.
- End-effector and linear modes must select a kinematics-only `profiles.curobo` with
  collision checks disabled. The canonical profile omits
  `kinematics.collision_cache`; a retained valid cache is ignored, so no collision
  cache is allocated;
  joint-only `joint_control` and `joint_delta` modes must omit that reference and
  allocate no cuRobo context. Tasks do not select a numerical backend.
- The action variant and action shape are construction-time facts. Native mode switches
  affect all robots and environments, are legal only between complete decisions, and
  are rejected while either SAME_STEP phase is outstanding.
- Kaleidoscope starts in position mode. Gymnasium, skrl, and
  `KaleidoscopeTrainingPort` do not expose the setter.
- Snapshot restore never changes control mode. Schema 2 requires the same active mode;
  schema 1 is position-only. Generation is not restored.
- Native callers reset done rows before the next step. skrl uses the maintained
  same-decision handshake.
- An engine state-writer failure poisons the runtime; close and recreate it.
- Viewport configuration is a launch-only object outside episode fingerprints. Only
  `selected_env` is renderer-facing; training physics ticks stay `render=False`, and
  explicit `env.render()` must not advance simulation time.

## Gymnasium Boundary

Gymnasium accepts NumPy and therefore performs host/device transfer every step. It is
not a GPU-residency reference. Supported autoreset modes are `disabled` and
`same_step`; the deferred mode is rejected.

## GPU Capacity

PhysX CUDA buffer sizes and Newton per-world contact/Jacobian capacities are fixed
before runtime creation and cannot be assumed to grow safely. Newton derives
`world_count` from the final mode-root `environments.num_envs`. The memory budget applies to the
simulator process, not just Torch. Warm-up and steady-state gates must include the
selected backend, representative environment count, contact load, task buffers, and
any optional batch IK context.

The `physics.memory` mapping in `configs/physics/physx/cuda.yaml` is the complete
`GpuMemoryBudget`; all four fields are required:

| Field | Constraint |
| --- | --- |
| `max_simulator_process_mib` | Upper bound for memory attributed by NVML to the simulator PID, including Kit, PhysX, Torch, and native CUDA allocators |
| `min_free_floor_mib` | Absolute free-device-memory floor at prelaunch, post-warmup, and steady-state samples |
| `min_free_fraction_after_warmup` | Required free-device-memory fraction at post-warmup and both steady-state samples; range `(0, 1]` |
| `max_steady_growth_mib` | Maximum simulator-PID growth from the steady baseline to steady final; zero is valid |

This gate is specific to the Kaleidoscope `physx_cuda` profile; it does not replace Newton's
capacity smoke. Probe failure, an invisible PID, or a budget violation fails closed.
Run either the maintained recipe or its underlying entrypoint in an Isaac/CUDA
environment:

```bash
just smoke-kaleidoscope-memory
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_physx_gpu_memory_budget.py \
  --profile physx_cuda --num-envs 2 --warmup-steps 8 --steady-steps 16
```

`just test-simulation` also runs the seven formal Kit closures, all four Mirror profiles,
both Kaleidoscope backends, Newton capacity, and this memory gate. CPU `quality` does
not start it implicitly.

See [Troubleshooting](troubleshooting.md) and the
[Configuration Reference](../reference/configuration.md).
