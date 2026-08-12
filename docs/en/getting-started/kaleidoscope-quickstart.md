# Kaleidoscope Quickstart

Language: [English](kaleidoscope-quickstart.md) | [中文](../../zh-CN/getting-started/kaleidoscope-quickstart.md)

Kaleidoscope is a GPU-native vector environment. Training entrypoints remain
headless; an explicit debug entrypoint can open a single-environment viewport for
either PhysX CUDA/Fabric or the project's multi-world Newton owner. Native
Torch is the fastest interface; Gymnasium is available when NumPy compatibility is
more important than device residency.

## 1. Prepare The Environment

```bash
uv sync --extra simulation --extra training
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH=src
```

The selected GPU must satisfy the configured PhysX buffer and process-memory gates.

## 2. Validate The Composition

```bash
.venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile physx_cuda
.venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile newton_cuda
```

Both graphs select the same T-block task, action contract, and position target space.
Their backend-specific composition and physics-derived drive bundles are:

| Profile | Physics | Internally derived environment realization | Headless Kit selected by training |
| --- | --- | --- | --- |
| `physx_cuda` | `physx/cuda` | GridCloner, 3.0 m spacing, environment-ID isolation | `apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` |
| `newton_cuda` | `newton/cuda` | Multi-world, zero spacing, separate worlds | `apps/linkerbot_sim.kaleidoscope.newton.python.kit` |

The mode root has exactly the required `scene`, `physics`, and `task` profile slots; it
has no `profiles.control`. It owns `environments.num_envs`, `base_env_path`, `env_prefix`, and
`origin_xyz`; there is no public replication profile. The only CUDA index is
`compute.cuda_device` in the mode root. The Newton Kit imports
the project's Newton/MuJoCo-Warp Python runtime directly and owns its worlds; it does
not load the Isaac Newton extension. Its `world_count` must equal the effective
`num_envs` and is derived from it, with one isolated world per environment.
`physics.engine` derives the `newton` controller bundle instead of PhysX gains.

The bundled roots use `joint_control` and therefore omit `profiles.curobo`. A custom
end-effector or linear task still owns only action semantics; its matching mode root
must add `profiles.curobo: kaleidoscope_batch_ik`. That numerical profile is
kinematics-only, has collision checks disabled, and cannot contain `motion_planner`.
Its canonical YAML omits `collision_cache`; validation also accepts a valid retained
cache, but runtime discards it and allocates no collision cache. Supplying cuRobo for
`joint_control`/`joint_delta`, or omitting it for an EE/linear action, fails before Kit
starts.

## 3. Run A Native Torch Step

Create `scratch_kaleidoscope.py` outside maintained source packages, or run this from
an interactive Python session in the checkout. The factory starts the Kit selected by
the profile; do not prelaunch a second experience:

```python
import torch

from linkerbot_sim.kaleidoscope import make_torch_env

env = make_torch_env(profile="physx_cuda", num_envs=64)
try:
    observations, info = env.reset()
    assert observations.is_cuda
    assert observations.shape == (env.num_envs, env.observation_dim)

    actions = torch.zeros(
        (env.num_envs, env.action_dim),
        device=env.device,
        dtype=torch.float32,
    )
    observations, rewards, terminated, truncated, info = env.step(actions)

    done_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
    if done_ids.numel() > 0:
        env.reset_idx(done_ids)
finally:
    env.close()
```

The native `step` contract requires every done row to be reset first. The precondition
is checked by one synchronous scalar readback so a recoverable lifecycle error is
raised before physics advances. The skrl same-decision adapter uses the tokenized
training entry and does not execute this native/debug guard, so its rollout path stays
on CUDA. Change the example to `profile="newton_cuda"` to select Newton. The Torch API,
action/observation shapes, and state contract do not change.

## 4. Exercise GPU State And Cloning

```python
env = make_torch_env(profile="physx_cuda", num_envs=4)
try:
    env.reset()
    source = torch.tensor([0, 1], device=env.device, dtype=torch.int64)
    target = torch.tensor([2, 3], device=env.device, dtype=torch.int64)

    state = env.get_state(source)
    snapshot = env.snapshot(source)
    env.clone_state(source, target, include_rng=True)
    env.restore_snapshot(snapshot, target_env_ids=target)
finally:
    env.close()
```

Selectors and payloads must be CUDA tensors on `env.device`. A snapshot owns cloned
GPU storage; it is not a serialized checkpoint. Source and target selectors for
`clone_state` must have equal lengths and may not overlap. State capture, restore,
snapshot, and cloning remain on device for both physics backends.

## 5. Use Gymnasium When Required

```python
import numpy as np

from linkerbot_sim.kaleidoscope import make_gymnasium_env

env = make_gymnasium_env(
    profile="physx_cuda", num_envs=64, autoreset_mode="same_step"
)
try:
    observations, info = env.reset(seed=7)
    actions = np.zeros((env.num_envs, env.env.action_dim), dtype=np.float32)
    observations, rewards, terminated, truncated, info = env.step(actions)
finally:
    env.close()
```

This adapter performs a full host/device transfer every step. It is a compatibility
surface, not the performance reference.

## 6. Use skrl Without Leaving CUDA

```python
from linkerbot_sim.kaleidoscope import make_torch_env
from linkerbot_sim.training.skrl import SkrlTorchAdapter

native = make_torch_env(profile="physx_cuda", num_envs=256)
env = SkrlTorchAdapter(native)
```

Pair this adapter with `CudaRolloutMemory` and `FinalObservationPPO`. The integration
is pinned to skrl 2.1.0 and validates the upstream methods it replaces before
training.

## 7. Open An Explicit Single-Environment Viewport

The maintained viewer uses the same interface for both physics backends:

```bash
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py \
  --profile physx_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py \
  --profile newton_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
```

The viewer calls `make_viewport_env()` and loads
`configs/visualization/kaleidoscope.yaml` independently from the training
graph. It selects `linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit` or
`linkerbot_sim.kaleidoscope.newton_viewport.python.kit`. Only `selected_env`
is materialized for renderer-facing USD; the other environments continue on GPU
without per-world RTX display state.

This launch-only configuration does not enter the task/physics graph or episode
snapshot/clone fingerprint. Every training decision still advances physics with
`step(render=False)`; the viewer explicitly calls `env.render()` according to
`render_every_n_steps`. This is a human viewport, not a camera observation pipeline:
it adds no SyntheticData, Replicator, recording, or image tensors. Gymnasium callers
may also select `render_mode="human"`, while retaining Gymnasium's NumPy transfer
boundary.

## 8. Run Real-Physics and Action Smokes

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile physx_cuda --num-envs 2 --steps 2
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 2 --steps 2 \
  --exercise-training-adapters
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 2 --steps 1 \
  --action-mode ee_delta_position
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 2 --steps 1 \
  --action-mode ee_linear_path_position
```

The smoke enters the production composition root, selects the corresponding Kit, and
checks reset/step, CUDA residency, snapshot restore, and row-to-row `clone_state`. It
also exercises Gymnasium/skrl SAME_STEP and the real non-collision cuRobo batch-IK and
fixed-waypoint linear action paths by constructing a temporary mode composition with
`profiles.curobo: kaleidoscope_batch_ik`; the task itself never selects that backend.
`just smoke-kaleidoscope` runs this same matrix.

Validate the formal 256-world Newton multi-world capacity separately:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 256 --steps 2
```

The probe also places the T-block at each world's TCP and requires live contact world
IDs to cover every environment without any cross-world contact. These are correctness
and capacity gates, not claims about unmeasured throughput or peak GPU memory. The capacity command is also available as
`just smoke-kaleidoscope-newton-capacity`.

## 9. Enforce The PhysX GPU Memory Budget

PhysX engine capacities are not a project configuration surface.
`configs/physics/physx/cuda.yaml` declares only the complete process-level
`GpuMemoryBudget`:

| Field | Gate |
| --- | --- |
| `max_simulator_process_mib` | NVML memory ceiling for the simulator PID |
| `min_free_floor_mib` | Absolute free-MiB floor at all four samples |
| `min_free_fraction_after_warmup` | Free-device-memory fraction after warmup and at both steady samples |
| `max_steady_growth_mib` | Maximum PID-memory growth from steady baseline to steady final |

Run the maintained recipe or the equivalent script on the target GPU:

```bash
just smoke-kaleidoscope-memory
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_physx_gpu_memory_budget.py \
  --profile physx_cuda --num-envs 2 --warmup-steps 8 --steady-steps 16
```

The script samples prelaunch, post-warmup, steady baseline, and steady final. Success
prints `LINKERBOT_PHYSX_GPU_MEMORY_BUDGET_OK`. It accounts for the complete simulator
PID rather than only the Torch allocator, accepts only the `physx_cuda` profile, and does not
replace the 256-world Newton capacity smoke.

Run `just test-simulation` for the complete simulation gate. It includes
`just smoke-runtime-kits` for all seven formal Kit closures, `just smoke-mirror` for
all four Mirror profiles, both Kaleidoscope backends, Newton capacity, and this memory
gate.

## Deliberate Omissions

Do not look for a Kaleidoscope JSON server, camera switch, planner backend, collision
avoidance toggle, playback queue, telemetry publisher, SyntheticData, Replicator, or
recording pipeline. Those capabilities are outside this product by design. The
explicit human viewport only displays one selected environment. Both physics backends
still resolve task contacts; neither constructs planning-collision or avoidance
resources.

Continue with the [Kaleidoscope API](../reference/kaleidoscope-api.md) and
[Configuration Reference](../reference/configuration.md).
