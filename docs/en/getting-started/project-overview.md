# Project Overview

Language: [English](project-overview.md) | [中文](../../zh-CN/getting-started/project-overview.md)

The repository is organized around two products and one infrastructure boundary.
They share assets and pure domain types, but they do not share a runtime facade.

```text
clients and trainers
        |
        +----------------------+-----------------------+
        |                      |                       |
 Mirror JSON / Python    native Torch / skrl    Gymnasium adapter
        |                      |                       |
 linkerbot_sim.mirror    linkerbot_sim.kaleidoscope   NumPy boundary
        |                      |
        +---------- linkerbot_sim.isaac --------------+
                              |
                    PhysX CPU / PhysX CUDA /
                       Newton runtime
```

Dependencies point downward. Product code owns behavior; `linkerbot_sim.isaac` owns
Kit, stage, and concrete physics runtime construction. An `IsaacSession` is the only
owner allowed to close its app, stage, and physics runtime.

## Mirror

Mirror represents one physical workspace as one simulation world. A world may contain
many robots and objects; the product boundary is about world ownership, not robot
count.

Mirror provides:

- a strict versioned JSON envelope over stdin, loopback TCP JSONL, or loopback
  WebSocket;
- bounded admission, duplicate-request protection, cancellation, emergency stop,
  reset, status, and orderly shutdown;
- joint goals, deltas, sampled trajectories, synchronized timelines, IK, linear TCP
  paths, cuRobo planning, collision refresh, and avoidance;
- process-local state access and versioned scene snapshots;
- PhysX CPU or Newton physics selected at the cold configuration boundary;
- rendering, cameras, logging, telemetry, and persistent output.

All Isaac, USD, physics, camera, and planner work executes on the runtime owner thread.
Ingress threads parse and enqueue JSON only.

Mirror closes resources in dependency order: ingress, outputs/cameras/planner,
controllers/views, then `IsaacSession`. A child timeout preserves ownership and blocks
premature session destruction.

## Kaleidoscope

Kaleidoscope is a homogeneous vector environment for large-scale reinforcement
learning. One template scene is replicated into `N` environments and accessed through
fixed-shape CUDA tensor views.

Kaleidoscope provides:

- either PhysX CUDA/Fabric or the project's multi-world Newton runtime, with
  one canonical CUDA device selected by `mode.compute.cuda_device`;
- native Torch `reset`, `reset_idx`, and `step` methods;
- GPU-resident `get_state`, `set_state`, `snapshot`, `restore_snapshot`, and
  `clone_state` operations;
- fixed action variants for joint deltas, batched end-effector IK, and synchronous
  waypoint-linear end-effector motion;
- task-owned observations, rewards, termination, randomization, and reset buffers;
- a NumPy Gymnasium `VectorEnv` adapter at an explicit host-transfer boundary;
- a CUDA-native skrl adapter with same-decision autoreset and final-observation
  preservation.

Both Kaleidoscope training backends are headless and GPU-native. An explicit debug
entrypoint can show one selected environment for either backend. Kaleidoscope
intentionally has no asynchronous or batched trajectory planner, path search,
collision-avoidance model, camera, SyntheticData, Replicator, recording, transport
server, telemetry publisher, or playback queue.
Physical contact remains active for task dynamics; what is removed is the planning
collision/query world and avoidance service.

## Physics Backends

| Product | Engine | Execution | Important consequence |
| --- | --- | --- | --- |
| Mirror | PhysX | CPU | Isaac World-backed physics; root CUDA device remains available to RTX and cuRobo |
| Mirror | Newton | CPU | MuJoCo CPU integration; exactly one product world; root CUDA remains available to RTX/cuRobo |
| Mirror | Newton | CUDA | CUDA stream/graph integration; exactly one product world |
| Kaleidoscope | PhysX | CUDA | CUDA tensor pipeline and Fabric; headless vector execution |
| Kaleidoscope | Newton | CUDA | Project-owned Model/State/Control/Solver; one isolated world per environment |

Newton's infrastructure can manage more than one world. Mirror derives one
world, while Kaleidoscope derives its world count from the final `environments.num_envs`. Both
use the shared per-world physics leaf and the project's `NewtonRuntime`; neither
loads the Isaac Newton extension or its tensor extension.

## Formal Kit Entry Matrix

The strict composition factory selects exactly one formal Kit. Callers do not pass an
arbitrary experience path or assemble physics/render extensions themselves:

| Product | Engine / execution | Render closure | Kit selected by the factory |
| --- | --- | --- | --- |
| Mirror | PhysX / CPU | Controlled by the outputs profile | `apps/linkerbot_sim.mirror.physx.python.kit` |
| Mirror | Newton / CPU or CUDA | Disabled | `apps/linkerbot_sim.mirror.newton.python.kit` |
| Mirror | Newton / CPU or CUDA | Enabled | `apps/linkerbot_sim.mirror.newton_render.python.kit` |
| Kaleidoscope | PhysX / CUDA | Training headless | `apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` |
| Kaleidoscope | Newton / CUDA | Training headless | `apps/linkerbot_sim.kaleidoscope.newton.python.kit` |
| Kaleidoscope | PhysX / CUDA | Explicit single-environment viewport | `apps/linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit` |
| Kaleidoscope | Newton / CUDA | Explicit single-environment viewport | `apps/linkerbot_sim.kaleidoscope.newton_viewport.python.kit` |

Mirror PhysX uses one experience containing the RTX resources and lets the session
render specification decide whether to render. Mirror Newton has separate
physics-only and Newton-render closures. Kaleidoscope keeps separate physics-only
training Kits and selects a matching viewport Kit only through the explicit viewer.
That viewport displays `selected_env` and adds no camera, SyntheticData, or Replicator.
Public mode profiles state execution explicitly: Mirror provides
`physx_cpu/newton_cpu/newton_cuda`, while Kaleidoscope provides
`physx_cuda/newton_cuda`.

## Configuration Ownership

A mode root is a composition file, not a parameter dump:

```text
configs/modes/mirror/{physx_cpu,newton_cpu,newton_cuda}.yaml
  -> compute + scene selector mirror/scene3 + physics + control + cuRobo + planning + outputs

configs/modes/kaleidoscope/{physx_cuda,newton_cuda}.yaml
  -> compute + environments + scene selector kaleidoscope/tblock_push + physics + task
  -> optional cuRobo numerical profile for EE/linear actions only

configs/scenes/mirror/scene3.yaml
  -> scene.id: scene3

configs/scenes/kaleidoscope/tblock_push.yaml
  -> scene.id: tblock_push

configs/visualization/kaleidoscope.yaml
  -> launch-only selected environment + window/renderer + scene visuals
```

Leaf values have one writer. In particular, the CUDA index appears only at
`mode.compute.cuda_device`; physics, Torch, cuRobo, and training derive their device
from it. Kaleidoscope environment count and path naming appear only in the mode-root
`environments` mapping. The selected engine then derives its internal replication
mechanism: PhysX uses GridCloner with environment-ID isolation, while Newton creates
one Newton-runtime world per environment. These are implementation plans, not public
configuration selectors.

Scene selectors, file paths, and scene identities are deliberately distinct. A mode
stores a product-qualified selector, the catalog resolves it below the corresponding
`configs/scenes/<product>/` directory, and `scene.id` remains the unqualified file
basename. Flat selectors and cross-product references are rejected because Mirror and
Kaleidoscope scenes have incompatible schemas.

Mirror uses the same `control: mirror` for both engines; the selected physics engine
derives the default controller bundle. Kaleidoscope has no control profile or control
object. Planning and cuRobo are separate owners. `configs/planning/mirror.yaml` contains only
backend-neutral request defaults, while `configs/curobo/*.yaml` contains numerical
IK batch capacity plus MotionPlanner seed, CUDA graph, collision-capability, and cache
facts. MotionPlanner is fixed to one request; its cache capacity remains explicit.
The backend fixes its
validated cuRobo 0.8.0 task bundle and float32 dtypes. A Kaleidoscope task never
selects a backend: the mode root conditionally adds `profiles.curobo` for an EE/linear
action and must omit it for the joint-only `joint_control` and `joint_delta` actions.

## Public Boundaries

Use these stable facades:

- `linkerbot_sim.configuration`
- `linkerbot_sim.isaac`
- `linkerbot_sim.mirror`
- `linkerbot_sim.kaleidoscope`
- `linkerbot_sim.training.skrl`

The facades are lazy. Importing them does not start Kit or initialize CUDA. Internal
scene builders, tensor ports, timeline compilers, and backend managers may change
without a compatibility promise.

## Next Steps

- [Choose a mode and API](choose-runtime-and-api.md)
- [Mirror quickstart](mirror-quickstart.md)
- [Kaleidoscope quickstart](kaleidoscope-quickstart.md)
- [Configuration reference](../reference/configuration.md)
