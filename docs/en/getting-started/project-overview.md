# Project Overview

Language: [English](project-overview.md) | [中文](../../zh-CN/getting-started/project-overview.md)

LinkerHand Simulation is a checkout-based Isaac Sim application for robot control,
motion planning, parallel environment execution, state capture, and sensor output.
It provides two runtime models with different execution contracts:

- Single Scene mode operates one physical world and can coordinate any number of robots
  declared by the selected env profile.
- Tiled Scene mode builds cloned environments and applies explicitly selected batched
  operations to their robot and object rows.

Both modes support strict YAML configuration, robot-ID discovery, JSON control,
cuRobo integration, canonical snapshots, Foxglove/MCAP telemetry, and sensor
cameras. Single Scene mode additionally provides shared-tick multi-robot timelines and
per-robot joint-tracking CSV output. Tiled Scene mode additionally provides batched
step actions, per-env state operations, trajectory buffers, and asynchronous
planning with bounded scheduling resources.

Use [Choose A Runtime And API](choose-runtime-and-api.md) before selecting an
entrypoint or Python integration surface.

## Execution Models

| Boundary | Single Scene | Tiled Scene |
| --- | --- | --- |
| Runtime owner | `SingleSceneRuntime` | `TiledSceneRuntime` |
| Topology | One World containing the configured robot and object instances | One source env cloned into `tiled.num_envs` rows |
| Robot count | One or more robots in the same scene | One or more robots in every cloned env |
| Selection | Session `robot_id` | Explicit `env_ids` plus session `robot_id` or `robot_ids` where required |
| Synchronous execution | Integer-tick robot timelines with one shared `world.step()` per tick | Batched fixed-tick `step` actions and trajectory-buffer playback |
| Planning | Single Scene compiler using `curobo` or joint-space `linear` backend | Synchronous batched IK for end-effector actions and a separate asynchronous planner manager |
| State operations | Single Scene reset and scene snapshot capture/restore | Per-env reset, debug state, snapshot broadcast, and env-to-env clone |
| Joint CSV | Supported through the selected logging profile | Not created by the Tiled Scene entrypoint |

Single Scene does not mean single robot. An env profile selected by Single Scene may contain several robot
instances, and one timeline can coordinate their tracks from a common tick zero.
Single Scene has no cloned `env_id` dimension.

Tiled Scene is not a wrapper around `SingleSceneRuntime`. It has a separate runtime class,
scene builder, command adapter, state shape, trajectory buffer, and planner
manager. A Tiled env may still contain multiple robots; `env_ids` select cloned
rows and robot IDs select robots within those rows.

## Shared Boundaries

The runtimes deliberately share domain contracts where the data has the same
meaning:

- Runtime profiles own process policy, selected domain profiles, resource
  limits, telemetry, output lifecycle, and shutdown timeouts.
- Env, robot, controller, object, cuRobo, and logging profiles use strict
  repository schemas and are validated before Isaac starts.
- Robots use session-local numeric IDs for control and stable labels/profile
  fingerprints for persistent matching.
- Planning requests and results use backend-neutral DTOs before a concrete
  `linear` or cuRobo adapter executes them.
- `linkerbot.snapshot` is the canonical logical snapshot schema used by Single Scene
  and Tiled Scene adapters.
- Coordinate values use meters and radians; public quaternions use `wxyz`.
- State telemetry, camera output, and file targets use bounded queues and
  explicit output policies.

Sharing these contracts does not make runtime-specific JSON interchangeable.
Always use the message schema for the selected runtime.

## Separate Boundaries

The following are intentionally runtime-specific:

- Single Scene timeline messages and Tiled Scene action messages have different envelopes,
  selectors, response fields, and state machines.
- Single Scene state describes one world. Tiled Scene state retains an explicit selected-env
  row dimension.
- Single Scene command status is tracked by command ID. Tiled Scene trajectory playback and
  asynchronous planning have independent lifecycle APIs.
- Scene collision coordination can freeze other robots as static planning
  obstacles. Tiled Scene planning treats selected env rows as request problems and
  does not expose the Single Scene coordination fields.
- Tiled Scene articulation control is position-based. A runtime profile requesting
  Tiled Scene velocity or effort control is rejected during configuration resolution.
- Single Scene and Tiled Scene own different Isaac objects and must each be closed through
  their own lifecycle facade.

See the [Single Scene JSON reference](../reference/single-scene-json.md) and
[Tiled Scene JSON reference](../reference/tiled-scene-json.md) for exact messages.

## Workspace Requirements

The repository is a workspace application, not an independently installable
Python library. It depends at runtime on checkout-local `configs/`, `assets/`,
`scripts/`, and cuRobo task resources. Wheel, source-distribution, and editable
builds are rejected by the local build backend.

The declared environment is Linux x86-64 with Python 3.11. Create the unified
environment from the checkout root:

```bash
uv sync --all-extras
```

Run project commands from that same root with `PYTHONPATH=src`. Before starting
Isaac Sim, read and accept the applicable NVIDIA/Kit EULA, then record that
acceptance in the deployment environment:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

The application accepts `Y`, `YES`, or `1`, case-insensitively. It never sets
this variable or accepts the EULA for the user. Missing acceptance fails before
`SimulationApp` is created.

## Configuration Ownership

| Location | Sole responsibility |
| --- | --- |
| `configs/runtime/` | Runtime mode, selected profiles, GUI/GPU/render settings, execution defaults, transports, planner/playback limits, telemetry, camera output policy, paths, and shutdown |
| `configs/envs/` | World settings, visual scene, sensor placement, robot/object instances, and optional Tiled topology/per-env pose overrides |
| `configs/robots/` | Isaac model, robot kind, joint groups, importer/PhysX settings, and optional cuRobo model/TCP binding |
| `configs/controllers/` | Position, velocity, and effort control methods, joints, gains, limits, and PhysX drive overrides |
| `configs/objects/` | Object asset, object kind, runtime physics/material, state summary, and optional simplified planning collision |
| `configs/curobo/` | Device, task bundle, IK/planner algorithms, seeds, tolerances, collision cache, and batch capacity |
| `configs/logging/` | Single Scene joint-tracking CSV enablement, path, sampling, flush interval, and columns |
| `tools/object_assets/` | Offline generated-asset geometry and authored USD/PhysX properties |

Runtime resolution order is code defaults, the selected runtime YAML, then only
the CLI fields explicitly supplied for that launch. Env profiles own scene facts;
they do not own planner, transport, telemetry, or process resource policy. Use
the [configuration guide](../guides/configuration.md) for ownership examples and
the [configuration reference](../reference/configuration.md) for validation
behavior.

## Operational Boundaries

### Network And Input

stdin, TCP JSONL, and WebSocket control paths accept strict JSON objects. Unknown
fields, duplicate YAML keys, non-finite JSON constants, invalid selectors, and
out-of-range values are rejected instead of guessed.

Every built-in control, state telemetry, and camera live listener is restricted
to loopback. The application provides no authentication or TLS. Remote access
requires an authenticated TLS proxy or SSH tunnel whose upstream remains a
loopback endpoint. Foxglove live is telemetry, not a JSON control transport.

### Thread Ownership

Isaac stage objects, articulation/PhysX views, Camera wrappers, and runtime
mutation are owned by the simulation main thread. Transport and planner workers
may parse messages or consume frozen NumPy/Python snapshots, but they must not
read or write Isaac objects. File and telemetry publishers receive already
captured immutable data.

### Resource And Output Limits

Transport connections, request/event queues, snapshot requests, planner work,
completed planner summaries, trajectory buffers, telemetry buffers, camera
queues, and camera directory bytes all have explicit bounds. Overflow behavior
is configured as rejection, backpressure, replacement, or a declared drop
policy; it is never an unbounded queue.

CSV, MCAP, and camera targets are planned and checked together before any writer
opens. Existing data requires an explicit `error`, `truncate`, `resume`, or
`timestamped_dir` policy, and a sink may reject a policy it cannot implement
safely. See [Telemetry](../guides/telemetry.md) and
[Cameras](../guides/cameras.md) for task workflows, and the
[Output Reference](../reference/outputs.md) for exact file and payload contracts.

### Mutation And Fail-Stop

Reset, Tiled Scene `set_state`, snapshot restore, and env cloning validate and capture
rollback state before their first mutation. Completed writes are compensated in
reverse order when a later setter fails. A complete rollback leaves the runtime
usable.

If rollback fails, or an error occurs after an irreversible queue/cache commit,
the runtime records its first fatal reason, requests shutdown, and rejects later
mutations. Recreate the runtime instead of continuing from state whose
controller/PhysX consistency cannot be proven.

### Shutdown

Entrypoints first stop new transport and publisher admission, then perform
bounded waits for background work, close planners/cameras/loggers and their
dependent sinks, release IK/planning resources, and finally close
`SimulationApp`. Independent timeout settings prevent one resource class from
consuming another class's shutdown budget. A worker that remains alive retains
its sink or runtime dependency so it is not closed concurrently; the owning
runtime can retry cleanup before releasing Kit.

The detailed invariants are collected in
[Known Constraints](../operations/constraints.md).

## Continue Reading

- [Choose A Runtime And API](choose-runtime-and-api.md)
- [Single Scene Quickstart](single-scene-quickstart.md)
- [Tiled Scene Quickstart](tiled-scene-quickstart.md)
- [Configuration Guide](../guides/configuration.md)
- [Single Scene CLI Reference](../reference/single-scene-cli.md)
- [Single Scene JSON Reference](../reference/single-scene-json.md)
- [Tiled Scene CLI Reference](../reference/tiled-scene-cli.md)
- [Tiled Scene JSON Reference](../reference/tiled-scene-json.md)
- [Control And Trajectories](../guides/control-and-trajectories.md)
- [Motion Planning](../guides/motion-planning.md)
- [Collision Models](../guides/collision-models.md)
- [Snapshot Data And Restore](../reference/snapshots.md)
- [Telemetry](../guides/telemetry.md)
- [Cameras](../guides/cameras.md)
- [Persistent And Live Outputs](../reference/outputs.md)
- [Troubleshooting](../operations/troubleshooting.md)
- [Object Assets](../development/object-assets.md)
