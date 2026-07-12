# Choose A Runtime And API

Language: [English](choose-runtime-and-api.md) | [中文](../../zh-CN/getting-started/choose-runtime-and-api.md)

Choose the runtime from the shape of the simulation, then choose JSON or Python
from the ownership boundary of the caller. Single Scene and Tiled Scene are both multi-robot
capable; the deciding factor is one shared world versus cloned env rows.

## Choose By Task

| Task | Runtime or tool | Interface | Reason |
| --- | --- | --- | --- |
| Coordinate one or more robots and objects in one physical world | Single Scene | Single Scene JSON | Shared integer-tick timeline applies every robot target before one world step |
| Run one robot or several robots without an env batch dimension | Single Scene | Single Scene JSON | Single Scene robot count is independent of runtime selection |
| Apply the same or row-specific command to selected cloned envs | Tiled Scene | Tiled Scene JSON | `env_ids` preserve an explicit batch row dimension |
| Run several robots inside every cloned env | Tiled Scene | Tiled Scene JSON | Env selectors and robot selectors address different dimensions |
| Execute fixed-duration batched end-effector motion | Tiled Scene | Tiled Scene `step` | Synchronous `ee_*` actions use the Tiled batch IK path |
| Submit planning without blocking physics and later play its trajectory | Tiled Scene | `plan`, `planner_status`, `step_trajectory` | Planner work and playback have separate bounded lifecycles |
| Execute a cross-robot sequence with explicit arm/hand group tracks | Single Scene | `plan_timeline` | Single Scene compiles all tracks onto one shared tick axis |
| Capture or restore persistent logical state | Single Scene or Tiled Scene | Snapshot JSON or `linkerbot_sim.snapshots` | Both adapters use the `linkerbot.snapshot` schema |
| Observe state, markers, or cameras | Single Scene or Tiled Scene | Foxglove live or MCAP | Telemetry observes runtime state and is not a control endpoint |
| Inspect config resolution without Isaac | Neither runtime | `validate_config.py` | Pure-Python graph validation resolves profiles and reports the effective runtime |
| Generate the bundled rope or T block USD asset | Offline tool | `build_asset.py` | Asset authoring is separate from simulation runtime construction |
| Embed runtime ownership in another in-process application | Single Scene or Tiled Scene | Python runtime facade | The caller accepts main-thread, startup, and shutdown responsibility |
| Use planning DTOs, snapshots, trajectories, or config parsing without a running world | Neither runtime | Pure-Python facade | These data/domain layers do not create Isaac by themselves |

Do not select Single Scene merely because an env contains one robot, and do not select
Tiled Scene merely because an env contains multiple robots. Select Tiled Scene when cloned
environment rows and `env_ids` are part of the problem.

## Choose JSON Or Python

| Caller boundary | Choose | Contract |
| --- | --- | --- |
| Another process, shell pipeline, test harness, or model-generated client | JSON | Strict message schema over stdin, TCP JSONL, or WebSocket |
| A service that should not own Isaac objects | JSON | Runtime process retains simulation-thread and shutdown ownership |
| In-process algorithm using validated planning/snapshot DTOs | Python facade | Caller imports an explicit public facade and handles returned errors/results |
| In-process application creating Single Scene or Tiled Scene runtime objects | Python runtime facade | Caller runs from the workspace, creates on the main thread, and closes every returned resource |
| One-off configuration inspection | `validate_config.py` | No Isaac import, GPU context, stage, or transport is created |

JSON is the canonical external control boundary. Python is an in-process
integration boundary, not an alternative installable SDK. The repository must
remain available because runtime profiles, assets, scripts, and task resources
are resolved from the checkout.

## JSON Control Transports

Single Scene and Tiled Scene expose their own message dialects through the same transport
families:

| Transport | Framing | Use |
| --- | --- | --- |
| stdin/stdout | One strict JSON object per line | Local process control and shell-driven automation |
| TCP JSONL | One strict JSON object per line, one direct response per request | Simple local clients with explicit request/response framing |
| WebSocket | One text JSON object per message | Browser or async clients and bounded event delivery |
| Foxglove live | Foxglove telemetry protocol | State/camera visualization only; it does not accept control JSON |

Binary WebSocket messages are rejected. Listener addresses must be loopback;
there is no authentication or TLS. Use an authenticated proxy or SSH tunnel for
remote access. Transport limits, connection admission, message size, and queue
overflow behavior come from the runtime profile.

Use the [Single Scene JSON reference](../reference/single-scene-json.md) for Single Scene commands and
the [Tiled Scene JSON reference](../reference/tiled-scene-json.md) for Tiled Scene commands. Do not
send a Single Scene message to a Tiled Scene handler or infer missing runtime selectors.

## Python Facade Boundary

The top-level `linkerbot_sim` package intentionally exports only `REPO_ROOT`.
Import from a documented package facade instead of relying on transitive
imports. Advanced owner-module symbols are integration points only when the
Python reference names the exact symbol and its lifecycle contract.

| Need | Documented package facade |
| --- | --- |
| Backend-neutral planning request/result types | `linkerbot_sim.planning` |
| cuRobo config, context, FK/IK, planning, and adapters | `linkerbot_sim.backends.curobo` |
| Canonical snapshot schema, compatibility, and runtime dispatch | `linkerbot_sim.snapshots` |
| Controller types and `JointController` | `linkerbot_sim.controllers` |
| Command execution steps | `linkerbot_sim.execution` |
| Object profile/runtime types | `linkerbot_sim.objects` |
| Robot capability and joint-group types | `linkerbot_sim.robots` |
| Sensor and camera configuration types | `linkerbot_sim.sensors` |
| Single Scene interactive entrypoint and loop | `linkerbot_sim.app.interactive.single_scene` |
| Tiled Scene runtime creation and message dispatch | `linkerbot_sim.app.interactive.tiled_scene` |

An exported facade does not remove lifecycle requirements:

1. Run from the checkout root with `PYTHONPATH=src`.
2. Set `OMNI_KIT_ACCEPT_EULA` only after accepting the NVIDIA/Kit EULA.
3. Resolve configuration before constructing a runtime.
4. Create and mutate Isaac runtime objects on the simulation main thread.
5. Pass only frozen Python/NumPy data to background workers.
6. Close transport/publishers and the runtime before releasing Kit.

Names beginning with `_`, test fakes, runtime service submodules, and helpers not
explicitly documented as integration points are implementation details. A package
`__all__` records exports but does not by itself create a support commitment. Pure
parsers may run without Isaac, but runtime handlers that mutate Single Scene or Tiled Scene state
must run on the simulation main thread.

The [Python Facade Reference](../reference/python-api.md) is the only complete
inventory of supported imports, exact symbols, signatures, and lifecycle
requirements. The [Motion Planning Guide](../guides/motion-planning.md) owns
planning behavior. The [Snapshot Reference](../reference/snapshots.md) owns the
shared snapshot payload, matching, and transaction contract; runtime references
own only their message envelopes and selectors.

## Validate Configuration

Use the validator before starting Isaac or after editing any referenced profile:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_single_scene

PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_tiled_scene
```

The validator traverses runtime, env, per-env fragments, robots, controller
bundles, objects, logging, and each enabled robot's merged cuRobo binding. It
does not launch `SimulationApp`. Inspect every resolved runtime leaf and its
source with:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_tiled_scene \
  --dump-effective-config
```

See the [configuration reference](../reference/configuration.md).

## Run The Interactive Entrypoints

After environment setup and EULA acceptance, start Single Scene with its Single Scene runtime
profile:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene
```

Start Tiled Scene only with a Tiled Scene runtime profile:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene
```

`--dump-effective-config` exits before Isaac starts. An explicit CLI value
overrides only its corresponding runtime field for that launch; omitted CLI
values preserve the selected runtime profile.

Follow the [Single Scene Quickstart](single-scene-quickstart.md) or
[Tiled Scene Quickstart](tiled-scene-quickstart.md) for a complete first client. Every
launch option is defined by the [Single Scene CLI](../reference/single-scene-cli.md) and
[Tiled Scene CLI](../reference/tiled-scene-cli.md) references.

## Generate Or Preview Assets

Runtime entrypoints consume existing assets and never rebuild them. Generate the
bundled assets through their offline entrypoints:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python \
  tools/object_assets/flexible/rope/build_asset.py

PYTHONPATH=src .venv/bin/python \
  tools/object_assets/rigid/tblock/build_asset.py
```

Both builders accept `--config` and `--output`, launch headless Isaac for USD and
PhysX schema authoring, save the asset, and close the app. Generated geometry is
owned by `tools/object_assets`; runtime placement and physics are owned by
`configs/objects` and `configs/envs`.

See [Object Assets](../development/object-assets.md),
[USD Preview](../development/usd-preview.md), and
[Collision Approximation](../development/collision-approximation.md).

## Decision Checklist

Before implementation, answer these questions in order:

1. Is there one physical world or a selected batch of cloned env rows?
2. Does the caller need process isolation, or will it own Isaac in-process?
3. Which profile owns every setting being changed?
4. Which robot and env selectors are required by the chosen message?
5. Does the request require direct control, IK, joint planning, or collision-aware planning?
6. Which bounded queue, file policy, and shutdown timeout apply to its outputs?
7. What terminal response proves completion, and what response requires runtime recreation?

Then follow the exact runtime reference instead of deriving a message from the
other runtime's examples.

## Continue Reading

- [Project Overview](project-overview.md)
- [Single Scene Quickstart](single-scene-quickstart.md)
- [Tiled Scene Quickstart](tiled-scene-quickstart.md)
- [Configuration Guide](../guides/configuration.md)
- [Single Scene CLI Reference](../reference/single-scene-cli.md)
- [Single Scene JSON Reference](../reference/single-scene-json.md)
- [Tiled Scene CLI Reference](../reference/tiled-scene-cli.md)
- [Tiled Scene JSON Reference](../reference/tiled-scene-json.md)
- [Python Facade Reference](../reference/python-api.md)
- [Motion Planning](../guides/motion-planning.md)
- [Telemetry](../guides/telemetry.md)
- [Known Constraints](../operations/constraints.md)
