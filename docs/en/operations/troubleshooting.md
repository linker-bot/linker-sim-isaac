# Troubleshooting

Language: [English](troubleshooting.md) | [中文](../../zh-CN/operations/troubleshooting.md)

Use this page to locate a failure domain. Exact fields and error responses remain owned by
the linked reference pages.

## Before Isaac Starts

| Symptom | Check |
| --- | --- |
| `OMNI_KIT_ACCEPT_EULA=Y` error | Read and accept the applicable NVIDIA/Kit EULA, then set `Y`, `YES`, or `1` without surrounding whitespace |
| Import or resource path fails | Run from the checkout root with `PYTHONPATH=src`; do not install only `src/` |
| Profile or unknown-field error | Run `scripts/validate_config.py --runtime-profile <name>` and inspect the full field path |
| Runtime mode mismatch | Use a `single_scene` runtime profile with the Single Scene entrypoint and a `tiled_scene` runtime profile with the Tiled Scene entrypoint |
| `--dump-effective-config` exits | This is expected: it resolves, prints, and exits before Isaac startup |
| Wheel/editable build is rejected | This project is intentionally a checkout workspace application |

Start with [Project Overview](../getting-started/project-overview.md) and
[Configuration](../guides/configuration.md).

## Process Starts But Is Not Ready

`SINGLE_SCENE_INTERACTIVE_READY` or `TILED_SCENE_INTERACTIVE_READY` means command transports can
accept requests. Before that marker, inspect the first startup exception rather than sending
control messages.

- Asset/import errors: verify every selected env, robot, object, controller, and cuRobo path.
- CUDA/cuRobo errors: verify the locked environment, GPU availability, and robot planning binding.
- Camera startup errors: distinguish a fatal exception from the known RTX/Fabric warnings in
  the [Camera Guide](../guides/cameras.md).
- Port errors: TCP, WebSocket, state Foxglove, and camera Foxglove endpoints need distinct
  available ports.

## Transport And Lifetime

| Symptom | Likely cause |
| --- | --- |
| Non-loopback host is rejected | Built-in listeners are loopback-only and have no authentication/TLS |
| TCP request has no delimiter | TCP JSONL requires one complete JSON object followed by `\n` |
| WebSocket binary frame is rejected | Only text JSON messages are accepted |
| JSON is rejected before dispatch | Check UTF-8, duplicate keys, trailing data, `NaN`/infinity, message size, and required object shape |
| New connection is refused | Single Scene TCP and WebSocket share one bounded connection quota |
| Process exits on stdin EOF | Select the documented EOF policy or keep an enabled service/output consumer alive |
| GUI/camera/telemetry stops refreshing while idle | Use `hold_step`; `pause` deliberately stops idle World stepping |

Do not expose a listener directly to an untrusted network. Use an authenticated TLS proxy or
SSH tunnel whose upstream remains on loopback.

## Single Scene Commands

- Discover the current `robot_id`, label, profile, joint groups, and capability flags with
  `status`; IDs are session-local.
- A `rejected` response means no command entered the queue.
- `accepted` proves queue admission, not completion. Poll `status` for `done`, `failed`, or
  `cancelled`, or consume WebSocket lifecycle events.
- If one segment fails during planning, the complete timeline fails before execution.
- Separate JSONL commands do not provide same-tick coordination; use `plan_timeline`.

See [Single Scene JSON](../reference/single-scene-json.md) and
[Control And Trajectories](../guides/control-and-trajectories.md).

## Tiled Scene Commands And Selectors

- Every env-scoped command requires explicit nonempty unique `env_ids`.
- Multi-robot commands require the robot selector defined for that message.
- `values` row count must be one or `len(env_ids)`; widths follow command-space or action rules.
- `get_state` is a transient in-process debugging shape; use Snapshot for persistent restore.
- Unselected envs still advance when the shared World steps, while holding their targets.

Inspect `status` for `num_envs`, robot command joints, env origins, queue resources, telemetry,
planner, and camera diagnostics. See [Tiled Scene JSON](../reference/tiled-scene-json.md).

## Planning And IK

| Symptom | Check |
| --- | --- |
| Task-space request is unsupported | Robot cuRobo binding, TCP frame, and request kind |
| `avoid_collisions=true` fails | Complete robot collision model, planning world, cache, and backend capability |
| Linear backend rejects a request | It supports joint interpolation only, not IK or collision avoidance |
| Batch is too large | Runtime `max_batch_problems` and `oversize_request_policy` |
| Planning request remains queued | Call `planner_status` or `step_trajectory` to dispatch/collect Tiled Scene work |
| Planner succeeds but trajectory is absent | Inspect `loaded` and `load_rejected` for playback admission failure |
| First request is slow | cuRobo context creation, kernel compilation, and warmup occur lazily per resource owner |

Use [Motion Planning](../guides/motion-planning.md) and
[Collision Models](../guides/collision-models.md).

## Trajectory Playback

- Verify finite strictly increasing `times`, position shape, and `joint_names` order.
- Check per-env queue-depth, sample-count, and duration limits in `trajectory_status`.
- Append accounts for existing plus new playback; replace validates the new sequence.
- Playback advances only through `step_trajectory`; loading does not step physics.
- The sparse `hand` path is not a synchronized Single Scene-style arm/hand timeline.

## Snapshots And State Mutation

- Use stable label/profile/fingerprint matching, not a cached session robot ID.
- `strict=true` requires equal joint/body-name sets for every mapped entry; extra target
  robots or objects remain unchanged. Use `label_map` only for an intentional rename.
- A pending snapshot request can time out before execution. An executing request waits for a
  real result except when shutdown returns the explicit running state.
- A rollback failure or failure after an irreversible commit puts the runtime in fail-stop.
  Recreate it; do not retry mutations on unprovable state.

See [Snapshot Reference](../reference/snapshots.md).

## Telemetry, Cameras, And Files

| Symptom | Check |
| --- | --- |
| No telemetry sink opens | Effective rate must be greater than zero and a live port or MCAP path must be configured |
| Foxglove connects to the wrong service | Control TCP/WebSocket and Foxglove use different ports and protocols |
| Joint effort is empty | Enable effort sampling and select the intended effort field |
| Segmentation image is absent in Foxglove | Segmentation is stored locally as `.npy`; RawImage channels are RGB/depth only |
| Output target already exists | Select an allowed explicit existing-data policy before startup |
| Camera publisher stops at quota | Increase the planned per-camera budget or start with a new target; inspect orphan cleanup status |
| Persistent output drops frames | Persistent sinks reject lossy queue policies; use bounded blocking or fail-fast behavior |

Use [Telemetry](../guides/telemetry.md), [Cameras](../guides/cameras.md), and
[Output Reference](../reference/outputs.md).

## Asset Tools

The rope and T block builders require EULA acceptance and launch a headless SimulationApp.
Run their `build_asset.py`, not the library `builder.py`, from the checkout root. If runtime
still loads old geometry, verify that `--output` matches the object profile's `asset_path` and
preview the generated USD before launching the scene.

See [Object Assets](../development/object-assets.md) and
[USD Preview](../development/usd-preview.md).

## Shutdown

A `*_SHUTDOWN_TIMEOUT` marker means shutdown is incomplete even if an outer script prints its
final step count. The owner retains timed-out resources for retry and must not close Kit below
a live child. Inspect transport, state publisher, camera publisher, planner, and runtime status
separately; they use independent timeout budgets.

For invariants and recovery boundaries, see [Known Constraints](constraints.md).
