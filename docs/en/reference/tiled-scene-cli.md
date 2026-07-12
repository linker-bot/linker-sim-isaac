# Tiled Scene CLI Reference

Language: [English](tiled-scene-cli.md) | [中文](../../zh-CN/reference/tiled-scene-cli.md)

The Tiled Scene entrypoint is `scripts/tiled_scene_interactive.py`. It creates the
independent `TiledSceneRuntime`; it does not create or wrap a
`SingleSceneRuntime`. The selected env profile owns clone count and topology, while
the runtime profile owns process, planner, transport, playback, telemetry, and
output policy.

For a first complete run, use the [Tiled Scene Quickstart](../getting-started/tiled-scene-quickstart.md).
For messages and selectors, use the [Tiled Scene JSON Reference](tiled-scene-json.md).

## Resolution Rules

Runtime values resolve from typed code defaults, then the selected runtime
YAML, then only the CLI fields explicitly supplied. Except for
`--runtime-profile` and `--dump-effective-config`, omitted parser values are
`None` and therefore do not override YAML.

Inspect the resolved values and field provenance without creating Isaac:

```bash
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene --dump-effective-config
```

## Complete Option Table

| Option | Argparse value when omitted | Bundled `default_tiled_scene` effective value | Contract |
|---|---|---|---|
| `-h`, `--help` | n/a | n/a | Print argparse help and exit before configuration or Isaac startup. |
| `--runtime-profile NAME` | `default_tiled_scene` | `default_tiled_scene` | Select `configs/runtime/NAME.yaml`; it must resolve with `mode: tiled_scene`. |
| `--dump-effective-config` | `false` | `false` | Print resolved config, field sources, and fingerprint, then exit before Isaac startup. |
| `--env NAME` | `None` | `scene3_tiled` | Override `runtime.profiles.env`; clone count and per-env structure still come from that env profile. |
| `--gui`, `--no-gui` | `None` | `false` | Explicitly enable or suppress the Isaac GUI. |
| `--default-decimation TICKS` | `None` | `2` | Positive physics-tick count used when a compatible action omits `decimation`. |
| `--planner-workers COUNT` | `None` | `2` | Positive async planner worker count. Each concurrent cuRobo worker owns separate GPU context/cache resources. |
| `--max-pending-requests COUNT` | `None` | `64` | Positive limit across queued and running async planner requests. |
| `--max-completed-results COUNT` | `None` | `256` | Completed planner-summary cache limit; `0` disables retention. |
| `--stdin`, `--no-stdin` | `None` | `true` | Explicitly enable or disable stdin JSONL. |
| `--stdin-eof-policy {exit,keep_alive}` | `None` | `exit` | Decide whether natural stdin EOF may request shutdown or keeps the process alive. |
| `--idle-physics-policy {pause,hold_step}` | `None` | `pause` | Pause while idle, or hold targets and continue stepping the shared World. |
| `--tcp-jsonl-host HOST` | `None` | `127.0.0.1` | Override the TCP bind host. A host-only override does not enable TCP. |
| `--tcp-jsonl-port PORT` | `None` | disabled (`null`) | Set the TCP JSONL port and enable that endpoint. |
| `--websocket-host HOST` | `None` | `127.0.0.1` | Override the WebSocket bind host. A host-only override does not enable WebSocket. |
| `--websocket-port PORT` | `None` | disabled (`null`) | Set the WebSocket port and enable that endpoint. |
| `--foxglove-live-host HOST` | `None` | `127.0.0.1` | Override the Foxglove live bind host. A host-only override does not enable live output. |
| `--foxglove-live-port PORT` | `None` | disabled (`null`) | Set the Foxglove live port and enable that endpoint. |
| `--foxglove-mcap-path PATH` | `None` | disabled (`null`) | Write Tiled Scene telemetry to this MCAP path, subject to output policy. |
| `--telemetry-env-ids IDS` | `None` | `0` | Comma-separated selected env IDs. Parsing and runtime resolution reject empty, duplicate, negative, or out-of-range selections. |
| `--telemetry-primary-env-id ID` | `None` | `0` | Env used for standard single-env topics; it must be included in selected env IDs. |
| `--telemetry-decimation TICKS` | `None` | `1` | Positive global-step interval between regular telemetry publications. |
| `--telemetry-rate-hz HZ` | `None` | `10.0` | Telemetry sampling rate. Exactly `0` disables Tiled Scene telemetry and opens no live or MCAP sink; negative values are rejected. |
| `--telemetry-full-batch-json`, `--no-telemetry-full-batch-json` | `None` | `true` | Include or omit selected-env JSON state. |
| `--telemetry-joint-states`, `--no-telemetry-joint-states` | `None` | `true` | Include or omit standard JointStates for the primary env. |

The paired Boolean forms write one field. If both are supplied, argparse keeps
the last occurrence; use only one form per invocation.

## Activation And Validation Relationships

A CLI listener port both sets the port and enables its endpoint. Changing only
the corresponding host keeps the YAML enablement unchanged. TCP, WebSocket,
and Foxglove live are separate services. Do not assign two services the same
host/port pair: config resolution does not preflight that collision, so the
later listener fails while binding. Distinct ports are the clearest setup.
Listener hosts are restricted to `localhost` or numeric loopback addresses.
Built-in services provide neither authentication nor TLS.

Telemetry selection is validated as one unit: `primary_env_id` must occur in
`selected_env_ids`, and every selected ID must be below the selected env's
`tiled.num_envs`. Overriding only one side can therefore make an otherwise valid
profile invalid. `--telemetry-rate-hz 0` suppresses telemetry sink creation even
if a live port or MCAP path is configured.

stdin lifetime and physics idling are independent. A long-running network
service can use `--no-stdin`; GUI, cameras, or continuously sampled telemetry
usually require `hold_step`, while a request-driven headless service can use
`pause`.

There are deliberately no CLI options for `num_envs`, robot selection, planner
backend, cuRobo profile, control mode, collision strategy, or planner batch
mode. Those values belong to env or runtime YAML. Tiled Scene control mode is
position-only. See [Configuration](configuration.md),
[Collision Models](../guides/collision-models.md), and
[Motion Planning](../guides/motion-planning.md).

## Startup And Exit Markers

Every live Isaac launch requires explicit EULA acceptance:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene --no-stdin --tcp-jsonl-port 8765
```

| Marker | Meaning |
|---|---|
| `TILED_SCENE_INTERACTIVE_CONFIG runtime_profile=<name> fingerprint=<hash>` | Configuration resolved; runtime creation is about to begin. |
| `TILED_SCENE_INTERACTIVE_READY` | Enabled network transports can enqueue requests; the configured stdin reader and main loop start immediately after this marker. |
| `TILED_SCENE_INTERACTIVE_EXIT` | The entrypoint reached its final shutdown block. Normal completion then exits with status 0; Tiled Scene has no separate success marker. |
| `TILED_SCENE_INTERACTIVE_FAILED <Exception>: <message>` | Startup or runtime escaped with an exception; the wrapper exits nonzero. |

Any `TILED_SCENE_INTERACTIVE_SHUTDOWN_TIMEOUT` line reports incomplete cleanup and
must be investigated even when `EXIT` follows. See
[Troubleshooting](../operations/troubleshooting.md).
