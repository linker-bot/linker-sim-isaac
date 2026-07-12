# Single Scene CLI Reference

Language: [English](single-scene-cli.md) | [中文](../../zh-CN/reference/single-scene-cli.md)

The Single Scene entrypoint is `scripts/single_scene_interactive.py`. It resolves a
Single Scene runtime profile, applies only explicitly supplied CLI overrides, creates
one `SingleSceneRuntime`, and serves the Single Scene JSON protocol. A Single Scene runtime can contain
any number of robots declared by its env profile.

For a first complete run, use the [Single Scene Quickstart](../getting-started/single-scene-quickstart.md).
For request and response fields, use the [Single Scene JSON Reference](single-scene-json.md).

## Resolution Rules

Runtime values resolve in this order:

1. Typed code defaults.
2. The YAML selected by `--runtime-profile`.
3. CLI fields explicitly present on this invocation.

Except for `--runtime-profile` and `--dump-effective-config`, the parser leaves
an omitted option as `None`. `None` means "do not override the selected YAML";
it is not the effective runtime value. The table therefore shows both the raw
argparse result and the effective value in the bundled `default_single_scene` profile.

Use the non-Isaac dry run to inspect the exact resolved values and their source:

```bash
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene --dump-effective-config
```

This command resolves the selected runtime and env configuration, prints JSON,
and exits before creating `SimulationApp`.

## Complete Option Table

| Option | Argparse value when omitted | Bundled `default_single_scene` effective value | Contract |
|---|---|---|---|
| `-h`, `--help` | n/a | n/a | Print argparse help and exit before configuration or Isaac startup. |
| `--runtime-profile NAME` | `default_single_scene` | `default_single_scene` | Select `configs/runtime/NAME.yaml`; the profile must resolve with `mode: single_scene`. |
| `--dump-effective-config` | `false` | `false` | Print the resolved config, field sources, and fingerprint, then exit before Isaac startup. |
| `--env NAME` | `None` | `scene1` | Override `runtime.profiles.env` with a profile under `configs/envs/`. |
| `--curobo-profile NAME` | `None` | `default` | Override `runtime.profiles.curobo`. It is consumed when cuRobo capabilities are created. |
| `--planner-backend {curobo,linear}` | `None` | `curobo` | Select cuRobo planning or executable joint-space interpolation. `linear` does not provide IK or collision checking. |
| `--logging-profile NAME` | `None` | `default_logger` | Select the Single Scene joint CSV profile under `configs/logging/`. |
| `--control-mode MODE` | `None` | `position` | Override articulation control with `position`, `velocity`, or `effort`; runtime validation rejects unsupported controller combinations. |
| `--gui`, `--no-gui` | `None` | `false` | Explicitly enable or suppress the Isaac GUI. |
| `--stdin-eof-policy {exit,keep_alive}` | `None` | `exit` | Decide whether natural stdin EOF may request shutdown or keeps the process alive. |
| `--idle-physics-policy {pause,hold_step}` | `None` | `hold_step` | Pause while idle, or hold current targets and continue stepping the World. |
| `--tcp-jsonl-host HOST` | `None` | `127.0.0.1` | Override the TCP bind host. A host-only override does not enable TCP. |
| `--tcp-jsonl-port PORT` | `None` | disabled (`null`) | Set the TCP JSONL port and enable that endpoint. |
| `--websocket-host HOST` | `None` | `127.0.0.1` | Override the WebSocket bind host. A host-only override does not enable WebSocket. |
| `--websocket-port PORT` | `None` | disabled (`null`) | Set the WebSocket port and enable that endpoint. |
| `--state-rate-hz HZ` | `None` | `60.0` | Override state sampling frequency. Exactly `0` suppresses state output and opens no state sink; negative values are rejected. |
| `--state-include-efforts`, `--no-state-include-efforts` | `None` | `false` | Include or omit commanded, measured, and applied effort samples. |
| `--state-include-objects`, `--no-state-include-objects` | `None` | `false` | Include or omit runtime object poses in state output. |
| `--foxglove-live-host HOST` | `None` | `127.0.0.1` | Override the Foxglove live bind host. A host-only override does not enable live output. |
| `--foxglove-live-port PORT` | `None` | disabled (`null`) | Set the Foxglove live port and enable that endpoint. |
| `--foxglove-mcap-path PATH` | `None` | disabled (`null`) | Write the state stream to this MCAP path, subject to the runtime output policy. |
| `--foxglove-joint-effort-field {none,commanded,measured,applied}` | `None` | `none` | Select the source for `/joint_states.effort`; any value other than `none` requires effort sampling. |

The paired Boolean options write one field. If both forms are present,
argparse keeps the last occurrence; use one form per invocation so logs remain
unambiguous.

## Endpoint And Lifetime Relationships

`--tcp-jsonl-port`, `--websocket-port`, and `--foxglove-live-port` each enable a
different service. Do not assign two services the same host/port pair; config
resolution does not preflight that collision, so the later listener fails while
binding. Using distinct ports is the clearest setup. Built-in listener validation
accepts only `localhost` or numeric loopback addresses. These services have no
authentication or TLS; use a loopback upstream behind an authenticated TLS proxy
or an SSH tunnel for remote access.

Process lifetime and idle stepping are independent:

- `stdin_eof_policy=exit` requests shutdown at natural EOF only when no TCP,
  WebSocket, state stream, or camera output still owns the process lifetime.
- `stdin_eof_policy=keep_alive` keeps a stdin-only process alive after EOF.
- `idle_physics_policy=hold_step` is required when GUI, cameras, or live state
  must continue refreshing without control requests.
- `idle_physics_policy=pause` avoids idle physics work for a request-driven
  service.

The Single Scene CLI does not expose `stdin_enabled`, transport queue sizes, camera
output policy, output collision policy, or shutdown timeouts. Configure those
in the runtime YAML. See [Configuration](configuration.md) and
[Telemetry](../guides/telemetry.md).

## Startup And Exit Markers

Every live Isaac launch requires explicit EULA acceptance in the deployment
environment:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene --tcp-jsonl-port 8765
```

The process prints these stable markers:

| Marker | Meaning |
|---|---|
| `SINGLE_SCENE_INTERACTIVE_CONFIG runtime_profile=<name> fingerprint=<hash>` | Configuration resolved; runtime creation is about to begin. |
| `SINGLE_SCENE_INTERACTIVE_READY` | Transports and the command queue can accept requests. Lazy rendering work may still occur on later rendered steps. |
| `SINGLE_SCENE_INTERACTIVE_EXIT` | The interactive loop has entered final shutdown. |
| `SINGLE_SCENE_INTERACTIVE_OK steps=<n>` | Shutdown returned normally; `<n>` is the final global simulation step. |
| `SINGLE_SCENE_INTERACTIVE_FAILED <Exception>: <message>` | Startup or runtime escaped with an exception; the wrapper exits nonzero. |

Shutdown timeout diagnostics may appear between `EXIT` and `OK`. Treat them as
resource cleanup failures even if the main function returned. See
[Troubleshooting](../operations/troubleshooting.md).
