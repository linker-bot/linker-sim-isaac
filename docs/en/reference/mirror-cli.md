# Mirror CLI Reference

Language: [English](mirror-cli.md) | [中文](../../zh-CN/reference/mirror-cli.md)

The supported process entrypoint is `scripts/mirror.py`. It loads one strict Mirror
composition, creates one `MirrorRuntime`, starts the selected ingress endpoints, and
closes every owned resource before exiting.

```bash
PYTHONPATH=src .venv/bin/python scripts/mirror.py [OPTIONS]
```

## Complete Option Table

| Option | Default | Meaning |
| --- | --- | --- |
| `--version` | - | Print the workspace compatibility version without starting Isaac, then exit. |
| `--profile NAME` | `physx_cpu` | Load `configs/modes/mirror/NAME.yaml`; accepted values are `physx_cpu`, `physx_cpu_hybrid`, `newton_cpu`, and `newton_cuda`. |
| `--stdin` / `--no-stdin` | enabled | Enable or disable one-request-per-line JSON on standard input. |
| `--tcp-jsonl HOST:PORT` | disabled | Start a loopback TCP JSONL listener. CLI ports are in `[1, 65535]`. |
| `--websocket HOST:PORT` | disabled | Start a loopback WebSocket text-frame listener. CLI ports are in `[1, 65535]`. |
| `--response-timeout-s SECONDS` | `30.0` | Maximum time an ingress worker waits for its request's response. Must be positive. |
| `--poll-timeout-s SECONDS` | `0.05` | Owner-thread admission poll interval. Must be positive. |

`HOST:PORT` must contain a host and decimal port. Network listeners accept only
`localhost` or a numeric loopback address. They intentionally reject wildcard and
LAN addresses.

## Examples

Default stdin service:

```bash
PYTHONPATH=src .venv/bin/python scripts/mirror.py
```

Newton CUDA composition (use `newton_cpu` for CPU physics):

```bash
PYTHONPATH=src .venv/bin/python scripts/mirror.py \
  --profile newton_cuda
```

240 Hz PhysX CPU hybrid-control composition:

```bash
PYTHONPATH=src .venv/bin/python scripts/mirror.py \
  --profile physx_cpu_hybrid
```

Loopback TCP and WebSocket without stdin:

```bash
PYTHONPATH=src .venv/bin/python scripts/mirror.py \
  --no-stdin \
  --tcp-jsonl 127.0.0.1:8765 \
  --websocket localhost:8766
```

## Process Markers

| Marker | Meaning |
| --- | --- |
| `MIRROR_INTERACTIVE_READY` | Runtime construction and ingress composition completed. |
| `MIRROR_INTERACTIVE_EXIT` | The event loop ended and all resources reported stopped. |
| `MIRROR_INTERACTIVE_FAILED TYPE: MESSAGE` | Startup, operation, or shutdown raised an unhandled error. The process exits nonzero. |

The ready marker does not authenticate a network client or certify task success. It
only identifies the process lifecycle boundary.

## Stdin Semantics

- Each nonblank line must contain exactly one UTF-8 JSON request.
- Each accepted line produces exactly one compact JSON response line.
- Invalid JSON produces an error response with request ID `invalid-request`; the
  stream remains usable.
- End-of-file requests runtime shutdown when stdin is enabled.
- Standard output also carries lifecycle markers, so a client must distinguish JSON
  lines from marker lines.

## Transport Security

The built-in TCP and WebSocket services provide no authentication, authorization,
rate-based abuse protection, or TLS. Loopback enforcement is the security boundary.
For remote operation, terminate an authenticated encrypted tunnel on the same host
and keep the built-in listener on loopback.

## Configuration Versus CLI

The CLI selects a mode profile and endpoints. Scene geometry, physics backend,
control defaults, cuRobo numerical capability, backend-neutral planning request
defaults, rendering, camera, logging, and telemetry remain YAML facts. There are no
CLI switches that partially override those leaf profiles.

Validate before starting Kit:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu
```

See [Mirror JSON](mirror-json.md) for requests and responses and
[Configuration](configuration.md) for profile ownership.
