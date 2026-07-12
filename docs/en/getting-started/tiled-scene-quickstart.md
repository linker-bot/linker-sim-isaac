# Tiled Scene Quickstart

Language: [English](tiled-scene-quickstart.md) | [中文](../../zh-CN/getting-started/tiled-scene-quickstart.md)

This workflow starts the independent Tiled Scene runtime, discovers env and robot
IDs, executes one synchronous hold step in env 0, verifies the resulting global
step, and shuts the process down. Tiled Scene uses cloned env rows and does not run
through `SingleSceneRuntime`.

## Prepare The Checkout

On Linux x86-64 with Python 3.11, Isaac Sim 5.1, and a compatible NVIDIA stack:

```bash
git clone https://gitea.linkerhub.work/LinkerOS/scene-replay-sim-Isaac.git
cd scene-replay-sim-Isaac
uv sync --all-extras
```

Run every project command from the checkout root. The workspace depends on its
checkout-local profiles, assets, scripts, and cuRobo task resources.

## Validate Before Isaac Starts

Validate the complete bundled Tiled Scene configuration graph without creating Isaac:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_tiled_scene
```

Success prints JSON with `"event": "config_validated"` and a configuration
fingerprint. Fix any `CONFIG_INVALID` result before launch.

## Start The Tiled Scene Service

Read and accept the applicable NVIDIA/Kit EULA, then record that choice in the
deployment environment. The project never accepts it automatically.

In terminal 1:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene \
  --no-stdin \
  --tcp-jsonl-port 8765
```

Wait for `TILED_SCENE_INTERACTIVE_READY`. The selected `scene3_tiled` env profile,
not a CLI count, determines `num_envs`. The bundled runtime is headless and
pauses physics while idle; add `--gui --idle-physics-policy hold_step` only when
continuous visual refresh is required.

## Discover, Step, Verify, And Quit

In terminal 2, run this standard-library TCP JSONL client from the checkout
root:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import socket


def request(stream, payload):
    stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
    stream.flush()
    line = stream.readline()
    if not line:
        raise ConnectionError("Tiled Scene service closed before responding")
    return json.loads(line)


with socket.create_connection(("127.0.0.1", 8765), timeout=5.0) as sock:
    sock.settimeout(30.0)
    stream = sock.makefile("rwb")
    try:
        discovery = request(stream, {"type": "status"})
        assert discovery["event"] == "status", discovery
        assert discovery["num_envs"] > 0, discovery
        assert discovery["robots"], discovery

        robot = discovery["robots"][0]
        robot_id = robot["robot_id"]
        print("discovered", discovery["num_envs"], robot_id, robot["label"])

        step = request(
            stream,
            {
                "type": "step",
                "kind": "hold",
                "env_ids": [0],
                "robot_id": robot_id,
            },
        )
        assert step["event"] == "step" and step["accepted"] is True, step
        assert step["ticks"] > 0, step

        after = request(stream, {"type": "status"})
        assert after["step"] >= step["step"], (step, after)
        print("terminal", step)
    finally:
        print("quit", request(stream, {"type": "quit"}))
PY
```

The `status` request is session discovery: `env_id` selects a cloned row, while
`robot_id` selects a robot definition repeated across rows. Rediscover both
dimensions after restart or profile changes. Unlike a Single Scene motion command,
`step` is synchronous; its direct response is the terminal result after all
requested physics ticks complete.

## Verify Shutdown

After the client receives `{"event":"quit","accepted":true}`, terminal 1
should print:

```text
TILED_SCENE_INTERACTIVE_EXIT
```

The process should then exit with status 0. Tiled Scene intentionally has no separate
success marker. A `TILED_SCENE_INTERACTIVE_FAILED` line, nonzero exit, or any shutdown
timeout diagnostic is a failed run.

Continue with the [Tiled Scene CLI Reference](../reference/tiled-scene-cli.md) for every
launch option and the [Tiled Scene JSON Reference](../reference/tiled-scene-json.md) for
selectors, state, trajectories, and asynchronous planning.
