# Single Scene Quickstart

Language: [English](single-scene-quickstart.md) | [中文](../../zh-CN/getting-started/single-scene-quickstart.md)

This workflow starts one Single Scene runtime, discovers the session's robots, submits
a minimal hold command, polls its terminal state, and shuts the process down.
A Single Scene is one physical World and may contain any number of robots from the
selected env profile; `robot_id` values must be discovered for each process.

## Prepare The Checkout

On Linux x86-64 with Python 3.11, Isaac Sim 5.1, and a compatible NVIDIA stack:

```bash
git clone https://gitea.linkerhub.work/LinkerOS/scene-replay-sim-Isaac.git
cd scene-replay-sim-Isaac
uv sync --all-extras
```

Run every project command from the checkout root. This is a workspace
application, so the checkout-local `configs/`, `assets/`, and `scripts/` are
part of the runtime.

## Validate Before Isaac Starts

Validate the complete bundled Single Scene configuration graph without importing or
creating Isaac:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py \
  --runtime-profile default_single_scene
```

Success prints JSON with `"event": "config_validated"`, the selected profile,
and a configuration fingerprint. Fix any `CONFIG_INVALID` result before
starting the runtime.

## Start The Single Scene Service

Read and accept the applicable NVIDIA/Kit EULA, then record that choice in the
same deployment environment. The project does not set this variable for you.

In terminal 1:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene \
  --tcp-jsonl-port 8765
```

Wait for `SINGLE_SCENE_INTERACTIVE_READY`. It means TCP and the shared command
queue can accept requests. A GUI is not required for this headless workflow;
add `--gui` only when visual inspection is needed.

## Discover, Hold, Check, And Quit

In terminal 2, run this standard-library TCP JSONL client from the checkout
root:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import socket
import time


def request(stream, payload):
    stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
    stream.flush()
    line = stream.readline()
    if not line:
        raise ConnectionError("Single Scene service closed before responding")
    return json.loads(line)


with socket.create_connection(("127.0.0.1", 8765), timeout=5.0) as sock:
    sock.settimeout(30.0)
    stream = sock.makefile("rwb")
    try:
        discovery = request(stream, {"type": "status"})
        assert discovery["event"] == "status", discovery
        assert discovery["robots"], discovery

        robot = discovery["robots"][0]
        robot_id = robot["robot_id"]
        groups = robot["joint_groups"]
        group = "arm" if groups.get("arm") else "hand"
        assert groups.get(group), robot
        print("discovered", robot_id, robot["label"], group)

        accepted = request(
            stream,
            {
                "type": "hold",
                "id": "quickstart-hold",
                "robot_id": robot_id,
                "group": group,
                "duration_s": 0.2,
            },
        )
        assert accepted["event"] == "accepted", accepted

        deadline = time.monotonic() + 30.0
        while True:
            status = request(
                stream,
                {"type": "status", "id": "quickstart-hold"},
            )
            assert status["commands"], status
            command = status["commands"][0]
            if command["state"] in {"done", "failed", "cancelled"}:
                assert command["state"] == "done", command
                print("terminal", command)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("quickstart-hold did not reach a terminal state")
            time.sleep(0.05)
    finally:
        print("quit", request(stream, {"type": "quit"}))
PY
```

The first `status` response is the session discovery contract. Do not cache its
`robot_id` across restarts or after changing the env robot order. The hold
submission is asynchronous, so the client polls the named command until
`done`, `failed`, or `cancelled`; TCP does not inject lifecycle events between
direct responses.

## Verify Shutdown

After the client receives `{"event":"quit"}`, terminal 1 should print, in
order:

```text
SINGLE_SCENE_INTERACTIVE_EXIT
SINGLE_SCENE_INTERACTIVE_OK steps=<n>
```

The process should exit with status 0. A `SINGLE_SCENE_INTERACTIVE_FAILED` line,
a nonzero exit, or any shutdown-timeout diagnostic is a failed run.

Continue with the [Single Scene CLI Reference](../reference/single-scene-cli.md) for every
launch option and the [Single Scene JSON Reference](../reference/single-scene-json.md) for
timelines, planning, reset, and snapshots.
