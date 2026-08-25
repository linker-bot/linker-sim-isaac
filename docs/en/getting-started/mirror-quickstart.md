# Mirror Quickstart

Language: [English](mirror-quickstart.md) | [中文](../../zh-CN/getting-started/mirror-quickstart.md)

This walkthrough validates a Mirror configuration, starts one reality-mapped world,
queries status, and requests an orderly shutdown. Complete the
[installation guide](installation.md) first if this is a new checkout.

## 1. Prepare The Environment

From the repository root:

```bash
uv sync --extra simulation --extra visualization
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH=src
```

Use a separate `.venv-dev` for CPU tests; do not install `usd-core` into the Isaac
environment.

The default `mirror/scene3` references an NVIDIA Warehouse visual payload that is
not redistributed. Its expected location and verification command are documented in
[Installation](installation.md). Configuration validation can succeed without that
payload, so verify it separately when warehouse visuals are part of the workflow.

## 2. Validate The Composition

```bash
.venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu
```

The `physx_cpu` profile, which is the CLI default, resolves:

- scene selector `mirror/scene3` to `configs/scenes/mirror/scene3.yaml`, whose
  internal identity is `scene.id: scene3`;
- `configs/physics/physx/cpu.yaml`;
- the shared `configs/control/mirror.yaml` and selected output profile;
- `configs/curobo/mirror.yaml` for IK batch capacity and the single-request
  MotionPlanner's warmup, seeds, CUDA graph, collision capability, and cache capacity;
- `configs/planning/mirror.yaml` for backend-neutral request defaults and the
  non-overridable per-request planner timeout.

`kinematics.max_batch_size` sizes IK only; the MotionPlanner context stays fixed at
`max_batch_size=1`. A wire request may override duration, sampling, avoidance, and
refresh, plus coordination at the wrapper/timeline level, but it cannot provide
`timeout_s`.

Use `--profile newton_cpu` or `--profile newton_cuda` to validate Newton on CPU or
CUDA. Both derive exactly one Mirror world and use `control: mirror`; the physics
engine derives the default controller bundle. CPU physics still keeps the root CUDA
device for cuRobo and RTX.

Use `--profile physx_cpu_hybrid` for the dedicated 240 Hz PhysX CPU composition. It
adds `profiles.hybrid_control: hybrid_force_position`; operate it through Mirror v3
after a wrench tare. The ordinary `physx_cpu` default does not silently enable hybrid
control.

## 3. Start The Service

```bash
.venv/bin/python scripts/mirror.py \
  --profile physx_cpu \
  --stdin
```

`--profile physx_cpu` is optional because `physx_cpu` is the Mirror CLI default. The
canonical control profile also enables wall-clock simulation pacing; set
`control.sync_simulation_to_wall_clock: false` to run without pacing.

With `outputs.render.gui: true`, Newton scene3 opens both the human `Viewport` and
the `NewtonCamera:world_rgbd` data-product window. Navigate only in `Viewport`:
`Alt + left drag` tumbles, middle drag pans, right drag looks, and the wheel zooms.
Camera navigation is disabled per `NewtonCamera:*` window so pointer input cannot
change RGB-D sensor extrinsics.

Wait for `MIRROR_INTERACTIVE_READY`. Blank stdin lines are ignored. End-of-file asks
the runtime to quit because stdin is enabled by default.

## 4. Query Status

Send one compact JSON object on one line:

```json
{"protocol":"linkerbot.mirror.v1","request_id":"status-1","operation":"runtime.status","arguments":{}}
```

The response repeats the protocol and request ID and contains either `ok: true` plus
`result`, or `ok: false` plus a structured error.

## 5. Capture A Snapshot

```json
{"protocol":"linkerbot.mirror.v1","request_id":"snapshot-1","operation":"snapshot.get","arguments":{}}
```

Save the returned snapshot object without changing its schema or identity fields. To
restore it, pass that object as `arguments.snapshot` to `snapshot.set`.

## 6. Stop Cleanly

```json
{"protocol":"linkerbot.mirror.v1","request_id":"quit-1","operation":"runtime.quit","arguments":{}}
```

Normal shutdown prints `MIRROR_INTERACTIVE_EXIT`. A close timeout is an error; Mirror
does not destroy the Isaac session while a child resource still owns a worker or
engine view.

## Optional Loopback Endpoints

```bash
.venv/bin/python scripts/mirror.py \
  --no-stdin \
  --tcp-jsonl 127.0.0.1:8765 \
  --websocket 127.0.0.1:8766
```

Both listeners accept loopback hosts only. They have no authentication or TLS. Use an
authenticated local proxy or SSH tunnel if another machine must connect.

## Python Embedding

```python
from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.mirror import create_mirror_runtime

runtime = create_mirror_runtime(load_mirror_config("physx_cpu"))
try:
    initial = runtime.capture_snapshot()
    runtime.step(render=False)
    runtime.restore_snapshot(initial)
finally:
    report = runtime.close()
    if not report.stopped:
        raise RuntimeError(report)
```

Run all four formal Mirror mode profiles with the maintained product gate. The seven-Kit
closure gate additionally exercises the renderer-free Newton physics-only experience:

```bash
just smoke-mirror
just smoke-runtime-kits
```

`just test-simulation` combines those recipes with both Kaleidoscope backends, Newton
capacity, and the PhysX process-memory budget.

Continue with [Mirror CLI](../reference/mirror-cli.md),
[Mirror JSON And Motion Examples](../reference/mirror-json.md), and
[Motion Planning](../guides/motion-planning.md).
