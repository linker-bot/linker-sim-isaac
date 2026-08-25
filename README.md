# linker-sim-isaac

[![Quality](https://github.com/linker-bot/linker-sim-isaac/actions/workflows/quality.yml/badge.svg)](https://github.com/linker-bot/linker-sim-isaac/actions/workflows/quality.yml)

Language: [English](README.md) | [中文](README_zh.md)

Start here: [Installation](docs/en/getting-started/installation.md) ·
[Choose a product and API](docs/en/getting-started/choose-runtime-and-api.md) ·
[Documentation](docs/en/index.md)

linker-sim-isaac is an Isaac Sim workspace for robotic manipulation, reality
replay, and GPU-parallel reinforcement learning. The repository exposes two product
modes with deliberately different contracts:

- **Mirror** maps one real workspace into one simulation world. It owns interactive
  control, full motion planning and collision avoidance, cameras, telemetry, and
  JSON transports.
- **Kaleidoscope** runs many homogeneous reinforcement-learning environments through
  either PhysX CUDA or the project's multi-world Newton runtime. Both backends
  keep a headless GPU-native training path and offer an explicit single-environment
  debug viewport. The product owns device-resident state, snapshots and
  cloning, batched IK, synchronous linear end-effector actions, Gymnasium integration,
  and a CUDA-native skrl path. It has no batch trajectory planner, avoidance service,
  camera, SyntheticData, Replicator, recording, transport, or telemetry.

Choose the product before choosing a physics backend or client API:

| If you need... | Start with |
| --- | --- |
| One reality-mapped workcell, interactive control, motion planning, cameras, or JSON | **Mirror** |
| Many homogeneous GPU environments for reinforcement learning | **Kaleidoscope** |
| Language-neutral process control | **Mirror JSON** |
| CUDA-resident training | **Kaleidoscope native Torch or skrl** |
| Gymnasium compatibility | **Kaleidoscope Gymnasium adapter** |

These names are the public API. There is no compatibility contract for the retired
mode names or entrypoints.

The product factory selects exactly one of seven formal Kit experiences:

| Product | Engine / execution | Render closure | Formal Kit experience |
| --- | --- | --- | --- |
| Mirror | PhysX / CPU | Controlled by the outputs profile | `apps/linkerbot_sim.mirror.physx.python.kit` |
| Mirror | Newton / CPU or CUDA | Disabled | `apps/linkerbot_sim.mirror.newton.python.kit` |
| Mirror | Newton / CPU or CUDA | Enabled | `apps/linkerbot_sim.mirror.newton_render.python.kit` |
| Kaleidoscope | PhysX / CUDA | Training headless | `apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` |
| Kaleidoscope | Newton / CUDA | Training headless | `apps/linkerbot_sim.kaleidoscope.newton.python.kit` |
| Kaleidoscope | PhysX / CUDA | Explicit selected-environment viewport | `apps/linkerbot_sim.kaleidoscope.physx_cuda_viewport.python.kit` |
| Kaleidoscope | Newton / CUDA | Explicit selected-environment viewport | `apps/linkerbot_sim.kaleidoscope.newton_viewport.python.kit` |

Call a Mirror or Kaleidoscope product entrypoint rather than assembling Kit manually;
the factory makes the unique choice from the validated physics and render specification.
Public selectors state the legal execution explicitly: Mirror provides `physx_cpu`,
`physx_cpu_hybrid`, `newton_cpu`, and `newton_cuda`; Kaleidoscope provides `physx_cuda` and `newton_cuda`.
The roots reference the product-namespaced scene selectors `mirror/scene3` and
`kaleidoscope/tblock_push`, respectively.

## Requirements

- Linux x86_64
- Python 3.12
- Isaac Sim 6.0.1
- PyTorch 2.11 with CUDA 12.8
- NVIDIA cuRobo 0.8.0 for planning or end-effector actions
- A compatible NVIDIA GPU for Kaleidoscope and Newton

For cloning, `uv` setup, dependency extras, GPU preflight, and the optional NVIDIA
Warehouse payload, read the
[installation guide](docs/en/getting-started/installation.md).

Install the complete simulation workspace from the checkout root:

```bash
uv sync --extra simulation --extra visualization --extra training
```

Keep the CPU development environment separate because its `usd-core` package must
not shadow Kit's `pxr` modules:

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev uv sync --extra dev --extra visualization
```

Accept the Isaac EULA before starting either product:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

This repository is a workspace application, not an installable wheel. Run commands
from the checkout root with `PYTHONPATH=src`.

> The default Mirror `scene3` references an NVIDIA Industrial Warehouse payload that
> is not redistributed by this repository. The project-owned wrapper and analytic
> floor remain in the checkout, but install the licensed payload at the documented path
> before relying on warehouse visuals. Kaleidoscope does not require this payload.

## Validate Configuration Without Starting Isaac

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu_hybrid
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile newton_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile newton_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile physx_cuda
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile newton_cuda
```

The validator reports the exact source files and a deterministic configuration
fingerprint. Mode roots live under `configs/modes/`; referenced leaf profiles remain
the only owners of their facts.

All Mirror physics profiles share `configs/control/mirror.yaml`; `physics.engine` derives the
default PhysX or Newton controller bundle. Kaleidoscope has no control slot at all:
its roots contain `scene`, `physics`, and `task`, plus optional `curobo` only for
end-effector or linear actions. Mirror logging is configured only under
`outputs.logging`. The cuRobo backend fixes its validated 0.8.0 task bundle and
float32 dtypes; YAML profiles contain only real numerical capacity choices.

## Start Mirror

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/mirror.py --profile physx_cpu
```

Mirror accepts strict `linkerbot.mirror.v1`, `v2`, and `v3` JSON through stdin. Optional TCP JSONL
and WebSocket listeners are loopback-only and provide neither authentication nor
TLS. See the [Mirror quickstart](docs/en/getting-started/mirror-quickstart.md).

## Start Kaleidoscope

```python
import torch

from linkerbot_sim.kaleidoscope import make_torch_env

env = make_torch_env(profile="physx_cuda", num_envs=256)
observations, info = env.reset()
actions = torch.zeros(
    (env.num_envs, env.action_dim), device=env.device, dtype=torch.float32
)
observations, rewards, terminated, truncated, info = env.step(actions)
env.close()
```

The native interface returns CUDA tensors. Its debug `step` performs one synchronous
done-scalar guard so an unreset terminal row is rejected before physics advances; the
skrl SAME_STEP path does not execute that guard. Use the Gymnasium adapter only when a
NumPy boundary is required. See the
[Kaleidoscope quickstart](docs/en/getting-started/kaleidoscope-quickstart.md).

Select `profile="newton_cuda"` for Newton. The factory selects
`apps/linkerbot_sim.kaleidoscope.physx_cuda.python.kit` or
`apps/linkerbot_sim.kaleidoscope.newton.python.kit` from the profile. The
Newton experience uses the project's Python runtime as the multi-world physics owner;
it does not load the Isaac Newton extension.

Run the real-physics smoke for both profiles:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile physx_cuda --num-envs 2 --steps 2
PYTHONPATH=src .venv/bin/python scripts/smoke_kaleidoscope_physics.py \
  --profile newton_cuda --num-envs 2 --steps 2 \
  --exercise-training-adapters
```

`just smoke-kaleidoscope` also exercises real Newton batch IK and synchronized linear
actions. See the [Kaleidoscope quickstart](docs/en/getting-started/kaleidoscope-quickstart.md)
for the full commands.

Launch the explicit viewport for either backend with:

```bash
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py \
  --profile physx_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
PYTHONPATH=src .venv/bin/python scripts/kaleidoscope_viewer.py \
  --profile newton_cuda --viewport-profile kaleidoscope --num-envs 4 --selected-env 0
```

The viewer reads `configs/visualization/kaleidoscope.yaml` separately from
the task/physics graph, so display choices do not change episode snapshot/clone
compatibility. Training steps still use `render=False`; only explicit `env.render()`
calls update the viewport. No camera, SyntheticData, Replicator, or recording pipeline
is added.

## Documentation

- [Documentation index](docs/en/index.md)
- [Installation and environment setup](docs/en/getting-started/installation.md)
- [Project overview](docs/en/getting-started/project-overview.md)
- [Choose a mode and API](docs/en/getting-started/choose-runtime-and-api.md)
- [Mirror CLI](docs/en/reference/mirror-cli.md)
- [Mirror JSON](docs/en/reference/mirror-json.md)
- [Kaleidoscope API](docs/en/reference/kaleidoscope-api.md)
- [Configuration reference](docs/en/reference/configuration.md)
- [Python API](docs/en/reference/python-api.md)
- [Troubleshooting](docs/en/operations/troubleshooting.md)
- [Source module map](docs/en/development/module-map.md)
- [Contributing](CONTRIBUTING.md)

## Quality Checks

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev just quality
```

Run the Isaac, CUDA, real-physics, Newton-capacity, and PhysX process-memory gates in
the simulation environment:

```bash
just test-simulation
```

The aggregate includes `smoke-runtime-kits` for all seven formal Kit closures,
`smoke-mirror` for all four Mirror mode profiles, both Kaleidoscope backends and action
variants, Newton's 256-world capacity, and the PhysX process-memory budget.
Trusted NVIDIA-runner automation, triggers, and setup requirements are documented in
[Simulation CI](docs/en/operations/simulation-ci.md).

## License

Released under the [MIT License](LICENSE), © Linkerbot (Beijing) Technology Co., Ltd.
Third-party software and asset licenses are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
