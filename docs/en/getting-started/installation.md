# Installation

Language: [English](installation.md) | [中文](../../zh-CN/getting-started/installation.md)

This project is a checkout application for Linux x86_64, not an installable wheel.
Commands, configuration profiles, assets, and Kit experiences are resolved relative
to the repository root.

## 1. Prerequisites

Install these host tools before creating the project environments:

- Git;
- [uv](https://docs.astral.sh/uv/);
- an NVIDIA driver and GPU compatible with the pinned Isaac Sim/CUDA stack when using
  Kaleidoscope, Newton CUDA, RTX rendering, or cuRobo;
- enough local storage for the Isaac Sim wheels and extension cache.

The repository pins Python 3.12, Isaac Sim 6.0.1, PyTorch 2.11/cu128, Warp 1.13.0,
and cuRobo 0.8.0. Treat `pyproject.toml` and `uv.lock` as the source of truth.

## 2. Clone The Workspace

```bash
git clone https://github.com/linker-bot/linker-sim-isaac.git
cd linker-sim-isaac
uv python install 3.12
```

Run every maintained command below from this checkout root.

## 3. Create The Simulation Environment

Choose only the extras the workflow needs:

| Extra | Adds | Typical use |
| --- | --- | --- |
| `simulation` | Isaac Sim, PyTorch CUDA, cuRobo, Warp, CUDA bindings | Required for either product |
| `visualization` | Foxglove SDK | Mirror telemetry visualization |
| `training` | Gymnasium and skrl | Kaleidoscope adapters and training |
| `dev` | pytest, coverage, Ruff, PyPI USD | CPU-only development checks |

A complete runtime environment is:

```bash
uv sync --extra simulation --extra visualization --extra training
```

A narrower Mirror environment is:

```bash
uv sync --extra simulation --extra visualization
```

A Kaleidoscope training environment is:

```bash
uv sync --extra simulation --extra training
```

Do not use `--all-extras`, and do not combine `dev` with `simulation`.
The `dev` extra installs PyPI `usd-core`, while Isaac must load `pxr` from Kit.
The project declares this pair as a uv conflict so an invalid sync fails instead of
silently contaminating the runtime environment.

## 4. Create The CPU Development Environment

Keep linting, pure tests, architecture checks, and documentation checks in
`.venv-dev`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev   uv sync --extra dev --extra visualization
```

The maintained `just quality` recipes use this environment and do not rewrite the
simulation `.venv`.

## 5. Accept The Isaac EULA

Set the EULA flag in every process environment that starts a Kit application:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

Review NVIDIA's terms before setting this variable. The project does not redistribute
Isaac Sim binaries.

## 6. Prepare The Optional Warehouse Payload

The default Mirror scene `mirror/scene3` uses the project-owned wrapper:

```text
assets/rigid_env_objects/industrial_warehouse_meters/industrial_warehouse_meters.usda
```

That wrapper references this NVIDIA payload:

```text
usd-material/extracted/Industrial_NVD_10012/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd
```

The NVIDIA asset is intentionally excluded from Git and must be obtained separately
under NVIDIA's license. Place it at the exact path above, preserving its companion
asset and texture layout. Verify the entry file with:

```bash
test -f usd-material/extracted/Industrial_NVD_10012/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd
```

Configuration validation checks the project configuration graph; it does not download
or license external content. Kaleidoscope does not use this warehouse payload. If the
payload is unavailable, use a Mirror scene profile that does not reference
`industrial_warehouse` rather than committing third-party content into this
repository.

## 7. Run Preflight Checks

Verify the pinned interpreter and, for GPU profiles, CUDA visibility:

```bash
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)'
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Validate both product graphs without starting Isaac:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode mirror --profile physx_cpu
PYTHONPATH=src .venv/bin/python scripts/validate_mode_config.py \
  --mode kaleidoscope --profile physx_cuda
```

The validator prints the resolved source files and deterministic configuration
fingerprint. It does not prove that the GPU, external payloads, or every Kit extension
can start.

## 8. Start With The Smallest Runtime Check

For Mirror:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_mirror_physics.py --profile physx_cpu --steps 8
```

For Kaleidoscope PhysX CUDA:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=src .venv/bin/python \
  scripts/smoke_kaleidoscope_physics.py \
  --profile physx_cuda --num-envs 2 --steps 2
```

For the complete GPU/Isaac acceptance matrix, run `just test-simulation`. It is
intentionally separate from the CPU `just quality` gate.

## Next Steps

- [Choose a mode and API](choose-runtime-and-api.md)
- [Mirror quickstart](mirror-quickstart.md)
- [Kaleidoscope quickstart](kaleidoscope-quickstart.md)
- [Troubleshooting](../operations/troubleshooting.md)
