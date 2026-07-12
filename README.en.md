# LinkerHand Simulation

Language: [English](README.en.md) | [中文](README.zh-CN.md)

LinkerHand Simulation is a checkout-based Isaac Sim manipulation workspace for
multi-robot scenes, cloned environments in Tiled Scene mode, cuRobo planning, trajectories,
snapshots, telemetry, cameras, and persistent experiment output.

The project has two runtime shapes. `SingleSceneRuntime` manages one scene graph with
any configured robot count. `TiledSceneRuntime` manages batched cloned
environments. They share configuration domains and data models where appropriate,
but Tiled Scene does not run through `SingleSceneRuntime`.

## Requirements

- Linux x86_64
- Python 3.11
- Isaac Sim 5.1
- PyTorch 2.7
- NVIDIA cuRobo 0.8.0 and a compatible CUDA GPU for cuRobo operations

Dependencies are declared in `pyproject.toml` and pinned by `uv.lock`:

```bash
uv sync --all-extras
```

Isaac entrypoints require explicit EULA acceptance in the deployment environment:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

This repository is a workspace application, not an installable wheel. Run commands
from the checkout root with `PYTHONPATH=src`; runtime profiles, scripts, assets, and
vendored task resources are part of the application.

## Validate Configuration

Validate both bundled runtime graphs without starting Isaac:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_config.py --runtime-profile default_single_scene
PYTHONPATH=src .venv/bin/python scripts/validate_config.py --runtime-profile default_tiled_scene
```

## Run

Single Scene runtime:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/single_scene_interactive.py \
  --runtime-profile default_single_scene --env scene1 --gui
```

Tiled Scene runtime:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
PYTHONPATH=src .venv/bin/python scripts/tiled_scene_interactive.py \
  --runtime-profile default_tiled_scene --env scene3_tiled --gui
```

Both processes accept strict JSON through stdin. They can also expose loopback-only
TCP JSONL and WebSocket listeners. The built-in listeners have no authentication or
TLS; remote access requires an authenticated TLS proxy or SSH tunnel terminating on
loopback.

## Documentation

- [Project overview](docs/en/getting-started/project-overview.md)
- [Choose Single Scene, Tiled Scene, JSON, or Python](docs/en/getting-started/choose-runtime-and-api.md)
- [Single Scene quickstart](docs/en/getting-started/single-scene-quickstart.md)
- [Tiled Scene quickstart](docs/en/getting-started/tiled-scene-quickstart.md)
- [Single Scene CLI reference](docs/en/reference/single-scene-cli.md)
- [Single Scene JSON and runtime reference](docs/en/reference/single-scene-json.md)
- [Tiled Scene CLI reference](docs/en/reference/tiled-scene-cli.md)
- [Tiled Scene JSON and runtime reference](docs/en/reference/tiled-scene-json.md)
- [Python facade reference](docs/en/reference/python-api.md)
- [Configuration guide](docs/en/guides/configuration.md)
- [Motion planning and cuRobo](docs/en/guides/motion-planning.md)
- [Cameras](docs/en/guides/cameras.md)
- [Source module map](docs/en/development/module-map.md)
- [Complete documentation index](docs/en/index.md)

## Quality Checks

```bash
just quality
```

The command checks formatting, lint, documentation links, the full CPU test suite,
coverage, and both bundled configuration graphs.
