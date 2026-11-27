# linker-lab

Source repository: [linker-bot/linker-sim-isaac](https://github.com/linker-bot/linker-sim-isaac).
The installable package and command are `linker-lab`; the Python import is `linker_lab`.

An Isaac Lab extension providing one Linker workstation, **LW-v0**: two AR5 arms with O6
hands, a table and three cameras, behaving like the real robot. The robot's files, its
drives, the real controller's motion limits and one physics setup are built in. Simulation
models these constraints; task performance still needs validation on the real workstation.

Use it to

- **drive the robot** by hand in the playground (`linker-lab play`);
- **run pretrained policies** from the gallery (`linker-lab gallery run ycb-grasp`);
- **write your own tasks** and train on them with Isaac Lab's trainers (`templates/task/`).

## Quick start

Needs Linux (x86_64 or aarch64) with glibc 2.34 or newer (Ubuntu 22.04 or newer), git, an NVIDIA GPU with
driver 580 or newer and 10 GB of free disk ([requirements](docs/requirements.md)).

```bash
git clone https://github.com/linker-bot/linker-sim-isaac.git
cd linker-sim-isaac
./install.sh                              # installs everything into .venv, then checks it
uv run linker-lab play                    # drive LW-v0 with the keyboard and mouse
uv run linker-lab gallery run ycb-grasp   # requires the separately published policy files
```

If `install.sh` installed uv, open a new terminal (or `source ~/.local/bin/env`) before
the `uv run` commands.

The playground and gallery need a working display; the doctor and task scripts can run
headless. Policy weights and object data are not included in the source checkout; see
[gallery asset availability](docs/gallery.md#artifact-availability) before running a policy.
Robot files are installed as the `linker-lw-v0` dependency; no submodule initialization is
needed for this checkout.

## Documentation

- [Getting started](docs/getting-started.md): install, the playground, the gallery
- [The LW-v0 environment](docs/environment.md): the scene, frames, arm and hand commands,
  objects, cameras, in your own scripts
- [The task template](templates/task/README.md): your own task, as your own project
- [The policy gallery](docs/gallery.md): the entries, and how to write one
- [API reference](docs/api.md): everything linker-lab provides, and what stays stable
- [Requirements](docs/requirements.md), [troubleshooting](docs/getting-started.md#troubleshooting)
- [Development and validation](docs/development.md), [changelog](CHANGELOG.md)
- [中文快速入门](docs/zh-CN/README.md)

## Develop

```bash
uv sync --locked --group dev
uv run --no-sync ruff check .
uv run --no-sync python -m pytest tests --ignore=tests/gpu --ignore=tests/distribution
```

These are the CPU checks, on a supported Linux installation. They still import Isaac Lab
and convert robot assets. Wheel-resource checks and GPU/viewer acceptance have separate
commands in [the development guide](docs/development.md); skipped GPU tests are not GPU
acceptance evidence.

## API and versions

[docs/api.md](docs/api.md) is the public API: from 1.0 it follows semantic versions, and the
[changelog](CHANGELOG.md) lists every change to it. LW-v0's files come from
[linker-sim-assets](https://github.com/linker-bot/linker-sim-assets) at the release tag
`v1.0.0`, which keeps LW-v0 the same robot throughout 1.x.

## Migrating from the previous workspace

The current package replaces `linkerbot_sim` and the Mirror/Kaleidoscope entry points.
Existing scripts and YAML configurations need to be adapted to `linker_lab.workstation`
and the new task API; there is no compatibility shim. The existing repository tag
`v1.0.0` contains the previous workspace, not the new `linker-lab` package. To use this
refactor, clone the current branch or use the commit pinned by the task template.

## License

Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). The gallery's policy weights
and objects have their own licences, shown by `linker-lab gallery info`.
