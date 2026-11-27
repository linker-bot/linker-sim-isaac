# linker-lab

An Isaac Lab extension providing one Linker workstation, **LW-v0**: two AR5 arms with O6
hands, a table and three cameras, behaving like the real robot. The robot's files, its
drives, the real controller's motion limits and one validated physics setup are built in, so
what you do in simulation is what the real workstation can do.

Use it to

- **drive the robot** by hand in the playground (`linker-lab play`);
- **run pretrained policies** from the gallery (`linker-lab gallery run ycb-grasp`);
- **write your own tasks** and train on them with Isaac Lab's trainers (`templates/task/`).

## Quick start

Needs Linux (x86_64 or aarch64) with glibc 2.34 or newer (Ubuntu 22.04 or newer), git, an NVIDIA GPU with
driver 580 or newer and 10 GB of free disk ([requirements](docs/requirements.md)).

```bash
git clone https://github.com/linker-bot/linker-lab
cd linker-lab
./install.sh                              # installs everything into .venv, then checks it
uv run linker-lab play                    # drive LW-v0 with the keyboard and mouse
uv run linker-lab gallery run ycb-grasp   # a pretrained grasp policy
```

If `install.sh` installed uv, open a new terminal (or `source ~/.local/bin/env`) before
the `uv run` commands.

## Documentation

- [Getting started](docs/getting-started.md): install, the playground, the gallery
- [The LW-v0 environment](docs/environment.md): the scene, frames, arm and hand commands,
  objects, cameras, in your own scripts
- [The task template](templates/task/README.md): your own task, as your own project
- [The policy gallery](docs/gallery.md): the entries, and how to write one
- [API reference](docs/api.md): everything linker-lab provides, and what stays stable
- [Requirements](docs/requirements.md), [changelog](CHANGELOG.md)
- [中文快速入门](docs/zh-CN/README.md)

## Develop

```bash
uv sync --locked
uv run ruff check .
uv run pytest           # GPU tests in tests/gpu skip themselves without CUDA
```

## API and versions

[docs/api.md](docs/api.md) is the public API: from 1.0 it follows semantic versions, and the
[changelog](CHANGELOG.md) lists every change to it. LW-v0's files come from
[linker-sim-assets](https://github.com/linker-bot/linker-sim-assets) at the release tag
`v1.0.0`, which keeps LW-v0 the same robot throughout 1.x.

## License

Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). The gallery's policy weights
and objects have their own licences, shown by `linker-lab gallery info`.
