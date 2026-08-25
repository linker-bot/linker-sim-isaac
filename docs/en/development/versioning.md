# Version And Revision Identity

Language: [English](versioning.md) | [中文](../../zh-CN/development/versioning.md)

linker-sim-isaac is a workspace application rather than an installable wheel, but
operators and bug reports still need a stable compatibility identifier. Read it
without starting Isaac:

```bash
PYTHONPATH=src python scripts/mirror.py --version
PYTHONPATH=src python scripts/kaleidoscope_viewer.py --version
```

Python integrations may read `linkerbot_sim.__version__`. Development checkouts must
also report `git rev-parse HEAD`: the compatibility version identifies the intended
API/configuration line, while the commit identifies an exact revision.

## Updating The Version

The authoritative release value is declared in `pyproject.toml` and mirrored in the
pure, import-safe `linkerbot_sim` facade because this workspace deliberately does not
install distribution metadata. The CPU quality gate requires both values to match.

Update both values in one focused release change. Do not derive the version by reading
the checkout during import: importing the top-level facade must remain free of file I/O,
Kit startup, CUDA initialization, and optional runtime dependencies.

The complete tag, GPU acceptance, archive, checksum, and GitHub Release procedure is
documented in [Releases](../operations/releases.md).
