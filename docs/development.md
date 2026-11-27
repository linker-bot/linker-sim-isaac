# Development and validation

Run these commands from the `linker-sim-isaac` repository root on a supported Linux system.
The package and CLI are named `linker-lab`; Python modules live in `src/linker_lab/`.

## Install the development environment

```bash
uv sync --locked --group dev
uv run --no-sync ruff check .
```

The `dev` dependency group adds pytest, Ruff and Ruckig; it is not a `--extra dev` feature.
The simulation runtime is still installed. CPU checks do not provide a CPU-only backend or
make this project usable on macOS/Windows. Keep `uv.lock` unchanged unless intentionally
updating dependencies; `uv lock --check` checks consistency without resolving a new graph.

## CPU checks

```bash
uv run --no-sync python -m pytest tests --ignore=tests/gpu --ignore=tests/distribution
```

These checks include API/documentation contracts, robot-data conversion, controller math,
snapshot files, gallery loading/downloads and repository/release policy. They need the
installed Isaac Lab/Newton/USD stack, but do not require a CUDA simulation. The top-level
`tests/conftest.py` prepares the compatibility patches before test collection.

## Installed wheel resources

```bash
uv pip install --python .venv/bin/python "hatchling>=1.27,<2"
uv run --no-sync python -m pytest tests/distribution --confcutdir=tests/distribution
```

If `UV_PROJECT_ENVIRONMENT` selects another environment, replace `.venv/bin/python` with
its interpreter. This gate builds an actual wheel and exercises gallery discovery from an
isolated directory. Hatchling is installed explicitly so the test cannot silently skip.
The separate conftest root keeps this distribution check independent of engine bootstrap.

## GPU and viewer acceptance

```bash
uv run --no-sync linker-lab doctor
uv run --no-sync python -m pytest tests/gpu -ra
uv run --no-sync linker-lab play --selftest --out tests/outputs/playground
```

A CUDA device is required. Window tests also require a working display (`DISPLAY` in the
current test suite). Headless scripts do not require it, but a GPU-only run without a
display skips viewer tests. Read the skipped-test summary: a successful CPU run, or GPU
checks that skipped, do not verify simulation/viewer acceptance. Gallery acceptance also
requires the exact policy/object files from its manifest; see
[artifact availability](gallery.md#artifact-availability).

The `Quality` workflow runs CPU and wheel checks. The manually dispatched `Simulation`
workflow runs on the configured self-hosted Linux x86_64 NVIDIA runner, requires a display,
and requires `OMNI_KIT_ACCEPT_EULA=Y` in the simulation environment after reviewing NVIDIA's
terms. Local aarch64 use and the workflow's x86_64 runner are separate validation targets.

## Package layout

- `assets`, `physics`, `objects`: robot data, conversion, solver configuration and rigid objects.
- `workstation`: the scene, Gym registration, action configurations, frames and cameras.
- `arm`, `hand`: controller models and Isaac Lab action terms.
- `snapshot`: supported physics/control/environment state; schema 2 also stores observations.
- `gallery`, `playground`, `cli`: policy discovery, interactive control and command dispatch.
- `engine`: the pinned compatibility patches; an internal module rather than a public API.
- `templates/task`: a separate task project; its sources and lock are independent of this checkout.

The public contract is [api.md](api.md). New Python examples are collected by
`tests/test_docs.py`; simulation examples execute separately in `tests/gpu/test_docs_gpu.py`.
When changing runtime behavior, update the matching API/guide text and the Unreleased
changelog section as well as adding a regression for the behavior.

## Releases

The existing repository tag `v1.0.0` is a legacy-workspace tag and does not contain the new
`linker-lab` package. Do not use it as this package's install source or move it to new code.
Choose a new, unused release version and keep `pyproject.toml`, `linker_lab.__version__` and
the dated changelog section consistent.

The `Release` workflow takes an existing annotated tag plus a successful `Simulation` run
for that exact commit. It reruns CPU/wheel checks and validates tag/version/evidence before
publishing the source archive, SHA-256 sums and release notes. It does not publish a PyPI
wheel or policy weights. Gallery artifacts need their own published files and matching
manifest URLs and hashes. The release validator is `scripts/prepare_release.py`.

## Separate runtime corrections

This documentation change does not include the implementation commits in
[PR #82](https://github.com/linker-bot/linker-sim-isaac/pull/82) (snapshot schema 2),
[PR #83](https://github.com/linker-bot/linker-sim-isaac/pull/83) (streamed motion limits),
[PR #84](https://github.com/linker-bot/linker-sim-isaac/pull/84) (gallery loading/downloads), or
[PR #85](https://github.com/linker-bot/linker-sim-isaac/pull/85) (doctor diagnostics).
Descriptions of those behaviors apply to builds containing the corresponding correction.
The task template's pinned commit predates them; use the local source option when testing
an unmerged change.
