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

The `Task template` workflow copies `templates/task` outside the checkout, installs it in
a new environment, checks the generated lock, and checks task registration and configuration
imports. It checks the default source dependency as users receive it; no `PYTHONPATH`
override or local library source is injected. Simulation still requires the GPU tests.

## GPU and viewer acceptance

```bash
uv run --no-sync linker-lab doctor
uv run --no-sync python -m pytest tests/gpu -ra
uv run --no-sync linker-lab play --selftest --out tests/outputs/playground
```

A CUDA device is required. Window tests also require a working display (`DISPLAY` in the
current test suite). Headless scripts do not require it, but a GPU-only run without a
display skips viewer tests, so read the summary of skipped tests: a skipped test checked
nothing. The gallery tests download the entries' files ([the files](gallery.md#the-files)).

The `Quality` workflow runs CPU and wheel checks. The manually dispatched `Simulation`
workflow runs on the configured self-hosted Linux x86_64 NVIDIA runner, requires a display,
and requires `OMNI_KIT_ACCEPT_EULA=Y` in the simulation environment after reviewing NVIDIA's
terms. Local aarch64 use and the workflow's x86_64 runner are separate validation targets.

## Package layout

- `assets`, `physics`, `objects`: robot data, conversion, solver configuration and rigid objects.
- `workstation`: the scene, Gym registration, action configurations, frames and cameras.
- `arm`, `hand`: controller models and Isaac Lab action terms.
- `snapshot`: save and restore an environment between two steps.
- `gallery`, `playground`, `cli`: policy discovery, interactive control and command dispatch.
- `engine`: the pinned compatibility patches; an internal module rather than a public API.
- `templates/task`: a separate task project; its sources and lock are independent of this checkout.

The public contract is [api.md](api.md). New Python examples are collected by
`tests/test_docs.py`; simulation examples execute separately in `tests/gpu/test_docs_gpu.py`.
A change in behaviour comes with its test, the matching text in the API reference and the
guides, and a line in the changelog under `## Unreleased`.

## Releases

A release is an annotated tag `vX.Y.Z` whose version is the one in `pyproject.toml`,
`linker_lab.__version__` and a dated changelog section. Before creating that tag, set the
task template's `linker-lab` source to `tag = "vX.Y.Z"`.
`scripts/prepare_release.py` requires all of these, including the matching template tag.
Between releases the template may use an existing complete commit ID, so it remains
installable before the next tag exists; `tests/test_release_links.py` rejects floating
branches and mismatched release tags. Run the independent template check again after the
candidate tag exists. The template does not carry a lockfile referencing its own release
commit; the copied task generates and maintains its own lock. Tags before `v2.0.0`
(up to `v1.0.0`) belong to this repository's earlier package.

The `Release` workflow takes an existing annotated tag plus a successful `Simulation` run
for that exact commit. It reruns CPU/wheel checks and validates tag/version/evidence before
publishing the source archive, SHA-256 sums and release notes
(`scripts/prepare_release.py`). It publishes no wheel and no gallery files.

A gallery entry's files are published by hand as a release named `<entry>-<version>`,
marked not latest, e.g.
`gh release create ycb-grasp-0.3.0 --latest=false policy.npz objects.npz`; its manifest
lists their URLs and sha256 (`tests/test_release_links.py` checks the URLs' form).
