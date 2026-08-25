# Contributing

Language: [English](CONTRIBUTING.md) | [中文](CONTRIBUTING_zh.md)

Contributions are welcome when they preserve the product, runtime-ownership, and
device boundaries described in the documentation. Start with the
[project overview](docs/en/getting-started/project-overview.md) and
[mode/API chooser](docs/en/getting-started/choose-runtime-and-api.md) before changing
public behavior.

## Before Opening A Change

- Search existing issues and pull requests.
- Use a focused issue for behavior that changes a public facade, configuration
  schema, wire protocol, Kit closure, physics owner, or third-party asset policy.
- Keep Mirror and Kaleidoscope capabilities separate. Do not add a Kaleidoscope
  camera, transport, planner, telemetry worker, or other Mirror-owned service without
  an explicit architecture decision.
- Report vulnerabilities through [SECURITY.md](SECURITY.md), not a public issue.

## Development Environments

Install the runtime and CPU development environments separately:

```bash
uv sync --extra simulation --extra visualization --extra training
UV_PROJECT_ENVIRONMENT=.venv-dev \
  uv sync --extra dev --extra visualization
```

Never combine `dev` and `simulation`, and do not use `--all-extras`. PyPI
`usd-core` in the development environment must not shadow Kit's `pxr` modules in
the simulation environment. See [Installation](docs/en/getting-started/installation.md).

## Make A Focused Change

- Use public facades under `linkerbot_sim.configuration`, `linkerbot_sim.isaac`,
  `linkerbot_sim.mirror`, `linkerbot_sim.kaleidoscope`, and
  `linkerbot_sim.training.skrl`.
- Keep configuration facts in their owning leaf; do not duplicate device, engine,
  environment-count, or output facts.
- Preserve owner-thread access to Isaac, USD, physics, camera, and planner resources.
- Keep native Kaleidoscope state and training data on the selected CUDA device.
- Preserve current comments around non-obvious ownership, lifecycle, device, and
  physics behavior; add concise explanations when introducing a new boundary.
- Do not commit NVIDIA Warehouse content or other third-party assets that the
  repository intentionally excludes.

## Documentation

English and Chinese documentation use matching relative paths. When changing public
behavior, update both language trees in the same pull request. Keep code, option
names, operation names, field names, and factual tables equivalent.

Run the maintained link checker through the quality gate. Do not add redirect pages
for removed product names or document internal helpers as stable APIs.

## Validation

Run the CPU gate for every change:

```bash
UV_PROJECT_ENVIRONMENT=.venv-dev just quality
```

The gate measures production modules marked `runtime: pure` in
`architecture/module_disposition.yaml` and enforces the
`tool.linkerbot_sim.coverage.pure_fail_under` floor. This architecture-derived scope
keeps missing Isaac/CUDA imports out of CPU coverage without allowing an arbitrary
file omit list. Runtime-label changes therefore change the coverage scope and require
the same architecture review as other module-map changes. The separate `just test`
recipe uses the simulation environment and retains the all-source coverage floor in
`tool.coverage.report`.

If source modules move, refresh and verify the architecture inventory:

```bash
just update-architecture
just test-architecture
```

Changes to Kit composition, Isaac lifecycle, physics backends, CUDA tensors, cuRobo,
rendering, cameras, or simulation assets also require the simulation gate on a
compatible NVIDIA host:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
just test-simulation
```

If the complete simulation matrix is too large for an iteration, run the narrowest
relevant recipe first, then record exactly which GPU gates remain outstanding.

## Pull Request Checklist

- The change has one clear scope and explains its product boundary.
- Public API/configuration/wire changes include tests and bilingual documentation.
- `just quality` passes.
- Relevant simulation smokes pass, or the pull request states why they could not run.
- New local Markdown links resolve.
- Architecture inventory is current when modules moved.
- No generated output, local environment, credential, internal path, or excluded
  third-party asset is committed.
- License and attribution changes are reflected in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
