## What and why

<!-- The problem, and how this change solves it. -->

## Public API

<!-- What changes in what docs/api.md lists, or "None". A change that breaks it needs a new major version. -->

## Validation

- [ ] `uv run --no-sync ruff check .` and `uv run --no-sync python -m pytest tests --ignore=tests/gpu --ignore=tests/distribution` pass
- [ ] `uv run --no-sync python -m pytest tests/gpu -ra` passes on a computer with an NVIDIA GPU and a display (or a `Simulation` run is linked below), or the reason it was not run is stated
- [ ] New or changed behaviour has a test, and `docs/api.md` and the guides describe it
- [ ] `CHANGELOG.md` has a line under `## Unreleased`, or users see no change
- [ ] `docs/zh-CN/README.md` still matches the English guides
- [ ] A dependency change updates both `pyproject.toml` and `uv.lock`
- [ ] No credentials, private paths, generated output or third-party files are committed

GPU tests or `Simulation` run (or why not run):

<!-- Paste the test summary or the Actions run URL. -->
