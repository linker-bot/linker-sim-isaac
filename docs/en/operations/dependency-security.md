# Dependency Security And Updates

Language: [English](dependency-security.md) | [中文](../../zh-CN/operations/dependency-security.md)

`pyproject.toml` declares the supported dependency profiles and `uv.lock` freezes
their complete Linux x86-64 graph. Dependency changes are never merged on the
strength of a newer version number alone: the Isaac, CUDA, Torch, Warp, and cuRobo
packages form a compatibility closure that must continue to pass the relevant
runtime acceptance tests.

## Automated Update Policy

`.github/dependabot.yml` checks the `uv` and GitHub Actions ecosystems weekly.
Dependabot does not merge changes. It only proposes reviewable pull requests, and
keeps `pyproject.toml`, `uv.lock`, action commit pins, and adjacent version comments
current.

Python updates are grouped by review boundary:

| Group | Contents | Required validation |
| --- | --- | --- |
| `simulation-runtime` | Isaac Sim, Torch, Warp, CUDA bindings, cuRobo, Newton, and related compatibility packages | CPU quality plus trusted Simulation CI |
| `development-tooling` | pytest, coverage, Ruff, and CPU-only USD tooling | CPU quality |
| `application-dependencies` | Remaining base and optional application dependencies | CPU quality; add Simulation CI when the changed package enters a simulation profile |
| `workflow-actions` | GitHub Actions used by repository workflows | Workflow-policy tests and the affected workflow |

The catch-all application group is deliberately last. A package that belongs to the
simulation or tooling boundary must not be absorbed into a broad application update.

## Pull Request Delta Gate

The `Dependency Audit` workflow checks changes to `pyproject.toml` and `uv.lock`. It
checks out the pull request base and head at their exact commit IDs, then executes
`scripts/check_dependency_audit_delta.py` from the trusted base checkout. The head
checkout supplies data only; it cannot replace the comparison policy in the same pull
request. The one-time rollout is the sole exception: when the base predates the
checker, the workflow uses the reviewed candidate copy and emits an explicit notice.

The checker runs `uv audit --frozen` against the complete locked graph for both
commits. It does not install or import a dependency. A pull request fails when the
head introduces either:

- a vulnerability whose identifier or alias was not present for the same package in
  the base graph; or
- a new adverse project status, such as a newly archived dependency.

The comparison uses advisory aliases so duplicate GHSA, CVE, and PYSEC records count
as one finding. Existing findings remain visible in the audit counts but do not make
every unrelated dependency pull request permanently fail. Audit execution, network,
schema, or lockfile errors fail closed instead of being treated as a clean report.

This is a regression gate, not an assertion that the current lock graph has no known
findings. Maintainers still need to triage existing advisories, including findings in
packages constrained by the simulator release.

## Reviewing An Update

1. Read the upstream release and security notes. For a GitHub Action, verify that the
   proposed commit belongs to the documented release tag.
2. Keep direct requirements and `uv.lock` in the same change. Run `uv lock --check`
   after resolving the update; do not hand-edit the lockfile.
3. Run `UV_PROJECT_ENVIRONMENT=.venv-dev just quality`.
4. For `simulation-runtime` or any package loaded by Kit/CUDA, run the trusted
   [Simulation CI](simulation-ci.md). Treat Isaac Sim, Torch, torchvision,
   torchaudio, Warp, CUDA bindings, and cuRobo as one tested closure.
5. Confirm that the Dependency Audit result contains no new findings. A lower total
   count is useful evidence, but it does not replace the compatibility tests.

For a local comparison, retain a clean base checkout and run:

```bash
python scripts/check_dependency_audit_delta.py \
  --base-project ../linker-sim-isaac-base \
  --head-project .
```

Running `uv audit --frozen --python-version 3.12` directly is useful for full triage;
it exits non-zero when the current graph contains any finding.

## Exceptions And Limitations

Do not add a repository-wide ignored-advisory list merely to make a pull request
green. If an upstream constraint prevents an immediate fix, record the affected
profile, reachability, mitigation, upstream issue, and retest condition in the pull
request. The delta gate will continue to prevent unrelated new findings while that
work is tracked.

The audit is only as complete as the package metadata and advisory service. In
particular, a Git dependency may not have package-index advisory coverage. The cuRobo
source therefore remains pinned to a full immutable commit, and any commit change
requires source review and Simulation CI even when the audit is clean.
