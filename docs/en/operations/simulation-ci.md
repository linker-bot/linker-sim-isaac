# Simulation CI

Language: [English](simulation-ci.md) | [中文](../../zh-CN/operations/simulation-ci.md)

The `Simulation` GitHub Actions workflow runs the maintained GPU/Isaac acceptance
matrix on a dedicated NVIDIA host. It does not replace the GitHub-hosted CPU `Quality`
workflow: the two jobs use incompatible Python environments and prove different
contracts.

## Trigger And Trust Boundary

The workflow runs after a relevant path is pushed to `master`, or when a maintainer
explicitly dispatches it. It has no `pull_request` or `pull_request_target` trigger.
This repository is public, and unreviewed pull-request code must never execute on a
self-hosted runner that may retain machine state.

For pre-merge evidence, open **Actions → Simulation → Run workflow**, select a reviewed
branch that belongs to this repository, and record the run URL in the pull request.
Fork code must first pass review and be copied to a trusted in-repository branch.

## Runner Contract

The runner must have every label below:

| Label | Contract |
| --- | --- |
| `self-hosted` | The host is maintained outside GitHub-hosted infrastructure. |
| `linux` | Isaac runs on Linux. |
| `x64` | The pinned wheel graph targets x86-64. |
| `nvidia-gpu` | `nvidia-smi` exposes a compatible NVIDIA GPU and driver. |
| `isaac-sim` | The host is dedicated to this repository's Isaac workload. |

Use an organization runner group restricted to this repository and, when the policy
is available, `.github/workflows/simulation.yml`. Prefer an ephemeral or resettable
machine. Persistent package and Isaac caches may live outside the Actions workspace,
but credentials, unrelated secrets, and mutable source checkouts must not.

The host also needs enough disk for Isaac wheels and extension caches. If a test needs
the optional licensed Warehouse payload, provision it outside Git and expose the exact
layout documented in [Installation](../getting-started/installation.md); the workflow
does not download or redistribute that content.

## Repository Environment

Create a GitHub environment named `simulation`. After reviewing NVIDIA's terms, add
the non-secret environment variable:

```text
OMNI_KIT_ACCEPT_EULA=Y
```

The workflow fails before dependency installation when the variable is absent or has
another value. No repository write permission or Git credential is retained by the
checkout step.

## Executed Gate

The workflow:

1. verifies Linux x86-64, the EULA decision, `nvidia-smi`, and the dependency lock;
2. syncs `simulation`, `visualization`, `training`, and `test` without the incompatible
   `dev` extra;
3. verifies the pinned Python, pytest, coverage, Torch, CUDA, and GPU stack;
4. runs `bash ci/simulation.sh`, the same entrypoint used by maintainers.

The job has a four-hour limit. In-progress jobs are not force-cancelled by concurrency
because terminating Kit in the middle of native teardown can leave a persistent runner
dirty. A failed or timed-out run is not acceptance evidence; inspect the first failing
recipe, reset the runner when native processes survive, and rerun the complete matrix.
