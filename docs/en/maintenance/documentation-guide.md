# Documentation Maintenance Guide

Language: [English](documentation-guide.md) | [中文](../../zh-CN/maintenance/documentation-guide.md)

The documentation must describe the current Mirror and Kaleidoscope contracts, not
internal migration history or removed entrypoints.

## Reader Path

Every new user should be able to answer, in order:

1. Which product owns the capability?
2. Which public interface should the consumer use?
3. Which profile owns the configuration fact?
4. What device/thread/resource boundary applies?
5. How is the behavior validated and shut down?

Keep [Project Overview](../getting-started/project-overview.md) and
[Mode And API Chooser](../getting-started/choose-runtime-and-api.md) as the first
decision pages.

## Fact Owners

| Fact | Documentation owner |
| --- | --- |
| Product capability boundary and call flow | Project Overview |
| First executable workflow | Product quickstart |
| Mirror process options | Mirror CLI Reference |
| Mirror request/response schema | Mirror JSON Protocol |
| Kaleidoscope tensors, state, cloning, adapters | Kaleidoscope API Reference |
| Exact profile fields and invariants | Configuration Reference |
| Planning concepts and tradeoffs | Motion Planning guide |
| Snapshot type/restore semantics | Snapshots reference |
| Stable import surface | Python API Reference |
| Source ownership and thread/device tags | Generated Module Map |

Other pages should link to the owner instead of duplicating large tables.

## Product Language

Use **Mirror** for the reality-mapped one-world product and **Kaleidoscope** for the
GPU vector reinforcement-learning product. State explicitly when a capability is
deliberately absent. Do not create compatibility redirect pages or list removed names
as alternate terminology.

## Code And Configuration Accuracy

Before editing a reference:

- inspect the current facade, typed configuration class, parser, or method signature;
- distinguish public facade from internal implementation;
- distinguish physical contacts from planning collision queries;
- distinguish native CUDA, Gymnasium NumPy, and skrl CUDA boundaries;
- verify backend and shutdown ownership.

Do not infer a CLI switch from a YAML field or document an internal helper as a public
API.

## Links And Filenames

Use product-qualified filenames such as `mirror-json.md` and
`kaleidoscope-api.md`. Relative links must resolve inside the repository. When a page
is replaced during a breaking change, update every inbound link and delete the old
page; do not leave a redirect stub.

Run:

```bash
PYTHONPATH=src .venv-dev/bin/python scripts/check_markdown_links.py
```

The repository gate also requires link targets to be tracked by Git.

## Examples

- Use commands that run from the repository root.
- Include `PYTHONPATH=src` where the surrounding context does not already export it.
- Use strict JSON with unique request IDs and finite values.
- Put CUDA tensors directly on `env.device` in native examples.
- Close runtimes in `finally` or a context manager.
- Never show a Kaleidoscope mode-level render, transport, telemetry, or planning
  switch. Human display must use the standalone viewport profile and must not be
  described as a camera/SyntheticData/Replicator API.

## Bilingual Structure

English and Chinese trees should contain the same maintained relative paths and
navigation shape. Translate explanations, but keep code, option names, operation
names, field names, and factual tables equivalent. The generated module maps must
have identical inventory facts.

## Review Checklist

- All new links resolve.
- Removed product terminology is absent outside generated inventory when source paths
  still require it.
- Mode capability tables agree across overview, chooser, and references.
- CUDA ownership is `mode.compute.cuda_device` only.
- Mirror and Kaleidoscope physics backends are not conflated.
- Kaleidoscope state, snapshot, and clone remain GPU-resident.
- Batch IK and synchronous linear motion are distinguished from trajectory planning.
- Shutdown order and fail-stop behavior are documented where relevant.
- New or materially changed implementation code includes useful Chinese comments as
  required by the project contribution policy.
