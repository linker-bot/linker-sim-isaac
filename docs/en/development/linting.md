# Lint And Format Policy

Language: [English](linting.md) | [中文](../../zh-CN/development/linting.md)

Ruff provides the repository's Python lint and format gates:

```bash
just lint
just format-check
```

Both commands are included in `just quality` and run with the release pinned by the
`dev` dependency group.

## Stable Rule Contract

`pyproject.toml` explicitly selects `E4`, `E7`, `E9`, and `F` rules and targets Python
3.12. The selection preserves the accepted repository baseline independently from
Ruff's release defaults. A dependency update must not silently add or remove policy.

Expand the selected rules in a focused change that fixes the resulting diagnostics,
documents intentional exceptions, and leaves `just quality` green. Do not hide a new
rule family behind a repository-wide ignore.

The format gate is scoped to Python source. Markdown maintenance remains governed by
the bilingual-tree and local-link contracts described in
[Documentation Maintenance](../maintenance/documentation-guide.md); Ruff does not
rewrite fenced examples as an implicit formatter-upgrade side effect.
