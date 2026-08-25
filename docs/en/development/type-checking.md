# Static Type Checking

Language: [English](type-checking.md) | [中文](../../zh-CN/development/type-checking.md)

The repository uses Pyright as a required CPU development gate. Run it directly with:

```bash
just type-check
```

`just quality` includes the same check. The command uses the pinned `dev` environment
and `pyrightconfig.ci.json`, so local and CI runs resolve the same Python 3.12 package
set.

## Configuration And Scope

The two Pyright files have different responsibilities:

- `pyrightconfig.json` keeps broad `src`, `scripts`, and `tests` discovery for editors.
- `pyrightconfig.ci.json` defines the required zero-diagnostic baseline for the
  configuration package, dependency and documentation checks, pure-coverage and
  architecture inventory tools, mode validation, and the workspace build backend.

The CI configuration uses standard checking and does not contain a global `ignore` or
disabled `report*` rule. The one line-local exception is the computed `__all__` in the
lazy `linkerbot_sim.configuration` facade. Architecture tests independently freeze and
verify that public export surface.

## Expanding The Gate

Add a module or directory to `pyrightconfig.ci.json` only after it reports zero
diagnostics in the CPU `dev` environment. Fix or narrow real type boundaries instead
of globally disabling a diagnostic. Keep a line-local exception only when the runtime
contract is intentional, narrowly documented, and independently tested.

The current scope deliberately avoids modules whose imports and types are owned by
Isaac Sim, Kit, CUDA, or other simulation-only distributions. Expanding into those
areas requires compatible runtime packages or maintained stubs. Static analysis does
not replace the simulation and GPU gates described in
[Simulation CI](../operations/simulation-ci.md).
