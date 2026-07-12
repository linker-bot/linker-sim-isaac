# Documentation Organization And Commit Guide

Language: [English](documentation-guide.md) | [中文](../../zh-CN/maintenance/documentation-guide.md)

This guide defines the content boundary, information architecture, fact ownership,
and review requirements for `docs/`. Its purpose is to give human users and language
models one route for selecting a runtime and invocation surface, followed by links to
one complete, source-aligned owner for each fact.

## 1. Scope And Audience

Project documentation serves three audiences:

- Application users who control simulation through CLI, YAML, and JSON.
- Algorithm and tooling developers who directly call explicitly documented Python facades.
- Maintainers who change runtime, planning, snapshots, telemetry, cameras, assets, or configuration.

Repository-root README files own only project positioning, minimal environment setup, minimal launch
commands, and documentation entry points. Complete field tables, state machines, data
structures, defaults, and error semantics belong to one corresponding reference page
under `docs/`.

## 2. User-Dependable Interface Boundary

Users may depend only on the following explicitly documented surfaces:

- The Single Scene, Tiled Scene, and configuration-validation CLIs:
  `scripts/single_scene_interactive.py`, `scripts/tiled_scene_interactive.py`, and
  `scripts/validate_config.py`.
- The documented offline asset-tool CLIs for the capsule rope and T block
  `build_asset.py` entrypoints.
- Project YAML profile structures, ownership, validation rules, and CLI overlay priority
  listed by the configuration guide and reference.
- Requests, responses, selectors, state machines, and transport semantics listed by the
  Single Scene and Tiled Scene JSON references.
- Facades, functions, types, parameters, return values, exceptions, and lifecycle
  constraints explicitly named by the Python reference.

The complete module map under `src/linkerbot_sim/` exists for implementation navigation;
it is not a declaration that every module is a public Python API. The existence of a
module, a symbol without a leading underscore, or an `__all__` declaration is not by
itself evidence of a user interface. Modules, classes, and functions not explicitly
listed by the Python reference are internal implementation. Maintenance documentation
may explain their responsibilities and call graph, but must not present them as stable
entry points on which users can depend.

Python pages must label each listed entry point with its runtime prerequisite:

- `pure`: importable in the ordinary project Python environment without starting Isaac Sim.
- `Isaac main thread`: callable only after Kit/Isaac startup and under the documented thread conditions.
- `cuRobo/CUDA`: requires the project-pinned GPU, Torch, Warp, and cuRobo environment.

This project is a checkout workspace application, not an installable Python library.
Python examples run from the repository root with `PYTHONPATH=src`; that execution model
does not broaden the user interface boundary.

## 3. Runtime Forms And Call Paths

Single Scene means one `SingleSceneRuntime`; it does not mean that a scene can contain only
one robot. Tiled Scene means parallel environments cloned from a homogeneous template, with
env-selective control, state, trajectory, and planning interfaces.

```text
optional preflight: runtime YAML + referenced profiles
  -> validate_config strict parsing and complete profile-graph validation
  -> validation report; no Isaac startup

runtime CLI: runtime YAML + referenced profiles + explicit CLI overlay
  -> strict effective-runtime resolution
  -> SingleSceneRuntime or Tiled Scene runtime composition

Single Scene JSON client
  -> stdin / TCP JSONL / WebSocket
  -> Single Scene protocol and timeline compiler
  -> SingleSceneRuntime
  -> controllers, execution and Isaac World

Tiled Scene JSON client
  -> stdin / TCP JSONL / WebSocket
  -> Tiled Scene protocol, selectors and command routing
  -> control, state, trajectory playback or asynchronous planning
  -> Isaac batched views and optional cuRobo services

Documented Python caller
  -> one explicitly documented domain facade
  -> its declared runtime and resource boundary
```

JSON is the process-level control protocol. A Python facade is an in-process domain call.
YAML describes launch configuration and scene facts; it is not a motion command. The
documentation entry point must first help readers choose Single Scene or Tiled Scene, then JSON or
Python, without collapsing those two decisions into one dimension.

## 4. Document Types And Directory Responsibilities

The English and Chinese trees use the same ASCII relative paths and the same topic boundaries:

| Directory | Responsibility |
| --- | --- |
| `getting-started/` | Project overview, runtime and interface selection, and complete minimal walkthroughs |
| `guides/` | Task-oriented configuration, planning, telemetry, camera, and operational instructions |
| `reference/` | Exhaustive CLI, JSON, configuration, and data-structure contracts |
| `operations/` | Diagnostics, runtime constraints, security boundaries, and capacity boundaries |
| `development/` | Source-domain navigation, naming, asset generation, collision, and preview tooling |
| `maintenance/` | Documentation ownership, commit boundaries, and review policy |

The opening or first relevant section must make the page's scope and required runtime
prerequisites clear. Tutorials may include minimal examples but must not copy complete
field tables from references. References must state exact names, types, units, shapes,
frames, defaults, terminal states, and rejection conditions.

Document types have these acceptance boundaries:

- A quickstart must form one executable chain from checkout setup and complete
  configuration validation through EULA acceptance, readiness, discovery, one minimal
  valid operation, terminal-result verification, and normal process exit.
- A CLI reference must cover every parser option, including both forms of paired Boolean
  flags. It distinguishes the argparse value when an option is omitted from the effective
  value supplied by the bundled profile.
- A JSON reference owns framing, messages, selectors, state transitions, and responses.
  It links to the CLI and configuration owners instead of copying their complete tables.
- Selector inventories come from the parser and dispatcher, including every command that
  requires an explicit environment selector. A numeric disable sentinel and rejected
  values must be described separately rather than collapsed into one range.

The source module map groups modules by domain and records responsibility, runtime
prerequisite, and related documentation. It must cover every `src/linkerbot_sim/**/*.py`
module while labeling each entry as a documented facade, an explicitly supported owner
path, or internal implementation, so navigation coverage cannot be mistaken for an API
commitment.

## 5. Single Documentation Owner For Each Fact

One detailed documentation owner must hold each field table, default, state machine, or
persistent format. Other pages retain only the minimum example needed for their task and
link to that owner.

| Fact | Source owner | Current documentation owner |
| --- | --- | --- |
| Single Scene launch options, config overlays, and process markers | `app.interactive.single_scene.cli` | [Single Scene CLI Reference](../reference/single-scene-cli.md) |
| Single Scene transport, commands, timeline, and responses | `app.interactive`, `app.motion.timeline` | [Single Scene JSON Reference](../reference/single-scene-json.md) |
| Tiled Scene launch options, config overlays, and process markers | `app.interactive.tiled_scene.cli` | [Tiled Scene CLI Reference](../reference/tiled-scene-cli.md) |
| Tiled Scene transport, selectors, step, state, trajectory, and planner | `app.interactive.tiled_scene`, `tiled` | [Tiled Scene JSON Reference](../reference/tiled-scene-json.md) |
| Control-path selection, timing, joint order, and playback lifecycle | `app.motion.timeline`, `tiled.control`, `tiled.playback` | [Control And Trajectory Guide](../guides/control-and-trajectories.md) |
| Profile layers, references, CLI overlays, and common configuration tasks | `configs.runtime`, domain config modules | [Configuration Guide](../guides/configuration.md) |
| Project YAML fields, per-env fragments, ownership, and complete graph validation | `configs`, domain config modules | [Configuration Reference](../reference/configuration.md) |
| Planning backends, frames, cuRobo binding, batching, and collision capability | `planning`, `backends.curobo` | [Motion Planning Guide](../guides/motion-planning.md) |
| PhysX, planning, and inter-env collision-layer selection | asset importers, planning collision providers, `tiled.scene.collision_filter` | [Collision Models](../guides/collision-models.md) |
| Snapshot data, identity matching, capture/restore, transactions, and failure semantics | `snapshots` | [Snapshot Reference](../reference/snapshots.md) |
| Telemetry topics, payloads, env selection, sampling, and live publication semantics | `telemetry` | [Telemetry Guide](../guides/telemetry.md) |
| Sensor cameras, frames, modalities, capture, and attachment semantics | `sensors.camera` | [Camera Guide](../guides/cameras.md) |
| CSV, MCAP, camera encodings and metadata, paths, existing-data policy, queues, quotas, and persistent-output shutdown | logging, telemetry, and camera output owners | [Output Reference](../reference/outputs.md) |
| Runtime security, threading, resources, simulation, and configuration constraints | Domain owners | [Runtime Constraints](../operations/constraints.md) |
| Symptom-to-owner diagnosis and recovery boundaries | Domain owners | [Troubleshooting](../operations/troubleshooting.md) |
| Asset, joint, link, body, and profile naming | `assets`, `robots`, `objects` | [Naming Rules](../development/naming.md) |
| Object asset generation and scene integration | `objects`, asset generation scripts | [Object Assets](../development/object-assets.md) |
| Importer collision approximation and USD/PhysX meaning | Asset importers | [Collision Approximation](../development/collision-approximation.md) |
| USD asset preview | Isaac asset tools | [USD Preview](../development/usd-preview.md) |

Python import paths, signatures, types, shapes, units, threading, resources, and
exceptions belong to one Python reference. For every resource-returning call, that page
also names the returned handle's supported shutdown method, the ownership boundary before
and after successful startup, the shutdown return value, and any retry condition. Detailed
domain state machines, matching behavior, timing policy, and persistent formats stay with
their domain owner and are linked rather than restated. The module map owns only
source-domain navigation.

## 6. English And Chinese Consistency

`docs/en/` and `docs/zh-CN/` use one-to-one paths, section responsibilities, and interface
semantics. Both languages must agree on:

- CLI names, options, defaults, and mutual exclusions.
- YAML and JSON fields, required conditions, types, enums, and rejection semantics.
- Python import paths, signatures, return types, shapes, units, frames, and lifecycle.
- Every current export of each documented facade and every explicitly supported advanced
  owner symbol; both language trees must expose identical symbol sets.
- Topics, file paths, payloads, metadata, capacity, and shutdown behavior.
- Error codes, terminal states, timeouts, backpressure, transactions, and fail-stop conditions.

Tutorial prose may follow each language's natural style, but it must not omit constraints
that change correct use. Every paired page should provide a language switch link, and both
indexes must expose the same descriptions and reading paths.

## 7. Content To Commit

The following are long-lived repository documentation and must be reviewed and committed
with corresponding code or configuration changes:

- README files, language indexes, getting-started pages, guides, references, operations,
  development, and maintenance pages.
- Current CLI, YAML, JSON, documented Python facade, and persistent-data contracts.
- Architecture boundaries, resource ownership, thread requirements, and runtime constraints
  that prevent misuse or regression.
- Executable, strictly parseable examples that match actual project entry points.
- The source module map with interface classification, runtime prerequisites, and fact-owner links.
- A colocated README that explains the purpose and boundary of a configuration resource.

Documentation describes the repository's current observable behavior. Interface inventories
and examples contain only names and fields accepted by the current implementation. They do
not record personal work logs, machine-local state, or implementation history.

## 8. Content Not To Commit

The following do not belong in `docs/` or Git:

- `docs/_build/`, `docs/tem/`, site output, and reproducible generated HTML/API pages.
- Session output produced by a runtime process, including `logs/`, MCAP, joint
  CSV, camera-frame directories, runtime `metadata.jsonl`, and the rest of the
  same run directory.
- Analysis, drafts, and experimental scripts under `design_plan/` that are not project contracts.
- Notebook checkpoints, editor files, tool caches, local agent state, and temporary files.
- Credentials, tokens, private addresses, user data, machine-specific absolute paths, and
  security-response records.
- Transient test counts, disk usage, personal execution timelines, and one-time audit output.

Generated API sites may be used as local or CI artifacts. The repository retains source
docstrings, a handwritten module map, and domain references rather than committing a
reproducible site.

The rule is based on provenance and ownership, not a global filename-extension ban.
Minimal data directly consumed by a test may be committed under `tests/data` or another
explicit fixture directory. A formal asset may be committed under `assets/` when its source
and purpose are documented. Do not rename a one-off run directory and present it as a fixture
or asset.

## 9. Documentation Checks For Source Changes

| Source or configuration change | Documentation owners to check |
| --- | --- |
| Single Scene CLI or overlay | Single Scene CLI reference and Single Scene quickstart |
| Single Scene parser, queue, transport, or timeline | Single Scene JSON reference, Single Scene quickstart, runtime constraints |
| Tiled Scene CLI or overlay | Tiled Scene CLI reference and Tiled Scene quickstart |
| Configuration validator parser, output, or exit behavior | Configuration reference and runtime chooser |
| Rope or T block asset-builder parser or output marker | Object-assets development guide |
| Tiled Scene selector, action, state, trajectory, or planner routing | Tiled Scene JSON reference, Tiled Scene quickstart, motion-planning guide |
| Runtime or domain YAML parser, defaults, validation, or CLI overlay | Configuration guide, configuration reference, example profiles |
| Planning request/result, frame, joint order, shape, or unit | Motion-planning guide; also the Python reference when a documented facade is affected |
| Snapshot capture, restore, identity, or transaction | Snapshot reference and Single Scene/Tiled Scene envelope sections |
| Telemetry topic, payload, sampling, backpressure, or MCAP | Telemetry guide and output reference |
| Camera frame, encoding, recorder, sink, path, or capacity | Camera guide and output reference |
| CSV columns, sampling, flushing, or existing-file policy | Output reference and configuration guide |
| Threading, queues, timeouts, shutdown, or fail-stop | Corresponding runtime reference and runtime constraints |
| Documented Python facade import, signature, type, exception, or resource ownership | Python reference and module map |
| Internal module addition, removal, movement, or responsibility change | Module map; Python reference only when the module contains a documented facade |
| Asset importer, naming, collision, or preview tool | Corresponding development page and configuration guide |

Code review must follow fact ownership across externally observable behavior rather than
checking only documentation files whose names resemble the changed source files.

## 10. Pre-Commit Verification

Run from the checkout root:

```bash
git status --short --untracked-files=all README*.md docs configs
git diff --check -- README*.md docs configs
PYTHONPATH=src .venv/bin/python scripts/check_markdown_links.py
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_markdown_links.py \
  tests/test_documented_module_map.py \
  tests/test_documentation_contracts.py
.venv/bin/python scripts/validate_config.py --runtime-profile default_single_scene
.venv/bin/python scripts/validate_config.py --runtime-profile default_tiled_scene
```

Manual review must also confirm:

- Every index and inline link targets a Git-tracked file, with paired language links.
- JSON/JSONL examples are strict JSON; YAML examples have no duplicate keys and match the current parser.
- CLI tables cover all paired flags, match the corresponding entry point's `--help`, and
  distinguish omitted parser values from bundled-profile effective values.
- Quickstarts complete the full validation, startup, discovery, operation, terminal, and
  shutdown chain; protocol selector inventories match their parsers and dispatchers.
- Python examples match source imports, signatures, shapes, units, frames, thread rules, and shutdown behavior.
- Python references name every facade export and supported owner symbol, and document
  startup-failure ownership plus the exact shutdown method and result for returned handles.
- Each detailed fact has one documentation owner; other pages link rather than copying the full definition.
- The module map covers every source module and correctly distinguishes documented facades, supported owner paths, and internal implementation.
- English and Chinese field, default, error, and lifecycle semantics agree.
- Documentation contains no credentials, machine-local data, run output, or irreproducible personal-environment conclusions.
