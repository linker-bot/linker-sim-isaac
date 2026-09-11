# linker-sim-isaac Documentation

Language: [English](index.md) | [中文](../zh-CN/index.md)

This documentation describes the current Mirror and Kaleidoscope product contracts.
The migration is intentionally breaking: removed names, scripts, configuration roots,
and message formats are not supported aliases.

## Start Here

| Goal | Read |
| --- | --- |
| Install the pinned environments and prepare optional assets | [Installation](getting-started/installation.md) |
| Understand the two products and their ownership boundaries | [Project Overview](getting-started/project-overview.md) |
| Choose between JSON, native Torch, Gymnasium, and skrl | [Mode And API Chooser](getting-started/choose-runtime-and-api.md) |
| Write joint, IK, planning, and synchronized motion JSON | [Mirror JSON Protocol And Motion Examples](reference/mirror-json.md) |
| Operate a reality-mapped world | [Mirror Quickstart](getting-started/mirror-quickstart.md) |
| Run a GPU vector environment | [Kaleidoscope Quickstart](getting-started/kaleidoscope-quickstart.md) |

## Guides

- [Configuration](guides/configuration.md)
- [Control And Trajectories](guides/control-and-trajectories.md)
- [Motion Planning](guides/motion-planning.md)
- [Collision Models](guides/collision-models.md)
- [Cameras](guides/cameras.md)
- [Telemetry](guides/telemetry.md)
- [Outputs](reference/outputs.md)
- [Foxglove](guides/foxglove.md)

Mirror owns every guide above. Kaleidoscope deliberately omits planning, avoidance,
camera, transport, telemetry, and persistent output from its hot runtime closure.

## References

- [Mirror CLI](reference/mirror-cli.md)
- [Mirror JSON Protocol](reference/mirror-json.md)
- [Kaleidoscope API](reference/kaleidoscope-api.md)
- [Python API](reference/python-api.md)
- [Configuration Schema](reference/configuration.md)
- [Snapshots And State](reference/snapshots.md)
- [Output Policy](reference/outputs.md)

## Operations

- [Runtime Constraints](operations/constraints.md)
- [Dependency Security And Updates](operations/dependency-security.md)
- [Repository Governance](operations/repository-governance.md)
- [Releases](operations/releases.md)
- [Simulation CI](operations/simulation-ci.md)
- [Troubleshooting](operations/troubleshooting.md)

## Development

- [Contributing](../../CONTRIBUTING.md)
- [Support](../../SUPPORT.md)
- [Changelog](../../CHANGELOG.md)
- [Naming And Ownership](development/naming.md)
- [Module Map](development/module-map.md)
- [Lint And Format Policy](development/linting.md)
- [Static Type Checking](development/type-checking.md)
- [Version And Revision Identity](development/versioning.md)
- [Object Assets](development/object-assets.md)
- [Collision Approximation](development/collision-approximation.md)
- [USD Preview](development/usd-preview.md)
- [Documentation Maintenance](maintenance/documentation-guide.md)
