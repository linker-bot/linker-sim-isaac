# Changelog

Language: [English](CHANGELOG.md) | [中文](CHANGELOG_zh.md)

All notable user-visible changes are recorded here. The project uses semantic
versions for its workspace contract; exact development revisions are identified by
their Git commit.

[Unreleased]: https://github.com/linker-bot/linker-sim-isaac/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/linker-bot/linker-sim-isaac/releases/tag/v0.3.0

## [Unreleased]

### Changed

- The GPU/Isaac `Simulation` workflow is temporarily manual-only while the
  self-hosted runner is stabilized.
- A maintainer-only release workflow now verifies an annotated version tag, CPU
  quality, and a successful Simulation run from the same commit before publishing a
  checksummed source workspace archive.
- Public contribution entry points now include a pull-request template and a support
  routing guide.
- The declared default-branch ruleset policy file is present for its tests and
  scheduled drift audit.

### Fixed

- Kaleidoscope PhysX CUDA seeded resets now restore native mimic-follower joint
  positions and velocities, then refresh derived articulation link poses. Repeating
  a seed is independent of the preceding episode's joint history.

## [0.3.0] - 2026-08-26

### Added

- Mirror provides a single-world reality-replay product with strict versioned JSON
  protocols, explicit runtime ownership, planning, cameras, telemetry, and bounded
  shutdown behavior.
- Kaleidoscope provides PhysX CUDA and project-owned Newton multi-world training
  backends with CUDA-resident state, snapshots, cloning, batched IK, Gymnasium, and
  skrl adapters.
- CPU quality, static typing, pure-module coverage, dependency audit, architecture
  inventory, repository ruleset, and dedicated GPU/Isaac acceptance contracts are
  maintained in the repository.
- Runtime and project metadata expose the same workspace version, while diagnostics
  and support requests retain the exact Git commit.

[中文变更记录](https://github.com/linker-bot/linker-sim-isaac/blob/v0.3.0/CHANGELOG_zh.md)
