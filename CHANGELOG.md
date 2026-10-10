# Changelog

Language: [English](CHANGELOG.md) | [中文](CHANGELOG_zh.md)

All notable user-visible changes are recorded here. The project uses semantic
versions for its workspace contract; exact development revisions are identified by
their Git commit.

[Unreleased]: https://github.com/linker-bot/linker-sim-isaac/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/linker-bot/linker-sim-isaac/releases/tag/v0.3.0

## [Unreleased]

- Added common L6/O6 flanges, optional Gemini 335L wrist assemblies and a top ZED module with named optical mounts and separate camera presets. Corrected USD pose readback and preserved live root rotation topology during snapshot restoration.

### Changed

- Set the Newton CPU/CUDA contact-response default to 10 ms in profiles and programmatic specs. Keep the 2 ms internal timestep bound, per-substep contact refresh and explicit source settings; revalidate task contact behavior after changing this numerical default.

- Move the supported runtime to Isaac Sim 6.1.0, Newton 1.5, Warp 1.16 and MuJoCo 3.11. Newton controls use the new target fields with an explicit DOF-layout requirement; model notifications and control snapshots follow the matched runtime. Recreate the simulation environment from the updated lockfile.

- Mirror now samples only due cameras with output consumers; `step(render=True)` permits scheduled acquisition rather than forcing every sensor each step. `render(camera_ids=...)` captures fresh selected cameras without advancing physics or recording. Idle sensor products stop rendering, and pause, estop and continuous queries service GUI independently.

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

- Preserve the configured Newton contact fallback through Isaac 6.1 MJCF conversion. Prepare an owned temporary source with the missing default, retaining explicitly authored and inherited `solref` values; source assets and PhysX imports are unchanged.

- Repair Isaac 6.1 PhysX imports so MJCF mimic targets and coefficients survive physics-variant routing, and existing MJCF/URDF world anchors carry the articulation-root metadata. Fixed camera assemblies stay fixed during integration. Task-space binding follows exact body relationships when the root is a joint.

- Renamed the AR5 arm-hand default `AR5V2_L/R_pinch_tcp` frames to
  `AR5V2_L/R_flange_tcp` to describe their zero-offset arm-flange reference.
  Explicit TCP selectors must use the new names; no old-name aliases remain.
  Frame transforms, physics, and separately configured task TCPs are unchanged.

- Newton refreshes contacts on every internal step and bounds internal timesteps to 2 ms by default; unauthored contacts use the configured response, now defaulting to 10 ms. Outer control/render clocks and authored per-shape contact parameters are preserved; diagnostics report effective integration settings.
  Existing Newton physics YAML leaves must add `max_substep_dt_s` and `default_contact_time_constant_s`; increased internal integration reduces throughput and changes contact trajectories.

- Joint effort telemetry identifies commanded, applied and projected sources,
  per-joint validity and missing-data reasons using the actual physics clock.
  Newton projected effort remains explicitly unsupported; gains, gravity and armature are unchanged.

- Planning now uses the measured fixed hand shape and mounted hardware, imported fixture bounds and carried-object geometry. Corrected world-to-base obstacle conversion, added scoped contact allowances and sampled path coverage diagnostics.

- Remove the USD parser serialization and Warp `func(module=...)` workaround now covered by the matched runtime. Newton equality audits read the MuJoCo namespace and preserve the native joint-equality executor; unauthored contact defaults no longer require resolver patches.

- Mirror cameras now refresh paused poses and headless captures using native frame completion, without stepping physics; records retain native render identity and stale frames are not retimestamped.

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
