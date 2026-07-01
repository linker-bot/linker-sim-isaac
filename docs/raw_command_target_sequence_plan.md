# Raw Command Target Sequence Plan

This plan is written for a code-modifying model. Follow the contract here before
changing implementation details.

## Goal

Add an interactive dual-arm mode that directly refreshes controller
command-space joint angle targets on physics steps. This mode is for external
real-time controllers or policy rollouts that already produce discrete joint
targets and do not want cuMotion planning, smoothstep retiming, or generated
acceleration profiles.

## Semantics

- The new command is explicit raw target playback, not a planned motion.
- Input targets are controller command-space joint positions, in radians.
- Each target sample is applied as a position target and then the Isaac world is
  advanced by one physics step.
- `step_interval` controls target refresh cadence:
  - `1`: apply the next sample every physics step.
  - `N > 1`: apply a sample, hold that same target for `N` physics steps, then
    advance to the next sample.
- The implementation does not interpolate between samples, retime them, smooth
  them, or enforce velocity/acceleration limits.
- Missing side means hold that side at the current command-space position for the
  same sample count.
- This mode should reuse `JointController.build_control_targets(...)` so mimic
  followers and partial command-space behavior remain centralized in the
  controller layer.

## Protocol Shape

Add a new interactive command type:

```json
{
  "type": "raw_joint_sequence",
  "left": {
    "joint_positions": [[0.1, 0.2], [0.11, 0.21]]
  },
  "right": {
    "joint_positions": {"joint_a": [0.3, 0.31]}
  },
  "step_interval": 1,
  "phase": "policy_step"
}
```

Side payload rules:

- `joint_positions` may be a matrix `samples x command_dof`. It must match the
  full command-space width for that side.
- `joint_positions` may also be a mapping of joint name to a sequence of sample
  values. Mapping samples may cover a subset of command-space joints; omitted
  command joints hold their current command value.
- At least one side payload is required.
- All provided sides must have the same number of samples.
- `step_interval` is optional and defaults to `1`; it must be a positive integer.

## Implementation Steps

1. Add `RawJointSequenceMoveSpec` to `src/linkerbot_sim/app/motion/specs.py` and
   include it in `MoveSpec`.
2. Extend `src/linkerbot_sim/app/interactive/protocol.py` so
   `type="raw_joint_sequence"` parses into `RawJointSequenceMoveSpec`.
3. Add low-level execution support in
   `src/linkerbot_sim/execution/dual_steps.py`:
   - add a `DualRawCommandTargetSequenceStep` dataclass;
   - add a function that builds targets for each side sample and advances the
     world exactly once per physics step;
   - support interruption through the existing `should_stop` callback;
   - zero articulation velocities in `finally`, matching trajectory playback.
4. Wire the move into `src/linkerbot_sim/app/motion/dual_arm.py` before the
   cuMotion-specific branch:
   - normalize matrix or mapping side payloads into full command-space sample
     matrices;
   - hold missing side commands;
   - return updated left/right command vectors from the final sample;
   - update merged cuMotion C-space from the resulting command vectors so later
     planned moves start from the raw target state.
5. Document the JSON command in `docs/interactive_simulation_usage.md`.
6. Add focused unit tests:
   - protocol parsing for matrix and mapping payloads;
   - low-level execution applies one target per physics step and honors
     `step_interval`;
   - dual-arm runtime helper expands mapping payloads and rejects sample count
     mismatch if practical without Isaac.

## Non-Goals

- Do not add collision checking, IK, cuMotion, or dynamic feasibility checks.
- Do not change existing `CommandPositionTrajectoryStep` semantics.
- Do not replace hand/arm motion commands; this is a separate explicit mode.
