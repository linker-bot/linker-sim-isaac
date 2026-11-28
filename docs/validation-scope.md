# Validation scope

LW-v0 provides a workstation model, controller models and task interfaces. Passing the
repository's tests establishes the properties those tests check. It does not establish
agreement with physical hardware or policy performance outside the tested distribution.

## What each check establishes

| Check | Evidence | Does not establish |
| --- | --- | --- |
| CPU controller tests | Numerical behavior for the tested inputs, limits and recorded outputs | Physical drive identification or behavior under all contacts |
| `tests/data/drive_reference.npz` | A six-second, two-environment GPU simulation recording used for drive-code regression | A recording from a physical AR5 or O6 |
| GPU acceptance tests | Execution and assertions on the recorded software and hardware configuration | Other GPUs, architectures, package versions or real-world performance |
| Gallery interaction tests | Viewer controls and scene transitions | A grasp success rate |
| Grasp evaluation | Per-trial outcomes under its recorded inputs and success criteria | Camera-based perception, untested objects or real-world transfer |

The drive recording's source and contents are described in
[`tests/test_drive_reference.py`](../tests/test_drive_reference.py). Tests reproduce the
controller's recorded outputs from its recorded inputs. A physical validation dataset
must be collected separately.

## State and collision coverage

The default policy observation contains 52 joint positions and velocities. The arm
trajectory generator, hand motors, managers and solver retain additional state. The
observation therefore does not determine the next state by itself. Tasks should choose
their inputs deliberately; [the environment guide](environment.md#observations-resets-episodes)
shows an observation/action-history configuration. Exact continuation has a separate
contract in [the snapshot API](api.md#linker_labsnapshot).

Collision filtering inside an arm unit retains contacts between different fingers, but
disables other same-articulation pairs. Full arm self-collision and hand-to-arm collision
avoidance are not provided. A motion that is feasible under joint and wrist limits still
requires collision checks against the physical robot and its surroundings.

The default camera sensor is shifted 6 mm forward of its optical frame to clear the
housing in the renderer. Intrinsics and optical-centre pose are different parts of camera
calibration. Use the sensor's actual pose when comparing depth or projecting points;
do not interpret the offset as a measured calibration error bound.

## Policy evidence

The `ycb-grasp` entry uses complete object geometry and simulation state. It performs one
actor evaluation and then executes a grasp open loop. It does not include a camera
perception pipeline. The manifest's 31 objects and resting poses were selected using a
reported training-side success threshold; that statement does not evaluate the full YCB
set or the current runtime.

The distributed artifacts identify the weights and object data by SHA-256. Training code,
the training dataset preparation and the underlying evaluation records are not included
here. Until those records are supplied, the training history in the manifest is provenance
information rather than independently reproducible evidence.

A new policy result should record:

- Source commit, policy and object artifact hashes, dependency lock, GPU and driver.
- Object list, resting-pose selection, placement bounds, random seeds and trial count.
- The success criterion, including required lift, hold duration and forbidden contacts.
- Every trial outcome, including planning failures, unstable placements and timeouts.
- Per-object and aggregate successes with their denominators; the sampling rule and any
  exclusions must be stated before interpreting the result.

An evaluation of the selected gallery objects should be labeled as such. Claims about a
larger population require an evaluation that covers that population. Failed placements or
plans must not disappear from the denominator of an end-to-end success rate.

## Physical validation

Before claiming agreement with hardware, record synchronized commands, measured joint
positions and velocities, and available torque/current measurements for the physical
workstation. Include free motion, reversals, motion near limits, loaded motion and contact
tests. Identify the robot, firmware, controller mode, gains, payload and measurement rates.

Replay the same commands with the same payload assumptions. Report tracking errors,
latency, velocity/acceleration limits, contact forces and grasp outcomes separately, with
the measurement uncertainty and acceptance thresholds. Include unsuccessful trials.
The repository currently supplies no such hardware dataset, so its regression fixtures
cannot substitute for this comparison.

## Dependency changes

The supported runtime is the pinned Isaac Lab/Newton stack in `pyproject.toml` and
`uv.lock`. [`engine.py`](../src/linker_lab/engine.py) applies compatibility patches before
Isaac Lab imports simulation modules. These patches cover this rigid-body application;
Newton-native actuators and deformable bodies are outside the supported configuration.

For a dependency upgrade, rerun bootstrap checks, CPU controller and snapshot tests,
installed-wheel checks, and GPU/viewer acceptance. Compare recorded physics and rendering
results before accepting the change. Remove an obsolete patch only after the corresponding
upstream behavior has been verified. A successful package resolution is not a simulation
compatibility test.

## Licenses

The repository code is Apache-2.0. The gallery manifest separately identifies the policy
weights as research-only, non-commercial, and the YCB object data as CC BY 4.0. The code
license does not change those artifact terms. Training reproducibility and commercial
reuse require the relevant artifacts, provenance and permissions; neither follows from
being able to run the viewer.
