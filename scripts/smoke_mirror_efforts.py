#!/usr/bin/env python3
"""Verify independent effort sources during real commands, cancellation and reset."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from linkerbot_sim.configuration import load_mirror_config  # noqa: E402
from linkerbot_sim.mirror import create_mirror_runtime  # noqa: E402
from linkerbot_sim.mirror.motion.timeline.executor import TimelineExecutionInterrupted  # noqa: E402
from linkerbot_sim.telemetry.state_snapshot import SceneRobotStateSampler  # noqa: E402
from scripts.runtime_worker_supervisor import in_runtime_worker, run_supervised_worker  # noqa: E402

MARKER = "LINKERBOT_MIRROR_EFFORT_RUNTIME_OK"


def run(profile):
    config = load_mirror_config(profile)
    output = config.outputs
    config = replace(
        config,
        scene=replace(config.scene, cameras=(), planning_startup="lazy"),
        control=replace(config.control, sync_simulation_to_wall_clock=False),
        outputs=replace(
            output,
            render=replace(output.render, enabled=False, gui=False),
            camera=replace(output.camera, enabled=False),
            telemetry=replace(output.telemetry, enabled=False),
        ),
    )
    runtime = create_mirror_runtime(config)
    try:
        resources = runtime.scene_resources
        robot = resources.robots_by_id[0]
        controller = robot.execution.joint_controller
        names = tuple(robot.execution.articulation.dof_names)
        joint = robot.joint_groups.names("arm")[0]
        index = names.index(joint)
        sampler = SceneRobotStateSampler(stage=None, include_efforts=True)
        reports = []

        def sample():
            snapshot = sampler.sample(resources, step=0)
            item = next(r for r in snapshot.robots if r.robot_id == robot.robot_id)
            assert snapshot.time_s == runtime.physics_runtime.simulation_time
            return item

        initial = sample()
        modes = initial.effort_metadata["control_modes"]
        if modes[index] == "position/implicit":
            assert not initial.effort_metadata["commanded"]["valid"][index]
        runtime.set_control_mode(
            "effort", expected_generation=runtime.get_control_mode().generation
        )
        dt = resources.physics.get_physics_dt()
        for effort, cancel in ((1.0, False), (-1.0, False), (0.0, False), (1.0, True)):
            samples = []

            def observe_and_cancel():
                current = sample()
                samples.append(current)
                return cancel and len(samples) >= 5

            try:
                runtime.motion.execute(
                    "motion.joint_effort",
                    {
                        "robot_id": robot.robot_id,
                        "group": "arm",
                        "duration_s": 8 * dt,
                        "joint_efforts": {joint: effort},
                    },
                    request_id=f"effort-{effort}-{cancel}",
                    should_cancel=observe_and_cancel,
                    protocol="linkerbot.mirror.v2",
                )
            except TimelineExecutionInterrupted:
                if not cancel:
                    raise
            else:
                assert not cancel, "cancellation did not interrupt the timeline"
            active = [
                s for s in samples if np.isclose(s.commanded_efforts[index], effort)
            ]
            assert active, (effort, cancel)
            for s in active:
                assert np.isclose(s.applied_efforts[index], effort, atol=1e-5), (
                    effort,
                    s.applied_efforts[index],
                )
                measured = s.effort_metadata["measured"]
                if config.physics.engine == "newton":
                    assert not measured["valid"][index]
                    assert measured["reason"].startswith("unsupported_backend:")
                    assert np.isnan(s.measured_efforts[index])
                else:
                    assert measured["valid"][index], measured
            assert np.allclose(
                controller.last_commanded_efforts[controller.command_indices], 0
            )
            s = active[-1]
            reports.append(
                {
                    "effort": effort,
                    "cancelled": cancel,
                    "samples": len(active),
                    "commanded": float(s.commanded_efforts[index]),
                    "applied": float(s.applied_efforts[index]),
                    "measured": float(s.measured_efforts[index])
                    if np.isfinite(s.measured_efforts[index])
                    else None,
                    "metadata": s.effort_metadata,
                }
            )
        runtime.reset(hold_after_reset=False)
        reset_sample = sample()
        # Reset invalidates the Python last-command cache. Check actual actuation
        # is zero without inventing a new explicit command for that missing cache.
        assert np.allclose(reset_sample.applied_efforts[controller.command_indices], 0)
        cached = controller.last_commanded_efforts[controller.command_indices]
        assert np.allclose(cached[np.isfinite(cached)], 0)
        runtime.set_control_mode(
            "position", expected_generation=runtime.get_control_mode().generation
        )
        report = {
            "profile": profile,
            "joint": joint,
            "units": "N*m",
            "cases": reports,
            "reset_zero": True,
            "reset_commanded_valid": reset_sample.effort_metadata["commanded"]["valid"],
            "restored_position_mode": True,
            "gravity_changed": False,
            "physical_parameters_changed": False,
        }
    except BaseException:
        traceback.print_exc()
        runtime.close()
        raise
    runtime.close(
        before_session_close=lambda _: print(
            MARKER + " " + json.dumps(report), flush=True
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("physx_cpu", "newton_cpu", "newton_cuda"),
        default="physx_cpu",
    )
    args = parser.parse_args()
    if not in_runtime_worker():
        return run_supervised_worker(
            script_path=Path(__file__),
            argv=sys.argv[1:],
            required_markers=(MARKER,),
            success_marker="LINKERBOT_MIRROR_EFFORT_SMOKE_OK",
        )
    run(args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
