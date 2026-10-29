"""Exercise the default AR5/O6 drive through Mirror's public motion entry."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from linkerbot_sim.mirror import create_mirror_runtime
from scripts.smoke_mirror_assemblies import assembly_config
from scripts.runtime_worker_supervisor import in_runtime_worker, run_supervised_worker

MARKER = "LINKERBOT_NEWTON_DRIVE_ENTRY_OK"


def trace_substeps(physics):
    """Record executed substeps on either device, including CUDA graph replay."""
    import warp as wp

    @wp.kernel
    def record(
        count: wp.array[int],
        history: wp.array2d[wp.float64],
        coordinates: wp.array2d[int],
        dofs: wp.array2d[int],
        command_q: wp.array[float],
        command_v: wp.array[float],
        native_q: wp.array[float],
        trace: wp.array3d[wp.float64],
    ):
        row = wp.atomic_add(count, 0, 1)
        if row < trace.shape[0]:
            for j in range(coordinates.shape[1]):
                for column in range(9):
                    trace[row, j, column] = history[0, 9 * j + column]
                trace[row, j, 9] = wp.float64(command_q[coordinates[0, j]])
                trace[row, j, 10] = wp.float64(command_v[dofs[0, j]])
                trace[row, j, 11] = wp.float64(native_q[coordinates[0, j]])

    drive = physics.compliant_drive
    count = wp.zeros(1, dtype=int, device=physics.device)
    trace = wp.zeros(
        (4096, drive.coordinates.shape[1], 12), dtype=wp.float64, device=physics.device
    )
    original = drive.apply

    def apply(state, substep):
        original(state, substep)
        wp.launch(
            record,
            dim=1,
            inputs=[
                count,
                drive.history,
                drive.coordinates,
                drive.dofs,
                physics.control.joint_target_q,
                physics.control.joint_target_qd,
                drive.control.joint_target_q,
                trace,
            ],
            device=physics.device,
            stream=physics.stream,
        )

    drive.apply = apply
    if physics.execution == "cuda":
        physics._graph = None
        physics._graph_state = "pending"
    return count, trace


def run(profile, output):
    config = assembly_config(
        SimpleNamespace(
            hand="o6",
            wrists="both",
            profile=profile,
            gui=False,
            record_root=None,
            depth_hz=30,
            top_hz=60,
            development_cameras=False,
        )
    )
    config = replace(
        config,
        scene=replace(config.scene, cameras=(), planning_startup="lazy"),
        outputs=replace(
            config.outputs,
            render=replace(config.outputs.render, enabled=False, gui=False),
            camera=replace(config.outputs.camera, enabled=False),
            logging=replace(config.outputs.logging, enabled=False),
        ),
    )
    runtime = create_mirror_runtime(config)
    try:
        physics = runtime.physics_runtime
        drive = physics.compliant_drive
        assert drive is not None
        assert not physics.solver.use_mujoco_cpu
        assert not physics.solver.mjw_model.opt.run_collision_detection
        assert drive.substeps == 8 and drive.command_dt == 1 / 60
        robots = runtime.scene_resources.robots_by_id
        expected_suffixes = (
            "thumb_cmc_pitch",
            "thumb_cmc_yaw",
            "index_mcp_pitch",
            "middle_mcp_pitch",
            "ring_mcp_pitch",
            "pinky_mcp_pitch",
        )
        for robot in robots.values():
            expected = tuple(
                f"hand_{'lh' if robot.robot_id == 0 else 'rh'}_{s}"
                for s in expected_suffixes
            )
            assert robot.joint_groups.hand == expected
            assert robot.execution.joint_controller.command_joint_names == (
                *robot.joint_groups.arm,
                *robot.joint_groups.hand,
            )

        count, trace = trace_substeps(physics)
        start_time = physics.simulation_time
        native_start = float(physics.solver.mjw_data.time.numpy()[0])

        def execute(operation, arguments, name):
            runtime.motion.execute(
                operation,
                arguments,
                request_id=name,
                should_cancel=lambda: False,
                protocol="linkerbot.mirror.v2",
            )

        left = robots[0]
        arm = left.joint_groups.arm[0]
        names = list(left.execution.articulation.dof_names)
        q0 = float(left.execution.articulation.get_joint_positions()[names.index(arm)])
        execute(
            "motion.joint_goal",
            {
                "robot_id": 0,
                "group": "arm",
                "duration_s": 2.0,
                "joint_positions": {arm: q0 + 0.005},
                "interpolation": "smoothstep",
            },
            "arm-velocity",
        )
        for robot_id in (0, 1):
            execute(
                "motion.joint_goal",
                {
                    "robot_id": robot_id,
                    "group": "hand",
                    "duration_s": 0.25,
                    "joint_positions": [0.03, 0.02, 0.04, 0.03, 0.02, 0.01],
                    "interpolation": "linear",
                },
                f"hand-order-{robot_id}",
            )

        physics._synchronize_owner_stream()
        n = int(count.numpy()[0])
        assert n == 1200, n  # 150 command ticks, each with eight physical steps.
        data = trace.numpy()[:n]
        np.testing.assert_array_equal(
            data[:, :, 8],
            np.tile(np.arange(1, 9) % 8, (150, 1))
            .reshape(-1, 1)
            .repeat(data.shape[1], axis=1),
        )
        intervals = data.reshape(150, 8, data.shape[1], 12)
        np.testing.assert_array_equal(
            intervals[:, :, :, 3:8], np.repeat(intervals[:, :1, :, 3:8], 8, axis=1)
        )
        params = drive.parameters.numpy()
        hands = np.flatnonzero(params[:, 0] == 1)
        u = (np.arange(1, 9) / 8)[None, :, None]
        expected = (
            intervals[:, :, hands, 3]
            + u * (intervals[:, :, hands, 4] - intervals[:, :, hands, 3])
            + intervals[:, :, hands, 7]
        )
        np.testing.assert_allclose(
            intervals[:, :, hands, 11], expected, atol=1e-7, rtol=0
        )
        arms = np.flatnonzero(params[:, 0] == 0)
        assert np.max(np.abs(data[:960, arms, 10])) > 0.001
        np.testing.assert_allclose(
            intervals[:120, 0, arms, 6], intervals[:120, 0, arms, 10], atol=1e-7
        )
        # This small public smoothstep stays below the arm force guard. Check
        # Hermite position and its analytic velocity feedforward at every substep.
        arm_index = int(arms[0])
        arm_data = intervals[:120, :, arm_index, :]
        fraction = np.arange(1, 9)[None, :] / 8
        q0, q1, v0, v1 = (arm_data[:, :, i] for i in (3, 4, 5, 6))
        dt = drive.command_dt
        qref = (
            (2 * fraction**3 - 3 * fraction**2 + 1) * q0
            + (fraction**3 - 2 * fraction**2 + fraction) * dt * v0
            + (-2 * fraction**3 + 3 * fraction**2) * q1
            + (fraction**3 - fraction**2) * dt * v1
        )
        vref = (
            (6 * fraction**2 - 6 * fraction) * q0
            + (3 * fraction**2 - 4 * fraction + 1) * dt * v0
            + (-6 * fraction**2 + 6 * fraction) * q1
            + (3 * fraction**2 - 2 * fraction) * dt * v1
        ) / dt
        arm_expected = qref + params[arm_index, 2] / params[arm_index, 1] * vref
        np.testing.assert_allclose(arm_data[:, :, 11], arm_expected, atol=1e-7, rtol=0)
        assert abs(physics.simulation_time - start_time - 2.5) < 1e-8
        assert (
            abs(float(physics.solver.mjw_data.time.numpy()[0]) - native_start - 2.5)
            < 0.0002
        )
        # The public executor's final hold clears supplied velocity, without advancing time.
        assert np.max(np.abs(physics.control.joint_target_qd.numpy())) == 0
        snapshot = runtime.get_state()
        history = drive.history.numpy().copy()
        runtime.set_state(snapshot)
        np.testing.assert_array_equal(drive.history.numpy(), history)
        assert np.isfinite(physics.state.joint_q.numpy()).all()
        diagnostics = physics.diagnostics()
        if physics.execution == "cuda":
            assert physics._graph_state == "captured", diagnostics
        result = dict(
            profile=profile,
            control_hz=60,
            physics_hz=480,
            command_ticks=150,
            physics_steps=n,
            arm_velocity_peak=float(np.max(np.abs(data[:960, arms, 10]))),
            arm_reference_max_error=float(
                np.max(np.abs(arm_data[:, :, 11] - arm_expected))
            ),
            interpolation_max_error=float(
                np.max(np.abs(intervals[:, :, hands, 11] - expected))
            ),
            diagnostics=diagnostics,
        )
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
        print(MARKER + " " + json.dumps(result), flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", required=True, choices=("newton_cpu", "newton_cuda")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not in_runtime_worker():
        raise SystemExit(
            run_supervised_worker(
                script_path=Path(__file__),
                argv=sys.argv[1:],
                required_markers=[MARKER],
                success_marker=MARKER + "_SUPERVISED",
                timeout_s=600,
            )
        )
    run(args.profile, args.output)
