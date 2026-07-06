"""通用初始姿态保持步骤。"""

from __future__ import annotations

import numpy as np

from linkerbot_sim.execution.runtime import ExecutionRuntime


def hold_current_pose(
    runtime: ExecutionRuntime,
    *,
    min_steps: int = 3,
    hold_until_app_closed: bool = False,
    phase: str = "initial_hold",
) -> int:
    """把当前完整 DOF 姿态作为目标保持若干 physics step。"""

    robot = runtime.articulation
    controller = runtime.joint_controller
    world = runtime.simulation_world
    full_target = np.asarray(robot.get_joint_positions(), dtype=float)
    full_velocity = np.zeros(robot.num_dof, dtype=float)

    step = 0
    while step < min_steps or (
        hold_until_app_closed
        and runtime.simulation_app is not None
        and runtime.simulation_app.is_running()
    ):
        targets = controller.targets_from_full_state(full_target, full_velocity)
        controller.apply_targets(runtime.articulation_action_type, targets)
        world.step(render=runtime.render_enabled)
        if runtime.camera_observer is not None:
            runtime.camera_observer.observe(world, step=step, phase=phase)
        if runtime.drive_logger is not None:
            driven_indices = controller.driven_indices
            if runtime.drive_logger.should_write(step):
                log_values = runtime.drive_logger.collect_step_values(
                    robot, controller, targets, driven_indices
                )
                runtime.drive_logger.write(
                    step=step,
                    time_s=(step + 1) * float(world.get_physics_dt()),
                    phase=phase,
                    drive_update=True,
                    **log_values,
                )
        step += 1
        if not hold_until_app_closed and step >= min_steps:
            break
    return step
