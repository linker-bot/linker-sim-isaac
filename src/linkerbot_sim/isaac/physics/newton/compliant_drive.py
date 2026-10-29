"""Newton-owned virtual shafts and substep references; no physical-state writes.

The hand equations reproduce the accepted direction-constrained backward-Euler
motor, using finite-difference active-joint velocity at the command clock. Native
joint equality alone transmits follower loads. CPU and CUDA use the same kernels.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import re

from linkerbot_sim.configuration.controllers import ControllerProfiles
from linkerbot_sim.isaac.physics.newton.constraints import read_mujoco_equality_column


@dataclass(frozen=True)
class DriveLayout:
    coordinates: tuple[tuple[int, ...], ...]
    dofs: tuple[tuple[int, ...], ...]
    parameters: tuple[tuple[float, ...], ...]


def _set_custom(builder, name, index, value):
    attribute = builder.custom_attributes[f"mujoco:{name}"]
    if attribute.values is None:
        attribute.values = {}
    if isinstance(attribute.values, list):
        while len(attribute.values) <= index:
            attribute.values.append(None)
    attribute.values[index] = value


def configure_compliant_drives(builder, robots, profiles) -> DriveLayout | None:
    """Cold builder configuration, confined to explicitly selected robot roots."""
    import newton
    import warp as wp

    selected = {
        name: p for name, p in profiles.items() if p.compliant_drive is not None
    }
    if not selected:
        return None
    world_count = builder.world_count
    coordinates, dofs = (
        [[] for _ in range(world_count)],
        [[] for _ in range(world_count)],
    )
    parameters = []
    for name, profile in selected.items():
        if not isinstance(profile, ControllerProfiles):
            raise TypeError("Expected parsed controller profiles")
        settings = profile.compliant_drive
        assert settings is not None
        robot = robots[name]
        unit = robot.asset_path.parent.name
        if not unit.startswith(("ar5_08_o6_", "ar5_o6_")):
            raise ValueError(
                f"Compliant drive supports only AR5/O6 assemblies, got {unit}"
            )
        roots = getattr(robot, "imported_root_paths", None) or (
            robot.imported_root_path,
        )
        if len(roots) != world_count:
            raise ValueError("Compliant drive robot roots must match world count")
        first_parameters = None
        for world, root in enumerate(roots):
            joints = {
                label.rsplit("/", 1)[-1]: j
                for j, label in enumerate(builder.joint_label)
                if label.startswith(str(root).rstrip("/") + "/")
                and int(builder.joint_type[j]) == int(newton.JointType.REVOLUTE)
            }
            arms = sorted(
                (n for n in joints if re.search(r"_joint_[1-7]$", n)),
                key=lambda n: int(n[-1]),
            )
            suffixes = (
                "thumb_cmc_yaw",
                "thumb_cmc_pitch",
                "index_mcp_pitch",
                "middle_mcp_pitch",
                "ring_mcp_pitch",
                "pinky_mcp_pitch",
            )
            active = [
                next((n for n in joints if n.endswith("_" + s)), "") for s in suffixes
            ]
            followers = [
                n
                for n in joints
                if n.endswith(
                    (
                        "_thumb_ip",
                        "_index_dip",
                        "_middle_dip",
                        "_ring_dip",
                        "_pinky_dip",
                    )
                )
            ]
            if (
                len(arms) != 7
                or len(set(active)) != 6
                or "" in active
                or len(followers) != 5
                or len(joints) != 18
            ):
                raise ValueError(
                    f"Expected seven AR5 joints and eleven O6 joints below {root}"
                )
            row_parameters = []
            for n in arms + active + followers:
                j = joints[n]
                dof, coord = builder.joint_qd_start[j], builder.joint_q_start[j]
                builder.joint_friction[dof] = 0.0
                builder.joint_armature[dof] = (
                    settings.arm_armature if n in arms else settings.hand_armature
                )
                if n in followers:
                    builder.joint_target_ke[dof] = 0.0
                    builder.joint_target_kd[dof] = 0.0
                    builder.joint_target_mode[dof] = int(newton.JointTargetMode.NONE)
                    builder.joint_damping[dof] = settings.follower_damping
                    continue
                builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
                kp, kd, cap = (
                    float(getattr(builder, key)[dof])
                    for key in (
                        "joint_target_ke",
                        "joint_target_kd",
                        "joint_effort_limit",
                    )
                )
                if kp <= 0 or kd < 0 or cap <= 0:
                    raise ValueError(
                        f"Compliant drive requires finite positive position gains/caps: {n}"
                    )
                is_arm = n in arms
                if is_arm:
                    if settings.arm_force_guard >= cap:
                        raise ValueError(
                            "Arm force guard must be below the native output cap"
                        )
                    builder.joint_damping[dof] = settings.arm_passive_damping
                    params = (
                        0,
                        kp,
                        kd,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        settings.arm_force_guard,
                    )
                else:
                    motor = (
                        settings.thumb_yaw
                        if n.endswith("_thumb_cmc_yaw")
                        else settings.thumb_flexion
                        if n.endswith("_thumb_cmc_pitch")
                        else settings.finger_flexion
                    )
                    if motor.advancing_torque > cap:
                        raise ValueError(
                            "Virtual advancing torque must not exceed native output cap"
                        )
                    builder.joint_damping[dof] = motor.passive_damping
                    params = (
                        1,
                        kp,
                        kd,
                        motor.output_damping,
                        settings.motor_stiffness,
                        motor.advancing_torque,
                        motor.viscosity,
                        motor.viscosity + motor.reflected_passive_damping,
                        settings.motor_speed_limit,
                        motor.lower,
                        motor.upper,
                        0,
                    )
                coordinates[world].append(coord)
                dofs[world].append(dof)
                row_parameters.append(tuple(float(x) for x in params))
                # Equal actuator and joint caps retain saturation-aware implicit damping.
                solver = newton.solvers.SolverMuJoCo
                builder.add_custom_values(
                    **{
                        "mujoco:actuator_trnid": wp.vec2i(dof, -1),
                        "mujoco:actuator_trntype": int(solver.TrnType.JOINT),
                        "mujoco:ctrl_source": int(solver.CtrlSource.JOINT_TARGET),
                        "mujoco:ctrl_type": int(solver.CtrlType.POSITION),
                        "mujoco:actuator_world": world,
                        "mujoco:actuator_ctrllimited": 0,
                        "mujoco:actuator_has_ctrlrange": 0,
                        "mujoco:actuator_forcelimited": 1,
                        "mujoco:actuator_has_forcerange": 1,
                        "mujoco:actuator_forcerange": wp.vec2(-cap, cap),
                    }
                )
            if first_parameters is None:
                first_parameters = row_parameters
                parameters.extend(row_parameters)
            elif row_parameters != first_parameters:
                raise ValueError("Replicated drive parameters must be homogeneous")
            selected_joints = set(joints.values())
            eq_joints = read_mujoco_equality_column(
                builder, "equality_constraint_joint1"
            )
            if eq_joints is None:
                raise ValueError("Compliant drive requires native joint equalities")
            for index, joint in enumerate(eq_joints):
                if int(joint) in selected_joints:
                    _set_custom(
                        builder,
                        "eq_solref",
                        index,
                        wp.vec2(*settings.equality_solref),
                    )
                    _set_custom(
                        builder,
                        "eq_solimp",
                        index,
                        settings.equality_solimp,
                    )
            for body, label in enumerate(builder.body_label):
                if label.startswith(str(root).rstrip("/") + "/"):
                    _set_custom(builder, "gravcomp", body, 1.0)
    return DriveLayout(
        tuple(map(tuple, coordinates)), tuple(map(tuple, dofs)), tuple(parameters)
    )


@lru_cache(maxsize=1)
def _kernels():
    import warp as wp

    @wp.kernel(enable_backward=False)
    def initialize(
        q: wp.array[float],
        coords: wp.array2d[int],
        params: wp.array2d[wp.float64],
        mask: wp.array[int],
        history: wp.array2d[wp.float64],
    ):
        world, joint = wp.tid()
        if mask[world]:
            value = wp.float64(q[coords[world, joint]])
            shaft = value
            if params[joint, 0] == wp.float64(1):
                shaft = wp.clamp(value, params[joint, 9], params[joint, 10])
            history[world, 9 * joint] = shaft
            history[world, 9 * joint + 1] = value
            history[world, 9 * joint + 2] = wp.float64(0)
            for column in range(3, 9):
                history[world, 9 * joint + column] = wp.float64(0)

    @wp.kernel(enable_backward=False)
    def prepare(
        q: wp.array[float],
        command: wp.array[float],
        velocity: wp.array[float],
        coords: wp.array2d[int],
        dofs: wp.array2d[int],
        params: wp.array2d[wp.float64],
        dt: wp.float64,
        history: wp.array2d[wp.float64],
    ):
        world, joint = wp.tid()
        if history[world, 9 * joint + 8] != wp.float64(0):
            return
        old = history[world, 9 * joint]
        measured = wp.float64(q[coords[world, joint]])
        target = wp.float64(command[coords[world, joint]])
        speed = wp.float64(velocity[dofs[world, joint]])
        previous_speed = history[world, 9 * joint + 2]
        offset = wp.float64(0)
        if params[joint, 0] == wp.float64(1):
            target = wp.clamp(target, params[joint, 9], params[joint, 10])
            measured_speed = (measured - history[world, 9 * joint + 1]) / dt
            error = target - old
            cap = params[joint, 5]
            drive = wp.clamp(params[joint, 4] * error, -cap, cap)
            direction = wp.sign(drive)
            predicted_q = measured + dt * measured_speed
            stiffness, damping = params[joint, 1], params[joint, 3]
            possible = (
                drive - stiffness * (old - predicted_q) + damping * measured_speed
            ) / (stiffness + (damping + params[joint, 6]) / dt)
            free_step = wp.min(
                dt * params[joint, 8], dt * wp.abs(drive) / params[joint, 7]
            )
            step = direction * wp.min(
                wp.max(direction * possible, wp.float64(0)),
                wp.min(wp.abs(error), free_step),
            )
            target = old + step
            speed = step / dt
            offset = damping / stiffness * speed
        history[world, 9 * joint + 3] = old
        history[world, 9 * joint + 4] = target
        history[world, 9 * joint + 5] = previous_speed
        history[world, 9 * joint + 6] = speed
        history[world, 9 * joint + 7] = offset
        history[world, 9 * joint] = target
        history[world, 9 * joint + 1] = measured
        history[world, 9 * joint + 2] = speed

    @wp.kernel(enable_backward=False)
    def apply(
        q: wp.array[float],
        qd: wp.array[float],
        coords: wp.array2d[int],
        dofs: wp.array2d[int],
        params: wp.array2d[wp.float64],
        history: wp.array2d[wp.float64],
        dt: wp.float64,
        command_steps: int,
        target_q: wp.array[float],
        target_v: wp.array[float],
    ):
        world, joint = wp.tid()
        phase = int(history[world, 9 * joint + 8])
        fraction = wp.float64(phase + 1) / wp.float64(command_steps)
        start, end = history[world, 9 * joint + 3], history[world, 9 * joint + 4]
        target = start + fraction * (end - start) + history[world, 9 * joint + 7]
        if params[joint, 0] == wp.float64(0):
            # Hermite interpolation preserves both endpoint positions and supplied
            # trajectory velocities, without task phases or a global retimer.
            u, v0, v1 = (
                fraction,
                history[world, 9 * joint + 5],
                history[world, 9 * joint + 6],
            )
            qref = (
                (wp.float64(2) * u * u * u - wp.float64(3) * u * u + wp.float64(1))
                * start
                + (u * u * u - wp.float64(2) * u * u + u) * dt * v0
                + (-wp.float64(2) * u * u * u + wp.float64(3) * u * u) * end
                + (u * u * u - u * u) * dt * v1
            )
            vref = (
                (wp.float64(6) * u * u - wp.float64(6) * u) * start
                + (wp.float64(3) * u * u - wp.float64(4) * u + wp.float64(1)) * dt * v0
                + (-wp.float64(6) * u * u + wp.float64(6) * u) * end
                + (wp.float64(3) * u * u - wp.float64(2) * u) * dt * v1
            ) / dt
            kp, kd, guard = params[joint, 1], params[joint, 2], params[joint, 11]
            target = qref + kd / kp * vref
            measured, speed = (
                wp.float64(q[coords[world, joint]]),
                wp.float64(qd[dofs[world, joint]]),
            )
            force = kp * (target - measured) - kd * speed
            if wp.abs(force) > guard:
                target = measured + (kd * speed + wp.clamp(force, -guard, guard)) / kp
        target_q[coords[world, joint]] = float(target)
        target_v[dofs[world, joint]] = 0.0
        history[world, 9 * joint + 8] = wp.float64((phase + 1) % command_steps)

    @wp.kernel(enable_backward=False)
    def restore(
        values: wp.array2d[wp.float64],
        mask: wp.array[wp.bool],
        history: wp.array2d[wp.float64],
    ):
        world, column = wp.tid()
        if mask[world]:
            history[world, column] = values[world, column]

    return initialize, prepare, apply, restore


class NewtonCompliantDrive:
    """Device-resident history and one native control buffer owned by Newton."""

    def __init__(self, model, state, layout, *, command_dt, substeps, stream):
        import warp as wp

        if type(substeps) is not int or substeps < 1:
            raise ValueError(
                "Command interval must contain a positive integer of physics steps"
            )
        if (
            type(command_dt) not in (float, int)
            or not math.isfinite(command_dt)
            or command_dt <= 0
        ):
            raise ValueError("Command interval must be finite and positive")
        self.model, self.stream = model, stream
        self.command_dt, self.substeps = float(command_dt), int(substeps)
        self.coordinates = wp.array(
            layout.coordinates, dtype=wp.int32, device=model.device
        )
        self.dofs = wp.array(layout.dofs, dtype=wp.int32, device=model.device)
        self.parameters = wp.array(
            layout.parameters, dtype=wp.float64, device=model.device
        )
        worlds, joints = self.coordinates.shape
        self.history = wp.zeros(
            (worlds, joints * 9), dtype=wp.float64, device=model.device
        )
        self.all_worlds = wp.full(worlds, True, dtype=wp.bool, device=model.device)
        self.reset_all_worlds = wp.full(worlds, 1, dtype=wp.int32, device=model.device)
        self.control = model.control()
        self.reset_from_positions(state.joint_q)
        self.baseline = wp.clone(self.history)

    def reset_from_positions(self, q, mask=None):
        import warp as wp

        wp.launch(
            _kernels()[0],
            dim=self.coordinates.shape,
            inputs=[
                q,
                self.coordinates,
                self.parameters,
                self.reset_all_worlds if mask is None else mask,
                self.history,
            ],
            device=self.model.device,
            stream=self.stream,
        )

    def prepare(self, state, command):
        import warp as wp

        for name in ("joint_f", "joint_target_q", "joint_target_qd", "joint_act"):
            source, destination = (
                getattr(command, name, None),
                getattr(self.control, name, None),
            )
            if source is not None and destination is not None:
                wp.copy(destination, source, stream=self.stream)
        wp.launch(
            _kernels()[1],
            dim=self.coordinates.shape,
            inputs=[
                state.joint_q,
                command.joint_target_q,
                command.joint_target_qd,
                self.coordinates,
                self.dofs,
                self.parameters,
                self.command_dt,
                self.history,
            ],
            device=self.model.device,
            stream=self.stream,
        )

    def apply(self, state, substep):
        import warp as wp

        wp.launch(
            _kernels()[2],
            dim=self.coordinates.shape,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.coordinates,
                self.dofs,
                self.parameters,
                self.history,
                self.command_dt,
                self.substeps,
                self.control.joint_target_q,
                self.control.joint_target_qd,
            ],
            device=self.model.device,
            stream=self.stream,
        )

    def restore(self, values, mask=None):
        import warp as wp

        if (
            values.dtype != wp.float64
            or values.shape != self.history.shape
            or values.device != self.history.device
        ):
            raise ValueError(
                "Controller history must match the live shape, float64 dtype and device"
            )
        if mask is not None and (
            mask.dtype != wp.bool
            or mask.shape != self.all_worlds.shape
            or mask.device != values.device
        ):
            raise ValueError(
                "Controller history mask must match the world layout and device"
            )
        wp.launch(
            _kernels()[3],
            dim=self.history.shape,
            inputs=[values, self.all_worlds if mask is None else mask, self.history],
            device=self.model.device,
            stream=self.stream,
        )

    def commit(self):
        import warp as wp

        wp.copy(self.baseline, self.history, stream=self.stream)
