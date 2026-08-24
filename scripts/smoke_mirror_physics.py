#!/usr/bin/env python3
"""用完整 MirrorRuntime 验证 PhysX/Newton、snapshot、相机与 cuRobo FK。"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from linkerbot_sim.isaac.physics.backend import (  # noqa: E402
    active_physics_backend,
    normalize_physics_backend,
)
from linkerbot_sim.configuration import load_mirror_config  # noqa: E402
from linkerbot_sim.mirror import create_mirror_runtime  # noqa: E402
from linkerbot_sim.snapshots.mirror_adapter import (  # noqa: E402
    get_mirror_snapshot,
    set_mirror_snapshot,
)
from linkerbot_sim.utils.json import strict_json_dumps  # noqa: E402
from scripts.runtime_worker_supervisor import (  # noqa: E402
    in_runtime_worker,
    run_supervised_worker,
)


RUNTIME_SUCCESS_MARKER = "LINKERBOT_MIRROR_PHYSICS_RUNTIME_OK"
SUCCESS_MARKER = "LINKERBOT_MIRROR_PHYSICS_SMOKE_OK"
DEFAULT_MIRROR_PROFILE = "physx_cpu"
DEFAULT_STEPS = 8
POSITION_DELTA_RAD = 1.0e-3
STATE_ATOL = 1.0e-6
CONTROL_RESPONSE_EPS = 1.0e-9
CAMERA_WARMUP_STEPS = 6
CAMERA_WARMUP_SAMPLE_PERIODS = 2
CAMERA_STEADY_RENDER_ATTEMPTS = 3
POST_STEP_JOINT_POSITION_ATOL = 1.0e-3
HYBRID_PROBE_ARM_POSITIONS = (0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0)
HYBRID_PROBE_MOVE_DURATION_S = 1.0
HYBRID_PROBE_SETTLE_DURATION_S = 0.25
HYBRID_PROBE_SETTLE_ATTEMPTS = 4


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析仅接受新 Mirror profile 的可重复物理 smoke 参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "physx_cpu",
            "physx_cpu_hybrid",
            "newton_cpu",
            "newton_cuda",
        ),
        default=DEFAULT_MIRROR_PROFILE,
    )
    parser.add_argument("--steps", type=_positive_int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--control-modes-only",
        action="store_true",
        help="only verify position/velocity/effort switching within the same runtime",
    )
    return parser.parse_args(argv)


def resolve_smoke_config(args: argparse.Namespace):
    """通过唯一 strict loader 解析完整 Mirror config graph。"""

    return load_mirror_config(str(args.profile))


def create_smoke_runtime(config: object):
    """通过正式 product composition root 创建完整 MirrorRuntime。"""

    return create_mirror_runtime(config)


def _scene_resources(runtime: object) -> object:
    """取得 probe 所需资源集合；测试 double 可以直接实现同一形状。"""

    resources = getattr(runtime, "scene_resources", None)
    return runtime if resources is None else resources


def _tensor_handle_valid(articulation: object) -> bool:
    """兼容 Isaac Core 与 Experimental Core 的 tensor handle 有效性接口。"""

    checker = getattr(articulation, "is_physics_handle_valid", None)
    if callable(checker):
        return bool(checker())
    initialized = getattr(articulation, "handles_initialized", None)
    if callable(initialized):
        return bool(initialized())
    if initialized is not None:
        return bool(initialized)
    raise RuntimeError("articulation does not expose tensor handle validity")


def _finite_vector(value: object, *, label: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != int(size):
        raise RuntimeError(
            f"{label} size mismatch: expected={int(size)}, actual={array.size}"
        )
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{label} contains non-finite values")
    return array.copy()


def _inspect_articulation(robot: object) -> dict[str, object]:
    label = str(robot.label)
    articulation = robot.execution.articulation
    if not _tensor_handle_valid(articulation):
        raise RuntimeError(f"robot {label!r} has an invalid articulation tensor handle")
    dof_names = tuple(str(name) for name in articulation.dof_names)
    if not dof_names or any(not name for name in dof_names):
        raise RuntimeError(f"robot {label!r} has empty articulation DOF names")
    if len(set(dof_names)) != len(dof_names):
        raise RuntimeError(f"robot {label!r} has duplicate articulation DOF names")
    num_dof = int(articulation.num_dof)
    if num_dof != len(dof_names):
        raise RuntimeError(
            f"robot {label!r} DOF metadata mismatch: "
            f"num_dof={num_dof}, names={len(dof_names)}"
        )
    _finite_vector(
        articulation.get_joint_positions(),
        label=f"robot {label!r} joint positions",
        size=num_dof,
    )
    _finite_vector(
        articulation.get_joint_velocities(),
        label=f"robot {label!r} joint velocities",
        size=num_dof,
    )
    return {
        "robot_id": int(robot.robot_id),
        "label": label,
        "tensor_handle_valid": True,
        "num_dof": num_dof,
        "dof_names": list(dof_names),
        "positions_finite": True,
        "velocities_finite": True,
    }


def _ordered_robots(runtime: object) -> tuple[object, ...]:
    robots = tuple(
        robot
        for _robot_id, robot in sorted(
            runtime.robots_by_id.items(), key=lambda item: int(item[0])
        )
    )
    if not robots:
        raise RuntimeError("MirrorSceneResources did not create any robot articulation")
    return robots


def _runtime_owner_identities(
    runtime: object,
    resources: object,
    robots: Sequence[object],
) -> dict[str, int]:
    owners = {
        "runtime": runtime,
        "session": getattr(runtime, "session", None),
        "physics_runtime": getattr(runtime, "physics_runtime", None),
        "motion": getattr(runtime, "motion", None),
        "motion_backend": getattr(getattr(runtime, "motion", None), "backend", None),
        "control_mode": getattr(runtime, "control_mode", None),
        "collision": getattr(runtime, "collision", None),
        "planning_registry": getattr(resources, "planning_registry", None),
    }
    for robot in robots:
        owners[f"articulation.{robot.label}"] = robot.execution.articulation
        owners[f"controller.{robot.label}"] = robot.execution.joint_controller
    missing = [name for name, value in owners.items() if value is None]
    if missing:
        raise RuntimeError(f"control-mode smoke is missing runtime owners: {missing}")
    return {name: id(value) for name, value in owners.items()}


def _require_control_state(
    runtime: object,
    *,
    mode: str,
    generation: int,
) -> object:
    getter = getattr(runtime, "get_control_mode", None)
    if not callable(getter):
        raise RuntimeError("Mirror runtime does not expose get_control_mode()")
    state = getter()
    actual = (
        getattr(state, "initial_mode", None),
        getattr(state, "active_mode", None),
        getattr(state, "generation", None),
        getattr(state, "scope", None),
    )
    expected = ("position", mode, generation, "all")
    if actual != expected:
        raise RuntimeError(
            f"unexpected Mirror control-mode state: actual={actual!r}, "
            f"expected={expected!r}"
        )
    return state


def _control_probe_joint(robot: object) -> tuple[str, str, int]:
    controller = robot.execution.joint_controller
    command_names = tuple(str(name) for name in controller.command_joint_names)
    command_indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    if len(command_names) != command_indices.size:
        raise RuntimeError(f"robot {robot.label!r} has inconsistent command metadata")
    groups = getattr(robot, "joint_groups", None)
    names_for_group = getattr(groups, "names", None)
    if not callable(names_for_group):
        raise RuntimeError(f"robot {robot.label!r} does not expose joint groups")
    by_name = dict(zip(command_names, command_indices.tolist(), strict=True))
    for group in ("arm", "hand"):
        for name in names_for_group(group):
            if name in by_name:
                return group, name, int(by_name[name])
    raise RuntimeError(f"robot {robot.label!r} has no arm/hand command joint")


def _execute_control_probe_motion(
    runtime: object,
    operation: str,
    arguments: Mapping[str, object],
    *,
    request_id: str,
) -> None:
    motion = getattr(runtime, "motion", None)
    execute = getattr(motion, "execute", None)
    if not callable(execute):
        raise RuntimeError("Mirror runtime does not expose its motion owner")
    execute(
        operation,
        arguments,
        request_id=request_id,
        should_cancel=lambda: False,
        protocol="linkerbot.mirror.v2",
    )


def _require_controller_mode(robots: Sequence[object], mode: str) -> None:
    for robot in robots:
        controller = robot.execution.joint_controller
        modes = tuple(str(value) for value in controller.command_target_modes)
        if not modes or modes != (mode,) * len(modes):
            raise RuntimeError(
                f"robot {robot.label!r} controller modes do not match {mode!r}: {modes}"
            )


def _require_terminal_neutral(robot: object, *, mode: str) -> None:
    controller = robot.execution.joint_controller
    indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    targets = controller.snapshot_control_targets_cache()
    if targets is None:
        raise RuntimeError(f"robot {robot.label!r} has no terminal target cache")
    values = (
        np.asarray(targets.velocities, dtype=float).reshape(-1)[indices]
        if mode == "velocity"
        else np.asarray(targets.efforts, dtype=float).reshape(-1)[indices]
    )
    if not np.all(np.isfinite(values)) or not np.allclose(
        values, 0.0, rtol=0.0, atol=STATE_ATOL
    ):
        raise RuntimeError(
            f"robot {robot.label!r} retained a non-neutral {mode} target: "
            f"{values.tolist()}"
        )


def _exercise_runtime_control_modes(
    runtime: object,
    resources: object,
    robots: Sequence[object],
    *,
    steps: int,
) -> dict[str, object]:
    """Exercise all Mirror control modes without replacing a runtime owner."""

    owners = _runtime_owner_identities(runtime, resources, robots)
    _require_control_state(runtime, mode="position", generation=0)
    _require_controller_mode(robots, "position")
    robot = robots[0]
    group, joint_name, joint_index = _control_probe_joint(robot)
    articulation = robot.execution.articulation
    duration_s = float(resources.physics.get_physics_dt()) * int(steps)

    def joint_position() -> float:
        positions = _finite_vector(
            articulation.get_joint_positions(),
            label=f"robot {robot.label!r} control probe positions",
            size=int(articulation.num_dof),
        )
        return float(positions[joint_index])

    _execute_control_probe_motion(
        runtime,
        "motion.joint_goal",
        {
            "robot_id": int(robot.robot_id),
            "robot_label": str(robot.label),
            "group": group,
            "duration_s": duration_s,
            "joint_positions": {joint_name: joint_position() + POSITION_DELTA_RAD},
        },
        request_id="smoke-control-position",
    )

    change = runtime.set_control_mode("velocity", expected_generation=0)
    if not change.changed or change.generation != 1:
        raise RuntimeError("position-to-velocity switch returned an invalid change")
    _require_control_state(runtime, mode="velocity", generation=1)
    _require_controller_mode(robots, "velocity")
    velocity_start = joint_position()
    _execute_control_probe_motion(
        runtime,
        "motion.joint_goal",
        {
            "robot_id": int(robot.robot_id),
            "robot_label": str(robot.label),
            "group": group,
            "duration_s": duration_s,
            "joint_positions": {joint_name: velocity_start + POSITION_DELTA_RAD},
        },
        request_id="smoke-control-velocity",
    )
    velocity_end = joint_position()
    if not np.isfinite(velocity_end) or abs(velocity_end - velocity_start) <= 1.0e-12:
        raise RuntimeError("velocity-mode joint goal produced no finite displacement")
    _require_terminal_neutral(robot, mode="velocity")

    change = runtime.set_control_mode("effort", expected_generation=1)
    if not change.changed or change.generation != 2:
        raise RuntimeError("velocity-to-effort switch returned an invalid change")
    _require_control_state(runtime, mode="effort", generation=2)
    _require_controller_mode(robots, "effort")
    controller = robot.execution.joint_controller
    prepared = controller.prepare_runtime()
    effort_limit = float(prepared.active_effort_limits[joint_index])
    if not np.isfinite(effort_limit) or effort_limit <= 0.0:
        raise RuntimeError("effort-mode control probe has no positive profile limit")
    requested_effort = effort_limit * 0.05
    _execute_control_probe_motion(
        runtime,
        "motion.joint_effort",
        {
            "robot_id": int(robot.robot_id),
            "robot_label": str(robot.label),
            "group": group,
            "duration_s": duration_s,
            "joint_efforts": {joint_name: requested_effort},
        },
        request_id="smoke-control-effort",
    )
    _finite_vector(
        articulation.get_joint_positions(),
        label=f"robot {robot.label!r} effort-mode positions",
        size=int(articulation.num_dof),
    )
    _require_terminal_neutral(robot, mode="effort")
    commanded_efforts = np.asarray(controller.last_commanded_efforts, dtype=float)
    if not np.allclose(
        commanded_efforts[np.asarray(controller.command_indices, dtype=int)],
        0.0,
        rtol=0.0,
        atol=STATE_ATOL,
    ):
        raise RuntimeError("effort-mode motion did not clear commanded efforts")

    change = runtime.set_control_mode("position", expected_generation=2)
    if not change.changed or change.generation != 3:
        raise RuntimeError("effort-to-position switch returned an invalid change")
    _require_control_state(runtime, mode="position", generation=3)
    _require_controller_mode(robots, "position")
    _execute_control_probe_motion(
        runtime,
        "motion.joint_goal",
        {
            "robot_id": int(robot.robot_id),
            "robot_label": str(robot.label),
            "group": group,
            "duration_s": duration_s,
            "joint_positions": {joint_name: joint_position() - POSITION_DELTA_RAD},
        },
        request_id="smoke-control-position-return",
    )

    runtime.reset(hold_after_reset=False)
    _require_control_state(runtime, mode="position", generation=3)
    if _runtime_owner_identities(runtime, resources, robots) != owners:
        raise RuntimeError("Mirror control-mode switching replaced a runtime owner")
    return {
        "verified": True,
        "sequence": ["position", "velocity", "effort", "position"],
        "generation": 3,
        "identity_owners": sorted(owners),
        "velocity_displacement_rad": velocity_end - velocity_start,
        "effort_fraction": 0.05,
        "terminal_velocity_zero": True,
        "terminal_effort_zero": True,
        "reset_preserved_mode": True,
    }


def _controller_mode_report(controller: object) -> dict[str, object]:
    settings = controller.settings
    arm = settings.arm
    hand = settings.hand
    if arm is None or hand is None:
        raise RuntimeError(
            "hybrid smoke requires explicit arm and hand controller settings"
        )
    return {
        "arm": {"mode": str(arm.mode), "method": str(arm.method)},
        "hand": {"mode": str(hand.mode), "method": str(hand.method)},
        "command_target_modes": [
            str(value) for value in controller.command_target_modes
        ],
    }


def _scaled_hybrid_parameter_updates(snapshot: object) -> dict[str, object]:
    values = snapshot.values
    updates: dict[str, object] = {}
    for field in (
        "motion_stiffness",
        "motion_damping",
        "force_proportional",
        "force_integral",
    ):
        current = tuple(float(value) for value in getattr(values, field))
        updates[field] = [0.9 * value for value in current]
    updates["posture_stiffness"] = 0.9 * float(values.posture_stiffness)
    updates["posture_damping"] = 0.9 * float(values.posture_damping)
    return updates


def _move_hybrid_robot_to_probe_pose(
    runtime: object, robot: object
) -> dict[str, object]:
    arm_names = tuple(str(name) for name in robot.joint_groups.arm)
    if len(arm_names) != len(HYBRID_PROBE_ARM_POSITIONS):
        raise RuntimeError(
            "hybrid smoke probe pose requires a seven-joint arm, "
            f"got {len(arm_names)} joints"
        )
    runtime.motion.execute(
        "motion.joint_goal",
        {
            "robot_id": int(robot.robot_id),
            "robot_label": str(robot.label),
            "group": "arm",
            "duration_s": HYBRID_PROBE_MOVE_DURATION_S,
            "joint_positions": dict(
                zip(arm_names, HYBRID_PROBE_ARM_POSITIONS, strict=True)
            ),
            "interpolation": "smoothstep",
        },
        request_id="smoke-hybrid-probe-pose",
        should_cancel=lambda: False,
        protocol="linkerbot.mirror.v2",
    )
    maximum_speed = math.inf
    settle_attempts = 0
    settings = runtime.config.hybrid_control
    for settle_attempts in range(1, HYBRID_PROBE_SETTLE_ATTEMPTS + 1):
        runtime.motion.execute(
            "motion.hold",
            {
                "robot_id": int(robot.robot_id),
                "robot_label": str(robot.label),
                "group": "arm",
                "duration_s": HYBRID_PROBE_SETTLE_DURATION_S,
            },
            request_id=f"smoke-hybrid-settle-{settle_attempts}",
            should_cancel=lambda: False,
            protocol="linkerbot.mirror.v2",
        )
        observation = robot.task_space_port.observe()
        maximum_speed = float(np.max(np.abs(observation.joint_velocities)))
        if maximum_speed <= 0.5 * float(settings.tare.maximum_joint_speed):
            break
    else:
        raise RuntimeError(
            "hybrid smoke probe pose did not settle below the tare speed margin: "
            f"observed={maximum_speed}, "
            f"required<={0.5 * float(settings.tare.maximum_joint_speed)}"
        )

    characteristic_length = float(settings.posture.characteristic_length_m)
    scale = np.diag(
        [
            1.0 / characteristic_length,
            1.0 / characteristic_length,
            1.0 / characteristic_length,
            1.0,
            1.0,
            1.0,
        ]
    )
    singular_values = np.linalg.svd(
        scale @ np.asarray(observation.jacobian, dtype=float),
        compute_uv=False,
    )
    minimum_singular_value = float(np.min(singular_values))
    condition_number = float(
        np.inf
        if minimum_singular_value <= 0.0
        else np.max(singular_values) / minimum_singular_value
    )
    if minimum_singular_value < float(
        settings.posture.minimum_singular_value
    ) or condition_number > float(settings.posture.maximum_condition_number):
        raise RuntimeError(
            "hybrid smoke probe pose is singular in the live runtime: "
            f"minimum_singular_value={minimum_singular_value}, "
            f"condition_number={condition_number}"
        )
    return {
        "arm_joint_names": list(arm_names),
        "arm_joint_positions": list(HYBRID_PROBE_ARM_POSITIONS),
        "move_duration_s": HYBRID_PROBE_MOVE_DURATION_S,
        "settle_duration_s": settle_attempts * HYBRID_PROBE_SETTLE_DURATION_S,
        "maximum_joint_speed": maximum_speed,
        "minimum_singular_value": minimum_singular_value,
        "condition_number": condition_number,
    }


def _cancelled_hybrid_segment(
    runtime: object,
    robot: object,
    *,
    request_id: str,
    force_axes: Sequence[bool],
    tare_generation: int,
    parameter_generation: int,
) -> dict[str, object]:
    controller = robot.execution.joint_controller
    port = robot.task_space_port
    binding = robot.physical_tcp_binding
    original_settings = controller.settings
    original_modes = tuple(str(value) for value in controller.command_target_modes)
    observation = port.observe()
    backend = runtime.motion.backend
    captured: dict[str, object] = {}

    def cancel_after_active_sample() -> bool:
        diagnostics = dict(backend.hybrid_diagnostics())
        if not bool(diagnostics.get("active", False)):
            return False
        captured.update(diagnostics)
        captured["controller"] = _controller_mode_report(controller)
        return True

    duration_s = 2.0 * float(runtime.physics_dt_s)
    arguments = {
        "robot_id": int(robot.robot_id),
        "robot_label": str(robot.label),
        "duration_s": duration_s,
        "tcp_frame_name": str(binding.tcp_frame_name),
        "reference_frame": "world",
        "target_position": np.asarray(observation.position, dtype=float).tolist(),
        "target_orientation_wxyz": np.asarray(
            observation.orientation_wxyz, dtype=float
        ).tolist(),
        "force_axes": [bool(value) for value in force_axes],
        "target_wrench": [0.0] * 6,
        "tare_generation": int(tare_generation),
        "hybrid_parameter_generation": int(parameter_generation),
        "phase": request_id,
    }
    try:
        runtime.motion.execute(
            "motion.hybrid_force_position",
            arguments,
            request_id=request_id,
            should_cancel=cancel_after_active_sample,
            protocol="linkerbot.mirror.v3",
        )
    except BaseException as exc:
        if getattr(exc, "code", None) != "cancelled":
            raise
    else:
        raise RuntimeError(
            "hybrid smoke segment completed before its cancellation probe"
        )

    expected_axes = [bool(value) for value in force_axes]
    if not captured:
        raise RuntimeError("hybrid smoke did not observe an active controller sample")
    if captured.get("request_id") != request_id:
        raise RuntimeError("hybrid smoke captured diagnostics for the wrong request")
    if captured.get("force_axes") != expected_axes:
        raise RuntimeError("hybrid smoke captured the wrong force-axis selection")
    if captured.get("hybrid_parameter_generation") != int(parameter_generation):
        raise RuntimeError("hybrid smoke captured the wrong parameter generation")
    active_controller = dict(captured["controller"])
    if active_controller.get("arm") != {"mode": "effort", "method": "direct"}:
        raise RuntimeError("hybrid smoke arm did not enter effort/direct control")
    if active_controller.get("hand") != {
        "mode": "position",
        "method": "implicit",
    }:
        raise RuntimeError("hybrid smoke hand did not remain position/implicit")
    if (
        controller.settings != original_settings
        or tuple(str(value) for value in controller.command_target_modes)
        != original_modes
    ):
        raise RuntimeError(
            "hybrid smoke did not restore the original controller runtime"
        )
    terminal_targets = controller.last_control_targets
    if terminal_targets is None:
        raise RuntimeError("hybrid smoke left no terminal handover target")
    target_efforts = np.asarray(terminal_targets.efforts, dtype=float).reshape(-1)
    commanded_efforts = np.asarray(
        controller.last_commanded_efforts, dtype=float
    ).reshape(-1)
    finite_commands = commanded_efforts[np.isfinite(commanded_efforts)]
    if (
        target_efforts.shape != commanded_efforts.shape
        or not np.all(np.isfinite(target_efforts))
        or not np.allclose(target_efforts, 0.0, rtol=0.0, atol=STATE_ATOL)
        or np.any(np.isinf(commanded_efforts))
        or not np.allclose(finite_commands, 0.0, rtol=0.0, atol=STATE_ATOL)
    ):
        raise RuntimeError("hybrid smoke left a non-zero terminal effort command")
    return {
        "request_id": request_id,
        "force_axes": expected_axes,
        "hybrid_parameter_generation": int(parameter_generation),
        "active_controller": active_controller,
        "restored_controller": _controller_mode_report(controller),
        "expected_cancellation": True,
        "terminal_effort_zero": True,
    }


def _exercise_runtime_hybrid_control(
    runtime: object,
    resources: object,
    robots: Sequence[object],
) -> dict[str, object]:
    config = getattr(runtime, "config", None)
    if getattr(config, "hybrid_control", None) is None:
        return {"performed": False, "reason": "not_configured"}
    candidates = tuple(
        robot
        for robot in robots
        if getattr(robot, "task_space_port", None) is not None
        and getattr(robot, "physical_tcp_binding", None) is not None
    )
    if not candidates:
        raise RuntimeError("hybrid smoke found no task-space capable robot")
    robot = candidates[0]
    parameters = getattr(runtime.controller, "hybrid_parameters", None)
    if parameters is None:
        raise RuntimeError("hybrid smoke has no parameter service")
    initial = parameters.snapshot()
    if int(initial.generation) != 0:
        raise RuntimeError("hybrid smoke requires the initial parameter generation")

    probe_pose = _move_hybrid_robot_to_probe_pose(runtime, robot)
    binding = robot.physical_tcp_binding
    tare = runtime.motion.tare_wrench(
        {
            "robot_id": int(robot.robot_id),
            "robot_label": str(robot.label),
            "tcp_frame_name": str(binding.tcp_frame_name),
            "reference_frame": "world",
        },
        request_id="smoke-hybrid-tare",
        should_cancel=lambda: False,
    )
    tare_generation = int(tare["tare_generation"])
    first = _cancelled_hybrid_segment(
        runtime,
        robot,
        request_id="smoke-hybrid-segment-z",
        force_axes=(False, False, True, False, False, False),
        tare_generation=tare_generation,
        parameter_generation=0,
    )
    change = parameters.set_parameters(
        _scaled_hybrid_parameter_updates(initial),
        expected_generation=0,
    )
    if not bool(change.changed) or int(change.generation) != 1:
        raise RuntimeError("hybrid smoke parameter update did not commit generation 1")
    second = _cancelled_hybrid_segment(
        runtime,
        robot,
        request_id="smoke-hybrid-segment-x",
        force_axes=(True, False, False, False, False, False),
        tare_generation=tare_generation,
        parameter_generation=1,
    )
    final = parameters.snapshot()
    if int(final.generation) != 1:
        raise RuntimeError("hybrid smoke lost the committed parameter generation")
    return {
        "performed": True,
        "robot_id": int(robot.robot_id),
        "robot_label": str(robot.label),
        "tcp_frame_name": str(binding.tcp_frame_name),
        "tare_generation": tare_generation,
        "tare_sample_count": int(tare["sample_count"]),
        "probe_pose": probe_pose,
        "parameter_generations": [0, 1],
        "updated_parameters": sorted(_scaled_hybrid_parameter_updates(initial)),
        "segments": [first, second],
        "global_control_mode": str(runtime.get_control_mode().active_mode),
    }


def _apply_position_targets(
    runtime: object, robots: Sequence[object], steps: int
) -> list[dict[str, object]]:
    """通过现有 JointController API 给全部 articulation 同步下发微小目标。"""

    commands: dict[int, np.ndarray] = {}
    probes: dict[int, tuple[int, float, float]] = {}
    target_readbacks: dict[int, float] = {}
    for robot in robots:
        articulation = robot.execution.articulation
        controller = robot.execution.joint_controller
        positions = np.asarray(articulation.get_joint_positions(), dtype=float).reshape(
            -1
        )
        indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
        if not indices.size:
            raise RuntimeError(f"robot {robot.label!r} has no command joints")
        if len(set(indices.tolist())) != indices.size:
            raise RuntimeError(f"robot {robot.label!r} has duplicate command indices")
        if np.any(indices < 0) or np.any(indices >= positions.size):
            raise RuntimeError(f"robot {robot.label!r} has invalid command indices")
        command = positions[indices].copy()
        command[0] += POSITION_DELTA_RAD
        commands[int(robot.robot_id)] = command
        probes[int(robot.robot_id)] = (
            int(indices[0]),
            float(positions[indices[0]]),
            float(command[0]),
        )

    for _step in range(int(steps)):
        for robot in robots:
            execution = robot.execution
            articulation = execution.articulation
            controller = execution.joint_controller
            positions = np.asarray(
                articulation.get_joint_positions(), dtype=float
            ).reshape(-1)
            targets = controller.build_control_targets(
                command_positions=commands[int(robot.robot_id)],
                command_velocities=np.zeros_like(commands[int(robot.robot_id)]),
                command_efforts=np.zeros_like(commands[int(robot.robot_id)]),
                base_positions=positions,
            )
            controller.apply_targets(execution.articulation_action_type, targets)
            robot_id = int(robot.robot_id)
            joint_index, _initial, expected = probes[robot_id]
            # Isaac 6 的 SingleArticulation 不再公开 get_joint_position_targets。生产
            # JointController 只有在全部 engine action 成功后才提交完整 target cache，
            # 因而统一通过下面的 backend-neutral drive-target 读取合同验证首个命令关节。
            position_targets, _velocity_targets = _capture_drive_targets((robot,))[
                str(robot.label)
            ]
            target_readback = float(position_targets[0])
            if not np.isfinite(target_readback) or not np.isclose(
                target_readback,
                expected,
                rtol=1.0e-5,
                atol=1.0e-7,
            ):
                raise RuntimeError(
                    f"robot {robot.label!r} position target was not committed to "
                    f"the probed DOF: expected={expected}, readback={target_readback}, "
                    f"joint_index={joint_index}"
                )
            target_readbacks[robot_id] = target_readback
        runtime.physics.step(render=False)

    responses: list[dict[str, object]] = []
    for robot in robots:
        robot_id = int(robot.robot_id)
        joint_index, initial, target = probes[robot_id]
        final = float(
            np.asarray(
                robot.execution.articulation.get_joint_positions(), dtype=float
            ).reshape(-1)[joint_index]
        )
        final_velocity = float(
            np.asarray(
                robot.execution.articulation.get_joint_velocities(), dtype=float
            ).reshape(-1)[joint_index]
        )
        runtime_controller = robot.execution.articulation.get_articulation_controller()
        stiffnesses, dampings = runtime_controller.get_gains()
        stiffness = float(np.asarray(stiffnesses, dtype=float).reshape(-1)[joint_index])
        damping = float(np.asarray(dampings, dtype=float).reshape(-1)[joint_index])
        requested_delta = target - initial
        observed_delta = final - initial
        if (
            not np.isfinite(final)
            or observed_delta * requested_delta <= CONTROL_RESPONSE_EPS
        ):
            raise RuntimeError(
                f"robot {robot.label!r} did not move toward its position target: "
                f"initial={initial}, target={target}, "
                f"target_readback={target_readbacks.get(robot_id)!r}, final={final}, "
                f"final_velocity={final_velocity}, stiffness={stiffness}, "
                f"damping={damping}"
            )
        responses.append(
            {
                "robot_id": robot_id,
                "label": str(robot.label),
                "joint_index": joint_index,
                "initial": initial,
                "target": target,
                "target_readback": target_readbacks[robot_id],
                "final": final,
                "final_velocity": final_velocity,
                "stiffness": stiffness,
                "damping": damping,
                "observed_delta": observed_delta,
            }
        )
    return responses


def _assert_no_active_mjc_actuators(runtime: object) -> list[str]:
    """确保 Newton 路径没有与项目 DriveAPI 竞争的 importer direct actuator。"""

    stage = runtime.session.stage
    active = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsActive() and str(prim.GetTypeName()) == "MjcActuator"
    ]
    if active:
        raise RuntimeError(
            "imported MjcActuator prims remain active alongside project DriveAPI: "
            + ", ".join(active)
        )
    return active


def _camera_probe(runtime: object) -> dict[str, object]:
    """Warm RTX while stepping, then prove steady rendering does not step physics."""

    resources = _scene_resources(runtime)
    cameras = tuple(getattr(resources, "sensor_cameras", ()))
    if not cameras:
        return {"performed": False, "reason": "no_enabled_camera"}

    from linkerbot_sim.sensors.camera.frame import sample_camera_frames

    frame_indices: dict[tuple[str, str], int] = {}
    ready_frames: dict[tuple[str, str], dict[str, object]] = {}
    cameras_by_name = {camera.name: camera for camera in cameras}
    expected = {
        (camera.name, modality)
        for camera in cameras
        for modality in camera.settings.modalities
    }
    warmup_physics_steps = 0
    warmup_step_limit = _camera_warmup_step_limit(resources, cameras)
    for _attempt in range(warmup_step_limit):
        _step_with_camera_render(runtime, resources=resources)
        warmup_physics_steps += 1
        simulation_step, time_s = _camera_physics_clock(resources)
        _collect_camera_probe_frames(
            cameras=cameras,
            cameras_by_name=cameras_by_name,
            expected=expected,
            frame_indices=frame_indices,
            ready_frames=ready_frames,
            simulation_step=simulation_step,
            time_s=time_s,
            sampler=sample_camera_frames,
        )
        if expected.issubset(ready_frames):
            break

    missing = sorted(expected.difference(ready_frames))
    if missing:
        raise RuntimeError(f"camera frames were not ready after warmup: {missing}")

    render = _render_only_callable(runtime, resources=resources)
    before_step, before_time_s = _camera_physics_clock(resources)
    steady_frames: dict[tuple[str, str], dict[str, object]] = {}
    render_only_updates = 0
    for _attempt in range(CAMERA_STEADY_RENDER_ATTEMPTS):
        render()
        render_only_updates += 1
        final_step, final_time_s = _camera_physics_clock(resources)
        if final_step != before_step or final_time_s != before_time_s:
            raise RuntimeError(
                "camera render-only verification advanced physics: "
                f"before=({before_step}, {before_time_s}), "
                f"after=({final_step}, {final_time_s})"
            )
        _collect_camera_probe_frames(
            cameras=cameras,
            cameras_by_name=cameras_by_name,
            expected=expected,
            frame_indices=frame_indices,
            ready_frames=steady_frames,
            simulation_step=before_step,
            time_s=before_time_s,
            sampler=sample_camera_frames,
        )
        if expected.issubset(steady_frames):
            break
    missing = sorted(expected.difference(steady_frames))
    if missing:
        raise RuntimeError(
            "camera frames were not ready after "
            f"{CAMERA_STEADY_RENDER_ATTEMPTS} render-only updates: {missing}"
        )

    return {
        "performed": True,
        "frames": [steady_frames[key] for key in sorted(expected)],
        "warmup_physics_steps": warmup_physics_steps,
        "render_only_updates": render_only_updates,
        "render_only_verified": True,
    }


def _camera_warmup_step_limit(resources: object, cameras: Sequence[object]) -> int:
    """Cover renderer latency in time, independent of the physics tick rate."""

    physics = getattr(resources, "physics", None)
    get_dt = getattr(physics, "get_physics_dt", None)
    if not callable(get_dt):
        return CAMERA_WARMUP_STEPS
    dt = float(get_dt())
    frequencies = [
        float(value)
        for camera in cameras
        if (value := getattr(getattr(camera, "settings", None), "frequency", None))
        is not None
    ]
    if (
        not np.isfinite(dt)
        or dt <= 0.0
        or not frequencies
        or any(not np.isfinite(value) or value <= 0.0 for value in frequencies)
    ):
        return CAMERA_WARMUP_STEPS
    slowest_frequency_hz = min(frequencies)
    period_steps = math.ceil(1.0 / (slowest_frequency_hz * dt) - 1.0e-12)
    return max(
        CAMERA_WARMUP_STEPS,
        CAMERA_WARMUP_SAMPLE_PERIODS * period_steps,
    )


def _collect_camera_probe_frames(
    *,
    cameras: Sequence[object],
    cameras_by_name: Mapping[str, object],
    expected: set[tuple[str, str]],
    frame_indices: dict[tuple[str, str], int],
    ready_frames: dict[tuple[str, str], dict[str, object]],
    simulation_step: int,
    time_s: float,
    sampler: Callable[..., Sequence[object]],
) -> None:
    """Sample one rendered generation and retain only contract-valid payloads."""

    for camera in cameras:
        for frame in sampler(
            camera,
            frame_indices=frame_indices,
            simulation_step=simulation_step,
            time_s=time_s,
        ):
            key = (frame.camera_name, frame.modality)
            configured_camera = cameras_by_name.get(frame.camera_name)
            if configured_camera is None or key not in expected:
                raise RuntimeError(f"camera probe received unexpected frame: {key}")
            item = _camera_frame_report_item(camera=configured_camera, frame=frame)
            if item is not None:
                ready_frames[key] = item


def _step_with_camera_render(runtime: object, *, resources: object) -> None:
    """优先走产品级 step/render，测试资源则回落到窄 physics adapter。"""

    step = getattr(runtime, "step", None)
    if runtime is not resources and callable(step):
        step(render=True)
        return
    resources.physics.step(render=True)


def _render_only_callable(
    runtime: object,
    *,
    resources: object | None = None,
) -> Callable[[], None]:
    """Resolve a camera render API that never advances the physics owner."""

    resources = _scene_resources(runtime) if resources is None else resources
    product_render = getattr(runtime, "render", None)
    if runtime is not resources and callable(product_render):
        return product_render
    session = getattr(resources, "session", None)
    manager = getattr(session, "physics_runtime", None)
    render = getattr(manager, "render", None)
    if not callable(render):
        render = getattr(getattr(resources, "physics", None), "render", None)
    if not callable(render):
        raise RuntimeError("Mirror camera probe requires a render-only API")
    return render


def _camera_physics_clock(runtime: object) -> tuple[int, float]:
    """Read a stable physics step/time pair for render-only frame metadata."""

    physics = getattr(runtime, "physics", None)
    if physics is None:
        raise RuntimeError("Mirror camera probe requires a physics adapter")
    manager = getattr(getattr(runtime, "session", None), "physics_runtime", None)
    manager_time = getattr(manager, "simulation_time", None)
    if manager_time is not None:
        time_s = float(manager_time)
        dt = float(physics.get_physics_dt())
        return int(round(time_s / dt)), time_s
    step = int(getattr(physics, "current_time_step_index", 0))
    world_time = getattr(physics, "current_time", None)
    time_s = (
        step * float(physics.get_physics_dt())
        if world_time is None
        else float(world_time)
    )
    return step, time_s


def _camera_frame_report_item(
    *,
    camera: object,
    frame: object,
) -> dict[str, object] | None:
    """Validate one sampled frame and return ``None`` while its pixels warm up."""

    camera_name = str(camera.name)
    modality = str(frame.modality)
    width, height = camera.settings.resolution
    data = np.asarray(frame.data)
    item: dict[str, object] = {
        "camera": camera_name,
        "modality": modality,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
    }
    if modality == "rgb":
        if data.shape != (height, width, 3) or data.dtype != np.uint8:
            raise RuntimeError(
                f"camera {camera_name!r} RGB contract mismatch: "
                f"shape={data.shape}, dtype={data.dtype}"
            )
    elif modality == "depth" and (
        data.shape != (height, width) or data.dtype != np.float32
    ):
        raise RuntimeError(
            f"camera {camera_name!r} depth contract mismatch: "
            f"shape={data.shape}, dtype={data.dtype}"
        )

    try:
        intrinsics = np.asarray(frame.intrinsics, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"camera {camera_name!r} intrinsics are invalid") from exc
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise RuntimeError(f"camera {camera_name!r} intrinsics are invalid")
    item["intrinsics"] = intrinsics.tolist()

    if modality == "rgb":
        nonzero = int(np.count_nonzero(data))
        if nonzero == 0:
            return None
        item["nonzero_values"] = nonzero
    elif modality == "depth":
        valid = np.isfinite(data) & (data > 0.0)
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            return None
        item["valid_values"] = valid_count
        item["min_valid"] = float(np.min(data[valid]))
    return item


def _newton_contact_probe(
    runtime: object,
    expected_backend: str,
) -> dict[str, object]:
    """读取 Newton CPU/CUDA 当前接触数量并拒绝容量耗尽。"""

    if expected_backend != "newton":
        return {"performed": False, "reason": "not_newton"}
    manager = getattr(getattr(runtime, "session", None), "physics_runtime", None)
    execution = str(getattr(manager, "execution", ""))
    if getattr(manager, "backend", None) != "newton" or execution not in {
        "cpu",
        "cuda",
    }:
        raise RuntimeError("Mirror Newton smoke requires the project Newton runtime")
    diagnostics = getattr(manager, "diagnostics", None)
    if not callable(diagnostics):
        raise RuntimeError("Newton runtime does not expose diagnostics")
    manager_diagnostics = dict(diagnostics())
    solver = getattr(manager, "solver", None)
    if solver is None:
        raise RuntimeError("Newton solver is unavailable during contact probe")
    if execution == "cuda":
        contact_counts = _tensor_int_vector(
            getattr(getattr(solver, "mjw_data", None), "nacon", None),
            label="Newton CUDA contact counts",
        )
    else:
        mj_data = getattr(solver, "mj_data", None)
        ncon = getattr(mj_data, "ncon", None)
        if type(ncon) is not int or ncon < 0:
            raise RuntimeError("Newton CPU contact count is unavailable")
        contact_counts = np.asarray([ncon], dtype=np.int64)
    nconmax = int(manager_diagnostics["nconmax_per_world"])
    max_contacts = int(np.max(contact_counts)) if contact_counts.size else 0
    if max_contacts >= nconmax:
        raise RuntimeError(
            f"Newton contact capacity exhausted: max_contacts={max_contacts}, "
            f"nconmax={nconmax}"
        )
    return {
        "performed": True,
        "owner": "newton_runtime",
        "execution": execution,
        "nconmax": nconmax,
        "max_contacts": max_contacts,
        "world_contact_counts": contact_counts.tolist(),
        "mujoco_actuator_count": int(solver.mj_model.nu),
    }


def _rigid_velocity_order_probe(
    runtime: object,
    expected_backend: str,
) -> dict[str, object]:
    """用 live rigid view 证明 Newton 速度 ABI 为线速度在前、角速度在后。"""

    if expected_backend != "newton":
        return {"performed": False, "reason": "not_newton"}
    candidates = [
        (str(name), view)
        for name, view in sorted(
            getattr(runtime, "object_state_views", {}).items(),
            key=lambda item: str(item[0]),
        )
        if getattr(view, "root_view", None) is not None
        and getattr(view, "body_view", None) is None
        and getattr(view, "immutable_position", None) is None
        and str(getattr(view, "velocity_capability", "")) == "complete"
    ]
    if not candidates:
        return {
            "performed": False,
            "reason": "no_independently_writable_free_rigid_root",
        }

    object_name, view = candidates[0]
    original = view.root_velocities()
    if original is None:
        raise RuntimeError(
            f"Newton rigid velocity order probe cannot read object {object_name!r}"
        )
    original_linear = _finite_vector(
        original[0], label="original rigid linear velocity", size=3
    )
    original_angular = _finite_vector(
        original[1], label="original rigid angular velocity", size=3
    )
    probe_values = (
        (
            "pure_linear",
            np.asarray([0.03125, -0.046875, 0.0625]),
            np.zeros(3, dtype=float),
        ),
        (
            "pure_angular",
            np.zeros(3, dtype=float),
            np.asarray([0.078125, -0.09375, 0.109375]),
        ),
    )
    results: list[dict[str, object]] = []
    probe_error: BaseException | None = None
    probe_traceback = None
    try:
        for kind, requested_linear, requested_angular in probe_values:
            view.set_root_velocities(requested_linear, requested_angular)
            actual = view.root_velocities()
            if actual is None:
                raise RuntimeError(
                    f"Newton rigid velocity {kind} probe returned no readback"
                )
            actual_linear = _finite_vector(
                actual[0], label=f"{kind} rigid linear velocity", size=3
            )
            actual_angular = _finite_vector(
                actual[1], label=f"{kind} rigid angular velocity", size=3
            )
            if not np.allclose(
                actual_linear,
                requested_linear,
                rtol=0.0,
                atol=STATE_ATOL,
            ) or not np.allclose(
                actual_angular,
                requested_angular,
                rtol=0.0,
                atol=STATE_ATOL,
            ):
                raise RuntimeError(
                    f"Newton rigid velocity {kind} order mismatch: "
                    f"requested_linear={requested_linear.tolist()}, "
                    f"requested_angular={requested_angular.tolist()}, "
                    f"actual_linear={actual_linear.tolist()}, "
                    f"actual_angular={actual_angular.tolist()}"
                )
            results.append(
                {
                    "kind": kind,
                    "requested_linear": requested_linear.tolist(),
                    "requested_angular": requested_angular.tolist(),
                    "readback_linear": actual_linear.tolist(),
                    "readback_angular": actual_angular.tolist(),
                }
            )
    except BaseException as exc:
        probe_error = exc
        probe_traceback = exc.__traceback__
    finally:
        try:
            view.set_root_velocities(original_linear, original_angular)
            restored = view.root_velocities()
            if restored is None:
                raise RuntimeError("restored rigid velocity readback is unavailable")
            restored_linear = _finite_vector(
                restored[0], label="restored rigid linear velocity", size=3
            )
            restored_angular = _finite_vector(
                restored[1], label="restored rigid angular velocity", size=3
            )
            if not np.allclose(
                restored_linear, original_linear, rtol=0.0, atol=STATE_ATOL
            ) or not np.allclose(
                restored_angular, original_angular, rtol=0.0, atol=STATE_ATOL
            ):
                raise RuntimeError(
                    "Newton rigid velocity probe failed to restore state"
                )
        except BaseException as restore_exc:
            if probe_error is None:
                raise
            probe_error.add_note(f"rigid velocity restore also failed: {restore_exc}")
    if probe_error is not None:
        raise probe_error.with_traceback(probe_traceback)

    return {
        "performed": True,
        "object": object_name,
        "contract": "linear_xyz_then_angular_xyz",
        "probes": results,
        "restored": True,
    }


def _tensor_int_vector(value: object, *, label: str) -> np.ndarray:
    """Read a small diagnostic vector without assuming one tensor provider."""

    if value is None:
        raise RuntimeError(f"{label} is unavailable")
    candidate = value
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    result = np.asarray(candidate, dtype=int).reshape(-1)
    if np.any(result < 0):
        raise RuntimeError(f"{label} contains a negative count")
    return result


def _physics_runtime_probe(runtime: object) -> dict[str, object]:
    """Report the actual physics owner without importing a second backend."""

    manager = getattr(getattr(runtime, "session", None), "physics_runtime", None)
    if manager is None:
        return {"performed": False, "reason": "session_runtime_unavailable"}
    diagnostics = getattr(manager, "diagnostics", None)
    if callable(diagnostics):
        return {"performed": True, **dict(diagnostics())}
    return {
        "performed": True,
        "backend": str(getattr(manager, "backend", "unknown")),
        "execution": str(getattr(manager, "execution", "unknown")),
    }


def _dynamic_chain_snapshot_probe(snapshot: object) -> dict[str, object]:
    """Require complete generalized-state round-trip fields for every chain."""

    reports = []
    for name, obj in sorted(getattr(snapshot, "objects", {}).items()):
        body_names = tuple(getattr(obj, "body_names", ()))
        if not body_names:
            continue
        required = (
            "body_positions_local",
            "body_orientations_wxyz",
            "body_linear_velocities",
            "body_angular_velocities",
        )
        missing = [field for field in required if getattr(obj, field, None) is None]
        if missing:
            raise RuntimeError(
                f"dynamic-chain snapshot {name!r} is incomplete: missing={missing}"
            )
        reports.append(
            {
                "name": str(name),
                "body_count": len(body_names),
                "complete_pose_velocity_state": True,
            }
        )
    return {
        "performed": bool(reports),
        "reason": None if reports else "no_dynamic_chain_objects",
        "objects": reports,
    }


def _curobo_fk_probe(runtime: object, robots: Sequence[object]) -> dict[str, object]:
    """按关节名把每个已启用机器人的 Isaac state 重排后执行 cuRobo FK。"""

    candidates = tuple(
        robot
        for robot in robots
        if bool(getattr(robot, "supports_planning", False))
        and getattr(robot, "curobo_config", None) is not None
    )
    if not candidates:
        return {
            "performed": False,
            "reason": "no_curobo_enabled_robot",
            "robot_count": 0,
            "robots": [],
        }

    reports = [_curobo_fk_robot_report(runtime, candidate) for candidate in candidates]
    return {
        "performed": True,
        "robot_count": len(reports),
        "robots": reports,
    }


def _curobo_fk_robot_report(runtime: object, candidate: object) -> dict[str, object]:
    """租用一个机器人的 context，并验证实时关节映射和 TCP FK。"""

    lease: Callable[..., AbstractContextManager[Any]] = runtime.planning_registry.lease
    with lease(int(candidate.robot_id), consumer_role="interactive") as context:
        fk = context.make_forward_kinematics()
        curobo_joint_names = tuple(str(name) for name in fk.joint_names())
        if not curobo_joint_names or len(set(curobo_joint_names)) != len(
            curobo_joint_names
        ):
            raise RuntimeError("cuRobo FK returned invalid joint names")
        articulation = candidate.execution.articulation
        articulation_names = tuple(str(name) for name in articulation.dof_names)
        articulation_index = {
            name: index for index, name in enumerate(articulation_names)
        }
        missing = sorted(set(curobo_joint_names).difference(articulation_index))
        if missing:
            raise RuntimeError(
                f"robot {candidate.label!r} state is missing cuRobo joints: {missing}"
            )
        state_indices = np.asarray(
            [articulation_index[name] for name in curobo_joint_names], dtype=int
        )
        full_positions = _finite_vector(
            articulation.get_joint_positions(),
            label=f"robot {candidate.label!r} joint positions for cuRobo",
            size=len(articulation_names),
        )
        curobo_positions = full_positions[state_indices]
        frame_names = tuple(str(name) for name in fk.frame_names())
        frame_name = str(getattr(context, "default_tcp_frame", ""))
        if not frame_name:
            if not frame_names:
                raise RuntimeError("cuRobo FK did not expose a TCP frame")
            frame_name = frame_names[0]
        if frame_name not in frame_names:
            raise RuntimeError(
                f"cuRobo default TCP frame {frame_name!r} is not registered"
            )
        pose = fk.compute_pose(curobo_positions, frame_name)
        position = _finite_vector(
            pose.position,
            label=f"robot {candidate.label!r} cuRobo FK position",
            size=3,
        )
        orientation = _finite_vector(
            pose.orientation,
            label=f"robot {candidate.label!r} cuRobo FK orientation",
            size=4,
        )
        rotation = np.asarray(pose.rotation_matrix, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise RuntimeError(
                f"robot {candidate.label!r} cuRobo FK rotation matrix is invalid"
            )

    return {
        "robot_id": int(candidate.robot_id),
        "label": str(candidate.label),
        "joint_names": list(curobo_joint_names),
        "articulation_indices": state_indices.tolist(),
        "tcp_frame": frame_name,
        "position": position.tolist(),
        "orientation_wxyz": orientation.tolist(),
    }


def _assert_array_close(
    label: str,
    expected: object,
    actual: object,
    *,
    atol: float = STATE_ATOL,
) -> None:
    if expected is None or actual is None:
        if expected is not actual:
            raise RuntimeError(f"snapshot roundtrip changed presence of {label}")
        return
    left = np.asarray(expected, dtype=float)
    right = np.asarray(actual, dtype=float)
    if left.shape != right.shape or not np.allclose(left, right, rtol=atol, atol=atol):
        raise RuntimeError(f"snapshot roundtrip changed {label}")


def _assert_derived_array_compatible(
    label: str,
    expected: object,
    actual: object,
) -> None:
    """Validate derived maximal state after exact generalized owner restore."""

    if expected is None or actual is None:
        if expected is not actual:
            raise RuntimeError(f"snapshot roundtrip changed presence of {label}")
        return
    left = np.asarray(expected, dtype=float)
    right = np.asarray(actual, dtype=float)
    if left.shape != right.shape:
        raise RuntimeError(f"snapshot roundtrip changed shape of {label}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise RuntimeError(f"snapshot roundtrip found non-finite {label}")


def _assert_quaternion_close(label: str, expected: object, actual: object) -> None:
    if expected is None or actual is None:
        if expected is not actual:
            raise RuntimeError(f"snapshot roundtrip changed presence of {label}")
        return
    left = np.asarray(expected, dtype=float).reshape(-1, 4)
    right = np.asarray(actual, dtype=float).reshape(-1, 4)
    if left.shape != right.shape:
        raise RuntimeError(f"snapshot roundtrip changed shape of {label}")
    equivalent = np.minimum(
        np.linalg.norm(left - right, axis=1),
        np.linalg.norm(left + right, axis=1),
    )
    if not np.all(equivalent <= STATE_ATOL):
        raise RuntimeError(f"snapshot roundtrip changed {label}")


def _assert_snapshot_roundtrip(expected: object, actual: object) -> None:
    """读取恢复后的 canonical state，并允许四元数的等价符号翻转。"""

    if set(expected.robots) != set(actual.robots):
        raise RuntimeError("snapshot roundtrip changed robot labels")
    for label, source in expected.robots.items():
        restored = actual.robots[label]
        if source.joint_names != restored.joint_names:
            raise RuntimeError(
                f"snapshot roundtrip changed robot {label!r} joint names"
            )
        if source.command_joint_names != restored.command_joint_names:
            raise RuntimeError(
                f"snapshot roundtrip changed robot {label!r} command joint names"
            )
        _assert_array_close(
            f"robot {label!r} joint positions",
            source.joint_positions,
            restored.joint_positions,
        )
        _assert_array_close(
            f"robot {label!r} joint velocities",
            source.joint_velocities,
            restored.joint_velocities,
        )
        _assert_array_close(
            f"robot {label!r} command targets",
            source.command_targets,
            restored.command_targets,
        )

    if set(expected.objects) != set(actual.objects):
        raise RuntimeError("snapshot roundtrip changed object names")
    array_fields = (
        "positions_local",
        "linear_velocities",
        "angular_velocities",
        "body_positions_local",
        "body_linear_velocities",
        "body_angular_velocities",
    )
    quaternion_fields = ("orientations_wxyz", "body_orientations_wxyz")
    for name, source in expected.objects.items():
        restored = actual.objects[name]
        if source.body_names != restored.body_names:
            raise RuntimeError(f"snapshot roundtrip changed object {name!r} bodies")
        generalized_owner = _assert_generalized_object_state_equal(
            name,
            source,
            restored,
        )
        for field_name in array_fields:
            label = f"object {name!r} {field_name}"
            if generalized_owner:
                _assert_derived_array_compatible(
                    label,
                    getattr(source, field_name),
                    getattr(restored, field_name),
                )
            else:
                _assert_array_close(
                    label,
                    getattr(source, field_name),
                    getattr(restored, field_name),
                )
        for field_name in quaternion_fields:
            label = f"object {name!r} {field_name}"
            if generalized_owner:
                _assert_derived_array_compatible(
                    label,
                    getattr(source, field_name),
                    getattr(restored, field_name),
                )
            else:
                _assert_quaternion_close(
                    label,
                    getattr(source, field_name),
                    getattr(restored, field_name),
                )


def _assert_generalized_object_state_equal(
    name: str,
    source: object,
    restored: object,
) -> bool:
    """Require exact generalized owner state before allowing twist projection."""

    source_q = getattr(source, "generalized_q", None)
    restored_q = getattr(restored, "generalized_q", None)
    if source_q is None or restored_q is None:
        if source_q is not restored_q:
            raise RuntimeError(
                f"snapshot roundtrip changed object {name!r} generalized state presence"
            )
        return False
    for field in (
        "generalized_signature",
        "generalized_q_names",
        "generalized_qd_names",
    ):
        if tuple(getattr(source, field)) != tuple(getattr(restored, field)):
            raise RuntimeError(f"snapshot roundtrip changed object {name!r} {field}")
    _assert_array_close(
        f"object {name!r} generalized q",
        source_q,
        restored_q,
    )
    _assert_array_close(
        f"object {name!r} generalized qd",
        getattr(source, "generalized_qd"),
        getattr(restored, "generalized_qd"),
    )
    return True


def _capture_drive_targets(
    robots: Sequence[object],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """读取每个 Single articulation 的真实 position/velocity drive targets。"""

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for robot in robots:
        articulation = robot.execution.articulation
        controller = robot.execution.joint_controller
        indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
        snapshot_targets = getattr(controller, "snapshot_control_targets_cache", None)
        cached = (
            snapshot_targets()
            if callable(snapshot_targets)
            else getattr(controller, "last_control_targets", None)
        )
        if cached is not None:
            full_positions = _finite_vector(
                getattr(cached, "positions", None),
                label=f"robot {robot.label!r} cached position drive targets",
                size=int(articulation.num_dof),
            )
            full_velocities = _finite_vector(
                getattr(cached, "velocities", None),
                label=f"robot {robot.label!r} cached velocity drive targets",
                size=int(articulation.num_dof),
            )
            positions = full_positions[indices]
            velocities = full_velocities[indices]
        else:
            get_positions = getattr(articulation, "get_joint_position_targets", None)
            get_velocities = getattr(articulation, "get_joint_velocity_targets", None)
            if callable(get_positions) and callable(get_velocities):
                positions = get_positions(joint_indices=indices)
                velocities = get_velocities(joint_indices=indices)
            else:
                get_action = getattr(articulation, "get_applied_action", None)
                action = get_action() if callable(get_action) else None
                if action is None:
                    raise RuntimeError(
                        f"robot {robot.label!r} does not expose articulation drive targets"
                    )
                full_positions = getattr(action, "joint_positions", None)
                full_velocities = getattr(action, "joint_velocities", None)
                if full_positions is None or full_velocities is None:
                    raise RuntimeError(
                        f"robot {robot.label!r} has incomplete articulation drive targets"
                    )
                positions = np.asarray(full_positions, dtype=float).reshape(-1)[indices]
                velocities = np.asarray(full_velocities, dtype=float).reshape(-1)[
                    indices
                ]
        result[str(robot.label)] = (
            _finite_vector(
                positions,
                label=f"robot {robot.label!r} position drive targets",
                size=indices.size,
            ),
            _finite_vector(
                velocities,
                label=f"robot {robot.label!r} velocity drive targets",
                size=indices.size,
            ),
        )
    return result


def _assert_restored_drive_targets(
    expected_snapshot: object,
    *,
    velocity_targets_before_restore: Mapping[str, tuple[np.ndarray, np.ndarray]],
    actual: Mapping[str, tuple[np.ndarray, np.ndarray]],
    phase: str,
) -> None:
    if set(expected_snapshot.robots) != set(actual):
        raise RuntimeError(f"{phase} changed articulation drive target availability")
    for label, source in expected_snapshot.robots.items():
        _assert_array_close(
            f"{phase} robot {label!r} position drive targets",
            source.command_targets,
            actual[label][0],
        )
        _assert_array_close(
            f"{phase} robot {label!r} velocity drive targets",
            velocity_targets_before_restore[label][1],
            actual[label][1],
        )


def _assert_post_step_joint_state_stable(expected: object, actual: object) -> None:
    for label, source in expected.robots.items():
        restored = actual.robots[label]
        left = np.asarray(source.joint_positions, dtype=float)
        right = np.asarray(restored.joint_positions, dtype=float)
        if left.shape != right.shape or not np.all(np.isfinite(right)):
            raise RuntimeError(f"post-step robot {label!r} joint state is invalid")
        max_difference = float(np.max(np.abs(left - right), initial=0.0))
        if max_difference > POST_STEP_JOINT_POSITION_ATOL:
            raise RuntimeError(
                f"post-step robot {label!r} positions drifted from restored state: "
                f"max_abs_diff={max_difference}, "
                f"tolerance={POST_STEP_JOINT_POSITION_ATOL}"
            )


def _capture_probe_snapshot(resources: object) -> object:
    """从 Mirror scene resources 捕获 typed SceneSnapshot。"""

    return get_mirror_snapshot(resources)


def _restore_probe_snapshot(resources: object, snapshot: object) -> object:
    """把 typed SceneSnapshot 严格恢复到同一 Mirror scene resources。"""

    return set_mirror_snapshot(resources, snapshot, strict=True)


def probe_mirror_runtime(
    runtime: object,
    *,
    expected_backend: object,
    steps: int,
    active_backend_getter: Callable[[], str] = active_physics_backend,
    snapshot_getter: Callable[[object], object] = _capture_probe_snapshot,
    snapshot_setter: Callable[[object, object], object] = _restore_probe_snapshot,
    contact_probe: Callable[[object, str], dict[str, object]] = _newton_contact_probe,
    rigid_velocity_probe: Callable[[object, str], dict[str, object]] = (
        _rigid_velocity_order_probe
    ),
    control_mode_probe: Callable[..., dict[str, object]] = (
        _exercise_runtime_control_modes
    ),
) -> dict[str, object]:
    """执行不拥有 runtime 生命周期的 Mirror 集成检查。"""

    expected = normalize_physics_backend(expected_backend)
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    active = normalize_physics_backend(active_backend_getter())
    if active != expected:
        raise RuntimeError(
            f"active physics backend mismatch: expected={expected!r}, actual={active!r}"
        )
    resources = _scene_resources(runtime)
    robots = _ordered_robots(resources)
    active_mjc_actuators = _assert_no_active_mjc_actuators(resources)
    initial_inspections = [_inspect_articulation(robot) for robot in robots]
    control_modes = control_mode_probe(
        runtime,
        resources,
        robots,
        steps=steps,
    )
    hybrid_control = _exercise_runtime_hybrid_control(runtime, resources, robots)
    original_snapshot = snapshot_getter(resources)
    dynamic_chain_snapshot = _dynamic_chain_snapshot_probe(original_snapshot)
    rigid_velocity_report = rigid_velocity_probe(resources, active)
    curobo_report = _curobo_fk_probe(resources, robots)
    camera_report = _camera_probe(runtime)

    control_response = _apply_position_targets(resources, robots, steps)
    contact_report = contact_probe(resources, active)
    stepped_inspections = [_inspect_articulation(robot) for robot in robots]
    drive_targets_before_restore = _capture_drive_targets(robots)
    restore = snapshot_setter(resources, original_snapshot)
    if not bool(getattr(restore, "accepted", False)):
        raise RuntimeError("snapshot restore was not accepted")
    if bool(getattr(restore, "partial", False)):
        raise RuntimeError("snapshot restore was partial")
    restored_snapshot = snapshot_getter(resources)
    _assert_snapshot_roundtrip(original_snapshot, restored_snapshot)
    restored_inspections = [_inspect_articulation(robot) for robot in robots]
    drive_targets_after_write = _capture_drive_targets(robots)
    _assert_restored_drive_targets(
        original_snapshot,
        velocity_targets_before_restore=drive_targets_before_restore,
        actual=drive_targets_after_write,
        phase="snapshot restore",
    )
    resources.physics.step(render=False)
    post_step_snapshot = snapshot_getter(resources)
    _assert_restored_drive_targets(
        original_snapshot,
        velocity_targets_before_restore=drive_targets_before_restore,
        actual=_capture_drive_targets(robots),
        phase="snapshot restore post-step",
    )
    _assert_post_step_joint_state_stable(original_snapshot, post_step_snapshot)

    return {
        "event": "mirror_physics_smoke",
        "physics_backend": active,
        "scene": str(getattr(getattr(resources, "scene", None), "scene_id", "")),
        "steps": int(steps),
        "robot_count": len(robots),
        "articulations_initial": initial_inspections,
        "articulations_after_step": stepped_inspections,
        "articulations_after_restore": restored_inspections,
        "control_response": control_response,
        "control_modes": control_modes,
        "hybrid_control": hybrid_control,
        "active_mjc_actuators": active_mjc_actuators,
        "camera": camera_report,
        "physics_runtime": _physics_runtime_probe(resources),
        "newton_contacts": contact_report,
        "rigid_velocity_order": rigid_velocity_report,
        "dynamic_chain_snapshot": dynamic_chain_snapshot,
        "snapshot": {
            "accepted": True,
            "partial": False,
            "robots": sorted(original_snapshot.robots),
            "objects": sorted(original_snapshot.objects),
            "readback_verified": True,
            "drive_targets_verified": True,
            "post_step_readback_verified": True,
            "post_step_physics_steps": 1,
        },
        "curobo_fk": curobo_report,
    }


def probe_mirror_control_modes(
    runtime: object,
    *,
    expected_backend: object,
    steps: int,
    active_backend_getter: Callable[[], str] = active_physics_backend,
) -> dict[str, object]:
    """Run the focused all-mode smoke without unrelated camera/planner probes."""

    expected = normalize_physics_backend(expected_backend)
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    active = normalize_physics_backend(active_backend_getter())
    if active != expected:
        raise RuntimeError(
            f"active physics backend mismatch: expected={expected!r}, actual={active!r}"
        )
    resources = _scene_resources(runtime)
    robots = _ordered_robots(resources)
    _assert_no_active_mjc_actuators(resources)
    initial = [_inspect_articulation(robot) for robot in robots]
    control_modes = _exercise_runtime_control_modes(
        runtime,
        resources,
        robots,
        steps=steps,
    )
    return {
        "event": "mirror_control_mode_smoke",
        "physics_backend": active,
        "scene": str(getattr(getattr(resources, "scene", None), "scene_id", "")),
        "steps": int(steps),
        "robot_count": len(robots),
        "articulations_initial": initial,
        "articulations_after_modes": [_inspect_articulation(robot) for robot in robots],
        "control_modes": control_modes,
        "physics_runtime": _physics_runtime_probe(resources),
    }


def _close_report_payload(close_report: object) -> dict[str, object]:
    if close_report is None:
        return {"stopped": True, "live_resources": []}
    stopped = bool(getattr(close_report, "stopped", False))
    live_resources = [str(name) for name in getattr(close_report, "live_resources", ())]
    if not stopped:
        raise RuntimeError(
            f"MirrorSceneResources close timed out with live resources: {live_resources}"
        )
    return {"stopped": True, "live_resources": live_resources}


def execute_smoke(
    args: argparse.Namespace,
    *,
    before_close: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """解析配置、创建 runtime、执行 smoke，并在所有路径关闭 runtime。"""

    config = resolve_smoke_config(args)
    if config.control.mode != "position":
        raise RuntimeError("Mirror physics smoke requires position control mode")
    runtime = create_smoke_runtime(config)
    expected_backend = str(config.physics.engine)
    try:
        if bool(getattr(args, "control_modes_only", False)):
            report = probe_mirror_control_modes(
                runtime,
                expected_backend=expected_backend,
                steps=args.steps,
            )
        else:
            report = probe_mirror_runtime(
                runtime,
                expected_backend=expected_backend,
                steps=args.steps,
            )
    except BaseException as exc:
        traceback.print_exception(exc)
        try:
            close_report = runtime.close()
            if close_report is not None and not bool(close_report.stopped):
                exc.add_note(
                    "runtime close timed out: "
                    f"live_resources={list(close_report.live_resources)}"
                )
        except BaseException as close_exc:
            exc.add_note(
                f"runtime close raised {type(close_exc).__name__}: {close_exc}"
            )
        raise
    report["shutdown"] = {"application_close_requested": True}
    if before_close is not None:
        before_close(report)
    report["shutdown"] = _close_report_payload(runtime.close())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """输出关闭前运行态 marker；完整成功还要求进程退出码为零。"""

    arguments = list(sys.argv[1:] if argv is None else argv)
    # 参数解析必须位于 supervisor 之前：--help 和非法参数是纯 CLI 冷路径，不应启动
    # Isaac worker，也不应因为缺少物理成功 marker 被改写为失败。
    parsed = parse_args(arguments)
    if not in_runtime_worker():
        return run_supervised_worker(
            script_path=Path(__file__),
            argv=arguments,
            required_markers=(RUNTIME_SUCCESS_MARKER,),
            success_marker=SUCCESS_MARKER,
        )

    def print_before_fast_shutdown(report: dict[str, object]) -> None:
        print(
            RUNTIME_SUCCESS_MARKER
            + " "
            + strict_json_dumps(report, ensure_ascii=True, sort_keys=True),
            flush=True,
        )

    execute_smoke(parsed, before_close=print_before_fast_shutdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
