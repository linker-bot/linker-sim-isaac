from __future__ import annotations

from dataclasses import replace
from threading import get_ident, Thread
from types import MappingProxyType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from linkerbot_sim.configuration import load_kaleidoscope_config  # noqa: E402
from linkerbot_sim.kaleidoscope.actions import (  # noqa: E402
    ActionMode,
    ActionSpec,
    IKRuntimeAction,
    JointControlRuntimeAction,
    JointDeltaActionTerm,
    JointDeltaRuntimeAction,
    KinematicsRobotBinding,
    LinearRuntimeAction,
    action_spec_from_configuration,
)
from linkerbot_sim.kaleidoscope.control_commands import (  # noqa: E402
    EffortControlTrajectory,
    PositionControlTrajectory,
    VelocityControlTrajectory,
)
from linkerbot_sim.kaleidoscope.env import TorchKaleidoscopeEnv  # noqa: E402
from linkerbot_sim.controllers.control_mode import (  # noqa: E402
    ControlModeLockedError,
)
from linkerbot_sim.kaleidoscope.collision_query import (  # noqa: E402
    ArticulatedCollisionQueryResult,
    ArticulatedCollisionQuerySpec,
    CollisionHitCategory,
    CollisionQueryTarget,
    CollisionQueryUnavailableError,
)
from linkerbot_sim.kaleidoscope.geometry import quaternion_slerp_wxyz  # noqa: E402
from linkerbot_sim.kaleidoscope.ik import (  # noqa: E402
    BatchIKTensorResult,
    BatchIKWaypointTensorResult,
)
from linkerbot_sim.kaleidoscope.linear_motion import solve_linear_motion_batch  # noqa: E402
from linkerbot_sim.kaleidoscope.observations import (  # noqa: E402
    TBlockState,
    tblock_heading,
)
from linkerbot_sim.kaleidoscope.reset_projection import (  # noqa: E402
    project_reset_description,
)
from linkerbot_sim.kaleidoscope.reset_sampling import (  # noqa: E402
    sample_reset_scene_description,
)
from linkerbot_sim.kaleidoscope.resets import (  # noqa: E402
    DeviceCounterRNG,
    TBlockResetCommand,
    build_tblock_reset_command,
    tblock_reset_command_from_payload,
)
from linkerbot_sim.kaleidoscope.runtime import (  # noqa: E402
    KaleidoscopeRuntime,
    SameStepToken,
)
from linkerbot_sim.kaleidoscope.snapshot import (  # noqa: E402
    KaleidoscopeEpisodeSnapshot,
)
from linkerbot_sim.kaleidoscope.state_api import (  # noqa: E402
    KaleidoscopeStateAPI,
    StateBinding,
)
from linkerbot_sim.kaleidoscope.tasks.tblock_push_v1 import (  # noqa: E402
    TBlockPushV1,
    TBlockPushV1Settings,
)
from linkerbot_sim.utils.rotations import rpy_xyz_to_quat_wxyz  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Kaleidoscope contracts"
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _task(
    *,
    num_envs: int = 4,
    command_dim: int = 6,
    settings: TBlockPushV1Settings | None = None,
) -> TBlockPushV1:
    device = _device()
    return TBlockPushV1(
        num_envs=num_envs,
        command_dim=command_dim,
        action_dim=command_dim,
        robot_count=2,
        device=device,
        dtype=torch.float32,
        nominal_joint_positions=torch.zeros(command_dim, device=device),
        nominal_block_position=torch.tensor([0.0, 0.0, -0.38], device=device),
        nominal_block_orientation_wxyz=torch.tensor(
            [0.70714084, 0.0, 0.70707272, 0.0], device=device
        ),
        settings=settings,
    )


def _state_from_reset(command, *, command_dim: int = 6) -> TBlockState:
    count = command.env_ids.numel()
    block = command.block_position
    tcp = torch.stack(
        (
            block + torch.tensor([0.0, -0.03, 0.02], device=block.device),
            block + torch.tensor([0.0, 0.03, 0.02], device=block.device),
        ),
        dim=1,
    )
    tcp_q = torch.zeros((count, 2, 4), device=block.device)
    tcp_q[..., 0] = 1.0
    return TBlockState(
        joint_positions=command.joint_positions[:, :command_dim],
        joint_velocities=command.joint_velocities[:, :command_dim],
        command_targets=command.joint_targets[:, :command_dim].clone(),
        tcp_positions_local=tcp,
        tcp_orientations_wxyz=tcp_q,
        block_position_local=command.block_position,
        block_orientation_wxyz=command.block_orientation_wxyz,
        block_com_velocity=command.block_velocity,
        external_safety_stop=torch.zeros(count, device=block.device, dtype=torch.bool),
    )


def test_action_spec_and_joint_delta_targets_stay_on_cuda() -> None:
    configured_action = replace(
        load_kaleidoscope_config().task.action,
        clip=0.25,
    )
    spec = action_spec_from_configuration(
        configured_action,
        robot_labels=("left", "right"),
        command_dims=(2, 2),
    )
    assert spec.action_dim == 4
    assert spec.robot_slices() == {"left": slice(0, 2), "right": slice(2, 4)}
    lower = torch.full((4,), -0.1, device=_device())
    upper = torch.full((4,), 0.1, device=_device())
    term = JointDeltaActionTerm(
        lower=lower,
        upper=upper,
        scale=torch.full((4,), 0.05, device=_device()),
        clip=spec.clip,
        num_envs=2,
        target=torch.zeros((2, 4), device=_device()),
    )
    term.reset_targets(torch.zeros((2, 4), device=_device()))
    result = term.apply(torch.full((2, 4), 10.0, device=_device()))
    torch.testing.assert_close(result, torch.full_like(result, 0.0125))
    runtime_action = JointDeltaRuntimeAction(term, physics_ticks_per_action=2)
    assert runtime_action.action_low == -0.25
    assert runtime_action.action_high == 0.25
    assert result.device.type == "cuda"


def _joint_action_term(*, target: torch.Tensor, clip: float = 2.0):
    width = target.shape[1]
    return JointDeltaActionTerm(
        lower=torch.full((width,), -0.5, device=target.device),
        upper=torch.full((width,), 0.5, device=target.device),
        scale=torch.tensor([0.1, 0.2], device=target.device)[:width],
        clip=clip,
        num_envs=target.shape[0],
        target=target,
    )


def test_joint_control_position_matches_joint_delta_numerically() -> None:
    initial = torch.tensor([[0.45, -0.45], [0.0, 0.1]], device=_device())
    joint_control_term = _joint_action_term(target=initial.clone())
    legacy_term = _joint_action_term(target=initial.clone())
    joint_control = JointControlRuntimeAction(
        joint_control_term,
        physics_ticks_per_action=3,
        velocity_scale_rad_s=4.0,
        effort_limits=torch.tensor([10.0, 20.0], device=_device()),
        effort_limit_fraction=0.25,
        physics_dt=0.02,
    )
    legacy = JointDeltaRuntimeAction(legacy_term, physics_ticks_per_action=3)
    actions = torch.tensor([[4.0, -4.0], [1.0, -1.0]], device=_device())
    state = SimpleNamespace(
        joint_positions=torch.zeros_like(initial),
        position_references=initial.clone(),
    )

    result = joint_control.apply(actions, state, "position")
    expected = legacy.apply(actions, state, "position")

    assert isinstance(result.control, PositionControlTrajectory)
    torch.testing.assert_close(result.position_reference, expected.position_reference)
    torch.testing.assert_close(result.control.positions, expected.control.positions)
    torch.testing.assert_close(
        result.control.velocities,
        torch.zeros_like(result.control.velocities),
    )


def test_joint_control_velocity_and_effort_use_normalized_units_and_reference_rules() -> (
    None
):
    initial = torch.tensor([[0.45, -0.45], [0.0, 0.1]], device=_device())
    term = _joint_action_term(target=initial.clone())
    action = JointControlRuntimeAction(
        term,
        physics_ticks_per_action=4,
        velocity_scale_rad_s=3.0,
        effort_limits=torch.tensor([10.0, 20.0], device=_device()),
        effort_limit_fraction=0.25,
        physics_dt=0.1,
    )
    state = SimpleNamespace(
        joint_positions=torch.tensor([[0.2, 0.3], [-0.1, 0.4]], device=_device()),
        position_references=initial.clone(),
    )
    actions = torch.tensor([[4.0, -1.0], [-4.0, 1.0]], device=_device())

    velocity = action.apply(actions, state, "velocity")
    expected_velocity = torch.tensor([[3.0, -1.5], [-3.0, 1.5]], device=_device())
    expected_reference = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device=_device())
    assert isinstance(velocity.control, VelocityControlTrajectory)
    torch.testing.assert_close(
        velocity.control.velocities,
        expected_velocity[None].expand(4, -1, -1),
    )
    torch.testing.assert_close(velocity.position_reference, expected_reference)
    torch.testing.assert_close(term.target, expected_reference)

    effort = action.apply(actions, state, "effort")
    expected_effort = torch.tensor([[2.5, -2.5], [-2.5, 2.5]], device=_device())
    assert isinstance(effort.control, EffortControlTrajectory)
    torch.testing.assert_close(
        effort.control.efforts,
        expected_effort[None].expand(4, -1, -1),
    )
    torch.testing.assert_close(effort.position_reference, state.joint_positions)
    torch.testing.assert_close(term.target, state.joint_positions)


def test_joint_control_rejects_wrong_action_width() -> None:
    target = torch.zeros((1, 2), device=_device())
    action = JointControlRuntimeAction(
        _joint_action_term(target=target),
        physics_ticks_per_action=2,
        velocity_scale_rad_s=1.0,
        effort_limits=torch.ones(2, device=_device()),
        effort_limit_fraction=1.0,
        physics_dt=0.01,
    )
    state = SimpleNamespace(
        joint_positions=target.clone(),
        position_references=target.clone(),
    )

    with pytest.raises(ValueError, match="shape/device"):
        action.apply(torch.zeros((1, 3), device=_device()), state, "velocity")


def test_joint_control_runs_finite_guard_before_interpreting_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.zeros((1, 2), device=_device())
    action = JointControlRuntimeAction(
        _joint_action_term(target=target),
        physics_ticks_per_action=2,
        velocity_scale_rad_s=1.0,
        effort_limits=torch.ones(2, device=_device()),
        effort_limit_fraction=1.0,
        physics_dt=0.01,
    )
    state = SimpleNamespace(
        joint_positions=target.clone(),
        position_references=target.clone(),
    )
    checked: list[str] = []

    def finite_guard(value: torch.Tensor, *, name: str) -> None:
        checked.append(name)
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError(f"{name} must be finite")

    monkeypatch.setattr(
        "linkerbot_sim.kaleidoscope.actions.assert_finite_async", finite_guard
    )
    with pytest.raises(ValueError, match="must be finite"):
        action.apply(
            torch.full((1, 2), float("nan"), device=_device()),
            state,
            "velocity",
        )
    assert checked == ["joint control actions"]


def test_gpu_snapshot_and_clone_are_owned_and_rng_aware() -> None:
    live = torch.arange(12, device=_device(), dtype=torch.float32).reshape(3, 4)
    rng = torch.tensor([11, 22, 33], device=_device(), dtype=torch.int64)
    api = KaleidoscopeStateAPI(
        {"q": StateBinding(live), "rng.key": StateBinding(rng, finite=False)},
        num_envs=3,
        rng_fields=("rng.key",),
    )
    source = torch.tensor([0], device=_device())
    target = torch.tensor([2], device=_device())
    snapshot = api.snapshot(source)
    assert (
        snapshot.fields["q"].untyped_storage().data_ptr()
        != live.untyped_storage().data_ptr()
    )
    api.clone_state(source, target)
    torch.cuda.synchronize()
    torch.testing.assert_close(live[2], live[0])
    assert rng[2].item() == rng[0].item()
    live[0].add_(100)
    torch.testing.assert_close(
        snapshot.fields["q"][0],
        torch.arange(4, device=_device(), dtype=torch.float32),
    )


def test_snapshot_rejects_incompatible_configuration_before_engine_write() -> None:
    source = torch.zeros((2, 3), device=_device())
    target = torch.ones((2, 3), device=_device())
    writes: list[torch.Tensor] = []
    source_api = KaleidoscopeStateAPI(
        {"robot.q": StateBinding(source)},
        num_envs=2,
        compatibility_fingerprint="task-profile-a",
    )
    target_api = KaleidoscopeStateAPI(
        {
            "robot.q": StateBinding(
                target,
                lambda _ids, values: writes.append(values.clone()),
            )
        },
        num_envs=2,
        compatibility_fingerprint="task-profile-b",
    )

    with pytest.raises(ValueError, match="incompatible"):
        target_api.restore_snapshot(source_api.snapshot())
    assert writes == []
    assert target_api.poisoned is False
    torch.testing.assert_close(target, torch.ones_like(target))


def test_snapshot_mode_metadata_restore_and_position_preflight_are_transactional() -> (
    None
):
    target = torch.zeros((2, 3), device=_device())
    reference = torch.zeros_like(target)
    writes: list[torch.Tensor] = []

    def writer(_ids, values) -> None:
        writes.append(values.clone())

    api = KaleidoscopeStateAPI(
        {
            "robot.target": StateBinding(target, writer),
            "robot.position_reference": StateBinding(reference),
        },
        num_envs=2,
        compatibility_fingerprint="mode-metadata",
    )
    mode = SimpleNamespace(active_mode="position", generation=4)
    provider_calls = 0

    def provider():
        nonlocal provider_calls
        provider_calls += 1
        return mode

    api.bind_control_mode_provider(provider)
    provider_calls = 0
    snapshot = api.snapshot()
    assert provider_calls == 1
    assert snapshot.control_mode == "position"
    assert snapshot.control_generation == 4

    mode.generation = 9
    api.restore_snapshot(snapshot)
    assert mode.generation == 9
    assert writes

    writes.clear()
    mode.active_mode = "velocity"
    with pytest.raises(ValueError, match="does not match runtime mode"):
        api.restore_snapshot(snapshot)
    assert writes == []

    mode.active_mode = "position"
    with pytest.raises(ValueError, match="must match"):
        api.set_state(
            {
                "robot.target": torch.zeros_like(target),
                "robot.position_reference": torch.ones_like(reference),
            }
        )
    assert writes == []


def test_schema_one_snapshot_uses_legacy_fingerprint_and_derives_reference() -> None:
    target = torch.zeros((2, 3), device=_device())
    reference = torch.zeros_like(target)

    def writer(ids, values) -> None:
        target.index_copy_(0, ids, values)

    current = KaleidoscopeStateAPI(
        {
            "robot.target": StateBinding(target, writer),
            "robot.position_reference": StateBinding(reference),
        },
        num_envs=2,
        compatibility_fingerprint="legacy-position",
    )
    legacy = KaleidoscopeStateAPI(
        {"robot.target": StateBinding(torch.zeros_like(target), writer)},
        num_envs=2,
        compatibility_fingerprint="legacy-position",
    )
    values = torch.full_like(target, 0.25)
    snapshot = KaleidoscopeEpisodeSnapshot(
        env_ids=torch.arange(2, device=_device()),
        fields={"robot.target": values},
        compatibility_fingerprint=legacy.compatibility_fingerprint,
        control_mode=None,
        schema_version=1,
    )

    current.restore_snapshot(snapshot)

    torch.testing.assert_close(target, values)
    torch.testing.assert_close(reference, values)


def test_masked_reset_preserves_every_non_done_task_and_rng_row() -> None:
    task = _task(num_envs=4)
    ids = torch.arange(4, device=_device(), dtype=torch.int64)
    initial = task.reset_command(ids)
    state = _state_from_reset(initial)
    task.initialize_after_reset(ids, state)
    task.step(state, torch.full((4, 6), 0.2, device=_device()))
    before = {name: value.clone() for name, value in task.state_fields().items()}
    mask = torch.tensor([True, False, True, False], device=_device())

    command = task.masked_reset_command(mask, state)
    non_done = ~mask
    torch.testing.assert_close(
        command.joint_positions[non_done], state.joint_positions[non_done]
    )
    torch.testing.assert_close(
        command.joint_velocities[non_done], state.joint_velocities[non_done]
    )
    torch.testing.assert_close(
        command.joint_targets[non_done], state.command_targets[non_done]
    )
    torch.testing.assert_close(
        command.block_position[non_done], state.block_position_local[non_done]
    )
    torch.testing.assert_close(
        command.block_orientation_wxyz[non_done],
        state.block_orientation_wxyz[non_done],
    )
    torch.testing.assert_close(
        command.block_velocity[non_done], state.block_com_velocity[non_done]
    )
    refreshed = _state_from_reset(command)
    task.initialize_after_masked_reset(mask, refreshed)
    after = task.state_fields()
    for name, previous in before.items():
        torch.testing.assert_close(after[name][non_done], previous[non_done])
    assert torch.all(after["rng.counter"][mask] > before["rng.counter"][mask])
    assert torch.all(after["task.episode_length"][mask] == 0)


def test_tblock_reset_step_and_numeric_failure_remain_finite() -> None:
    task = _task(num_envs=4)
    ids = torch.arange(4, device=_device(), dtype=torch.int64)
    command = task.reset_command(ids)
    state = _state_from_reset(command)
    task.initialize_after_reset(ids, state)
    result = task.step(state, torch.zeros((4, 6), device=_device()))
    assert result.observations.shape == (4, task.observation_dim)
    assert torch.all(torch.isfinite(result.observations))
    assert torch.all(torch.isfinite(result.rewards))

    broken_q = state.block_orientation_wxyz.clone()
    broken_q[1, 0] = torch.nan
    broken = replace(state, block_orientation_wxyz=broken_q)
    failed = task.step(broken, torch.full((4, 6), 0.75, device=_device()))
    torch.cuda.synchronize()
    assert failed.truncated[1]
    assert failed.rewards[1] == 0.0
    assert torch.all(torch.isfinite(failed.observations))
    failed_observation_action = failed.observations[1, -(task.action_dim + 1) : -1]
    torch.testing.assert_close(
        failed_observation_action,
        task.buffers.previous_action[1],
    )
    torch.testing.assert_close(
        task.buffers.previous_action[1],
        torch.zeros(task.action_dim, device=_device()),
    )


def test_tblock_heading_is_relative_to_nominal_for_reset_goal_and_observation() -> None:
    settings = TBlockPushV1Settings(
        block_yaw_delta_rad=(0.0, 0.0),
        goal_yaw_delta_rad=(0.0, 0.0),
    )
    task = _task(num_envs=3, settings=settings)
    ids = torch.arange(3, device=_device(), dtype=torch.int64)
    command = task.reset_command(ids)
    state = _state_from_reset(command)
    task.initialize_after_reset(ids, state)
    torch.cuda.synchronize()

    # nominal quaternion 的绝对 +Y heading 约为 pi/2；任务坐标系必须将其归零。
    torch.testing.assert_close(command.goal_yaw, torch.zeros_like(command.goal_yaw))
    torch.testing.assert_close(
        task.buffers.previous_heading_error,
        torch.zeros_like(task.buffers.previous_heading_error),
        atol=1.0e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("seed", "sample_id", "reset_index"),
    (
        (0, 0, 0),
        (7, 12, 4),
        (-17, 2**31 + 5, 2),
    ),
)
def test_pure_reset_description_matches_cuda_reset_command(
    seed: int,
    sample_id: int,
    reset_index: int,
) -> None:
    config = load_kaleidoscope_config()
    description = sample_reset_scene_description(
        config,
        seed=seed,
        sample_id=sample_id,
        reset_index=reset_index,
    )
    randomization = config.task.randomization
    nominal_joint_positions = torch.tensor(
        [
            value
            for robot in config.scene.robots
            for value in robot.ordered_reset_joint_positions()
        ],
        device=_device(),
        dtype=torch.float32,
    )
    dynamic_object = next(
        item for item in config.scene.objects if item.name == config.task.dynamic_object
    )
    nominal_block_position = torch.tensor(
        dynamic_object.root_pose.xyz,
        device=_device(),
        dtype=torch.float32,
    )
    nominal_block_orientation = torch.tensor(
        rpy_xyz_to_quat_wxyz(dynamic_object.root_pose.rpy),
        device=_device(),
        dtype=torch.float32,
    )
    heading_axis = torch.tensor(
        config.task.heading_axis,
        device=_device(),
        dtype=torch.float32,
    )
    nominal_heading, finite = tblock_heading(
        nominal_block_orientation[None, :],
        heading_axis=heading_axis,
    )
    assert torch.all(finite)

    draw_count = nominal_joint_positions.numel() + 6
    rng_counter = torch.tensor(
        [reset_index * draw_count],
        device=_device(),
        dtype=torch.int64,
    )
    rng = DeviceCounterRNG(
        torch.tensor(
            [description.generator.rng_key],
            device=_device(),
            dtype=torch.int64,
        ),
        rng_counter,
    )
    command = build_tblock_reset_command(
        env_ids=torch.zeros(1, device=_device(), dtype=torch.int64),
        rng=rng,
        nominal_joint_positions=nominal_joint_positions,
        nominal_block_position=nominal_block_position,
        nominal_block_orientation_wxyz=nominal_block_orientation,
        heading_axis=heading_axis,
        nominal_heading=nominal_heading[0],
        joint_delta_range=randomization.robot_joint_delta_rad,
        block_x_range=randomization.block_x_delta_m,
        block_y_range=randomization.block_y_delta_m,
        block_yaw_range=randomization.block_yaw_delta_rad,
        goal_x_range=randomization.goal_x_delta_m,
        goal_y_range=randomization.goal_y_delta_m,
        goal_yaw_range=randomization.goal_yaw_delta_rad,
    )

    expected_joint_positions = torch.tensor(
        [value for robot in description.robots for value in robot.joint_positions],
        device=_device(),
        dtype=torch.float32,
    )[None, :]
    expected_block_position = torch.tensor(
        description.dynamic_object.position_xyz,
        device=_device(),
        dtype=torch.float32,
    )[None, :]
    expected_block_orientation = torch.tensor(
        description.dynamic_object.orientation_wxyz,
        device=_device(),
        dtype=torch.float32,
    )[None, :]
    expected_goal_position = torch.tensor(
        description.task.goal_position,
        device=_device(),
        dtype=torch.float32,
    )[None, :]
    expected_goal_yaw = torch.tensor(
        [description.task.goal_yaw],
        device=_device(),
        dtype=torch.float32,
    )

    torch.cuda.synchronize()
    assert rng_counter.item() == description.generator.rng_counter_after
    torch.testing.assert_close(
        command.joint_positions,
        expected_joint_positions,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        command.joint_velocities, torch.zeros_like(command.joint_velocities)
    )
    torch.testing.assert_close(command.joint_targets, expected_joint_positions)
    torch.testing.assert_close(
        command.block_position,
        expected_block_position,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        command.block_velocity, torch.zeros_like(command.block_velocity)
    )
    torch.testing.assert_close(
        command.goal_position,
        expected_goal_position,
        rtol=0.0,
        atol=0.0,
    )
    # CUDA trigonometric kernels can differ from Python libm by a few float32 ULPs.
    torch.testing.assert_close(
        command.block_orientation_wxyz,
        expected_block_orientation,
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        command.goal_yaw,
        expected_goal_yaw,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_validated_reset_payload_copies_to_owned_one_row_cuda_command() -> None:
    config = load_kaleidoscope_config()
    description = sample_reset_scene_description(
        config,
        seed=7,
        sample_id=12,
        reset_index=4,
    )
    payload = project_reset_description(config, description)

    command = tblock_reset_command_from_payload(
        payload,
        env_id=2,
        num_envs=4,
        device=_device(),
    )

    assert command.env_ids.tolist() == [2]
    assert command.joint_positions.shape == (1, 26)
    assert command.joint_velocities.shape == (1, 26)
    assert command.joint_targets.shape == (1, 26)
    assert command.block_position.shape == (1, 3)
    assert command.block_orientation_wxyz.shape == (1, 4)
    assert command.block_velocity.shape == (1, 6)
    assert command.goal_position.shape == (1, 3)
    assert command.goal_yaw.shape == (1,)
    tensors = (
        command.env_ids,
        command.joint_positions,
        command.joint_velocities,
        command.joint_targets,
        command.block_position,
        command.block_orientation_wxyz,
        command.block_velocity,
        command.goal_position,
        command.goal_yaw,
        command.rng_key,
        command.rng_counter_after,
    )
    assert all(value is not None and value.device == _device() for value in tensors)
    assert all(value is not None and value.is_contiguous() for value in tensors)
    assert command.rng_key is not None
    assert command.rng_counter_after is not None
    assert command.rng_key.item() == description.generator.rng_key
    assert command.rng_counter_after.item() == description.generator.rng_counter_after
    torch.testing.assert_close(
        command.joint_positions,
        torch.tensor(
            [payload.joint_positions],
            device=_device(),
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        command.goal_yaw,
        torch.tensor([payload.goal_yaw], device=_device(), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="env_id"):
        tblock_reset_command_from_payload(
            payload,
            env_id=4,
            num_envs=4,
            device=_device(),
        )
    with pytest.raises(ValueError, match="CUDA device"):
        tblock_reset_command_from_payload(
            payload,
            env_id=0,
            num_envs=4,
            device=torch.device("cpu"),
        )


def test_task_stages_one_description_goal_and_rng_row_transactionally() -> None:
    config = load_kaleidoscope_config()
    description = sample_reset_scene_description(
        config,
        seed=7,
        sample_id=12,
        reset_index=4,
    )
    payload = project_reset_description(config, description)
    command = tblock_reset_command_from_payload(
        payload,
        env_id=2,
        num_envs=4,
        device=_device(),
    )
    task = _task(num_envs=4, command_dim=26)
    task.buffers.goal_position.fill_(-9.0)
    task.buffers.goal_yaw.fill_(-9.0)
    task.buffers.rng_key.fill_(-9)
    task.buffers.rng_counter.fill_(99)

    task.stage_reset_description(command)

    torch.testing.assert_close(
        task.buffers.goal_position[2],
        command.goal_position[0],
    )
    torch.testing.assert_close(task.buffers.goal_yaw[2], command.goal_yaw[0])
    assert command.rng_key is not None
    assert command.rng_counter_after is not None
    assert task.buffers.rng_key[2] == command.rng_key[0]
    assert task.buffers.rng_counter[2] == command.rng_counter_after[0]
    untouched = torch.tensor([0, 1, 3], device=_device(), dtype=torch.int64)
    torch.testing.assert_close(
        task.buffers.goal_position.index_select(0, untouched),
        torch.full((3, 3), -9.0, device=_device()),
    )
    torch.testing.assert_close(
        task.buffers.goal_yaw.index_select(0, untouched),
        torch.full((3,), -9.0, device=_device()),
    )
    assert torch.all(task.buffers.rng_key.index_select(0, untouched) == -9)
    assert torch.all(task.buffers.rng_counter.index_select(0, untouched) == 99)

    before = {
        "goal_position": task.buffers.goal_position.clone(),
        "goal_yaw": task.buffers.goal_yaw.clone(),
        "rng_key": task.buffers.rng_key.clone(),
        "rng_counter": task.buffers.rng_counter.clone(),
    }
    invalid_goal = command.goal_position.clone()
    invalid_goal[0, 0] = torch.nan
    invalid_counter = torch.full((1,), -1, device=_device(), dtype=torch.int64)
    invalid_commands = (
        (replace(command, rng_key=None), "RNG key"),
        (replace(command, goal_position=invalid_goal), "goal must be finite"),
        (
            replace(command, rng_counter_after=invalid_counter),
            "counter must be non-negative",
        ),
        (
            replace(
                command,
                env_ids=torch.tensor([4], device=_device(), dtype=torch.int64),
            ),
            "out of range",
        ),
    )
    for invalid, expected in invalid_commands:
        with pytest.raises(ValueError, match=expected):
            task.stage_reset_description(invalid)
        for name, previous in before.items():
            torch.testing.assert_close(getattr(task.buffers, name), previous)


def test_step_observation_previous_action_matches_same_step_buffer() -> None:
    task = _task(num_envs=2)
    ids = torch.arange(2, device=_device(), dtype=torch.int64)
    command = task.reset_command(ids)
    state = _state_from_reset(command)
    task.initialize_after_reset(ids, state)

    for value in (0.2, -0.35):
        action = torch.full((2, task.action_dim), value, device=_device())
        result = task.step(state, action)
        observation_action = result.observations[:, -(task.action_dim + 1) : -1]
        torch.testing.assert_close(observation_action, action)
        torch.testing.assert_close(observation_action, task.buffers.previous_action)


def test_tblock_success_requires_five_consecutive_decisions() -> None:
    task = _task(num_envs=2)
    ids = torch.arange(2, device=_device(), dtype=torch.int64)
    command = task.reset_command(ids)
    state = _state_from_reset(command)
    heading_axis = torch.tensor([0.0, 1.0, 0.0], device=_device())
    nominal_orientation = torch.tensor(
        [[0.70714084, 0.0, 0.70707272, 0.0]], device=_device()
    )
    nominal_heading, nominal_finite = tblock_heading(
        nominal_orientation,
        heading_axis=heading_axis,
    )
    heading, finite = tblock_heading(
        state.block_orientation_wxyz,
        heading_axis=heading_axis,
        nominal_heading=nominal_heading[0],
    )
    assert torch.all(nominal_finite)
    assert torch.all(finite)
    task.buffers.goal_position.copy_(state.block_position_local)
    task.buffers.goal_yaw.copy_(heading)
    task.initialize_after_reset(ids, state)
    for index in range(5):
        result = task.step(state, torch.zeros((2, 6), device=_device()))
        if index < 4:
            assert not torch.any(result.terminated)
    torch.cuda.synchronize()
    assert torch.all(result.terminated)
    assert torch.all(result.info["success"])


class _WaypointSolver:
    def __init__(self, device: torch.device, command_dim: int) -> None:
        self.device = device
        self.command_dim = command_dim
        self.inputs = None

    def solve_waypoints(self, **kwargs):
        self.inputs = SimpleNamespace(**kwargs)
        positions = kwargs["target_positions"]
        seeds = kwargs["seeds"]
        steps, num_envs = positions.shape[:2]
        q = seeds[None, :, :].expand(steps, -1, -1).clone()
        q.add_(torch.arange(1, steps + 1, device=self.device)[:, None, None] * 0.01)
        return BatchIKWaypointTensorResult(
            joint_positions=q,
            success=torch.ones(num_envs, device=self.device, dtype=torch.bool),
            first_failure_step=torch.full(
                (num_envs,), -1, device=self.device, dtype=torch.int64
            ),
            position_error=torch.zeros(num_envs, device=self.device),
        )


class _CartesianWaypointSolver:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.command_dim = 1
        self.inputs = None

    def solve_waypoints(self, **kwargs):
        self.inputs = SimpleNamespace(**kwargs)
        positions = kwargs["target_positions"]
        num_envs = positions.shape[1]
        return BatchIKWaypointTensorResult(
            joint_positions=positions[..., :1].clone(),
            success=torch.ones(num_envs, device=self.device, dtype=torch.bool),
            first_failure_step=torch.full(
                (num_envs,), -1, device=self.device, dtype=torch.int64
            ),
            position_error=torch.zeros(num_envs, device=self.device),
        )


class _IKSolver:
    def __init__(self, device: torch.device, command_dim: int) -> None:
        self.device = device
        self.command_dim = command_dim

    def solve(self, **kwargs):
        seeds = kwargs["seeds"]
        count = seeds.shape[0]
        success = torch.ones(count, device=self.device, dtype=torch.bool)
        success[-1] = False
        return BatchIKTensorResult(
            joint_positions=seeds + 0.25,
            success=success,
            position_error=torch.zeros(count, device=self.device),
            orientation_error=torch.zeros(count, device=self.device),
        )


def test_linear_motion_uses_cuda_slerp_and_fixed_tick_output() -> None:
    device = _device()
    solver = _WaypointSolver(device, command_dim=3)
    start_p = torch.zeros((2, 3), device=device)
    target_p = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], device=device)
    start_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device)
    target_q = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2, device=device)
    result = solve_linear_motion_batch(
        solver,
        start_positions=start_p,
        target_positions=target_p,
        start_orientations_wxyz=start_q,
        target_orientations_wxyz=target_q,
        seeds=torch.zeros((2, 3), device=device),
        waypoint_count=4,
        physics_ticks_per_action=7,
        orientation_mode="target",
        progress_mode="smoothstep",
    )
    assert result.joint_positions.shape == (7, 2, 3)
    assert result.joint_positions.device.type == "cuda"
    assert solver.inputs.target_positions.shape == (4, 2, 3)
    assert solver.inputs.target_orientations_wxyz.shape == (4, 2, 4)
    midpoint = quaternion_slerp_wxyz(
        start_q,
        target_q,
        torch.tensor([0.5], device=device),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(midpoint, dim=-1), torch.ones((1, 2), device=device)
    )


def test_linear_motion_applies_smoothstep_only_during_tick_resampling() -> None:
    device = _device()
    solver = _CartesianWaypointSolver(device)
    result = solve_linear_motion_batch(
        solver,
        start_positions=torch.zeros((1, 3), device=device),
        target_positions=torch.tensor([[1.0, 0.0, 0.0]], device=device),
        start_orientations_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        target_orientations_wxyz=None,
        seeds=torch.zeros((1, 1), device=device),
        waypoint_count=4,
        physics_ticks_per_action=4,
        orientation_mode="free",
        progress_mode="smoothstep",
    )

    expected_waypoints = torch.tensor([0.25, 0.5, 0.75, 1.0], device=device).reshape(
        4, 1, 1
    )
    expected_ticks = torch.tensor([0.15625, 0.5, 0.84375, 1.0], device=device).reshape(
        4, 1, 1
    )
    double_smoothed_ticks = expected_ticks.square() * (3.0 - 2.0 * expected_ticks)
    torch.testing.assert_close(
        solver.inputs.target_positions[..., :1], expected_waypoints
    )
    torch.testing.assert_close(result.joint_positions, expected_ticks)
    assert not torch.allclose(result.joint_positions, double_smoothed_ticks)


def test_multi_robot_ik_and_linear_action_terms_have_fixed_cuda_shapes() -> None:
    state_task = _task(num_envs=3)
    ids = torch.arange(3, device=_device())
    command = state_task.reset_command(ids)
    state = _state_from_reset(command)
    bindings = (
        KinematicsRobotBinding(
            label="left",
            action_slice=slice(0, 7),
            command_slice=slice(0, 3),
            tcp_index=0,
            solver=_IKSolver(_device(), 3),
        ),
        KinematicsRobotBinding(
            label="right",
            action_slice=slice(7, 14),
            command_slice=slice(3, 6),
            tcp_index=1,
            solver=_IKSolver(_device(), 3),
        ),
    )
    pose_spec = ActionSpec(
        ActionMode.EE_POSE_FULL,
        ("left", "right"),
        (3, 3),
        physics_ticks_per_action=2,
    )
    pose_term = IKRuntimeAction(spec=pose_spec, bindings=bindings, command_dim=6)
    assert pose_term.action_low == -float("inf")
    assert pose_term.action_high == float("inf")
    pose_action = torch.zeros((3, 14), device=_device())
    pose_action[:, 3] = 1.0
    pose_action[:, 10] = 1.0
    pose_result = pose_term.apply(pose_action, state)
    assert pose_result.joint_targets.shape == (2, 3, 6)
    assert pose_result.failure_mask.tolist() == [False, False, True]

    state.position_references[-1].add_(0.5)
    velocity_spec = replace(pose_spec, reference_velocity_limit=0.1)
    velocity_term = IKRuntimeAction(
        spec=velocity_spec,
        bindings=bindings,
        command_dim=6,
        physics_dt=0.1,
    )
    velocity_result = velocity_term.apply(pose_action, state, "velocity")
    assert isinstance(velocity_result.control, VelocityControlTrajectory)
    torch.testing.assert_close(
        velocity_result.control.velocities[0, :2],
        torch.full((2, 6), 0.1, device=_device()),
    )
    torch.testing.assert_close(
        velocity_result.control.velocities[1, :2],
        torch.zeros((2, 6), device=_device()),
    )
    torch.testing.assert_close(
        velocity_result.control.velocities[:, -1],
        torch.zeros((2, 6), device=_device()),
    )
    assert velocity_result.info["control_velocity_saturated"].tolist() == [
        True,
        True,
        False,
    ]

    linear_bindings = tuple(
        replace(binding, solver=_WaypointSolver(_device(), 3)) for binding in bindings
    )
    linear_spec = ActionSpec(
        ActionMode.EE_LINEAR_PATH_FULL,
        ("left", "right"),
        (3, 3),
        physics_ticks_per_action=5,
        waypoint_count=3,
        orientation_mode="target",
    )
    linear_term = LinearRuntimeAction(
        spec=linear_spec, bindings=linear_bindings, command_dim=6
    )
    assert linear_term.action_low == -float("inf")
    assert linear_term.action_high == float("inf")
    linear_action = pose_action.clone()
    linear_result = linear_term.apply(linear_action, state)
    assert linear_result.joint_targets.shape == (5, 3, 6)
    assert linear_result.failure_mask.device.type == "cuda"


class _PhysicsRuntime:
    kind = "physx_cuda"

    def __init__(self) -> None:
        self.steps = 0
        self.forwards = 0

    def step(self, *, render: bool) -> None:
        assert render is False
        self.steps += 1

    def forward(self) -> None:
        self.forwards += 1


class _Session:
    def __init__(self) -> None:
        self.physics_runtime = _PhysicsRuntime()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Views:
    def __init__(self, num_envs: int, command_dim: int) -> None:
        self.num_envs = num_envs
        self.device = _device()
        self.q = torch.zeros((num_envs, command_dim), device=self.device)
        self.qd = torch.zeros_like(self.q)
        self.target = torch.zeros_like(self.q)
        self.block_p = torch.zeros((num_envs, 3), device=self.device)
        self.block_p[:, 2] = -0.38
        self.block_q = torch.tensor(
            [[0.70714084, 0.0, 0.70707272, 0.0]] * num_envs,
            device=self.device,
        )
        self.block_v = torch.zeros((num_envs, 6), device=self.device)
        self.tcp_p = torch.zeros((num_envs, 2, 3), device=self.device)
        self.tcp_q = torch.zeros((num_envs, 2, 4), device=self.device)
        self.tcp_q[..., 0] = 1.0
        self.stop = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.reset_selectors: list[torch.Tensor] = []
        self.closed = False

    def write_reset(self, command) -> None:
        ids = command.env_ids
        self.reset_selectors.append(ids)
        self.q.index_copy_(0, ids, command.joint_positions)
        self.qd.index_copy_(0, ids, command.joint_velocities)
        self.target.index_copy_(0, ids, command.joint_targets)
        self.block_p.index_copy_(0, ids, command.block_position)
        self.block_q.index_copy_(0, ids, command.block_orientation_wxyz)
        self.block_v.index_copy_(0, ids, command.block_velocity)
        tcp = torch.stack(
            (
                command.block_position
                + torch.tensor([0.0, -0.03, 0.02], device=self.device),
                command.block_position
                + torch.tensor([0.0, 0.03, 0.02], device=self.device),
            ),
            dim=1,
        )
        self.tcp_p.index_copy_(0, ids, tcp)

    def write_joint_targets(self, targets: torch.Tensor) -> None:
        self.target.copy_(targets)

    def refresh(self, env_ids: torch.Tensor | None = None) -> TBlockState:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )

        def select(value: torch.Tensor) -> torch.Tensor:
            return value.index_select(0, ids).clone()

        return TBlockState(
            joint_positions=select(self.q),
            joint_velocities=select(self.qd),
            command_targets=select(self.target),
            tcp_positions_local=select(self.tcp_p),
            tcp_orientations_wxyz=select(self.tcp_q),
            block_position_local=select(self.block_p),
            block_orientation_wxyz=select(self.block_q),
            block_com_velocity=select(self.block_v),
            external_safety_stop=select(self.stop),
        )

    def close(self) -> None:
        self.closed = True


def _description_command(*, env_id: int, command_dim: int = 6) -> TBlockResetCommand:
    device = _device()
    joint_positions = torch.linspace(
        -0.12,
        0.12,
        command_dim,
        device=device,
        dtype=torch.float32,
    )[None, :]
    return TBlockResetCommand(
        env_ids=torch.tensor([env_id], device=device, dtype=torch.int64),
        joint_positions=joint_positions,
        joint_velocities=torch.zeros_like(joint_positions),
        joint_targets=joint_positions.clone(),
        block_position=torch.tensor(
            [[0.12, 0.01, -0.38]], device=device, dtype=torch.float32
        ),
        block_orientation_wxyz=torch.tensor(
            [[0.70714084, 0.0, 0.70707272, 0.0]],
            device=device,
            dtype=torch.float32,
        ),
        block_velocity=torch.zeros((1, 6), device=device, dtype=torch.float32),
        goal_position=torch.tensor(
            [[0.22, -0.02, -0.38]], device=device, dtype=torch.float32
        ),
        goal_yaw=torch.tensor([0.25], device=device, dtype=torch.float32),
        rng_key=torch.tensor([314159], device=device, dtype=torch.int64),
        rng_counter_after=torch.tensor([96], device=device, dtype=torch.int64),
    )


def _description_runtime(*, factory):
    num_envs = 3
    task = _task(num_envs=num_envs)
    views = _Views(num_envs, command_dim=6)
    session = _Session()
    controller = JointDeltaActionTerm(
        lower=torch.full((6,), -1.0, device=_device()),
        upper=torch.full((6,), 1.0, device=_device()),
        scale=torch.full((6,), 0.05, device=_device()),
        clip=1.0,
        num_envs=num_envs,
        target=views.target,
    )
    action = JointDeltaRuntimeAction(controller, physics_ticks_per_action=2)
    state_api = SimpleNamespace(
        num_envs=num_envs,
        device=_device(),
        poisoned=False,
    )
    runtime = KaleidoscopeRuntime(
        session=session,
        views=views,
        action_term=action,
        task=task,
        state_api=state_api,
        reset_description_factory=factory,
    )
    return runtime, task, views, session, action


def test_runtime_reset_from_description_uses_existing_reset_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    received: list[tuple[object, int]] = []
    description = object()
    command = _description_command(env_id=1)

    def factory(value: object, env_id: int) -> TBlockResetCommand:
        events.append("factory")
        received.append((value, env_id))
        return command

    runtime, task, views, session, action = _description_runtime(factory=factory)

    def wrap(target: object, name: str, event: str) -> None:
        original = getattr(target, name)

        def recorded(*args, **kwargs):
            events.append(event)
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, recorded)

    wrap(task, "stage_reset_description", "stage")
    wrap(runtime.control_mode, "write_reset", "control_write")
    wrap(action, "reset", "action_reset")
    wrap(session.physics_runtime, "forward", "forward")
    wrap(views, "refresh", "refresh")
    wrap(task, "initialize_after_reset", "initialize")

    observations, info = runtime.reset_from_description(description, 1)

    assert received == [(description, 1)]
    assert events == [
        "factory",
        "stage",
        "control_write",
        "action_reset",
        "forward",
        "refresh",
        "initialize",
    ]
    torch.testing.assert_close(views.q[1], command.joint_positions[0])
    torch.testing.assert_close(views.block_p[1], command.block_position[0])
    torch.testing.assert_close(task.buffers.goal_position[1], command.goal_position[0])
    torch.testing.assert_close(task.buffers.goal_yaw[1], command.goal_yaw[0])
    assert int(task.buffers.rng_key[1].item()) == 314159
    assert int(task.buffers.rng_counter[1].item()) == 96
    torch.testing.assert_close(
        observations,
        task.buffers.last_finite_observation.index_select(0, command.env_ids),
    )
    torch.testing.assert_close(info["env_ids"], command.env_ids)
    assert session.physics_runtime.steps == 0
    runtime.close()


def test_torch_env_exposes_single_description_reset() -> None:
    description = {"schema": "test-only-sentinel"}
    observations = object()
    info = {"env_ids": object()}
    calls: list[tuple[object, int]] = []

    def reset_from_description(value: object, env_id: int):
        calls.append((value, env_id))
        return observations, info

    runtime = SimpleNamespace(
        num_envs=4,
        device=_device(),
        action_dim=3,
        action_low=-1.0,
        action_high=1.0,
        observation_dim=7,
        viewport_enabled=False,
        render_every_n_steps=0,
        reset_from_description=reset_from_description,
    )
    env = TorchKaleidoscopeEnv(runtime)

    actual_observations, actual_info = env.reset_from_description(
        description,
        env_id=2,
    )

    assert calls == [(description, 2)]
    assert actual_observations is observations
    assert actual_info is info


def test_runtime_reset_description_conversion_failure_precedes_device_writes() -> None:
    description = object()
    calls: list[tuple[object, int]] = []

    def fail(value: object, env_id: int) -> TBlockResetCommand:
        calls.append((value, env_id))
        raise ValueError("description rejected")

    runtime, task, views, session, _action = _description_runtime(factory=fail)
    goal_before = task.buffers.goal_position.clone()

    with pytest.raises(ValueError, match="description rejected"):
        runtime.reset_from_description(description, 2)

    assert calls == [(description, 2)]
    assert views.reset_selectors == []
    assert session.physics_runtime.forwards == 0
    torch.testing.assert_close(task.buffers.goal_position, goal_before)
    assert runtime._failed is True
    with pytest.raises(RuntimeError, match="fail-stop"):
        runtime.reset()


def test_runtime_reset_description_failure_after_write_is_fail_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _description_command(env_id=2)
    runtime, _task_value, views, session, action = _description_runtime(
        factory=lambda _description, _env_id: command
    )

    def fail_action_reset(*_args, **_kwargs) -> None:
        raise RuntimeError("action reset failed")

    monkeypatch.setattr(action, "reset", fail_action_reset)

    with pytest.raises(RuntimeError, match="action reset failed"):
        runtime.reset_from_description(object(), 2)

    assert len(views.reset_selectors) == 1
    torch.testing.assert_close(views.q[2], command.joint_positions[0])
    assert session.physics_runtime.forwards == 0
    assert runtime._failed is True
    with pytest.raises(RuntimeError, match="fail-stop"):
        runtime.reset_from_description(object(), 2)


def test_runtime_owns_physx_cuda_flow_and_same_step_token() -> None:
    num_envs = 3
    task = _task(num_envs=num_envs)
    views = _Views(num_envs, command_dim=6)
    session = _Session()
    controller = JointDeltaActionTerm(
        lower=torch.full((6,), -1.0, device=_device()),
        upper=torch.full((6,), 1.0, device=_device()),
        scale=torch.full((6,), 0.05, device=_device()),
        clip=1.0,
        num_envs=num_envs,
        target=views.target,
    )
    action = JointDeltaRuntimeAction(controller, physics_ticks_per_action=2)

    def writer(target: torch.Tensor):
        return lambda ids, values: target.index_copy_(0, ids, values)

    bindings = {
        "robot.q": StateBinding(views.q, writer(views.q)),
        "robot.qd": StateBinding(views.qd, writer(views.qd)),
        "robot.target": StateBinding(views.target, writer(views.target)),
        "object.position": StateBinding(views.block_p, writer(views.block_p)),
        "object.orientation": StateBinding(views.block_q, writer(views.block_q)),
        "object.velocity": StateBinding(views.block_v, writer(views.block_v)),
        **{
            name: StateBinding(value, finite=value.is_floating_point())
            for name, value in task.state_fields().items()
        },
    }
    state_api = KaleidoscopeStateAPI(
        bindings,
        num_envs=num_envs,
        rng_fields=("rng.key", "rng.counter"),
    )
    runtime = KaleidoscopeRuntime(
        session=session,
        views=views,
        action_term=action,
        task=task,
        state_api=state_api,
    )
    observations, _info = runtime.reset()
    assert observations.shape == (num_envs, task.observation_dim)
    runtime.reset()
    assert len(views.reset_selectors) == 2
    assert views.reset_selectors[0] is runtime._all_env_ids
    assert views.reset_selectors[1] is runtime._all_env_ids
    result = runtime.step(torch.zeros((num_envs, 6), device=_device()))
    assert result.rewards.device.type == "cuda"
    assert session.physics_runtime.steps == 2

    task.buffers.needs_reset[0] = True
    steps_before_rejected_step = session.physics_runtime.steps
    with pytest.raises(RuntimeError, match="done environments must be reset"):
        runtime.step(torch.zeros((num_envs, 6), device=_device()))
    assert session.physics_runtime.steps == steps_before_rejected_step
    runtime.reset_idx(torch.tensor([0], device=_device()))

    token = runtime.issue_same_step_token()
    with pytest.raises(RuntimeError, match="has not been stepped"):
        runtime.complete_same_step_reset(token)
    forged = SameStepToken(token.generation)
    with pytest.raises(RuntimeError, match="forged"):
        runtime.tokenized_step(forged, torch.zeros((num_envs, 6), device=_device()))
    runtime.tokenized_step(token, torch.zeros((num_envs, 6), device=_device()))
    steps_after_token = session.physics_runtime.steps
    state_before_complete = {
        name: value.clone() for name, value in state_api.get_state().items()
    }
    with pytest.raises(RuntimeError, match="already been stepped"):
        runtime.tokenized_step(token, torch.zeros((num_envs, 6), device=_device()))
    assert session.physics_runtime.steps == steps_after_token
    done_mask = runtime.complete_same_step_reset(token)
    assert done_mask.shape == (num_envs,)
    assert done_mask.dtype == torch.bool
    assert not torch.any(done_mask)
    for name, previous in state_before_complete.items():
        torch.testing.assert_close(state_api.get_state(fields=(name,))[name], previous)
    with pytest.raises(RuntimeError, match="completed"):
        runtime.complete_same_step_reset(token)
    next_token = runtime.issue_same_step_token()
    assert next_token.generation == token.generation + 1
    runtime.tokenized_step(next_token, torch.zeros((num_envs, 6), device=_device()))
    runtime.complete_same_step_reset(next_token)

    runtime.clone_state(
        torch.tensor([0], device=_device()), torch.tensor([1], device=_device())
    )
    runtime.step(torch.zeros((num_envs, 6), device=_device()))
    torch.testing.assert_close(controller.target[1], controller.target[0])
    runtime.close()
    assert views.closed and session.closed


class _CollisionQueryAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.fail_query = False

    def query(self, **kwargs: object) -> ArticulatedCollisionQueryResult:
        self.calls.append(kwargs)
        positions = kwargs["joint_positions"]
        env_ids = kwargs["env_ids"]
        assert isinstance(positions, torch.Tensor)
        assert isinstance(env_ids, torch.Tensor)
        if self.fail_query:
            raise RuntimeError("injected collision query failure")
        return ArticulatedCollisionQueryResult(
            collision_mask=torch.zeros(
                (env_ids.numel(), positions.shape[1]),
                device=positions.device,
                dtype=torch.bool,
            ),
            first_hit_sample=torch.full(
                (env_ids.numel(),),
                -1,
                device=positions.device,
                dtype=torch.int64,
            ),
            first_hit_category=torch.full(
                (env_ids.numel(),),
                CollisionHitCategory.NONE,
                device=positions.device,
                dtype=torch.int64,
            ),
        )

    def close(self) -> None:
        self.closed = True


def _collision_query_runtime(
    *,
    backend_kind: str = "physx_cuda",
    adapter: _CollisionQueryAdapter | None = None,
    query_specs: object | None = None,
) -> KaleidoscopeRuntime:
    runtime = object.__new__(KaleidoscopeRuntime)
    runtime.session = SimpleNamespace(
        physics_runtime=SimpleNamespace(kind=backend_kind)
    )
    runtime.collision_query_resources = None
    if adapter is not None:
        runtime.collision_query_resources = SimpleNamespace(
            query_adapter=adapter,
            query_specs=(MappingProxyType({}) if query_specs is None else query_specs),
        )
    runtime.state_api = SimpleNamespace(poisoned=False)
    runtime._owner_thread_id = get_ident()
    runtime._closed = False
    runtime._closing_started = False
    runtime._failed = False
    runtime._fatal_error = None
    runtime._phase = "idle"
    runtime._outstanding_token = None
    return runtime


def test_runtime_collision_query_specs_are_cold_backend_metadata() -> None:
    adapter = _CollisionQueryAdapter()
    spec = ArticulatedCollisionQuerySpec(
        robot_label="left",
        command_joint_names=("joint",),
        joint_lower=(-1.0,),
        joint_upper=(1.0,),
    )
    specs = MappingProxyType({"left": spec})
    runtime = _collision_query_runtime(adapter=adapter, query_specs=specs)

    assert runtime.articulated_collision_query_specs is specs
    assert adapter.calls == []
    runtime._phase = "step"
    assert runtime.articulated_collision_query_specs is specs
    runtime._closed = True
    assert runtime.articulated_collision_query_specs is specs

    newton = _collision_query_runtime(
        backend_kind="newton_cuda",
        adapter=adapter,
        query_specs=specs,
    )
    assert dict(newton.articulated_collision_query_specs) == {}
    with pytest.raises(TypeError):
        newton.articulated_collision_query_specs["left"] = spec  # type: ignore[index]


def test_runtime_collision_query_dispatch_and_lifecycle_guards() -> None:
    adapter = _CollisionQueryAdapter()
    runtime = _collision_query_runtime(adapter=adapter)
    positions = torch.zeros((1, 3, 2), device=_device())
    env_ids = torch.tensor((1,), device=_device(), dtype=torch.int64)
    targets = (
        CollisionQueryTarget.TABLE,
        CollisionQueryTarget.DYNAMIC_OBJECT,
    )

    assert runtime.supports_articulated_collision_queries is True

    result = runtime.query_articulated_collisions(
        robot_label="left",
        joint_positions=positions,
        env_ids=env_ids,
        targets=targets,
    )
    assert result.collision_mask.shape == (1, 3)
    assert adapter.calls == [
        {
            "robot_label": "left",
            "joint_positions": positions,
            "env_ids": env_ids,
            "targets": targets,
        }
    ]
    assert runtime._phase == "idle"

    adapter.fail_query = True
    with pytest.raises(RuntimeError, match="injected collision query failure"):
        runtime.query_articulated_collisions(
            robot_label="left",
            joint_positions=positions,
            env_ids=env_ids,
            targets=targets,
        )
    assert runtime._phase == "idle"
    assert runtime._failed is False
    adapter.fail_query = False

    runtime._outstanding_token = object()
    with pytest.raises(RuntimeError, match="active SAME_STEP transaction"):
        runtime.query_articulated_collisions(
            robot_label="left",
            joint_positions=positions,
            env_ids=env_ids,
            targets=targets,
        )
    runtime._outstanding_token = None

    runtime._phase = "step"
    with pytest.raises(ControlModeLockedError, match="collision_query"):
        runtime.query_articulated_collisions(
            robot_label="left",
            joint_positions=positions,
            env_ids=env_ids,
            targets=targets,
        )
    runtime._phase = "idle"

    errors: list[BaseException] = []

    def query_from_worker() -> None:
        try:
            runtime.query_articulated_collisions(
                robot_label="left",
                joint_positions=positions,
                env_ids=env_ids,
                targets=targets,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=query_from_worker)
    worker.start()
    worker.join()
    assert len(errors) == 1
    assert "owner thread" in str(errors[0])

    runtime._failed = True
    with pytest.raises(RuntimeError, match="fail-stop"):
        runtime.query_articulated_collisions(
            robot_label="left",
            joint_positions=positions,
            env_ids=env_ids,
            targets=targets,
        )
    runtime._failed = False
    runtime._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        runtime.query_articulated_collisions(
            robot_label="left",
            joint_positions=positions,
            env_ids=env_ids,
            targets=targets,
        )


def test_description_reset_then_collision_query_is_read_only_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter = _CollisionQueryAdapter()
    joint_names = tuple(f"joint_{index}" for index in range(6))
    spec = ArticulatedCollisionQuerySpec(
        robot_label="left",
        command_joint_names=joint_names,
        joint_lower=(-1.0,) * 6,
        joint_upper=(1.0,) * 6,
    )

    def close_query_resources() -> None:
        events.append("collision_query_resources")
        adapter.close()

    runtime, task, views, session, action = _description_runtime(
        factory=lambda _description, _env_id: _description_command(env_id=1)
    )
    runtime.collision_query_resources = SimpleNamespace(
        query_adapter=adapter,
        query_specs=MappingProxyType({"left": spec}),
        close=close_query_resources,
    )
    env = TorchKaleidoscopeEnv(runtime)

    def record_close(target: object, name: str, event: str) -> None:
        close = getattr(target, name)

        def recorded(*args, **kwargs):
            events.append(event)
            return close(*args, **kwargs)

        monkeypatch.setattr(target, name, recorded)

    record_close(action, "close", "action_term")
    record_close(views, "close", "views")
    record_close(task, "close", "task")
    record_close(session, "close", "session")

    try:
        env.reset_from_description(object(), env_id=1)
        assert env.supports_articulated_collision_queries is True
        assert env.articulated_collision_query_specs["left"] is spec

        views_before = {
            name: getattr(views, name).clone()
            for name in ("q", "qd", "target", "block_p", "block_q", "block_v")
        }
        task_before = {
            name: value.clone() for name, value in task.state_fields().items()
        }
        task_counters_before = {
            name: getattr(task.buffers, name).clone()
            for name in (
                "rng_key",
                "rng_counter",
                "episode_length",
                "episode_physics_steps",
                "reward",
                "goal_position",
                "goal_yaw",
            )
        }
        physics_clock_before = (
            session.physics_runtime.steps,
            session.physics_runtime.forwards,
        )

        result = env.query_articulated_collisions(
            robot_label="left",
            joint_positions=torch.zeros((1, 2, 6), device=_device()),
            env_ids=torch.tensor([1], device=_device(), dtype=torch.int64),
            targets=(CollisionQueryTarget.TABLE,),
        )

        assert result.collision_mask.shape == (1, 2)
        assert runtime._phase == "idle"
        assert runtime._failed is False
        assert (session.physics_runtime.steps, session.physics_runtime.forwards) == (
            physics_clock_before
        )
        for name, before in views_before.items():
            torch.testing.assert_close(getattr(views, name), before)
        for name, before in task_before.items():
            torch.testing.assert_close(task.state_fields()[name], before)
        for name, before in task_counters_before.items():
            torch.testing.assert_close(getattr(task.buffers, name), before)
        assert len(adapter.calls) == 1
    finally:
        env.close()

    env.close()
    assert events == [
        "action_term",
        "collision_query_resources",
        "views",
        "task",
        "session",
    ]
    assert adapter.closed is True


@pytest.mark.parametrize("backend_kind", ("physx_cuda", "newton_cuda"))
def test_runtime_collision_query_reports_backend_unavailable(
    backend_kind: str,
) -> None:
    runtime = _collision_query_runtime(backend_kind=backend_kind)

    assert runtime.supports_articulated_collision_queries is False

    with pytest.raises(CollisionQueryUnavailableError) as captured:
        runtime.query_articulated_collisions(
            robot_label="left",
            joint_positions=torch.zeros((1, 1, 2), device=_device()),
            env_ids=torch.zeros(1, device=_device(), dtype=torch.int64),
            targets=(CollisionQueryTarget.TABLE,),
        )

    assert captured.value.backend_kind == backend_kind


@pytest.mark.parametrize(
    ("action_low", "action_high"),
    [(float("nan"), 1.0), (-1.0, float("nan"))],
)
def test_runtime_rejects_nan_action_bounds(
    action_low: float, action_high: float
) -> None:
    device = _device()
    with pytest.raises(ValueError, match="bounds must be ordered"):
        KaleidoscopeRuntime(
            session=SimpleNamespace(physics_runtime=SimpleNamespace(kind="physx_cuda")),
            views=SimpleNamespace(num_envs=2, device=device),
            action_term=SimpleNamespace(
                action_dim=1,
                action_low=action_low,
                action_high=action_high,
                physics_ticks_per_action=1,
            ),
            task=SimpleNamespace(
                num_envs=2,
                device=device,
                action_dim=1,
                observation_dim=1,
                settings=SimpleNamespace(physics_ticks_per_action=1),
            ),
            state_api=SimpleNamespace(num_envs=2, device=device),
        )


def test_runtime_allows_unbounded_ee_action_metadata() -> None:
    device = _device()
    runtime = KaleidoscopeRuntime(
        session=SimpleNamespace(physics_runtime=SimpleNamespace(kind="physx_cuda")),
        views=SimpleNamespace(num_envs=2, device=device),
        action_term=SimpleNamespace(
            action_dim=7,
            action_low=-float("inf"),
            action_high=float("inf"),
            physics_ticks_per_action=1,
        ),
        task=SimpleNamespace(
            num_envs=2,
            device=device,
            action_dim=7,
            observation_dim=1,
            settings=SimpleNamespace(physics_ticks_per_action=1),
        ),
        state_api=SimpleNamespace(num_envs=2, device=device),
    )
    assert runtime.action_low == -float("inf")
    assert runtime.action_high == float("inf")
