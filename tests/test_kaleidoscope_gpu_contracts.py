from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Kaleidoscope contracts"
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _task(
    *,
    num_envs: int = 4,
    settings: TBlockPushV1Settings | None = None,
) -> TBlockPushV1:
    device = _device()
    return TBlockPushV1(
        num_envs=num_envs,
        command_dim=6,
        action_dim=6,
        robot_count=2,
        device=device,
        dtype=torch.float32,
        nominal_joint_positions=torch.zeros(6, device=device),
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
