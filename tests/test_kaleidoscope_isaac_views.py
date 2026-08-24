from __future__ import annotations

import pytest
import torch

from linkerbot_sim.kaleidoscope.isaac_views import (
    KaleidoscopeTensorViews,
    _owned_rows,
)
from linkerbot_sim.kaleidoscope.resets import TBlockResetCommand
from linkerbot_sim.kaleidoscope.state_api import KaleidoscopeStateAPI, StateBinding


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for PhysX tensor views"
)


class _RobotPort:
    def __init__(self, label: str, origins: torch.Tensor, local_x: float) -> None:
        self.label = label
        self.command_dim = 2
        self.state_dim = 3
        self.device = origins.device
        self.command_state_indices = torch.tensor(
            [0, 2], device=self.device, dtype=torch.int64
        )
        self.q = torch.zeros((3, 3), device=self.device)
        self.qd = torch.zeros_like(self.q)
        self.target = torch.zeros((3, 2), device=self.device)
        self.tcp_world = origins + torch.tensor(
            [local_x, 0.0, -0.3], device=self.device
        )
        self.tcp_q = torch.zeros((3, 4), device=self.device)
        self.tcp_q[:, 0] = 1.0
        self.closed = False
        self.close_calls = 0
        self.fail_close_once = False

    def read_joint_positions(self, ids):
        return self.q.index_select(0, ids).index_select(1, self.command_state_indices)

    def read_joint_velocities(self, ids):
        return self.qd.index_select(0, ids).index_select(1, self.command_state_indices)

    def read_all_joint_positions(self, ids):
        return self.q.index_select(0, ids)

    def read_all_joint_velocities(self, ids):
        return self.qd.index_select(0, ids)

    def read_tcp_pose_wxyz(self, ids):
        return self.tcp_world.index_select(0, ids), self.tcp_q.index_select(0, ids)

    def write_joint_positions(self, ids, values):
        selected = self.q.index_select(0, ids)
        selected.index_copy_(1, self.command_state_indices, values)
        self.q.index_copy_(0, ids, selected)

    def write_joint_velocities(self, ids, values):
        selected = self.qd.index_select(0, ids)
        selected.index_copy_(1, self.command_state_indices, values)
        self.qd.index_copy_(0, ids, selected)

    def write_joint_targets(self, ids, values):
        self.target.index_copy_(0, ids, values)

    def write_all_joint_positions(self, ids, values):
        self.q.index_copy_(0, ids, values)

    def write_all_joint_velocities(self, ids, values):
        self.qd.index_copy_(0, ids, values)

    def close(self):
        self.close_calls += 1
        if self.fail_close_once and self.close_calls == 1:
            raise RuntimeError("retry robot close")
        self.closed = True


class _ObjectPort:
    label = "tblock"

    def __init__(self, origins: torch.Tensor) -> None:
        self.device = origins.device
        self.position_world = origins + torch.tensor(
            [0.1, 0.2, -0.38], device=self.device
        )
        self.orientation = torch.zeros((3, 4), device=self.device)
        self.orientation[:, 0] = 1.0
        self.velocity = torch.zeros((3, 6), device=self.device)
        self.closed = False

    def read_pose_wxyz(self, ids):
        return self.position_world.index_select(0, ids), self.orientation.index_select(
            0, ids
        )

    def read_com_velocity(self, ids):
        return self.velocity.index_select(0, ids)

    def write_pose_wxyz(self, ids, positions_world, orientations_wxyz):
        self.position_world.index_copy_(0, ids, positions_world)
        self.orientation.index_copy_(0, ids, orientations_wxyz)

    def write_velocity(self, ids, values):
        self.velocity.index_copy_(0, ids, values)

    def close(self):
        self.closed = True


class _PhysicsStatePort:
    field_name = "solver.persistent"

    def __init__(self, origins: torch.Tensor) -> None:
        self.device = origins.device
        self.num_envs = origins.shape[0]
        self.tensor = torch.arange(
            self.num_envs * 5,
            device=self.device,
            dtype=torch.float32,
        ).reshape(self.num_envs, 5)
        self.reset_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        self.closed = False

    def write(self, ids, values) -> None:
        self.tensor.index_copy_(0, ids, values)

    def reset(self, ids, *, reset_mask=None) -> None:
        self.reset_calls.append(
            (ids.clone(), None if reset_mask is None else reset_mask.clone())
        )

    def close(self) -> None:
        self.closed = True


class _SelectorRobotPort(_RobotPort):
    def __init__(self, label: str, origins: torch.Tensor, local_x: float) -> None:
        super().__init__(label, origins, local_x)
        self.selectors: list[torch.Tensor] = []

    def read_all_joint_positions(self, ids):
        self.selectors.append(ids)
        return super().read_all_joint_positions(ids)

    def write_joint_targets(self, ids, values):
        self.selectors.append(ids)
        super().write_joint_targets(ids, values)


class _ControlRobotPort(_RobotPort):
    def __init__(self, label: str, origins: torch.Tensor, local_x: float) -> None:
        super().__init__(label, origins, local_x)
        self.control_targets = {
            mode: torch.zeros((3, self.command_dim), device=self.device)
            for mode in ("position", "velocity", "effort")
        }
        self.control_events: list[str] = []

    def write_joint_position_targets(self, ids, values):
        self.control_events.append("position")
        self.control_targets["position"].index_copy_(0, ids, values)

    def write_joint_velocity_targets(self, ids, values):
        self.control_events.append("velocity")
        self.control_targets["velocity"].index_copy_(0, ids, values)

    def write_joint_effort_targets(self, ids, values):
        self.control_events.append("effort")
        self.control_targets["effort"].index_copy_(0, ids, values)


def test_physx_views_keep_canonical_pose_local_and_clone_adds_target_origin() -> None:
    origins = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        device="cuda",
    )
    left = _RobotPort("left", origins, -0.1)
    right = _RobotPort("right", origins, 0.1)
    block = _ObjectPort(origins)
    views = KaleidoscopeTensorViews(
        robot_ports=(left, right), object_port=block, env_origins=origins
    )
    state = views.refresh()
    torch.testing.assert_close(
        state.block_position_local,
        torch.tensor([[0.1, 0.2, -0.38]] * 3, device="cuda"),
    )
    torch.testing.assert_close(
        state.tcp_positions_local[:, 0],
        torch.tensor([[-0.1, 0.0, -0.3]] * 3, device="cuda"),
    )

    task_counter = torch.arange(3, device="cuda", dtype=torch.int64)
    api = KaleidoscopeStateAPI(
        views.state_bindings({"task.counter": task_counter}), num_envs=3
    )
    api.clone_state(torch.tensor([0], device="cuda"), torch.tensor([2], device="cuda"))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        block.position_world[2], torch.tensor([20.1, 0.2, -0.38], device="cuda")
    )
    torch.testing.assert_close(
        views.block_position_local[2], views.block_position_local[0]
    )
    assert task_counter[2] == task_counter[0]


def test_state_api_snapshots_and_clones_full_robot_dofs() -> None:
    origins = torch.zeros((3, 3), device="cuda")
    robot = _RobotPort("left", origins, 0.0)
    robot.q.copy_(torch.arange(9, device="cuda").reshape(3, 3))
    block = _ObjectPort(origins)
    views = KaleidoscopeTensorViews(
        robot_ports=(robot,), object_port=block, env_origins=origins
    )
    views.refresh()
    api = KaleidoscopeStateAPI(
        views.state_bindings(
            {"task.counter": torch.zeros(3, device="cuda", dtype=torch.int64)}
        ),
        num_envs=3,
    )

    snapshot = api.snapshot(torch.tensor([0], device="cuda"))
    assert snapshot.fields["robot.q"].shape == (1, 3)
    api.clone_state(torch.tensor([0], device="cuda"), torch.tensor([2], device="cuda"))
    torch.testing.assert_close(robot.q[2], robot.q[0])
    torch.testing.assert_close(views.joint_positions[2], robot.q[0, [0, 2]])


def test_physx_views_copy_borrowed_rows_and_close_all_ports() -> None:
    origins = torch.zeros((3, 3), device="cuda")
    left = _RobotPort("left", origins, -0.1)
    right = _RobotPort("right", origins, 0.1)
    block = _ObjectPort(origins)
    views = KaleidoscopeTensorViews(
        robot_ports=(left, right), object_port=block, env_origins=origins
    )
    state = views.refresh()
    left.q.add_(10.0)
    torch.testing.assert_close(
        state.joint_positions, torch.zeros_like(state.joint_positions)
    )
    views.close()
    assert left.closed and right.closed and block.closed


def test_physx_views_close_keeps_retry_progress_after_partial_failure() -> None:
    origins = torch.zeros((3, 3), device="cuda")
    left = _RobotPort("left", origins, -0.1)
    right = _RobotPort("right", origins, 0.1)
    right.fail_close_once = True
    block = _ObjectPort(origins)
    views = KaleidoscopeTensorViews(
        robot_ports=(left, right), object_port=block, env_origins=origins
    )

    with pytest.raises(RuntimeError, match="retry robot close"):
        views.close()
    assert views._closed is False
    assert left.closed and block.closed and not right.closed
    with pytest.raises(RuntimeError, match="teardown has started"):
        views.refresh()

    views.close()
    assert views._closed is True
    assert right.close_calls == 2
    assert left.close_calls == 1


def test_physics_state_binding_participates_in_clone_and_masked_reset() -> None:
    origins = torch.zeros((3, 3), device="cuda")
    robot = _RobotPort("left", origins, 0.0)
    block = _ObjectPort(origins)
    physics = _PhysicsStatePort(origins)
    views = KaleidoscopeTensorViews(
        robot_ports=(robot,),
        object_port=block,
        env_origins=origins,
        physics_state_port=physics,
    )
    views.refresh()
    api = KaleidoscopeStateAPI(views.state_bindings({}), num_envs=3)
    source_before = physics.tensor[0].clone()

    api.clone_state(
        torch.tensor([0], device="cuda"),
        torch.tensor([2], device="cuda"),
    )
    torch.testing.assert_close(physics.tensor[2], source_before)
    assert "solver.persistent" in api.get_state()

    ids = torch.arange(3, device="cuda", dtype=torch.int64)
    mask = torch.tensor([False, True, False], device="cuda")
    views.write_reset(
        TBlockResetCommand(
            env_ids=ids,
            joint_positions=views.joint_positions.clone(),
            joint_velocities=views.joint_velocities.clone(),
            joint_targets=views.command_targets.clone(),
            block_position=views.block_position_local.clone(),
            block_orientation_wxyz=views.block_orientation_wxyz.clone(),
            block_velocity=views.block_com_velocity.clone(),
            goal_position=torch.zeros((3, 3), device="cuda"),
            goal_yaw=torch.zeros(3, device="cuda"),
            device_reset_mask=mask,
        )
    )
    reset_ids, reset_mask = physics.reset_calls[-1]
    assert torch.equal(reset_ids, ids)
    assert reset_mask is not None and torch.equal(reset_mask, mask)
    views.close()
    assert physics.closed


def test_views_reuse_one_all_environment_selector_on_hot_paths() -> None:
    origins = torch.zeros((3, 3), device="cuda")
    robot = _SelectorRobotPort("left", origins, 0.0)
    views = KaleidoscopeTensorViews(
        robot_ports=(robot,),
        object_port=_ObjectPort(origins),
        env_origins=origins,
    )

    views.refresh()
    views.refresh()
    views.write_joint_targets(torch.zeros((3, 2), device="cuda"))

    assert robot.selectors
    assert all(selector is views._all_env_ids for selector in robot.selectors)
    assert views._all_env_ids.dtype == torch.int64
    assert views._all_env_ids.device == views.device


def test_views_split_three_control_channels_and_write_position_feedforward_first() -> (
    None
):
    origins = torch.zeros((3, 3), device="cuda")
    left = _ControlRobotPort("left", origins, -0.1)
    right = _ControlRobotPort("right", origins, 0.1)
    views = KaleidoscopeTensorViews(
        robot_ports=(left, right),
        object_port=_ObjectPort(origins),
        env_origins=origins,
    )
    positions = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    velocities = positions + 20.0
    efforts = positions - 10.0

    views.write_position_targets(positions, velocities)
    assert left.control_events == ["velocity", "position"]
    assert right.control_events == ["velocity", "position"]
    torch.testing.assert_close(left.control_targets["position"], positions[:, :2])
    torch.testing.assert_close(left.control_targets["velocity"], velocities[:, :2])
    torch.testing.assert_close(right.control_targets["position"], positions[:, 2:])
    torch.testing.assert_close(right.control_targets["velocity"], velocities[:, 2:])

    views.write_effort_targets(efforts)
    torch.testing.assert_close(left.control_targets["effort"], efforts[:, :2])
    torch.testing.assert_close(right.control_targets["effort"], efforts[:, 2:])
    torch.testing.assert_close(views.control_targets, efforts)


def test_view_selection_and_borrowed_rows_remain_owned_with_one_copy() -> None:
    base = torch.arange(24, device="cuda", dtype=torch.float32).reshape(3, 8)
    borrowed = base[:, ::2]
    expected = borrowed.clone()

    owned = _owned_rows(
        borrowed,
        name="non-contiguous borrowed rows",
        rows=3,
        width=4,
        device=base.device,
    )
    assert owned.is_contiguous()
    assert owned.data_ptr() != borrowed.data_ptr()
    base.add_(100.0)
    torch.testing.assert_close(owned, expected)

    origins = torch.zeros((3, 3), device="cuda")
    views = KaleidoscopeTensorViews(
        robot_ports=(_RobotPort("left", origins, 0.0),),
        object_port=_ObjectPort(origins),
        env_origins=origins,
    )
    selected = views.refresh(torch.tensor([0, 2], device="cuda"))
    selected_before = selected.joint_positions.clone()
    views.joint_positions.add_(7.0)
    torch.testing.assert_close(selected.joint_positions, selected_before)


def test_view_port_device_validation_does_not_require_device_tensor() -> None:
    origins = torch.zeros((3, 3), device="cuda")
    robot = _RobotPort("left", origins, 0.0)
    robot.device = torch.device("cpu")

    with pytest.raises(ValueError, match="must share environment origins device"):
        KaleidoscopeTensorViews(
            robot_ports=(robot,),
            object_port=_ObjectPort(origins),
            env_origins=origins,
        )


def test_state_api_makes_one_contiguous_owned_copy_of_noncontiguous_payload() -> None:
    canonical = torch.zeros((3, 2), device="cuda")
    writer_values: list[torch.Tensor] = []

    def writer(ids: torch.Tensor, values: torch.Tensor) -> None:
        del ids
        writer_values.append(values)

    api = KaleidoscopeStateAPI(
        {"value": StateBinding(canonical, writer)},
        num_envs=3,
    )
    base = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    payload = base[:, ::2]
    expected = payload.clone()

    api.set_state({"value": payload})
    assert len(writer_values) == 1
    assert writer_values[0].is_contiguous()
    assert writer_values[0].data_ptr() != payload.data_ptr()
    base.add_(100.0)
    torch.testing.assert_close(canonical, expected)

    selected = api.get_state(torch.tensor([0, 2], device="cuda"))["value"]
    selected_before = selected.clone()
    canonical.add_(50.0)
    torch.testing.assert_close(selected, selected_before)


def test_nonfinite_tcp_pose_holds_last_finite_value() -> None:
    # Gitea #67: at scale PhysX/Fabric intermittently returns a non-finite (sticky) TCP link
    # transform for an env; the readback must hold that env's last-finite pose so no consumer
    # (observation / cuRobo IK target) ever sees NaN.
    origins = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        device="cuda",
    )
    robot = _RobotPort("arm", origins, 0.1)
    block = _ObjectPort(origins)
    views = KaleidoscopeTensorViews(
        robot_ports=(robot,), object_port=block, env_origins=origins
    )

    # First refresh records a finite last-known TCP for every env.
    first = views.refresh()
    torch.testing.assert_close(
        first.tcp_positions_local[:, 0],
        torch.tensor([[0.1, 0.0, -0.3]] * 3, device="cuda"),
    )
    assert int(views._nonfinite_tcp_holds.item()) == 0

    # PhysX/Fabric returns a non-finite link transform for env 1 only; env 0 moves.
    robot.tcp_world[1] = float("nan")
    robot.tcp_q[1] = float("nan")
    robot.tcp_world[0] = origins[0] + torch.tensor([0.5, 0.0, -0.3], device="cuda")

    second = views.refresh()
    # env 1 holds its last-finite pose; env 0 updates; nothing is NaN.
    assert torch.isfinite(second.tcp_positions_local).all()
    assert torch.isfinite(second.tcp_orientations_wxyz).all()
    torch.testing.assert_close(
        second.tcp_positions_local[1, 0],
        torch.tensor([0.1, 0.0, -0.3], device="cuda"),
    )
    torch.testing.assert_close(
        second.tcp_orientations_wxyz[1, 0],
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"),
    )
    torch.testing.assert_close(
        second.tcp_positions_local[0, 0],
        torch.tensor([0.5, 0.0, -0.3], device="cuda"),
    )
    assert int(views._nonfinite_tcp_holds.item()) == 1
