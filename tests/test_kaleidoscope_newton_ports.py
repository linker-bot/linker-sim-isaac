from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")

from linkerbot_sim.isaac.physics.newton.views import (  # noqa: E402
    NewtonArticulationView,
    NewtonRigidBodyView,
)
from linkerbot_sim.controllers.runtime_projection import (  # noqa: E402
    CommandRuntimeProjection,
)
from linkerbot_sim.kaleidoscope.isaac_views import (  # noqa: E402
    RigidObjectTensorPort,
    RobotTensorPort,
)
from linkerbot_sim.kaleidoscope.newton_ports import (  # noqa: E402
    NewtonArticulationTensorPort,
    NewtonRigidObjectTensorPort,
    NewtonSolverIntegrationTensorPort,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Newton tensor ports"
)


class _Manager:
    def __init__(self) -> None:
        self.closed = False
        self.stream = wp.Stream(wp.get_device("cuda:0"))
        self.events: list[tuple[str, str, tuple[int, ...]]] = []
        self.device_mask_events: list[tuple[str, str, torch.Tensor]] = []
        self._views: set[object] = set()

    def register_newton_view(self, view: object) -> None:
        self._views.add(view)

    def release_newton_view(self, view: object) -> None:
        wp.synchronize_stream(self.stream)
        self._views.discard(view)
        view._release_from_manager()

    def on_newton_view_write(
        self,
        *,
        view: object,
        category: str,
        field: str,
        world_indices: tuple[int, ...],
        device_row_mask: torch.Tensor | None = None,
    ) -> None:
        del view
        self.events.append((category, field, world_indices))
        if device_row_mask is not None:
            self.device_mask_events.append((category, field, device_row_mask))


def _warp_array(values: object, dtype: object, manager: _Manager) -> object:
    with wp.ScopedStream(manager.stream, sync_enter=False, sync_exit=False):
        return wp.array(values, dtype=dtype, device="cuda:0")


def _runtime() -> tuple[_Manager, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    manager = _Manager()
    robot_paths = tuple(f"/World/envs/env_{world}/robot" for world in range(3))
    tcp_paths = tuple(f"{path}/tcp" for path in robot_paths)
    object_paths = tuple(f"/World/envs/env_{world}/Tblock" for world in range(3))
    joint_labels = [
        name
        for root in robot_paths
        for name in (f"{root}/joint_a", f"{root}/joint_follower")
    ] + [f"{path}/free_joint" for path in object_paths]
    joint_q_start = (0, 1, 2, 3, 4, 5, 6, 13, 20, 27)
    joint_qd_start = (0, 1, 2, 3, 4, 5, 6, 12, 18, 24)
    joint_world = (0, 0, 1, 1, 2, 2, 0, 1, 2)
    body_labels = tuple(
        name for world in range(3) for name in (tcp_paths[world], object_paths[world])
    )
    joint_child = (0, 0, 2, 2, 4, 4, 1, 3, 5)
    manager.model = SimpleNamespace(
        device="cuda:0",
        world_count=3,
        articulation_label=list(robot_paths),
        articulation_world=_warp_array((0, 1, 2), wp.int32, manager),
        articulation_start=_warp_array((0, 2, 4, 6), wp.int32, manager),
        joint_label=joint_labels,
        joint_world=_warp_array(joint_world, wp.int32, manager),
        joint_type=_warp_array((1, 1, 1, 1, 1, 1, 4, 4, 4), wp.int32, manager),
        joint_q_start=_warp_array(joint_q_start, wp.int32, manager),
        joint_qd_start=_warp_array(joint_qd_start, wp.int32, manager),
        joint_child=_warp_array(joint_child, wp.int32, manager),
        joint_parent=_warp_array((-1,) * 9, wp.int32, manager),
        body_label=list(body_labels),
        body_world=_warp_array((0, 0, 1, 1, 2, 2), wp.int32, manager),
        joint_target_ke=_warp_array((100.0,) * 24, wp.float32, manager),
        joint_target_kd=_warp_array((10.0,) * 24, wp.float32, manager),
        joint_effort_limit=_warp_array((50.0,) * 24, wp.float32, manager),
    )
    joint_q = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
    joint_q.extend([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    joint_q.extend([11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    joint_q.extend([12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    body_q = []
    body_qd = []
    half = 2.0**-0.5
    for world in range(3):
        body_q.extend(
            (
                wp.transform((float(world), 0.0, 0.0), (0.0, 0.0, half, half)),
                wp.transform((10.0 + world, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            )
        )
        body_qd.extend(
            (
                wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                wp.spatial_vector(
                    1.0 + world,
                    2.0 + world,
                    3.0 + world,
                    4.0 + world,
                    5.0 + world,
                    6.0 + world,
                ),
            )
        )
    manager.state = SimpleNamespace(
        joint_q=_warp_array(joint_q, wp.float32, manager),
        joint_qd=_warp_array(
            [0.1, 1.0, 0.2, 2.0, 0.3, 3.0] + [0.0] * 18,
            wp.float32,
            manager,
        ),
        body_q=_warp_array(body_q, wp.transform, manager),
        body_qd=_warp_array(body_qd, wp.spatial_vector, manager),
    )
    manager.control = SimpleNamespace(
        joint_target_pos=_warp_array(
            [4.0, 40.0, 5.0, 50.0, 6.0, 60.0] + [0.0] * 18,
            wp.float32,
            manager,
        ),
        joint_target_vel=_warp_array([0.0] * 24, wp.float32, manager),
        joint_f=_warp_array([0.0] * 24, wp.float32, manager),
    )
    return manager, robot_paths, tcp_paths, object_paths


def _torch_alias(value: object) -> torch.Tensor:
    return torch.from_dlpack(wp.to_dlpack(value))


def _robot_port(
    manager: _Manager,
    robot_paths: tuple[str, ...],
    tcp_paths: tuple[str, ...],
    *,
    tcp_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[NewtonArticulationTensorPort, NewtonArticulationView, NewtonRigidBodyView]:
    articulation = NewtonArticulationView(
        manager,
        paths=robot_paths,
        world_indices=(0, 1, 2),
        controllable_dof_names=("joint_a",),
    )
    tcp = NewtonRigidBodyView(
        manager,
        paths=tcp_paths,
        world_indices=(0, 1, 2),
    )
    port = NewtonArticulationTensorPort(
        label="arm",
        view=articulation,
        tcp_view=tcp,
        command_dof_names=("joint_a",),
        device=torch.device("cuda:0"),
        tcp_offset_xyz=tcp_offset_xyz,
    )
    return port, articulation, tcp


def _projection(
    mode: str,
    method: str,
    *,
    stiffness: float,
    damping: float,
    limit: float,
) -> CommandRuntimeProjection:
    physical = mode if method == "implicit" else "effort"
    return CommandRuntimeProjection(
        joint_names=("joint_a",),
        components=("arm",),
        modes=(mode,),
        methods=(method,),
        physical_modes=(physical,),
        stiffness=np.asarray([stiffness], dtype=float),
        damping=np.asarray([damping], dtype=float),
        effort_limits=np.asarray([limit], dtype=float),
        drive_stiffness=np.asarray(
            [stiffness if method == "implicit" else 0.0], dtype=float
        ),
        drive_damping=np.asarray(
            [damping if method == "implicit" else 0.0], dtype=float
        ),
    )


def test_articulation_port_maps_cuda_env_rows_and_reuses_borrowed_output() -> None:
    manager, robot_paths, tcp_paths, _object_paths = _runtime()
    port, articulation, tcp = _robot_port(manager, robot_paths, tcp_paths)
    assert isinstance(port, RobotTensorPort)
    assert port.command_dim == 1

    ids = torch.tensor([2, 0], device="cuda", dtype=torch.int64)
    first = port.read_joint_positions(ids)
    torch.testing.assert_close(first, torch.tensor([[3.0], [1.0]], device="cuda"))
    pointer = first.data_ptr()
    _torch_alias(manager.state.joint_q)[4] = 33.0
    second = port.read_joint_positions(ids)
    assert second.data_ptr() == pointer
    torch.testing.assert_close(second, torch.tensor([[33.0], [1.0]], device="cuda"))
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        port.read_joint_velocities(ids),
        torch.tensor([[0.3], [0.1]], device="cuda"),
    )

    port.write_joint_positions(ids, torch.tensor([[9.0], [7.0]], device="cuda"))
    port.write_joint_velocities(ids, torch.tensor([[0.9], [0.7]], device="cuda"))
    port.write_joint_targets(ids, torch.tensor([[8.0], [6.0]], device="cuda"))
    torch.testing.assert_close(
        _torch_alias(manager.state.joint_q)[:6],
        torch.tensor([7.0, 10.0, 2.0, 20.0, 9.0, 30.0], device="cuda"),
    )
    torch.testing.assert_close(
        _torch_alias(manager.state.joint_qd)[:6],
        torch.tensor([0.7, 1.0, 0.2, 2.0, 0.9, 3.0], device="cuda"),
    )
    torch.testing.assert_close(
        _torch_alias(manager.control.joint_target_pos)[:6],
        torch.tensor([6.0, 40.0, 5.0, 50.0, 8.0, 60.0], device="cuda"),
    )
    assert ("state", "joint_q", (0, 1, 2)) in manager.events
    assert ("control", "joint_target_pos", (0, 1, 2)) in manager.events

    torch.testing.assert_close(
        port.read_all_joint_positions(ids),
        torch.tensor([[9.0, 30.0], [7.0, 10.0]], device="cuda"),
    )
    port.write_all_joint_positions(
        ids,
        torch.tensor([[11.0, 12.0], [13.0, 14.0]], device="cuda"),
    )
    torch.testing.assert_close(
        port.read_all_joint_positions(ids),
        torch.tensor([[11.0, 12.0], [13.0, 14.0]], device="cuda"),
    )
    assert ("state", "joint_q_full", (0, 1, 2)) in manager.events

    port.close()
    assert not articulation.valid and not tcp.valid


def test_newton_control_runtime_prewarms_and_preserves_owner_addresses() -> None:
    manager, robot_paths, tcp_paths, _object_paths = _runtime()
    port, articulation, _tcp = _robot_port(manager, robot_paths, tcp_paths)
    ids = torch.arange(3, device="cuda", dtype=torch.int64)
    owner_arrays = {
        "ke": manager.model.joint_target_ke,
        "kd": manager.model.joint_target_kd,
        "limit": manager.model.joint_effort_limit,
        "position": manager.control.joint_target_pos,
        "velocity": manager.control.joint_target_vel,
        "effort": manager.control.joint_f,
    }
    owner_pointers = {
        name: _torch_alias(value).data_ptr() for name, value in owner_arrays.items()
    }
    model_before = {
        name: _torch_alias(owner_arrays[name]).clone() for name in ("ke", "kd", "limit")
    }

    position = port.prepare_control_runtime(
        _projection(
            "position",
            "explicit",
            stiffness=2.0,
            damping=4.0,
            limit=6.0,
        )
    )
    velocity = port.prepare_control_runtime(
        _projection(
            "velocity",
            "explicit",
            stiffness=0.0,
            damping=3.0,
            limit=2.0,
        )
    )
    effort = port.prepare_control_runtime(
        _projection(
            "effort",
            "direct",
            stiffness=0.0,
            damping=0.0,
            limit=1.5,
        )
    )

    for name, before in model_before.items():
        torch.testing.assert_close(_torch_alias(owner_arrays[name]), before)
    raw_selection = articulation._qd.selection(None, (0,))
    assert {
        "mode_position_zero",
        "mode_velocity_zero",
        "mode_effort_zero",
        "stiffness",
        "damping",
        "max_efforts",
    }.issubset(raw_selection.staging)
    assert {
        "mode_position_stiffness",
        "mode_position_damping",
        "mode_velocity_damping",
    }.issubset(raw_selection.outputs)
    assert port._q._selected and port._qd._selected
    assert port._effort_target._selected
    prewarmed_pointers = {
        "raw_stiffness": int(raw_selection.staging["stiffness"].ptr),
        "q_selector": next(iter(port._q._selected.values())).data_ptr(),
        "qd_output": next(iter(port._qd._outputs.values())).data_ptr(),
        "effort_selector": next(
            iter(port._effort_target._selected.values())
        ).data_ptr(),
    }

    port.apply_prepared_control_runtime(position)
    desired_qd = torch.tensor([[1.5], [-1.0], [2.0]], device="cuda")
    desired_q = torch.tensor([[5.0], [8.0], [-1.0]], device="cuda")
    port.write_joint_velocity_targets(ids, desired_qd)
    port.write_joint_position_targets(ids, desired_q)
    actual_q = torch.tensor([[1.0], [2.0], [3.0]], device="cuda")
    actual_qd = torch.tensor([[0.1], [0.2], [0.3]], device="cuda")
    expected_pd = torch.clamp(
        2.0 * (desired_q - actual_q) + 4.0 * (desired_qd - actual_qd),
        min=-6.0,
        max=6.0,
    )
    torch.testing.assert_close(
        _torch_alias(manager.control.joint_f)[[0, 2, 4]][:, None],
        expected_pd,
    )

    port.apply_prepared_control_runtime(velocity)
    desired_velocity = torch.tensor([[4.0], [-4.0], [5.0]], device="cuda")
    port.write_joint_velocity_targets(ids, desired_velocity)
    expected_d = torch.clamp(3.0 * (desired_velocity - actual_qd), -2.0, 2.0)
    torch.testing.assert_close(
        _torch_alias(manager.control.joint_f)[[0, 2, 4]][:, None],
        expected_d,
    )

    port.apply_prepared_control_runtime(effort)
    port.write_joint_effort_targets(
        ids,
        torch.tensor([[8.0], [-8.0], [0.5]], device="cuda"),
    )
    torch.testing.assert_close(
        _torch_alias(manager.control.joint_f)[[0, 2, 4]],
        torch.tensor([1.5, -1.5, 0.5], device="cuda"),
    )
    torch.testing.assert_close(
        _torch_alias(manager.control.joint_f)[[1, 3, 5]],
        torch.zeros(3, device="cuda"),
    )
    torch.testing.assert_close(
        _torch_alias(manager.model.joint_target_ke)[[1, 3, 5]],
        torch.full((3,), 100.0, device="cuda"),
    )
    torch.testing.assert_close(
        _torch_alias(manager.model.joint_target_kd)[[1, 3, 5]],
        torch.full((3,), 10.0, device="cuda"),
    )
    torch.testing.assert_close(
        _torch_alias(manager.model.joint_effort_limit)[[1, 3, 5]],
        torch.full((3,), 50.0, device="cuda"),
    )

    assert {
        name: _torch_alias(value).data_ptr() for name, value in owner_arrays.items()
    } == owner_pointers
    assert (
        int(raw_selection.staging["stiffness"].ptr)
        == prewarmed_pointers["raw_stiffness"]
    )
    assert (
        next(iter(port._q._selected.values())).data_ptr()
        == prewarmed_pointers["q_selector"]
    )
    assert (
        next(iter(port._qd._outputs.values())).data_ptr()
        == prewarmed_pointers["qd_output"]
    )
    assert (
        next(iter(port._effort_target._selected.values())).data_ptr()
        == prewarmed_pointers["effort_selector"]
    )
    port.close()


def test_empty_same_step_mask_preserves_newton_owner_state_bitwise() -> None:
    manager, robot_paths, tcp_paths, object_paths = _runtime()
    robot, _articulation, _tcp = _robot_port(manager, robot_paths, tcp_paths)
    rigid_view = NewtonRigidBodyView(
        manager,
        paths=object_paths,
        world_indices=(0, 1, 2),
    )
    rigid = NewtonRigidObjectTensorPort(
        label="Tblock",
        view=rigid_view,
        device=torch.device("cuda:0"),
    )
    ids = torch.arange(3, device="cuda", dtype=torch.int64)
    empty = torch.zeros(3, device="cuda", dtype=torch.bool)
    original = {
        name: _torch_alias(getattr(owner, name)).clone()
        for owner, names in (
            (manager.state, ("joint_q", "joint_qd", "body_q", "body_qd")),
            (manager.control, ("joint_target_pos",)),
        )
        for name in names
    }

    robot.set_device_reset_mask(empty)
    rigid.set_device_reset_mask(empty)
    robot.write_joint_positions(ids, torch.full((3, 1), 99.0, device="cuda"))
    robot.write_joint_velocities(ids, torch.full((3, 1), 88.0, device="cuda"))
    robot.write_joint_targets(ids, torch.full((3, 1), 77.0, device="cuda"))
    rigid.write_pose_wxyz(
        ids,
        torch.full((3, 3), 66.0, device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, device="cuda"),
    )
    rigid.write_velocity(ids, torch.full((3, 6), 55.0, device="cuda"))
    robot.set_device_reset_mask(None)
    rigid.set_device_reset_mask(None)
    wp.synchronize_stream(manager.stream)

    for owner, names in (
        (manager.state, ("joint_q", "joint_qd", "body_q", "body_qd")),
        (manager.control, ("joint_target_pos",)),
    ):
        for name in names:
            assert torch.equal(_torch_alias(getattr(owner, name)), original[name])
    assert {field for _category, field, _mask in manager.device_mask_events} == {
        "joint_q",
        "joint_qd",
        "joint_target_pos",
        "body_q",
        "body_qd",
    }
    assert all(mask is empty for _category, _field, mask in manager.device_mask_events)

    robot.close()
    rigid.close()


def test_articulation_tcp_pose_uses_wxyz_and_fixed_offset_on_gpu() -> None:
    manager, robot_paths, tcp_paths, _object_paths = _runtime()
    port, _articulation, _tcp = _robot_port(
        manager,
        robot_paths,
        tcp_paths,
        tcp_offset_xyz=(1.0, 0.0, 0.0),
    )
    ids = torch.tensor([1], device="cuda", dtype=torch.int64)
    position, orientation = port.read_tcp_pose_wxyz(ids)
    half = 2.0**-0.5
    torch.testing.assert_close(
        position,
        torch.tensor([[1.0, 1.0, 0.0]], device="cuda"),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        orientation,
        torch.tensor([[half, 0.0, 0.0, half]], device="cuda"),
        atol=1.0e-6,
        rtol=0.0,
    )
    port.close()


def test_articulation_port_close_retries_only_failed_raw_view() -> None:
    manager, robot_paths, tcp_paths, _object_paths = _runtime()
    port, articulation, tcp = _robot_port(manager, robot_paths, tcp_paths)
    original_close = articulation.close
    attempts = 0

    def flaky_close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected articulation close failure")
        original_close()

    articulation.close = flaky_close
    with pytest.raises(RuntimeError, match="injected articulation"):
        port.close()
    assert not tcp.valid
    assert articulation.valid

    port.close()
    assert attempts == 2
    assert not articulation.valid


def test_rigid_object_port_double_writes_root_generalized_state() -> None:
    manager, _robot_paths, tcp_paths, object_paths = _runtime()
    view = NewtonRigidBodyView(
        manager,
        paths=object_paths,
        world_indices=(0, 1, 2),
    )
    port = NewtonRigidObjectTensorPort(
        label="Tblock",
        view=view,
        device=torch.device("cuda:0"),
    )
    assert isinstance(port, RigidObjectTensorPort)
    ids = torch.tensor([2, 0], device="cuda", dtype=torch.int64)
    position, orientation = port.read_pose_wxyz(ids)
    pose_pointer = position.data_ptr()
    torch.testing.assert_close(
        position,
        torch.tensor([[12.0, 0.0, 0.0], [10.0, 0.0, 0.0]], device="cuda"),
    )
    torch.testing.assert_close(
        orientation,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device="cuda"),
    )
    torch.testing.assert_close(
        port.read_com_velocity(ids),
        torch.tensor(
            [[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            device="cuda",
        ),
    )
    assert port.read_pose_wxyz(ids)[0].data_ptr() == pose_pointer

    selected = torch.tensor([1], device="cuda", dtype=torch.int64)
    port.write_pose_wxyz(
        selected,
        torch.tensor([[7.0, 8.0, 9.0]], device="cuda"),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], device="cuda"),
    )
    velocity = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], device="cuda")
    port.write_velocity(selected, velocity)
    body_q = _torch_alias(manager.state.body_q)
    body_qd = _torch_alias(manager.state.body_qd)
    torch.testing.assert_close(
        body_q[3],
        torch.tensor([7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.0], device="cuda"),
    )
    torch.testing.assert_close(body_qd[3], velocity[0])
    torch.testing.assert_close(_torch_alias(manager.state.joint_q)[13:20], body_q[3])
    torch.testing.assert_close(_torch_alias(manager.state.joint_qd)[12:18], velocity[0])
    assert ("state", "body_q", (0, 1, 2)) in manager.events
    assert ("state", "body_qd", (0, 1, 2)) in manager.events

    port.close()
    assert not view.valid

    read_only = NewtonRigidBodyView(
        manager,
        paths=tcp_paths,
        world_indices=(0, 1, 2),
    )
    with pytest.raises(ValueError, match="world-root FREE"):
        NewtonRigidObjectTensorPort(
            label="not_a_root",
            view=read_only,
            device=torch.device("cuda:0"),
        )
    read_only.close()


def test_newton_port_source_has_no_host_tensor_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "linkerbot_sim"
        / "kaleidoscope"
        / "newton_ports.py"
    ).read_text(encoding="utf-8")
    for forbidden in (".numpy(", ".cpu(", "import numpy", "np."):
        assert forbidden not in source
    assert "from_dlpack" in source


class _SolverStateRuntime:
    kind = "newton_cuda"

    def __init__(self, *, worlds: int = 4, width: int = 6) -> None:
        self.world_count = worlds
        self.solver_integration_state_width = width
        self.stream = wp.Stream(wp.get_device("cuda:0"))
        self._warp_state = wp.zeros((worlds, width), dtype=wp.float32, device="cuda:0")
        self.state = _torch_alias(self._warp_state)
        self.state.copy_(
            torch.arange(worlds * width, device="cuda", dtype=torch.float32).reshape(
                worlds, width
            )
            + 100.0
        )
        self.baseline = torch.arange(
            worlds * width, device="cuda", dtype=torch.float32
        ).reshape(worlds, width)
        self.active_history: list[torch.Tensor] = []

    def borrow_solver_integration_state(self):
        return self._warp_state

    def set_solver_integration_state(self, values, *, active_world_mask) -> None:
        payload = _torch_alias(values)
        active = _torch_alias_bool(active_world_mask)
        self.active_history.append(active.clone())
        self.state.copy_(torch.where(active[:, None], payload, self.state))

    def reset_solver_integration_state(self, active_world_mask) -> None:
        active = _torch_alias_bool(active_world_mask)
        self.active_history.append(active.clone())
        self.state.copy_(torch.where(active[:, None], self.baseline, self.state))


def _torch_alias_bool(value: object) -> torch.Tensor:
    return torch.from_dlpack(wp.to_dlpack(value))


@pytest.mark.parametrize(
    ("case", "ids", "mask", "expected_active"),
    (
        ("ordinary", (0, 1, 2, 3), None, (True, True, True, True)),
        ("partial", (1, 3), None, (False, True, False, True)),
        ("empty", (0, 1, 2, 3), (False, False, False, False), (False,) * 4),
        ("mixed", (0, 1, 2, 3), (False, True, False, True), (False, True, False, True)),
        ("all_done", (0, 1, 2, 3), (True, True, True, True), (True,) * 4),
    ),
)
def test_solver_integration_port_resets_only_device_selected_worlds(
    case: str,
    ids: tuple[int, ...],
    mask: tuple[bool, ...] | None,
    expected_active: tuple[bool, ...],
) -> None:
    del case
    runtime = _SolverStateRuntime()
    port = NewtonSolverIntegrationTensorPort(runtime=runtime, device="cuda:0")
    before = runtime.state.clone()
    selector = torch.tensor(ids, device="cuda", dtype=torch.int64)
    reset_mask = (
        None if mask is None else torch.tensor(mask, device="cuda", dtype=torch.bool)
    )

    port.reset(selector, reset_mask=reset_mask)
    wp.synchronize_stream(runtime.stream)

    active = torch.tensor(expected_active, device="cuda", dtype=torch.bool)
    torch.testing.assert_close(
        runtime.state,
        torch.where(active[:, None], runtime.baseline, before),
    )
    assert torch.equal(runtime.active_history[-1], active)
    port.close()


def test_solver_integration_port_clone_writer_is_device_only_and_row_isolated() -> None:
    runtime = _SolverStateRuntime()
    port = NewtonSolverIntegrationTensorPort(runtime=runtime, device="cuda:0")
    before = runtime.state.clone()
    source = torch.tensor([0], device="cuda", dtype=torch.int64)
    target = torch.tensor([3], device="cuda", dtype=torch.int64)
    payload = port.tensor.index_select(0, source).clone()

    port.write(target, payload)
    wp.synchronize_stream(runtime.stream)

    torch.testing.assert_close(runtime.state[3], before[0])
    torch.testing.assert_close(runtime.state[:3], before[:3])
    assert torch.equal(
        runtime.active_history[-1],
        torch.tensor([False, False, False, True], device="cuda"),
    )
    port.close()
