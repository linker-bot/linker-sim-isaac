from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.core_api import ArticulationCoreView, RigidPrimCoreView
from linkerbot_sim.isaac.physics.newton.views import (
    NewtonArticulationBinding,
    NewtonArticulationView,
    NewtonRigidBodyView,
    NewtonViewBindingError,
)


wp = pytest.importorskip("warp")


class _Manager:
    def __init__(self, model: object, state: object, control: object) -> None:
        self.model = model
        self.state = state
        self.control = control
        self.closed = False
        self.events: list[tuple[str, str, tuple[int, ...]]] = []

    def on_newton_view_write(
        self,
        *,
        view: object,
        category: str,
        field: str,
        world_indices: tuple[int, ...],
    ) -> None:
        del view
        self.events.append((category, field, world_indices))


def _array(values: object, dtype: object) -> object:
    return wp.array(values, dtype=dtype, device="cpu")


def _runtime() -> _Manager:
    articulation_paths = (
        "/World/envs/env_0/robot",
        "/World/envs/env_1/robot",
    )
    object_paths = (
        "/World/envs/env_0/object",
        "/World/envs/env_1/object",
    )
    joint_labels = [
        f"{articulation_paths[0]}/root_joint",
        f"{articulation_paths[0]}/joint_a",
        f"{articulation_paths[0]}/joint_follower",
        f"{articulation_paths[1]}/root_joint",
        f"{articulation_paths[1]}/joint_a",
        f"{articulation_paths[1]}/joint_follower",
        f"{object_paths[0]}/free_joint",
        f"{object_paths[1]}/free_joint",
    ]
    # Two fixed-root articulations with two scalar DOFs each, followed by two
    # world-root FREE rigid objects.
    joint_q_start = [0, 0, 1, 2, 2, 3, 4, 11, 18]
    joint_qd_start = [0, 0, 1, 2, 2, 3, 4, 10, 16]
    model = SimpleNamespace(
        device="cpu",
        world_count=2,
        articulation_label=list(articulation_paths),
        articulation_world=_array([0, 1], wp.int32),
        articulation_start=_array([0, 3, 6], wp.int32),
        joint_label=joint_labels,
        joint_world=_array([0, 0, 0, 1, 1, 1, 0, 1], wp.int32),
        joint_type=_array([3, 1, 1, 3, 1, 1, 4, 4], wp.int32),
        joint_q_start=_array(joint_q_start, wp.int32),
        joint_qd_start=_array(joint_qd_start, wp.int32),
        joint_child=_array([2, 3, 4, 5, 6, 7, 0, 1], wp.int32),
        joint_parent=_array([-1] * 8, wp.int32),
        body_label=list(object_paths),
        body_world=_array([0, 1], wp.int32),
        joint_target_ke=_array([100.0, 0.0, 110.0, 0.0] + [0.0] * 12, wp.float32),
        joint_target_kd=_array([10.0, 0.0, 11.0, 0.0] + [0.0] * 12, wp.float32),
        joint_effort_limit=_array([50.0] * 16, wp.float32),
    )
    joint_q = [1.0, 2.0, 10.0, 20.0]
    joint_q.extend([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    joint_q.extend([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0])
    joint_qd = [0.1, 0.2, 1.0, 2.0]
    joint_qd.extend([0.0] * 12)
    state = SimpleNamespace(
        joint_q=_array(joint_q, wp.float32),
        joint_qd=_array(joint_qd, wp.float32),
        body_q=_array(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
            ],
            wp.transform,
        ),
        body_qd=_array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            wp.spatial_vector,
        ),
    )
    control = SimpleNamespace(
        joint_target_pos=_array([3.0, 30.0, 4.0, 40.0] + [0.0] * 12, wp.float32),
        joint_target_vel=_array([0.3, 3.0, 0.4, 4.0] + [0.0] * 12, wp.float32),
        joint_f=_array([0.0] * 16, wp.float32),
    )
    return _Manager(model, state, control)


def _numpy(value: object) -> np.ndarray:
    return np.asarray(value.numpy()).copy()


def test_articulation_view_binds_exact_world_rows_and_scatters_targets() -> None:
    manager = _runtime()
    paths = tuple(manager.model.articulation_label)
    view = NewtonArticulationView(
        manager,
        paths=paths,
        world_indices=(0, 1),
        controllable_dof_names=("joint_a",),
    )

    assert view.count == 2
    assert view.num_dofs == 2
    assert view.dof_names == ["joint_a", "joint_follower"]
    assert view.controllable_dof_names == ("joint_a",)
    np.testing.assert_allclose(_numpy(view.get_dof_positions()), [[1, 2], [10, 20]])
    np.testing.assert_allclose(
        _numpy(view.get_dof_velocities(indices=[1], dof_indices=[0])), [[1.0]]
    )

    # The first target call creates its selector/buffer cache.  Repeating the
    # same selection reuses it and writes only the mapped global DOFs.
    view.prepare_dof_selection(dof_indices=[0])
    view.set_dof_position_targets([[7.0], [8.0]], dof_indices=[0])
    view.set_dof_velocity_targets(
        wp.array([[0.7], [0.8]], dtype=wp.float32, device="cpu"),
        dof_indices=[0],
    )
    np.testing.assert_allclose(
        _numpy(manager.control.joint_target_pos)[:4],
        [7.0, 30.0, 8.0, 40.0],
    )
    np.testing.assert_allclose(
        _numpy(manager.control.joint_target_vel)[:4],
        [0.7, 3.0, 0.8, 4.0],
    )
    with pytest.raises(RuntimeError, match="follower"):
        view.set_dof_position_targets([[99.0]], indices=[0], dof_indices=[1])

    assert ("control", "joint_target_pos", (0, 1)) in manager.events
    assert ("control", "joint_target_vel", (0, 1)) in manager.events


def test_articulation_state_writes_preserve_targets_and_core_cold_contract() -> None:
    manager = _runtime()
    raw = NewtonArticulationView(
        manager,
        paths=manager.model.articulation_label,
        controllable_dof_names=("joint_a",),
    )
    core = ArticulationCoreView(raw, physics_backend="newton")
    before_targets = _numpy(manager.control.joint_target_pos)

    core.set_joint_positions([[42.0]], indices=[1], joint_indices=[0])
    core.set_joint_velocities([[4.2]], indices=[1], joint_indices=[0])

    np.testing.assert_allclose(
        core.get_joint_positions(indices=[0, 1], joint_indices=[0]),
        [[1.0], [42.0]],
    )
    np.testing.assert_allclose(
        core.get_joint_velocities(indices=[0, 1], joint_indices=[0]),
        [[0.1], [4.2]],
    )
    np.testing.assert_array_equal(
        _numpy(manager.control.joint_target_pos), before_targets
    )
    assert ("state", "joint_q", (1,)) in manager.events
    assert ("state", "joint_qd", (1,)) in manager.events


def test_articulation_modes_restore_persistent_default_gains() -> None:
    manager = _runtime()
    view = NewtonArticulationView(
        manager,
        paths=manager.model.articulation_label,
        controllable_dof_names=("joint_a",),
    )

    view.set_dof_gains(
        stiffnesses=[[120.0], [130.0]],
        dampings=[[12.0], [13.0]],
        dof_indices=[0],
    )
    view.switch_dof_control_mode("velocity", dof_indices=[0])
    stiffness, damping = view.get_dof_gains(dof_indices=[0])
    np.testing.assert_allclose(_numpy(stiffness), [[0.0], [0.0]])
    np.testing.assert_allclose(_numpy(damping), [[12.0], [13.0]])

    view.switch_dof_control_mode("position", dof_indices=[0])
    stiffness, damping = view.get_dof_gains(dof_indices=[0])
    np.testing.assert_allclose(_numpy(stiffness), [[120.0], [130.0]])
    np.testing.assert_allclose(_numpy(damping), [[12.0], [13.0]])


def test_articulation_binding_rejects_non_exact_or_nonhomogeneous_rows() -> None:
    manager = _runtime()
    with pytest.raises(NewtonViewBindingError, match="exactly one"):
        NewtonArticulationView(
            manager,
            paths=["/World/envs/env_0/robot*"],
            controllable_dof_names=("joint_a",),
        )
    with pytest.raises(NewtonViewBindingError, match="requested world order"):
        NewtonArticulationView(
            manager,
            paths=list(reversed(manager.model.articulation_label)),
            world_indices=(0, 1),
            controllable_dof_names=("joint_a",),
        )

    manager.model.joint_label[4] = "/World/envs/env_1/robot/different_joint"
    with pytest.raises(NewtonViewBindingError, match="identical ordered DOFs"):
        NewtonArticulationView(
            manager,
            paths=manager.model.articulation_label,
            controllable_dof_names=("joint_a",),
        )


def test_articulation_target_write_requires_command_dof_binding() -> None:
    manager = _runtime()
    view = NewtonArticulationView(
        manager,
        paths=manager.model.articulation_label,
    )

    with pytest.raises(RuntimeError, match="must be bound"):
        view.set_dof_position_targets([[1.0], [2.0]], dof_indices=[0])


def test_articulation_infers_follower_exclusion_from_manager_audit() -> None:
    manager = _runtime()
    manager.constraint_audit = SimpleNamespace(
        bindings=(
            SimpleNamespace(follower_qd_index=1),
            SimpleNamespace(follower_qd_index=3),
        )
    )
    view = NewtonArticulationView(
        manager,
        paths=manager.model.articulation_label,
    )

    assert view.controllable_dof_names == ("joint_a",)
    view.set_dof_position_targets([[8.0], [9.0]], dof_indices=[0])
    with pytest.raises(RuntimeError, match="follower"):
        view.set_dof_position_targets([[8.0], [9.0]], dof_indices=[1])


def test_articulation_explicit_controls_and_gains_reject_native_followers() -> None:
    manager = _runtime()
    manager.constraint_audit = SimpleNamespace(
        bindings=(
            SimpleNamespace(follower_qd_index=1),
            SimpleNamespace(follower_qd_index=3),
        )
    )

    with pytest.raises(RuntimeError, match="native-equality followers"):
        NewtonArticulationView(
            manager,
            paths=manager.model.articulation_label,
            controllable_dof_names=("joint_a", "joint_follower"),
        )

    view = NewtonArticulationView(
        manager,
        paths=manager.model.articulation_label,
    )
    with pytest.raises(RuntimeError, match="gains.*follower"):
        view.set_dof_gains(
            stiffnesses=[[5.0], [6.0]],
            dof_indices=[1],
        )


@pytest.mark.parametrize("world_count", (1, 16, 1024))
def test_articulation_binding_has_no_small_world_count_limit(world_count: int) -> None:
    paths = tuple(f"/World/envs/env_{world}/robot" for world in range(world_count))
    model = SimpleNamespace(
        articulation_label=list(paths),
        articulation_world=np.arange(world_count, dtype=np.int32),
        articulation_start=np.arange(world_count + 1, dtype=np.int32),
        joint_label=[f"{path}/joint" for path in paths],
        joint_world=np.arange(world_count, dtype=np.int32),
        joint_q_start=np.arange(world_count + 1, dtype=np.int32),
        joint_qd_start=np.arange(world_count + 1, dtype=np.int32),
    )

    binding = NewtonArticulationBinding.from_model(
        model,
        paths,
        world_indices=tuple(range(world_count)),
    )

    assert binding.count == world_count
    assert binding.num_dofs == 1
    assert binding.world_indices[0] == 0
    assert binding.world_indices[-1] == world_count - 1


def test_rigid_view_updates_body_and_free_joint_state_in_selected_world() -> None:
    manager = _runtime()
    raw = NewtonRigidBodyView(
        manager,
        paths=manager.model.body_label,
        world_indices=(0, 1),
    )
    core = RigidPrimCoreView(raw, physics_backend="newton")

    positions, orientations = core.get_world_poses()
    np.testing.assert_allclose(positions, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(orientations, [[1, 0, 0, 0], [1, 0, 0, 0]])

    core.set_world_poses(
        positions=[[7.0, 8.0, 9.0]],
        orientations=[[0.0, 1.0, 0.0, 0.0]],
        indices=[1],
    )
    core.set_velocities(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        indices=[1],
    )

    positions, orientations = core.get_world_poses(indices=[1])
    np.testing.assert_allclose(positions, [[7, 8, 9]])
    np.testing.assert_allclose(orientations, [[0, 1, 0, 0]])
    np.testing.assert_allclose(
        core.get_velocities(indices=[1]),
        [[1, 2, 3, 4, 5, 6]],
    )
    joint_q = _numpy(manager.state.joint_q)
    joint_qd = _numpy(manager.state.joint_qd)
    np.testing.assert_allclose(joint_q[11:18], [7, 8, 9, 1, 0, 0, 0])
    np.testing.assert_allclose(joint_qd[10:16], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(joint_q[4:11], [1, 2, 3, 0, 0, 0, 1])
    assert ("state", "body_q", (1,)) in manager.events
    assert ("state", "body_qd", (1,)) in manager.events


def test_rigid_view_rejects_writes_without_world_root_free_joint() -> None:
    manager = _runtime()
    manager.model.joint_type = _array([3, 1, 1, 3, 1, 1, 3, 4], wp.int32)
    view = NewtonRigidBodyView(manager, paths=manager.model.body_label)

    with pytest.raises(RuntimeError, match="FREE bodies"):
        view.set_world_poses(positions=[[0.0, 0.0, 0.0]], indices=[0])
