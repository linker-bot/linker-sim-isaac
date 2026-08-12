from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from linkerbot_sim.isaac.physics.core_api import (
    ArticulationCoreView,
    ExperimentalArticulationAction,
    ExperimentalArticulationController,
    RigidPrimCoreView,
    SingleArticulationCoreView,
    create_articulation_core_view,
    create_rigid_prim_core_view,
    create_single_articulation_core_view,
    use_experimental_core,
)
from linkerbot_sim.isaac.physics.core_api import (
    _newton_articulation_ids_for_rigid_rows,
    _newton_rigid_tensor_context,
)


class _WarpLike:
    def __init__(self, value: object) -> None:
        self._value = np.asarray(value)

    def numpy(self) -> np.ndarray:
        return self._value.copy()


class _ExperimentalArticulation:
    dof_names = ["joint_0", "joint_1", "joint_2"]
    num_dofs = 3

    def __init__(self) -> None:
        self.calls: list[
            tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]
        ] = []

    def __len__(self) -> int:
        return 2

    def get_dof_positions(self, *, indices=None, dof_indices=None):
        return _WarpLike([[1.0, 3.0], [4.0, 6.0]])

    def get_dof_velocities(self, *, indices=None, dof_indices=None):
        return _WarpLike([[0.1, 0.3], [0.4, 0.6]])

    def get_dof_position_targets(self, *, indices=None, dof_indices=None):
        return _WarpLike([[0.2, 0.4], [0.5, 0.7]])

    def get_dof_velocity_targets(self, *, indices=None, dof_indices=None):
        return _WarpLike([[-0.2, -0.4], [-0.5, -0.7]])

    def _record(self, name, values, indices=None, dof_indices=None):
        self.calls.append(
            (
                name,
                np.asarray(values),
                None if indices is None else np.asarray(indices),
                None if dof_indices is None else np.asarray(dof_indices),
            )
        )

    def set_dof_positions(self, values, *, indices=None, dof_indices=None):
        self._record("positions", values, indices, dof_indices)

    def set_dof_velocities(self, values, *, indices=None, dof_indices=None):
        self._record("velocities", values, indices, dof_indices)

    def set_dof_position_targets(self, values, *, indices=None, dof_indices=None):
        self._record("position_targets", values, indices, dof_indices)

    def set_dof_velocity_targets(self, values, *, indices=None, dof_indices=None):
        self._record("velocity_targets", values, indices, dof_indices)


class _ExperimentalSingleArticulation(_ExperimentalArticulation):
    def __init__(self) -> None:
        super().__init__()
        self.gains = (
            np.asarray([[1.0, 2.0, 3.0]], dtype=float),
            np.asarray([[0.1, 0.2, 0.3]], dtype=float),
        )
        self.max_efforts = np.asarray([[10.0, 20.0, 30.0]], dtype=float)
        self.control_modes: list[tuple[str, np.ndarray | None]] = []
        self.drive_modes: list[tuple[str, np.ndarray | None]] = []

    def __len__(self) -> int:
        return 1

    def get_dof_positions(self, *, indices=None, dof_indices=None):
        return _WarpLike([[1.0, 2.0, 3.0]])

    def get_dof_velocities(self, *, indices=None, dof_indices=None):
        return _WarpLike([[0.1, 0.2, 0.3]])

    def get_dof_position_targets(self, *, indices=None, dof_indices=None):
        return _WarpLike([[0.2, 0.4, 0.6]])

    def get_dof_velocity_targets(self, *, indices=None, dof_indices=None):
        return _WarpLike([[-0.2, -0.4, -0.6]])

    def get_dof_efforts(self, *, indices=None, dof_indices=None):
        return _WarpLike([[4.0, 5.0, 6.0]])

    def get_dof_projected_joint_forces(self, *, indices=None, dof_indices=None):
        return _WarpLike([[7.0, 8.0, 9.0]])

    def set_dof_efforts(self, values, *, indices=None, dof_indices=None):
        self._record("efforts", values, indices, dof_indices)

    def get_dof_gains(self, *, indices=None, dof_indices=None):
        return _WarpLike(self.gains[0]), _WarpLike(self.gains[1])

    def set_dof_gains(
        self,
        stiffnesses=None,
        dampings=None,
        *,
        indices=None,
        dof_indices=None,
    ):
        self.gains = (
            np.asarray(stiffnesses, dtype=float).reshape(1, -1),
            np.asarray(dampings, dtype=float).reshape(1, -1),
        )

    def get_dof_max_efforts(self, *, indices=None, dof_indices=None):
        return _WarpLike(self.max_efforts)

    def set_dof_max_efforts(self, values, *, indices=None, dof_indices=None):
        self.max_efforts = np.asarray(values, dtype=float).reshape(1, -1)

    def switch_dof_control_mode(self, mode, *, indices=None, dof_indices=None):
        self.control_modes.append(
            (
                str(mode),
                None if dof_indices is None else np.asarray(dof_indices),
            )
        )

    def set_dof_drive_types(self, mode, *, indices=None, dof_indices=None):
        self.drive_modes.append(
            (
                str(mode),
                None if dof_indices is None else np.asarray(dof_indices),
            )
        )


def test_experimental_articulation_preserves_partial_joint_api() -> None:
    raw = _ExperimentalArticulation()
    view = ArticulationCoreView(raw)

    np.testing.assert_allclose(
        view.get_joint_positions(indices=[0, 1], joint_indices=[0, 2]),
        [[1.0, 3.0], [4.0, 6.0]],
    )
    view.set_joint_positions([[7.0, 9.0]], indices=[1], joint_indices=[0, 2])
    view.apply_action(
        SimpleNamespace(
            joint_positions=[[0.5, 0.7]],
            joint_velocities=[[1.5, 1.7]],
            joint_efforts=None,
            joint_indices=[0, 2],
        )
    )

    assert [call[0] for call in raw.calls] == [
        "positions",
        "position_targets",
        "velocity_targets",
    ]
    np.testing.assert_array_equal(raw.calls[0][2], [1])
    np.testing.assert_array_equal(raw.calls[0][3], [0, 2])
    assert view.count == 2
    assert view.num_dof == 3


def test_newton_articulation_teleports_selected_rows_through_full_batch(
    monkeypatch,
) -> None:
    float32 = object()
    int32 = object()
    conversions: list[tuple[np.ndarray, object, object]] = []
    warp = ModuleType("warp")
    warp.float32 = float32
    warp.int32 = int32

    def from_numpy(values, *, dtype, device):
        array = np.asarray(values).copy()
        conversions.append((array, dtype, device))
        return array

    warp.from_numpy = from_numpy
    monkeypatch.setitem(sys.modules, "warp", warp)

    class Raw:
        dof_names = ["joint_0", "joint_1", "joint_2"]
        num_dofs = 3

        def __init__(self) -> None:
            self.positions = np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=float)
            self.velocities = np.zeros_like(self.positions)
            self.position_targets = np.asarray(
                [[0.3, 0.4, 0.5], [1.3, 1.4, 1.5]], dtype=float
            )
            self.velocity_targets = np.asarray(
                [[-0.3, -0.4, -0.5], [-1.3, -1.4, -1.5]], dtype=float
            )
            self.calls: list[tuple[str, np.ndarray, object, object]] = []
            self._physics_articulation_view = TensorView(self)

        def __len__(self) -> int:
            return 2

        def get_dof_positions(self, *, indices=None, dof_indices=None):
            return _WarpLike(self.positions)

        def get_dof_velocities(self, *, indices=None, dof_indices=None):
            return _WarpLike(self.velocities)

        def get_dof_position_targets(self, *, indices=None, dof_indices=None):
            return _WarpLike(self.position_targets)

        def get_dof_velocity_targets(self, *, indices=None, dof_indices=None):
            return _WarpLike(self.velocity_targets)

        def set_dof_positions(self, values, *, indices=None, dof_indices=None):
            self.calls.append(
                ("public_positions", np.asarray(values), indices, dof_indices)
            )
            self.positions = np.asarray(values, dtype=float).copy()
            self.position_targets = np.asarray(values, dtype=float).copy()

        def set_dof_velocities(self, values, *, indices=None, dof_indices=None):
            self.calls.append(
                ("public_velocities", np.asarray(values), indices, dof_indices)
            )
            self.velocities = np.asarray(values, dtype=float).copy()
            self.velocity_targets = np.asarray(values, dtype=float).copy()

    class TensorView:
        count = 2
        max_dofs = 3

        def __init__(self, owner: Raw) -> None:
            self.owner = owner

        def get_dof_positions(self):
            return SimpleNamespace(device="cuda:7")

        def get_dof_velocities(self):
            return SimpleNamespace(device="cuda:7")

        def set_dof_positions(self, values, indices) -> None:
            selected = np.asarray(indices, dtype=int)
            self.owner.calls.append(
                ("tensor_positions", np.asarray(values), selected, None)
            )
            self.owner.positions[selected] = np.asarray(values)[selected]

        def set_dof_velocities(self, values, indices) -> None:
            selected = np.asarray(indices, dtype=int)
            self.owner.calls.append(
                ("tensor_velocities", np.asarray(values), selected, None)
            )
            self.owner.velocities[selected] = np.asarray(values)[selected]

    raw = Raw()
    view = ArticulationCoreView(raw, physics_backend="newton")

    view.set_joint_positions([[7.0, 9.0]], indices=[1], joint_indices=[0, 2])
    view.set_joint_velocities([[0.7, 0.9]], indices=[1], joint_indices=[0, 2])

    np.testing.assert_allclose(
        view.get_joint_positions(indices=[1], joint_indices=[0, 2]),
        [[7.0, 9.0]],
    )
    np.testing.assert_allclose(
        view.get_joint_velocities(indices=[1], joint_indices=[0, 2]),
        [[0.7, 0.9]],
    )
    np.testing.assert_allclose(
        view.get_joint_position_targets(indices=[1], joint_indices=[0, 2]),
        [[1.3, 1.5]],
    )
    np.testing.assert_allclose(
        view.get_joint_velocity_targets(indices=[1], joint_indices=[0, 2]),
        [[-1.3, -1.5]],
    )
    np.testing.assert_allclose(raw.positions, [[0.0, 0.1, 0.2], [7.0, 1.1, 9.0]])
    np.testing.assert_allclose(raw.velocities, [[0.0, 0.0, 0.0], [0.7, 0.0, 0.9]])
    np.testing.assert_allclose(raw.position_targets, [[0.3, 0.4, 0.5], [1.3, 1.4, 1.5]])
    np.testing.assert_allclose(
        raw.velocity_targets,
        [[-0.3, -0.4, -0.5], [-1.3, -1.4, -1.5]],
    )
    assert [call[0] for call in raw.calls] == [
        "tensor_positions",
        "tensor_velocities",
    ]
    for call in raw.calls:
        np.testing.assert_array_equal(call[2], [0, 1])
    assert len(conversions) == 4
    for index, (values, dtype, device) in enumerate(conversions):
        assert dtype is (float32 if index % 2 == 0 else int32)
        assert device == "cuda:7"
        assert values.dtype == (np.float32 if index % 2 == 0 else np.int32)
        assert values.shape == ((2, 3) if index % 2 == 0 else (2,))


def test_newton_articulation_rejects_state_shape_that_differs_from_raw_view(
    monkeypatch,
) -> None:
    class TensorView:
        count = 2
        max_dofs = 4

    class Raw:
        dof_names = ["joint_0", "joint_1", "joint_2"]
        num_dofs = 3
        _physics_articulation_view = TensorView()

        def __len__(self) -> int:
            return 2

        def get_dof_positions(self, *, indices=None, dof_indices=None):
            return _WarpLike(np.zeros((2, 3), dtype=np.float32))

    view = ArticulationCoreView(Raw(), physics_backend="newton")

    with pytest.raises(
        ValueError,
        match=r"expected=\(2, 4\), actual=\(2, 3\)",
    ):
        view.set_joint_positions(np.zeros((2, 3), dtype=np.float32))


def test_experimental_single_articulation_restores_legacy_contract() -> None:
    raw = _ExperimentalSingleArticulation()
    view = SingleArticulationCoreView(
        raw,
        prim_path="/World/Robot",
        name="robot",
        physics_backend="newton",
    )

    np.testing.assert_allclose(view.get_joint_positions(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(view.get_joint_velocities(), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(view.get_joint_position_targets(), [0.2, 0.4, 0.6])
    np.testing.assert_allclose(view.get_joint_velocity_targets(), [-0.2, -0.4, -0.6])
    np.testing.assert_allclose(view.get_applied_joint_efforts(), [4.0, 5.0, 6.0])
    view.apply_action(
        ExperimentalArticulationAction(
            joint_positions=[0.4, 0.6],
            joint_velocities=[1.4, 1.6],
            joint_efforts=[2.4, 2.6],
            joint_indices=[0, 2],
        )
    )

    assert [call[0] for call in raw.calls] == [
        "position_targets",
        "velocity_targets",
        "efforts",
    ]
    for call in raw.calls:
        np.testing.assert_array_equal(call[3], [0, 2])
    assert view.num_dof == 3
    assert view.dof_names == ["joint_0", "joint_1", "joint_2"]
    assert view.requires_scene_registration is False
    assert view.supports_per_link_gravity is False
    with pytest.raises(RuntimeError, match="per-link gravity"):
        view.disable_gravity()
    with pytest.raises(RuntimeError, match="measured joint efforts"):
        view.get_measured_joint_efforts()


def test_newton_single_state_setters_keep_internal_batch_dimension() -> None:
    raw = _ExperimentalSingleArticulation()
    view = SingleArticulationCoreView(
        raw,
        prim_path="/World/Robot",
        name="robot",
        physics_backend="newton",
    )
    writes: list[dict[str, np.ndarray]] = []
    view._set_newton_tensor_dof_state = lambda **values: writes.append(  # type: ignore[method-assign]
        {name: np.asarray(value).copy() for name, value in values.items()}
    )

    view.set_joint_positions([7.0, 9.0], joint_indices=[0, 2])
    view.set_joint_velocities([0.7, 0.9], joint_indices=[0, 2])

    assert len(writes) == 2
    np.testing.assert_allclose(writes[0]["positions"], [[7.0, 2.0, 9.0]])
    np.testing.assert_allclose(writes[1]["velocities"], [[0.7, 0.2, 0.9]])


def test_experimental_single_controller_maps_dof_configuration() -> None:
    raw = _ExperimentalSingleArticulation()
    view = SingleArticulationCoreView(
        raw,
        prim_path="/World/Robot",
        name="robot",
        physics_backend="newton",
    )
    controller = view.get_articulation_controller()

    kps, kds = controller.get_gains()
    np.testing.assert_allclose(kps, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(kds, [0.1, 0.2, 0.3])
    controller.set_gains(kps=[4.0, 5.0, 6.0], kds=[0.4, 0.5, 0.6])
    controller.set_max_efforts([40.0, 50.0, 60.0])
    controller.switch_dof_control_mode(dof_index=1, mode="velocity")
    controller.set_effort_modes("force", joint_indices=[0, 2])

    np.testing.assert_allclose(controller.get_gains()[0], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(controller.get_max_efforts(), [40.0, 50.0, 60.0])
    assert raw.control_modes[0][0] == "velocity"
    np.testing.assert_array_equal(raw.control_modes[0][1], [1])
    assert raw.drive_modes[0][0] == "force"
    np.testing.assert_array_equal(raw.drive_modes[0][1], [0, 2])


def test_newton_runtime_controller_slices_model_writes_to_audited_dofs() -> None:
    class NewtonRaw:
        dof_names = ["joint_0", "native_follower", "joint_2"]
        controllable_dof_names = ("joint_0", "joint_2")

        def __init__(self) -> None:
            self.gain_write = None
            self.effort_write = None
            self.control_mode_writes = []
            self.drive_type_writes = []

        def set_dof_gains(
            self,
            stiffnesses=None,
            dampings=None,
            *,
            dof_indices=None,
        ) -> None:
            self.gain_write = (
                np.asarray(stiffnesses),
                np.asarray(dampings),
                np.asarray(dof_indices),
            )

        def set_dof_max_efforts(self, values, *, dof_indices=None) -> None:
            self.effort_write = (np.asarray(values), np.asarray(dof_indices))

        def switch_dof_control_mode(
            self,
            mode,
            *,
            dof_indices=None,
        ) -> None:
            self.control_mode_writes.append((str(mode), np.asarray(dof_indices)))

        def set_dof_drive_types(
            self,
            mode,
            *,
            dof_indices=None,
        ) -> None:
            self.drive_type_writes.append((str(mode), np.asarray(dof_indices)))

    NewtonRaw.__module__ = "linkerbot_sim.isaac.physics.newton.views"
    raw = NewtonRaw()
    controller = ExperimentalArticulationController(raw)

    controller.set_gains(kps=[4.0, 0.0, 6.0], kds=[0.4, 0.0, 0.6])
    controller.set_max_efforts([40.0, 0.0, 60.0])
    controller.switch_control_mode("position")
    controller.set_effort_modes("force")

    np.testing.assert_allclose(raw.gain_write[0], [4.0, 6.0])
    np.testing.assert_allclose(raw.gain_write[1], [0.4, 0.6])
    np.testing.assert_array_equal(raw.gain_write[2], [0, 2])
    np.testing.assert_allclose(raw.effort_write[0], [40.0, 60.0])
    np.testing.assert_array_equal(raw.effort_write[1], [0, 2])
    assert raw.control_mode_writes[0][0] == "position"
    np.testing.assert_array_equal(raw.control_mode_writes[0][1], [0, 2])
    assert raw.drive_type_writes[0][0] == "force"
    np.testing.assert_array_equal(raw.drive_type_writes[0][1], [0, 2])

    with pytest.raises(ValueError, match="complete legacy DOF axis"):
        controller.set_gains(kps=[4.0, 6.0])
    with pytest.raises(RuntimeError, match="audited controllable set"):
        controller.set_max_efforts([50.0], joint_indices=[1])
    with pytest.raises(RuntimeError, match="audited controllable set"):
        controller.switch_dof_control_mode(dof_index=1, mode="velocity")
    with pytest.raises(RuntimeError, match="audited controllable set"):
        controller.set_effort_modes("force", joint_indices=[1])

    assert len(raw.control_mode_writes) == 1
    assert len(raw.drive_type_writes) == 1


def test_experimental_rigid_view_combines_and_splits_velocities() -> None:
    class Raw:
        def __init__(self) -> None:
            self.written = None

        def get_velocities(self, *, indices=None):
            return _WarpLike([[1.0, 2.0, 3.0]]), _WarpLike([[4.0, 5.0, 6.0]])

        def set_velocities(self, linear, angular, *, indices=None):
            self.written = (
                np.asarray(linear),
                np.asarray(angular),
                np.asarray(indices),
            )

    raw = Raw()
    view = RigidPrimCoreView(raw)
    np.testing.assert_allclose(view.get_velocities(indices=[2]), [[1, 2, 3, 4, 5, 6]])
    view.set_velocities([[7, 8, 9, 10, 11, 12]], indices=[2])
    np.testing.assert_allclose(raw.written[0], [[7, 8, 9]])
    np.testing.assert_allclose(raw.written[1], [[10, 11, 12]])
    np.testing.assert_array_equal(raw.written[2], [2])


def test_newton_rigid_teleports_selected_rows_through_full_batch() -> None:
    class Raw:
        def __init__(self) -> None:
            self.positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            self.orientations = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
            self.linear = np.zeros((2, 3), dtype=float)
            self.angular = np.zeros((2, 3), dtype=float)
            self.pose_indices = object()
            self.velocity_indices = object()

        def __len__(self) -> int:
            return 2

        def get_world_poses(self, *, indices=None):
            return _WarpLike(self.positions), _WarpLike(self.orientations)

        def set_world_poses(self, positions=None, orientations=None, *, indices=None):
            self.pose_indices = indices
            if positions is not None:
                self.positions = np.asarray(positions, dtype=float).copy()
            if orientations is not None:
                self.orientations = np.asarray(orientations, dtype=float).copy()

        def get_velocities(self, *, indices=None):
            return _WarpLike(self.linear), _WarpLike(self.angular)

        def set_velocities(self, linear, angular, *, indices=None):
            self.velocity_indices = indices
            self.linear = np.asarray(linear, dtype=float).copy()
            self.angular = np.asarray(angular, dtype=float).copy()

    raw = Raw()
    view = RigidPrimCoreView(raw, physics_backend="newton")

    view.set_world_poses(
        positions=[[2.0, 3.0, 4.0]],
        orientations=[[0.0, 1.0, 0.0, 0.0]],
        indices=[1],
    )
    view.set_velocities([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], indices=[1])

    selected_positions, selected_orientations = view.get_world_poses(indices=[1])
    np.testing.assert_allclose(selected_positions, [[2.0, 3.0, 4.0]])
    np.testing.assert_allclose(selected_orientations, [[0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        view.get_velocities(indices=[1]),
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
    )
    np.testing.assert_allclose(raw.positions, [[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    np.testing.assert_allclose(
        raw.orientations,
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(raw.linear, [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(raw.angular, [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
    assert raw.pose_indices is None
    assert raw.velocity_indices is None


def test_newton_rigid_tensor_velocity_read_uses_linear_angular_order() -> None:
    class TensorView:
        def get_velocities(self):
            return _WarpLike(
                [
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
                ]
            )

    class Raw:
        _physics_rigid_body_view = TensorView()

        def __len__(self) -> int:
            return 2

    view = RigidPrimCoreView(Raw(), physics_backend="newton")

    np.testing.assert_allclose(
        view.get_velocities(),
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        ],
    )


def test_newton_rigid_full_batch_velocity_write_uses_linear_angular_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warp = ModuleType("warp")
    warp.float32 = object()  # type: ignore[attr-defined]
    warp.int32 = object()  # type: ignore[attr-defined]
    warp.from_numpy = lambda values, **_kwargs: np.asarray(values).copy()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "warp", warp)

    class TensorView:
        def __init__(self) -> None:
            self.values = np.zeros((2, 6), dtype=np.float32)
            self.indices = None

        def get_velocities(self):
            return SimpleNamespace(device="cuda:0")

        def set_velocities(self, values, indices) -> None:
            self.values = np.asarray(values).copy()
            self.indices = np.asarray(indices).copy()

    tensor_view = TensorView()

    class Raw:
        _physics_rigid_body_view = tensor_view

        def __len__(self) -> int:
            return 2

        def set_velocities(self, *_args, **_kwargs) -> None:
            raise AssertionError(
                "Newton full-batch writes must use the raw tensor view"
            )

    view = RigidPrimCoreView(Raw(), physics_backend="newton")
    view.set_velocities(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        ]
    )

    np.testing.assert_allclose(
        tensor_view.values,
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        ],
    )
    np.testing.assert_array_equal(tensor_view.indices, [0, 1])


def test_newton_rigid_rows_map_to_only_their_articulations() -> None:
    tensor_view = SimpleNamespace(
        count=4,
        body_paths=["/a/0", "/a/1", "/b/0", "/b/1"],
    )
    model = SimpleNamespace(
        body_label=tensor_view.body_paths,
        joint_child=_WarpLike([0, 1, 2, 3]),
        joint_articulation=_WarpLike([3, 3, 7, 7]),
    )

    articulation_ids = _newton_articulation_ids_for_rigid_rows(
        raw_view=SimpleNamespace(paths=tensor_view.body_paths),
        tensor_view=tensor_view,
        model=model,
        row_indices=np.asarray([2, 3]),
    )

    np.testing.assert_array_equal(articulation_ids, [7])


def test_newton_rigid_row_mapping_rejects_tensor_path_reordering() -> None:
    model = SimpleNamespace(
        body_label=["/a", "/b"],
        joint_child=_WarpLike([0, 1]),
        joint_articulation=_WarpLike([3, 7]),
    )

    with pytest.raises(RuntimeError, match="path order differs"):
        _newton_articulation_ids_for_rigid_rows(
            raw_view=SimpleNamespace(paths=["/a", "/b"]),
            tensor_view=SimpleNamespace(count=2, prim_paths=["/b", "/a"]),
            model=model,
            row_indices=np.asarray([0, 1]),
        )


def test_newton_rigid_context_uses_public_active_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace()
    stage = SimpleNamespace(model=model, state_0=SimpleNamespace())
    newton_module = ModuleType("isaacsim.physics.newton")
    newton_module.acquire_stage = lambda: stage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.physics", ModuleType("isaacsim.physics"))
    monkeypatch.setitem(sys.modules, "isaacsim.physics.newton", newton_module)

    context = _newton_rigid_tensor_context(SimpleNamespace(_backend=SimpleNamespace()))

    assert context.stage is stage
    assert context.model is model
    assert context.state is stage.state_0


class _FakeWarpArray:
    def __init__(self, value: object, *, device: str = "cpu") -> None:
        self.value = np.asarray(value).copy()
        self.device = device

    def __len__(self) -> int:
        return len(self.value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def numpy(self) -> np.ndarray:
        return self.value.copy()

    def assign(self, value: object) -> None:
        source = value.value if isinstance(value, _FakeWarpArray) else value
        self.value[...] = np.asarray(source)


def _install_fake_newton_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    eval_ik,
) -> None:
    warp = ModuleType("warp")
    warp.float32 = "float32"  # type: ignore[attr-defined]
    warp.int32 = "int32"  # type: ignore[attr-defined]
    warp.from_numpy = (  # type: ignore[attr-defined]
        lambda value, **kwargs: _FakeWarpArray(
            value,
            device=str(kwargs.get("device") or "cpu"),
        )
    )
    warp.clone = lambda value: _FakeWarpArray(  # type: ignore[attr-defined]
        value.value, device=value.device
    )

    def copy(destination, source) -> None:
        destination.value[...] = source.value

    warp.copy = copy  # type: ignore[attr-defined]
    newton = ModuleType("newton")
    newton.eval_ik = eval_ik  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "warp", warp)
    monkeypatch.setitem(sys.modules, "newton", newton)


def _newton_articulated_rigid_fixture(*, corrupt_readback: bool = False):
    transforms = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros((4, 6), dtype=np.float32)
    state = SimpleNamespace(
        body_q=_FakeWarpArray(transforms),
        body_qd=_FakeWarpArray(velocities),
        joint_q=_FakeWarpArray([10.0, 20.0, 30.0, 40.0]),
        joint_qd=_FakeWarpArray([1.0, 2.0, 3.0, 4.0]),
    )
    model = SimpleNamespace(
        device="cpu",
        body_label=["/a/0", "/a/1", "/b/0", "/b/1"],
        joint_child=_FakeWarpArray([0, 1, 2, 3]),
        joint_articulation=_FakeWarpArray([3, 3, 7, 7]),
    )
    stage = SimpleNamespace(state_0=state, model=model)

    class NewtonRigidBodyView:
        def __init__(self) -> None:
            self._newton_stage = stage
            self._model = model

    class TensorApiRigidBodyView:
        count = 4
        prim_paths = ["/a/0", "/a/1", "/b/0", "/b/1"]

        def __init__(self) -> None:
            self._backend = NewtonRigidBodyView()
            self.corrupt_readback = corrupt_readback

        @property
        def transforms(self) -> np.ndarray:
            return state.body_q.value

        @property
        def velocities(self) -> np.ndarray:
            return state.body_qd.value

        def get_transforms(self):
            result = self.transforms.copy()
            if self.corrupt_readback and result[2, 0] != transforms[2, 0]:
                result[2, 0] += 0.25
            return _FakeWarpArray(result)

        def get_velocities(self):
            return _FakeWarpArray(self.velocities)

        def set_transforms(self, data, indices) -> None:
            selected = np.asarray(indices.value, dtype=int)
            self.transforms[selected] = data.value[selected]
            # Newton's setter also writes free-joint coordinates. The atomic
            # restore must still recover the original q if a later step fails.
            state.joint_q.value[0] = -100.0

        def set_velocities(self, data, indices) -> None:
            selected = np.asarray(indices.value, dtype=int)
            self.velocities[selected] = data.value[selected]

    tensor_view = TensorApiRigidBodyView()
    raw = SimpleNamespace(
        paths=tensor_view.prim_paths,
        _physics_rigid_body_view=tensor_view,
    )
    return (
        RigidPrimCoreView(raw, physics_backend="newton"),
        tensor_view,
        state,
        transforms,
        velocities,
    )


def test_newton_articulated_restore_limits_ik_and_preserves_other_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[np.ndarray] = []

    def eval_ik(_model, _state, joint_q, joint_qd, *, indices) -> None:
        calls.append(indices.numpy())
        joint_q.value[2:] = [77.0, 88.0]
        joint_qd.value[2:] = [7.7, 8.8]

    _install_fake_newton_modules(monkeypatch, eval_ik=eval_ik)
    view, tensor_view, state, original_transforms, _ = (
        _newton_articulated_rigid_fixture()
    )
    target_cache = np.asarray([123.0, 456.0])

    view.set_articulated_body_states(
        positions=[[20.0, 1.0, 2.0], [30.0, 3.0, 4.0]],
        orientations=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        velocities=[[1, 2, 3, 0, 0, 0], [0, 0, 0, 10, 11, 12]],
        indices=[2, 3],
    )

    np.testing.assert_array_equal(calls, [[7]])
    np.testing.assert_allclose(state.joint_q.value, [10.0, 20.0, 77.0, 88.0])
    np.testing.assert_allclose(state.joint_qd.value, [1.0, 2.0, 7.7, 8.8])
    np.testing.assert_allclose(tensor_view.transforms[:2], original_transforms[:2])
    np.testing.assert_allclose(tensor_view.transforms[2:, :3], [[20, 1, 2], [30, 3, 4]])
    np.testing.assert_allclose(
        tensor_view.velocities[2:],
        [[1, 2, 3, 0, 0, 0], [0, 0, 0, 10, 11, 12]],
    )
    np.testing.assert_allclose(target_cache, [123.0, 456.0])


def test_newton_articulated_restore_rejects_partial_chain_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[np.ndarray] = []

    def eval_ik(_model, _state, _joint_q, _joint_qd, *, indices) -> None:
        calls.append(indices.numpy())

    _install_fake_newton_modules(monkeypatch, eval_ik=eval_ik)
    view, tensor_view, state, original_transforms, original_velocities = (
        _newton_articulated_rigid_fixture()
    )

    with pytest.raises(RuntimeError, match="complete articulation body coverage"):
        view.set_articulated_body_states(
            positions=[[20.0, 1.0, 2.0]],
            orientations=[[1.0, 0.0, 0.0, 0.0]],
            velocities=[[1, 2, 3, 4, 5, 6]],
            indices=[2],
        )

    assert calls == []
    np.testing.assert_allclose(state.joint_q.value, [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(state.joint_qd.value, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(tensor_view.transforms, original_transforms)
    np.testing.assert_allclose(tensor_view.velocities, original_velocities)


def test_newton_articulated_restore_rolls_back_q_and_bodies_on_readback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def eval_ik(_model, _state, joint_q, joint_qd, *, indices) -> None:
        joint_q.value[:] = -7.0
        joint_qd.value[:] = -8.0

    _install_fake_newton_modules(monkeypatch, eval_ik=eval_ik)
    view, tensor_view, state, original_transforms, original_velocities = (
        _newton_articulated_rigid_fixture(corrupt_readback=True)
    )

    with pytest.raises(RuntimeError, match="failed immediate readback"):
        view.set_articulated_body_states(
            positions=[[20.0, 1.0, 2.0], [30.0, 3.0, 4.0]],
            orientations=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            velocities=[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            indices=[2, 3],
        )

    np.testing.assert_allclose(state.joint_q.value, [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(state.joint_qd.value, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(tensor_view.transforms, original_transforms)
    np.testing.assert_allclose(tensor_view.velocities, original_velocities)


def test_newton_or_environment_selects_experimental_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LINKERBOT_EXPERIMENTAL_CORE", raising=False)
    assert use_experimental_core(physics_backend="newton") is True
    assert use_experimental_core(physics_backend="physx") is False
    monkeypatch.setenv("LINKERBOT_EXPERIMENTAL_CORE", "yes")
    assert use_experimental_core(physics_backend="physx") is True
    monkeypatch.setenv("LINKERBOT_EXPERIMENTAL_CORE", "sometimes")
    with pytest.raises(ValueError, match="must be a boolean value"):
        use_experimental_core(physics_backend="physx")


def test_core_factories_preserve_legacy_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LINKERBOT_EXPERIMENTAL_CORE", raising=False)
    prims = ModuleType("isaacsim.core.prims")

    class Articulation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class RigidPrim:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.initialized = False

        def initialize(self):
            self.initialized = True

    prims.Articulation = Articulation  # type: ignore[attr-defined]
    prims.RigidPrim = RigidPrim  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim", ModuleType("isaacsim"))
    monkeypatch.setitem(sys.modules, "isaacsim.core", ModuleType("isaacsim.core"))
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", prims)
    scene = SimpleNamespace(add=lambda view: view)

    articulation = create_articulation_core_view(
        paths=["/World/Robot"],
        name="robot",
        world_scene=scene,
        physics_backend="physx",
    )
    rigid = create_rigid_prim_core_view(
        paths=["/World/Object"], name="object", physics_backend="physx"
    )

    assert isinstance(articulation, Articulation)
    assert articulation.kwargs["prim_paths_expr"] == ["/World/Robot"]
    assert isinstance(rigid, RigidPrim)
    assert rigid.initialized is True


def test_single_factory_requires_installed_project_newton_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "isaacsim", raising=False)
    monkeypatch.delitem(sys.modules, "isaacsim.core.prims", raising=False)

    with pytest.raises(RuntimeError, match="no physics manager is active"):
        create_single_articulation_core_view(
            prim_path="/World/Robot",
            name="robot",
            physics_backend="newton",
        )

    assert "isaacsim.core.prims" not in sys.modules
