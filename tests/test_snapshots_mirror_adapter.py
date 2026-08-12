"""Mirror CPU/NumPy SceneSnapshot adapter 合同。"""

from __future__ import annotations

from copy import deepcopy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.controllers.joint_controller import JointController
from linkerbot_sim.controllers.types import (
    ComponentControlSettings,
    ControlTargets,
    JointControlSettings,
)
from linkerbot_sim.objects.state_views import SceneObjectStateView
from linkerbot_sim.robots.classification import RobotComponentMapping
from linkerbot_sim.snapshots.mirror_adapter import (
    CONTROL_MODE_INFO_KEY,
    CONTROLLER_PROFILE_FINGERPRINTS_INFO_KEY,
    NEWTON_SOLVER_STATE_INFO_KEY,
    get_mirror_snapshot,
    set_mirror_snapshot,
)
from linkerbot_sim.snapshots.runtime_objects import COMMAND_TARGET_MODES_INFO_KEY
from linkerbot_sim.snapshots.schema import ObjectSnapshot, SceneSnapshot
from linkerbot_sim.snapshots.transactions import (
    RuntimeMutationRejected,
    SnapshotRollbackError,
)


class _Articulation:
    dof_names = ("fixed", "j0", "j1")

    def __init__(self) -> None:
        self.positions = np.asarray([9.0, 0.1, 0.2], dtype=float)
        self.velocities = np.asarray([8.0, 0.3, 0.4], dtype=float)
        self.position_targets = np.asarray([7.0, 0.5, 0.6], dtype=float)

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def get_joint_position_targets(self):
        return self.position_targets.copy()

    def set_joint_positions(self, values):
        self.positions = np.asarray(values, dtype=float).copy()

    def set_joint_velocities(self, values):
        self.velocities = np.asarray(values, dtype=float).copy()

    def set_joint_position_targets(self, values, *, joint_indices):
        self.position_targets[np.asarray(joint_indices, dtype=int)] = np.asarray(
            values, dtype=float
        )


class _LegacyAction:
    def __init__(self, *, joint_positions, joint_indices) -> None:
        self.joint_positions = np.asarray(joint_positions, dtype=float).copy()
        self.joint_indices = np.asarray(joint_indices, dtype=int).copy()


class _LegacyArticulation:
    """只暴露 Isaac legacy SingleArticulation 的 target action API。"""

    dof_names = ("fixed", "j0", "j1")

    def __init__(self) -> None:
        self.positions = np.asarray([9.0, 0.1, 0.2], dtype=float)
        self.velocities = np.asarray([8.0, 0.3, 0.4], dtype=float)
        self.position_targets = np.asarray([7.0, 0.5, 0.6], dtype=float)
        self.applied_actions: list[_LegacyAction] = []

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def set_joint_positions(self, values):
        self.positions = np.asarray(values, dtype=float).copy()

    def set_joint_velocities(self, values):
        self.velocities = np.asarray(values, dtype=float).copy()

    def get_applied_action(self):
        return SimpleNamespace(joint_positions=self.position_targets.copy())

    def apply_action(self, action: _LegacyAction) -> None:
        self.applied_actions.append(action)
        self.position_targets[action.joint_indices] = action.joint_positions


class _ModeAction:
    def __init__(
        self,
        *,
        joint_positions=None,
        joint_velocities=None,
        joint_efforts=None,
        joint_indices=None,
    ) -> None:
        self.joint_positions = _optional_array(joint_positions)
        self.joint_velocities = _optional_array(joint_velocities)
        self.joint_efforts = _optional_array(joint_efforts)
        self.joint_indices = np.asarray(joint_indices, dtype=int).reshape(-1)


def _optional_array(values):
    return None if values is None else np.asarray(values, dtype=float).reshape(-1)


class _ModeIsaacController:
    def __init__(self, dof_count: int) -> None:
        self.kps = np.zeros(dof_count, dtype=float)
        self.kds = np.zeros(dof_count, dtype=float)
        self.max_efforts = np.zeros(dof_count, dtype=float)
        self.mode_switches: list[tuple[int, str]] = []

    def get_gains(self):
        return self.kps.copy(), self.kds.copy()

    def set_gains(self, *, kps, kds) -> None:
        self.kps = np.asarray(kps, dtype=float).copy()
        self.kds = np.asarray(kds, dtype=float).copy()

    def set_max_efforts(self, values, joint_indices=None) -> None:
        self.max_efforts = np.asarray(values, dtype=float).copy()

    def set_effort_modes(self, mode, joint_indices=None) -> None:
        return None

    def switch_dof_control_mode(self, *, dof_index: int, mode: str) -> None:
        self.mode_switches.append((int(dof_index), str(mode)))


class _ModeArticulation:
    dof_names = ("p", "v", "e")
    num_dof = 3

    def __init__(self) -> None:
        self.positions = np.asarray([0.1, 0.2, 0.3], dtype=float)
        self.velocities = np.asarray([0.01, 0.02, 0.03], dtype=float)
        self.position_targets = np.asarray([0.4, 0.5, 0.6], dtype=float)
        self.velocity_targets = np.asarray([0.04, 0.05, 0.06], dtype=float)
        self.applied_efforts = np.asarray([0.7, 0.8, 0.9], dtype=float)
        self.controller = _ModeIsaacController(self.num_dof)
        self.actions: list[_ModeAction] = []
        self.position_state_writes = 0
        self.velocity_state_writes = 0

    def get_articulation_controller(self):
        return self.controller

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def get_joint_position_targets(self):
        return self.position_targets.copy()

    def get_joint_velocity_targets(self):
        return self.velocity_targets.copy()

    def get_applied_joint_efforts(self):
        return self.applied_efforts.copy()

    def set_joint_positions(self, values):
        self.position_state_writes += 1
        self.positions = np.asarray(values, dtype=float).copy()

    def set_joint_velocities(self, values):
        self.velocity_state_writes += 1
        self.velocities = np.asarray(values, dtype=float).copy()

    def apply_action(self, action: _ModeAction) -> None:
        self.actions.append(action)
        indices = action.joint_indices
        if action.joint_positions is not None:
            self.position_targets[indices] = action.joint_positions
        if action.joint_velocities is not None:
            self.velocity_targets[indices] = action.joint_velocities
        if action.joint_efforts is not None:
            self.applied_efforts[indices] = action.joint_efforts


def _mode_aware_runtime(
    *,
    prime_cache: bool,
    fail_collision_invalidation: bool = False,
):
    articulation = _ModeArticulation()
    mapping = RobotComponentMapping.from_profile(
        {
            "joint_groups": {
                "arm": ["p"],
                "hand": ["v"],
                "passive": ["e"],
            }
        }
    )
    controller = JointController(
        articulation,
        joint_names=["all"],
        settings=JointControlSettings(
            arm=ComponentControlSettings(
                mode="position",
                method="implicit",
                max_force=10.0,
            ),
            hand=ComponentControlSettings(
                mode="velocity",
                method="implicit",
                max_force=10.0,
            ),
            default=ComponentControlSettings(
                mode="effort",
                method="direct",
                max_force=10.0,
            ),
        ),
        component_mapping=mapping,
    )
    controller.configure_runtime()
    if prime_cache:
        controller.apply_targets(
            _ModeAction,
            ControlTargets(
                positions=np.asarray([1.1, 1.2, 1.3]),
                velocities=np.asarray([2.1, 2.2, 2.3]),
                efforts=np.asarray([3.1, 3.2, 3.3]),
            ),
        )
        articulation.actions.clear()
    observer = _ResetObserver()
    execution = SimpleNamespace(
        articulation=articulation,
        articulation_action_type=_ModeAction,
        joint_controller=controller,
        state_observer=observer,
        camera_observer=None,
    )
    robot = SimpleNamespace(
        label="mixed",
        profile_name=None,
        imported=None,
        execution=execution,
    )

    def mark_dirty() -> None:
        if fail_collision_invalidation:
            raise RuntimeError("collision invalidation failed")

    runtime = SimpleNamespace(
        robots_by_id={0: robot},
        robot_id_by_label={"mixed": 0},
        robot_registry=object(),
        session=SimpleNamespace(stage=None),
        object_handles=(),
        object_state_views={},
        collision_registry=SimpleNamespace(mark_dirty=mark_dirty),
        quit_event=threading.Event(),
        robot_by_label=lambda label: robot if label == "mixed" else None,
    )
    return runtime, articulation, controller


class _ResetObserver:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _FailingArticulation(_Articulation):
    def __init__(self, *, offset: float, fail_position_calls: set[int]) -> None:
        super().__init__()
        self.positions += float(offset)
        self.velocities += float(offset)
        self.position_calls = 0
        self.fail_position_calls = set(fail_position_calls)

    def set_joint_positions(self, values):
        self.position_calls += 1
        if self.position_calls in self.fail_position_calls:
            raise RuntimeError(f"position setter {self.position_calls} failed")
        super().set_joint_positions(values)


class _AtomicSceneBodyView:
    def __init__(self) -> None:
        self.positions = np.asarray([[0.1, 0.0, 0.0], [0.4, 0.2, 0.0]])
        self.orientations = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self.velocities = np.asarray(
            [[0.1, 0.2, 0.3, 1.1, 1.2, 1.3], [0.4, 0.5, 0.6, 1.4, 1.5, 1.6]]
        )
        self.write_count = 0

    def get_world_poses(self, *, indices):
        selected = np.asarray(indices, dtype=int)
        return self.positions[selected].copy(), self.orientations[selected].copy()

    def get_velocities(self, *, indices):
        return self.velocities[np.asarray(indices, dtype=int)].copy()

    def set_articulated_body_states(
        self,
        *,
        positions,
        orientations,
        velocities,
        indices,
    ) -> None:
        selected = np.asarray(indices, dtype=int)
        self.positions[selected] = np.asarray(positions, dtype=float)
        self.orientations[selected] = np.asarray(orientations, dtype=float)
        self.velocities[selected] = np.asarray(velocities, dtype=float)
        self.write_count += 1


class _GeneralizedSceneBodyView(_AtomicSceneBodyView):
    q_coordinate_names = ("@root/a|translation.x", "joint|q[0]")
    qd_coordinate_names = ("@root/a|linear.x",)
    generalized_coordinate_signature = ("newton-generalized-state-v1", "topology")
    generalized_world_translation_q_indices = (0,)

    def __init__(self) -> None:
        super().__init__()
        self.q = np.asarray([[0.1, 0.2]], dtype=float)
        self.qd = np.asarray([[0.3]], dtype=float)
        self.generalized_write_count = 0
        self.validation_count = 0

    def get_generalized_state(self, *, indices):
        selected = np.asarray(indices, dtype=int)
        return self.q[selected].copy(), self.qd[selected].copy()

    def validate_generalized_state(
        self,
        *,
        signature,
        q_names,
        qd_names,
        q,
        qd,
        indices,
    ) -> None:
        self.validation_count += 1
        if tuple(signature) != self.generalized_coordinate_signature:
            raise ValueError("generalized coordinate signature mismatch")
        if tuple(q_names) != self.q_coordinate_names:
            raise ValueError("generalized q coordinate names mismatch")
        if tuple(qd_names) != self.qd_coordinate_names:
            raise ValueError("generalized qd coordinate names mismatch")
        selected = np.asarray(indices, dtype=int).reshape(-1)
        q_array = np.asarray(q, dtype=float)
        qd_array = np.asarray(qd, dtype=float)
        if q_array.shape != (selected.size, 2) or qd_array.shape != (
            selected.size,
            1,
        ):
            raise ValueError("generalized state shape mismatch")
        if not np.all(np.isfinite(q_array)) or not np.all(np.isfinite(qd_array)):
            raise ValueError("generalized state must be finite")

    def set_generalized_state(self, **kwargs) -> None:
        self.validate_generalized_state(**kwargs)
        selected = np.asarray(kwargs["indices"], dtype=int)
        self.q[selected] = np.asarray(kwargs["q"], dtype=float)
        self.qd[selected] = np.asarray(kwargs["qd"], dtype=float)
        self.generalized_write_count += 1


def _partial_chain_runtime(*, fail_collision_invalidation: bool):
    body_view = _AtomicSceneBodyView()
    state_view = SceneObjectStateView(
        body_view=body_view,
        body_names=("a", "b"),
        reference_body="a",
    )
    handle = SimpleNamespace(
        runtime_handle="rope",
        kind="dynamic_chain",
        config=SimpleNamespace(object_profile="rope"),
        model={"prim_path": "/rope", "bodies": ("/rope/a", "/rope/b")},
    )

    def mark_dirty() -> None:
        if fail_collision_invalidation:
            raise RuntimeError("collision invalidation failed")

    runtime = SimpleNamespace(
        robots_by_id={},
        robot_id_by_label={},
        robot_registry=object(),
        session=SimpleNamespace(stage=object()),
        object_handles=(handle,),
        object_state_views={"rope": state_view},
        collision_registry=SimpleNamespace(mark_dirty=mark_dirty),
        quit_event=threading.Event(),
        robot_by_label=lambda label: pytest.fail(f"unexpected robot lookup: {label}"),
    )
    snapshot = SceneSnapshot(
        robots={},
        objects={
            "rope": ObjectSnapshot(
                name="rope",
                object_profile="rope",
                positions_local=np.asarray([8.0, 8.1, 8.2]),
                orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
                body_names=("b",),
                body_positions_local=np.asarray([[8.0, 8.1, 8.2]]),
                body_orientations_wxyz=np.asarray([[0.0, 1.0, 0.0, 0.0]]),
                body_linear_velocities=np.asarray([[2.0, 3.0, 4.0]]),
                body_angular_velocities=np.asarray([[5.0, 6.0, 7.0]]),
            )
        },
    )
    return runtime, body_view, snapshot


def _transaction_mirror_runtime(
    articulations: dict[str, _FailingArticulation],
) -> SimpleNamespace:
    robots = {}
    for robot_id, (label, articulation) in enumerate(articulations.items()):
        controller = SimpleNamespace(
            command_indices=np.asarray([1, 2], dtype=int),
            command_joint_names=("j0", "j1"),
            last_commanded_efforts=np.asarray(
                [robot_id + 0.1, robot_id + 0.2, robot_id + 0.3],
                dtype=float,
            ),
        )
        robots[robot_id] = SimpleNamespace(
            label=label,
            profile_name=None,
            imported=None,
            execution=SimpleNamespace(
                articulation=articulation,
                articulation_action_type=_LegacyAction,
                joint_controller=controller,
                state_observer=_ResetObserver(),
                camera_observer=None,
            ),
        )
    by_label = {robot.label: robot for robot in robots.values()}
    collision_registry = SimpleNamespace(mark_dirty=lambda: None)
    return SimpleNamespace(
        robots_by_id=robots,
        robot_id_by_label={label: index for index, label in enumerate(by_label)},
        robot_registry=object(),
        session=SimpleNamespace(stage=None),
        object_handles=(),
        object_state_views={},
        collision_registry=collision_registry,
        quit_event=threading.Event(),
        robot_by_label=by_label.__getitem__,
    )


def _changed_mirror_snapshot(runtime: object) -> dict[str, object]:
    payload = get_mirror_snapshot(runtime).as_dict()
    for index, robot in enumerate(payload["robots"]):
        robot["joint_positions"] = [10.0 + index, 20.0 + index]
        robot["joint_velocities"] = [30.0 + index, 40.0 + index]
        robot["command_targets"] = [50.0 + index, 60.0 + index]
    return payload


def _attach_generalized_chain(runtime: object) -> _GeneralizedSceneBodyView:
    body_view = _GeneralizedSceneBodyView()
    runtime.session.stage = object()
    runtime.object_handles = (
        SimpleNamespace(
            runtime_handle="rope",
            kind="dynamic_chain",
            config=SimpleNamespace(object_profile="rope", prim_path="/rope"),
            model={"prim_path": "/rope", "bodies": ("/rope/a", "/rope/b")},
        ),
    )
    runtime.object_state_views = {
        "rope": SceneObjectStateView(
            body_view=body_view,
            body_names=("a", "b"),
            reference_body="a",
        )
    }
    return body_view


def test_single_generalized_signature_preflight_precedes_robot_mutation() -> None:
    articulation = _FailingArticulation(offset=0.0, fail_position_calls=set())
    runtime = _transaction_mirror_runtime({"arm": articulation})
    body_view = _attach_generalized_chain(runtime)
    payload = get_mirror_snapshot(runtime).as_dict()
    payload["robots"][0]["joint_positions"] = [8.0, 9.0]
    payload["objects"]["rope"]["generalized_signature"] = ["wrong-abi"]
    before = articulation.positions.copy()

    with pytest.raises(ValueError, match="coordinate signature mismatch"):
        set_mirror_snapshot(runtime, payload)

    np.testing.assert_array_equal(articulation.positions, before)
    assert articulation.position_calls == 0
    assert body_view.generalized_write_count == 0
    assert not runtime.quit_event.is_set()


def test_single_partial_body_mapping_does_not_use_generalized_fast_path() -> None:
    runtime, _body_view, snapshot = _partial_chain_runtime(
        fail_collision_invalidation=False
    )
    body_view = _attach_generalized_chain(runtime)
    source = snapshot.objects["rope"]
    partial = ObjectSnapshot(
        name=source.name,
        object_profile=source.object_profile,
        positions_local=source.positions_local,
        orientations_wxyz=source.orientations_wxyz,
        body_names=source.body_names,
        body_positions_local=source.body_positions_local,
        body_orientations_wxyz=source.body_orientations_wxyz,
        body_linear_velocities=source.body_linear_velocities,
        body_angular_velocities=source.body_angular_velocities,
        generalized_signature=body_view.generalized_coordinate_signature,
        generalized_q_names=body_view.q_coordinate_names,
        generalized_qd_names=body_view.qd_coordinate_names,
        generalized_q=np.asarray([4.0, 5.0]),
        generalized_qd=np.asarray([6.0]),
    )

    result = set_mirror_snapshot(
        runtime,
        SceneSnapshot(robots={}, objects={"rope": partial}),
        strict=False,
    )

    assert result.partial is True
    assert body_view.generalized_write_count == 0
    assert body_view.write_count == 1


def test_mirror_uses_body_fallback_for_replicated_generalized_world_origin() -> None:
    runtime, _body_view, _snapshot = _partial_chain_runtime(
        fail_collision_invalidation=False
    )
    body_view = _attach_generalized_chain(runtime)
    source = get_mirror_snapshot(runtime).objects["rope"]
    replicated_source = ObjectSnapshot(
        **{
            **source.__dict__,
            "generalized_world_origin": np.asarray([2.0, 0.0, 0.0]),
        }
    )

    result = set_mirror_snapshot(
        runtime,
        SceneSnapshot(robots={}, objects={"rope": replicated_source}),
    )

    assert result.accepted
    assert body_view.generalized_write_count == 0
    assert body_view.write_count == 1


def test_mirror_snapshot_dispatch_restores_only_command_joints_and_caches() -> None:
    articulation = _Articulation()
    observer = _ResetObserver()
    controller = SimpleNamespace(
        command_indices=np.asarray([1, 2], dtype=int),
        command_joint_names=("j0", "j1"),
        last_commanded_efforts=np.zeros(3, dtype=float),
    )
    execution = SimpleNamespace(
        articulation=articulation,
        joint_controller=controller,
        state_observer=observer,
        camera_observer=None,
    )
    robot = SimpleNamespace(
        label="arm",
        profile_name=None,
        imported=None,
        execution=execution,
    )
    collision_registry = SimpleNamespace(mark_dirty_calls=0)

    def mark_dirty() -> None:
        collision_registry.mark_dirty_calls += 1

    collision_registry.mark_dirty = mark_dirty
    runtime = SimpleNamespace(
        robots_by_id={0: robot},
        robot_id_by_label={"arm": 0},
        robot_registry=object(),
        session=SimpleNamespace(stage=None),
        object_handles=(),
        config_fingerprint="scene-config",
        collision_registry=collision_registry,
        robot_by_label=lambda label: robot if label == "arm" else None,
    )

    snapshot = get_mirror_snapshot(runtime)

    assert snapshot.metadata.source_runtime == "mirror"
    np.testing.assert_allclose(snapshot.robots["arm"].joint_positions, [0.1, 0.2])
    np.testing.assert_allclose(snapshot.robots["arm"].joint_velocities, [0.3, 0.4])
    np.testing.assert_allclose(snapshot.robots["arm"].command_targets, [0.5, 0.6])
    articulation.positions[:] = [7.0, 1.0, 2.0]
    articulation.velocities[:] = [6.0, 3.0, 4.0]
    articulation.position_targets[:] = [5.0, 3.0, 4.0]

    result = set_mirror_snapshot(runtime, snapshot)

    assert result.accepted is True
    assert result.robots == ("arm",)
    np.testing.assert_allclose(articulation.positions, [7.0, 0.1, 0.2])
    np.testing.assert_allclose(articulation.velocities, [6.0, 0.3, 0.4])
    np.testing.assert_allclose(articulation.position_targets, [5.0, 0.5, 0.6])
    assert np.isnan(controller.last_commanded_efforts).all()
    assert observer.reset_calls == 1
    assert collision_registry.mark_dirty_calls == 1


def test_mirror_mixed_mode_snapshot_captures_and_restores_target_semantics() -> None:
    runtime, articulation, controller = _mode_aware_runtime(prime_cache=True)

    snapshot = get_mirror_snapshot(runtime)

    np.testing.assert_allclose(
        snapshot.robots["mixed"].command_targets,
        [1.1, 2.2, 3.3],
    )
    assert snapshot.metadata.info[COMMAND_TARGET_MODES_INFO_KEY] == {
        "mixed": {"p": "position", "v": "velocity", "e": "effort"}
    }
    payload = snapshot.as_dict()
    payload["robots"][0]["joint_positions"] = [4.1, 4.2, 4.3]
    payload["robots"][0]["joint_velocities"] = [5.1, 5.2, 5.3]
    payload["robots"][0]["command_targets"] = [6.1, 6.2, 6.3]

    result = set_mirror_snapshot(runtime, payload)

    assert result.accepted is True
    np.testing.assert_allclose(articulation.positions, [4.1, 4.2, 4.3])
    np.testing.assert_allclose(articulation.velocities, [5.1, 5.2, 5.3])
    np.testing.assert_allclose(articulation.position_targets[0], 6.1)
    np.testing.assert_allclose(articulation.velocity_targets[1], 6.2)
    np.testing.assert_allclose(articulation.applied_efforts[2], 6.3)
    restored_cache = controller.last_control_targets
    assert restored_cache is not None
    np.testing.assert_allclose(restored_cache.positions[0], 6.1)
    np.testing.assert_allclose(restored_cache.velocities[1], 6.2)
    np.testing.assert_allclose(restored_cache.efforts[2], 6.3)


def test_mirror_mode_aware_capture_falls_back_to_runtime_targets() -> None:
    runtime, _articulation, _controller = _mode_aware_runtime(prime_cache=False)

    snapshot = get_mirror_snapshot(runtime)

    # position 来自 position target，velocity 来自 velocity target，effort 来自 applied effort。
    np.testing.assert_allclose(
        snapshot.robots["mixed"].command_targets,
        [0.4, 0.05, 0.9],
    )


def test_mirror_snapshot_control_mode_rejects_cross_mode_before_write() -> None:
    articulation = _FailingArticulation(offset=0.0, fail_position_calls=set())
    runtime = _transaction_mirror_runtime({"arm": articulation})
    runtime.control_mode_state_provider = lambda: SimpleNamespace(
        active_mode="velocity",
        generation=3,
    )
    payload = get_mirror_snapshot(runtime).as_dict()

    assert payload["metadata"]["info"][CONTROL_MODE_INFO_KEY] == {
        "active_mode": "velocity",
        "generation": 3,
    }
    runtime.control_mode_state_provider = lambda: SimpleNamespace(
        active_mode="effort",
        generation=9,
    )

    with pytest.raises(ValueError, match="control mode mismatch"):
        set_mirror_snapshot(runtime, payload)

    assert articulation.position_calls == 0


def test_mirror_snapshot_generation_is_diagnostic_not_restored() -> None:
    articulation = _FailingArticulation(offset=0.0, fail_position_calls=set())
    runtime = _transaction_mirror_runtime({"arm": articulation})
    runtime.control_mode_state_provider = lambda: SimpleNamespace(
        active_mode="position",
        generation=2,
    )
    payload = get_mirror_snapshot(runtime).as_dict()
    runtime.control_mode_state_provider = lambda: SimpleNamespace(
        active_mode="position",
        generation=7,
    )

    assert set_mirror_snapshot(runtime, payload).accepted is True
    assert articulation.position_calls == 1


def test_mirror_snapshot_controller_profile_fingerprint_preflights_restore() -> None:
    articulation = _FailingArticulation(offset=0.0, fail_position_calls=set())
    runtime = _transaction_mirror_runtime({"arm": articulation})
    robot = runtime.robot_by_label("arm")
    robot.controller_profile_fingerprint = "profile-a"
    payload = get_mirror_snapshot(runtime).as_dict()

    assert payload["metadata"]["info"][CONTROLLER_PROFILE_FINGERPRINTS_INFO_KEY] == {
        "arm": "profile-a"
    }
    robot.controller_profile_fingerprint = "profile-b"

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        set_mirror_snapshot(runtime, payload)

    assert articulation.position_calls == 0


@pytest.mark.parametrize("metadata_change", ["missing", "mismatch"])
def test_mirror_mode_metadata_rejects_before_first_state_write(
    metadata_change: str,
) -> None:
    runtime, articulation, _controller = _mode_aware_runtime(prime_cache=True)
    payload = get_mirror_snapshot(runtime).as_dict()
    modes = payload["metadata"]["info"][COMMAND_TARGET_MODES_INFO_KEY]
    if metadata_change == "missing":
        payload["metadata"]["info"].pop(COMMAND_TARGET_MODES_INFO_KEY)
        match = "missing"
    else:
        modes["mixed"]["v"] = "position"
        match = "mode mismatch"

    with pytest.raises(ValueError, match=match):
        set_mirror_snapshot(runtime, payload)

    assert articulation.position_state_writes == 0
    assert articulation.velocity_state_writes == 0


def test_mirror_mixed_mode_rollback_restores_exact_controller_caches() -> None:
    runtime, articulation, controller = _mode_aware_runtime(
        prime_cache=True,
        fail_collision_invalidation=True,
    )
    payload = get_mirror_snapshot(runtime).as_dict()
    payload["robots"][0]["joint_positions"] = [4.1, 4.2, 4.3]
    payload["robots"][0]["joint_velocities"] = [5.1, 5.2, 5.3]
    payload["robots"][0]["command_targets"] = [6.1, 6.2, 6.3]
    original_positions = articulation.positions.copy()
    original_velocities = articulation.velocities.copy()
    original_position_targets = articulation.position_targets.copy()
    original_velocity_targets = articulation.velocity_targets.copy()
    original_efforts = articulation.applied_efforts.copy()
    original_controller_efforts = controller.last_commanded_efforts.copy()
    original_cache = controller.snapshot_control_targets_cache()

    with pytest.raises(RuntimeError, match="collision invalidation failed"):
        set_mirror_snapshot(runtime, payload)

    np.testing.assert_allclose(articulation.positions, original_positions)
    np.testing.assert_allclose(articulation.velocities, original_velocities)
    np.testing.assert_allclose(articulation.position_targets, original_position_targets)
    np.testing.assert_allclose(articulation.velocity_targets, original_velocity_targets)
    np.testing.assert_allclose(articulation.applied_efforts, original_efforts)
    np.testing.assert_allclose(
        controller.last_commanded_efforts,
        original_controller_efforts,
        equal_nan=True,
    )
    assert original_cache is not None
    restored_cache = controller.last_control_targets
    assert restored_cache is not None
    np.testing.assert_allclose(restored_cache.positions, original_cache.positions)
    np.testing.assert_allclose(restored_cache.velocities, original_cache.velocities)
    np.testing.assert_allclose(restored_cache.efforts, original_cache.efforts)


def test_mirror_snapshot_uses_legacy_applied_action_target_api() -> None:
    articulation = _LegacyArticulation()
    runtime = _transaction_mirror_runtime({"arm": articulation})
    payload = get_mirror_snapshot(runtime).as_dict()
    payload["robots"][0]["joint_positions"] = [1.1, 1.2]
    payload["robots"][0]["joint_velocities"] = [2.1, 2.2]
    payload["robots"][0]["command_targets"] = [3.1, 3.2]

    result = set_mirror_snapshot(runtime, payload)

    assert result.accepted is True
    np.testing.assert_allclose(articulation.positions, [9.0, 1.1, 1.2])
    np.testing.assert_allclose(articulation.velocities, [8.0, 2.1, 2.2])
    np.testing.assert_allclose(articulation.position_targets, [7.0, 3.1, 3.2])
    assert len(articulation.applied_actions) == 1
    np.testing.assert_array_equal(articulation.applied_actions[0].joint_indices, [1, 2])


def test_mirror_snapshot_without_targets_holds_restored_positions() -> None:
    articulation = _Articulation()
    runtime = _transaction_mirror_runtime({"arm": articulation})
    payload = get_mirror_snapshot(runtime).as_dict()
    payload["robots"][0]["joint_positions"] = [1.1, 1.2]
    payload["robots"][0].pop("command_targets")
    payload["metadata"]["info"].pop(COMMAND_TARGET_MODES_INFO_KEY)

    result = set_mirror_snapshot(runtime, payload)

    assert result.accepted is True
    np.testing.assert_allclose(articulation.position_targets, [7.0, 1.1, 1.2])


def test_mirror_public_non_strict_partial_chain_preserves_unmapped_body() -> None:
    runtime, body_view, snapshot = _partial_chain_runtime(
        fail_collision_invalidation=False
    )
    original_a = (
        body_view.positions[0].copy(),
        body_view.orientations[0].copy(),
        body_view.velocities[0].copy(),
    )

    result = set_mirror_snapshot(runtime, snapshot, strict=False)

    assert result.accepted is True
    assert result.partial is True
    np.testing.assert_allclose(body_view.positions[0], original_a[0])
    np.testing.assert_allclose(body_view.orientations[0], original_a[1])
    np.testing.assert_allclose(body_view.velocities[0], original_a[2])
    np.testing.assert_allclose(body_view.positions[1], [8.0, 8.1, 8.2])
    np.testing.assert_allclose(body_view.velocities[1], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_mirror_partial_chain_rolls_back_after_later_runtime_failure() -> None:
    runtime, body_view, snapshot = _partial_chain_runtime(
        fail_collision_invalidation=True
    )
    original_positions = body_view.positions.copy()
    original_orientations = body_view.orientations.copy()
    original_velocities = body_view.velocities.copy()

    with pytest.raises(RuntimeError, match="collision invalidation failed"):
        set_mirror_snapshot(runtime, snapshot, strict=False)

    np.testing.assert_allclose(body_view.positions, original_positions)
    np.testing.assert_allclose(body_view.orientations, original_orientations)
    np.testing.assert_allclose(body_view.velocities, original_velocities)
    assert body_view.write_count == 2
    assert runtime.quit_event.is_set()
    assert "collision invalidation failed" in runtime.fatal_error


def test_mirror_snapshot_rolls_back_all_robots_and_runtime_can_continue() -> None:
    articulations = {
        "left": _FailingArticulation(offset=0.0, fail_position_calls=set()),
        "right": _FailingArticulation(offset=1.0, fail_position_calls={1}),
    }
    runtime = _transaction_mirror_runtime(articulations)
    payload = _changed_mirror_snapshot(runtime)
    original_positions = {
        name: articulation.positions.copy()
        for name, articulation in articulations.items()
    }
    original_targets = {
        name: articulation.position_targets.copy()
        for name, articulation in articulations.items()
    }
    original_efforts = {
        robot.label: robot.execution.joint_controller.last_commanded_efforts.copy()
        for robot in runtime.robots_by_id.values()
    }

    with pytest.raises(RuntimeError, match="position setter 1 failed"):
        set_mirror_snapshot(runtime, payload)

    for name, articulation in articulations.items():
        np.testing.assert_allclose(articulation.positions, original_positions[name])
        np.testing.assert_allclose(
            articulation.position_targets, original_targets[name]
        )
        np.testing.assert_allclose(
            runtime.robot_by_label(
                name
            ).execution.joint_controller.last_commanded_efforts,
            original_efforts[name],
        )
    assert not hasattr(runtime, "fatal_error")
    assert not runtime.quit_event.is_set()

    result = set_mirror_snapshot(runtime, payload)
    assert result.accepted
    np.testing.assert_allclose(articulations["left"].positions[1:], [10.0, 20.0])
    np.testing.assert_allclose(articulations["right"].positions[1:], [11.0, 21.0])
    np.testing.assert_allclose(articulations["left"].position_targets[1:], [50.0, 60.0])
    np.testing.assert_allclose(
        articulations["right"].position_targets[1:], [51.0, 61.0]
    )


def test_mirror_snapshot_rollback_failure_fail_stops_future_mutations() -> None:
    articulations = {
        "left": _FailingArticulation(offset=0.0, fail_position_calls={2}),
        "right": _FailingArticulation(offset=1.0, fail_position_calls={1}),
    }
    runtime = _transaction_mirror_runtime(articulations)
    payload = _changed_mirror_snapshot(runtime)

    with pytest.raises(SnapshotRollbackError) as exc_info:
        set_mirror_snapshot(runtime, payload)

    assert isinstance(exc_info.value.cause, RuntimeError)
    assert "position setter 1 failed" in str(exc_info.value.cause)
    assert runtime.quit_event.is_set()
    assert "rollback_errors" in runtime.fatal_error
    calls = {name: item.position_calls for name, item in articulations.items()}

    with pytest.raises(RuntimeMutationRejected, match="requires rebuild"):
        set_mirror_snapshot(runtime, payload)
    assert {name: item.position_calls for name, item in articulations.items()} == calls


class _NewtonSolverStateRuntime:
    """只模拟 Mirror adapter 所需的 Newton 冷快照合同。"""

    def __init__(
        self,
        values: list[list[float]],
        *,
        baseline_values: list[list[float]] | None = None,
    ) -> None:
        self.payload = {
            "schema": "linkerbot.newton-solver-integration-state.v1",
            "source_execution": "cpu",
            "world_count": 1,
            "state_signature": 41,
            "state_width": len(values[0]),
            "simulation_time_s": float(values[0][0]),
            "values": deepcopy(values),
        }
        baseline = values if baseline_values is None else baseline_values
        self.baseline_payload = {
            **deepcopy(self.payload),
            "simulation_time_s": float(baseline[0][0]),
            "values": deepcopy(baseline),
        }
        self.validate_calls = 0
        self.set_values: list[list[list[float]]] = []
        self.reset_calls = 0

    def capture_solver_integration_state_host(self) -> dict[str, object]:
        return deepcopy(self.payload)

    def validate_solver_integration_state_host(self, payload) -> None:
        self.validate_calls += 1
        if payload.get("state_width") != len(payload.get("values", [[]])[0]):
            raise ValueError("solver state width mismatch")

    def set_solver_integration_state_host(self, payload) -> None:
        self.validate_solver_integration_state_host(payload)
        self.payload = deepcopy(dict(payload))
        self.set_values.append(deepcopy(self.payload["values"]))

    def reset_solver_integration_state_host(self) -> None:
        self.payload = deepcopy(self.baseline_payload)
        self.reset_calls += 1


def test_mirror_newton_snapshot_restores_solver_persistent_state() -> None:
    runtime = _transaction_mirror_runtime(
        {"arm": _FailingArticulation(offset=0.0, fail_position_calls=set())}
    )
    physics = _NewtonSolverStateRuntime([[0.25, 1.0, 2.0]])
    runtime.session.physics_runtime = physics
    snapshot = get_mirror_snapshot(runtime)

    assert snapshot.metadata.info[NEWTON_SOLVER_STATE_INFO_KEY]["values"] == [
        [0.25, 1.0, 2.0]
    ]
    physics.payload["values"] = [[1.5, 8.0, 9.0]]
    physics.payload["simulation_time_s"] = 1.5

    result = set_mirror_snapshot(runtime, snapshot)

    assert result.accepted
    assert physics.payload["values"] == [[0.25, 1.0, 2.0]]
    assert physics.payload["simulation_time_s"] == 0.25


def test_mirror_physx_snapshot_resets_newton_solver_to_committed_baseline() -> None:
    articulation = _FailingArticulation(offset=0.0, fail_position_calls=set())
    runtime = _transaction_mirror_runtime({"arm": articulation})
    # 在没有 Newton runtime 时取得的快照等价于 PhysX-origin：它只携带跨引擎逻辑状态。
    physx_snapshot = get_mirror_snapshot(runtime)
    physics = _NewtonSolverStateRuntime(
        [[2.5, 8.0, 9.0]],
        baseline_values=[[0.0, 0.0, 0.0]],
    )
    runtime.session.physics_runtime = physics

    result = set_mirror_snapshot(runtime, physx_snapshot)

    assert result.accepted
    assert physics.reset_calls == 1
    assert physics.payload["values"] == [[0.0, 0.0, 0.0]]
    assert physics.payload["simulation_time_s"] == 0.0


def test_mirror_physx_to_newton_baseline_reset_rolls_back_after_late_failure() -> None:
    articulation = _FailingArticulation(offset=0.0, fail_position_calls=set())
    runtime = _transaction_mirror_runtime({"arm": articulation})
    physx_snapshot = get_mirror_snapshot(runtime)
    physics = _NewtonSolverStateRuntime(
        [[2.5, 8.0, 9.0]],
        baseline_values=[[0.0, 0.0, 0.0]],
    )
    runtime.session.physics_runtime = physics
    runtime.collision_registry.mark_dirty = lambda: (_ for _ in ()).throw(
        RuntimeError("collision invalidation failed")
    )

    with pytest.raises(RuntimeError, match="collision invalidation failed"):
        set_mirror_snapshot(runtime, physx_snapshot)

    assert physics.reset_calls == 1
    assert physics.set_values == [[[2.5, 8.0, 9.0]]]
    assert physics.payload["values"] == [[2.5, 8.0, 9.0]]
    assert physics.payload["simulation_time_s"] == 2.5


def test_mirror_newton_solver_state_preflight_precedes_robot_write() -> None:
    articulation = _FailingArticulation(offset=0.0, fail_position_calls=set())
    runtime = _transaction_mirror_runtime({"arm": articulation})
    physics = _NewtonSolverStateRuntime([[0.0, 1.0, 2.0]])
    runtime.session.physics_runtime = physics
    payload = get_mirror_snapshot(runtime).as_dict()
    payload["metadata"]["info"][NEWTON_SOLVER_STATE_INFO_KEY]["state_width"] = 4

    with pytest.raises(ValueError, match="solver state width mismatch"):
        set_mirror_snapshot(runtime, payload)

    assert articulation.position_calls == 0
    assert physics.set_values == []


def test_mirror_newton_solver_state_rolls_back_after_late_failure() -> None:
    articulation = _FailingArticulation(offset=0.0, fail_position_calls=set())
    runtime = _transaction_mirror_runtime({"arm": articulation})
    physics = _NewtonSolverStateRuntime([[0.25, 1.0, 2.0]])
    runtime.session.physics_runtime = physics
    snapshot = get_mirror_snapshot(runtime)
    physics.payload["values"] = [[1.5, 8.0, 9.0]]
    physics.payload["simulation_time_s"] = 1.5

    def fail_after_restore() -> None:
        raise RuntimeError("collision invalidation failed")

    runtime.collision_registry.mark_dirty = fail_after_restore

    with pytest.raises(RuntimeError, match="collision invalidation failed"):
        set_mirror_snapshot(runtime, snapshot)

    assert physics.set_values == [
        [[0.25, 1.0, 2.0]],
        [[1.5, 8.0, 9.0]],
    ]
    assert physics.payload["values"] == [[1.5, 8.0, 9.0]]
