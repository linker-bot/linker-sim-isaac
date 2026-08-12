from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import linkerbot_sim.isaac.physics.newton.constraints as newton_constraints
from linkerbot_sim.isaac.physics.newton.constraints import (
    COLD_STATE_PROJECTION_SCOPE,
    NATIVE_JOINT_EQUALITY_EXECUTOR,
    ExpectedMasterFollowerConstraint,
    MasterFollowerExecutorMetadata,
    NewtonColdStateProjector,
    NewtonConstraintAuditError,
    NewtonDeviceWorldMasks,
    audit_native_master_follower_constraints,
    make_selected_world_mask,
)


def _constraint_topology(
    *,
    representation: str,
    world_count: int = 2,
    relations_per_world: int = 10,
) -> tuple[SimpleNamespace, tuple[ExpectedMasterFollowerConstraint, ...]]:
    joint_labels: list[str] = []
    joint_world: list[int] = []
    equality_joint1: list[int] = []
    equality_joint2: list[int] = []
    equality_world: list[int] = []
    equality_polycoef: list[list[float]] = []
    equality_labels: list[str] = []
    expectations: list[ExpectedMasterFollowerConstraint] = []

    for world in range(world_count):
        for relation in range(relations_per_world):
            follower_label = f"/World/envs/env_{world}/hand/follower_{relation}"
            master_label = f"/World/envs/env_{world}/hand/master_{relation}"
            follower_joint = len(joint_labels)
            master_joint = follower_joint + 1
            joint_labels.extend((follower_label, master_label))
            joint_world.extend((world, world))
            equality_joint1.append(follower_joint)
            equality_joint2.append(master_joint)
            equality_world.append(world)
            coefficient = 1.1 + relation * 0.01
            polycoef = [0.05, coefficient, 0.1, 0.0, 0.0]
            equality_polycoef.append(polycoef)
            equality_label = f"{follower_label}:equality"
            equality_labels.append(equality_label)
            expectations.append(
                ExpectedMasterFollowerConstraint(
                    world=world,
                    follower_joint_label=follower_label,
                    master_joint_label=master_label,
                    polycoef=tuple(polycoef),
                    constraint_label=equality_label,
                )
            )

    joint_count = len(joint_labels)
    values: dict[str, object] = {
        "world_count": world_count,
        "joint_label": joint_labels,
        "joint_world": joint_world,
        "joint_type": [1] * joint_count,
        "joint_q_start": list(range(joint_count)),
        "joint_qd_start": list(range(joint_count)),
        "joint_coord_count": joint_count,
        "joint_dof_count": joint_count,
        "joint_q": [0.0] * joint_count,
        "joint_qd": [0.0] * joint_count,
        "equality_constraint_type": [2] * len(expectations),
        "equality_constraint_joint1": equality_joint1,
        "equality_constraint_joint2": equality_joint2,
        "equality_constraint_enabled": [True] * len(expectations),
        "equality_constraint_world": equality_world,
        "equality_constraint_label": equality_labels,
        "equality_constraint_polycoef": equality_polycoef,
        "constraint_mimic_joint0": [],
        "constraint_mimic_joint1": [],
        "constraint_mimic_coef0": [],
        "constraint_mimic_coef1": [],
        "constraint_mimic_enabled": [],
        "constraint_mimic_label": [],
        "constraint_mimic_world": [],
    }
    if representation == "model":
        numeric_columns = (
            "joint_world",
            "joint_type",
            "equality_constraint_type",
            "equality_constraint_joint1",
            "equality_constraint_joint2",
            "equality_constraint_enabled",
            "equality_constraint_world",
            "equality_constraint_polycoef",
        )
        for name in numeric_columns:
            values[name] = np.asarray(values[name])
        values["joint_q_start"] = np.arange(joint_count + 1, dtype=np.int32)
        values["joint_qd_start"] = np.arange(joint_count + 1, dtype=np.int32)
        values["equality_constraint_count"] = len(expectations)
        values["constraint_mimic_count"] = 0
    return SimpleNamespace(**values), tuple(expectations)


def _metadata(**changes: object) -> MasterFollowerExecutorMetadata:
    values: dict[str, object] = {
        "dynamic_executor": NATIVE_JOINT_EQUALITY_EXECUTOR,
        "state_projection_scope": COLD_STATE_PROJECTION_SCOPE,
        "runtime_target_writer": None,
        "follower_drive_prim_paths": (),
        "follower_actuator_labels": (),
    }
    values.update(changes)
    return MasterFollowerExecutorMetadata(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("representation", ("builder", "model"))
def test_audit_accepts_exact_ten_joint_equalities_per_world(
    representation: str,
) -> None:
    source, expected = _constraint_topology(representation=representation)

    result = audit_native_master_follower_constraints(
        source,
        expected,
        expected_world_count=2,
        executor_metadata=_metadata(),
    )

    assert result.representation == representation
    assert result.relation_count == 20
    assert len(result.bindings_for_world(0)) == 10
    assert len(result.bindings_for_world(1)) == 10
    first = result.bindings[0]
    assert first.follower_joint_index == 0
    assert first.master_joint_index == 1
    assert first.follower_q_index == 0
    assert first.master_q_index == 1
    assert first.polycoef == (0.05, 1.1, 0.1, 0.0, 0.0)


def test_audit_rejects_any_constraint_mimic_representation() -> None:
    source, expected = _constraint_topology(representation="builder")
    source.constraint_mimic_joint0.append(0)

    with pytest.raises(NewtonConstraintAuditError, match="zero constraint_mimic"):
        audit_native_master_follower_constraints(
            source,
            expected,
            expected_world_count=2,
            executor_metadata=_metadata(),
        )

    source, expected = _constraint_topology(representation="builder")
    del source.constraint_mimic_world
    with pytest.raises(NewtonConstraintAuditError, match="missing=.*mimic_world"):
        audit_native_master_follower_constraints(
            source,
            expected,
            expected_world_count=2,
            executor_metadata=_metadata(),
        )


def test_audit_rejects_wrong_polycoef_and_disabled_equality() -> None:
    source, expected = _constraint_topology(representation="builder")
    source.equality_constraint_polycoef[3][1] = 9.0

    with pytest.raises(NewtonConstraintAuditError, match="polycoef differs"):
        audit_native_master_follower_constraints(
            source,
            expected,
            expected_world_count=2,
            executor_metadata=_metadata(),
        )

    source.equality_constraint_polycoef[3][1] = expected[3].polycoef[1]
    source.equality_constraint_enabled[3] = False
    with pytest.raises(NewtonConstraintAuditError, match="row is disabled"):
        audit_native_master_follower_constraints(
            source,
            expected,
            expected_world_count=2,
            executor_metadata=_metadata(),
        )


def test_audit_allows_anonymous_unrelated_equality_but_not_anonymous_joint() -> None:
    source, expected = _constraint_topology(representation="builder")
    source.equality_constraint_type.append(0)
    source.equality_constraint_joint1.append(-1)
    source.equality_constraint_joint2.append(-1)
    source.equality_constraint_enabled.append(True)
    source.equality_constraint_world.append(0)
    source.equality_constraint_label.append("")
    source.equality_constraint_polycoef.append([0.0] * 5)

    result = audit_native_master_follower_constraints(
        source,
        expected,
        expected_world_count=2,
        executor_metadata=_metadata(),
    )
    assert result.relation_count == 20

    source.equality_constraint_label[0] = ""
    with pytest.raises(NewtonConstraintAuditError, match="non-empty exact label"):
        audit_native_master_follower_constraints(
            source,
            expected,
            expected_world_count=2,
            executor_metadata=_metadata(),
        )


def test_audit_ignores_duplicate_labels_on_unrelated_joints() -> None:
    source, expected = _constraint_topology(representation="builder")
    source.joint_label.extend(("joint_44", "joint_44"))
    source.joint_world.extend((0, 1))
    source.joint_type.extend((4, 4))
    source.joint_q_start.extend((len(source.joint_q), len(source.joint_q) + 7))
    source.joint_qd_start.extend((len(source.joint_qd), len(source.joint_qd) + 6))
    source.joint_q.extend([0.0] * 14)
    source.joint_qd.extend([0.0] * 12)
    source.joint_coord_count += 14
    source.joint_dof_count += 12

    result = audit_native_master_follower_constraints(
        source,
        expected,
        expected_world_count=2,
        executor_metadata=_metadata(),
    )

    assert result.relation_count == 20


def test_audit_rejects_duplicate_label_used_by_expected_relation() -> None:
    source, expected = _constraint_topology(representation="builder")
    source.joint_label[2] = source.joint_label[0]

    with pytest.raises(NewtonConstraintAuditError, match="resolve exactly once"):
        audit_native_master_follower_constraints(
            source,
            expected,
            expected_world_count=2,
            executor_metadata=_metadata(),
        )


@pytest.mark.parametrize(
    "changes,match",
    (
        ({"runtime_target_writer": "python_follower_targets"}, "runtime target writer"),
        ({"follower_drive_prim_paths": ("/World/follower",)}, "follower drives"),
        ({"follower_actuator_labels": ("follower_motor",)}, "follower actuators"),
        ({"dynamic_executor": "drive"}, "must be native EqType.JOINT"),
        ({"state_projection_scope": "every_step"}, "reset/restore-only"),
    ),
)
def test_audit_rejects_second_executor_metadata(
    changes: dict[str, object], match: str
) -> None:
    source, expected = _constraint_topology(representation="builder")

    with pytest.raises(NewtonConstraintAuditError, match=match):
        audit_native_master_follower_constraints(
            source,
            expected,
            expected_world_count=2,
            executor_metadata=_metadata(**changes),
        )


def test_audit_rejects_wrong_world_and_duplicate_follower() -> None:
    source, expected = _constraint_topology(representation="builder")
    source.equality_constraint_world[10] = 0

    with pytest.raises(NewtonConstraintAuditError, match="distributed evenly"):
        audit_native_master_follower_constraints(
            source,
            expected,
            expected_world_count=2,
            executor_metadata=_metadata(),
        )

    source, expected = _constraint_topology(representation="builder")
    duplicate = list(expected)
    duplicate[1] = ExpectedMasterFollowerConstraint(
        world=0,
        follower_joint_label=expected[0].follower_joint_label,
        master_joint_label=expected[1].master_joint_label,
        polycoef=expected[1].polycoef,
        constraint_label=expected[1].constraint_label,
    )
    with pytest.raises(NewtonConstraintAuditError, match="only one native equality"):
        audit_native_master_follower_constraints(
            source,
            duplicate,
            expected_world_count=2,
            executor_metadata=_metadata(),
        )


def test_cold_projector_updates_only_selected_world_on_warp_cpu() -> None:
    wp = pytest.importorskip("warp")
    source, expected = _constraint_topology(representation="model")
    audit = audit_native_master_follower_constraints(
        source,
        expected,
        expected_world_count=2,
        executor_metadata=_metadata(),
    )
    projector = NewtonColdStateProjector(audit, device="cpu")
    joint_q_np = np.zeros(source.joint_coord_count, dtype=np.float32)
    joint_qd_np = np.zeros(source.joint_dof_count, dtype=np.float32)
    for binding in audit.bindings:
        joint_q_np[binding.follower_q_index] = -10.0 - binding.world
        joint_q_np[binding.master_q_index] = 0.2 + 0.01 * binding.equality_index
        joint_qd_np[binding.follower_qd_index] = -20.0 - binding.world
        joint_qd_np[binding.master_qd_index] = 0.3 + 0.01 * binding.equality_index
    original_q = joint_q_np.copy()
    original_qd = joint_qd_np.copy()
    joint_q = wp.array(joint_q_np, dtype=wp.float32, device="cpu")
    joint_qd = wp.array(joint_qd_np, dtype=wp.float32, device="cpu")
    selected = make_selected_world_mask([1], world_count=2, device="cpu")

    projector.project(
        joint_q=joint_q,
        joint_qd=joint_qd,
        selected_world_mask=selected,
    )
    actual_q = joint_q.numpy()
    actual_qd = joint_qd.numpy()

    for binding in audit.bindings:
        if binding.world == 0:
            assert (
                actual_q[binding.follower_q_index]
                == original_q[binding.follower_q_index]
            )
            assert (
                actual_qd[binding.follower_qd_index]
                == original_qd[binding.follower_qd_index]
            )
            continue
        master_q = original_q[binding.master_q_index]
        master_qd = original_qd[binding.master_qd_index]
        c0, c1, c2, c3, c4 = binding.polycoef
        expected_q = (
            c0 + c1 * master_q + c2 * master_q**2 + c3 * master_q**3 + c4 * master_q**4
        )
        derivative = (
            c1 + 2 * c2 * master_q + 3 * c3 * master_q**2 + 4 * c4 * master_q**3
        )
        assert actual_q[binding.follower_q_index] == pytest.approx(expected_q)
        assert actual_qd[binding.follower_qd_index] == pytest.approx(
            derivative * master_qd
        )


def test_make_selected_world_mask_rejects_duplicates_and_range() -> None:
    pytest.importorskip("warp")

    with pytest.raises(ValueError, match="unique indices"):
        make_selected_world_mask([1, 1], world_count=2, device="cpu")
    with pytest.raises(ValueError, match="unique indices"):
        make_selected_world_mask([2], world_count=2, device="cpu")


def test_projector_allocations_mask_upload_and_launch_use_owner_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wp = pytest.importorskip("warp")
    source, expected = _constraint_topology(representation="model")
    audit = audit_native_master_follower_constraints(
        source,
        expected,
        expected_world_count=2,
        executor_metadata=_metadata(),
    )
    joint_q = wp.zeros(source.joint_coord_count, dtype=wp.float32, device="cpu")
    joint_qd = wp.zeros(source.joint_dof_count, dtype=wp.float32, device="cpu")
    owner_stream = object()
    active_streams: list[object] = []
    scopes: list[tuple[object, bool, bool]] = []
    allocation_streams: list[object | None] = []
    launches: list[tuple[object, object | None]] = []

    class _ScopedStream:
        def __init__(
            self,
            stream: object,
            sync_enter: bool = True,
            sync_exit: bool = False,
        ) -> None:
            self.stream = stream
            self.sync_enter = sync_enter
            self.sync_exit = sync_exit

        def __enter__(self) -> object:
            scopes.append((self.stream, self.sync_enter, self.sync_exit))
            active_streams.append(self.stream)
            return self.stream

        def __exit__(self, *args: object) -> None:
            del args
            assert active_streams.pop() is self.stream

    original_array = wp.array

    def _record_array(*args: object, **kwargs: object) -> object:
        allocation_streams.append(active_streams[-1] if active_streams else None)
        return original_array(*args, **kwargs)

    kernel = object()

    def _record_launch(value: object, **kwargs: object) -> None:
        assert kwargs["stream"] is owner_stream
        launches.append((value, active_streams[-1] if active_streams else None))

    monkeypatch.setattr(wp, "ScopedStream", _ScopedStream)
    monkeypatch.setattr(wp, "array", _record_array)
    monkeypatch.setattr(wp, "launch", _record_launch)
    monkeypatch.setattr(newton_constraints, "_cold_projection_kernel", lambda: kernel)

    projector = NewtonColdStateProjector(
        audit,
        device="cpu",
        stream=owner_stream,
    )
    selected = make_selected_world_mask(
        [1],
        world_count=2,
        device="cpu",
        stream=owner_stream,
    )
    projector.project(
        joint_q=joint_q,
        joint_qd=joint_qd,
        selected_world_mask=selected,
    )

    assert allocation_streams
    assert all(stream is owner_stream for stream in allocation_streams)
    assert scopes
    assert all(scope == (owner_stream, False, False) for scope in scopes)
    # 第一条 launch 在 device 上标记 selected world，第二条才是 equality projector；
    # mask 构造不再通过 NumPy/wp.array 动态上传。
    assert len(launches) == 2
    assert launches[-1] == (kernel, owner_stream)


def test_device_world_masks_map_empty_and_partial_rows_without_host_readback() -> None:
    wp = pytest.importorskip("warp")
    articulation_world = wp.array([0, 1, 1, 2], dtype=wp.int32, device="cpu")
    row_world = wp.array([2, 0, 1], dtype=wp.int32, device="cpu")
    selector = NewtonDeviceWorldMasks(
        world_count=3,
        articulation_world=articulation_world,
        device="cpu",
    )

    empty = wp.array([False, False, False], dtype=wp.bool, device="cpu")
    world_mask = selector.world_mask((), masked_rows=((row_world, empty),))
    articulation_mask = selector.articulation_mask(world_mask)
    np.testing.assert_array_equal(world_mask.numpy(), [0, 0, 0])
    np.testing.assert_array_equal(
        articulation_mask.numpy(), [False, False, False, False]
    )

    partial = wp.array([True, False, True], dtype=wp.bool, device="cpu")
    world_mask = selector.world_mask((), masked_rows=((row_world, partial),))
    articulation_mask = selector.articulation_mask(world_mask)
    np.testing.assert_array_equal(world_mask.numpy(), [0, 1, 1])
    np.testing.assert_array_equal(articulation_mask.numpy(), [False, True, True, True])

    # host 冷路径 selector 与 device mask 可合并，且复用数组前会在 device 上清零旧选择。
    world_mask = selector.world_mask((0,), masked_rows=((row_world, empty),))
    np.testing.assert_array_equal(world_mask.numpy(), [1, 0, 0])


def test_audit_real_newton_builder_and_finalized_model() -> None:
    newton = pytest.importorskip("newton")
    wp = pytest.importorskip("warp")

    prototype = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(prototype)
    inertia = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)
    parent = -1
    joints: list[int] = []
    bodies: list[int] = []
    prototype_relations: list[tuple[str, str, str, tuple[float, ...]]] = []
    for relation in range(10):
        pair: list[int] = []
        for role in ("follower", "master"):
            relative_label = f"hand/{role}_{relation}"
            body = prototype.add_link(
                mass=1.0,
                inertia=inertia,
                label=f"{relative_label}/body",
            )
            joint = prototype.add_joint_revolute(
                parent=parent,
                child=body,
                axis=(0.0, 0.0, 1.0),
                label=relative_label,
            )
            parent = body
            bodies.append(body)
            joints.append(joint)
            pair.append(joint)
        coefficient = 1.125676 if relation < 9 else 1.226495
        polycoef = (0.0, coefficient, 0.0, 0.0, 0.0)
        constraint_label = f"hand/couple_{relation}"
        prototype.add_equality_constraint_joint(
            joint1=pair[0],
            joint2=pair[1],
            polycoef=list(polycoef),
            label=constraint_label,
        )
        prototype_relations.append(
            (
                f"hand/follower_{relation}",
                f"hand/master_{relation}",
                constraint_label,
                polycoef,
            )
        )
    prototype.add_articulation(joints, label="hand")
    # An unrelated equality may be anonymous and must not be counted as one of
    # the ten native master/follower rows.
    prototype.add_equality_constraint_connect(body1=bodies[0], body2=bodies[-1])

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    expected: list[ExpectedMasterFollowerConstraint] = []
    for world in range(2):
        world_root = f"/World/envs/env_{world}"
        builder.begin_world()
        builder.add_builder(prototype, label_prefix=world_root)
        builder.end_world()
        expected.extend(
            ExpectedMasterFollowerConstraint(
                world=world,
                follower_joint_label=f"{world_root}/{follower}",
                master_joint_label=f"{world_root}/{master}",
                polycoef=polycoef,
                constraint_label=f"{world_root}/{constraint}",
            )
            for follower, master, constraint, polycoef in prototype_relations
        )

    builder_audit = audit_native_master_follower_constraints(
        builder,
        expected,
        expected_world_count=2,
        executor_metadata=_metadata(),
    )
    model = builder.finalize(device="cpu")
    model_audit = audit_native_master_follower_constraints(
        model,
        expected,
        expected_world_count=2,
        executor_metadata=_metadata(),
    )

    assert builder_audit.representation == "builder"
    assert model_audit.representation == "model"
    assert model_audit.relation_count == 20
    assert model.equality_constraint_count == 22
    assert model.constraint_mimic_count == 0
