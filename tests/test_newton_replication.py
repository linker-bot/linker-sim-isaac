from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import pytest

from linkerbot_sim.isaac.physics.newton import replication


@dataclass
class _CustomAttribute:
    dtype: object
    frequency: object
    references: str | None
    values: object


class _SchemaResolverNewton:
    pass


class _SchemaResolverMjc:
    pass


class _SchemaResolverPhysx:
    pass


class _FakeWarp:
    @staticmethod
    def transform_identity() -> int:
        return 0

    @staticmethod
    def transform_inverse(value: int) -> int:
        return -value

    @staticmethod
    def transform_multiply(left: int, right: int) -> int:
        return left + right


class _FakeSolverMuJoCo:
    @staticmethod
    def register_custom_attributes(builder: _FakeBuilder) -> None:
        builder.registered_before_parse = True
        builder.custom_attributes = {
            "test:target_label": _CustomAttribute(
                dtype=str,
                frequency="test:actuator",
                references=None,
                values=[],
            ),
            "test:actuator_world": _CustomAttribute(
                dtype=int,
                frequency="test:actuator",
                references="world",
                values=[],
            ),
        }


class _FakePrim:
    def __init__(self, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:
        return self._valid


class _FakeStage:
    def __init__(self, valid_paths: set[str]) -> None:
        self._valid_paths = valid_paths

    def GetPrimAtPath(self, path: str) -> _FakePrim:
        return _FakePrim(path in self._valid_paths)


class _FakeBuilder:
    instances: list[_FakeBuilder] = []
    finalize_calls = 0
    add_global_body = False
    fail_copy = False

    def __init__(self, *, up_axis: str) -> None:
        self.up_axis = up_axis
        self.registered_before_parse = False
        self.add_usd_calls: list[tuple[object, dict[str, object]]] = []
        self.add_builder_xforms: list[int] = []
        self.current_world = -1
        self.world_count = 0
        self.particle_count = 0
        self.body_count = 0
        self.shape_count = 0
        self.joint_count = 0
        self.articulation_count = 0
        self.particle_world: list[int] = []
        self.body_world: list[int] = []
        self.shape_world: list[int] = []
        self.joint_world: list[int] = []
        self.articulation_world: list[int] = []
        self.body_label: list[str] = []
        self.shape_label: list[str] = []
        self.joint_label: list[str] = []
        self.articulation_label: list[str] = []
        self.equality_constraint_type: list[int] = []
        self.equality_constraint_joint1: list[int] = []
        self.equality_constraint_joint2: list[int] = []
        self.equality_constraint_world: list[int] = []
        self.equality_constraint_label: list[str] = []
        self.constraint_mimic_joint0: list[int] = []
        self.constraint_mimic_joint1: list[int] = []
        self.constraint_mimic_world: list[int] = []
        self.constraint_mimic_label: list[str] = []
        self.custom_attributes: dict[str, _CustomAttribute] = {}
        type(self).instances.append(self)

    def add_usd(self, stage: object, **kwargs: object) -> dict[str, object]:
        assert self.registered_before_parse
        self.add_usd_calls.append((stage, kwargs))
        root_path = kwargs.get("root_path")
        if root_path is None:
            self.shape_count = 1
            self.shape_world = [-1]
            self.shape_label = ["/World/ground"]
            if type(self).add_global_body:
                self.body_count = 1
                self.body_world = [-1]
                self.body_label = ["/World/global_dynamic_body"]
            return {"scope": "global"}

        source = str(root_path)
        # One configured environment contains both hands: ten native MuJoCo
        # EqType.JOINT rows and no Newton/PhysX constraint_mimic rows.
        self.body_count = 20
        self.shape_count = 20
        self.joint_count = 20
        self.articulation_count = 2
        self.body_world = [-1] * self.body_count
        self.shape_world = [-1] * self.shape_count
        self.joint_world = [-1] * self.joint_count
        self.articulation_world = [-1] * self.articulation_count
        self.body_label = [f"{source}/body_{i}" for i in range(self.body_count)]
        self.shape_label = [f"{source}/shape_{i}" for i in range(self.shape_count)]
        self.joint_label = [f"{source}/joint_{i}" for i in range(self.joint_count)]
        self.articulation_label = [f"{source}/arm_{i}" for i in range(2)]
        self.equality_constraint_type = [2] * 10
        self.equality_constraint_joint1 = list(range(0, 20, 2))
        self.equality_constraint_joint2 = list(range(1, 20, 2))
        self.equality_constraint_world = [-1] * 10
        self.equality_constraint_label = [f"{source}/equality_{i}" for i in range(10)]
        self.custom_attributes["test:target_label"].values = [f"{source}/joint_1"]
        self.custom_attributes["test:actuator_world"].values = [-1]
        return {"scope": "prototype", "root": source}

    def begin_world(self) -> None:
        assert self.current_world == -1
        self.current_world = self.world_count
        self.world_count += 1

    def end_world(self) -> None:
        assert self.current_world >= 0
        self.current_world = -1

    def add_builder(self, prototype: _FakeBuilder, *, xform: int) -> None:
        if type(self).fail_copy:
            raise RuntimeError("copy failed")
        assert self.current_world >= 0
        self.add_builder_xforms.append(xform)
        world = self.current_world
        joint_offset = self.joint_count

        self.particle_count += prototype.particle_count
        self.body_count += prototype.body_count
        self.shape_count += prototype.shape_count
        self.joint_count += prototype.joint_count
        self.articulation_count += prototype.articulation_count
        self.particle_world.extend([world] * prototype.particle_count)
        self.body_world.extend([world] * prototype.body_count)
        self.shape_world.extend([world] * prototype.shape_count)
        self.joint_world.extend([world] * prototype.joint_count)
        self.articulation_world.extend([world] * prototype.articulation_count)
        self.body_label.extend(prototype.body_label)
        self.shape_label.extend(prototype.shape_label)
        self.joint_label.extend(prototype.joint_label)
        self.articulation_label.extend(prototype.articulation_label)

        self.equality_constraint_type.extend(prototype.equality_constraint_type)
        self.equality_constraint_joint1.extend(
            index + joint_offset for index in prototype.equality_constraint_joint1
        )
        self.equality_constraint_joint2.extend(
            index + joint_offset for index in prototype.equality_constraint_joint2
        )
        self.equality_constraint_world.extend(
            [world] * len(prototype.equality_constraint_type)
        )
        self.equality_constraint_label.extend(prototype.equality_constraint_label)
        self.constraint_mimic_joint0.extend(
            index + joint_offset for index in prototype.constraint_mimic_joint0
        )
        self.constraint_mimic_joint1.extend(
            index + joint_offset for index in prototype.constraint_mimic_joint1
        )
        self.constraint_mimic_world.extend(
            [world] * len(prototype.constraint_mimic_joint0)
        )
        self.constraint_mimic_label.extend(prototype.constraint_mimic_label)

        target_values = prototype.custom_attributes["test:target_label"].values
        world_values = prototype.custom_attributes["test:actuator_world"].values
        assert isinstance(target_values, list)
        assert isinstance(world_values, list)
        main_targets = self.custom_attributes["test:target_label"].values
        main_worlds = self.custom_attributes["test:actuator_world"].values
        assert isinstance(main_targets, list)
        assert isinstance(main_worlds, list)
        main_targets.extend(target_values)
        main_worlds.extend([world] * len(world_values))

    def finalize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        type(self).finalize_calls += 1


@pytest.fixture(autouse=True)
def _reset_fake_builder() -> None:
    _FakeBuilder.instances = []
    _FakeBuilder.finalize_calls = 0
    _FakeBuilder.add_global_body = False
    _FakeBuilder.fail_copy = False


@pytest.fixture
def fake_dependencies(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    dependencies = SimpleNamespace(
        model_builder_type=_FakeBuilder,
        solver_mujoco_type=_FakeSolverMuJoCo,
        schema_resolver_newton_type=_SchemaResolverNewton,
        schema_resolver_mjc_type=_SchemaResolverMjc,
        schema_resolver_physx_type=_SchemaResolverPhysx,
        warp=_FakeWarp,
    )
    monkeypatch.setattr(replication, "_load_newton_dependencies", lambda: dependencies)
    return dependencies


def test_upstream_mjc_resolver_maps_native_joint_frictionloss() -> None:
    """Newton 必须直接消费 importer author 的 mjc:frictionloss。"""

    pytest.importorskip("newton")
    resolver = replication._load_newton_dependencies().schema_resolver_mjc_type
    friction_attributes = [
        fields["friction"]
        for fields in resolver.mapping.values()
        if "friction" in fields
    ]

    assert len(friction_attributes) == 1
    assert friction_attributes[0].name == "mjc:frictionloss"
    assert friction_attributes[0].default == 0.0


def test_build_parses_once_and_replicates_native_joint_equalities(
    fake_dependencies: SimpleNamespace,
) -> None:
    del fake_dependencies
    stage = _FakeStage({"/World/envs/env_0"})

    result = replication.build_replicated_newton_builder(
        stage,
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0", "/World/envs/env_1"),
        world_transforms=(10, 20),
        global_ignore_paths=("/World/debug", "/World/debug"),
    )

    assert len(_FakeBuilder.instances) == 2
    builder, prototype = _FakeBuilder.instances
    assert result.builder is builder
    assert result.prototype_builder is prototype
    assert result.global_stage_info == {"scope": "global"}
    assert result.prototype_stage_info == {
        "scope": "prototype",
        "root": "/World/envs/env_0",
    }
    assert result.num_worlds == 2
    assert len(builder.add_usd_calls) == 1
    assert len(prototype.add_usd_calls) == 1
    assert builder.add_usd_calls[0][1]["ignore_paths"] == [
        "/World/envs",
        "/World/debug",
    ]
    assert prototype.add_usd_calls[0][1]["root_path"] == "/World/envs/env_0"
    for current in (builder, prototype):
        assert current.add_usd_calls[0][1][
            "joint_drive_gains_scaling"
        ] == pytest.approx(math.pi / 180.0)
        resolvers = current.add_usd_calls[0][1]["schema_resolvers"]
        assert [type(value) for value in resolvers] == [
            _SchemaResolverNewton,
            _SchemaResolverMjc,
            _SchemaResolverPhysx,
        ]
        assert current.registered_before_parse

    assert builder.world_count == 2
    assert builder.current_world == -1
    assert builder.add_builder_xforms == [0, 10]
    assert builder.equality_constraint_type == [2] * 20
    assert builder.equality_constraint_world == [0] * 10 + [1] * 10
    assert builder.equality_constraint_joint1 == list(range(0, 40, 2))
    assert builder.equality_constraint_joint2 == list(range(1, 40, 2))
    assert builder.constraint_mimic_joint0 == []
    assert builder.equality_constraint_label[0] == "/World/envs/env_0/equality_0"
    assert builder.equality_constraint_label[10] == "/World/envs/env_1/equality_0"
    assert builder.shape_label[0] == "/World/ground"
    assert builder.shape_label[1] == "/World/envs/env_0/shape_0"
    assert builder.shape_label[21] == "/World/envs/env_1/shape_0"
    custom_labels = builder.custom_attributes["test:target_label"].values
    assert custom_labels == [
        "/World/envs/env_0/joint_1",
        "/World/envs/env_1/joint_1",
    ]
    assert _FakeBuilder.finalize_calls == 0


def test_build_forwards_explicit_joint_drive_gain_scaling(
    fake_dependencies: SimpleNamespace,
) -> None:
    del fake_dependencies

    replication.build_replicated_newton_builder(
        _FakeStage({"/World/envs/env_0"}),
        prototype_root="/World/envs/env_0",
        destination_roots=("/World/envs/env_0",),
        joint_drive_gains_scaling=0.25,
    )

    builder, prototype = _FakeBuilder.instances
    for current in (builder, prototype):
        assert current.add_usd_calls[0][1]["joint_drive_gains_scaling"] == 0.25


def test_identity_source_preserves_nonzero_absolute_world_origins(
    fake_dependencies: SimpleNamespace,
) -> None:
    """An unpositioned prototype must retain the replicated layout translation."""

    del fake_dependencies
    result = replication.build_replicated_newton_builder(
        _FakeStage({"/World/envs/env_0"}),
        prototype_root="/World/envs/env_0",
        destination_roots=(
            "/World/envs/env_0",
            "/World/envs/env_1",
            "/World/envs/env_2",
        ),
        world_transforms=(10, 20, 40),
        source_world_transform=0,
    )

    builder, _prototype = _FakeBuilder.instances
    assert builder.add_builder_xforms == [10, 20, 40]
    assert result.world_transforms == (10, 20, 40)
    assert result.source_world_transform == 0


def test_single_world_uses_identity_and_only_ignores_prototype(
    fake_dependencies: SimpleNamespace,
) -> None:
    del fake_dependencies
    stage = _FakeStage({"/World/robot"})

    result = replication.build_replicated_newton_builder(
        stage,
        prototype_root="/World/robot",
        destination_roots=("/World/robot",),
    )

    builder, _ = _FakeBuilder.instances
    assert builder.add_usd_calls[0][1]["ignore_paths"] == ["/World/robot"]
    assert builder.add_builder_xforms == [0]
    assert result.environment_root == "/World/robot"
    assert result.world_transforms == (0,)
    assert result.source_world_transform == 0
    assert len(builder.equality_constraint_type) == 10


def test_label_rewrite_is_exact_boundary_safe_and_supports_mapping_columns() -> None:
    builder = SimpleNamespace(
        body_label=[
            "/Source",
            "/Source/link",
            "/SourceSibling/link",
            "display name",
        ],
        body_world=[0, 1, 1, 0],
        equality_constraint_key=["/Source/equality"],
        equality_constraint_world=[1],
        constraint_mimic_label=["/Source/mimic"],
        constraint_mimic_world=[0],
        custom_attributes={
            "test:label": _CustomAttribute(
                dtype=str,
                frequency="test:item",
                references=None,
                values={3: "/Source", 4: "/SourceSibling/item"},
            ),
            "test:world": _CustomAttribute(
                dtype=int,
                frequency="test:item",
                references="world",
                values={3: 0, 4: 1},
            ),
        },
    )

    replication.rename_replicated_builder_labels(
        builder,
        source_root="/Source/",
        destination_roots=("/Dest/zero", "/Dest/one"),
    )

    assert builder.body_label == [
        "/Dest/zero",
        "/Dest/one/link",
        "/SourceSibling/link",
        "display name",
    ]
    assert builder.equality_constraint_key == ["/Dest/one/equality"]
    assert builder.constraint_mimic_label == ["/Dest/zero/mimic"]
    assert builder.custom_attributes["test:label"].values == {
        3: "/Dest/zero",
        4: "/SourceSibling/item",
    }


def test_global_dynamic_entities_are_rejected(
    fake_dependencies: SimpleNamespace,
) -> None:
    del fake_dependencies
    _FakeBuilder.add_global_body = True

    with pytest.raises(RuntimeError, match="global Newton scope contains a body"):
        replication.build_replicated_newton_builder(
            _FakeStage({"/World/envs/env_0"}),
            prototype_root="/World/envs/env_0",
            destination_roots=("/World/envs/env_0", "/World/envs/env_1"),
        )


def test_copy_failure_closes_world_context(
    fake_dependencies: SimpleNamespace,
) -> None:
    del fake_dependencies
    _FakeBuilder.fail_copy = True

    with pytest.raises(RuntimeError, match="copy failed"):
        replication.build_replicated_newton_builder(
            _FakeStage({"/World/envs/env_0"}),
            prototype_root="/World/envs/env_0",
            destination_roots=("/World/envs/env_0",),
        )

    builder, _ = _FakeBuilder.instances
    assert builder.current_world == -1
    assert _FakeBuilder.finalize_calls == 0
