from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from linkerbot_sim.configuration.objects import object_profile_from_mapping
from linkerbot_sim.isaac.replicated_scene import newton_builder


class _Runtime:
    kind = "newton_cuda"

    def __init__(self) -> None:
        self.world_count = 0
        self.calls: list[dict[str, object]] = []

    def initialize_worlds(self, **kwargs: object) -> None:
        self.calls.append(dict(kwargs))
        self.world_count = len(kwargs["env_root_paths"])  # type: ignore[arg-type]


class _ArticulationView:
    def __init__(
        self,
        runtime: object,
        *,
        paths: tuple[str, ...],
        world_indices: tuple[int, ...],
        name: str,
    ) -> None:
        self.runtime = runtime
        self.paths = paths
        self.world_indices = world_indices
        self.name = name
        self.dof_names = ["joint_a", "joint_b"]
        self.bound: tuple[str, ...] | None = None
        self.prepared: tuple[int, ...] | None = None

    def bind_controllable_dofs(self, names: tuple[str, ...]) -> None:
        self.bound = tuple(names)

    def prepare_dof_selection(self, *, dof_indices: tuple[int, ...]) -> None:
        self.prepared = tuple(dof_indices)


def _source_robot() -> SimpleNamespace:
    controller_profiles = object()
    return SimpleNamespace(
        robot_id=0,
        label="left",
        profile_name="robot_profile",
        profile={},
        controller_bundle_name="newton",
        controller_profiles=controller_profiles,
        execution=object(),
        asset_path=Path("robot.usd"),
        asset_type="usd",
        articulation_path="/World/envs/env_0/Robots/left/articulation",
        imported_root_path="/World/envs/env_0/Robots/left",
        controlled_joints=("joint_b", "joint_a"),
        tcp_frame_name="tcp",
        tcp_parent_frame_name="flange",
        tcp_parent_body_path="/World/envs/env_0/Robots/left/flange",
        tcp_offset_xyz=(0.0, 0.0, 0.1),
        tcp_offset_rpy=(0.0, 0.0, 0.0),
    )


def test_newton_scene_imports_one_prototype_then_finalizes_all_worlds(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    source_robot = _source_robot()
    object_profile = object_profile_from_mapping(
        {
            "object": {
                "name": "Tblock",
                "kind": "rigid",
                "source": "usd",
                "asset_path": "tblock.usda",
                "physics": {"static": False},
            }
        },
        profile_name="TblockV1_default",
    )
    object_config = SimpleNamespace(
        name="Tblock",
        kind="rigid",
        object_profile="TblockV1_default",
        profile=object_profile,
        prim_path="/World/envs/env_0/TBlock",
    )
    object_handle = SimpleNamespace(name="Tblock", kind="rigid")
    calls: list[tuple[str, object]] = []
    source_render_intents: list[bool] = []
    object_render_intents: list[bool] = []
    object_backends: list[str] = []

    monkeypatch.setattr(
        newton_builder,
        "define_source_environment",
        lambda stage, root, *, prepare_newton_render_topology: (
            source_render_intents.append(prepare_newton_render_topology)
            or calls.append(("define", root))
        ),
    )
    monkeypatch.setattr(
        newton_builder,
        "source_object_configs",
        lambda scene, *, env_root: (object_config,),
    )
    monkeypatch.setattr(
        newton_builder,
        "import_source_objects",
        lambda stage, *, configs, physics_backend, prepare_newton_render_topology: (
            object_backends.append(physics_backend)
            or object_render_intents.append(prepare_newton_render_topology)
            or (object_handle,)
        ),
    )

    def import_robots(stage, **kwargs):
        calls.append(("robots", kwargs))
        return (source_robot,)

    monkeypatch.setattr(newton_builder, "import_source_robots", import_robots)
    monkeypatch.setattr(
        newton_builder,
        "NewtonArticulationView",
        _ArticulationView,
    )

    environments = SimpleNamespace(
        base_env_path="/World/envs",
        env_prefix="env",
        origin_xyz=(1.0, -1.0, 0.0),
    )
    scene = newton_builder.build_replicated_newton_scene(
        stage=object(),
        runtime=runtime,
        scene_settings=object(),
        environment_settings=environments,
        num_envs=3,
        dynamic_object_name="Tblock",
        controller_bundle="newton",
        controller_bundles={},
        prepare_newton_render_topology=True,
    )

    roots = tuple(f"/World/envs/env_{index}" for index in range(3))
    assert calls[0] == ("define", roots[0])
    assert source_render_intents == [True]
    assert object_backends == ["newton"]
    assert object_render_intents == [True]
    assert calls[1][1]["prepare_newton_render_topology"] is True
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["env_root_paths"] == roots
    manager_robot = runtime.calls[0]["robots"]["left"]
    assert manager_robot.asset_path == source_robot.asset_path
    assert manager_robot.imported_root_paths == tuple(
        f"{root}/Robots/left" for root in roots
    )
    assert runtime.calls[0]["object_handles"] == (object_handle,)
    np.testing.assert_allclose(
        runtime.calls[0]["env_origins"],
        [[1.0, -1.0, 0.0]] * 3,
    )

    assert scene.collision_isolation_strategy == "separate_worlds"
    assert scene.env_root_paths == roots
    robot = scene.robots[0]
    assert robot.controller_bundle_name == "newton"
    assert robot.controller_profiles is source_robot.controller_profiles
    assert robot.articulation_paths == tuple(
        f"{root}/Robots/left/articulation" for root in roots
    )
    assert robot.tcp_body_paths == tuple(f"{root}/Robots/left/flange" for root in roots)
    assert scene.object_prim_paths["Tblock"] == tuple(
        f"{root}/TBlock" for root in roots
    )
    assert robot.command_joint_names == ("joint_b", "joint_a")
    np.testing.assert_array_equal(robot.command_joint_indices, [1, 0])
    assert robot.articulation_view.bound == ("joint_b", "joint_a")
    assert robot.articulation_view.prepared == (1, 0)


def test_newton_scene_rejects_non_newton_runtime_before_stage_mutation() -> None:
    runtime = SimpleNamespace(kind="physx_cuda")
    try:
        newton_builder.build_replicated_newton_scene(
            stage=object(),
            runtime=runtime,
            scene_settings=object(),
            environment_settings=object(),
            num_envs=1,
            dynamic_object_name="Tblock",
            controller_bundle="newton",
            controller_bundles={},
            prepare_newton_render_topology=False,
        )
    except TypeError as exc:
        assert "newton_cuda" in str(exc)
    else:
        raise AssertionError("non-Newton runtime must be rejected")


def test_newton_scene_rejects_dynamic_chain_before_stage_or_world_mutation(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    mutations: list[str] = []
    chain_profile = object_profile_from_mapping(
        {
            "object": {
                "name": "rope",
                "kind": "dynamic_chain",
                "source": "usd",
                "asset_path": "rope.usda",
                "root_path": "/Rope",
                "state_summary": {"reference_body": "segment_0"},
            }
        },
        profile_name="capsule_rope",
    )
    chain = SimpleNamespace(
        name="rope",
        kind="dynamic_chain",
        object_profile="capsule_rope",
        profile=chain_profile,
        prim_path="/World/envs/env_0/Rope",
    )
    monkeypatch.setattr(
        newton_builder,
        "source_object_configs",
        lambda scene, *, env_root: (chain,),
    )
    monkeypatch.setattr(
        newton_builder,
        "define_source_environment",
        lambda *_args: mutations.append("stage"),
    )
    monkeypatch.setattr(
        newton_builder,
        "import_source_objects",
        lambda *_args, **_kwargs: mutations.append("objects"),
    )
    environments = SimpleNamespace(
        base_env_path="/World/envs",
        env_prefix="env",
        origin_xyz=(0.0, 0.0, 0.0),
    )

    try:
        newton_builder.build_replicated_newton_scene(
            stage=object(),
            runtime=runtime,
            scene_settings=object(),
            environment_settings=environments,
            num_envs=2,
            dynamic_object_name="rope",
            controller_bundle="newton",
            controller_bundles={},
            prepare_newton_render_topology=False,
        )
    except ValueError as exc:
        assert "dynamic_chain" in str(exc)
    else:
        raise AssertionError("dynamic_chain must fail the state ownership contract")
    assert mutations == []
    assert runtime.calls == []
