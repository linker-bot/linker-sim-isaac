from __future__ import annotations

from dataclasses import dataclass, replace
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.isaac.replicated_scene.assets import (
    _bind_fixed_joints_to_environment_anchor,
    _physical_tcp_binding,
    source_object_configs,
    validate_single_dynamic_rigid_object,
)
from linkerbot_sim.configuration import load_kaleidoscope_config
from linkerbot_sim.configuration.controllers import controller_profiles_from_mappings
from linkerbot_sim.configuration.objects import object_profile_from_mapping
from linkerbot_sim.configuration.robots import RobotProfileSettings
from linkerbot_sim.isaac.replicated_scene.layout import (
    env_local_prim_path,
    environment_origins,
    environment_root_paths,
    paths_from_suffix,
    relative_prim_suffix,
)
from linkerbot_sim.isaac.replicated_scene.newton_builder import NewtonWorldPlan
from linkerbot_sim.isaac.replicated_scene import physx_builder
from linkerbot_sim.isaac.replicated_scene import assets as replicated_assets
from linkerbot_sim.isaac.replicated_scene.physx_builder import PhysxGridClonePlan
from linkerbot_sim.isaac.replicated_scene.types import ReplicatedPhysxScene
from linkerbot_sim.isaac.replicated_scene.views import (
    finalize_replicated_robot_views,
)


def _environments() -> SimpleNamespace:
    return SimpleNamespace(
        base_env_path="/World/envs",
        env_prefix="env",
        origin_xyz=(1.0, -2.0, 0.5),
    )


def _strict_controller_document(target: str) -> dict[str, object]:
    """为 replicated-scene 单元测试提供完整、无配置默认的 controller。"""

    follower = {"stiffness": 0.0, "damping": 0.0, "max_force": 1.0}
    return {
        "target": target,
        "position_control": {
            "method": "implicit",
            "active_joints": {
                "stiffness": 1.0,
                "damping": 0.1,
                "max_force": 1.0,
            },
            "follower_joints": dict(follower),
        },
        "velocity_control": {
            "method": "explicit",
            "active_joints": {"damping": 0.1, "max_force": 1.0},
            "follower_joints": dict(follower),
        },
        "effort_control": {
            "method": "direct",
            "active_joints": {"effort_limit": 1.0},
            "follower_joints": dict(follower),
        },
    }


def test_layout_and_paths_are_deterministic_row_major() -> None:
    settings = _environments()
    roots = environment_root_paths(settings, num_envs=5)
    assert roots == tuple(f"/World/envs/env_{index}" for index in range(5))
    np.testing.assert_allclose(
        environment_origins(settings, num_envs=5, spacing_m=3.0),
        [
            [1.0, -2.0, 0.5],
            [4.0, -2.0, 0.5],
            [7.0, -2.0, 0.5],
            [1.0, 1.0, 0.5],
            [4.0, 1.0, 0.5],
        ],
    )
    source = env_local_prim_path(roots[0], "/World/Robots/left")
    assert source == "/World/envs/env_0/Robots/left"
    suffix = relative_prim_suffix(roots[0], source)
    assert suffix == "Robots/left"
    assert paths_from_suffix(roots[:2], suffix) == (
        "/World/envs/env_0/Robots/left",
        "/World/envs/env_1/Robots/left",
    )


def test_zero_spacing_colocates_separate_world_origins() -> None:
    settings = _environments()

    np.testing.assert_array_equal(
        environment_origins(settings, num_envs=4, spacing_m=0.0),
        np.repeat([[1.0, -2.0, 0.5]], 4, axis=0),
    )


@pytest.mark.parametrize(
    ("prepare_render_topology", "expected_ops", "expected_reset"),
    (
        (True, ("xformOp:transform:newtonRenderWorld",), True),
        (False, (), False),
    ),
)
def test_source_environment_root_respects_render_intent_at_definition(
    prepare_render_topology: bool,
    expected_ops: tuple[str, ...],
    expected_reset: bool,
) -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    replicated_assets.define_source_environment(
        stage,
        "/World/envs/env_0",
        prepare_newton_render_topology=prepare_render_topology,
    )

    xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0"))
    assert (
        tuple(str(op.GetOpName()) for op in xform.GetOrderedXformOps()) == expected_ops
    )
    assert bool(xform.GetResetXformStack()) is expected_reset


def test_physics_builders_derive_unique_replication_mechanisms() -> None:
    """同一环境事实不能重新组合成另一物理引擎的复制策略。"""

    environments = _environments()

    physx = PhysxGridClonePlan.from_environment_settings(environments, num_envs=4)
    newton = NewtonWorldPlan.from_environment_settings(environments, num_envs=4)

    assert physx.spacing_m == 3.0
    assert physx.replicate_physics is True
    assert physx.copy_from_source is True
    assert physx.enable_env_ids is True
    np.testing.assert_allclose(
        physx.env_origins,
        [
            [1.0, -2.0, 0.5],
            [4.0, -2.0, 0.5],
            [1.0, 1.0, 0.5],
            [4.0, 1.0, 0.5],
        ],
    )
    assert newton.world_count == 4
    np.testing.assert_array_equal(
        newton.env_origins,
        np.repeat([[1.0, -2.0, 0.5]], 4, axis=0),
    )


def test_physx_clone_plan_forces_verified_grid_cloner_contract(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _GridCloner:
        def __init__(self, **kwargs: object) -> None:
            calls["constructor"] = kwargs

        def clone(self, **kwargs: object) -> np.ndarray:
            calls["clone"] = kwargs
            return plan.env_origins

    isaacsim = ModuleType("isaacsim")
    core = ModuleType("isaacsim.core")
    cloner = ModuleType("isaacsim.core.cloner")
    cloner.GridCloner = _GridCloner  # type: ignore[attr-defined]
    isaacsim.core = core  # type: ignore[attr-defined]
    core.cloner = cloner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "isaacsim.core", core)
    monkeypatch.setitem(sys.modules, "isaacsim.core.cloner", cloner)

    plan = PhysxGridClonePlan.from_environment_settings(
        _environments(),
        num_envs=4,
    )
    stage = object()
    actual = physx_builder._clone_environments(stage=stage, plan=plan)

    np.testing.assert_array_equal(actual, plan.env_origins)
    assert calls["constructor"] == {
        "spacing": 3.0,
        "num_per_row": 2,
        "stage": stage,
    }
    clone_call = calls["clone"]
    assert clone_call["replicate_physics"] is True  # type: ignore[index]
    assert clone_call["copy_from_source"] is True  # type: ignore[index]
    assert clone_call["enable_env_ids"] is True  # type: ignore[index]


def test_canonical_kaleidoscope_scene_has_one_state_owned_dynamic_rigid() -> None:
    config = load_kaleidoscope_config()
    objects = source_object_configs(
        config.scene,
        env_root="/World/envs/env_0",
    )

    validate_single_dynamic_rigid_object(
        objects,
        expected_name=config.task.dynamic_object,
    )


def test_source_objects_fail_closed_without_resolved_profile() -> None:
    scene = SimpleNamespace(
        objects=(
            SimpleNamespace(
                name="block",
                object_profile="TblockV1_default",
                prim_path="/World/TBlock",
                root_pose=SimpleNamespace(
                    xyz=(0.0, 0.0, 0.0),
                    rpy=(0.0, 0.0, 0.0),
                ),
            ),
        )
    )

    with pytest.raises(RuntimeError, match="no resolved object profile"):
        source_object_configs(scene, env_root="/World/envs/env_0")


def test_newton_source_robot_freezes_render_topology_after_root_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = RobotProfileSettings.from_mapping(
        {
            "robot": {
                "kind": "hand",
                "name": "unit_hand",
                "asset_type": "mjcf",
                "asset_path": "unit_hand.xml",
            },
            "curobo": {"enabled": False},
            "joint_groups": {"arm": [], "hand": ["hand_joint"]},
        }
    )
    scene = SimpleNamespace(
        robots=(
            SimpleNamespace(
                label="arm",
                robot_profile="unit_hand",
                resolved_profile=profile,
                root_pose=SimpleNamespace(
                    xyz=(0.0, 0.0, 0.0),
                    rpy=(0.0, 0.0, 0.0),
                ),
                controller_profile=None,
            ),
        )
    )
    controllers = controller_profiles_from_mappings(
        {
            component: _strict_controller_document(component)
            for component in ("arm", "hand")
        }
    )
    events: list[str] = []
    render_intents: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        replicated_assets,
        "import_robot_asset",
        lambda _robot, *, physics_backend, prepare_newton_render_topology, root_pose: (
            render_intents.append(("import", prepare_newton_render_topology))
            or (
                "/World/envs/env_0/Robots/arm/Articulation",
                SimpleNamespace(physics_backend=physics_backend),
                "/World/envs/env_0/Robots/arm",
            )
        ),
    )
    monkeypatch.setattr(
        replicated_assets,
        "apply_root_pose",
        lambda *_args, **kwargs: (
            render_intents.append(
                ("root_pose", kwargs["prepare_newton_render_topology"])
            )
            or events.append("root_pose")
        ),
    )
    monkeypatch.setattr(
        replicated_assets,
        "prepare_newton_render_subtree",
        lambda **kwargs: events.append(f"render:{kwargs['subtree_root']}"),
    )
    monkeypatch.setattr(
        replicated_assets,
        "robot_usd_override_configs",
        lambda _controllers: {},
    )
    monkeypatch.setattr(
        replicated_assets,
        "apply_robot_usd_overrides",
        lambda *_args, **_kwargs: events.append("usd_overrides"),
    )
    monkeypatch.setattr(
        replicated_assets,
        "apply_robot_gravity_policy",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        replicated_assets,
        "_physical_tcp_binding",
        lambda **_kwargs: (
            "tcp",
            "link",
            "/World/envs/env_0/Robots/arm/link",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        replicated_assets,
        "mjcf_fixed_root_joint_paths_without_body0",
        lambda *_args: ("/World/envs/env_0/Robots/arm/rootJoint",),
    )
    monkeypatch.setattr(
        replicated_assets,
        "_bind_fixed_joints_to_environment_anchor",
        lambda *_args, **_kwargs: (
            events.append("bind_anchor") or "/World/envs/env_0/__fixed_world_anchor"
        ),
    )

    robots = replicated_assets.import_source_robots(
        object(),
        scene_settings=scene,
        env_root="/World/envs/env_0",
        controller_bundle="bundle",
        controller_bundles={"bundle": controllers},
        solver_type=None,
        physics_backend="newton",
        prepare_newton_render_topology=True,
        object_configs=(),
    )

    assert len(robots) == 1
    assert render_intents == [("import", True), ("root_pose", True)]
    assert events == [
        "root_pose",
        "render:/World/envs/env_0/Robots/arm",
        "usd_overrides",
        "bind_anchor",
        "render:/World/envs/env_0/__fixed_world_anchor",
    ]


def _runtime_object(
    name: str,
    *,
    kind: str = "rigid",
    static: bool = False,
) -> SimpleNamespace:
    object_data: dict[str, object] = {
        "name": name,
        "kind": kind,
        "source": "usd",
        "asset_path": f"{name}.usda",
    }
    if kind == "rigid":
        object_data["physics"] = {"static": static}
    else:
        object_data.update(
            {
                "root_path": f"/{name}",
                "state_summary": {"reference_body": "segment_0"},
            }
        )
    profile = object_profile_from_mapping(
        {"object": object_data},
        profile_name=f"{name}_profile",
    )
    return SimpleNamespace(
        name=name,
        kind=kind,
        object_profile=f"{name}_profile",
        profile=profile,
    )


@pytest.mark.parametrize(
    ("objects", "expected_name", "message"),
    [
        (
            (_runtime_object("fixture", static=True),),
            "block",
            "exactly one non-static rigid object",
        ),
        (
            (_runtime_object("block"), _runtime_object("other")),
            "block",
            "exactly one non-static rigid object",
        ),
        (
            (_runtime_object("block"),),
            "other",
            "does not match",
        ),
        (
            (
                _runtime_object("block"),
                _runtime_object("rope", kind="dynamic_chain"),
            ),
            "block",
            "does not support dynamic_chain",
        ),
    ],
)
def test_single_dynamic_rigid_contract_rejects_incomplete_state_ownership(
    objects: tuple[SimpleNamespace, ...],
    expected_name: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_single_dynamic_rigid_object(
            objects,  # type: ignore[arg-type]
            expected_name=expected_name,
        )


def test_mjcf_world_fixed_joint_is_rebound_to_cloned_environment_anchor() -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    body = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    joint = UsdPhysics.FixedJoint.Define(
        stage,
        "/World/envs/env_0/Robot/rootJoint_robot",
    )
    joint.CreateBody1Rel().SetTargets([Sdf.Path(str(body.GetPath()))])
    assert joint.GetBody0Rel().GetTargets() == []

    anchor = _bind_fixed_joints_to_environment_anchor(
        stage,
        env_root="/World/envs/env_0",
        joint_paths=(str(joint.GetPath()),),
    )

    assert anchor == "/World/envs/env_0/__fixed_world_anchor"
    assert joint.GetBody0Rel().GetTargets() == [Sdf.Path(anchor)]
    anchor_prim = stage.GetPrimAtPath(anchor)
    assert anchor_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.RigidBodyAPI(anchor_prim).GetKinematicEnabledAttr().Get() is True


def test_physical_tcp_binding_consumes_catalog_typed_robot_profile() -> None:
    """replicated scene 不得把 catalog 的 RobotProfileSettings 当作 raw mapping。"""

    from pxr import Usd, UsdGeom, UsdPhysics

    profile = load_kaleidoscope_config().scene.robots[0].resolved_profile
    assert profile is not None
    stage = Usd.Stage.CreateInMemory()
    root_path = "/World/envs/env_0/Robots/left"
    UsdGeom.Xform.Define(stage, root_path)
    parent_name = "AR5V2_L_arm_flan_link"
    parent = UsdGeom.Xform.Define(stage, f"{root_path}/{parent_name}").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(parent)

    binding = _physical_tcp_binding(
        stage=stage,
        imported_root_path=root_path,
        profile=profile,
    )

    assert binding == (
        "AR5V2_L_pinch_tcp",
        parent_name,
        f"{root_path}/{parent_name}",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )


@dataclass(frozen=True)
class _Robot:
    label: str
    articulation_view: object
    controlled_joints: tuple[str, ...]
    asset_type: str = "usd"
    asset_path: object = None
    command_joint_names: tuple[str, ...] = ()
    command_joint_indices: np.ndarray | None = None

    def with_command_binding(self, *, names, indices):
        return replace(
            self,
            command_joint_names=tuple(names),
            command_joint_indices=np.asarray(indices, dtype=np.int64),
        )


def test_finalize_views_freezes_articulation_order_without_product_dependency() -> None:
    robot = _Robot(
        label="arm",
        articulation_view=SimpleNamespace(dof_names=["j0", "j1", "j2"]),
        controlled_joints=("j2", "j0"),
    )
    scene = ReplicatedPhysxScene(
        env_root_paths=("/World/envs/env_0",),
        env_origins=np.zeros((1, 3), dtype=np.float32),
        robots=(robot,),  # type: ignore[arg-type]
        object_handles=(),
        object_prim_paths={},
    )
    finalized = finalize_replicated_robot_views(scene)
    assert finalized.robots[0].command_joint_names == ("j2", "j0")
    np.testing.assert_array_equal(
        finalized.robots[0].command_joint_indices,
        [2, 0],
    )
