from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from linkerbot_sim.assets.usd_overrides import RobotUsdOverrideConfig
from linkerbot_sim.configuration.robots import RobotGravityPolicy
from linkerbot_sim.execution import setup
from linkerbot_sim.isaac.physics.backend import PhysicsCompatibilityWarning
from linkerbot_sim.robots.classification import RobotComponentMapping


class _Articulation:
    num_dof = 2

    def __init__(self, *, supports_per_link_gravity: bool | None) -> None:
        if supports_per_link_gravity is not None:
            self.supports_per_link_gravity = supports_per_link_gravity
        self.disable_gravity_calls = 0
        self.velocities: np.ndarray | None = None

    def disable_gravity(self) -> None:
        self.disable_gravity_calls += 1

    def set_joint_velocities(self, values: object) -> None:
        self.velocities = np.asarray(values, dtype=float)


class _JointController:
    instances: list["_JointController"] = []

    def __init__(self, robot: object, **kwargs: object) -> None:
        self.robot = robot
        self.kwargs = kwargs
        self.configure_calls = 0
        self.instances.append(self)

    def configure_runtime(self) -> None:
        self.configure_calls += 1


def _imported(
    articulation: object,
    *,
    gravity_counts: dict[str, int] | None = None,
) -> setup.ImportedRobot:
    return setup.ImportedRobot(
        articulation=articulation,
        articulation_path="/World/Robot",
        imported_root_path="/World/Robot",
        asset_path=Path("robot.usd"),
        asset_type="usd",
        controlled_joints=("joint_0", "joint_1"),
        gravity_policy=RobotGravityPolicy(default=False),
        component_mapping=RobotComponentMapping(),
        solver_counts={"configured": 0},
        gravity_counts=(
            {"configured": 0} if gravity_counts is None else gravity_counts
        ),
    )


def _install_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    _JointController.instances.clear()
    monkeypatch.setattr(setup, "JointController", _JointController)
    monkeypatch.setattr(
        setup,
        "joint_control_settings",
        lambda profiles, *, mode: (profiles, mode),
    )


def _robot_execution_for_projection_test(
    *,
    physx_overrides: object,
    contact_material: object | None = None,
) -> object:
    return SimpleNamespace(
        robot=SimpleNamespace(
            asset_type="mjcf",
            component_mapping=RobotComponentMapping(),
            gravity_policy=RobotGravityPolicy(default=False),
            contact_material=contact_material,
            name="robot",
            physx=SimpleNamespace(
                overrides=physx_overrides,
                solver_iterations=object(),
            ),
        ),
        controlled_joints=("joint_a",),
        root_pose=object(),
    )


@pytest.mark.parametrize("prepare_render_topology", (False, True))
def test_newton_import_projects_only_backend_neutral_robot_usd_fields(
    monkeypatch: pytest.MonkeyPatch,
    prepare_render_topology: bool,
) -> None:
    class _PhysxOverrides:
        def apply_to_configs(self, _configs: object) -> object:
            raise AssertionError("Newton must not project robot.physics.physx")

    shared = {"default": RobotUsdOverrideConfig(contact_static_friction=0.7)}

    class _ContactMaterial:
        def apply_to_configs(self, configs: object) -> object:
            assert configs == base
            return shared

    projected: list[dict[str, RobotUsdOverrideConfig]] = []
    author_events: list[str] = []
    render_intents: list[tuple[str, bool]] = []
    gravity_backends: list[str] = []
    base = {"default": RobotUsdOverrideConfig(drive_stiffness_seed=123.0)}
    monkeypatch.setattr(
        setup,
        "import_robot_asset",
        lambda _config, *, physics_backend, prepare_newton_render_topology, root_pose: (
            render_intents.append(("import", prepare_newton_render_topology))
            or (
                "/World/Robot",
                Path(f"robot-{physics_backend}.xml"),
                "/World/Robot",
            )
        ),
    )
    monkeypatch.setattr(
        setup,
        "apply_root_pose",
        lambda *_args, **kwargs: (
            render_intents.append(
                ("root_pose", kwargs["prepare_newton_render_topology"])
            )
            or author_events.append("root_pose")
        ),
    )
    monkeypatch.setattr(
        setup,
        "prepare_newton_render_subtree",
        lambda **_kwargs: (
            author_events.append("render_topology") or ("/World/Robot/link",)
        ),
    )
    monkeypatch.setattr(setup, "robot_usd_override_configs", lambda _profiles: base)
    monkeypatch.setattr(
        setup,
        "apply_robot_usd_overrides",
        lambda _root, configs, **_kwargs: (
            author_events.append("usd_overrides") or projected.append(configs)
        ),
    )
    monkeypatch.setattr(
        setup,
        "apply_solver_iteration_overrides",
        lambda *_args, **_kwargs: pytest.fail("Newton must not apply PhysX solver"),
    )
    monkeypatch.setattr(
        setup,
        "apply_robot_gravity_policy",
        lambda *_args, **kwargs: (
            gravity_backends.append(kwargs["physics_backend"]) or {"configured": 0}
        ),
    )

    imported = setup.import_execution_robot_to_stage(
        world=object(),
        stage=object(),
        single_articulation_type=object(),
        robot_execution=_robot_execution_for_projection_test(
            physx_overrides=_PhysxOverrides(),
            contact_material=_ContactMaterial(),
        ),
        controller_profiles=object(),
        scene_solver=None,
        physics_backend="newton",
        prepare_newton_render_topology=prepare_render_topology,
        defer_articulation_binding=True,
    )

    assert projected == [shared]
    assert render_intents == [
        ("import", prepare_render_topology),
        ("root_pose", prepare_render_topology),
    ]
    assert author_events == [
        "root_pose",
        *(["render_topology"] if prepare_render_topology else []),
        "usd_overrides",
    ]
    assert imported.solver_counts == {"configured": 0}
    assert gravity_backends == ["newton"]


def test_physx_import_projects_robot_leaf_and_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = {"default": RobotUsdOverrideConfig(contact_static_friction=0.7)}
    merged = {
        "default": RobotUsdOverrideConfig(
            contact_static_friction=0.7,
            joint_friction=0.5,
        )
    }

    class _ContactMaterial:
        def apply_to_configs(self, configs: object) -> object:
            assert configs == {"default": RobotUsdOverrideConfig()}
            return shared

    class _PhysxOverrides:
        def apply_to_configs(self, configs: object) -> object:
            assert configs == shared
            return merged

    solver = object()
    scene_solver = object()
    applied_usd: list[object] = []
    applied_solver: list[object] = []
    render_intents: list[tuple[str, bool]] = []
    gravity_backends: list[str] = []
    monkeypatch.setattr(
        setup,
        "import_robot_asset",
        lambda _config, *, physics_backend, prepare_newton_render_topology, root_pose: (
            render_intents.append(("import", prepare_newton_render_topology))
            or (
                "/World/Robot",
                Path(f"robot-{physics_backend}.xml"),
                "/World/Robot",
            )
        ),
    )
    monkeypatch.setattr(
        setup,
        "apply_root_pose",
        lambda *_args, **kwargs: render_intents.append(
            ("root_pose", kwargs["prepare_newton_render_topology"])
        ),
    )
    monkeypatch.setattr(
        setup,
        "prepare_newton_render_subtree",
        lambda **_kwargs: pytest.fail("PhysX must not author Newton render topology"),
    )
    monkeypatch.setattr(
        setup,
        "robot_usd_override_configs",
        lambda _profiles: {"default": RobotUsdOverrideConfig()},
    )
    monkeypatch.setattr(
        setup,
        "apply_robot_usd_overrides",
        lambda _root, configs, **_kwargs: applied_usd.append(configs),
    )
    monkeypatch.setattr(setup, "merge_solver_configs", lambda _scene, _robot: solver)
    monkeypatch.setattr(
        setup,
        "apply_solver_iteration_overrides",
        lambda _stage, _root, config, **_kwargs: (
            applied_solver.append(config) or {"configured": 1}
        ),
    )
    monkeypatch.setattr(
        setup,
        "apply_robot_gravity_policy",
        lambda *_args, **kwargs: (
            gravity_backends.append(kwargs["physics_backend"]) or {"configured": 0}
        ),
    )

    imported = setup.import_execution_robot_to_stage(
        world=object(),
        stage=object(),
        single_articulation_type=object(),
        robot_execution=_robot_execution_for_projection_test(
            physx_overrides=_PhysxOverrides(),
            contact_material=_ContactMaterial(),
        ),
        controller_profiles=object(),
        scene_solver=scene_solver,
        physics_backend="physx",
        prepare_newton_render_topology=False,
        defer_articulation_binding=True,
    )

    assert applied_usd == [merged]
    assert applied_solver == [solver]
    assert render_intents == [("import", False), ("root_pose", False)]
    assert imported.solver_counts == {"configured": 1}
    assert gravity_backends == ["physx"]


def test_finalize_robot_warns_and_skips_unsupported_newton_link_gravity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_controller(monkeypatch)
    articulation = _Articulation(supports_per_link_gravity=False)

    with pytest.warns(PhysicsCompatibilityWarning) as caught:
        prepared = setup.finalize_robot_controller(
            imported=_imported(articulation),
            controller_profiles=object(),
            control_mode="position",
        )

    assert prepared.articulation is articulation
    assert articulation.disable_gravity_calls == 0
    np.testing.assert_array_equal(articulation.velocities, [0.0, 0.0])
    assert _JointController.instances[0].configure_calls == 1
    warning = caught[0].message
    assert warning.backend == "newton"
    assert warning.feature == "robot per-link gravity"
    assert warning.skipped_fields == ("robot.physics.gravity",)


@pytest.mark.parametrize("capability", [None, True])
def test_finalize_robot_preserves_legacy_gravity_behavior(
    monkeypatch: pytest.MonkeyPatch,
    capability: bool | None,
) -> None:
    _install_controller(monkeypatch)
    articulation = _Articulation(supports_per_link_gravity=capability)

    setup.finalize_robot_controller(
        imported=_imported(articulation),
        controller_profiles=object(),
        control_mode="position",
    )

    assert articulation.disable_gravity_calls == 1


def test_finalize_robot_accepts_projected_newton_gravity_compensation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_controller(monkeypatch)
    articulation = _Articulation(supports_per_link_gravity=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        setup.finalize_robot_controller(
            imported=_imported(
                articulation,
                gravity_counts={
                    "enabled": 0,
                    "disabled": 2,
                    "newton_gravcomp": 2,
                },
            ),
            controller_profiles=object(),
            control_mode="position",
        )

    assert not [
        item for item in caught if isinstance(item.message, PhysicsCompatibilityWarning)
    ]
    assert articulation.disable_gravity_calls == 0


def test_bind_deferred_single_articulation_after_direct_initialization() -> None:
    imported = _imported(None)
    created = []

    def articulation_type(*, prim_path: str, name: str) -> object:
        created.append((prim_path, name))
        articulation = _Articulation(supports_per_link_gravity=False)
        articulation.requires_scene_registration = False
        return articulation

    bound = setup.bind_imported_robot_articulation(
        imported,
        world=object(),
        single_articulation_type=articulation_type,
        name="left",
    )

    assert imported.articulation is None
    assert isinstance(bound.articulation, _Articulation)
    assert created == [("/World/Robot", "left")]


def test_finalize_robot_rejects_unbound_direct_articulation() -> None:
    with pytest.raises(RuntimeError, match="must be bound before controller"):
        setup.finalize_robot_controller(
            imported=_imported(None),
            controller_profiles=object(),
            control_mode="position",
        )
