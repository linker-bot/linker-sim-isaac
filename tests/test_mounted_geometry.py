from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.backends.curobo.context import CuroboContext
from linkerbot_sim.configuration.objects import RigidObjectPlanningCollisionConfig
from linkerbot_sim.configuration.robots import RobotPlanningCollisionSettings
from linkerbot_sim.mirror.collision.collider_bounds import read_link_collider_boxes
from linkerbot_sim.planning.mounted_geometry import (
    MountedCollisionModel,
    cover_box_with_spheres,
)


def test_box_spheres_cover_volume_including_corners():
    dimensions = (0.024, 0.047, 0.103)
    spheres = cover_box_with_spheres(dimensions)
    # Boundary samples expose uncovered corners in surface-only sphere fits.
    axes = [np.linspace(-length / 2, length / 2, 13) for length in dimensions]
    samples = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 3)
    distance = np.linalg.norm(samples[:, None] - spheres[None, :, :3], axis=-1)
    assert np.all(np.min(distance - spheres[None, :, 3], axis=1) <= 1e-14)
    assert len(spheres) == 8


@pytest.mark.parametrize(
    "size", [(0, 1, 1), (-1, 1, 1), (1, np.nan, 1), (1, 2), (100, 100, 100)]
)
def test_invalid_or_unbounded_sphere_covers_are_rejected(size):
    with pytest.raises(ValueError):
        cover_box_with_spheres(size)


def test_imported_collider_bounds_ignore_render_visibility_and_purpose():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/robot")
    root.AddTranslateOp().Set((4, 5, 6))
    root.AddRotateZOp().Set(90)
    mesh = UsdGeom.Mesh.Define(stage, "/robot/hidden")
    mesh.CreatePointsAttr([(-0.01, -0.02, -0.03), (0.01, 0.02, 0.03)])
    mesh.CreatePurposeAttr("guide")
    mesh.CreateVisibilityAttr("invisible")
    mesh.AddTranslateOp().Set((0, 0, 0.1))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    # Unrelated visible geometry must not inflate a physical collision model.
    UsdGeom.Cube.Define(stage, "/robot/visual").CreateSizeAttr(30)
    boxes = read_link_collider_boxes(stage, "/robot", {"robot"})
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0].center, (0, 0, 0.1), atol=1e-8)
    np.testing.assert_allclose(boxes[0].size, (0.02, 0.04, 0.06), atol=1e-8)
    del Gf


def test_mounted_model_refresh_invalidates_every_lazy_solver_once():
    context = CuroboContext.__new__(CuroboContext)
    context._mounted_collision_model = None
    context._ik_solver = SimpleNamespace(destroy=lambda: events.append("ik"))
    context._motion_planner = SimpleNamespace(destroy=lambda: events.append("planner"))
    events = []
    context._make_kinematics = lambda **kw: (
        events.append(kw["mounted_model"].fingerprint) or object()
    )
    model = MountedCollisionModel(0, "flange", ((0.0, 0.0, 0.1, 0.03),), ("hand",))
    context.sync_mounted_geometry(model)
    assert events == [model.fingerprint, "planner", "ik"]
    assert context._ik_solver is None and context._motion_planner is None
    context.sync_mounted_geometry(model)
    assert len(events) == 3
    changed = MountedCollisionModel(0, "flange", ((0.0, 0.0, 0.2, 0.03),), ("hand",))
    context.sync_mounted_geometry(changed)
    assert events[-1] == changed.fingerprint
    assert context._synced_scene_version is None


def test_collision_sources_are_explicit_and_mutually_exclusive():
    assert (
        RobotPlanningCollisionSettings.from_mapping(
            {"mounted_urdf": "robot.urdf"}
        ).mounted_urdf
        == "robot.urdf"
    )
    with pytest.raises(ValueError):
        RobotPlanningCollisionSettings.from_mapping(
            {"mounted_urdf": "robot.urdf", "spheres": []}
        )
    parsed = RigidObjectPlanningCollisionConfig.from_mapping(
        {"source": "colliders"}, label="object"
    )
    assert parsed.source == "colliders"
    with pytest.raises(ValueError):
        RigidObjectPlanningCollisionConfig.from_mapping(
            {"source": "colliders", "size": [1, 1, 1]}, label="object"
        )


def test_path_guard_checks_midpoints_and_keeps_nonallowed_links_active():
    from linkerbot_sim.planning.collision_objects import CollisionObject
    from linkerbot_sim.planning.collision_validation import validate_sampled_path

    obstacle = CollisionObject("post", "cuboid", np.eye(4), (0.1, 0.1, 0.1))

    def spheres(q):
        output = np.zeros((len(q), 2, 4))
        output[:, :, 0] = q[:, :1]
        output[:, :, 3] = 0.02
        return output

    result = validate_sampled_path(
        np.array([[-1.0], [1.0]]),
        sample_spheres=spheres,
        sphere_links=("finger", "camera"),
        obstacles=(obstacle,),
        allowed_pairs=frozenset({("finger", "post")}),
    )
    assert not result["valid"]
    assert result["link_name"] == "camera"
    assert 0 < result["sample_index"] < result["samples"] - 1
    assert result["max_joint_step_rad"] == 0.02


def test_path_guard_allowed_pair_is_specific_to_the_obstacle():
    from linkerbot_sim.planning.collision_objects import CollisionObject
    from linkerbot_sim.planning.collision_validation import validate_sampled_path

    objects = tuple(
        CollisionObject(name, "cuboid", np.eye(4), (0.1, 0.1, 0.1))
        for name in ("grasped", "fixture")
    )
    result = validate_sampled_path(
        np.zeros((1, 1)),
        sample_spheres=lambda q: np.array([[[0, 0, 0, 0.02]]]),
        sphere_links=("finger",),
        obstacles=objects,
        allowed_pairs=frozenset({("finger", "grasped")}),
    )
    assert not result["valid"]
    assert result["geometry_name"] == "fixture"


def test_payload_world_membership_refresh_and_release():
    from linkerbot_sim.mirror.collision.registry import SceneCollisionRegistry
    from linkerbot_sim.planning.collision_objects import CollisionObject

    class Robot:
        label = "arm"

        def mounted_collision_model(self):
            return MountedCollisionModel(
                0,
                "flange",
                ((0.0, 0.0, 0.1, 0.03),),
                ("finger",),
                sphere_links=("finger",),
            )

        def flange_world_pose(self):
            return np.eye(4)

        def collision_objects(self):
            return ()

    registry = SceneCollisionRegistry()
    pose = np.eye(4)
    pose[0, 3] = 0.5
    registry.register_provider("robot", Robot(), owner_robot_id=0, source="robot")
    registry.register_provider(
        "object",
        lambda: (CollisionObject("block", "cuboid", pose, (0.03, 0.03, 0.03)),),
        source="object",
    )
    registry.attach_object("block", 0, touch_links=("finger",))
    attached = registry.snapshot()
    assert not attached.collision_objects_for(0)
    assert len(attached.collision_objects_for(1)) == 1
    assert attached.mounted_models[0].sphere_links == ("finger", "payload:block")
    assert attached.mounted_models[0].payload_collision() is None
    saved = registry.capture_attachments()
    assert registry.prepare_attachments(saved) == (("block", 0, ("finger",)),)
    pose[0, 3] += 0.1
    registry.mark_dirty()
    updated = registry.snapshot()
    assert (
        attached.mounted_models[0].fingerprint != updated.mounted_models[0].fingerprint
    )
    registry.detach_object("block")
    released = registry.snapshot()
    assert released.collision_objects_for(0)[0].name == "block"
    assert released.mounted_models[0].sphere_links == ("finger",)


def test_payload_contact_does_not_disable_camera_collision():
    model = MountedCollisionModel(
        0,
        "flange",
        ((0.0, 0.0, 0.0, 0.02), (0.0, 0.0, 0.0, 0.02), (0.0, 0.0, 0.0, 0.02)),
        ("finger", "camera", "payload:block"),
        sphere_links=("finger", "camera", "payload:block"),
        payload_contacts=(("payload:block", "finger"),),
    )
    assert model.payload_collision()["link_name"] == "camera"


def test_mounted_coverage_reports_both_directions_of_flange_exclusions():
    context = CuroboContext.__new__(CuroboContext)
    context._mounted_collision_model = MountedCollisionModel(
        0, "flange", ((0.0, 0.0, 0.1, 0.03),), ("hand",)
    )
    context._robot_mapping = lambda model: {
        "robot_cfg": {
            "kinematics": {
                "self_collision_ignore": {
                    "link4": ["link5", "flange"],
                    "flange": ["link7"],
                }
            }
        }
    }
    report = context.mounted_geometry_diagnostics()
    assert report["ignored_own_arm_links"] == ["link4", "link7"]
    assert report["internal_mounted_self_collision_checked"] is False
