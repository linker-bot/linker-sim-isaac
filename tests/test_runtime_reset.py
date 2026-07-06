from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from linkerbot_sim.app.runtime.reset import reset_dual_robot_runtime
from linkerbot_sim.assets.robot_loader import RootPoseConfig


class _PhysicsContext:
    def __init__(self) -> None:
        self.gravity = None

    def set_gravity(self, value) -> None:
        self.gravity = value


class _World:
    def __init__(self) -> None:
        self.reset_count = 0
        self.physics_context = _PhysicsContext()

    def reset(self) -> None:
        self.reset_count += 1

    def get_physics_context(self):
        return self.physics_context


class _GravityPolicy:
    def disables_all_known_components(self) -> bool:
        return True


class _Articulation:
    def __init__(self) -> None:
        self.num_dof = 2
        self.gravity_disabled = False
        self.velocities = None

    def disable_gravity(self) -> None:
        self.gravity_disabled = True

    def set_joint_velocities(self, values) -> None:
        self.velocities = np.asarray(values, dtype=float)


class _Controller:
    def __init__(self) -> None:
        self.configure_count = 0
        self.last_commanded_efforts = np.asarray([1.0, 2.0], dtype=float)

    def configure_runtime(self) -> None:
        self.configure_count += 1


class _Observer:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _DualConfig:
    def __init__(self) -> None:
        self.root_poses = {
            "left": RootPoseConfig(xyz=(1.0, 0.0, 0.0)),
            "right": RootPoseConfig(xyz=(-1.0, 0.0, 0.0)),
        }

    def side(self, side: str):
        return SimpleNamespace(root_pose=self.root_poses[side])


def test_reset_dual_robot_runtime_restores_runtime_state() -> None:
    world = _World()
    left_articulation = _Articulation()
    right_articulation = _Articulation()
    left_controller = _Controller()
    right_controller = _Controller()
    state_observer = _Observer()
    camera_observer = _Observer()
    root_pose_calls = []

    runtime = SimpleNamespace(
        session=SimpleNamespace(stage=object(), world=world),
        env_config={
            "env": {
                "gravity_z": -3.0,
                "physics_frequency": 240.0,
                "render_frequency": 60.0,
            },
            "robots": {},
        },
        object_handles=(),
        imported={
            "left": SimpleNamespace(imported_root_path="/World/Left"),
            "right": SimpleNamespace(imported_root_path="/World/Right"),
        },
        dual_config=_DualConfig(),
        prepared={
            "left": SimpleNamespace(
                articulation=left_articulation,
                joint_controller=left_controller,
                gravity_policy=_GravityPolicy(),
            ),
            "right": SimpleNamespace(
                articulation=right_articulation,
                joint_controller=right_controller,
                gravity_policy=_GravityPolicy(),
            ),
        },
        execution=SimpleNamespace(
            state_observer=state_observer,
            camera_observer=camera_observer,
        ),
    )

    result = reset_dual_robot_runtime(
        runtime,
        robot_root_pose_applier=lambda stage, path, pose: root_pose_calls.append(
            (path, pose.xyz)
        ),
    )

    assert result.step == 0
    assert world.reset_count == 1
    assert world.physics_context.gravity == -3.0
    assert root_pose_calls == [
        ("/World/Left", (1.0, 0.0, 0.0)),
        ("/World/Right", (-1.0, 0.0, 0.0)),
    ]
    np.testing.assert_allclose(left_articulation.velocities, [0.0, 0.0])
    np.testing.assert_allclose(right_articulation.velocities, [0.0, 0.0])
    assert left_articulation.gravity_disabled is True
    assert right_articulation.gravity_disabled is True
    assert left_controller.configure_count == 1
    assert right_controller.configure_count == 1
    assert np.isnan(left_controller.last_commanded_efforts).all()
    assert np.isnan(right_controller.last_commanded_efforts).all()
    assert state_observer.reset_count == 1
    assert camera_observer.reset_count == 1
