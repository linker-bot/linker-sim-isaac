from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.app.interactive.single_scene.protocol import (
    parse_interactive_motion_message,
)
from linkerbot_sim.app.motion.timeline.builders import (
    duration_to_ticks,
    make_goal_segment,
    sequential_group_track,
    sequential_robot_track,
)
from linkerbot_sim.app.motion.timeline.executor import (
    TimelinePostStepError,
    execute_robot_timeline,
)
from linkerbot_sim.snapshots.transactions import RuntimeMutationRejected
from linkerbot_sim.app.motion.timeline.model import (
    RobotMotionUnit,
    RobotTimeline,
)
from linkerbot_sim.app.motion.timeline.compiler import (
    TimelinePlanningError,
    TimelinePlanningLocation,
    TimelinePlanningSession,
)
from linkerbot_sim.app.runtime.robot_registry import RobotRegistry
from linkerbot_sim.app.runtime.collision.registry import SceneCollisionRegistry
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.controllers.types import ControlTargets
from linkerbot_sim.configs.runtime import RuntimeCommandDefaults
from linkerbot_sim.robots.capabilities import PlanningCapability, RobotKind
from linkerbot_sim.robots.joint_groups import JointGroupLayout


class _Articulation:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.dof_names = names
        self.num_dof = len(names)
        self.positions = np.zeros(len(names), dtype=float)
        self.zero_velocity_calls = 0

    def get_joint_positions(self):
        return self.positions.copy()

    def set_joint_velocities(self, values) -> None:
        self.zero_velocity_calls += 1


class _Controller:
    def __init__(self, articulation: _Articulation) -> None:
        self.articulation = articulation
        self.command_joint_names = articulation.dof_names
        self.command_indices = np.arange(articulation.num_dof, dtype=int)
        self.driven_indices = self.command_indices.copy()
        self.apply_log: list[np.ndarray] = []

    def build_control_targets(
        self,
        command_positions,
        command_velocities,
        command_efforts,
        *,
        base_positions,
    ) -> ControlTargets:
        return ControlTargets(
            np.asarray(command_positions, dtype=float),
            np.asarray(command_velocities, dtype=float),
            np.asarray(command_efforts, dtype=float),
        )

    def apply_targets(self, action_type, targets: ControlTargets) -> None:
        self.articulation.positions = targets.positions.copy()
        self.apply_log.append(targets.positions.copy())


class _World:
    def __init__(self, dt: float = 0.1) -> None:
        self.dt = dt
        self.step_calls = 0

    def get_physics_dt(self) -> float:
        return self.dt

    def step(self, *, render: bool = False) -> None:
        self.step_calls += 1


class _CountingCollisionRegistry(SceneCollisionRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_calls = 0

    def snapshot(self, *, force: bool = False):
        self.snapshot_calls += 1
        return super().snapshot(force=force)


def _robot(robot_id: int, label: str):
    names = (f"a{robot_id}", f"h{robot_id}")
    articulation = _Articulation(names)
    controller = _Controller(articulation)
    execution = SimpleNamespace(
        articulation=articulation,
        joint_controller=controller,
        articulation_action_type=object,
        render_enabled=False,
        drive_logger=None,
        state_observer=None,
        camera_observer=None,
    )
    return SimpleNamespace(
        robot_id=robot_id,
        label=label,
        kind=RobotKind.ARM_HAND,
        profile_name="profile",
        execution=execution,
        joint_groups=JointGroupLayout(
            names,
            arm=(names[0],),
            hand=(names[1],),
        ),
        planning_capability=PlanningCapability(
            RobotKind.ARM_HAND,
            False,
            None,
            False,
            False,
        ),
        scene_instance=SimpleNamespace(root_pose=RootPoseConfig()),
    )


def _runtime(count: int = 2):
    robots = tuple(_robot(index, f"robot_{index}") for index in range(count))
    registry = RobotRegistry(robots)
    world = _World()
    return SimpleNamespace(
        world=world,
        session=SimpleNamespace(app=None),
        robot_registry=registry,
        robots_by_id=registry.robots_by_id,
        robot_id_by_label=registry.robot_id_by_label,
        collision_registry=SceneCollisionRegistry(),
        camera_output=None,
    )


def test_duration_rounds_up_and_zero_motion_is_rejected() -> None:
    assert duration_to_ticks(0.0, 0.1) == 0
    assert duration_to_ticks(0.100001, 0.1) == 2
    with pytest.raises(ValueError, match="zero-duration"):
        make_goal_segment(
            joint_names=("a",),
            start_positions=(0.0,),
            target_positions=(1.0,),
            duration_s=0.0,
            physics_dt=0.1,
        )


def test_timeline_planning_error_includes_underlying_cause() -> None:
    error = TimelinePlanningError(
        "segment planning failed",
        location=TimelinePlanningLocation(0, "left_arm", 0, 0, "arm", 0),
        cause=TypeError("invalid planner config"),
    )

    assert (
        str(error)
        == "segment planning failed; robot_id=0 label='left_arm' track=0 unit=0 "
        "group=arm segment=0; cause=TypeError: invalid planner config"
    )


def test_multi_robot_executor_applies_all_targets_before_one_world_step() -> None:
    runtime = _runtime(2)
    first = runtime.robots_by_id[0]
    second = runtime.robots_by_id[1]
    track0 = sequential_robot_track(
        0,
        (
            RobotMotionUnit(
                (
                    sequential_group_track(
                        "arm",
                        (
                            make_goal_segment(
                                joint_names=("a0",),
                                start_positions=(0.0,),
                                target_positions=(1.0,),
                                duration_s=0.2,
                                physics_dt=0.1,
                            ),
                        ),
                    ),
                )
            ),
        ),
    )
    track1 = sequential_robot_track(
        1,
        (
            RobotMotionUnit(
                (
                    sequential_group_track(
                        "hand",
                        (
                            make_goal_segment(
                                joint_names=("h1",),
                                start_positions=(0.0,),
                                target_positions=(2.0,),
                                duration_s=0.4,
                                physics_dt=0.1,
                            ),
                        ),
                    ),
                )
            ),
        ),
    )
    timeline = RobotTimeline((track0, track1), physics_dt=0.1)

    step = execute_robot_timeline(runtime, timeline)

    assert step == 4
    assert runtime.world.step_calls == 4
    assert len(first.execution.joint_controller.apply_log) == 4
    assert len(second.execution.joint_controller.apply_log) == 4
    np.testing.assert_allclose(first.execution.articulation.positions, [1.0, 0.0])
    np.testing.assert_allclose(second.execution.articulation.positions, [0.0, 2.0])


def test_timeline_post_step_failure_reports_committed_step_and_dirty_scene() -> None:
    runtime = _runtime(1)
    dirty_calls = 0

    class CollisionRegistry:
        def mark_dirty(self) -> None:
            nonlocal dirty_calls
            dirty_calls += 1

    class FailingObserver:
        def observe(self, *_args, **_kwargs) -> None:
            raise RuntimeError("observer failed")

    runtime.collision_registry = CollisionRegistry()
    runtime.state_observer = FailingObserver()
    timeline = RobotTimeline(
        (
            sequential_robot_track(
                0,
                (
                    RobotMotionUnit(
                        (
                            sequential_group_track(
                                "arm",
                                (
                                    make_goal_segment(
                                        joint_names=("a0",),
                                        start_positions=(0.0,),
                                        target_positions=(1.0,),
                                        duration_s=0.1,
                                        physics_dt=0.1,
                                    ),
                                ),
                            ),
                        )
                    ),
                ),
            ),
        ),
        physics_dt=0.1,
    )

    with pytest.raises(TimelinePostStepError, match="observer failed") as exc_info:
        execute_robot_timeline(runtime, timeline)

    assert exc_info.value.step == 1
    assert runtime.world.step_calls == 1
    assert dirty_calls == 1


def test_timeline_rejects_fatal_runtime_before_accessing_world() -> None:
    runtime = SimpleNamespace(fatal_error="snapshot rollback failed")

    with pytest.raises(RuntimeMutationRejected, match="requires rebuild"):
        execute_robot_timeline(runtime, object())


def test_protocol_and_planner_compile_arm_hand_tracks_by_robot_id() -> None:
    runtime = _runtime(1)
    message = {
        "type": "plan_timeline",
        "tracks": [
            {
                "robot_id": 0,
                "robot_label": "robot_0",
                "units": [
                    {
                        "group_tracks": [
                            {
                                "group": "arm",
                                "segments": [
                                    {
                                        "kind": "joint_goal",
                                        "joint_positions": [1.0],
                                        "duration_s": 0.2,
                                    }
                                ],
                            },
                            {
                                "group": "hand",
                                "segments": [
                                    {"kind": "hold", "duration_s": 0.1},
                                    {
                                        "kind": "joint_goal",
                                        "joint_positions": [2.0],
                                        "duration_s": 0.3,
                                    },
                                ],
                            },
                        ]
                    }
                ],
            }
        ],
    }
    command = parse_interactive_motion_message(message)

    timeline = TimelinePlanningSession(runtime).compile(command.timeline)

    assert timeline.duration_ticks == 4
    assert timeline.scene_version == runtime.collision_registry.version
    assert runtime.collision_registry.dirty is True
    assert "scene_fingerprint" not in timeline.metadata


@pytest.mark.parametrize(
    "message",
    (
        {"type": "hold", "robot_id": 0, "duration_s": 0.1},
        {
            "type": "joint_goal",
            "robot_id": 0,
            "joint_positions": [0.5],
            "duration_s": 0.1,
        },
        {
            "type": "joint_delta",
            "robot_id": 0,
            "joint_deltas": [0.5],
            "duration_s": 0.1,
        },
        {
            "type": "joint_trajectory",
            "robot_id": 0,
            "joint_positions": [[0.25], [0.5]],
            "times_s": [0.1, 0.2],
            "duration_s": 0.2,
        },
    ),
    ids=("hold", "joint_goal", "joint_delta", "joint_trajectory"),
)
def test_direct_segments_do_not_materialize_collision_snapshot(
    message: dict[str, object],
) -> None:
    runtime = _runtime(1)
    registry = _CountingCollisionRegistry()
    registry.mark_dirty()
    runtime.collision_registry = registry
    message = {**message, "id": f"direct-{message['type']}"}
    command = parse_interactive_motion_message(message)

    timeline = TimelinePlanningSession(runtime).compile(command.timeline)

    assert registry.snapshot_calls == 0
    assert registry.dirty is True
    assert timeline.scene_version == registry.version
    assert timeline.metadata == {"command_id": message["id"]}


def test_multiple_planning_segments_share_one_collision_snapshot() -> None:
    runtime = _runtime(1)
    registry = _CountingCollisionRegistry()
    runtime.collision_registry = registry
    command = parse_interactive_motion_message(
        {
            "type": "plan_timeline",
            "id": "two-plans",
            "tracks": [
                {
                    "robot_id": 0,
                    "segments": [
                        {
                            "kind": "plan_cspace_goal",
                            "joint_positions": [0.5],
                            "duration_s": 0.1,
                        },
                        {
                            "kind": "plan_cspace_delta",
                            "joint_deltas": [0.25],
                            "duration_s": 0.1,
                        },
                    ],
                }
            ],
        }
    )

    timeline = TimelinePlanningSession(runtime, planner_backend="linear").compile(
        command.timeline
    )

    assert registry.snapshot_calls == 1
    assert registry.dirty is False
    assert timeline.scene_version == registry.version
    assert "scene_fingerprint" in timeline.metadata


def test_scene_linear_backend_compiles_cspace_request_without_curobo() -> None:
    runtime = _runtime(1)
    command = parse_interactive_motion_message(
        {
            "type": "plan_cspace_goal",
            "robot_id": 0,
            "joint_positions": [1.0],
            "duration_s": 0.2,
            "sample_dt_s": 0.05,
        }
    )

    timeline = TimelinePlanningSession(runtime, planner_backend="linear").compile(
        command.timeline
    )
    segment = timeline.tracks[0].units[0].group_tracks[0].segments[0]

    assert timeline.duration_ticks == 2
    assert runtime.collision_registry.dirty is False
    assert "scene_fingerprint" in timeline.metadata
    assert segment.metadata["backend"] == "linear"
    assert segment.metadata["sample_dt_s"] == 0.05
    np.testing.assert_allclose(segment.positions, [[0.5], [1.0]])


def test_scene_planner_request_defaults_sample_dt_to_world_physics_dt() -> None:
    runtime = _runtime(1)
    command = parse_interactive_motion_message(
        {
            "type": "plan_cspace_goal",
            "robot_id": 0,
            "joint_positions": [1.0],
            "duration_s": 0.2,
        }
    )

    timeline = TimelinePlanningSession(runtime, planner_backend="linear").compile(
        command.timeline
    )
    segment = timeline.tracks[0].units[0].group_tracks[0].segments[0]

    assert segment.metadata["sample_dt_s"] == pytest.approx(0.1)


def test_scene_joint_interpolation_default_changes_compiled_samples() -> None:
    runtime = _runtime(1)
    command = parse_interactive_motion_message(
        {
            "type": "joint_goal",
            "robot_id": 0,
            "joint_positions": [1.0],
            "duration_s": 0.3,
        },
        command_defaults=RuntimeCommandDefaults(joint_interpolation="linear"),
    )

    timeline = TimelinePlanningSession(runtime).compile(command.timeline)
    segment = timeline.tracks[0].units[0].group_tracks[0].segments[0]

    assert segment.metadata["interpolation"] == "linear"
    np.testing.assert_allclose(segment.positions[:, 0], [1.0 / 3.0, 2.0 / 3.0, 1.0])
