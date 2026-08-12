from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.controllers.control_mode import ControlModeIncompatibleError
from linkerbot_sim.mirror.motion.request_parser import (
    parse_mirror_motion_request,
)
from linkerbot_sim.mirror.motion.backend import MirrorTimelineBackend
from linkerbot_sim.configuration import MirrorConfig, load_mirror_config
from linkerbot_sim.mirror.motion.timeline.builders import (
    duration_to_ticks,
    make_goal_segment,
    sequential_group_track,
    sequential_robot_track,
)
from linkerbot_sim.mirror.motion.timeline.executor import (
    TimelineExecutionInterrupted,
    TimelinePostStepError,
    execute_robot_timeline,
)
from linkerbot_sim.mirror.rendering import CameraBundle, RenderCoordinator
from linkerbot_sim.mirror.timing import WallClockStepSynchronizer
from linkerbot_sim.snapshots.transactions import RuntimeMutationRejected
from linkerbot_sim.mirror.motion.timeline.model import (
    RobotMotionUnit,
    RobotTimeline,
)
from linkerbot_sim.mirror.motion.timeline.compiler import (
    TimelinePlanningError,
    TimelinePlanningLocation,
    TimelinePlanningSession,
)
from linkerbot_sim.mirror.robots import RobotRegistry
from linkerbot_sim.mirror.collision.registry import SceneCollisionRegistry
from linkerbot_sim.assets.root_pose import RootPoseConfig
from linkerbot_sim.controllers.types import ControlTargets
from linkerbot_sim.robots.capabilities import PlanningCapability, RobotKind
from linkerbot_sim.robots.joint_groups import JointGroupLayout


_MIRROR_CONFIG = load_mirror_config()


def _parse_motion(
    message: Mapping[str, object],
    *,
    config: MirrorConfig = _MIRROR_CONFIG,
    allow_effort: bool = False,
):
    """把测试用紧凑 mapping 投影成正式 Mirror v1 motion 调用。"""

    payload = dict(message)
    kind = str(payload.pop("type"))
    request_id = str(payload.pop("id", f"test-{kind}"))
    return parse_mirror_motion_request(
        f"motion.{kind}",
        payload,
        request_id=request_id,
        config=config,
        allow_effort=allow_effort,
    )


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
        self.target_log: list[ControlTargets] = []

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
        self.target_log.append(
            ControlTargets(targets.positions, targets.velocities, targets.efforts)
        )


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
        physics=world,
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

    paced_dt: list[float] = []
    step = execute_robot_timeline(
        runtime,
        timeline,
        before_step=paced_dt.append,
    )

    assert step == 4
    assert runtime.physics.step_calls == 4
    assert paced_dt == pytest.approx([0.1] * 4)
    assert len(first.execution.joint_controller.apply_log) == 5
    assert len(second.execution.joint_controller.apply_log) == 5
    np.testing.assert_allclose(first.execution.articulation.positions, [1.0, 0.0])
    np.testing.assert_allclose(second.execution.articulation.positions, [0.0, 2.0])


def test_timeline_uses_one_newton_snapshot_and_each_camera_render_budget() -> None:
    runtime = _runtime(1)
    events: list[str] = []

    class DirectPhysics(_World):
        def step(self, *, render: bool = False) -> None:
            events.append(f"step:{render}")
            super().step(render=render)

        def pre_render(self) -> None:
            events.append("pre_render")

        def render_update(self) -> None:
            events.append("render_update")

        def render(self) -> None:
            raise AssertionError("timeline must use the split direct render contract")

    class DirectCamera:
        def __init__(self, name: str, count: int) -> None:
            self.name = name
            self.render_update_count = count

        def set_render_active(self, active: bool) -> None:
            events.append(f"active:{self.name}:{active}")

    physics = DirectPhysics()
    runtime.physics = physics
    first = DirectCamera("first", 2)
    second = DirectCamera("second", 3)
    coordinator = RenderCoordinator(
        physics_runtime=physics,
        cameras=CameraBundle(cameras=(first, second)),
    )
    track = sequential_robot_track(
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
                                target_positions=(0.2,),
                                duration_s=0.1,
                                physics_dt=0.1,
                            ),
                        ),
                    ),
                )
            ),
        ),
    )

    assert (
        execute_robot_timeline(
            runtime,
            RobotTimeline((track,), physics_dt=0.1),
            render_frame=coordinator.render_only,
        )
        == 1
    )
    assert events == [
        "step:False",
        "pre_render",
        "active:first:True",
        "active:second:False",
        "render_update",
        "render_update",
        "active:first:False",
        "active:second:True",
        "render_update",
        "render_update",
        "render_update",
        "active:first:True",
        "active:second:True",
    ]


def test_timeline_backend_rebases_step_and_holds_all_robots_after_reset() -> None:
    runtime = _runtime(2)
    backend = MirrorTimelineBackend(runtime, config=_MIRROR_CONFIG)
    renders: list[str] = []
    backend.bind_render_frame(lambda: renders.append("render"))

    assert backend.after_scene_reset(hold_duration_s=0.2) == 2
    assert backend.step_count == 2
    assert runtime.physics.step_calls == 2
    assert renders == ["render", "render"]
    for robot in runtime.robots_by_id.values():
        assert len(robot.execution.joint_controller.apply_log) == 3

    assert backend.after_scene_reset(hold_duration_s=None) == 0
    assert backend.step_count == 0
    assert runtime.physics.step_calls == 2


def test_timeline_backend_commits_interrupted_motion_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(1)
    backend = MirrorTimelineBackend(runtime, config=_MIRROR_CONFIG)
    backend.bind_render_frame(lambda: None)

    def interrupt(*_args, **_kwargs):
        raise TimelineExecutionInterrupted("cancelled", step=7)

    monkeypatch.setattr(
        "linkerbot_sim.mirror.motion.backend.execute_robot_timeline",
        interrupt,
    )

    with pytest.raises(TimelineExecutionInterrupted, match="cancelled"):
        backend.execute(
            "motion.hold",
            {"robot_id": 0, "duration_s": 0.1},
            request_id="cancel-after-seven",
            should_cancel=lambda: False,
        )

    assert backend.step_count == 7


def test_timeline_backend_passes_its_bound_synchronizer_to_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(1)
    backend = MirrorTimelineBackend(runtime, config=_MIRROR_CONFIG)
    backend.bind_render_frame(lambda: None)
    synchronizer = WallClockStepSynchronizer(enabled=False)
    backend.bind_step_synchronizer(synchronizer)
    received_callbacks = []

    def execute(_runtime, _timeline, **kwargs):
        received_callbacks.append(kwargs["before_step"])
        return 1

    monkeypatch.setattr(
        "linkerbot_sim.mirror.motion.backend.execute_robot_timeline",
        execute,
    )

    backend.execute(
        "motion.hold",
        {"robot_id": 0, "duration_s": 0.1},
        request_id="paced-hold",
        should_cancel=lambda: False,
    )

    assert len(received_callbacks) == 1
    assert received_callbacks[0].__self__ is synchronizer


def test_timeline_backend_commits_reset_hold_post_step_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(1)
    backend = MirrorTimelineBackend(runtime, config=_MIRROR_CONFIG)
    backend.bind_render_frame(lambda: None)

    def fail_after_step(*_args, **_kwargs):
        raise TimelinePostStepError("observer failed", step=1)

    monkeypatch.setattr(
        "linkerbot_sim.mirror.motion.backend.execute_robot_timeline",
        fail_after_step,
    )

    with pytest.raises(TimelinePostStepError, match="observer failed"):
        backend.after_scene_reset(hold_duration_s=0.1)

    assert backend.step_count == 1


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
    assert runtime.physics.step_calls == 1
    assert dirty_calls == 1
    terminal = runtime.robots_by_id[0].execution.joint_controller.target_log[-1]
    np.testing.assert_allclose(terminal.velocities, 0.0)
    np.testing.assert_allclose(terminal.efforts, 0.0)


def test_effort_timeline_cancellation_neutralizes_active_target() -> None:
    runtime = _runtime(1)
    request = _parse_motion(
        {
            "type": "joint_effort",
            "robot_id": 0,
            "joint_efforts": [2.5],
            "duration_s": 0.3,
        },
        allow_effort=True,
    )
    timeline = TimelinePlanningSession(runtime).compile(request)
    checks = 0

    def should_stop() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    with pytest.raises(TimelineExecutionInterrupted):
        execute_robot_timeline(runtime, timeline, should_stop=should_stop)

    targets = runtime.robots_by_id[0].execution.joint_controller.target_log
    assert len(targets) == 2
    assert targets[0].efforts[0] == pytest.approx(2.5)
    np.testing.assert_allclose(targets[-1].velocities, 0.0)
    np.testing.assert_allclose(targets[-1].efforts, 0.0)


def test_effort_timeline_post_step_error_neutralizes_active_target() -> None:
    runtime = _runtime(1)

    class FailingObserver:
        def observe(self, *_args, **_kwargs) -> None:
            raise RuntimeError("observer failed")

    runtime.state_observer = FailingObserver()
    request = _parse_motion(
        {
            "type": "joint_effort",
            "robot_id": 0,
            "joint_efforts": [2.5],
            "duration_s": 0.2,
        },
        allow_effort=True,
    )
    timeline = TimelinePlanningSession(runtime).compile(request)

    with pytest.raises(TimelinePostStepError, match="observer failed"):
        execute_robot_timeline(runtime, timeline)

    targets = runtime.robots_by_id[0].execution.joint_controller.target_log
    assert targets[0].efforts[0] == pytest.approx(2.5)
    np.testing.assert_allclose(targets[-1].velocities, 0.0)
    np.testing.assert_allclose(targets[-1].efforts, 0.0)


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
    command = _parse_motion(message)

    timeline = TimelinePlanningSession(runtime).compile(command)

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
    command = _parse_motion(message)

    timeline = TimelinePlanningSession(runtime).compile(command)

    assert registry.snapshot_calls == 0
    assert registry.dirty is True
    assert timeline.scene_version == registry.version
    assert timeline.metadata == {"command_id": message["id"]}


@pytest.mark.parametrize(
    "joint_efforts",
    ({"a0": 2.5}, [2.5]),
    ids=("named", "flat"),
)
def test_v2_effort_segment_is_constant_and_terminal_is_neutral(
    joint_efforts: object,
) -> None:
    runtime = _runtime(1)
    registry = _CountingCollisionRegistry()
    runtime.collision_registry = registry
    request = _parse_motion(
        {
            "type": "joint_effort",
            "id": "effort",
            "robot_id": 0,
            "group": "arm",
            "joint_efforts": joint_efforts,
            "duration_s": 0.2,
            "phase": "contact_push",
        },
        allow_effort=True,
    )

    timeline = TimelinePlanningSession(runtime).compile(request)
    segment = timeline.tracks[0].units[0].group_tracks[0].segments[0]

    assert registry.snapshot_calls == 0
    np.testing.assert_allclose(segment.positions, [[0.0], [0.0]])
    np.testing.assert_allclose(segment.velocities, [[0.0], [0.0]])
    np.testing.assert_allclose(segment.efforts, [[2.5], [2.5]])
    terminal = segment.terminal_sample()
    assert terminal is not None
    np.testing.assert_allclose(terminal.efforts, [0.0])

    assert execute_robot_timeline(runtime, timeline) == 2
    targets = runtime.robots_by_id[0].execution.joint_controller.target_log
    assert len(targets) == 3
    np.testing.assert_allclose(
        [target.efforts[0] for target in targets], [2.5, 2.5, 0.0]
    )


def test_v1_timeline_cannot_smuggle_effort_segment() -> None:
    with pytest.raises(ValueError, match="require linkerbot.mirror.v2"):
        _parse_motion(
            {
                "type": "plan_timeline",
                "tracks": [
                    {
                        "robot_id": 0,
                        "segments": [
                            {
                                "kind": "joint_effort",
                                "joint_efforts": [1.0],
                                "duration_s": 0.1,
                            }
                        ],
                    }
                ],
            }
        )


@pytest.mark.parametrize("invalid", [True, "1.0", float("nan"), float("inf")])
def test_effort_request_rejects_non_numeric_or_nonfinite_values(
    invalid: object,
) -> None:
    with pytest.raises(ValueError):
        _parse_motion(
            {
                "type": "joint_effort",
                "robot_id": 0,
                "joint_efforts": [invalid],
                "duration_s": 0.1,
            },
            allow_effort=True,
        )


def test_mode_compatibility_fails_before_collision_snapshot_or_planner() -> None:
    runtime = _runtime(1)
    registry = _CountingCollisionRegistry()
    registry.mark_dirty()
    runtime.collision_registry = registry
    backend = MirrorTimelineBackend(runtime, config=_MIRROR_CONFIG)
    backend.bind_render_frame(lambda: None)
    backend.bind_control_mode_provider(lambda: "effort")

    with pytest.raises(ControlModeIncompatibleError) as captured:
        backend.execute(
            "motion.plan_cspace_goal",
            {
                "robot_id": 0,
                "joint_positions": [0.5],
                "duration_s": 0.1,
                "avoid_collisions": False,
            },
            request_id="incompatible-plan",
            should_cancel=lambda: False,
            protocol="linkerbot.mirror.v2",
        )

    assert registry.snapshot_calls == 0
    assert captured.value.details == {
        "active_mode": "effort",
        "operation": "motion.plan_cspace_goal",
        "location": {
            "track_index": 0,
            "unit_index": 0,
            "robot_id": 0,
            "robot_label": None,
            "group": "arm",
            "segment_index": 0,
            "segment_kind": "plan_cspace_goal",
        },
    }


def test_position_mode_rejects_effort_before_any_physics_write() -> None:
    runtime = _runtime(1)
    backend = MirrorTimelineBackend(runtime, config=_MIRROR_CONFIG)
    backend.bind_render_frame(lambda: None)
    backend.bind_control_mode_provider(lambda: "position")

    with pytest.raises(ControlModeIncompatibleError):
        backend.execute(
            "motion.joint_effort",
            {
                "robot_id": 0,
                "joint_efforts": [1.0],
                "duration_s": 0.1,
            },
            request_id="position-effort",
            should_cancel=lambda: False,
            protocol="linkerbot.mirror.v2",
        )

    assert runtime.physics.step_calls == 0
    assert runtime.robots_by_id[0].execution.joint_controller.apply_log == []


def test_multiple_planning_segments_share_one_collision_snapshot() -> None:
    runtime = _runtime(1)
    registry = _CountingCollisionRegistry()
    runtime.collision_registry = registry
    command = _parse_motion(
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
                            "avoid_collisions": False,
                        },
                        {
                            "kind": "plan_cspace_delta",
                            "joint_deltas": [0.25],
                            "duration_s": 0.1,
                            "avoid_collisions": False,
                        },
                    ],
                }
            ],
        }
    )

    timeline = TimelinePlanningSession(runtime, planner_backend="linear").compile(
        command
    )

    assert registry.snapshot_calls == 1
    assert registry.dirty is False
    assert timeline.scene_version == registry.version
    assert "scene_fingerprint" in timeline.metadata


def test_scene_linear_backend_compiles_cspace_request_without_curobo() -> None:
    runtime = _runtime(1)
    command = _parse_motion(
        {
            "type": "plan_cspace_goal",
            "robot_id": 0,
            "joint_positions": [1.0],
            "duration_s": 0.2,
            "sample_dt_s": 0.05,
            "avoid_collisions": False,
        }
    )

    timeline = TimelinePlanningSession(runtime, planner_backend="linear").compile(
        command
    )
    segment = timeline.tracks[0].units[0].group_tracks[0].segments[0]

    assert timeline.duration_ticks == 2
    assert runtime.collision_registry.dirty is False
    assert "scene_fingerprint" in timeline.metadata
    assert segment.metadata["backend"] == "linear"
    assert segment.metadata["sample_dt_s"] == 0.05
    np.testing.assert_allclose(segment.positions, [[0.5], [1.0]])


def test_scene_planner_request_uses_strict_planning_period_and_timeout() -> None:
    runtime = _runtime(1)
    command = _parse_motion(
        {
            "type": "plan_cspace_goal",
            "robot_id": 0,
            "joint_positions": [1.0],
            "duration_s": 0.2,
            "avoid_collisions": False,
        }
    )

    timeline = TimelinePlanningSession(runtime, planner_backend="linear").compile(
        command
    )
    segment = timeline.tracks[0].units[0].group_tracks[0].segments[0]

    assert command.tracks[0].units[0].group_tracks[0].segments[0].timeout_s == 30.0
    assert segment.metadata["sample_dt_s"] == pytest.approx(0.02)


def test_scene_joint_interpolation_default_changes_compiled_samples() -> None:
    runtime = _runtime(1)
    command = _parse_motion(
        {
            "type": "joint_goal",
            "robot_id": 0,
            "joint_positions": [1.0],
            "duration_s": 0.3,
        },
        config=replace(
            _MIRROR_CONFIG,
            control=replace(
                _MIRROR_CONFIG.control,
                joint_interpolation="linear",
            ),
        ),
    )

    timeline = TimelinePlanningSession(runtime).compile(command)
    segment = timeline.tracks[0].units[0].group_tracks[0].segments[0]

    assert segment.metadata["interpolation"] == "linear"
    np.testing.assert_allclose(segment.positions[:, 0], [1.0 / 3.0, 2.0 / 3.0, 1.0])
