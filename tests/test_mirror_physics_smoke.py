"""Mirror 真实物理 smoke runner 的纯合同。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.configuration import load_mirror_config
from linkerbot_sim.configuration.scenes import MirrorSceneSettings, ViewportSettings
from linkerbot_sim.mirror.hybrid_parameters import HybridParameterService
from linkerbot_sim.snapshots.schema import (
    ObjectSnapshot,
    RobotSnapshot,
    SceneSnapshot,
    SnapshotRestoreResult,
)
from scripts import smoke_mirror_physics as smoke


def _mirror_generalized_rope_snapshot(
    *,
    owner_q: float = 0.25,
    body_velocity_y: float = 0.0,
) -> SceneSnapshot:
    rope = ObjectSnapshot(
        name="rope",
        positions_local=np.zeros(3),
        orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        linear_velocities=np.asarray([0.0, body_velocity_y, 0.0]),
        angular_velocities=np.zeros(3),
        body_names=("segment",),
        body_positions_local=np.zeros((1, 3)),
        body_orientations_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        body_linear_velocities=np.asarray([[0.0, body_velocity_y, 0.0]]),
        body_angular_velocities=np.zeros((1, 3)),
        generalized_signature=("newton-generalized-state-v1",),
        generalized_q_names=("root|q[0]",),
        generalized_qd_names=("root|qd[0]",),
        generalized_q=np.asarray([owner_q]),
        generalized_qd=np.asarray([0.5]),
    )
    return SceneSnapshot(robots={}, objects={"rope": rope})


class _FakeArticulation:
    def __init__(self, names: tuple[str, ...], offset: float = 0.0) -> None:
        self.dof_names = names
        self.num_dof = len(names)
        self.positions = np.arange(len(names), dtype=float) * 0.1 + offset
        self.velocities = np.zeros(len(names), dtype=float)
        self.position_targets = self.positions.copy()
        self.velocity_targets = self.velocities.copy()
        self.valid = True

    def is_physics_handle_valid(self) -> bool:
        return self.valid

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def set_joint_positions(self, values) -> None:
        self.positions = np.asarray(values, dtype=float).copy()

    def set_joint_velocities(self, values) -> None:
        self.velocities = np.asarray(values, dtype=float).copy()

    def get_joint_position_targets(self, *, joint_indices=None):
        values = self.position_targets
        return values if joint_indices is None else values[np.asarray(joint_indices)]

    def get_joint_velocity_targets(self, *, joint_indices=None):
        values = self.velocity_targets
        return values if joint_indices is None else values[np.asarray(joint_indices)]

    def get_articulation_controller(self):
        return SimpleNamespace(
            get_gains=lambda: (
                np.full(self.num_dof, 200.0, dtype=float),
                np.full(self.num_dof, 20.0, dtype=float),
            )
        )


class _FakeController:
    def __init__(self, articulation: _FakeArticulation) -> None:
        self.articulation = articulation
        self.command_indices = np.arange(articulation.num_dof, dtype=int)
        self.command_joint_names = tuple(articulation.dof_names)
        self.applied = 0
        self.last_targets = None

    def build_control_targets(
        self,
        *,
        command_positions,
        command_velocities,
        command_efforts,
        base_positions,
    ):
        positions = np.asarray(base_positions, dtype=float).copy()
        positions[self.command_indices] = np.asarray(command_positions, dtype=float)
        return SimpleNamespace(
            positions=positions,
            velocities=np.asarray(command_velocities, dtype=float),
            efforts=np.asarray(command_efforts, dtype=float),
        )

    def apply_targets(self, _action_type, targets) -> None:
        self.applied += 1
        self.last_targets = SimpleNamespace(
            positions=np.asarray(targets.positions, dtype=float).copy(),
            velocities=np.asarray(targets.velocities, dtype=float).copy(),
            efforts=np.asarray(targets.efforts, dtype=float).copy(),
        )
        self.articulation.position_targets = np.asarray(
            targets.positions, dtype=float
        ).copy()
        self.articulation.velocity_targets = np.asarray(
            targets.velocities, dtype=float
        ).copy()
        current = self.articulation.positions
        self.articulation.positions = current + 0.5 * (targets.positions - current)

    def snapshot_control_targets_cache(self):
        if self.last_targets is None:
            return None
        return SimpleNamespace(
            positions=self.last_targets.positions.copy(),
            velocities=self.last_targets.velocities.copy(),
            efforts=self.last_targets.efforts.copy(),
        )


class _FakeWorld:
    def __init__(self, manager: _FakeRenderManager | None = None) -> None:
        self.steps = 0
        self.manager = manager

    def step(self, *, render: bool) -> None:
        assert isinstance(render, bool)
        self.steps += 1
        if self.manager is not None:
            self.manager.simulation_time += self.get_physics_dt()

    def get_physics_dt(self) -> float:
        return 1.0 / 120.0


class _FakeRenderManager:
    def __init__(self) -> None:
        self.render_calls = 0
        self.simulation_time = 0.25

    def render(self) -> None:
        self.render_calls += 1


def _camera_frame(
    modality: str,
    data: np.ndarray,
    *,
    intrinsics: np.ndarray | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        camera_name="debug",
        modality=modality,
        data=data,
        intrinsics=np.eye(3) if intrinsics is None else intrinsics,
    )


def _camera_probe_runtime() -> SimpleNamespace:
    camera = SimpleNamespace(
        name="debug",
        settings=SimpleNamespace(
            modalities=("rgb", "depth"),
            resolution=(2, 1),
        ),
    )
    manager = _FakeRenderManager()
    return SimpleNamespace(
        sensor_cameras=(camera,),
        physics=_FakeWorld(manager),
        session=SimpleNamespace(physics_runtime=manager),
    )


class _FakeFk:
    def __init__(self) -> None:
        self.received = None

    def joint_names(self) -> list[str]:
        return ["j1", "j0"]

    def frame_names(self) -> list[str]:
        return ["tool"]

    def compute_pose(self, positions, frame_name: str):
        self.received = np.asarray(positions, dtype=float).copy()
        assert frame_name == "tool"
        return SimpleNamespace(
            position=np.asarray([0.1, 0.2, 0.3]),
            orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3),
        )


class _FakePlanningRegistry:
    def __init__(self, fk: _FakeFk) -> None:
        self.fk = fk
        self.leases: list[tuple[int, str]] = []

    @contextmanager
    def lease(self, robot_id: int, *, consumer_role: str):
        self.leases.append((robot_id, consumer_role))
        yield SimpleNamespace(
            default_tcp_frame="tool",
            make_forward_kinematics=lambda: self.fk,
        )


class _FakeRigidStateView:
    has_live_root = True
    velocity_capability = "complete"
    root_view = object()
    body_view = None
    immutable_position = None

    def __init__(self, *, corrupt_first_probe_readback: bool = False) -> None:
        self.linear = np.asarray([0.4, -0.5, 0.6], dtype=float)
        self.angular = np.asarray([-0.7, 0.8, -0.9], dtype=float)
        self.writes: list[tuple[np.ndarray, np.ndarray]] = []
        self.read_count = 0
        self.corrupt_first_probe_readback = corrupt_first_probe_readback

    def root_velocities(self):
        self.read_count += 1
        if self.corrupt_first_probe_readback and self.read_count == 2:
            return self.angular.copy(), self.linear.copy()
        return self.linear.copy(), self.angular.copy()

    def set_root_velocities(self, linear, angular) -> None:
        self.linear = np.asarray(linear, dtype=float).copy()
        self.angular = np.asarray(angular, dtype=float).copy()
        self.writes.append((self.linear.copy(), self.angular.copy()))


class _FakeRuntime:
    def __init__(self) -> None:
        self.physics = _FakeWorld()
        self.session = SimpleNamespace(stage=SimpleNamespace(Traverse=lambda: ()))
        self.scene = MirrorSceneSettings(
            scene_id="scene3",
            description="Mirror physics smoke fixture",
            gravity_z=-9.81,
            add_ground=False,
            ground_height=0.0,
            physics_frequency_hz=120.0,
            render_frequency_hz=30.0,
            robots=(),
            objects=(),
            cameras=(),
            viewport=ViewportSettings(enabled=False),
            lights=(),
        )
        self.sensor_cameras = ()
        self.fk = _FakeFk()
        self.planning_registry = _FakePlanningRegistry(self.fk)
        self.object_state_views = {"Tblock": _FakeRigidStateView()}
        self.close_calls = 0
        robots = {}
        for robot_id, (label, names) in enumerate(
            (("left", ("j0", "j1")), ("right", ("k0", "k1", "k2")))
        ):
            articulation = _FakeArticulation(names, offset=float(robot_id))
            controller = _FakeController(articulation)
            execution = SimpleNamespace(
                articulation=articulation,
                joint_controller=controller,
                articulation_action_type=object,
            )
            robots[robot_id] = SimpleNamespace(
                robot_id=robot_id,
                label=label,
                execution=execution,
                supports_planning=robot_id == 0,
                curobo_config=object() if robot_id == 0 else None,
            )
        self.robots_by_id = robots

    def close(self):
        self.close_calls += 1
        return SimpleNamespace(stopped=True, live_resources=())


def _get_fake_snapshot(runtime: _FakeRuntime) -> SceneSnapshot:
    robots = {}
    for robot_id, robot in runtime.robots_by_id.items():
        articulation = robot.execution.articulation
        controller = robot.execution.joint_controller
        indices = controller.command_indices
        robots[robot.label] = RobotSnapshot(
            label=robot.label,
            robot_id=robot_id,
            joint_names=controller.command_joint_names,
            joint_positions=articulation.positions[indices],
            joint_velocities=articulation.velocities[indices],
            command_joint_names=controller.command_joint_names,
            command_targets=articulation.position_targets[indices],
        )
    return SceneSnapshot(robots=robots)


def _set_fake_snapshot(
    runtime: _FakeRuntime, snapshot: SceneSnapshot
) -> SnapshotRestoreResult:
    for robot in runtime.robots_by_id.values():
        source = snapshot.robots[robot.label]
        execution = robot.execution
        execution.articulation.set_joint_positions(source.joint_positions)
        execution.articulation.set_joint_velocities(source.joint_velocities)
        execution.articulation.position_targets[
            execution.joint_controller.command_indices
        ] = source.command_targets
        execution.joint_controller.last_targets = SimpleNamespace(
            positions=execution.articulation.position_targets.copy(),
            velocities=execution.articulation.velocity_targets.copy(),
            efforts=np.zeros(execution.articulation.num_dof, dtype=float),
        )
    return SnapshotRestoreResult(
        accepted=True,
        robots=tuple(snapshot.robots),
    )


def test_parse_args_defaults_to_strict_mirror_profile() -> None:
    args = smoke.parse_args([])

    assert args.profile == "physx_cpu"
    assert args.steps == 8
    assert args.control_modes_only is False


def test_parse_args_selects_newton_through_mirror_profile_only() -> None:
    args = smoke.parse_args(
        ["--profile", "newton_cuda", "--steps", "3", "--control-modes-only"]
    )

    assert args.profile == "newton_cuda"
    assert args.steps == 3
    assert args.control_modes_only is True


def test_parse_args_accepts_hybrid_physx_profile() -> None:
    args = smoke.parse_args(["--profile", "physx_cpu_hybrid", "--steps", "2"])

    assert args.profile == "physx_cpu_hybrid"
    assert args.steps == 2


def test_parse_args_rejects_non_positive_steps() -> None:
    with pytest.raises(SystemExit):
        smoke.parse_args(["--steps", "0"])


def test_probe_checks_articulations_targets_snapshot_and_curobo_mapping() -> None:
    runtime = _FakeRuntime()
    initial_left = runtime.robots_by_id[0].execution.articulation.positions.copy()

    report = smoke.probe_mirror_runtime(
        runtime,
        expected_backend="newton",
        steps=3,
        active_backend_getter=lambda: "newton",
        snapshot_getter=_get_fake_snapshot,
        snapshot_setter=_set_fake_snapshot,
        contact_probe=lambda _runtime, _backend: {
            "performed": True,
            "nconmax": 200,
            "max_contacts": 3,
        },
        control_mode_probe=lambda *_args, **_kwargs: {
            "verified": False,
            "reason": "test_double",
        },
    )

    assert report["physics_backend"] == "newton"
    assert report["scene"] == "scene3"
    assert report["steps"] == 3
    assert report["robot_count"] == 2
    assert runtime.physics.steps == 4
    assert all(
        item["tensor_handle_valid"] for item in report["articulations_after_step"]
    )
    assert report["snapshot"]["readback_verified"] is True
    assert report["snapshot"]["drive_targets_verified"] is True
    assert report["snapshot"]["post_step_readback_verified"] is True
    assert report["snapshot"]["post_step_physics_steps"] == 1
    assert len(report["control_response"]) == 2
    assert report["control_modes"] == {
        "verified": False,
        "reason": "test_double",
    }
    assert report["hybrid_control"] == {
        "performed": False,
        "reason": "not_configured",
    }
    for response in report["control_response"]:
        assert response["target_readback"] == pytest.approx(response["target"])
        assert response["stiffness"] == 200.0
        assert response["damping"] == 20.0
        assert response["final_velocity"] == pytest.approx(0.0)
    assert report["active_mjc_actuators"] == []
    assert report["camera"] == {
        "performed": False,
        "reason": "no_enabled_camera",
    }
    assert report["physics_runtime"] == {
        "performed": False,
        "reason": "session_runtime_unavailable",
    }
    assert report["dynamic_chain_snapshot"] == {
        "performed": False,
        "reason": "no_dynamic_chain_objects",
        "objects": [],
    }
    assert report["curobo_fk"]["robot_count"] == 1
    assert report["curobo_fk"]["robots"][0]["joint_names"] == ["j1", "j0"]
    assert report["curobo_fk"]["robots"][0]["articulation_indices"] == [1, 0]
    assert report["rigid_velocity_order"]["contract"] == ("linear_xyz_then_angular_xyz")
    assert [item["kind"] for item in report["rigid_velocity_order"]["probes"]] == [
        "pure_linear",
        "pure_angular",
    ]
    assert report["rigid_velocity_order"]["restored"] is True
    rigid_view = runtime.object_state_views["Tblock"]
    np.testing.assert_allclose(rigid_view.linear, [0.4, -0.5, 0.6])
    np.testing.assert_allclose(rigid_view.angular, [-0.7, 0.8, -0.9])
    np.testing.assert_allclose(runtime.fk.received, initial_left[[1, 0]])
    np.testing.assert_allclose(
        runtime.robots_by_id[0].execution.articulation.positions,
        initial_left,
    )
    assert runtime.planning_registry.leases == [(0, "interactive")]
    for robot in runtime.robots_by_id.values():
        assert robot.execution.joint_controller.applied == 3


def test_runtime_control_mode_probe_exercises_all_modes_without_replacing_owners() -> (
    None
):
    runtime = _FakeRuntime()
    robots = tuple(runtime.robots_by_id.values())
    runtime.physics_runtime = object()
    runtime.session.physics_runtime = runtime.physics_runtime
    runtime.control_mode = object()
    runtime.collision = object()
    runtime.motion = SimpleNamespace(backend=object())
    mode_state = {"mode": "position", "generation": 0}
    motion_operations: list[str] = []
    reset_calls: list[bool] = []

    for robot in robots:
        names = tuple(robot.execution.joint_controller.command_joint_names)
        robot.joint_groups = SimpleNamespace(
            names=lambda group, names=names: names if group == "arm" else ()
        )
        controller = robot.execution.joint_controller
        controller.command_target_modes = ("position",) * len(names)
        controller.last_commanded_efforts = np.zeros(
            robot.execution.articulation.num_dof,
            dtype=float,
        )
        controller.prepare_runtime = lambda robot=robot: SimpleNamespace(
            active_effort_limits=np.full(
                robot.execution.articulation.num_dof,
                10.0,
                dtype=float,
            )
        )

    def get_control_mode():
        return SimpleNamespace(
            initial_mode="position",
            active_mode=mode_state["mode"],
            generation=mode_state["generation"],
            scope="all",
        )

    def set_control_mode(mode: str, *, expected_generation: int):
        assert expected_generation == mode_state["generation"]
        previous = mode_state["mode"]
        mode_state["mode"] = mode
        mode_state["generation"] += 1
        for robot in robots:
            width = len(robot.execution.joint_controller.command_joint_names)
            robot.execution.joint_controller.command_target_modes = (mode,) * width
        return SimpleNamespace(
            previous_mode=previous,
            active_mode=mode,
            generation=mode_state["generation"],
            changed=True,
        )

    def execute_motion(
        operation: str,
        arguments: Mapping[str, object],
        *,
        request_id: str,
        should_cancel,
        protocol: str,
    ) -> None:
        assert request_id.startswith("smoke-control-")
        assert should_cancel() is False
        assert protocol == "linkerbot.mirror.v2"
        motion_operations.append(operation)
        robot = runtime.robots_by_id[int(arguments["robot_id"])]
        articulation = robot.execution.articulation
        controller = robot.execution.joint_controller
        if operation == "motion.joint_goal":
            targets = dict(arguments["joint_positions"])
            for name, value in targets.items():
                articulation.positions[articulation.dof_names.index(name)] = float(
                    value
                )
        else:
            assert operation == "motion.joint_effort"
            assert list(dict(arguments["joint_efforts"]).values()) == [0.5]
        controller.last_targets = SimpleNamespace(
            positions=articulation.positions.copy(),
            velocities=np.zeros(articulation.num_dof, dtype=float),
            efforts=np.zeros(articulation.num_dof, dtype=float),
        )
        controller.last_commanded_efforts.fill(0.0)
        runtime.physics.step(render=False)

    runtime.get_control_mode = get_control_mode
    runtime.set_control_mode = set_control_mode
    runtime.motion.execute = execute_motion
    runtime.reset = lambda *, hold_after_reset: reset_calls.append(hold_after_reset)

    report = smoke.probe_mirror_control_modes(
        runtime,
        expected_backend="physx",
        steps=2,
        active_backend_getter=lambda: "physx",
    )

    assert report["event"] == "mirror_control_mode_smoke"
    assert report["control_modes"]["verified"] is True
    assert report["control_modes"]["sequence"] == [
        "position",
        "velocity",
        "effort",
        "position",
    ]
    assert report["control_modes"]["generation"] == 3
    assert report["control_modes"]["terminal_velocity_zero"] is True
    assert report["control_modes"]["terminal_effort_zero"] is True
    assert motion_operations == [
        "motion.joint_goal",
        "motion.joint_goal",
        "motion.joint_effort",
        "motion.joint_goal",
    ]
    assert reset_calls == [False]


def test_runtime_hybrid_probe_updates_gains_between_axis_selections() -> None:
    settings = load_mirror_config("physx_cpu_hybrid").hybrid_control
    assert settings is not None
    parameters = HybridParameterService(settings)
    position_settings = SimpleNamespace(
        arm=SimpleNamespace(mode="position", method="implicit"),
        hand=SimpleNamespace(mode="position", method="implicit"),
    )
    hybrid_settings = SimpleNamespace(
        arm=SimpleNamespace(mode="effort", method="direct"),
        hand=SimpleNamespace(mode="position", method="implicit"),
    )
    controller = SimpleNamespace(
        settings=position_settings,
        command_target_modes=("position", "position"),
        last_commanded_efforts=np.zeros(2, dtype=float),
        last_control_targets=SimpleNamespace(efforts=np.zeros(2, dtype=float)),
    )
    port = SimpleNamespace(
        observe=lambda: SimpleNamespace(
            position=np.asarray([0.1, 0.2, 0.3]),
            orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            joint_velocities=np.zeros(7, dtype=float),
            jacobian=np.column_stack((np.eye(6), np.zeros(6))),
        )
    )
    arm_names = tuple(f"joint_{index}" for index in range(7))
    robot = SimpleNamespace(
        robot_id=0,
        label="left",
        task_space_port=port,
        physical_tcp_binding=SimpleNamespace(tcp_frame_name="tcp"),
        joint_groups=SimpleNamespace(arm=arm_names),
        execution=SimpleNamespace(joint_controller=controller),
    )
    backend = SimpleNamespace(sample={"active": False})
    backend.hybrid_diagnostics = lambda: dict(backend.sample)
    requests: list[dict[str, object]] = []

    class ExpectedCancellation(RuntimeError):
        code = "cancelled"

    class Motion:
        def __init__(self) -> None:
            self.backend = backend

        def tare_wrench(self, arguments, **kwargs):
            assert arguments == {
                "robot_id": 0,
                "robot_label": "left",
                "tcp_frame_name": "tcp",
                "reference_frame": "world",
            }
            assert kwargs["request_id"] == "smoke-hybrid-tare"
            assert kwargs["should_cancel"]() is False
            return {"tare_generation": 4, "sample_count": 120}

        def execute(self, operation, arguments, **kwargs):
            if operation == "motion.joint_goal":
                assert arguments["joint_positions"] == dict(
                    zip(
                        arm_names,
                        smoke.HYBRID_PROBE_ARM_POSITIONS,
                        strict=True,
                    )
                )
                assert kwargs["protocol"] == "linkerbot.mirror.v2"
                assert kwargs["should_cancel"]() is False
                return {"event": "motion_completed"}
            if operation == "motion.hold":
                assert arguments["group"] == "arm"
                assert kwargs["protocol"] == "linkerbot.mirror.v2"
                assert kwargs["should_cancel"]() is False
                return {"event": "motion_completed"}
            assert operation == "motion.hybrid_force_position"
            assert kwargs["protocol"] == "linkerbot.mirror.v3"
            requests.append(dict(arguments))
            controller.settings = hybrid_settings
            controller.command_target_modes = ("effort", "position")
            backend.sample = {
                "active": True,
                "request_id": kwargs["request_id"],
                "force_axes": list(arguments["force_axes"]),
                "hybrid_parameter_generation": arguments["hybrid_parameter_generation"],
            }
            try:
                assert kwargs["should_cancel"]() is True
            finally:
                controller.settings = position_settings
                controller.command_target_modes = ("position", "position")
                controller.last_commanded_efforts.fill(np.nan)
                controller.last_control_targets = SimpleNamespace(
                    efforts=np.zeros(2, dtype=float)
                )
                backend.sample = {"active": False}
            raise ExpectedCancellation

    runtime = SimpleNamespace(
        config=SimpleNamespace(hybrid_control=settings),
        controller=SimpleNamespace(hybrid_parameters=parameters),
        motion=Motion(),
        physics_dt_s=1.0 / 240.0,
        get_control_mode=lambda: SimpleNamespace(active_mode="position"),
    )

    report = smoke._exercise_runtime_hybrid_control(
        runtime,
        SimpleNamespace(),
        (robot,),
    )

    assert report["performed"] is True
    assert report["tare_generation"] == 4
    assert report["tare_sample_count"] == 120
    assert report["probe_pose"]["arm_joint_positions"] == list(
        smoke.HYBRID_PROBE_ARM_POSITIONS
    )
    assert report["probe_pose"]["maximum_joint_speed"] == 0.0
    assert report["probe_pose"]["minimum_singular_value"] == pytest.approx(1.0)
    assert report["probe_pose"]["condition_number"] == pytest.approx(10.0)
    assert report["parameter_generations"] == [0, 1]
    assert report["updated_parameters"] == [
        "force_integral",
        "force_proportional",
        "motion_damping",
        "motion_stiffness",
        "posture_damping",
        "posture_stiffness",
    ]
    assert [item["force_axes"] for item in report["segments"]] == [
        [False, False, True, False, False, False],
        [True, False, False, False, False, False],
    ]
    assert [item["hybrid_parameter_generation"] for item in report["segments"]] == [
        0,
        1,
    ]
    assert all(item["expected_cancellation"] for item in report["segments"])
    assert all(item["terminal_effort_zero"] for item in report["segments"])
    assert all(
        item["active_controller"]["arm"] == {"mode": "effort", "method": "direct"}
        for item in report["segments"]
    )
    assert controller.settings is position_settings
    assert controller.command_target_modes == ("position", "position")
    assert [request["hybrid_parameter_generation"] for request in requests] == [0, 1]


def test_position_probe_uses_controller_cache_without_removed_isaac_getters() -> None:
    runtime = _FakeRuntime()
    robots = tuple(runtime.robots_by_id.values())
    for robot in robots:
        articulation = robot.execution.articulation
        articulation.get_joint_position_targets = None
        articulation.get_joint_velocity_targets = None

    responses = smoke._apply_position_targets(runtime, robots, steps=1)

    assert len(responses) == 2
    assert all(
        item["target_readback"] == pytest.approx(item["target"]) for item in responses
    )


def test_camera_probe_retries_blank_payload_then_accepts_valid_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _camera_probe_runtime()

    def sample_camera_frames(
        _camera: object,
        *,
        simulation_step: int,
        time_s: float,
        **_kwargs: object,
    ) -> tuple[SimpleNamespace, ...]:
        assert simulation_step == 30 + runtime.physics.steps
        assert time_s == pytest.approx(
            0.25 + runtime.physics.steps * runtime.physics.get_physics_dt()
        )
        if runtime.physics.steps == 1:
            return (
                _camera_frame("rgb", np.zeros((1, 2, 3), dtype=np.uint8)),
                _camera_frame("depth", np.full((1, 2), np.inf, dtype=np.float32)),
            )
        return (
            _camera_frame("rgb", np.ones((1, 2, 3), dtype=np.uint8)),
            _camera_frame("depth", np.asarray([[0.5, np.inf]], dtype=np.float32)),
        )

    monkeypatch.setattr(
        "linkerbot_sim.sensors.camera.frame.sample_camera_frames",
        sample_camera_frames,
    )

    report = smoke._camera_probe(runtime)

    assert runtime.physics.steps == 2
    assert runtime.session.physics_runtime.render_calls == 1
    assert runtime.session.physics_runtime.simulation_time == pytest.approx(
        0.25 + 2.0 * runtime.physics.get_physics_dt()
    )
    assert report["performed"] is True
    assert report["warmup_physics_steps"] == 2
    assert report["render_only_updates"] == 1
    assert report["render_only_verified"] is True
    assert report["frames"][0]["valid_values"] == 1
    assert report["frames"][1]["nonzero_values"] == 6


def test_camera_probe_retries_incomplete_render_only_modalities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _camera_probe_runtime()
    manager = runtime.session.physics_runtime

    def sample_camera_frames(*_args, **_kwargs):
        rgb = _camera_frame("rgb", np.ones((1, 2, 3), dtype=np.uint8))
        depth = _camera_frame("depth", np.ones((1, 2), dtype=np.float32))
        if manager.render_calls == 1:
            return (rgb,)
        if manager.render_calls == 2:
            return (depth,)
        return (rgb, depth)

    monkeypatch.setattr(
        "linkerbot_sim.sensors.camera.frame.sample_camera_frames",
        sample_camera_frames,
    )

    report = smoke._camera_probe(runtime)

    assert runtime.physics.steps == 1
    assert manager.render_calls == 2
    assert report["render_only_updates"] == 2
    assert {item["modality"] for item in report["frames"]} == {"rgb", "depth"}
    assert report["render_only_verified"] is True


def test_camera_warmup_scales_with_physics_rate_and_camera_period() -> None:
    resources = SimpleNamespace(
        physics=SimpleNamespace(get_physics_dt=lambda: 1.0 / 240.0)
    )
    cameras = (
        SimpleNamespace(settings=SimpleNamespace(frequency=20.0)),
        SimpleNamespace(settings=SimpleNamespace(frequency=60.0)),
    )

    assert smoke._camera_warmup_step_limit(resources, cameras) == 24


def test_camera_probe_reports_warmup_missing_when_payload_stays_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _camera_probe_runtime()
    monkeypatch.setattr(
        "linkerbot_sim.sensors.camera.frame.sample_camera_frames",
        lambda *_args, **_kwargs: (
            _camera_frame("rgb", np.zeros((1, 2, 3), dtype=np.uint8)),
            _camera_frame("depth", np.zeros((1, 2), dtype=np.float32)),
        ),
    )

    with pytest.raises(RuntimeError, match="not ready after warmup.*depth.*rgb"):
        smoke._camera_probe(runtime)

    assert runtime.physics.steps == smoke.CAMERA_WARMUP_STEPS
    assert runtime.session.physics_runtime.render_calls == 0


def test_camera_probe_rejects_malformed_shape_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _camera_probe_runtime()
    monkeypatch.setattr(
        "linkerbot_sim.sensors.camera.frame.sample_camera_frames",
        lambda *_args, **_kwargs: (
            _camera_frame("rgb", np.ones((1, 2), dtype=np.uint8)),
            _camera_frame("depth", np.ones((1, 2), dtype=np.float32)),
        ),
    )

    with pytest.raises(RuntimeError, match="RGB contract mismatch"):
        smoke._camera_probe(runtime)

    assert runtime.physics.steps == 1
    assert runtime.session.physics_runtime.render_calls == 0


def test_camera_probe_rejects_render_only_physics_clock_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _camera_probe_runtime()
    manager = runtime.session.physics_runtime

    def advancing_render() -> None:
        manager.render_calls += 1
        manager.simulation_time += runtime.physics.get_physics_dt()

    manager.render = advancing_render
    monkeypatch.setattr(
        "linkerbot_sim.sensors.camera.frame.sample_camera_frames",
        lambda *_args, **_kwargs: (
            _camera_frame("rgb", np.ones((1, 2, 3), dtype=np.uint8)),
            _camera_frame("depth", np.ones((1, 2), dtype=np.float32)),
        ),
    )

    with pytest.raises(RuntimeError, match="render-only verification advanced physics"):
        smoke._camera_probe(runtime)

    assert runtime.physics.steps == 1
    assert manager.render_calls == 1


def test_mirror_snapshot_roundtrip_checks_owner_before_maximal_projection() -> None:
    expected = _mirror_generalized_rope_snapshot()
    projected = _mirror_generalized_rope_snapshot(body_velocity_y=10.0)

    projected = SceneSnapshot(
        robots={},
        objects={
            "rope": replace(
                projected.objects["rope"],
                positions_local=np.asarray([5.0, -2.0, 1.0]),
                orientations_wxyz=np.asarray([0.0, 1.0, 0.0, 0.0]),
                body_positions_local=np.asarray([[5.0, -2.0, 1.0]]),
                body_orientations_wxyz=np.asarray([[0.0, 1.0, 0.0, 0.0]]),
            )
        },
    )

    smoke._assert_snapshot_roundtrip(expected, projected)

    wrong_owner = _mirror_generalized_rope_snapshot(owner_q=0.26)
    with pytest.raises(RuntimeError, match="generalized q"):
        smoke._assert_snapshot_roundtrip(expected, wrong_owner)

    missing_body_twist = SceneSnapshot(
        robots={},
        objects={
            "rope": replace(
                projected.objects["rope"],
                body_angular_velocities=None,
            )
        },
    )
    with pytest.raises(RuntimeError, match="presence.*body_angular_velocities"):
        smoke._assert_snapshot_roundtrip(expected, missing_body_twist)


def test_curobo_fk_probe_checks_every_planning_enabled_robot() -> None:
    runtime = _FakeRuntime()
    right_fk = _FakeFk()
    fks = {0: runtime.fk, 1: right_fk}
    runtime.robots_by_id[1].supports_planning = True
    runtime.robots_by_id[1].curobo_config = object()
    right_fk.joint_names = lambda: ["k2", "k0"]

    class PlanningRegistry:
        def __init__(self) -> None:
            self.leases: list[tuple[int, str]] = []

        @contextmanager
        def lease(self, robot_id: int, *, consumer_role: str):
            self.leases.append((robot_id, consumer_role))
            yield SimpleNamespace(
                default_tcp_frame="tool",
                make_forward_kinematics=lambda: fks[robot_id],
            )

    registry = PlanningRegistry()
    runtime.planning_registry = registry

    report = smoke._curobo_fk_probe(
        runtime,
        tuple(runtime.robots_by_id.values()),
    )

    assert report["performed"] is True
    assert report["robot_count"] == 2
    assert [item["label"] for item in report["robots"]] == ["left", "right"]
    assert [item["articulation_indices"] for item in report["robots"]] == [
        [1, 0],
        [2, 0],
    ]
    assert registry.leases == [(0, "interactive"), (1, "interactive")]
    np.testing.assert_allclose(runtime.fk.received, [0.1, 0.0])
    np.testing.assert_allclose(right_fk.received, [1.2, 1.0])


def test_newton_rigid_velocity_order_probe_restores_after_mismatch() -> None:
    view = _FakeRigidStateView(corrupt_first_probe_readback=True)
    runtime = SimpleNamespace(object_state_views={"block": view})

    with pytest.raises(RuntimeError, match="pure_linear order mismatch"):
        smoke._rigid_velocity_order_probe(runtime, "newton")

    np.testing.assert_allclose(view.linear, [0.4, -0.5, 0.6])
    np.testing.assert_allclose(view.angular, [-0.7, 0.8, -0.9])
    assert len(view.writes) == 2


def test_newton_rigid_velocity_probe_skips_dynamic_chain_and_static_views() -> None:
    runtime = SimpleNamespace(
        object_state_views={
            "rope": SimpleNamespace(
                root_view=None,
                body_view=object(),
                immutable_position=None,
                velocity_capability="complete",
            ),
            "workstation": SimpleNamespace(
                root_view=None,
                body_view=None,
                immutable_position=(0.0, 0.0, 0.0),
                velocity_capability="complete",
            ),
        }
    )

    assert smoke._rigid_velocity_order_probe(runtime, "newton") == {
        "performed": False,
        "reason": "no_independently_writable_free_rigid_root",
    }


def test_newton_runtime_contact_probe_uses_manager_without_legacy_extension() -> None:
    solver = SimpleNamespace(
        mjw_data=SimpleNamespace(nacon=np.asarray([3], dtype=int)),
        mj_model=SimpleNamespace(nu=7),
    )
    manager = SimpleNamespace(
        backend="newton",
        execution="cuda",
        solver=solver,
        diagnostics=lambda: {
            "backend": "newton",
            "execution": "cuda",
            "nconmax_per_world": 200,
        },
    )
    runtime = SimpleNamespace(
        session=SimpleNamespace(physics_runtime=manager),
    )

    report = smoke._newton_contact_probe(runtime, "newton")

    assert report == {
        "performed": True,
        "owner": "newton_runtime",
        "execution": "cuda",
        "nconmax": 200,
        "max_contacts": 3,
        "world_contact_counts": [3],
        "mujoco_actuator_count": 7,
    }


def test_newton_cpu_contact_probe_reads_single_mujoco_world() -> None:
    solver = SimpleNamespace(
        mj_data=SimpleNamespace(ncon=2),
        mj_model=SimpleNamespace(nu=5),
    )
    manager = SimpleNamespace(
        backend="newton",
        execution="cpu",
        solver=solver,
        diagnostics=lambda: {"nconmax_per_world": 200},
    )
    runtime = SimpleNamespace(
        session=SimpleNamespace(physics_runtime=manager),
    )

    report = smoke._newton_contact_probe(runtime, "newton")

    assert report == {
        "performed": True,
        "owner": "newton_runtime",
        "execution": "cpu",
        "nconmax": 200,
        "max_contacts": 2,
        "world_contact_counts": [2],
        "mujoco_actuator_count": 5,
    }


def test_dynamic_chain_snapshot_probe_requires_complete_body_state() -> None:
    rope = ObjectSnapshot(
        name="rope",
        positions_local=np.zeros(3),
        orientations_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        body_names=("a", "b"),
        body_positions_local=np.zeros((2, 3)),
        body_orientations_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        body_linear_velocities=np.zeros((2, 3)),
        body_angular_velocities=np.zeros((2, 3)),
    )

    assert smoke._dynamic_chain_snapshot_probe(
        SceneSnapshot(robots={}, objects={"rope": rope})
    ) == {
        "performed": True,
        "reason": None,
        "objects": [
            {
                "name": "rope",
                "body_count": 2,
                "complete_pose_velocity_state": True,
            }
        ],
    }


def test_probe_fails_closed_on_backend_mismatch() -> None:
    runtime = _FakeRuntime()

    with pytest.raises(RuntimeError, match="active physics backend mismatch"):
        smoke.probe_mirror_runtime(
            runtime,
            expected_backend="newton",
            steps=1,
            active_backend_getter=lambda: "physx",
            snapshot_getter=_get_fake_snapshot,
            snapshot_setter=_set_fake_snapshot,
        )

    assert runtime.physics.steps == 0


def test_probe_rejects_non_positive_steps_for_direct_call() -> None:
    runtime = _FakeRuntime()

    with pytest.raises(ValueError, match="steps must be positive"):
        smoke.probe_mirror_runtime(
            runtime,
            expected_backend="physx",
            steps=0,
            active_backend_getter=lambda: "physx",
            snapshot_getter=_get_fake_snapshot,
            snapshot_setter=_set_fake_snapshot,
        )

    assert runtime.physics.steps == 0


@pytest.mark.parametrize("failure", ("handle", "position", "velocity"))
def test_probe_rejects_invalid_tensor_state(failure: str) -> None:
    runtime = _FakeRuntime()
    articulation = runtime.robots_by_id[0].execution.articulation
    if failure == "handle":
        articulation.valid = False
    elif failure == "position":
        articulation.positions[0] = np.nan
    else:
        articulation.velocities[0] = np.inf

    with pytest.raises(RuntimeError):
        smoke.probe_mirror_runtime(
            runtime,
            expected_backend="physx",
            steps=1,
            active_backend_getter=lambda: "physx",
            snapshot_getter=_get_fake_snapshot,
            snapshot_setter=_set_fake_snapshot,
        )


def test_execute_smoke_closes_runtime_and_reports_shutdown(monkeypatch) -> None:
    runtime = _FakeRuntime()
    config = SimpleNamespace(
        control=SimpleNamespace(mode="position"),
        physics=SimpleNamespace(engine="physx", execution="cpu"),
    )
    monkeypatch.setattr(smoke, "resolve_smoke_config", lambda _args: config)
    monkeypatch.setattr(smoke, "create_smoke_runtime", lambda *_args: runtime)
    monkeypatch.setattr(
        smoke,
        "probe_mirror_runtime",
        lambda *_args, **_kwargs: {"event": "mirror_physics_smoke"},
    )

    report = smoke.execute_smoke(smoke.parse_args([]))

    assert report["shutdown"] == {"stopped": True, "live_resources": []}
    assert runtime.close_calls == 1


def test_execute_smoke_reports_before_runtime_close(monkeypatch) -> None:
    runtime = _FakeRuntime()
    config = SimpleNamespace(
        control=SimpleNamespace(mode="position"),
        physics=SimpleNamespace(engine="physx", execution="cpu"),
    )
    monkeypatch.setattr(smoke, "resolve_smoke_config", lambda _args: config)
    monkeypatch.setattr(smoke, "create_smoke_runtime", lambda *_args: runtime)
    monkeypatch.setattr(
        smoke,
        "probe_mirror_runtime",
        lambda *_args, **_kwargs: {"event": "mirror_physics_smoke"},
    )
    observed: list[tuple[dict[str, object], int]] = []

    smoke.execute_smoke(
        smoke.parse_args([]),
        before_close=lambda report: observed.append(
            (dict(report), runtime.close_calls)
        ),
    )

    assert observed == [
        (
            {
                "event": "mirror_physics_smoke",
                "shutdown": {"application_close_requested": True},
            },
            0,
        )
    ]
    assert runtime.close_calls == 1


def test_execute_smoke_closes_runtime_when_probe_fails(monkeypatch) -> None:
    runtime = _FakeRuntime()
    config = SimpleNamespace(
        control=SimpleNamespace(mode="position"),
        physics=SimpleNamespace(engine="physx", execution="cpu"),
    )
    monkeypatch.setattr(smoke, "resolve_smoke_config", lambda _args: config)
    monkeypatch.setattr(smoke, "create_smoke_runtime", lambda *_args: runtime)

    def fail(*_args, **_kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(smoke, "probe_mirror_runtime", fail)

    with pytest.raises(RuntimeError, match="probe failed"):
        smoke.execute_smoke(smoke.parse_args([]))

    assert runtime.close_calls == 1


def test_main_outputs_pre_shutdown_runtime_json_marker(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LINKERBOT_ISAAC_RUNTIME_WORKER", "1")

    def fake_execute(_args, *, before_close):
        report = {
            "event": "mirror_physics_smoke",
            "physics_backend": "newton",
            "shutdown": {"application_close_requested": True},
        }
        before_close(report)
        return report

    monkeypatch.setattr(smoke, "execute_smoke", fake_execute)

    assert smoke.main(["--profile", "newton_cuda"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    runtime_marker, runtime_payload = lines[0].split(" ", maxsplit=1)
    assert runtime_marker == smoke.RUNTIME_SUCCESS_MARKER
    assert json.loads(runtime_payload)["shutdown"] == {
        "application_close_requested": True
    }
    assert json.loads(runtime_payload)["physics_backend"] == "newton"


def test_main_help_exits_before_starting_runtime_worker(monkeypatch, capsys) -> None:
    supervised: list[object] = []
    monkeypatch.delenv("LINKERBOT_ISAAC_RUNTIME_WORKER", raising=False)
    monkeypatch.setattr(
        smoke,
        "run_supervised_worker",
        lambda **kwargs: supervised.append(kwargs),
    )

    with pytest.raises(SystemExit) as raised:
        smoke.main(["--help"])

    assert raised.value.code == 0
    assert "--profile" in capsys.readouterr().out
    assert supervised == []
