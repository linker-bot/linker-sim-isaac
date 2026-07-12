from __future__ import annotations

import importlib.util
import inspect
import queue
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.app.interactive import tiled_scene as tiled_scene_interactive
from linkerbot_sim.app.interactive.tiled_scene import (
    command_utils as tiled_scene_command_utils,
)
from linkerbot_sim.app.interactive.tiled_scene.runtime import (
    TiledSceneRuntime,
)
from linkerbot_sim.app.interactive.tiled_scene.runtime import (
    factory as tiled_scene_runtime_factory,
)
from linkerbot_sim.app.interactive.tiled_scene.runtime import (
    ik as tiled_scene_runtime_ik,
)
from linkerbot_sim.app.interactive.tiled_scene.runtime import (
    state as tiled_scene_runtime_state,
)
from linkerbot_sim.app.interactive.tiled_scene.runtime import (
    stepping as tiled_scene_runtime_stepping,
)
from linkerbot_sim.app.interactive.tiled_scene import (
    telemetry_publish as tiled_scene_telemetry_publish,
)
from linkerbot_sim.app.interactive.tiled_scene import transport as tiled_scene_transport
from linkerbot_sim.configs.runtime import (
    PlannerRequestDefaults,
    RuntimeCommandDefaults,
)
from tests.fakes.tiled_scene_runtime_fake import (
    DebugBatchIKBackend,
    DebugTiledSceneRuntime,
)
from linkerbot_sim.planning.batch_ik import BatchIKResult
from linkerbot_sim.snapshots.transactions import (
    RuntimeMutationRejected,
    SnapshotRollbackError,
)
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.control.adapter import TiledCommandAdapter
from linkerbot_sim.tiled.control.types import TiledCommandAction
from linkerbot_sim.tiled.planning.manager import TiledPlannerManager
from linkerbot_sim.tiled.planning.types import TiledPlanningResult
from linkerbot_sim.tiled.playback.buffer import TiledTrajectoryBuffer
from linkerbot_sim.utils.rotations import matrix_to_quat_wxyz


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "tiled_scene_interactive.py"
)


def load_tiled_scene_interactive_module():
    """返回包内 Tiled Scene interactive runtime 模块。"""

    return tiled_scene_interactive


def load_tiled_scene_interactive_script_wrapper():
    """按文件路径导入 CLI wrapper，确认 scripts/ 入口仍能使用。"""

    spec = importlib.util.spec_from_file_location(
        "tiled_scene_interactive", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tiled_scene_script_wrapper_reexports_main_entrypoint() -> None:
    module = load_tiled_scene_interactive_script_wrapper()

    assert module.main is tiled_scene_interactive.main


def make_runtime(*, fail_env_ids=frozenset(), failure_policy: str = "hold_failed_env"):
    return DebugTiledSceneRuntime.create(
        env_name="unit",
        env_config={
            "env": {"physics_frequency": 100.0},
            "tiled": {
                "enabled": True,
                "num_envs": 2,
                "spacing": 2.0,
            },
        },
        command_dim=3,
        default_decimation=2,
        tcp_frame_name="tcp",
        ik_solver=DebugBatchIKBackend(fail_env_ids=frozenset(fail_env_ids)),
        failure_policy=failure_policy,
    )


def test_isaac_tiled_step_world_refreshes_all_mimics_and_samples_camera_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWorld:
        def __init__(self) -> None:
            self.render_calls: list[bool] = []

        def step(self, *, render: bool) -> None:
            self.render_calls.append(bool(render))

        def get_physics_dt(self) -> float:
            return 0.1

    class FakeObserver:
        def __init__(self) -> None:
            self.calls: list[tuple[object, int, str | None]] = []

        def observe(self, world, *, step: int, phase: str | None = None) -> None:
            self.calls.append((world, step, phase))

    world = FakeWorld()
    observer = FakeObserver()
    articulations = {"left": object(), "right": object()}
    mimic_calls: list[object] = []
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.runtime.stepping._apply_runtime_mimic_targets",
        mimic_calls.append,
    )
    runtime = TiledSceneRuntime(
        env_name="unit",
        env_config={},
        session=SimpleNamespace(world=world, app=None),
        scene=SimpleNamespace(
            config=SimpleNamespace(num_envs=2),
            articulation_views=articulations,
        ),
        render=True,
        default_decimation=1,
        robot_names=("left", "right"),
        episode_steps=np.zeros(2, dtype=int),
        episode_ids=np.zeros(2, dtype=int),
        initial_joint_positions={},
        initial_joint_velocities={},
        target_positions={},
        initial_object_states={},
        command_adapters={},
        ik_solvers={},
        tcp_positions_world={},
        tcp_orientations_wxyz={},
        trajectory_buffer=TiledTrajectoryBuffer(num_envs=2),
        planner_manager=SimpleNamespace(),
        sensor_cameras=(),
        camera_output=SimpleNamespace(observer=observer),
        quit_event=threading.Event(),
    )

    runtime._step_world(phase="action")

    assert world.render_calls == [True]
    assert mimic_calls == [articulations["left"], articulations["right"]]
    assert observer.calls == [(world, 0, "action")]
    assert runtime.step == 1
    assert runtime.episode_steps.tolist() == [1, 1]


def test_isaac_tiled_step_world_commits_counters_before_observer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class World:
        def __init__(self) -> None:
            self.steps = 0

        def step(self, *, render: bool) -> None:
            self.steps += 1

    class FailingObserver:
        def observe(self, *_args, **_kwargs) -> None:
            raise RuntimeError("camera observer failed")

    world = World()
    articulation = object()
    runtime = SimpleNamespace(
        session=SimpleNamespace(world=world),
        render=False,
        step=4,
        episode_steps=np.asarray([2, 3], dtype=int),
        camera_output=SimpleNamespace(observer=FailingObserver()),
        _selected_runtime_items=lambda _selection: (("robot", articulation),),
    )
    monkeypatch.setattr(
        tiled_scene_runtime_stepping,
        "_apply_runtime_mimic_targets",
        lambda _articulation: None,
    )

    with pytest.raises(RuntimeError, match="camera observer failed"):
        tiled_scene_runtime_stepping.step_world(runtime, phase="action")

    assert world.steps == 1
    assert runtime.step == 5
    assert runtime.episode_steps.tolist() == [3, 4]


def test_tiled_scene_runtime_close_releases_resources_in_owner_order(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Resource:
        def __init__(self, name: str, method: str = "close") -> None:
            setattr(self, method, lambda: calls.append(name))

    runtime = object.__new__(TiledSceneRuntime)
    runtime._closed = False
    runtime.planner_manager = Resource("planner", "shutdown")
    runtime.ik_solvers = {
        "arm": SimpleNamespace(solver=SimpleNamespace(context=Resource("ik_context")))
    }
    runtime.camera_output = Resource("camera")
    runtime.session = SimpleNamespace(app=object())
    monkeypatch.setattr(
        "linkerbot_sim.app.runtime.simulation_app_lifecycle.close_simulation_app",
        lambda app: calls.append("simulation_app"),
    )

    runtime.close()
    runtime.close()

    assert calls == ["planner", "camera", "ik_context", "simulation_app"]


def test_tiled_scene_runtime_camera_timeout_is_retryable_and_fail_closed(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Resource:
        def __init__(self, name: str, method: str = "close") -> None:
            setattr(self, method, lambda: calls.append(name))

    class RetryCamera:
        def __init__(self) -> None:
            self.attempts = 0

        def close(self) -> bool:
            self.attempts += 1
            calls.append("camera")
            return self.attempts > 1

    runtime = object.__new__(TiledSceneRuntime)
    runtime._closed = False
    runtime.planner_manager = Resource("planner", "shutdown")
    runtime.ik_solvers = {
        "arm": SimpleNamespace(solver=SimpleNamespace(context=Resource("ik_context")))
    }
    runtime.camera_output = RetryCamera()
    runtime.session = SimpleNamespace(app=object())
    monkeypatch.setattr(
        "linkerbot_sim.app.runtime.simulation_app_lifecycle.close_simulation_app",
        lambda _app: calls.append("simulation_app"),
    )

    assert runtime.close() is False
    assert runtime._closed is False
    assert calls == ["planner", "camera", "ik_context"]

    assert runtime.close() is True
    assert runtime._closed is True
    assert calls == ["planner", "camera", "ik_context", "camera", "simulation_app"]


@pytest.mark.parametrize("failing_resource", ("planner", "camera", "ik_context"))
def test_tiled_scene_runtime_close_exception_still_releases_other_resources(
    failing_resource: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class RetriableResource:
        def __init__(self, name: str, method: str) -> None:
            self.name = name
            self.attempts = 0
            setattr(self, method, self._close)

        def _close(self) -> None:
            self.attempts += 1
            calls.append(self.name)
            if self.name == failing_resource and self.attempts == 1:
                raise RuntimeError(f"{self.name} close failed")

    planner = RetriableResource("planner", "shutdown")
    camera = RetriableResource("camera", "close")
    ik_context = RetriableResource("ik_context", "close")
    runtime = object.__new__(TiledSceneRuntime)
    runtime._closed = False
    runtime.planner_manager = planner
    runtime.camera_output = camera
    runtime.ik_solvers = {
        "arm": SimpleNamespace(solver=SimpleNamespace(context=ik_context))
    }
    runtime.session = SimpleNamespace(app=object())
    monkeypatch.setattr(
        "linkerbot_sim.app.runtime.simulation_app_lifecycle.close_simulation_app",
        lambda _app: calls.append("simulation_app"),
    )

    with pytest.raises(RuntimeError, match=f"{failing_resource} close failed"):
        runtime.close()
    assert calls == ["planner", "camera", "ik_context"]
    assert runtime._closed is False
    assert runtime._app_closed is False

    assert runtime.close() is True
    assert calls == [
        "planner",
        "camera",
        "ik_context",
        failing_resource,
        "simulation_app",
    ]


def test_failed_tiled_creation_cleanup_attempts_resources_after_errors_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_planner() -> None:
        calls.append("planner")
        raise RuntimeError("planner close failed")

    def timeout_camera() -> bool:
        calls.append("camera")
        return False

    def fail_ik() -> None:
        calls.append("failing_ik")
        raise RuntimeError("ik close failed")

    monkeypatch.setattr(
        "linkerbot_sim.app.runtime.simulation_app_lifecycle.close_simulation_app",
        lambda _app: calls.append("simulation_app"),
    )
    stopped = tiled_scene_runtime_factory._close_runtime_resources(
        planner_manager=SimpleNamespace(shutdown=fail_planner),
        camera_output=SimpleNamespace(close=timeout_camera),
        ik_solvers={
            "failing": SimpleNamespace(
                solver=SimpleNamespace(context=SimpleNamespace(close=fail_ik))
            ),
            "stable": SimpleNamespace(
                solver=SimpleNamespace(
                    context=SimpleNamespace(close=lambda: calls.append("stable_ik"))
                )
            ),
        },
        session=SimpleNamespace(app=object()),
        suppress_errors=True,
    )

    assert stopped is False
    assert calls == [
        "planner",
        "camera",
        "failing_ik",
        "stable_ik",
        "simulation_app",
    ]


def test_parse_tiled_joint_action_from_interactive_message() -> None:
    module = load_tiled_scene_interactive_module()

    action = module.parse_tiled_action(
        {
            "type": "step",
            "kind": "joint_delta_pos",
            "values": [0.1, -0.2, 0.3],
            "decimation": 3,
        }
    )

    assert action.kind == "joint_delta_pos"
    assert action.decimation == 3
    np.testing.assert_allclose(action.values, [0.1, -0.2, 0.3])


def test_parse_tiled_linear_path_action_with_fixed_duration() -> None:
    module = load_tiled_scene_interactive_module()

    action = module.parse_tiled_action(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_linear_path",
            "values": [0.0, 0.0, 0.1],
            "duration_s": 0.04,
            "interpolation": "linear",
        }
    )

    assert action.kind == "ee_linear_path"
    assert action.duration_s == 0.04
    assert action.decimation is None
    assert action.orientation_mode == "current"


def test_parse_tiled_linear_path_action_with_canonical_pose_fields() -> None:
    module = load_tiled_scene_interactive_module()

    action = module.parse_tiled_action(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_linear_path",
            "target_position": [[0.3, 0.0, 0.2], [0.4, 0.0, 0.2]],
            "orientation_mode": "target",
            "target_orientation_quat_wxyz": [0.0, 0.0, 0.0, 1.0],
            "pose_reference_frame": "env",
            "duration_s": 0.04,
            "sample_dt_s": 0.02,
        }
    )

    assert action.values is None
    assert action.orientation_mode == "target"
    assert action.sample_dt_s == pytest.approx(0.02)
    np.testing.assert_allclose(
        action.target_position,
        [[0.3, 0.0, 0.2], [0.4, 0.0, 0.2]],
    )
    np.testing.assert_allclose(
        action.target_orientation_wxyz,
        [0.0, 0.0, 0.0, 1.0],
    )


def test_tiled_sync_linear_path_resolves_duration_and_orientation_defaults() -> None:
    module = load_tiled_scene_interactive_module()
    planner_defaults = PlannerRequestDefaults(duration_s=0.25)
    command_defaults = RuntimeCommandDefaults(orientation_mode="current")
    inferred_target = module.parse_tiled_action(
        {
            "type": "step",
            "kind": "ee_linear_path",
            "target_offset": [0.0, 0.0, 0.1],
            "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        planner_defaults=planner_defaults,
        command_defaults=command_defaults,
    )
    explicit_duration = module.parse_tiled_action(
        {
            "type": "step",
            "kind": "ee_linear_path",
            "target_offset": [0.0, 0.0, 0.1],
            "duration_s": 0.4,
        },
        planner_defaults=planner_defaults,
        command_defaults=command_defaults,
    )
    explicit_decimation = module.parse_tiled_action(
        {
            "type": "step",
            "kind": "ee_linear_path",
            "target_offset": [0.0, 0.0, 0.1],
            "decimation": 3,
        },
        planner_defaults=planner_defaults,
        command_defaults=command_defaults,
    )

    assert inferred_target.duration_s == pytest.approx(0.25)
    assert inferred_target.orientation_mode == "target"
    assert explicit_duration.duration_s == pytest.approx(0.4)
    assert explicit_decimation.duration_s is None
    assert explicit_decimation.decimation == 3


@pytest.mark.parametrize("field", ("duration_s", "sample_dt_s"))
def test_tiled_sync_linear_path_rejects_explicit_null_numbers(field: str) -> None:
    module = load_tiled_scene_interactive_module()
    with pytest.raises(ValueError, match=f"{field} must be a number"):
        module.parse_tiled_action(
            {
                "type": "step",
                "kind": "ee_linear_path",
                "target_offset": [0.0, 0.0, 0.1],
                field: None,
            },
            planner_defaults=PlannerRequestDefaults(duration_s=0.25),
        )


@pytest.mark.parametrize("orientation_mode", ("free", "current"))
def test_tiled_sync_rejects_non_target_mode_with_target_orientation(
    orientation_mode: str,
) -> None:
    module = load_tiled_scene_interactive_module()
    with pytest.raises(ValueError, match="cannot be combined"):
        module.parse_tiled_action(
            {
                "type": "step",
                "kind": "ee_linear_path",
                "target_offset": [0.0, 0.0, 0.1],
                "orientation_mode": orientation_mode,
                "target_orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            command_defaults=RuntimeCommandDefaults(orientation_mode="target"),
        )


def test_tiled_sync_runtime_defaults_are_forwarded_by_protocol() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = SimpleNamespace(
        planner_request_defaults=PlannerRequestDefaults(duration_s=0.3),
        command_defaults=RuntimeCommandDefaults(),
    )
    captured: dict[str, object] = {}

    def step_action(action, **_kwargs):
        captured["action"] = action
        return {"event": "step"}

    runtime.step_action = step_action
    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_linear_path",
            "target_offset": [0.0, 0.0, 0.1],
        },
        runtime,
    )

    assert response["event"] == "step"
    assert captured["action"].duration_s == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("coordination", "static_others"),
        ("force_collision_refresh", True),
        ("interpolaton", "linear"),
    ),
)
def test_tiled_action_rejects_unconsumed_fields(field: str, value: object) -> None:
    module = load_tiled_scene_interactive_module()
    with pytest.raises(ValueError, match="unknown fields"):
        module.parse_tiled_action(
            {
                "type": "step",
                "kind": "joint_delta_pos",
                "values": [0.1, 0.0, 0.0],
                field: value,
            }
        )


def test_tiled_hold_rejects_unconsumed_interpolation() -> None:
    module = load_tiled_scene_interactive_module()
    with pytest.raises(ValueError, match="unknown fields"):
        module.parse_tiled_action(
            {
                "type": "step",
                "kind": "hold",
                "interpolation": "linear",
            }
        )


def test_tiled_message_handlers_have_domain_owned_module_paths() -> None:
    from linkerbot_sim.app.interactive.tiled_scene.action_messages import (
        parse_tiled_action,
    )
    from linkerbot_sim.app.interactive.tiled_scene.hand_messages import (
        load_interactive_hand_motion,
    )
    from linkerbot_sim.app.interactive.tiled_scene.plan_messages import (
        planning_request_from_message,
    )
    from linkerbot_sim.app.interactive.tiled_scene.trajectory_messages import (
        load_interactive_trajectory,
    )

    assert parse_tiled_action.__module__.endswith(".action_messages")
    assert planning_request_from_message.__module__.endswith(".plan_messages")
    assert load_interactive_trajectory.__module__.endswith(".trajectory_messages")
    assert load_interactive_hand_motion.__module__.endswith(".hand_messages")
    assert (
        importlib.util.find_spec(
            "linkerbot_sim.app.interactive.tiled_scene.planning_messages"
        )
        is None
    )


def test_parse_args_uses_isaac_runtime_without_backend_switch(monkeypatch) -> None:
    module = load_tiled_scene_interactive_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiled_scene_interactive.py",
            "--gui",
            "--default-decimation",
            "3",
            "--stdin-eof-policy",
            "keep_alive",
            "--idle-physics-policy",
            "hold_step",
            "--telemetry-rate-hz",
            "5",
            "--telemetry-primary-env-id",
            "3",
            "--max-pending-requests",
            "8",
            "--max-completed-results",
            "9",
        ],
    )

    args = module.parse_args()

    assert not hasattr(args, "backend")
    assert args.runtime_profile == "default_tiled_scene"
    assert args.env is None
    assert args.gui is True
    assert not hasattr(args, "num_envs")
    assert not hasattr(args, "robots")
    assert args.default_decimation == 3
    assert not hasattr(args, "hold")
    assert args.stdin_eof_policy == "keep_alive"
    assert args.idle_physics_policy == "hold_step"
    assert not hasattr(args, "ik_backend")
    assert not hasattr(args, "planner_backend")
    assert args.telemetry_rate_hz == 5.0
    assert args.telemetry_primary_env_id == 3
    assert args.max_pending_requests == 8
    assert args.max_completed_results == 9


def test_tiled_planner_backend_is_rejected_in_env_yaml() -> None:
    with pytest.raises(ValueError, match=r"unsupported keys: runtime.*tiled\.runtime"):
        TiledEnvConfig.from_env_config(
            {
                "tiled": {
                    "runtime": {
                        "planner": {
                            "backend": "linear",
                            "curobo_profile": "fast",
                            "joint_batch_mode": "per_env",
                        }
                    }
                },
            }
        )


def test_tiled_env_config_has_no_planner_runtime_section() -> None:
    config = TiledEnvConfig.from_env_config({})

    assert not hasattr(config, "runtime")


def test_parse_args_rejects_num_envs_cli_override(monkeypatch) -> None:
    module = load_tiled_scene_interactive_module()

    monkeypatch.setattr(
        sys,
        "argv",
        ["tiled_scene_interactive.py", "--num-envs", "8"],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_parse_args_rejects_unknown_backend_option(monkeypatch) -> None:
    module = load_tiled_scene_interactive_module()

    monkeypatch.setattr(
        sys,
        "argv",
        ["tiled_scene_interactive.py", "--backend", "debug"],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_parse_args_rejects_unknown_robot_option(monkeypatch) -> None:
    module = load_tiled_scene_interactive_module()

    monkeypatch.setattr(
        sys,
        "argv",
        ["tiled_scene_interactive.py", "--robots", "left"],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_runtime_steps_batched_joint_delta_synchronously() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "joint_delta_pos",
            "values": [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
            "decimation": 4,
        },
        runtime,
    )

    assert response["event"] == "step"
    assert response["ticks"] == 4
    assert response["step"] == 4
    np.testing.assert_allclose(
        runtime.current_positions,
        [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
    )


def test_step_message_accepts_top_level_kind() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "decimation": 1,
        },
        runtime,
    )

    assert response["event"] == "step"
    np.testing.assert_allclose(
        runtime.current_positions,
        [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
    )


@pytest.mark.parametrize(
    "message",
    (
        {"type": "joint_delta_pos", "values": [0.1, 0.0, 0.0]},
        {
            "type": "step",
            "action": {"kind": "joint_delta_pos", "values": [0.1, 0.0, 0.0]},
        },
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "joint_deltas": [0.1, 0.0, 0.0],
        },
        {
            "type": "step",
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "action": {},
        },
        {"type": "step", "kind": "hold", "values": []},
    ),
)
def test_tiled_action_rejects_unknown_shapes(message) -> None:
    module = load_tiled_scene_interactive_module()
    response = module.handle_tiled_interactive_message(
        {**message, "env_ids": [0, 1]}, make_runtime()
    )

    assert response["event"] == "rejected"


def test_step_env_ids_updates_selected_envs_and_holds_others() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "kind": "joint_delta_pos",
            "env_ids": [1],
            "values": [[0.1, 0.0, 0.0]],
            "decimation": 2,
        },
        runtime,
    )

    assert response["event"] == "step"
    assert response["env_ids"] == [1]
    np.testing.assert_allclose(runtime.current_positions[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(runtime.current_positions[1], [2.1, 2.0, 2.0])
    assert runtime.episode_steps.tolist() == [2, 2]


def test_step_message_passes_robot_selection_to_runtime() -> None:
    module = load_tiled_scene_interactive_module()

    class FakeRuntime:
        quit_event = None
        robot_names = ("robot_0", "robot_1")

        def step_action(self, action, *, env_ids=None, robot_names=None):
            return {
                "event": "step",
                "kind": action.kind,
                "env_ids": None if env_ids is None else env_ids.tolist(),
                "robots": list(robot_names),
            }

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "env_ids": [0],
            "robot_ids": [1],
        },
        FakeRuntime(),
    )

    assert response == {
        "event": "step",
        "kind": "joint_delta_pos",
        "env_ids": [0],
        "robot_ids": [1],
    }


def test_tiled_responses_replace_robot_keyed_info_and_trajectory_with_ids() -> None:
    module = load_tiled_scene_interactive_module()

    class FakeRuntime:
        quit_event = None
        robot_names = ("left_arm", "right_arm")

        def step_action(self, action, *, env_ids=None, robot_names=None):
            return {
                "event": "step",
                "robots": ["left_arm", "right_arm"],
                "info": {
                    "left_arm": {"command_width": 6},
                    "right_arm": {"command_width": 7},
                },
            }

        def step_trajectory(self, *, env_ids=None, robot_names=None, decimation=None):
            return {
                "event": "trajectory_step",
                "trajectory": {
                    "left_arm": {"active_env_ids": [0]},
                    "right_arm": {"active_env_ids": [1]},
                },
            }

    runtime = FakeRuntime()
    stepped = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "joint_delta_pos",
            "values": [0.1],
        },
        runtime,
    )
    trajectory = module.handle_tiled_interactive_message(
        {"type": "step_trajectory", "env_ids": [0, 1], "robot_ids": "all"},
        runtime,
    )

    assert stepped == {
        "event": "step",
        "robot_ids": [0, 1],
        "info": [
            {"robot_id": 0, "command_width": 6},
            {"robot_id": 1, "command_width": 7},
        ],
    }
    assert trajectory == {
        "event": "trajectory_step",
        "trajectory": [
            {"robot_id": 0, "active_env_ids": [0]},
            {"robot_id": 1, "active_env_ids": [1]},
        ],
    }


def test_step_message_preserves_explicit_all_robot_selection() -> None:
    module = load_tiled_scene_interactive_module()
    from linkerbot_sim.app.interactive.tiled_scene.selectors import ALL_ROBOTS

    class FakeRuntime:
        quit_event = None
        robot_names = ("robot_0", "robot_1")

        def step_action(self, action, *, env_ids=None, robot_names=None):
            return {
                "event": "step",
                "robot_names_is_all": robot_names is ALL_ROBOTS,
            }

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "robot_ids": "all",
        },
        FakeRuntime(),
    )

    assert response == {"event": "step", "robot_names_is_all": True}


def test_tiled_message_rejects_unknown_selector_field() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "unexpected_selector": 0,
        },
        runtime,
    )

    assert response["event"] == "rejected"
    assert "unknown fields" in response["error"]


def test_runtime_ee_delta_reports_ik_mask_and_fallback() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime(fail_env_ids={1})
    runtime.current_positions[:] = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "decimation": 1,
        },
        runtime,
    )

    assert response["event"] == "step"
    assert response["info"]["ik_success"] == [True, False]
    assert response["info"]["failed_env_ids"] == [1]
    # env 0 使用 debug IK 目标；env 1 失败后保持 seed/current_positions。
    np.testing.assert_allclose(runtime.current_positions[0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(runtime.current_positions[1], [2.0, 2.0, 2.0])


def test_runtime_reject_request_reports_only_selected_failed_envs_atomically() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime(
        fail_env_ids={0, 1},
        failure_policy="reject_request",
    )
    runtime.current_positions[:] = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    runtime.adapter.last_target = np.asarray(
        [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=float
    )
    initial_positions = runtime.current_positions.copy()
    initial_last_target = runtime.adapter.last_target.copy()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "kind": "ee_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "env_ids": [0],
            "decimation": 2,
        },
        runtime,
    )

    assert response == {
        "event": "rejected",
        "accepted": False,
        "code": "ik_failure",
        "error": "synchronous IK request rejected; failed env_ids: [0]",
        "failure_policy": "reject_request",
        "failed_env_ids": [0],
    }
    assert runtime.step == 0
    assert runtime.episode_steps.tolist() == [0, 0]
    np.testing.assert_allclose(runtime.current_positions, initial_positions)
    np.testing.assert_allclose(runtime.adapter.last_target, initial_last_target)


def test_planner_status_continues_after_one_playback_capacity_rejection() -> None:
    runtime = make_runtime()
    runtime.trajectory_buffer = TiledTrajectoryBuffer(
        num_envs=2,
        max_samples_per_env=2,
    )

    def result(request_id: str, env_id: int, samples: int) -> TiledPlanningResult:
        times = np.linspace(0.0, 0.1, samples)
        return TiledPlanningResult(
            request_id=request_id,
            robot_name="debug",
            env_ids=(env_id,),
            success=True,
            status="SUCCESS",
            message="",
            times=times,
            positions=np.zeros((1, samples, 3), dtype=float),
            joint_names=("joint_0", "joint_1", "joint_2"),
        )

    ready = (
        result("first", 0, 2),
        result("oversized", 0, 3),
        result("third", 1, 2),
    )
    runtime.planner_manager = SimpleNamespace(
        collect_ready=lambda *, timeout_s=0.0: ready,
        status=lambda: {"completed_count": 3},
    )

    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {"type": "planner_status"}, runtime
    )

    assert response["event"] == "planner_status"
    assert [item["request_id"] for item in response["ready"]] == [
        "first",
        "oversized",
        "third",
    ]
    assert [item["request_id"] for item in response["loaded"]] == ["first", "third"]
    assert response["load_rejected"] == [
        {
            "request_id": "oversized",
            "robot_id": 0,
            "env_ids": [0],
            "error": "trajectory playback capacity exceeded; env 0: samples=3>2",
            "code": "playback_capacity_exceeded",
        }
    ]
    status = runtime.trajectory_buffer.status(robot_name="debug")
    assert [item["request_id"] for item in status["robots"]["debug"]["envs"]] == [
        "first",
        "third",
    ]


@pytest.mark.parametrize("second_behavior", ("failed", "raises"))
def test_isaac_multi_robot_ik_rejection_preflights_before_any_physics_write(
    monkeypatch: pytest.MonkeyPatch,
    second_behavior: str,
) -> None:
    module = load_tiled_scene_interactive_module()
    apply_calls: list[object] = []
    solver_calls: list[str] = []

    class FakeView:
        def __init__(self, positions: np.ndarray) -> None:
            self.positions = np.asarray(positions, dtype=float).copy()

        def get_joint_positions(self, *, joint_indices=None):
            columns = (
                np.arange(self.positions.shape[1])
                if joint_indices is None
                else np.asarray(joint_indices, dtype=int)
            )
            return self.positions[:, columns]

    class FakeWorld:
        steps = 0

        def get_physics_dt(self) -> float:
            return 0.01

        def step(self, *, render: bool) -> None:
            self.steps += 1

    class FakeSolver:
        tcp_frame_name = "tool"

        def __init__(
            self,
            name: str,
            *,
            fail_env_ids: set[int],
            raises: bool = False,
        ) -> None:
            self.name = name
            self.fail_env_ids = fail_env_ids
            self.raises = raises

        def solve(self, *, target_positions, seeds, **_kwargs):
            solver_calls.append(self.name)
            if self.raises:
                raise RuntimeError("second robot IK failed")
            success = np.ones(len(seeds), dtype=bool)
            success[list(self.fail_env_ids)] = False
            return BatchIKResult(
                joint_positions=np.asarray(seeds, dtype=float) + 1.0,
                success=success,
                position_error=np.zeros(len(seeds), dtype=float),
            )

    world = FakeWorld()
    names = ("left", "right")
    initial_targets = {
        "left": np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=float),
        "right": np.asarray([[3.0, 3.0], [4.0, 4.0]], dtype=float),
    }
    views = {name: FakeView(initial_targets[name]) for name in names}
    adapters = {
        "left": TiledCommandAdapter(
            num_envs=2,
            command_dim=2,
            tcp_frame_name="tool",
            ik_solver=FakeSolver("left", fail_env_ids=set()),
            failure_policy="reject_request",
        ),
        "right": TiledCommandAdapter(
            num_envs=2,
            command_dim=2,
            tcp_frame_name="tool",
            ik_solver=FakeSolver(
                "right",
                fail_env_ids=({1} if second_behavior == "failed" else set()),
                raises=second_behavior == "raises",
            ),
            failure_policy="reject_request",
        ),
    }
    for name, adapter in adapters.items():
        adapter.last_target = initial_targets[name].copy()
    runtime = TiledSceneRuntime(
        env_name="unit",
        env_config={},
        session=SimpleNamespace(world=world, stage=object(), app=object()),
        scene=SimpleNamespace(
            config=SimpleNamespace(num_envs=2),
            env_origins=np.zeros((2, 3), dtype=float),
            articulation_views={
                name: SimpleNamespace(
                    view=views[name],
                    command_joint_indices=np.asarray([0, 1], dtype=int),
                )
                for name in names
            },
        ),
        render=False,
        default_decimation=2,
        robot_names=names,
        episode_steps=np.zeros(2, dtype=int),
        episode_ids=np.zeros(2, dtype=int),
        initial_joint_positions={},
        initial_joint_velocities={},
        target_positions={
            name: values.copy() for name, values in initial_targets.items()
        },
        initial_object_states={},
        command_adapters=adapters,
        ik_solvers={name: object() for name in names},
        tcp_positions_world={name: np.zeros((2, 3), dtype=float) for name in names},
        tcp_orientations_wxyz={
            name: np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)) for name in names
        },
        trajectory_buffer=TiledTrajectoryBuffer(num_envs=2),
        planner_manager=SimpleNamespace(),
        sensor_cameras=(),
        camera_output=None,
        quit_event=threading.Event(),
    )
    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.runtime.stepping._apply_joint_targets",
        lambda *args, **kwargs: apply_calls.append((args, kwargs)),
    )

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "robot_ids": "all",
        },
        runtime,
    )

    assert response["event"] == "rejected"
    if second_behavior == "failed":
        assert response["failed_env_ids"] == [1]
    else:
        assert response["error"] == "second robot IK failed"
        assert "failed_env_ids" not in response
    assert solver_calls == ["left", "right"]
    assert apply_calls == []
    assert world.steps == 0
    assert runtime.step == 0
    assert runtime.episode_steps.tolist() == [0, 0]
    for name in names:
        np.testing.assert_allclose(
            runtime.target_positions[name], initial_targets[name]
        )
        np.testing.assert_allclose(adapters[name].last_target, initial_targets[name])


def test_runtime_ee_linear_path_uses_equal_duration_and_ticks_for_all_envs() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_linear_path",
            "values": [0.0, 0.0, 0.2],
            "duration_s": 0.04,
            "interpolation": "linear",
        },
        runtime,
    )

    assert response["event"] == "step"
    assert response["ticks"] == 4
    assert response["duration_s"] == pytest.approx(0.04)
    assert response["episode_steps"] == [4, 4]
    assert response["info"]["ik_success"] == [True, True]
    assert response["info"]["ik_completed_steps"] == [4, 4]
    assert "ik_orientation_error" in response["info"]
    np.testing.assert_allclose(
        runtime.current_tcp_positions,
        runtime.origins + np.asarray([0.0, 0.0, 0.2]),
    )


def test_runtime_ee_linear_path_supports_absolute_target_and_target_orientation() -> (
    None
):
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "kind": "ee_linear_path",
            "env_ids": [1],
            "target_position": [0.3, 0.1, 0.2],
            "pose_reference_frame": "env",
            "orientation_mode": "target",
            "target_orientation_quat_wxyz": [0.0, 0.0, 0.0, 1.0],
            "duration_s": 0.04,
            "interpolation": "linear",
        },
        runtime,
    )

    assert response["event"] == "step"
    assert response["info"]["ik_success"] == [True, True]
    np.testing.assert_allclose(runtime.current_tcp_positions[0], runtime.origins[0])
    np.testing.assert_allclose(
        runtime.current_tcp_positions[1],
        runtime.origins[1] + np.asarray([0.3, 0.1, 0.2]),
    )
    np.testing.assert_allclose(
        runtime.current_tcp_orientations_wxyz[1],
        [0.0, 0.0, 0.0, 1.0],
        atol=1.0e-8,
    )


@pytest.mark.parametrize(
    "message,error",
    (
        (
            {
                "type": "step",
                "kind": "ee_linear_path",
                "target_offset": [0.1, 0.0, 0.0],
                "target_position": [0.2, 0.0, 0.0],
            },
            "exactly one",
        ),
        (
            {
                "type": "step",
                "kind": "ee_linear_path",
                "target_offset": [0.1, 0.0, 0.0],
                "orientation_mode": "target",
            },
            "target_orientation_quat_wxyz",
        ),
        (
            {
                "type": "step",
                "kind": "ee_linear_path",
                "target_offset": [0.1, 0.0, 0.0],
                "orientation_mode": "none",
            },
            "orientation_mode",
        ),
    ),
)
def test_runtime_ee_linear_path_rejects_invalid_canonical_targets(
    message, error
) -> None:
    module = load_tiled_scene_interactive_module()

    response = module.handle_tiled_interactive_message(
        {**message, "env_ids": [0, 1]}, make_runtime()
    )

    assert response["event"] == "rejected"
    assert error in response["error"]


def test_runtime_ee_linear_path_freezes_failed_env_without_shortening_step() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime(fail_env_ids={1})
    runtime.current_positions[:] = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_linear_path",
            "values": [0.1, 0.0, 0.0],
            "decimation": 3,
            "interpolation": "linear",
        },
        runtime,
    )

    assert response["event"] == "step"
    assert response["ticks"] == 3
    assert response["episode_steps"] == [3, 3]
    assert response["info"]["ik_success"] == [True, False]
    assert response["info"]["ik_first_failure_step"] == [-1, 1]
    assert response["info"]["ik_completed_steps"] == [3, 0]
    np.testing.assert_allclose(runtime.current_positions[1], [2.0, 2.0, 2.0])


def test_runtime_ee_linear_path_solver_exception_is_atomic() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    initial_positions = runtime.current_positions.copy()

    class RaisingSolver:
        calls = 0

        def solve(
            self,
            *,
            target_positions,
            target_orientations_wxyz,
            seeds,
            tcp_frame_name,
        ):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("IK backend failed")
            return BatchIKResult(
                joint_positions=np.asarray(seeds, dtype=float),
                success=np.ones(len(seeds), dtype=bool),
                position_error=np.zeros(len(seeds), dtype=float),
            )

    runtime.adapter.ik_solver = RaisingSolver()
    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_linear_path",
            "values": [0.1, 0.0, 0.0],
            "decimation": 3,
        },
        runtime,
    )

    assert response["event"] == "rejected"
    assert "IK backend failed" in response["error"]
    assert runtime.step == 0
    assert runtime.episode_steps.tolist() == [0, 0]
    np.testing.assert_allclose(runtime.current_positions, initial_positions)


def test_runtime_ee_linear_path_allows_unaligned_duration_and_sparse_ik() -> None:
    module = load_tiled_scene_interactive_module()

    runtime = make_runtime()
    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_linear_path",
            "target_offset": [0.1, 0.0, 0.0],
            "duration_s": 0.025,
            "sample_dt_s": 0.02,
            "interpolation": "linear",
        },
        runtime,
    )

    assert response["event"] == "step"
    assert response["ticks"] == 3
    assert response["duration_s"] == pytest.approx(0.03)
    assert response["sample_dt_s"] == pytest.approx(0.02)
    assert response["ik_waypoints"] == 2
    assert response["info"]["ik_completed_steps"] == [2, 2]
    np.testing.assert_allclose(
        runtime.current_tcp_positions,
        runtime.origins + np.asarray([0.1, 0.0, 0.0]),
    )


def test_runtime_ee_linear_path_rejects_duration_with_decimation() -> None:
    module = load_tiled_scene_interactive_module()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0, 1],
            "kind": "ee_linear_path",
            "values": [0.1, 0.0, 0.0],
            "duration_s": 0.04,
            "decimation": 4,
        },
        make_runtime(),
    )

    assert response["event"] == "rejected"
    assert "cannot both be set" in response["error"]


def test_runtime_status_reset_and_quit_controls() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    status = module.handle_tiled_interactive_message({"type": "status"}, runtime)
    reset = module.handle_tiled_interactive_message(
        {"type": "reset", "env_ids": [0, 1]}, runtime
    )
    quit_response = module.handle_tiled_interactive_message({"type": "quit"}, runtime)

    assert status["event"] == "status"
    assert status["num_envs"] == 2
    assert reset["event"] == "reset"
    assert reset["accepted"] is True
    assert quit_response == {"event": "quit", "accepted": True}
    assert runtime.quit_event.is_set()


@pytest.mark.parametrize(
    "message_type",
    (
        "reset",
        "get_state",
        "set_state",
        "load_trajectory",
        "step_trajectory",
        "trajectory_status",
        "clear_trajectory",
        "hand",
        "plan",
        "cancel_plan",
        "step",
    ),
)
def test_env_scoped_protocol_requires_env_ids(message_type: str) -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {"type": message_type},
        make_runtime(),
    )

    assert response["event"] == "rejected"
    assert response["error"] == f"{message_type}.env_ids is required"


@pytest.mark.parametrize(
    "method_name",
    (
        "reset",
        "step_action",
        "get_state",
        "set_state",
        "load_trajectory",
        "step_trajectory",
        "submit_plan",
        "submit_hand_motion",
        "trajectory_status",
        "clear_trajectory",
    ),
)
def test_runtime_env_scoped_apis_require_env_ids(method_name: str) -> None:
    signature = inspect.signature(getattr(TiledSceneRuntime, method_name))

    assert signature.parameters["env_ids"].default is inspect.Parameter.empty


def test_cancel_plan_by_request_id_does_not_require_env_ids() -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {"type": "cancel_plan", "request_id": "plan-missing"}, make_runtime()
    )

    assert response["event"] == "plan_cancelled"


@pytest.mark.parametrize("request_id", (None, "", "   "))
def test_cancel_plan_rejects_invalid_request_id_without_env_ids(
    request_id: object,
) -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {"type": "cancel_plan", "request_id": request_id},
        make_runtime(),
    )

    assert response["event"] == "rejected"
    assert "cancel_plan.request_id must be a non-empty string" in response["error"]


def test_protocol_allows_scoped_and_global_controls() -> None:
    runtime = make_runtime()

    reset = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {"type": "reset", "env_ids": [1]},
        runtime,
    )
    status = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {"type": "status"},
        runtime,
    )

    assert reset["event"] == "reset"
    assert reset["env_ids"] == [1]
    assert status["event"] == "status"


@pytest.mark.parametrize(
    "message",
    (
        {"type": "status", "id": "status-1"},
        {"type": "reset", "env_ids": [0], "id": "reset-1"},
        {"type": "planner_status", "request_id": "plan-1"},
        {"type": "quit", "accepted": True},
    ),
)
def test_tiled_controls_reject_unknown_fields(message: dict[str, object]) -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        message, make_runtime()
    )

    assert response["event"] == "rejected"
    assert "unknown fields" in response["error"]


@pytest.mark.parametrize("invalid", (None, True, "0.1", float("inf")))
def test_planner_timeout_requires_finite_json_number(invalid: object) -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {"type": "planner_status", "wait_timeout_s": invalid}, make_runtime()
    )

    assert response["event"] == "rejected"
    assert "planner_status.wait_timeout_s must be" in response["error"]


@pytest.mark.parametrize("invalid_value", (False, 1.9, "2"))
@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("env_id", {"type": "get_snapshot"}),
        (
            "source_env_id",
            {"type": "clone_state", "target_env_ids": [1]},
        ),
        ("env_ids", {"type": "reset"}),
        (
            "target_env_ids",
            {"type": "clone_state", "source_env_id": 0},
        ),
    ),
)
def test_protocol_env_selectors_require_json_integers(
    field: str,
    message: dict[str, object],
    invalid_value: object,
) -> None:
    selector_value = (
        [invalid_value] if field in {"env_ids", "target_env_ids"} else invalid_value
    )

    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {**message, field: selector_value}, make_runtime()
    )

    assert response["event"] == "rejected"
    assert response["error"] == (
        f"{field}[0] must be a JSON integer"
        if isinstance(selector_value, list)
        else f"{field} must be a JSON integer"
    )


@pytest.mark.parametrize("invalid_value", (False, 1.0, "1"))
@pytest.mark.parametrize("field", ("robot_id", "robot_ids"))
def test_protocol_robot_selectors_require_nonnegative_json_integers(
    field: str,
    invalid_value: object,
) -> None:
    value = [invalid_value] if field == "robot_ids" else invalid_value
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {"type": "step_trajectory", "env_ids": [0, 1], field: value},
        make_runtime(),
    )

    assert response["event"] == "rejected"
    assert "robot ID must be a non-negative integer" in response["error"]


@pytest.mark.parametrize(
    ("message", "error"),
    (
        ({"type": "reset", "env_ids": [0, 0]}, "env_ids cannot contain duplicates"),
        (
            {
                "type": "clone_state",
                "source_env_id": 0,
                "target_env_ids": [1, 1],
            },
            "target_env_ids cannot contain duplicates",
        ),
        ({"type": "get_snapshot", "env_id": 2}, "env_id is out of range"),
        (
            {
                "type": "clone_state",
                "source_env_id": 2,
                "target_env_ids": [1],
            },
            "env_id is out of range",
        ),
        (
            {
                "type": "clone_state",
                "source_env_id": 0,
                "target_env_ids": [2],
            },
            "env_ids contains out-of-range env id",
        ),
    ),
)
def test_protocol_env_selectors_still_validate_duplicates_and_range(
    message: dict[str, object], error: str
) -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        message, make_runtime()
    )

    assert response == {"event": "rejected", "error": error}


def test_transport_rejects_missing_env_ids() -> None:
    response = tiled_scene_transport._handle_json_line(
        '{"type":"reset"}',
        make_runtime(),
    )

    assert response == {"event": "rejected", "error": "reset.env_ids is required"}


def test_interactive_loop_processes_queued_requests_on_main_loop() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    request_queue = queue.Queue()
    status_queue = queue.Queue()
    quit_queue = queue.Queue()
    request_queue.put(
        tiled_scene_transport._InteractiveRequest(
            line='{"type":"status"}',
            source="unit",
            response_queue=status_queue,
        )
    )
    request_queue.put(
        tiled_scene_transport._InteractiveRequest(
            line='{"type":"quit"}',
            source="unit",
            response_queue=quit_queue,
        )
    )

    module.run_interactive_loop(
        runtime,
        telemetry=None,
        request_queue=request_queue,
        telemetry_rate_hz=0.0,
    )

    assert status_queue.get_nowait()["event"] == "status"
    assert quit_queue.get_nowait() == {"event": "quit", "accepted": True}
    assert runtime.quit_event.is_set()


def test_interactive_loop_idles_gui_runtime_with_hold_step_policy() -> None:
    module = load_tiled_scene_interactive_module()

    class FakeRuntime:
        render = True
        idle_period_s = 0.001

        def __init__(self) -> None:
            self.quit_event = threading.Event()
            self.idle_steps = 0
            self.session = SimpleNamespace(
                world=SimpleNamespace(get_physics_dt=lambda: 0.001)
            )

        def idle_step(self) -> None:
            self.idle_steps += 1
            self.quit_event.set()

    runtime = FakeRuntime()

    module.run_interactive_loop(
        runtime,
        telemetry=None,
        request_queue=queue.Queue(),
        telemetry_rate_hz=0.0,
        idle_physics_policy="hold_step",
        idle_step_duration_s=0.001,
    )

    assert runtime.idle_steps == 1


def test_interactive_loop_does_not_idle_gui_runtime_with_pause_policy() -> None:
    module = load_tiled_scene_interactive_module()

    class FakeRuntime:
        render = True
        idle_period_s = 0.001

        def __init__(self) -> None:
            self.quit_event = threading.Event()
            self.idle_steps = 0

        def idle_step(self) -> None:
            self.idle_steps += 1

        def quit(self) -> dict[str, object]:
            self.quit_event.set()
            return {"event": "quit", "accepted": True}

    runtime = FakeRuntime()
    request_queue: queue.Queue[object] = queue.Queue()
    request_queue.put(
        tiled_scene_transport._InteractiveRequest(
            line='{"type":"quit"}',
            source="unit",
        )
    )

    module.run_interactive_loop(
        runtime,
        telemetry=None,
        request_queue=request_queue,
        telemetry_rate_hz=0.0,
    )

    assert runtime.idle_steps == 0


def test_stdin_eof_quit_policy_keeps_tcp_and_telemetry_alive() -> None:
    assert (
        tiled_scene_transport._quit_on_stdin_eof(
            stdin_eof_policy="exit",
            tcp_jsonl_port=None,
            telemetry=None,
        )
        is True
    )
    assert (
        tiled_scene_transport._quit_on_stdin_eof(
            stdin_eof_policy="keep_alive",
            tcp_jsonl_port=None,
            telemetry=None,
        )
        is False
    )
    assert (
        tiled_scene_transport._quit_on_stdin_eof(
            stdin_eof_policy="exit",
            tcp_jsonl_port=8765,
            telemetry=None,
        )
        is False
    )
    assert (
        tiled_scene_transport._quit_on_stdin_eof(
            stdin_eof_policy="exit",
            tcp_jsonl_port=None,
            telemetry=object(),
        )
        is False
    )
    assert (
        tiled_scene_transport._quit_on_stdin_eof(
            stdin_eof_policy="exit",
            tcp_jsonl_port=None,
            telemetry=None,
            keepalive_consumer_active=True,
        )
        is False
    )


def test_interactive_loop_zero_telemetry_rate_never_samples_state() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.quit_event = threading.Event()
            self.get_state_calls = 0

        def status(self) -> dict[str, object]:
            return {"event": "status"}

        def quit(self) -> dict[str, object]:
            self.quit_event.set()
            return {"event": "quit", "accepted": True}

        def get_state(self, **_kwargs: object) -> dict[str, object]:
            self.get_state_calls += 1
            return {"step": 0, "time_s": 0.0, "env_ids": [0], "state": {}}

    runtime = Runtime()
    requests = tiled_scene_transport.BoundedInteractiveRequestQueue(capacity=2)
    requests.put_nowait(
        tiled_scene_transport._InteractiveRequest(
            line='{"type":"status"}', source="test"
        )
    )
    requests.put_nowait(
        tiled_scene_transport._InteractiveRequest(line='{"type":"quit"}', source="test")
    )

    tiled_scene_transport.run_interactive_loop(
        runtime,
        telemetry=object(),  # type: ignore[arg-type]
        request_queue=requests,
        telemetry_rate_hz=0.0,
        queue_poll_timeout_s=0.01,
    )

    assert runtime.get_state_calls == 0


def test_publish_state_telemetry_samples_selected_envs() -> None:
    load_tiled_scene_interactive_module()
    runtime = make_runtime()
    published = []

    class FakeTelemetry:
        config = SimpleNamespace(selected_env_ids=(1,))

        def publish_interactive_state(
            self, state_response, *, event, trigger_response=None
        ):
            published.append(
                {
                    "state_response": state_response,
                    "event": event,
                    "trigger_response": trigger_response,
                }
            )
            return True

    tiled_scene_telemetry_publish._publish_state_telemetry(
        FakeTelemetry(), runtime, event="state"
    )

    assert published[0]["event"] == "state"
    assert published[0]["state_response"]["env_ids"] == [1]


def test_publish_response_telemetry_resamples_configured_envs() -> None:
    load_tiled_scene_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]
    published = []

    class FakeTelemetry:
        config = SimpleNamespace(selected_env_ids=(0,))

        def publish_interactive_state(
            self, state_response, *, event, trigger_response=None
        ):
            published.append(
                {
                    "state_response": state_response,
                    "event": event,
                    "trigger_response": trigger_response,
                }
            )
            return True

    tiled_scene_telemetry_publish._publish_response_telemetry(
        FakeTelemetry(),
        runtime,
        {
            "event": "state",
            "env_ids": [1],
            "state": {
                "robots": [{"robot_id": 0, "joint_positions": [[9.0, 0.0, 0.0]]}]
            },
        },
    )

    assert published[0]["state_response"]["env_ids"] == [0]
    assert published[0]["state_response"]["state"]["robots"]["debug"][
        "joint_positions"
    ] == [[1.0, 0.0, 0.0]]


def test_restore_tiled_object_pose_snapshot_uses_selected_env_paths(
    monkeypatch,
) -> None:
    from linkerbot_sim.tiled.state.object_io import (
        restore_tiled_object_pose_snapshot,
    )

    calls = []

    def fake_apply(stage, prim_path, position, orientation):
        calls.append(
            (prim_path, np.asarray(position).tolist(), np.asarray(orientation).tolist())
        )
        return True

    monkeypatch.setattr(
        "linkerbot_sim.tiled.state.object_io.apply_prim_local_pose_and_zero_velocity",
        fake_apply,
    )

    restored = restore_tiled_object_pose_snapshot(
        stage=object(),
        object_prim_paths={
            "Tblock": (
                "/World/envs/env_0/TBlock",
                "/World/envs/env_1/TBlock",
            )
        },
        snapshot={
            "Tblock": {
                "env_ids": np.asarray([0, 1], dtype=int),
                "positions_local": np.asarray(
                    [[0.1, 0.0, -0.4], [0.2, 0.0, -0.4]],
                    dtype=float,
                ),
                "orientations_wxyz": np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    dtype=float,
                ),
            }
        },
        env_ids=np.asarray([1], dtype=int),
    )

    assert restored == 1
    assert calls == [
        (
            "/World/envs/env_1/TBlock",
            [0.2, 0.0, -0.4],
            [0.0, 0.0, 0.0, 1.0],
        )
    ]


def test_read_tiled_object_states_prefers_rigid_view_world_pose() -> None:
    from linkerbot_sim.tiled.state.object_io import (
        read_tiled_object_states,
    )

    class FakeRigidView:
        def get_world_poses(self, *, indices):
            assert np.asarray(indices, dtype=int).tolist() == [1]
            return (
                np.asarray([[3.2, 0.1, -0.4]], dtype=float),
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
            )

    state = read_tiled_object_states(
        stage=object(),
        object_prim_paths={
            "Tblock": (
                "/World/envs/env_0/TBlock",
                "/World/envs/env_1/TBlock",
            )
        },
        env_origins=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        env_ids=np.asarray([1], dtype=int),
        object_pose_views={"Tblock": FakeRigidView()},
    )

    assert state == {
        "Tblock": {
            "env_ids": [1],
            "positions_world": [[3.2, 0.1, -0.4]],
            "positions_local": [[0.20000000000000018, 0.1, -0.4]],
            "orientations_wxyz": [[1.0, 0.0, 0.0, 0.0]],
        }
    }


def test_restore_tiled_object_pose_snapshot_uses_rigid_view_when_available(
    monkeypatch,
) -> None:
    from linkerbot_sim.tiled.state.object_io import (
        restore_tiled_object_pose_snapshot,
    )

    usd_calls = []
    view_calls = []

    def fake_apply(stage, prim_path, position, orientation):
        usd_calls.append((prim_path, position, orientation))
        return True

    class FakeRigidView:
        def set_world_poses(self, *, positions, orientations, indices):
            view_calls.append(
                (
                    "poses",
                    np.asarray(positions, dtype=float).tolist(),
                    np.asarray(orientations, dtype=float).tolist(),
                    np.asarray(indices, dtype=int).tolist(),
                )
            )

        def set_velocities(self, velocities, *, indices):
            view_calls.append(
                (
                    "velocities",
                    np.asarray(velocities, dtype=float).tolist(),
                    np.asarray(indices, dtype=int).tolist(),
                )
            )

    monkeypatch.setattr(
        "linkerbot_sim.tiled.state.object_io.apply_prim_local_pose_and_zero_velocity",
        fake_apply,
    )

    restored = restore_tiled_object_pose_snapshot(
        stage=object(),
        object_prim_paths={
            "Tblock": (
                "/World/envs/env_0/TBlock",
                "/World/envs/env_1/TBlock",
            )
        },
        snapshot={
            "Tblock": {
                "env_ids": np.asarray([0, 1], dtype=int),
                "positions_local": np.asarray(
                    [[0.1, 0.0, -0.4], [0.2, 0.1, -0.4]],
                    dtype=float,
                ),
                "orientations_wxyz": np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    dtype=float,
                ),
            }
        },
        env_ids=np.asarray([1], dtype=int),
        env_origins=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        object_pose_views={"Tblock": FakeRigidView()},
    )

    assert restored == 1
    assert usd_calls == []
    assert view_calls == [
        (
            "poses",
            [[3.2, 0.1, -0.4]],
            [[0.0, 0.0, 0.0, 1.0]],
            [1],
        ),
        ("velocities", [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [1]),
    ]


def test_read_tiled_object_states_raises_when_rigid_view_fails() -> None:
    from linkerbot_sim.tiled.state.object_io import (
        read_tiled_object_states,
    )

    class FailingRigidView:
        def get_world_poses(self, *, indices):
            raise RuntimeError("view read failed")

    with pytest.raises(RuntimeError, match="failed to read tiled object 'Tblock'"):
        read_tiled_object_states(
            stage=object(),
            object_prim_paths={
                "Tblock": (
                    "/World/envs/env_0/TBlock",
                    "/World/envs/env_1/TBlock",
                )
            },
            env_origins=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
            env_ids=np.asarray([1], dtype=int),
            object_pose_views={"Tblock": FailingRigidView()},
        )


def test_restore_tiled_object_pose_snapshot_raises_when_rigid_view_fails(
    monkeypatch,
) -> None:
    from linkerbot_sim.tiled.state.object_io import (
        restore_tiled_object_pose_snapshot,
    )

    usd_calls = []

    def fake_apply(stage, prim_path, position, orientation):
        usd_calls.append((prim_path, position, orientation))
        return True

    class FailingRigidView:
        def set_world_poses(self, *, positions, orientations, indices):
            raise RuntimeError("view write failed")

        def set_velocities(self, velocities, *, indices):
            raise AssertionError("velocity reset should not run after pose failure")

    monkeypatch.setattr(
        "linkerbot_sim.tiled.state.object_io.apply_prim_local_pose_and_zero_velocity",
        fake_apply,
    )

    with pytest.raises(RuntimeError, match="failed to restore tiled object 'Tblock'"):
        restore_tiled_object_pose_snapshot(
            stage=object(),
            object_prim_paths={
                "Tblock": (
                    "/World/envs/env_0/TBlock",
                    "/World/envs/env_1/TBlock",
                )
            },
            snapshot={
                "Tblock": {
                    "env_ids": np.asarray([0, 1], dtype=int),
                    "positions_local": np.asarray(
                        [[0.1, 0.0, -0.4], [0.2, 0.1, -0.4]],
                        dtype=float,
                    ),
                    "orientations_wxyz": np.asarray(
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        dtype=float,
                    ),
                }
            },
            env_ids=np.asarray([1], dtype=int),
            env_origins=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
            object_pose_views={"Tblock": FailingRigidView()},
        )

    assert usd_calls == []


def test_tiled_dynamic_chain_object_view_captures_and_restores_child_bodies(
    monkeypatch,
) -> None:
    from linkerbot_sim.tiled.state.object_io import (
        capture_tiled_object_pose_snapshot,
        restore_tiled_object_pose_snapshot,
    )
    from linkerbot_sim.tiled.state.object_views import (
        TiledDynamicChainObjectPoseView,
    )

    usd_calls = []
    view_calls = []

    def fake_apply(stage, prim_path, position, orientation):
        usd_calls.append((prim_path, position, orientation))
        return True

    class FakeRigidBodyView:
        def get_world_poses(self, *, indices):
            assert np.asarray(indices, dtype=int).tolist() == [0, 1, 2, 3]
            return (
                np.asarray(
                    [
                        [0.0, 0.0, -0.1],
                        [0.2, 0.0, -0.1],
                        [3.0, 0.1, -0.1],
                        [3.2, 0.1, -0.1],
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                ),
            )

        def set_world_poses(self, *, positions, orientations, indices):
            view_calls.append(
                (
                    "poses",
                    np.asarray(positions, dtype=float).tolist(),
                    np.asarray(orientations, dtype=float).tolist(),
                    np.asarray(indices, dtype=int).tolist(),
                )
            )

        def set_velocities(self, velocities, *, indices):
            view_calls.append(
                (
                    "velocities",
                    np.asarray(velocities, dtype=float).tolist(),
                    np.asarray(indices, dtype=int).tolist(),
                )
            )

    monkeypatch.setattr(
        "linkerbot_sim.tiled.state.object_io.apply_prim_local_pose_and_zero_velocity",
        fake_apply,
    )

    object_view = TiledDynamicChainObjectPoseView(
        view=FakeRigidBodyView(),
        body_names=("endpoint_0", "segment_0"),
        body_paths_by_env=(
            (
                "/World/envs/env_0/CapsuleRope/endpoint_0",
                "/World/envs/env_0/CapsuleRope/segment_0",
            ),
            (
                "/World/envs/env_1/CapsuleRope/endpoint_0",
                "/World/envs/env_1/CapsuleRope/segment_0",
            ),
        ),
        reference_body="endpoint_0",
    )

    snapshot = capture_tiled_object_pose_snapshot(
        stage=object(),
        object_prim_paths={
            "rope": (
                "/World/envs/env_0/CapsuleRope",
                "/World/envs/env_1/CapsuleRope",
            )
        },
        env_origins=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        env_ids=np.asarray([0, 1], dtype=int),
        object_pose_views={"rope": object_view},
    )

    assert snapshot["rope"]["body_names"] == ("endpoint_0", "segment_0")
    np.testing.assert_allclose(
        snapshot["rope"]["positions_local"],
        np.asarray([[0.0, 0.0, -0.1], [0.0, 0.1, -0.1]], dtype=float),
    )
    np.testing.assert_allclose(
        snapshot["rope"]["body_positions_local"],
        np.asarray(
            [
                [[0.0, 0.0, -0.1], [0.2, 0.0, -0.1]],
                [[0.0, 0.1, -0.1], [0.2, 0.1, -0.1]],
            ],
            dtype=float,
        ),
    )

    restored = restore_tiled_object_pose_snapshot(
        stage=object(),
        object_prim_paths={
            "rope": (
                "/World/envs/env_0/CapsuleRope",
                "/World/envs/env_1/CapsuleRope",
            )
        },
        snapshot=snapshot,
        env_ids=np.asarray([1], dtype=int),
        env_origins=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        object_pose_views={"rope": object_view},
    )

    assert restored == 1
    assert usd_calls == []
    assert view_calls == [
        (
            "poses",
            [[3.0, 0.1, -0.1], [3.2, 0.1, -0.1]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [2, 3],
        ),
        (
            "velocities",
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [2, 3],
        ),
    ]


def test_create_tiled_object_pose_views_raises_when_dynamic_view_fails(
    monkeypatch,
) -> None:
    from linkerbot_sim.app.interactive.tiled_scene.runtime.factory import (
        _create_tiled_object_pose_views,
    )

    isaacsim_module = ModuleType("isaacsim")
    core_module = ModuleType("isaacsim.core")
    prims_module = ModuleType("isaacsim.core.prims")

    class FailingRigidPrim:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("view create failed")

    prims_module.RigidPrim = FailingRigidPrim
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core", core_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", prims_module)

    scene = SimpleNamespace(
        object_prim_paths={"Tblock": ("/World/envs/env_0/TBlock",)},
        object_handles=(
            SimpleNamespace(
                name="Tblock", kind="rigid", model=SimpleNamespace(static=False)
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="failed to create tiled object rigid view"):
        _create_tiled_object_pose_views(scene)


def test_create_tiled_object_pose_views_creates_dynamic_chain_body_view(
    monkeypatch,
) -> None:
    from linkerbot_sim.app.interactive.tiled_scene.runtime.factory import (
        _create_tiled_object_pose_views,
    )
    from linkerbot_sim.tiled.state.object_views import (
        TiledDynamicChainObjectPoseView,
    )

    created = []

    isaacsim_module = ModuleType("isaacsim")
    core_module = ModuleType("isaacsim.core")
    prims_module = ModuleType("isaacsim.core.prims")

    class FakeRigidPrim:
        def __init__(self, *, prim_paths_expr, name, reset_xform_properties):
            created.append(
                {
                    "prim_paths_expr": tuple(prim_paths_expr),
                    "name": name,
                    "reset_xform_properties": reset_xform_properties,
                }
            )

        def initialize(self):
            created[-1]["initialized"] = True

    class FakePrim:
        def __init__(self, path, name):
            self._path = path
            self._name = name

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._name

    prims_module.RigidPrim = FakeRigidPrim
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core", core_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", prims_module)

    scene = SimpleNamespace(
        object_prim_paths={
            "rope": (
                "/World/envs/env_0/CapsuleRope",
                "/World/envs/env_1/CapsuleRope",
            )
        },
        object_handles=(
            SimpleNamespace(
                name="rope",
                kind="dynamic_chain",
                state_summary=SimpleNamespace(reference_body="segment_0"),
                model={
                    "bodies": (
                        FakePrim(
                            "/World/envs/env_0/CapsuleRope/endpoint_0",
                            "endpoint_0",
                        ),
                        FakePrim(
                            "/World/envs/env_0/CapsuleRope/segment_0",
                            "segment_0",
                        ),
                    )
                },
            ),
        ),
    )

    result = _create_tiled_object_pose_views(scene)

    assert isinstance(result["rope"], TiledDynamicChainObjectPoseView)
    assert result["rope"].body_names == ("endpoint_0", "segment_0")
    assert result["rope"].reference_body == "segment_0"
    assert result["rope"].body_paths_by_env == (
        (
            "/World/envs/env_0/CapsuleRope/endpoint_0",
            "/World/envs/env_0/CapsuleRope/segment_0",
        ),
        (
            "/World/envs/env_1/CapsuleRope/endpoint_0",
            "/World/envs/env_1/CapsuleRope/segment_0",
        ),
    )
    assert created == [
        {
            "prim_paths_expr": (
                "/World/envs/env_0/CapsuleRope/endpoint_0",
                "/World/envs/env_0/CapsuleRope/segment_0",
                "/World/envs/env_1/CapsuleRope/endpoint_0",
                "/World/envs/env_1/CapsuleRope/segment_0",
            ),
            "name": "tiled_object_rope_bodies",
            "reset_xform_properties": False,
            "initialized": True,
        }
    ]


def test_create_tiled_object_pose_views_raises_when_dynamic_chain_has_no_bodies() -> (
    None
):
    from linkerbot_sim.app.interactive.tiled_scene.runtime.factory import (
        _create_tiled_object_pose_views,
    )

    scene = SimpleNamespace(
        object_prim_paths={"rope": ("/World/envs/env_0/CapsuleRope",)},
        object_handles=(
            SimpleNamespace(name="rope", kind="dynamic_chain", model={"bodies": ()}),
        ),
    )

    with pytest.raises(RuntimeError, match="has no child rigid bodies"):
        _create_tiled_object_pose_views(scene)


def test_apply_tiled_object_pose_reuses_double_precision_xform_ops() -> None:
    from pxr import Gf, Usd, UsdGeom

    from linkerbot_sim.tiled.state.usd_pose import (
        apply_prim_local_pose_and_zero_velocity,
    )

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(1.0, 2.0, 3.0)
    )
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0))

    applied = apply_prim_local_pose_and_zero_velocity(
        stage,
        "/World/Object",
        np.asarray([0.2, 0.3, 0.4], dtype=float),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    )

    assert applied is True
    assert str(prim.GetAttribute("xformOp:orient").GetTypeName()) == "quatd"
    assert str(prim.GetAttribute("xformOp:translate").GetTypeName()) == "double3"
    assert [str(item) for item in xform.GetXformOpOrderAttr().Get()] == [
        "xformOp:translate",
        "xformOp:orient",
    ]


def test_apply_tiled_object_pose_reuses_float_precision_xform_ops() -> None:
    from pxr import Gf, Usd, UsdGeom

    from linkerbot_sim.tiled.state.usd_pose import (
        apply_prim_local_pose_and_zero_velocity,
    )

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(1.0, 2.0, 3.0)
    )
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0))

    applied = apply_prim_local_pose_and_zero_velocity(
        stage,
        "/World/Object",
        np.asarray([0.2, 0.3, 0.4], dtype=float),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    )

    assert applied is True
    assert str(prim.GetAttribute("xformOp:orient").GetTypeName()) == "quatf"
    assert str(prim.GetAttribute("xformOp:translate").GetTypeName()) == "float3"
    assert [str(item) for item in xform.GetXformOpOrderAttr().Get()] == [
        "xformOp:translate",
        "xformOp:orient",
    ]


def test_apply_tiled_object_pose_preserves_existing_scale_op() -> None:
    from pxr import Gf, Usd, UsdGeom

    from linkerbot_sim.tiled.state.usd_pose import (
        apply_prim_local_pose_and_zero_velocity,
    )

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(1.0, 2.0, 3.0)
    )
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(2.0, 3.0, 4.0)
    )

    applied = apply_prim_local_pose_and_zero_velocity(
        stage,
        "/World/Object",
        np.asarray([0.2, 0.3, 0.4], dtype=float),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    )

    assert applied is True
    # 复位后 scale 必须保留，且排在 orient 之后（否则带缩放物体几何被破坏）。
    order = [str(item) for item in xform.GetXformOpOrderAttr().Get()]
    assert order == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    assert tuple(prim.GetAttribute("xformOp:scale").Get()) == (2.0, 3.0, 4.0)


def test_partial_reset_only_resets_selected_env() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    runtime.episode_steps[:] = [5, 6]

    response = module.handle_tiled_interactive_message(
        {"type": "reset", "env_ids": [1]},
        runtime,
    )

    assert response["event"] == "reset"
    assert response["env_ids"] == [1]
    np.testing.assert_allclose(runtime.current_positions[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(runtime.current_positions[1], [0.0, 0.0, 0.0])
    assert runtime.episode_steps.tolist() == [5, 0]
    assert runtime.episode_ids.tolist() == [0, 1]


def test_isaac_reset_rolls_back_later_robot_failure_and_can_retry() -> None:
    class View:
        def __init__(self, values: np.ndarray, fail_calls: set[int]) -> None:
            self.positions = np.asarray(values, dtype=float).copy()
            self.velocities = self.positions + 10.0
            self.position_calls = 0
            self.fail_calls = set(fail_calls)

        def get_joint_positions(self, *, indices=None):
            return self.positions[np.asarray(indices, dtype=int)].copy()

        def get_joint_velocities(self, *, indices=None):
            return self.velocities[np.asarray(indices, dtype=int)].copy()

        def set_joint_positions(self, values, *, indices=None):
            self.position_calls += 1
            if self.position_calls in self.fail_calls:
                raise RuntimeError(f"reset setter {self.position_calls} failed")
            self.positions[np.asarray(indices, dtype=int)] = np.asarray(
                values, dtype=float
            )

        def set_joint_velocities(self, values, *, indices=None):
            self.velocities[np.asarray(indices, dtype=int)] = np.asarray(
                values, dtype=float
            )

    class Adapter:
        def __init__(self, target: np.ndarray) -> None:
            self.last_target = target.copy()

        def reset(self) -> None:
            self.last_target = None

    views = {
        "left": View(np.asarray([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]), set()),
        "right": View(
            np.asarray([[2.0, 2.1, 2.2], [2.3, 2.4, 2.5]]),
            {1},
        ),
    }
    articulations = {
        name: SimpleNamespace(
            view=view,
            command_joint_indices=np.asarray([0, 1], dtype=int),
        )
        for name, view in views.items()
    }
    targets = {name: view.positions[:, :2].copy() for name, view in views.items()}
    adapters = {name: Adapter(target + 20.0) for name, target in targets.items()}
    tcp_positions = {
        name: np.full((2, 3), index + 0.5, dtype=float)
        for index, name in enumerate(views)
    }
    tcp_orientations = {name: np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)) for name in views}
    runtime = SimpleNamespace(
        scene=SimpleNamespace(
            config=SimpleNamespace(num_envs=2),
            object_prim_paths={},
            env_origins=np.zeros((2, 3), dtype=float),
        ),
        session=SimpleNamespace(stage=object()),
        object_pose_views={},
        initial_object_states={},
        initial_joint_positions={
            name: np.zeros_like(view.positions) for name, view in views.items()
        },
        initial_joint_velocities={
            name: np.zeros_like(view.velocities) for name, view in views.items()
        },
        target_positions=targets,
        tcp_positions_world=tcp_positions,
        tcp_orientations_wxyz=tcp_orientations,
        episode_steps=np.asarray([5, 6], dtype=int),
        episode_ids=np.asarray([7, 8], dtype=int),
        trajectory_buffer=SimpleNamespace(clear=lambda **_kwargs: None),
        planner_manager=SimpleNamespace(cancel_matching=lambda **_kwargs: None),
        quit_event=threading.Event(),
        step=0,
        time_s=0.0,
        _selected_runtime_items=lambda _selection: tuple(articulations.items()),
        _command_adapter=adapters.__getitem__,
        _refresh_tcp_state=lambda *_args, **_kwargs: None,
    )
    original_positions = {name: view.positions.copy() for name, view in views.items()}
    original_velocities = {name: view.velocities.copy() for name, view in views.items()}
    original_targets = {name: value.copy() for name, value in targets.items()}
    original_adapter_targets = {
        name: adapter.last_target.copy() for name, adapter in adapters.items()
    }

    with pytest.raises(RuntimeError, match="reset setter 1 failed"):
        tiled_scene_runtime_stepping.reset(runtime, np.asarray([0], dtype=int))

    for name, view in views.items():
        np.testing.assert_allclose(view.positions, original_positions[name])
        np.testing.assert_allclose(view.velocities, original_velocities[name])
        np.testing.assert_allclose(targets[name], original_targets[name])
        np.testing.assert_allclose(
            adapters[name].last_target,
            original_adapter_targets[name],
        )
    assert runtime.episode_steps.tolist() == [5, 6]
    assert runtime.episode_ids.tolist() == [7, 8]
    assert not hasattr(runtime, "fatal_error")

    response = tiled_scene_runtime_stepping.reset(runtime, np.asarray([0], dtype=int))
    assert response["accepted"] is True
    for view in views.values():
        np.testing.assert_allclose(view.positions[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(view.velocities[0], [0.0, 0.0, 0.0])
    assert runtime.episode_steps.tolist() == [0, 6]
    assert runtime.episode_ids.tolist() == [8, 8]


def test_get_state_and_set_state_support_selected_envs() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    set_response = module.handle_tiled_interactive_message(
        {
            "type": "set_state",
            "env_ids": [1],
            "state": {
                "robots": [
                    {
                        "robot_id": 0,
                        "joint_positions": [[0.3, 0.2, 0.1]],
                    }
                ],
                "episode_steps": [7],
            },
        },
        runtime,
    )
    state_response = module.handle_tiled_interactive_message(
        {
            "type": "get_state",
            "env_ids": [1],
            "fields": ["robots", "episode_steps"],
        },
        runtime,
    )

    assert set_response["event"] == "set_state"
    assert state_response["event"] == "state"
    assert state_response["env_ids"] == [1]
    assert state_response["state"]["episode_steps"] == [7]
    assert state_response["state"]["robots"][0]["robot_id"] == 0
    assert state_response["state"]["robots"][0]["joint_positions"] == [[0.3, 0.2, 0.1]]


def test_set_state_converts_public_robot_array_to_internal_labels() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    captured = {}

    def set_state(state, *, env_ids=None):
        captured["state"] = state
        captured["env_ids"] = env_ids.tolist()
        return {"event": "set_state", "accepted": True}

    runtime.set_state = set_state
    response = module.handle_tiled_interactive_message(
        {
            "type": "set_state",
            "env_ids": [1],
            "state": {
                "robots": [
                    {
                        "robot_id": 0,
                        "joint_positions": [[0.3, 0.2, 0.1]],
                    }
                ]
            },
        },
        runtime,
    )

    assert response["event"] == "set_state"
    assert captured == {
        "state": {
            "robots": {
                runtime.robot_names[0]: {
                    "joint_positions": [[0.3, 0.2, 0.1]],
                }
            }
        },
        "env_ids": [1],
    }


def test_set_state_rejects_internal_label_keyed_robot_map() -> None:
    module = load_tiled_scene_interactive_module()
    response = module.handle_tiled_interactive_message(
        {
            "type": "set_state",
            "env_ids": [0, 1],
            "state": {"robots": {"debug": {"joint_positions": [[0.0]]}}},
        },
        make_runtime(),
    )

    assert response["event"] == "rejected"
    assert "label-keyed robot maps are internal" in response["error"]


@pytest.mark.parametrize(
    ("state", "error"),
    (
        (
            {"joint_positions": [[0.0]]},
            "set_state.state has unknown fields",
        ),
        (
            {"robots": [{"robot_id": 0, "label": "debug"}]},
            "set_state.state.robots[0] has unknown fields",
        ),
        (
            {"robots": ({"robot_id": 0},)},
            "set_state.state.robots must be an array",
        ),
        (
            {"robots": [{"robot_id": 0, "joint_positions": [["0.1", 0.0, 0.0]]}]},
            "joint_positions[0][0] must be a number",
        ),
        (
            {"robots": [{"robot_id": 0, "joint_velocities": [[True, 0.0, 0.0]]}]},
            "joint_velocities[0][0] must be a number",
        ),
        (
            {"robots": [{"robot_id": 0, "joint_positions": None}]},
            "joint_positions must be an array",
        ),
        (
            {"episode_steps": [1.0]},
            "episode_steps[0] must be a JSON integer",
        ),
        (
            {"episode_ids": [True]},
            "episode_ids[0] must be a JSON integer",
        ),
        (
            {"episode_steps": [-1]},
            "episode_steps[0] must be nonnegative",
        ),
        (
            {"episode_ids": None},
            "episode_ids must be an array",
        ),
    ),
)
def test_set_state_accepts_only_current_native_json_shape(
    state: object, error: str
) -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {"type": "set_state", "env_ids": [0], "state": state},
        make_runtime(),
    )

    assert response["event"] == "rejected"
    assert error in response["error"]


@pytest.mark.parametrize("invalid", ("0.1", True, None, float("inf")))
def test_tiled_action_values_require_finite_json_numbers(invalid: object) -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {
            "type": "step",
            "env_ids": [0],
            "kind": "joint_delta_pos",
            "values": [invalid, 0.0, 0.0],
        },
        make_runtime(),
    )

    assert response["event"] == "rejected"
    assert "joint_delta_pos.values[0] must be" in response["error"]


def test_plan_duration_requires_finite_json_number() -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        {
            "type": "plan",
            "env_ids": [0],
            "kind": "joint_position_target",
            "joint_positions": [0.0, 0.0, 0.0],
            "duration_s": float("inf"),
        },
        make_runtime(),
    )

    assert response["event"] == "rejected"
    assert response["error"] == "plan.duration_s must be finite"


def test_isaac_set_state_validates_every_robot_before_first_setter() -> None:
    class View:
        def __init__(self) -> None:
            self.positions = np.zeros((2, 2), dtype=float)
            self.velocities = np.zeros((2, 2), dtype=float)
            self.setter_calls = 0

        def get_joint_positions(self, *, indices=None, joint_indices=None):
            rows = np.asarray(indices, dtype=int)
            columns = np.asarray(joint_indices, dtype=int)
            return self.positions[np.ix_(rows, columns)]

        def get_joint_velocities(self, *, indices=None, joint_indices=None):
            rows = np.asarray(indices, dtype=int)
            columns = np.asarray(joint_indices, dtype=int)
            return self.velocities[np.ix_(rows, columns)]

        def set_joint_positions(self, values, *, indices=None, joint_indices=None):
            self.setter_calls += 1

        def set_joint_velocities(self, values, *, indices=None, joint_indices=None):
            self.setter_calls += 1

    views = {"left": View(), "right": View()}
    articulations = {
        name: SimpleNamespace(
            view=view,
            command_joint_indices=np.asarray([0, 1], dtype=int),
        )
        for name, view in views.items()
    }
    runtime = SimpleNamespace(
        scene=SimpleNamespace(config=SimpleNamespace(num_envs=2)),
        target_positions={name: view.positions.copy() for name, view in views.items()},
        episode_steps=np.zeros(2, dtype=int),
        episode_ids=np.zeros(2, dtype=int),
        trajectory_buffer=SimpleNamespace(clear=lambda **_kwargs: None),
        planner_manager=SimpleNamespace(cancel_matching=lambda **_kwargs: None),
        step=0,
        time_s=0.0,
        _selected_runtime_items=lambda _selection: tuple(articulations.items()),
        _refresh_tcp_state=lambda *_args, **_kwargs: None,
        _command_adapter=lambda _name: SimpleNamespace(reset=lambda: None),
    )

    with pytest.raises(ValueError, match="right.joint_positions"):
        tiled_scene_runtime_state.set_state(
            runtime,
            {
                "robots": {
                    "left": {"joint_positions": [[1.0, 2.0]]},
                    "right": {"joint_positions": [[3.0, 4.0, 5.0]]},
                }
            },
            env_ids=np.asarray([0]),
        )

    assert views["left"].setter_calls == 0
    assert views["right"].setter_calls == 0


def test_isaac_set_state_rolls_back_when_later_robot_setter_fails() -> None:
    class View:
        def __init__(self, values: np.ndarray, *, fail_once: bool = False) -> None:
            self.positions = np.asarray(values, dtype=float).copy()
            self.velocities = np.zeros_like(self.positions)
            self.fail_once = fail_once

        def get_joint_positions(self, *, indices=None, joint_indices=None):
            rows = np.asarray(indices, dtype=int)
            columns = np.asarray(joint_indices, dtype=int)
            return self.positions[np.ix_(rows, columns)]

        def get_joint_velocities(self, *, indices=None, joint_indices=None):
            rows = np.asarray(indices, dtype=int)
            columns = np.asarray(joint_indices, dtype=int)
            return self.velocities[np.ix_(rows, columns)]

        def set_joint_positions(self, values, *, indices=None, joint_indices=None):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("second robot setter failed")
            rows = np.asarray(indices, dtype=int)
            columns = np.asarray(joint_indices, dtype=int)
            self.positions[np.ix_(rows, columns)] = np.asarray(values, dtype=float)

        def set_joint_velocities(self, values, *, indices=None, joint_indices=None):
            rows = np.asarray(indices, dtype=int)
            columns = np.asarray(joint_indices, dtype=int)
            self.velocities[np.ix_(rows, columns)] = np.asarray(values, dtype=float)

    original = {
        "left": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "right": np.asarray([[1.1, 1.2], [1.3, 1.4]], dtype=float),
    }
    views = {
        "left": View(original["left"]),
        "right": View(original["right"], fail_once=True),
    }
    articulations = {
        name: SimpleNamespace(
            view=view,
            command_joint_indices=np.asarray([0, 1], dtype=int),
        )
        for name, view in views.items()
    }
    targets = {name: values.copy() for name, values in original.items()}
    runtime = SimpleNamespace(
        scene=SimpleNamespace(config=SimpleNamespace(num_envs=2)),
        target_positions=targets,
        episode_steps=np.asarray([5, 6], dtype=int),
        episode_ids=np.asarray([7, 8], dtype=int),
        trajectory_buffer=SimpleNamespace(clear=lambda **_kwargs: None),
        planner_manager=SimpleNamespace(cancel_matching=lambda **_kwargs: None),
        step=0,
        time_s=0.0,
        _selected_runtime_items=lambda _selection: tuple(articulations.items()),
        _refresh_tcp_state=lambda *_args, **_kwargs: None,
        _command_adapter=lambda _name: SimpleNamespace(reset=lambda: None),
    )

    with pytest.raises(RuntimeError, match="second robot setter failed"):
        tiled_scene_runtime_state.set_state(
            runtime,
            {
                "robots": {
                    "left": {"joint_positions": [[9.0, 9.0]]},
                    "right": {"joint_positions": [[8.0, 8.0]]},
                },
                "episode_steps": [99],
            },
            env_ids=np.asarray([0]),
        )

    for name in ("left", "right"):
        np.testing.assert_allclose(views[name].positions, original[name])
        np.testing.assert_allclose(targets[name], original[name])
    assert runtime.episode_steps.tolist() == [5, 6]
    assert runtime.episode_ids.tolist() == [7, 8]


@pytest.mark.parametrize(
    ("state", "error"),
    (
        ({"unexpected": 1}, "set_state.state has unknown fields"),
        (
            {"objects": {}},
            "set_state.state.objects is unsupported; use set_snapshot",
        ),
        (
            {"robots": {"missing": {}}},
            "set_state.state.robots has unknown robots",
        ),
        (
            {"robots": {"arm": {"joint_efforts": [[0.0, 0.0]]}}},
            "set_state.state.robots.arm has unknown fields",
        ),
    ),
)
def test_internal_set_state_rejects_unknown_or_unsupported_fields_before_write(
    state: dict[str, object],
    error: str,
) -> None:
    articulation = SimpleNamespace(
        command_joint_indices=np.asarray([0, 1], dtype=int),
        view=SimpleNamespace(),
    )
    runtime = SimpleNamespace(
        scene=SimpleNamespace(config=SimpleNamespace(num_envs=1)),
        _selected_runtime_items=lambda _selection: (("arm", articulation),),
    )

    with pytest.raises(ValueError, match=error):
        tiled_scene_runtime_state.set_state(
            runtime,
            state,
            env_ids=np.asarray([0], dtype=int),
        )


@pytest.mark.parametrize(
    ("value", "error"),
    (
        ([1.5], "must be an integer"),
        ([True], "must be an integer"),
        (["1"], "must be an integer"),
        ([-1], "must be nonnegative"),
    ),
)
def test_internal_set_state_requires_nonnegative_integer_episode_counters(
    value: object,
    error: str,
) -> None:
    runtime = SimpleNamespace(
        scene=SimpleNamespace(config=SimpleNamespace(num_envs=1)),
        _selected_runtime_items=lambda _selection: (),
    )

    with pytest.raises(ValueError, match=error):
        tiled_scene_runtime_state.set_state(
            runtime,
            {"episode_steps": value},
            env_ids=np.asarray([0], dtype=int),
        )


def test_set_state_rollback_failure_fail_stops_and_preserves_forward_cause() -> None:
    class View:
        def __init__(self, values: np.ndarray, fail_calls: set[int]) -> None:
            self.positions = np.asarray(values, dtype=float).copy()
            self.velocities = np.zeros_like(self.positions)
            self.position_calls = 0
            self.fail_calls = set(fail_calls)

        def get_joint_positions(self, *, indices=None, joint_indices=None):
            return self.positions[
                np.ix_(
                    np.asarray(indices, dtype=int),
                    np.asarray(joint_indices, dtype=int),
                )
            ].copy()

        def get_joint_velocities(self, *, indices=None, joint_indices=None):
            return self.velocities[
                np.ix_(
                    np.asarray(indices, dtype=int),
                    np.asarray(joint_indices, dtype=int),
                )
            ].copy()

        def set_joint_positions(self, values, *, indices=None, joint_indices=None):
            self.position_calls += 1
            if self.position_calls in self.fail_calls:
                raise RuntimeError(f"position setter {self.position_calls} failed")
            self.positions[
                np.ix_(
                    np.asarray(indices, dtype=int),
                    np.asarray(joint_indices, dtype=int),
                )
            ] = np.asarray(values, dtype=float)

        def set_joint_velocities(self, values, *, indices=None, joint_indices=None):
            self.velocities[
                np.ix_(
                    np.asarray(indices, dtype=int),
                    np.asarray(joint_indices, dtype=int),
                )
            ] = np.asarray(values, dtype=float)

    views = {
        "left": View(np.asarray([[0.1, 0.2]]), {2}),
        "right": View(np.asarray([[1.1, 1.2]]), {1}),
    }
    articulations = {
        name: SimpleNamespace(
            view=view,
            command_joint_indices=np.asarray([0, 1], dtype=int),
        )
        for name, view in views.items()
    }
    adapters = {
        name: SimpleNamespace(
            last_target=np.asarray([[3.0, 4.0]], dtype=float),
            reset=lambda: None,
        )
        for name in views
    }
    runtime = SimpleNamespace(
        scene=SimpleNamespace(config=SimpleNamespace(num_envs=1)),
        target_positions={name: view.positions.copy() for name, view in views.items()},
        episode_steps=np.asarray([5], dtype=int),
        episode_ids=np.asarray([7], dtype=int),
        trajectory_buffer=SimpleNamespace(clear=lambda **_kwargs: None),
        planner_manager=SimpleNamespace(cancel_matching=lambda **_kwargs: None),
        quit_event=threading.Event(),
        step=0,
        time_s=0.0,
        _selected_runtime_items=lambda _selection: tuple(articulations.items()),
        _refresh_tcp_state=lambda *_args, **_kwargs: None,
        _command_adapter=adapters.__getitem__,
    )
    state = {
        "robots": {
            "left": {"joint_positions": [[9.0, 9.0]]},
            "right": {"joint_positions": [[8.0, 8.0]]},
        }
    }

    with pytest.raises(SnapshotRollbackError) as exc_info:
        tiled_scene_runtime_state.set_state(
            runtime,
            state,
            env_ids=np.asarray([0]),
        )

    assert str(exc_info.value.cause) == "position setter 1 failed"
    assert runtime.quit_event.is_set()
    assert "rollback_errors" in runtime.fatal_error
    call_counts = {name: view.position_calls for name, view in views.items()}

    with pytest.raises(RuntimeMutationRejected, match="requires rebuild"):
        tiled_scene_runtime_state.set_state(
            runtime,
            state,
            env_ids=np.asarray([0]),
        )
    assert {name: view.position_calls for name, view in views.items()} == call_counts

    with pytest.raises(RuntimeMutationRejected, match="requires rebuild"):
        tiled_scene_runtime_stepping.step_action(
            runtime,
            SimpleNamespace(kind="hold"),
            env_ids=np.asarray([0]),
        )


def test_set_state_commit_cache_failure_rolls_back_and_fail_stops() -> None:
    class View:
        def __init__(self) -> None:
            self.positions = np.asarray([[0.1, 0.2]], dtype=float)
            self.velocities = np.zeros_like(self.positions)

        def get_joint_positions(self, *, indices=None, joint_indices=None):
            return self.positions[
                np.ix_(
                    np.asarray(indices, dtype=int),
                    np.asarray(joint_indices, dtype=int),
                )
            ].copy()

        def get_joint_velocities(self, *, indices=None, joint_indices=None):
            return self.velocities[
                np.ix_(
                    np.asarray(indices, dtype=int),
                    np.asarray(joint_indices, dtype=int),
                )
            ].copy()

        def set_joint_positions(self, values, *, indices=None, joint_indices=None):
            self.positions[
                np.ix_(
                    np.asarray(indices, dtype=int),
                    np.asarray(joint_indices, dtype=int),
                )
            ] = np.asarray(values, dtype=float)

        def set_joint_velocities(self, values, *, indices=None, joint_indices=None):
            self.velocities[
                np.ix_(
                    np.asarray(indices, dtype=int),
                    np.asarray(joint_indices, dtype=int),
                )
            ] = np.asarray(values, dtype=float)

    view = View()
    adapter = SimpleNamespace(
        last_target=np.asarray([[3.0, 4.0]], dtype=float),
        reset=lambda: setattr(adapter, "last_target", None),
    )
    original_positions = view.positions.copy()
    original_targets = view.positions.copy()
    original_adapter_target = adapter.last_target.copy()
    cause = RuntimeError("planner cancellation failed")

    def fail_cancel(**_kwargs) -> None:
        raise cause

    runtime = SimpleNamespace(
        scene=SimpleNamespace(config=SimpleNamespace(num_envs=1)),
        target_positions={"arm": original_targets.copy()},
        episode_steps=np.asarray([5], dtype=int),
        episode_ids=np.asarray([7], dtype=int),
        trajectory_buffer=SimpleNamespace(clear=lambda **_kwargs: None),
        planner_manager=SimpleNamespace(cancel_matching=fail_cancel),
        quit_event=threading.Event(),
        step=0,
        time_s=0.0,
        _selected_runtime_items=lambda _selection: (
            (
                "arm",
                SimpleNamespace(
                    view=view,
                    command_joint_indices=np.asarray([0, 1], dtype=int),
                ),
            ),
        ),
        _refresh_tcp_state=lambda *_args, **_kwargs: None,
        _command_adapter=lambda _name: adapter,
    )

    with pytest.raises(RuntimeError) as exc_info:
        tiled_scene_runtime_state.set_state(
            runtime,
            {
                "robots": {"arm": {"joint_positions": [[9.0, 9.0]]}},
                "episode_steps": [99],
            },
            env_ids=np.asarray([0]),
        )

    assert exc_info.value is cause
    np.testing.assert_allclose(view.positions, original_positions)
    np.testing.assert_allclose(runtime.target_positions["arm"], original_targets)
    np.testing.assert_allclose(adapter.last_target, original_adapter_target)
    assert runtime.episode_steps.tolist() == [5]
    assert runtime.episode_ids.tolist() == [7]
    assert runtime.quit_event.is_set()
    assert "irreversible_steps" in runtime.fatal_error


def test_snapshot_protocol_restores_selected_envs() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[0.1, 0.2, 0.3], [9.0, 9.0, 9.0]]

    snapshot_response = module.handle_tiled_interactive_message(
        {"type": "get_snapshot", "env_id": 0},
        runtime,
    )
    set_response = module.handle_tiled_interactive_message(
        {
            "type": "set_snapshot",
            "env_ids": [1],
            "snapshot": snapshot_response["snapshot"],
        },
        runtime,
    )

    assert snapshot_response["event"] == "snapshot"
    assert set_response["event"] == "snapshot_restored"
    assert set_response["accepted"] is True
    assert set_response["env_ids"] == [1]
    np.testing.assert_allclose(runtime.current_positions[1], [0.1, 0.2, 0.3])


def test_snapshot_protocol_rejects_unknown_fields() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    snapshot = module.handle_tiled_interactive_message(
        {"type": "get_snapshot", "env_id": 0}, runtime
    )["snapshot"]

    response = module.handle_tiled_interactive_message(
        {
            "type": "set_snapshot",
            "env_ids": [1],
            "snapshot": snapshot,
            "unknown_selector": {"debug": "debug"},
        },
        runtime,
    )

    assert response["event"] == "rejected"
    assert "unknown fields" in response["error"]


def test_snapshot_protocol_applies_explicit_label_map() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[0.1, 0.2, 0.3], [9.0, 9.0, 9.0]]
    snapshot = module.handle_tiled_interactive_message(
        {"type": "get_snapshot", "env_id": 0}, runtime
    )["snapshot"]
    snapshot["robots"][0]["label"] = "renamed_source"

    response = module.handle_tiled_interactive_message(
        {
            "type": "set_snapshot",
            "env_ids": [1],
            "snapshot": snapshot,
            "label_map": {"renamed_source": "debug"},
        },
        runtime,
    )

    assert response["accepted"] is True
    np.testing.assert_allclose(runtime.current_positions[1], [0.1, 0.2, 0.3])


def test_clone_state_protocol_copies_source_env() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[0.1, 0.2, 0.3], [9.0, 9.0, 9.0]]

    response = module.handle_tiled_interactive_message(
        {
            "type": "clone_state",
            "source_env_id": 0,
            "target_env_ids": [1],
        },
        runtime,
    )

    assert response["event"] == "state_cloned"
    assert response["accepted"] is True
    assert response["source_env_id"] == 0
    assert response["target_env_ids"] == [1]
    np.testing.assert_allclose(runtime.current_positions[1], [0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    "message",
    (
        {"type": "get_snapshot", "env_ids": [0]},
        {"type": "clone_state", "source_env_id": 0, "env_ids": [1]},
    ),
)
def test_snapshot_commands_reject_unknown_env_selector_fields(message) -> None:
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        message, make_runtime()
    )

    assert response["event"] == "rejected"


@pytest.mark.parametrize(
    "message",
    (
        {
            "type": "clone_state",
            "source_env_id": 0,
            "target_env_ids": [1],
            "strict": "false",
        },
        {
            "type": "load_trajectory",
            "times": [0.0],
            "positions": [[0.0]],
            "replace": "false",
        },
        {
            "type": "load_trajectory",
            "times": [0.0],
            "positions": [[0.0]],
            "queue": 1,
        },
        {
            "type": "hand",
            "duration_s": 0.1,
            "joint_positions": {"joint_2": 0.8},
            "replace": "false",
        },
        {
            "type": "hand",
            "duration_s": 0.1,
            "joint_positions": {"joint_2": 0.8},
            "queue": 0,
        },
    ),
)
def test_external_boolean_fields_require_json_booleans(message) -> None:
    if message["type"] in {"load_trajectory", "hand"}:
        message = {**message, "env_ids": [0, 1]}
    response = load_tiled_scene_interactive_module().handle_tiled_interactive_message(
        message, make_runtime()
    )

    assert response["event"] == "rejected"
    assert "must be a boolean" in response["error"]


def test_load_and_step_trajectory_replays_test_runtime_fake() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    loaded = module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "robot_id": 0,
            "env_ids": [0, 1],
            "times": [0.0, 0.1],
            "positions": [
                [[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]],
                [[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]],
            ],
            "joint_names": ["j0", "j1", "j2"],
            "request_id": "manual-1",
        },
        runtime,
    )
    stepped = module.handle_tiled_interactive_message(
        {
            "type": "step_trajectory",
            "env_ids": [0, 1],
            "robot_id": 0,
            "decimation": 5,
        },
        runtime,
    )

    assert loaded["event"] == "trajectory_loaded"
    assert loaded["robot_id"] == 0
    assert stepped["event"] == "trajectory_step"
    assert stepped["ticks"] == 5
    # make_runtime 使用 100Hz physics，所以 5 tick 后应在 0.05s，即轨迹中点。
    np.testing.assert_allclose(runtime.current_positions[0], [0.5, 0.25, -0.25])
    np.testing.assert_allclose(runtime.current_positions[1], [1.0, 0.5, -0.5])


def test_trajectory_load_prefix_fills_missing_command_joints() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "env_ids": [1],
            "times": [0.0, 0.1],
            "positions": [[0.0], [10.0]],
        },
        runtime,
    )
    response = module.handle_tiled_interactive_message(
        {
            "type": "step_trajectory",
            "env_ids": [1],
            "decimation": 10,
        },
        runtime,
    )

    assert response["event"] == "trajectory_step"
    np.testing.assert_allclose(runtime.current_positions[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(runtime.current_positions[1], [10.0, 5.0, 6.0])


def test_trajectory_status_and_clear_controls() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "env_ids": [0, 1],
            "times": [0.0, 1.0],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        },
        runtime,
    )
    status = module.handle_tiled_interactive_message(
        {"type": "trajectory_status", "env_ids": [0, 1]},
        runtime,
    )
    cleared = module.handle_tiled_interactive_message(
        {"type": "clear_trajectory", "env_ids": [0]},
        runtime,
    )

    assert status["event"] == "trajectory_status"
    assert status["trajectory"]["robots"][0]["robot_id"] == 0
    assert status["trajectory"]["robots"][0]["active_env_ids"] == [0, 1]
    assert cleared["event"] == "trajectory_cleared"
    assert cleared["cleared"] == [{"robot_id": 0, "env_ids": [0]}]


def test_reset_clears_selected_trajectory_env() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "env_ids": [0, 1],
            "times": [0.0, 1.0],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        },
        runtime,
    )
    module.handle_tiled_interactive_message({"type": "reset", "env_ids": [1]}, runtime)
    status = module.handle_tiled_interactive_message(
        {"type": "trajectory_status", "env_ids": [0, 1]},
        runtime,
    )

    assert status["trajectory"]["robots"][0]["robot_id"] == 0
    assert status["trajectory"]["robots"][0]["active_env_ids"] == [0]


def test_planner_status_loads_ready_trajectory_and_step_replays_it() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    submitted = module.handle_tiled_interactive_message(
        {
            "type": "plan",
            "robot_id": 0,
            "env_ids": [0, 1],
            "request_id": "plan-1",
            "duration_s": 0.1,
            "sample_dt_s": 0.05,
            "kind": "joint_position_target",
            "joint_positions": [
                [1.0, 0.5, -0.5],
                [2.0, 1.0, -1.0],
            ],
        },
        runtime,
    )
    status = module.handle_tiled_interactive_message(
        {"type": "planner_status", "wait_timeout_s": 1.0},
        runtime,
    )
    stepped = module.handle_tiled_interactive_message(
        {
            "type": "step_trajectory",
            "env_ids": [0, 1],
            "robot_id": 0,
            "decimation": 10,
        },
        runtime,
    )

    assert submitted["event"] == "plan_submitted"
    assert submitted["request_id"] == "plan-1"
    assert status["event"] == "planner_status"
    assert status["loaded"] == [
        {"request_id": "plan-1", "robot_id": 0, "env_ids": [0, 1]}
    ]
    assert stepped["event"] == "trajectory_step"
    np.testing.assert_allclose(runtime.current_positions[0], [1.0, 0.5, -0.5])
    np.testing.assert_allclose(runtime.current_positions[1], [2.0, 1.0, -1.0])


def test_independent_hand_motion_queues_after_active_trajectory() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[0.0, 0.0, 0.2], [0.0, 0.0, 0.4]]

    module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "env_ids": [0, 1],
            "robot_id": 0,
            "times": [0.0, 0.1],
            "positions": [[0.0], [1.0]],
        },
        runtime,
    )
    queued = module.handle_tiled_interactive_message(
        {
            "type": "hand",
            "env_ids": [0, 1],
            "robot_id": 0,
            "duration_s": 0.1,
            "joint_positions": {"joint_2": 0.8},
        },
        runtime,
    )
    main = module.handle_tiled_interactive_message(
        {
            "type": "step_trajectory",
            "env_ids": [0, 1],
            "robot_id": 0,
            "decimation": 10,
        },
        runtime,
    )
    hand = module.handle_tiled_interactive_message(
        {
            "type": "step_trajectory",
            "env_ids": [0, 1],
            "robot_id": 0,
            "decimation": 10,
        },
        runtime,
    )

    assert queued["event"] == "hand_motion_queued"
    assert queued["motions"][0]["queued"] is True
    assert queued["motions"][0]["joint_track_count"] == 1
    np.testing.assert_allclose(main["joint_positions"][0], [1.0, 0.0, 0.2])
    np.testing.assert_allclose(hand["joint_positions"][0], [1.0, 0.0, 0.8])

    with pytest.raises(ValueError, match="type='hand'"):
        runtime.submit_hand_motion(
            {
                "duration_s": 0.1,
                "joint_positions": {"joint_2": 0.8},
            },
            env_ids=np.asarray([0, 1], dtype=int),
        )


def test_clear_completed_planner_results_from_interactive_message() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    module.handle_tiled_interactive_message(
        {
            "type": "plan",
            "env_ids": [0, 1],
            "request_id": "plan-clear",
            "duration_s": 0.1,
            "sample_dt_s": 0.05,
            "kind": "joint_position_target",
            "joint_positions": [1.0, 0.5, -0.5],
        },
        runtime,
    )
    module.handle_tiled_interactive_message(
        {"type": "planner_status", "wait_timeout_s": 1.0},
        runtime,
    )
    cleared = module.handle_tiled_interactive_message(
        {"type": "clear_completed", "request_id": "plan-clear"},
        runtime,
    )

    assert cleared["event"] == "completed_cleared"
    assert cleared["result"] == {
        "cleared": ["plan-clear"],
        "missing": [],
        "count": 1,
    }


def test_plan_joint_delta_uses_current_snapshot() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

    module.handle_tiled_interactive_message(
        {
            "type": "plan",
            "env_ids": [1],
            "request_id": "plan-delta",
            "duration_s": 0.1,
            "sample_dt_s": 0.1,
            "kind": "joint_delta_pos",
            "joint_deltas": [[0.5, -0.5]],
        },
        runtime,
    )
    module.handle_tiled_interactive_message(
        {"type": "planner_status", "wait_timeout_s": 1.0},
        runtime,
    )
    module.handle_tiled_interactive_message(
        {"type": "step_trajectory", "env_ids": [1], "decimation": 10},
        runtime,
    )

    np.testing.assert_allclose(runtime.current_positions[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(runtime.current_positions[1], [2.5, 1.5, 2.0])


def test_plan_rejects_unknown_kind() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "plan",
            "env_ids": [0, 1],
            "kind": "unknown_kind",
            "robot_id": 0,
            "duration_s": 1.0,
        },
        runtime,
    )

    assert response["event"] == "rejected"
    assert "unsupported tiled planning kind" in response["error"]


@pytest.mark.parametrize(
    ("message", "error"),
    (
        ({"type": "plan", "robot_id": 0, "joint_positions": [1.0]}, "plan.kind"),
        (
            {
                "type": "plan",
                "robot_id": 0,
                "kind": "joint_position_target",
                "joint_positions": [1.0],
                "unexpected_field": True,
            },
            "unknown fields",
        ),
    ),
)
def test_tiled_planning_validates_current_payload(message, error: str) -> None:
    module = load_tiled_scene_interactive_module()
    response = module.handle_tiled_interactive_message(
        {**message, "env_ids": [0, 1]}, make_runtime()
    )

    assert response["event"] == "rejected"
    assert error in response["error"]


def test_plan_linear_pose_path_submits_async_plan() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "plan",
            "env_ids": [0, 1],
            "kind": "linear_pose_path",
            "robot_id": 0,
            "target_offset": [0.0, 0.0, 0.1],
            "duration_s": 1.0,
        },
        runtime,
    )

    assert response["event"] == "plan_submitted"
    assert response["robot_id"] == 0
    assert response["segments"] == ["linear_pose_path"]


def test_unknown_message_type_is_rejected() -> None:
    module = load_tiled_scene_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "unknown_type",
        },
        runtime,
    )

    assert response["event"] == "rejected"
    assert "type='step'" in response["error"]


def test_filter_isaac_state_fields_supports_nested_robot_paths() -> None:
    filtered = tiled_scene_command_utils._filter_isaac_state_fields(
        {
            "robots": {
                "robot_a": {
                    "joint_names": ["j1", "j2"],
                    "joint_positions": [[0.1, 0.2]],
                    "joint_velocities": [[0.0, 0.0]],
                },
                "robot_b": {
                    "joint_names": ["j1", "j2"],
                    "joint_positions": [[0.3, 0.4]],
                    "joint_velocities": [[0.0, 0.0]],
                },
            },
            "objects": {
                "Tblock": {
                    "positions_world": [[0.2, 0.0, -0.4]],
                    "positions_local": [[0.2, 0.0, -0.4]],
                    "orientations_wxyz": [[1.0, 0.0, 0.0, 0.0]],
                }
            },
            "episode_steps": [5],
            "episode_ids": [2],
        },
        (
            "robots.robot_a.joint_positions",
            "objects.Tblock.positions_world",
            "episode_steps",
            "unknown",
        ),
    )

    assert filtered == {
        "robots": {"robot_a": {"joint_positions": [[0.1, 0.2]]}},
        "objects": {"Tblock": {"positions_world": [[0.2, 0.0, -0.4]]}},
        "episode_steps": [5],
    }


def test_world_frame_batched_ik_solver_converts_between_world_and_base() -> None:
    calls = []
    rotation_x_90 = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    local_orientation = matrix_to_quat_wxyz(rotation_x_90)

    class FakeSolver:
        tcp_frame_name = "tool"

        def solve(self, **kwargs):
            calls.append(kwargs)
            return BatchIKResult(
                joint_positions=np.asarray(kwargs["seeds"], dtype=float),
                success=np.ones(2, dtype=bool),
                position_error=np.zeros(2, dtype=float),
                orientation_error=None,
                status=("SUCCESS", "SUCCESS"),
            )

        def compute_tcp_poses(self, command_positions, *, tcp_frame_name=None):
            return (
                np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                np.repeat(local_orientation.reshape(1, 4), 2, axis=0),
            )

    rotation_z_90 = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    wrapper = tiled_scene_runtime_ik._WorldFrameBatchIKBackend(
        solver=FakeSolver(),
        root_positions_world=np.asarray([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        root_rotations_world_from_base=np.repeat(
            rotation_z_90.reshape(1, 3, 3),
            2,
            axis=0,
        ),
        root_quats_world_wxyz=np.repeat(
            matrix_to_quat_wxyz(rotation_z_90).reshape(1, 4),
            2,
            axis=0,
        ),
    )

    wrapper.solve(
        target_positions=np.asarray([[10.0, 1.0, 0.0], [19.0, 0.0, 0.0]]),
        target_orientations_wxyz=None,
        seeds=np.zeros((2, 2)),
        tcp_frame_name="tool",
    )
    tcp_positions, tcp_orientations = wrapper.command_tcp_world_poses(np.zeros((2, 2)))

    np.testing.assert_allclose(
        calls[0]["target_positions"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        tcp_positions,
        [[10.0, 1.0, 0.0], [19.0, 0.0, 0.0]],
        atol=1.0e-8,
    )
    expected_world_orientation = matrix_to_quat_wxyz(rotation_z_90 @ rotation_x_90)
    np.testing.assert_allclose(
        np.abs(tcp_orientations @ expected_world_orientation),
        [1.0, 1.0],
        atol=1.0e-8,
    )


def test_ee_linear_path_converts_absolute_base_target_to_world() -> None:
    rotation_z_90 = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    root_quaternion = matrix_to_quat_wxyz(rotation_z_90)
    solver = SimpleNamespace(
        root_positions_world=np.asarray(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=float
        ),
        root_rotations_world_from_base=np.repeat(
            rotation_z_90.reshape(1, 3, 3), 2, axis=0
        ),
        root_quats_world_wxyz=np.repeat(root_quaternion.reshape(1, 4), 2, axis=0),
    )
    runtime = SimpleNamespace(ik_solvers={"robot": solver})

    converted = tiled_scene_runtime_ik.action_for_robot_reference(
        runtime,
        TiledCommandAction(
            "ee_linear_path",
            target_position=np.asarray([1.0, 0.0, 0.0]),
            orientation_mode="target",
            target_orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            pose_reference_frame="base",
        ),
        robot_name="robot",
        env_ids=np.asarray([0, 1]),
    )

    assert converted.pose_reference_frame == "world"
    np.testing.assert_allclose(
        converted.target_position,
        [[10.0, 1.0, 0.0], [20.0, 1.0, 0.0]],
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        converted.target_orientation_wxyz,
        np.repeat(root_quaternion.reshape(1, 4), 2, axis=0),
        atol=1.0e-8,
    )

    converted_relative = tiled_scene_runtime_ik.action_for_robot_reference(
        runtime,
        TiledCommandAction(
            "ee_linear_path",
            target_offset=np.asarray([1.0, 0.0, 0.0]),
            orientation_mode="target",
            target_orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            pose_reference_frame="base",
        ),
        robot_name="robot",
        env_ids=np.asarray([0, 1]),
    )

    np.testing.assert_allclose(
        converted_relative.target_offset,
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        converted_relative.target_orientation_wxyz,
        np.repeat(root_quaternion.reshape(1, 4), 2, axis=0),
        atol=1.0e-8,
    )


def test_isaac_runtime_actions_apply_fixed_tick_targets(
    monkeypatch,
) -> None:
    module = load_tiled_scene_interactive_module()
    applied = []

    class FakeView:
        count = 2
        num_dof = 2
        dof_names = ("j0", "j1")

        def __init__(self):
            self.positions = np.zeros((2, 2), dtype=float)
            self.velocities = np.zeros((2, 2), dtype=float)

        def get_joint_positions(self, *, indices=None, joint_indices=None):
            rows = np.arange(2) if indices is None else np.asarray(indices, dtype=int)
            cols = (
                np.arange(2)
                if joint_indices is None
                else np.asarray(joint_indices, dtype=int)
            )
            return self.positions[np.ix_(rows, cols)]

        def get_joint_velocities(self, *, indices=None, joint_indices=None):
            rows = np.arange(2) if indices is None else np.asarray(indices, dtype=int)
            cols = (
                np.arange(2)
                if joint_indices is None
                else np.asarray(joint_indices, dtype=int)
            )
            return self.velocities[np.ix_(rows, cols)]

    class FakeWorld:
        def __init__(self):
            self.steps = 0

        def step(self, *, render):
            self.steps += 1

        def get_physics_dt(self):
            return 0.01

    class FakeIKSolver:
        tcp_frame_name = "tool"

        def solve(
            self,
            *,
            target_positions,
            target_orientations_wxyz,
            seeds,
            tcp_frame_name,
        ):
            return BatchIKResult(
                joint_positions=np.asarray(target_positions, dtype=float)[:, :2],
                success=np.ones(2, dtype=bool),
                position_error=np.zeros(2, dtype=float),
                orientation_error=np.zeros(2, dtype=float),
            )

    view = FakeView()
    view.positions[1, :] = [10.0, 20.0]
    world = FakeWorld()
    target_positions = np.asarray([[0.0, 0.0], [7.0, 8.0]], dtype=float)

    def fake_apply_joint_targets(view_arg, targets, *, joint_indices):
        applied.append(np.asarray(targets, dtype=float).copy())
        view_arg.positions[:, np.asarray(joint_indices, dtype=int)] = targets

    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled_scene.runtime.stepping._apply_joint_targets",
        fake_apply_joint_targets,
    )
    monkeypatch.setattr(
        module.TiledSceneRuntime,
        "_refresh_tcp_state",
        lambda self, robot_name, env_ids=None: None,
    )
    runtime = module.TiledSceneRuntime(
        env_name="unit",
        env_config={},
        session=SimpleNamespace(world=world, stage=object(), app=object()),
        scene=SimpleNamespace(
            config=SimpleNamespace(
                num_envs=2,
                runtime=SimpleNamespace(
                    inspect_env_ids=(0,),
                ),
            ),
            env_root_paths=("/World/envs/env_0", "/World/envs/env_1"),
            env_origins=np.zeros((2, 3), dtype=float),
            articulation_views={
                "left": SimpleNamespace(
                    view=view,
                    command_joint_indices=np.asarray([0, 1], dtype=int),
                    command_joint_names=("j0", "j1"),
                )
            },
            object_prim_paths={},
        ),
        render=False,
        default_decimation=2,
        robot_names=("left",),
        episode_steps=np.zeros(2, dtype=int),
        episode_ids=np.zeros(2, dtype=int),
        initial_joint_positions={"left": np.zeros((2, 2), dtype=float)},
        initial_joint_velocities={"left": np.zeros((2, 2), dtype=float)},
        target_positions={"left": target_positions},
        initial_object_states={},
        command_adapters={
            "left": TiledCommandAdapter(
                num_envs=2,
                command_dim=2,
                default_decimation=2,
                tcp_frame_name="tool",
                ik_solver=FakeIKSolver(),
            )
        },
        ik_solvers={"left": object()},
        tcp_positions_world={"left": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
        tcp_orientations_wxyz={
            "left": np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        },
        trajectory_buffer=TiledTrajectoryBuffer(num_envs=2),
        planner_manager=TiledPlannerManager(max_workers=1),
        sensor_cameras=(),
        camera_output=None,
        quit_event=SimpleNamespace(is_set=lambda: False, set=lambda: None),
    )
    try:
        response = runtime.step_action(
            TiledCommandAction(
                kind="joint_position_target",
                values=np.asarray([[1.0, 2.0]], dtype=float),
                decimation=2,
                interpolation="linear",
            ),
            env_ids=np.asarray([0], dtype=int),
        )
        path_response = runtime.step_action(
            TiledCommandAction(
                kind="ee_linear_path",
                values=np.asarray([[0.2, 0.0, 0.0]], dtype=float),
                duration_s=0.02,
                interpolation="linear",
            ),
            env_ids=np.asarray([0, 1], dtype=int),
        )
    finally:
        runtime.close = lambda: None
        runtime.planner_manager.shutdown()

    assert response["ticks"] == 2
    assert response["env_ids"] == [0]
    assert response["step"] == 2
    np.testing.assert_allclose(applied[0], [[0.5, 1.0], [7.0, 8.0]])
    np.testing.assert_allclose(applied[1], [[1.0, 2.0], [7.0, 8.0]])
    assert path_response["ticks"] == 2
    assert path_response["duration_s"] == pytest.approx(0.02)
    assert path_response["info"]["left"]["ik"]["ik_completed_steps"] == [2, 2]
    assert world.steps == 4
    np.testing.assert_allclose(applied[2], [[0.1, 0.0], [1.1, 0.0]])
    np.testing.assert_allclose(applied[3], [[0.2, 0.0], [1.2, 0.0]])


def test_isaac_runtime_requires_message_robot_selection_for_multi_robot_actions() -> (
    None
):
    module = load_tiled_scene_interactive_module()
    from linkerbot_sim.app.interactive.tiled_scene.selectors import ALL_ROBOTS

    runtime = object.__new__(module.TiledSceneRuntime)
    runtime.robot_names = ("left", "right")
    runtime.scene = SimpleNamespace(
        articulation_views={
            "left": object(),
            "right": object(),
        }
    )

    with pytest.raises(ValueError, match="robots is required"):
        runtime._selected_runtime_items(None, require_explicit=True)

    selected = runtime._selected_runtime_items(ALL_ROBOTS, require_explicit=True)
    assert [name for name, _ in selected] == ["left", "right"]
