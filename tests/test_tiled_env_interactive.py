from __future__ import annotations

import importlib.util
import queue
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from linkerbot_sim.app.interactive import tiled as tiled_interactive
from linkerbot_sim.app.interactive.tiled import command_utils as tiled_command_utils
from linkerbot_sim.app.interactive.tiled.isaac_runtime import (
    IsaacTiledInteractiveRuntime,
)
from linkerbot_sim.app.interactive.tiled import isaac_ik_solver as tiled_isaac_ik_solver
from linkerbot_sim.app.interactive.tiled import (
    telemetry_publish as tiled_telemetry_publish,
)
from linkerbot_sim.app.interactive.tiled import transport as tiled_transport
from tests.fakes.tiled_runtime_fake import (
    DebugBatchedIKSolver,
    DebugTiledInteractiveRuntime,
)
from linkerbot_sim.tiled import (
    BatchedIKResult,
    TiledCommandAction,
    TiledCommandAdapter,
    TiledPlannerManager,
)
from linkerbot_sim.tiled import TiledTrajectoryBuffer
from linkerbot_sim.utils.rotations import matrix_to_quat_wxyz


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "tiled_env_interactive.py"
)


def load_tiled_interactive_module():
    """返回包内 tiled interactive runtime 模块。"""

    return tiled_interactive


def load_tiled_interactive_script_wrapper():
    """按文件路径导入 CLI wrapper，确认 scripts/ 入口仍能使用。"""

    spec = importlib.util.spec_from_file_location("tiled_env_interactive", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_wrapper_reexports_main_entrypoint() -> None:
    module = load_tiled_interactive_script_wrapper()

    assert module.main is tiled_interactive.main


def make_runtime(*, fail_env_ids=frozenset()):
    return DebugTiledInteractiveRuntime.create(
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
        ik_solver=DebugBatchedIKSolver(fail_env_ids=frozenset(fail_env_ids)),
    )


def test_isaac_tiled_step_world_samples_camera_output() -> None:
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
    runtime = IsaacTiledInteractiveRuntime(
        env_name="unit",
        env_config={},
        session=SimpleNamespace(world=world, app=None),
        scene=SimpleNamespace(config=SimpleNamespace(num_envs=2)),
        render=True,
        default_decimation=1,
        robot_names=(),
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
    assert observer.calls == [(world, 0, "action")]
    assert runtime.step == 1
    assert runtime.episode_steps.tolist() == [1, 1]


def test_parse_tiled_joint_action_from_interactive_message() -> None:
    module = load_tiled_interactive_module()

    action = module.parse_tiled_action(
        {
            "type": "joint_delta_pos",
            "joint_deltas": [0.1, -0.2, 0.3],
            "decimation": 3,
        }
    )

    assert action.kind == "joint_delta_pos"
    assert action.decimation == 3
    np.testing.assert_allclose(action.values, [0.1, -0.2, 0.3])


def test_parse_args_uses_isaac_runtime_without_backend_switch(monkeypatch) -> None:
    module = load_tiled_interactive_module()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tiled_env_interactive.py",
            "--gui",
            "--default-decimation",
            "3",
            "--hold",
            "--telemetry-rate-hz",
            "5",
            "--max-pending-requests",
            "8",
            "--max-completed-results",
            "9",
        ],
    )

    args = module.parse_args()

    assert not hasattr(args, "backend")
    assert args.env == "scene3_tiled"
    assert args.gui is True
    assert not hasattr(args, "num_envs")
    assert not hasattr(args, "robots")
    assert args.default_decimation == 3
    assert args.hold is True
    assert args.planner_backend == "linear"
    assert args.telemetry_rate_hz == 5.0
    assert args.max_pending_requests == 8
    assert args.max_completed_results == 9


def test_parse_args_rejects_num_envs_cli_override(monkeypatch) -> None:
    module = load_tiled_interactive_module()

    monkeypatch.setattr(
        sys,
        "argv",
        ["tiled_env_interactive.py", "--num-envs", "8"],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_parse_args_rejects_removed_debug_backend_switch(monkeypatch) -> None:
    module = load_tiled_interactive_module()

    monkeypatch.setattr(
        sys,
        "argv",
        ["tiled_env_interactive.py", "--backend", "debug"],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_parse_args_rejects_removed_robots_switch(monkeypatch) -> None:
    module = load_tiled_interactive_module()

    monkeypatch.setattr(
        sys,
        "argv",
        ["tiled_env_interactive.py", "--robots", "left"],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_runtime_steps_batched_joint_delta_synchronously() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "joint_delta_pos",
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
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
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


def test_step_env_ids_updates_selected_envs_and_holds_others() -> None:
    module = load_tiled_interactive_module()
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


def test_step_message_passes_robot_names_to_runtime() -> None:
    module = load_tiled_interactive_module()

    class FakeRuntime:
        quit_event = None

        def step_action(self, action, *, env_ids=None, robot_names=None):
            return {
                "event": "step",
                "kind": action.kind,
                "env_ids": None if env_ids is None else env_ids.tolist(),
                "robot_names": robot_names,
            }

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "env_ids": [0],
            "robots": ["left"],
        },
        FakeRuntime(),
    )

    assert response == {
        "event": "step",
        "kind": "joint_delta_pos",
        "env_ids": [0],
        "robot_names": ("left",),
    }


def test_step_message_preserves_explicit_all_robot_selection() -> None:
    module = load_tiled_interactive_module()
    from linkerbot_sim.app.interactive.tiled.protocol import ALL_ROBOTS

    class FakeRuntime:
        quit_event = None

        def step_action(self, action, *, env_ids=None, robot_names=None):
            return {
                "event": "step",
                "robot_names_is_all": robot_names is ALL_ROBOTS,
            }

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "robots": "all",
        },
        FakeRuntime(),
    )

    assert response == {"event": "step", "robot_names_is_all": True}


def test_step_message_rejects_removed_robot_names_input_alias() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "robot_names": ["left"],
        },
        runtime,
    )

    assert response["event"] == "rejected"


def test_tiled_message_rejects_removed_side_alias() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "step",
            "kind": "joint_delta_pos",
            "values": [0.1, 0.0, 0.0],
            "side": "left",
        },
        runtime,
    )

    assert response["event"] == "rejected"
    assert "side is not supported" in response["error"]


def test_runtime_ee_delta_reports_ik_mask_and_fallback() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime(fail_env_ids={1})
    runtime.current_positions[:] = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

    response = module.handle_tiled_interactive_message(
        {
            "type": "ee_delta_pos",
            "offset": [0.1, 0.0, 0.0],
            "decimation": 1,
        },
        runtime,
    )

    assert response["event"] == "step"
    assert response["info"]["ik_success"] == [True, False]
    # env 0 使用 debug IK 目标；env 1 失败后保持 seed/current_positions。
    np.testing.assert_allclose(runtime.current_positions[0], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(runtime.current_positions[1], [2.0, 2.0, 2.0])


def test_runtime_status_reset_and_quit_controls() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    status = module.handle_tiled_interactive_message({"type": "status"}, runtime)
    reset = module.handle_tiled_interactive_message({"type": "reset"}, runtime)
    quit_response = module.handle_tiled_interactive_message({"type": "quit"}, runtime)

    assert status["event"] == "status"
    assert status["num_envs"] == 2
    assert reset["event"] == "reset"
    assert reset["accepted"] is True
    assert quit_response == {"event": "quit", "accepted": True}
    assert runtime.quit_event.is_set()


def test_interactive_loop_processes_queued_requests_on_main_loop() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()
    request_queue = queue.Queue()
    status_queue = queue.Queue()
    quit_queue = queue.Queue()
    request_queue.put(
        tiled_transport._InteractiveRequest(
            line='{"type":"status"}',
            source="unit",
            response_queue=status_queue,
        )
    )
    request_queue.put(
        tiled_transport._InteractiveRequest(
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


def test_interactive_loop_idles_gui_runtime_with_hold_without_requests() -> None:
    module = load_tiled_interactive_module()

    class FakeRuntime:
        render = True
        idle_period_s = 0.001

        def __init__(self) -> None:
            self.quit_event = threading.Event()
            self.idle_steps = 0

        def idle_step(self) -> None:
            self.idle_steps += 1
            self.quit_event.set()

    runtime = FakeRuntime()

    module.run_interactive_loop(
        runtime,
        telemetry=None,
        request_queue=queue.Queue(),
        telemetry_rate_hz=0.0,
        hold=True,
    )

    assert runtime.idle_steps == 1


def test_interactive_loop_does_not_idle_gui_runtime_without_hold() -> None:
    module = load_tiled_interactive_module()

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
        tiled_transport._InteractiveRequest(
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
        tiled_transport._quit_on_stdin_eof(
            hold=False,
            tcp_jsonl_port=None,
            telemetry=None,
        )
        is True
    )
    assert (
        tiled_transport._quit_on_stdin_eof(
            hold=True,
            tcp_jsonl_port=None,
            telemetry=None,
        )
        is False
    )
    assert (
        tiled_transport._quit_on_stdin_eof(
            hold=False,
            tcp_jsonl_port=8765,
            telemetry=None,
        )
        is False
    )
    assert (
        tiled_transport._quit_on_stdin_eof(
            hold=False,
            tcp_jsonl_port=None,
            telemetry=object(),
        )
        is False
    )


def test_publish_state_telemetry_samples_selected_envs() -> None:
    load_tiled_interactive_module()
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

    tiled_telemetry_publish._publish_state_telemetry(
        FakeTelemetry(), runtime, event="state"
    )

    assert published[0]["event"] == "state"
    assert published[0]["state_response"]["env_ids"] == [1]


def test_publish_response_telemetry_resamples_configured_envs() -> None:
    load_tiled_interactive_module()
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

    tiled_telemetry_publish._publish_response_telemetry(
        FakeTelemetry(),
        runtime,
        {
            "event": "state",
            "env_ids": [1],
            "state": {"joint_positions": [[9.0, 0.0, 0.0]]},
        },
    )

    assert published[0]["state_response"]["env_ids"] == [0]
    assert published[0]["state_response"]["state"]["joint_positions"] == [
        [1.0, 0.0, 0.0]
    ]


def test_restore_tiled_object_pose_snapshot_uses_selected_env_paths(
    monkeypatch,
) -> None:
    from linkerbot_sim.app.interactive.tiled.object_states import (
        _restore_tiled_object_pose_snapshot,
    )

    calls = []

    def fake_apply(stage, prim_path, position, orientation):
        calls.append(
            (prim_path, np.asarray(position).tolist(), np.asarray(orientation).tolist())
        )
        return True

    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled.object_states._apply_prim_local_pose_and_zero_velocity",
        fake_apply,
    )

    restored = _restore_tiled_object_pose_snapshot(
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
    from linkerbot_sim.app.interactive.tiled.object_states import (
        _read_tiled_object_states,
    )

    class FakeRigidView:
        def get_world_poses(self, *, indices):
            assert np.asarray(indices, dtype=int).tolist() == [1]
            return (
                np.asarray([[3.2, 0.1, -0.4]], dtype=float),
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
            )

    state = _read_tiled_object_states(
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
    from linkerbot_sim.app.interactive.tiled.object_states import (
        _restore_tiled_object_pose_snapshot,
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
        "linkerbot_sim.app.interactive.tiled.object_states._apply_prim_local_pose_and_zero_velocity",
        fake_apply,
    )

    restored = _restore_tiled_object_pose_snapshot(
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
    from linkerbot_sim.app.interactive.tiled.object_states import (
        _read_tiled_object_states,
    )

    class FailingRigidView:
        def get_world_poses(self, *, indices):
            raise RuntimeError("view read failed")

    with pytest.raises(RuntimeError, match="failed to read tiled object 'Tblock'"):
        _read_tiled_object_states(
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
    from linkerbot_sim.app.interactive.tiled.object_states import (
        _restore_tiled_object_pose_snapshot,
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
        "linkerbot_sim.app.interactive.tiled.object_states._apply_prim_local_pose_and_zero_velocity",
        fake_apply,
    )

    with pytest.raises(RuntimeError, match="failed to restore tiled object 'Tblock'"):
        _restore_tiled_object_pose_snapshot(
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
    from linkerbot_sim.app.interactive.tiled.object_states import (
        TiledDynamicChainObjectPoseView,
        _capture_tiled_object_pose_snapshot,
        _restore_tiled_object_pose_snapshot,
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
        "linkerbot_sim.app.interactive.tiled.object_states._apply_prim_local_pose_and_zero_velocity",
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
    )

    snapshot = _capture_tiled_object_pose_snapshot(
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
        np.asarray([[0.1, 0.0, -0.1], [0.1, 0.1, -0.1]], dtype=float),
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

    restored = _restore_tiled_object_pose_snapshot(
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
    from linkerbot_sim.app.interactive.tiled.isaac_runtime import (
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
    from linkerbot_sim.app.interactive.tiled.isaac_runtime import (
        _create_tiled_object_pose_views,
    )
    from linkerbot_sim.app.interactive.tiled.object_states import (
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
    from linkerbot_sim.app.interactive.tiled.isaac_runtime import (
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

    from linkerbot_sim.app.interactive.tiled.object_states import (
        _apply_prim_local_pose_and_zero_velocity,
    )

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(1.0, 2.0, 3.0)
    )
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0))

    applied = _apply_prim_local_pose_and_zero_velocity(
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

    from linkerbot_sim.app.interactive.tiled.object_states import (
        _apply_prim_local_pose_and_zero_velocity,
    )

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(1.0, 2.0, 3.0)
    )
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0))

    applied = _apply_prim_local_pose_and_zero_velocity(
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


def test_partial_reset_only_resets_selected_env() -> None:
    module = load_tiled_interactive_module()
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


def test_get_state_and_set_state_support_selected_envs() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    set_response = module.handle_tiled_interactive_message(
        {
            "type": "set_state",
            "env_ids": [1],
            "state": {
                "joint_positions": [[0.3, 0.2, 0.1]],
                "tcp_positions_world": [[2.2, 0.1, 0.0]],
                "episode_steps": [7],
            },
        },
        runtime,
    )
    state_response = module.handle_tiled_interactive_message(
        {
            "type": "get_state",
            "env_ids": [1],
            "fields": ["joint_positions", "episode_steps"],
        },
        runtime,
    )

    assert set_response["event"] == "set_state"
    assert state_response["event"] == "state"
    assert state_response["env_ids"] == [1]
    assert state_response["state"] == {
        "joint_positions": [[0.3, 0.2, 0.1]],
        "episode_steps": [7],
    }


def test_snapshot_protocol_restores_selected_envs() -> None:
    module = load_tiled_interactive_module()
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


def test_clone_state_protocol_copies_source_env() -> None:
    module = load_tiled_interactive_module()
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


def test_load_and_step_trajectory_replays_test_runtime_fake() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    loaded = module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "robot": "left",
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
            "robot": "left",
            "decimation": 5,
        },
        runtime,
    )

    assert loaded["event"] == "trajectory_loaded"
    assert loaded["robot"] == "left"
    assert stepped["event"] == "trajectory_step"
    assert stepped["ticks"] == 5
    # make_runtime 使用 100Hz physics，所以 5 tick 后应在 0.05s，即轨迹中点。
    np.testing.assert_allclose(runtime.current_positions[0], [0.5, 0.25, -0.25])
    np.testing.assert_allclose(runtime.current_positions[1], [1.0, 0.5, -0.5])


def test_trajectory_load_prefix_fills_missing_command_joints() -> None:
    module = load_tiled_interactive_module()
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
            "decimation": 10,
        },
        runtime,
    )

    assert response["event"] == "trajectory_step"
    np.testing.assert_allclose(runtime.current_positions[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(runtime.current_positions[1], [10.0, 5.0, 6.0])


def test_load_trajectory_accepts_sync_hand_overlay() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[0.0, 0.0, 0.2], [0.0, 0.0, 0.4]]

    loaded = module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "robot": "left",
            "env_ids": [0, 1],
            "times": [0.0, 0.1],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "joint_names": ["joint_0", "joint_1", "joint_2"],
            "overlays": [
                {
                    "timing": "sync",
                    "left_hand": {
                        "joint_positions": {"joint_2": 0.8},
                    },
                }
            ],
        },
        runtime,
    )
    stepped = module.handle_tiled_interactive_message(
        {"type": "step_trajectory", "robot": "left", "decimation": 5},
        runtime,
    )

    assert loaded["event"] == "trajectory_loaded"
    assert loaded["overlay_count"] == 1
    assert stepped["event"] == "trajectory_step"
    np.testing.assert_allclose(runtime.current_positions[0], [0.5, 0.0, 0.5])
    np.testing.assert_allclose(runtime.current_positions[1], [0.5, 0.0, 0.6])


def test_load_trajectory_accepts_before_and_after_hand_overlays() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    loaded = module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "robot": "left",
            "env_ids": [0, 1],
            "times": [0.0, 0.1],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "joint_names": ["joint_0", "joint_1", "joint_2"],
            "overlays": [
                {
                    "timing": "before",
                    "left_hand": {
                        "duration_s": 0.1,
                        "joint_positions": {"joint_2": 0.2},
                    },
                },
                {
                    "timing": "sync",
                    "left_hand": {
                        "duration_s": 0.1,
                        "joint_positions": {"joint_2": 0.8},
                    },
                },
                {
                    "timing": "after",
                    "left_hand": {
                        "duration_s": 0.1,
                        "joint_positions": {"joint_2": 0.0},
                    },
                },
            ],
        },
        runtime,
    )
    before = module.handle_tiled_interactive_message(
        {"type": "step_trajectory", "robot": "left", "decimation": 10},
        runtime,
    )
    main = module.handle_tiled_interactive_message(
        {"type": "step_trajectory", "robot": "left", "decimation": 10},
        runtime,
    )
    after = module.handle_tiled_interactive_message(
        {"type": "step_trajectory", "robot": "left", "decimation": 10},
        runtime,
    )

    assert loaded["event"] == "trajectory_loaded"
    assert before["event"] == "trajectory_step"
    np.testing.assert_allclose(before["joint_positions"][0], [0.0, 0.0, 0.2])
    np.testing.assert_allclose(main["joint_positions"][0], [1.0, 0.0, 0.8])
    np.testing.assert_allclose(after["joint_positions"][0], [1.0, 0.0, 0.0], atol=1e-12)


def test_trajectory_status_and_clear_controls() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "times": [0.0, 1.0],
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        },
        runtime,
    )
    status = module.handle_tiled_interactive_message(
        {"type": "trajectory_status"},
        runtime,
    )
    cleared = module.handle_tiled_interactive_message(
        {"type": "clear_trajectory", "env_ids": [0]},
        runtime,
    )

    assert status["event"] == "trajectory_status"
    assert status["trajectory"]["robots"]["debug"]["active_env_ids"] == [0, 1]
    assert cleared["event"] == "trajectory_cleared"
    assert cleared["cleared"] == {"debug": [0]}


def test_reset_clears_selected_trajectory_env() -> None:
    module = load_tiled_interactive_module()
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
        {"type": "trajectory_status"},
        runtime,
    )

    assert status["trajectory"]["robots"]["debug"]["active_env_ids"] == [0]


def test_plan_status_loads_ready_trajectory_and_step_replays_it() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    submitted = module.handle_tiled_interactive_message(
        {
            "type": "plan",
            "robot": "left",
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
            "robot": "left",
            "decimation": 10,
        },
        runtime,
    )

    assert submitted["event"] == "plan_submitted"
    assert submitted["request_id"] == "plan-1"
    assert status["event"] == "planner_status"
    assert status["loaded"] == [
        {"request_id": "plan-1", "robot": "left", "env_ids": [0, 1]}
    ]
    assert stepped["event"] == "trajectory_step"
    np.testing.assert_allclose(runtime.current_positions[0], [1.0, 0.5, -0.5])
    np.testing.assert_allclose(runtime.current_positions[1], [2.0, 1.0, -1.0])


def test_async_plan_sync_hand_overlay_is_loaded_with_ready_result() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[0.0, 0.0, 0.2], [0.0, 0.0, 0.4]]

    submitted = module.handle_tiled_interactive_message(
        {
            "type": "plan",
            "robot": "left",
            "request_id": "plan-overlay",
            "duration_s": 0.1,
            "sample_dt_s": 0.05,
            "kind": "joint_position_target",
            "joint_positions": [1.0, 0.0, 0.0],
            "overlays": [
                {
                    "timing": "sync",
                    "left_hand": {
                        "duration_s": 0.05,
                        "joint_positions": {"joint_2": 0.8},
                    },
                }
            ],
        },
        runtime,
    )
    status = module.handle_tiled_interactive_message(
        {"type": "planner_status", "wait_timeout_s": 1.0},
        runtime,
    )
    stepped = module.handle_tiled_interactive_message(
        {"type": "step_trajectory", "robot": "left", "decimation": 5},
        runtime,
    )

    assert submitted["event"] == "plan_submitted"
    assert status["ready"][0]["overlay_count"] == 1
    assert status["loaded"] == [
        {"request_id": "plan-overlay", "robot": "left", "env_ids": [0, 1]}
    ]
    assert stepped["event"] == "trajectory_step"
    np.testing.assert_allclose(runtime.current_positions[0], [0.5, 0.0, 0.8])
    np.testing.assert_allclose(runtime.current_positions[1], [0.5, 0.0, 0.8])


def test_independent_hand_motion_queues_after_active_trajectory() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()
    runtime.current_positions[:] = [[0.0, 0.0, 0.2], [0.0, 0.0, 0.4]]

    module.handle_tiled_interactive_message(
        {
            "type": "load_trajectory",
            "robot": "left",
            "times": [0.0, 0.1],
            "positions": [[0.0], [1.0]],
        },
        runtime,
    )
    queued = module.handle_tiled_interactive_message(
        {
            "type": "hand",
            "robot": "left",
            "duration_s": 0.1,
            "joint_positions": {"joint_2": 0.8},
        },
        runtime,
    )
    main = module.handle_tiled_interactive_message(
        {"type": "step_trajectory", "robot": "left", "decimation": 10},
        runtime,
    )
    hand = module.handle_tiled_interactive_message(
        {"type": "step_trajectory", "robot": "left", "decimation": 10},
        runtime,
    )

    assert queued["event"] == "hand_motion_queued"
    assert queued["motions"][0]["queued"] is True
    np.testing.assert_allclose(main["joint_positions"][0], [1.0, 0.0, 0.2])
    np.testing.assert_allclose(hand["joint_positions"][0], [1.0, 0.0, 0.8])


def test_clear_completed_planner_results_from_interactive_message() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    module.handle_tiled_interactive_message(
        {
            "type": "plan",
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
    module = load_tiled_interactive_module()
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
        {"type": "step_trajectory", "decimation": 10},
        runtime,
    )

    np.testing.assert_allclose(runtime.current_positions[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(runtime.current_positions[1], [2.5, 1.5, 2.0])


def test_plan_task_space_line_submits_async_plan() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "plan",
            "kind": "task_space_line",
            "robot": "left",
            "target_offset": [0.0, 0.0, 0.1],
            "duration_s": 1.0,
        },
        runtime,
    )

    assert response["event"] == "plan_submitted"
    assert response["robot"] == "left"
    assert response["segments"] == ["task_space_line"]
    status = module.handle_tiled_interactive_message(
        {"type": "planner_status", "wait_timeout_s": 1.0},
        runtime,
    )
    assert status["ready"][0]["success"] is False
    assert status["ready"][0]["status"] == "UNSUPPORTED"


def test_old_task_space_line_message_is_rejected() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "task_space_line",
            "side": "left",
            "target_offset": [0.0, 0.0, 0.1],
            "duration_s": 1.0,
        },
        runtime,
    )

    assert response["event"] == "rejected"
    assert "unsupported tiled action" in response["error"]


def test_plan_queue_old_cspace_move_specs_are_rejected() -> None:
    module = load_tiled_interactive_module()
    runtime = make_runtime()

    response = module.handle_tiled_interactive_message(
        {
            "type": "plan_queue",
            "robot": "left",
            "request_id": "queue-1",
            "moves": [
                {
                    "type": "cspace_delta",
                    "duration_s": 0.05,
                    "sample_dt_s": 0.05,
                    "joint_deltas": [1.0, 0.0, 0.0],
                },
                {
                    "type": "cspace_goal",
                    "duration_s": 0.05,
                    "sample_dt_s": 0.05,
                    "joint_positions": [1.0, 2.0, 0.0],
                },
            ],
        },
        runtime,
    )

    assert response["event"] == "rejected"
    assert "unsupported tiled action" in response["error"]


def test_filter_isaac_state_fields_supports_nested_robot_paths() -> None:
    filtered = tiled_command_utils._filter_isaac_state_fields(
        {
            "robots": {
                "left": {
                    "joint_names": ["j1", "j2"],
                    "joint_positions": [[0.1, 0.2]],
                    "joint_velocities": [[0.0, 0.0]],
                },
                "right": {
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
            "robots.left.joint_positions",
            "objects.Tblock.positions_world",
            "episode_steps",
            "unknown",
        ),
    )

    assert filtered == {
        "robots": {"left": {"joint_positions": [[0.1, 0.2]]}},
        "objects": {"Tblock": {"positions_world": [[0.2, 0.0, -0.4]]}},
        "episode_steps": [5],
    }


def test_world_frame_batched_ik_solver_converts_between_world_and_base() -> None:
    calls = []

    class FakeSolver:
        tcp_frame_name = "tool"

        def solve(self, **kwargs):
            calls.append(kwargs)
            return BatchedIKResult(
                joint_positions=np.asarray(kwargs["seeds"], dtype=float),
                success=np.ones(2, dtype=bool),
                position_error=np.zeros(2, dtype=float),
                orientation_error=None,
                status=("SUCCESS", "SUCCESS"),
            )

        def compute_tcp_poses(self, command_positions, *, tcp_frame_name=None):
            return (
                np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                    dtype=float,
                ),
            )

    rotation_z_90 = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    wrapper = tiled_isaac_ik_solver._WorldFrameBatchedIKSolver(
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
    tcp_positions, _ = wrapper.command_tcp_world_poses(np.zeros((2, 2)))

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


def test_isaac_runtime_joint_action_applies_interpolated_tick_targets(
    monkeypatch,
) -> None:
    module = load_tiled_interactive_module()
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

    view = FakeView()
    view.positions[1, :] = [10.0, 20.0]
    world = FakeWorld()
    target_positions = np.asarray([[0.0, 0.0], [7.0, 8.0]], dtype=float)

    def fake_apply_joint_targets(view_arg, targets, *, joint_indices):
        applied.append(np.asarray(targets, dtype=float).copy())
        view_arg.positions[:, np.asarray(joint_indices, dtype=int)] = targets

    monkeypatch.setattr(
        "linkerbot_sim.app.interactive.tiled.isaac_runtime._apply_joint_targets",
        fake_apply_joint_targets,
    )
    monkeypatch.setattr(
        module.IsaacTiledInteractiveRuntime,
        "_refresh_tcp_state",
        lambda self, robot_name, env_ids=None: None,
    )
    runtime = module.IsaacTiledInteractiveRuntime(
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
            )
        },
        ik_solvers={},
        tcp_positions_world={},
        tcp_orientations_wxyz={},
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
    finally:
        runtime.close = lambda: None
        runtime.planner_manager.shutdown()

    assert response["ticks"] == 2
    assert response["env_ids"] == [0]
    assert world.steps == 2
    np.testing.assert_allclose(applied[0], [[0.5, 1.0], [7.0, 8.0]])
    np.testing.assert_allclose(applied[1], [[1.0, 2.0], [7.0, 8.0]])


def test_isaac_runtime_requires_message_robot_selection_for_multi_robot_actions() -> (
    None
):
    module = load_tiled_interactive_module()
    from linkerbot_sim.app.interactive.tiled.protocol import ALL_ROBOTS

    runtime = object.__new__(module.IsaacTiledInteractiveRuntime)
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
