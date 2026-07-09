#!/usr/bin/env python3
"""双 tiled world MPC 闭环测试脚本。

这个脚本是一个一次性集成测试入口，不提供 CLI，也不走 tiled interactive
TCP/JSONL。主进程只负责调度；EXEC 和 PLAN 各自运行在独立 worker 进程中，从而隔离
Isaac ``SimulationApp``、``World`` 和全局 USD stage。
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import multiprocessing as mp
import os
import sys
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


EXEC_ENV = "scene_exec_tiled"
PLAN_ENV = "scene_plan_tiled"
EXEC_ENVS = 1
PLAN_ENVS = 32
CYCLES = 50
PLAN_HORIZON = 25
EXECUTE_STEPS = 5
DECIMATION = 4
DELTA_SCALE = 0.1
ARM_WIDTH = 7
EE_WIDTH = 3
SEED = 0
WORKER_POLL_S = 0.5
WORKER_CLOSE_TIMEOUT_S = 10.0
PROGRESS_LOG_INTERVAL = 5
ROBOT_STEP_ORDER = ("left", "right")
TBLOCK_OBJECT_NAME = "Tblock"
TARGET_TBLOCK_XY_YAW = np.asarray([0.2, 0.0, 0.0], dtype=float)
TOP_CANDIDATE_COUNT = 3

INITIAL_RIGHT = np.asarray(
    [1.64, -1.2, -1.5707, 1.57, 0.37, 0.0, 0.0],
    dtype=float,
)
INITIAL_LEFT = np.asarray(
    [1.5, 1.2, -1.5707, 1.57, -0.37, 0.0, 0.0],
    dtype=float,
)


class WorkerError(RuntimeError):
    """worker 进程返回失败或异常退出时抛出。"""


class RuntimeWorker:
    """主进程侧的 worker 连接句柄。"""

    def __init__(
        self,
        *,
        role: str,
        env_name: str,
        gui: bool,
        expected_num_envs: int,
        context: mp.context.BaseContext,
    ) -> None:
        """创建但不启动 worker 进程。"""

        self.role = role
        self.env_name = env_name
        self.gui = bool(gui)
        self.expected_num_envs = int(expected_num_envs)
        self._parent_conn, child_conn = context.Pipe()
        self._process = context.Process(
            target=_worker_main,
            args=(role, env_name, bool(gui), int(expected_num_envs), child_conn),
            name=f"tiled-mpc-{role.lower()}",
        )

    def start(self) -> dict[str, object]:
        """启动 worker，并等待 runtime 创建完成。"""

        self._process.start()
        ready = self._recv_checked("ready")
        if ready.get("event") != "ready":
            raise WorkerError(f"{self.role} expected ready, got {ready!r}")
        return ready

    def request(self, command: dict[str, object]) -> dict[str, object]:
        """发送一个同步命令，并返回 worker result。"""

        self._ensure_alive()
        self._parent_conn.send(command)
        return self._recv_checked(str(command.get("cmd", "")))

    def close(self) -> None:
        """尽量优雅关闭 worker；超时则 terminate。"""

        if self._process.is_alive():
            try:
                self._parent_conn.send({"cmd": "close"})
                deadline = time.monotonic() + WORKER_CLOSE_TIMEOUT_S
                while time.monotonic() < deadline:
                    if self._parent_conn.poll(WORKER_POLL_S):
                        self._parent_conn.recv()
                        break
                    if not self._process.is_alive():
                        break
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._process.join(timeout=WORKER_CLOSE_TIMEOUT_S)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=WORKER_CLOSE_TIMEOUT_S)
        self._parent_conn.close()

    def _recv_checked(self, label: str) -> dict[str, object]:
        """等待 worker 响应；如果进程提前退出则报告错误。"""

        while True:
            if self._parent_conn.poll(WORKER_POLL_S):
                response = self._parent_conn.recv()
                break
            self._ensure_alive()
        if not isinstance(response, dict):
            raise WorkerError(f"{self.role} returned non-dict response for {label}: {response!r}")
        if not response.get("ok", False):
            error = response.get("error", "unknown worker error")
            tb = response.get("traceback", "")
            raise WorkerError(f"{self.role} failed during {label}: {error}\n{tb}")
        result = response.get("result", {})
        return result if isinstance(result, dict) else {"value": result}

    def _ensure_alive(self) -> None:
        """确认 worker 进程仍在运行。"""

        if self._process.is_alive():
            return
        exitcode = self._process.exitcode
        raise WorkerError(f"{self.role} worker exited unexpectedly, exitcode={exitcode}")


def main() -> None:
    """执行 50 个 MPC outer cycles。"""

    # spawn 避免 fork 继承父进程里潜在的 Omni/Isaac 全局状态。
    context = mp.get_context("spawn")
    rng = np.random.default_rng(SEED)
    exec_worker = RuntimeWorker(
        role="EXEC",
        env_name=EXEC_ENV,
        gui=True,
        expected_num_envs=EXEC_ENVS,
        context=context,
    )
    plan_worker = RuntimeWorker(
        role="PLAN",
        env_name=PLAN_ENV,
        gui=False,
        expected_num_envs=PLAN_ENVS,
        context=context,
    )

    try:
        exec_ready = exec_worker.start()
        print(f"MPC_EXEC_READY {exec_ready}", flush=True)
        plan_ready = plan_worker.start()
        print(f"MPC_PLAN_READY {plan_ready}", flush=True)

        init_result = exec_worker.request({"cmd": "set_initial_exec_pose"})
        print(f"MPC_EXEC_INITIALIZED {init_result}", flush=True)

        for cycle in range(CYCLES):
            cycle_label = f"{cycle + 1}/{CYCLES}"
            cycle_started_at = time.monotonic()
            print(f"MPC_CYCLE_BEGIN cycle={cycle_label}", flush=True)

            print(f"MPC_SNAPSHOT_BEGIN cycle={cycle_label}", flush=True)
            snapshot_response = exec_worker.request({"cmd": "get_snapshot"})
            snapshot = snapshot_response["snapshot"]
            print(f"MPC_SNAPSHOT_DONE cycle={cycle_label}", flush=True)

            print(
                f"MPC_PLAN_SYNC_BEGIN cycle={cycle_label} env_count={PLAN_ENVS}",
                flush=True,
            )
            plan_sync = plan_worker.request(
                {
                    "cmd": "set_snapshot_all",
                    "snapshot": snapshot,
                }
            )
            print(
                "MPC_PLAN_SYNC_DONE "
                f"cycle={cycle_label} "
                f"env_count={len(plan_sync.get('env_ids', []))}",
                flush=True,
            )

            ee_actions = rng.uniform(
                -DELTA_SCALE,
                DELTA_SCALE,
                size=(PLAN_ENVS, PLAN_HORIZON, EE_WIDTH),
            )

            print(
                "MPC_PLAN_ROLLOUT_BEGIN "
                f"cycle={cycle_label} "
                f"env_count={PLAN_ENVS} "
                "action_kind=ee_delta_pos "
                f"target_tblock_xy_yaw={TARGET_TBLOCK_XY_YAW.tolist()} "
                f"high_level_steps={PLAN_HORIZON} "
                f"physics_ticks={PLAN_HORIZON * DECIMATION}",
                flush=True,
            )
            plan_rollout = plan_worker.request(
                {
                    "cmd": "rollout_ee_deltas",
                    "deltas": ee_actions,
                    "progress_label": f"cycle={cycle_label}",
                }
            )
            selected_env = int(plan_rollout["best_env"])
            print(
                "MPC_PLAN_SELECTION_DONE "
                f"cycle={cycle_label} "
                f"selected_env={selected_env} "
                f"best_cost={float(plan_rollout['best_cost']):.6f} "
                f"best_tblock_xy_yaw={plan_rollout['best_tblock_xy_yaw']} "
                f"best_error_xy_yaw={plan_rollout['best_error_xy_yaw']} "
                f"top_candidates={plan_rollout.get('top_candidates', [])}",
                flush=True,
            )
            print(
                "MPC_EXEC_APPLY_BEGIN "
                f"cycle={cycle_label} "
                f"selected_env={selected_env} "
                "action_kind=ee_delta_pos "
                f"high_level_steps={EXECUTE_STEPS} "
                f"physics_ticks={EXECUTE_STEPS * DECIMATION}",
                flush=True,
            )
            exec_result = exec_worker.request(
                {
                    "cmd": "execute_ee_deltas",
                    "deltas": ee_actions[selected_env, :EXECUTE_STEPS, :],
                    "progress_label": f"cycle={cycle_label} selected_env={selected_env}",
                }
            )
            print(
                "MPC_CYCLE_DONE "
                f"cycle={cycle_label} "
                f"selected_env={selected_env} "
                f"best_cost={float(plan_rollout['best_cost']):.6f} "
                f"exec_tblock_xy_yaw={exec_result.get('best_tblock_xy_yaw')} "
                f"plan_step={plan_rollout['runtime_step']} "
                f"exec_step={exec_result['runtime_step']} "
                f"plan_synced_env_count={len(plan_sync.get('env_ids', []))} "
                f"duration_s={time.monotonic() - cycle_started_at:.3f}",
                flush=True,
            )
    finally:
        plan_worker.close()
        exec_worker.close()
        print("MPC_DONE", flush=True)


def _worker_main(
    role: str,
    env_name: str,
    gui: bool,
    expected_num_envs: int,
    conn: Connection,
) -> None:
    """worker 进程入口：创建 runtime 并执行主进程发来的同步命令。"""

    runtime = None
    try:
        runtime, config_source = _create_runtime(
            role=role,
            env_name=env_name,
            gui=gui,
            expected_num_envs=expected_num_envs,
        )
        conn.send(
            {
                "ok": True,
                "result": {
                    "event": "ready",
                    "role": role,
                    "env_name": env_name,
                    "num_envs": _runtime_num_envs(runtime),
                    "gui": bool(gui),
                    "config_source": config_source,
                },
            }
        )
        while True:
            message = conn.recv()
            if not isinstance(message, dict):
                conn.send(_error_response(ValueError("worker command must be a dict")))
                continue
            command = str(message.get("cmd", ""))
            if command == "close":
                _close_runtime(runtime)
                runtime = None
                conn.send({"ok": True, "result": {"event": "closed", "role": role}})
                return
            try:
                result = _handle_worker_command(runtime, message)
            except Exception as exc:
                conn.send(_error_response(exc))
            else:
                conn.send({"ok": True, "result": result})
    except EOFError:
        pass
    except BaseException as exc:
        try:
            conn.send(_error_response(exc))
        except Exception:
            pass
    finally:
        _close_runtime(runtime)
        conn.close()


def _create_runtime(
    *,
    role: str,
    env_name: str,
    gui: bool,
    expected_num_envs: int,
):
    """在 worker 内创建 Isaac tiled runtime。"""

    from linkerbot_sim.app.interactive.tiled.isaac_runtime import (
        IsaacTiledInteractiveRuntime,
    )

    env_config, config_source = _load_worker_env_config(
        env_name,
        expected_num_envs=expected_num_envs,
    )
    runtime = IsaacTiledInteractiveRuntime.create(
        env_name=env_name,
        env_config=env_config,
        gui=bool(gui),
        default_decimation=DECIMATION,
        planner_backend="linear",
        planner_workers=1,
        max_pending_requests=64,
        max_completed_results=256,
    )
    _validate_runtime(runtime, role=role, expected_num_envs=expected_num_envs)
    return runtime, config_source


def _load_worker_env_config(
    env_name: str,
    *,
    expected_num_envs: int,
) -> tuple[dict[str, Any], str]:
    """加载 env profile，并把 tiled env 数修正为脚本硬编码值。"""

    from linkerbot_sim.configs.profiles import load_profile_yaml
    from linkerbot_sim.utils.config import load_yaml
    from linkerbot_sim.utils.paths import CONFIGS_ROOT

    config_source = env_name
    try:
        config = load_profile_yaml("env", env_name)
    except FileNotFoundError:
        base_path = CONFIGS_ROOT / "envs" / env_name / "base.yaml"
        if base_path.is_file():
            config = load_yaml(base_path)
            config_source = str(base_path)
        else:
            config = load_profile_yaml("env", "scene3_tiled")
            config_source = "scene3_tiled fallback"

    config = copy.deepcopy(config)
    tiled = config.setdefault("tiled", {})
    if not isinstance(tiled, dict):
        raise ValueError("env config tiled section must be a mapping")
    tiled["enabled"] = True
    tiled["num_envs"] = int(expected_num_envs)
    per_env = tiled.get("per_env")
    if isinstance(per_env, list):
        tiled["per_env"] = [
            item
            for item in per_env
            if isinstance(item, dict)
            and 0 <= int(item.get("env_id", -1)) < int(expected_num_envs)
        ]
    return config, config_source


def _handle_worker_command(runtime: object, message: dict[str, object]) -> dict[str, object]:
    """执行主进程发来的单条 worker 命令。"""

    command = str(message.get("cmd", ""))
    if command == "set_initial_exec_pose":
        return _set_initial_exec_pose(runtime)
    if command == "get_snapshot":
        return runtime.get_snapshot(env_id=0)
    if command == "set_snapshot_all":
        snapshot = message.get("snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("set_snapshot_all requires snapshot dict")
        env_ids = np.arange(_runtime_num_envs(runtime), dtype=int)
        return runtime.set_snapshot(snapshot, env_ids=env_ids)
    if command == "rollout_ee_deltas":
        env_count = _runtime_num_envs(runtime)
        deltas = _ee_delta_sequence(
            message.get("deltas"),
            env_count=env_count,
            label="deltas",
        )
        return _run_ee_delta_sequence(
            runtime,
            deltas=deltas,
            env_ids=np.arange(env_count, dtype=int),
            phase="mpc_plan",
            progress_label=str(message.get("progress_label", "")),
        )
    if command == "execute_ee_deltas":
        deltas = _ee_delta_sequence(
            message.get("deltas"),
            env_count=1,
            label="deltas",
        )
        return _run_ee_delta_sequence(
            runtime,
            deltas=deltas,
            env_ids=np.asarray([0], dtype=int),
            phase="mpc_exec",
            progress_label=str(message.get("progress_label", "")),
        )
    raise ValueError(f"unsupported worker command: {command!r}")


def _set_initial_exec_pose(runtime: object) -> dict[str, object]:
    """将 EXEC env0 设置到脚本指定的左右臂初始关节状态。"""

    selected = np.asarray([0], dtype=int)
    runtime.reset(env_ids=selected)
    positions: dict[str, np.ndarray] = {}
    for robot_name, arm_target in (("left", INITIAL_LEFT), ("right", INITIAL_RIGHT)):
        view_runtime = runtime.scene.articulation_views[robot_name]
        command_indices = np.asarray(view_runtime.command_joint_indices, dtype=int).reshape(-1)
        if command_indices.size < ARM_WIDTH:
            raise ValueError(
                f"{robot_name} command width {command_indices.size} is smaller than {ARM_WIDTH}"
            )
        q = np.asarray(
            view_runtime.view.get_joint_positions(
                indices=selected,
                joint_indices=command_indices,
            ),
            dtype=float,
        ).reshape(1, -1)
        q[:, :ARM_WIDTH] = arm_target.reshape(1, ARM_WIDTH)
        dq = np.zeros_like(q)
        view_runtime.view.set_joint_positions(
            q,
            indices=selected,
            joint_indices=command_indices,
        )
        view_runtime.view.set_joint_velocities(
            dq,
            indices=selected,
            joint_indices=command_indices,
        )
        runtime.target_positions[robot_name][selected, :] = q
        runtime._command_adapter(robot_name).reset()
        runtime._refresh_tcp_state(robot_name, env_ids=selected)
        positions[robot_name] = q[0, :ARM_WIDTH].copy()

    runtime.trajectory_buffer.clear(env_ids=selected)
    runtime.planner_manager.cancel_matching(env_ids=selected)
    return {
        "event": "initial_pose_set",
        "env_ids": [0],
        "left": positions["left"].tolist(),
        "right": positions["right"].tolist(),
        "runtime_step": int(runtime.step),
        "time_s": float(runtime.time_s),
    }


def _run_ee_delta_sequence(
    runtime: object,
    *,
    deltas: np.ndarray,
    env_ids: np.ndarray,
    phase: str,
    progress_label: str = "",
) -> dict[str, object]:
    """顺序执行一段末端 xyz 微动序列，并用 Tblock pose cost 选择候选 env。

    ``deltas`` 的 shape 是 ``(env, high_level_step, 3)``，三个通道分别是 TCP 在
    world frame 下的 ``dx/dy/dz`` 微动。每个 high-level step 只驱动一个机器人：
    step 0 驱动 left，step 1 驱动 right，后续按 ``ROBOT_STEP_ORDER`` 交替。
    """

    high_level_steps = int(deltas.shape[1])
    physics_ticks = int(high_level_steps * DECIMATION)
    env_count = int(np.asarray(env_ids, dtype=int).reshape(-1).size)
    started_at = time.monotonic()
    label_suffix = f" {progress_label}" if progress_label else ""

    # worker 进程执行 PLAN rollout 时，主进程会同步等待 Pipe 返回；如果这里没有直接
    # 打印，长时间的 physics tick 会看起来像“卡住”。因此进度日志放在 worker
    # 内部，随仿真步推进实时 flush 到 stdout。
    print(
        "MPC_WORKER_SEQUENCE_BEGIN "
        f"phase={phase}{label_suffix} "
        "action_kind=ee_delta_pos "
        f"env_count={env_count} "
        f"high_level_steps={high_level_steps} "
        f"physics_ticks={physics_ticks}",
        flush=True,
    )
    for step_index in range(high_level_steps):
        robot_name = _robot_name_for_step(step_index)
        _step_single_arm_ee_delta(
            runtime,
            robot_name=robot_name,
            delta=deltas[:, step_index, :],
            env_ids=env_ids,
        )
        completed_steps = step_index + 1
        if completed_steps % PROGRESS_LOG_INTERVAL == 0 or completed_steps == high_level_steps:
            print(
                "MPC_WORKER_SEQUENCE_PROGRESS "
                f"phase={phase}{label_suffix} "
                f"robot={robot_name} "
                f"step={completed_steps}/{high_level_steps} "
                f"physics_ticks={completed_steps * DECIMATION}/{physics_ticks} "
                f"runtime_step={int(runtime.step)} "
                f"elapsed_s={time.monotonic() - started_at:.3f}",
                flush=True,
            )
    score = _score_tblock_pose(runtime, env_ids=env_ids)
    duration_s = time.monotonic() - started_at
    print(
        "MPC_WORKER_SEQUENCE_DONE "
        f"phase={phase}{label_suffix} "
        f"best_env={score['best_env']} "
        f"best_cost={score['best_cost']:.6f} "
        f"best_tblock_xy_yaw={score['best_tblock_xy_yaw']} "
        f"high_level_steps={high_level_steps} "
        f"physics_ticks={physics_ticks} "
        f"runtime_step={int(runtime.step)} "
        f"duration_s={duration_s:.3f}",
        flush=True,
    )
    return {
        "event": f"{phase}_done",
        "high_level_steps": high_level_steps,
        "physics_ticks": physics_ticks,
        "env_ids": [int(item) for item in np.asarray(env_ids, dtype=int).reshape(-1)],
        "runtime_step": int(runtime.step),
        "time_s": float(runtime.time_s),
        "duration_s": duration_s,
        **score,
    }


def _step_single_arm_ee_delta(
    runtime: object,
    *,
    robot_name: str,
    delta: np.ndarray,
    env_ids: np.ndarray,
) -> None:
    """对单只手臂执行一次 batched TCP xyz 微动。

    这里故意每个 high-level step 只调用一次 ``step_action``，并且只传入一个
    ``robot_name``。这样 left/right 不会在同一个高层 step 内同时动作，满足“左右臂交替、
    每步只动一个臂”的测试语义，同时复用 runtime 已有的 batched IK 路径。
    """

    from linkerbot_sim.app.interactive.tiled.command_utils import _normalize_env_ids
    from linkerbot_sim.tiled import TiledCommandAction

    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    delta_rows = _delta_rows(
        delta,
        row_count=selected.size,
        width=EE_WIDTH,
        label=f"{robot_name}.ee_delta_pos",
    )
    action = TiledCommandAction(
        kind="ee_delta_pos",
        values=delta_rows,
        decimation=DECIMATION,
    )
    runtime.step_action(action, env_ids=selected, robot_names=robot_name)


def _ee_delta_sequence(
    values: object,
    *,
    env_count: int,
    label: str,
) -> np.ndarray:
    """把 worker 输入的末端微动动作规范化为 ``(env, steps, 3)``。"""

    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and int(env_count) == 1:
        array = array.reshape(1, array.shape[0], array.shape[1])
    if array.ndim != 3:
        raise ValueError(f"{label} actions must have shape (env, steps, 3)")
    if array.shape[0] != int(env_count):
        raise ValueError(f"{label} action env dimension must be {int(env_count)}")
    if array.shape[1] < 1:
        raise ValueError(f"{label} actions cannot be empty")
    if array.shape[2] != EE_WIDTH:
        raise ValueError(f"{label} action width must be {EE_WIDTH}")
    return array.astype(float, copy=True)


def _delta_rows(
    values: np.ndarray,
    *,
    row_count: int,
    width: int,
    label: str,
) -> np.ndarray:
    """把单个 high-level step 的 delta 规范化为 ``(selected_envs, width)``。"""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{label} delta must be 1D or 2D")
    if array.shape[1] != int(width):
        raise ValueError(f"{label} delta width must be {int(width)}")
    if array.shape[0] == 1 and int(row_count) != 1:
        array = np.repeat(array, int(row_count), axis=0)
    if array.shape[0] != int(row_count):
        raise ValueError(f"{label} delta row count must be 1 or {int(row_count)}")
    return array.astype(float, copy=True)


def _robot_name_for_step(step_index: int) -> str:
    """按 high-level step 序号返回当前要动作的手臂。"""

    if not ROBOT_STEP_ORDER:
        raise ValueError("ROBOT_STEP_ORDER cannot be empty")
    return str(ROBOT_STEP_ORDER[int(step_index) % len(ROBOT_STEP_ORDER)])


def _score_tblock_pose(runtime: object, *, env_ids: np.ndarray) -> dict[str, object]:
    """按 Tblock 当前 x/y/yaw 到目标位姿的差距，为 PLAN env 打分。

    目标语义来自本脚本顶部的 ``TARGET_TBLOCK_XY_YAW``。这里明确只比较：
    ``positions_local.x``、``positions_local.y`` 和 yaw；z、roll、pitch 不参与 cost。
    """

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    object_states = runtime._object_states(selected)
    object_name, object_state = _select_object_state(object_states, TBLOCK_OBJECT_NAME)
    object_env_ids = np.asarray(object_state.get("env_ids", selected), dtype=int).reshape(-1)
    positions_local = np.asarray(
        object_state.get("positions_local", ()), dtype=float
    ).reshape(-1, 3)
    orientations = np.asarray(
        object_state.get("orientations_wxyz", ()), dtype=float
    ).reshape(-1, 4)
    row_count = min(object_env_ids.size, positions_local.shape[0], orientations.shape[0])
    if row_count < 1:
        raise RuntimeError(f"object {object_name!r} has no pose rows for scoring")
    object_env_ids = object_env_ids[:row_count]
    positions_local = positions_local[:row_count]
    orientations = orientations[:row_count]

    yaw = _yaw_from_quat_wxyz(orientations)
    pose_xy_yaw = np.column_stack(
        (positions_local[:, 0], positions_local[:, 1], yaw)
    )
    error_xy_yaw = pose_xy_yaw - TARGET_TBLOCK_XY_YAW.reshape(1, 3)
    error_xy_yaw[:, 2] = _wrap_angle(error_xy_yaw[:, 2])
    costs = np.sum(error_xy_yaw * error_xy_yaw, axis=1)
    best_row = int(np.argmin(costs))
    return {
        "object_name": object_name,
        "target_tblock_xy_yaw": TARGET_TBLOCK_XY_YAW.astype(float).tolist(),
        "best_env": int(object_env_ids[best_row]),
        "best_cost": float(costs[best_row]),
        "best_tblock_xy_yaw": pose_xy_yaw[best_row].astype(float).tolist(),
        "best_error_xy_yaw": error_xy_yaw[best_row].astype(float).tolist(),
        "top_candidates": _top_tblock_candidates(
            env_ids=object_env_ids,
            costs=costs,
            pose_xy_yaw=pose_xy_yaw,
            error_xy_yaw=error_xy_yaw,
        ),
    }


def _select_object_state(
    object_states: Mapping[str, object],
    object_name: str,
) -> tuple[str, Mapping[str, object]]:
    """按名字读取 object state；大小写不一致时做一次宽松匹配。"""

    if object_name in object_states and isinstance(object_states[object_name], Mapping):
        return object_name, object_states[object_name]
    lowered = object_name.lower()
    for name, state in object_states.items():
        if str(name).lower() == lowered and isinstance(state, Mapping):
            return str(name), state
    available = ", ".join(str(name) for name in object_states)
    raise RuntimeError(f"object {object_name!r} not found; available objects: {available}")


def _yaw_from_quat_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """从 wxyz 四元数中提取 yaw。

    这里的 yaw 是绕 z 轴的 RPY 第三项；roll/pitch 即使存在也不会进入 cost。
    """

    quat = np.asarray(quaternions, dtype=float).reshape(-1, 4)
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Tblock orientation quaternion must be non-zero")
    quat = quat / norms[:, None]
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    """把角度差归一化到 ``[-pi, pi]``，避免 2*pi 跳变影响 cost。"""

    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _top_tblock_candidates(
    *,
    env_ids: np.ndarray,
    costs: np.ndarray,
    pose_xy_yaw: np.ndarray,
    error_xy_yaw: np.ndarray,
) -> list[dict[str, object]]:
    """返回若干个 cost 最低的候选，方便日志中观察 PLAN 选择是否合理。"""

    order = np.argsort(np.asarray(costs, dtype=float))
    result: list[dict[str, object]] = []
    for row in order[: min(TOP_CANDIDATE_COUNT, order.size)]:
        index = int(row)
        result.append(
            {
                "env": int(env_ids[index]),
                "cost": float(costs[index]),
                "tblock_xy_yaw": pose_xy_yaw[index].astype(float).tolist(),
                "error_xy_yaw": error_xy_yaw[index].astype(float).tolist(),
            }
        )
    return result


def _validate_runtime(runtime: object, *, role: str, expected_num_envs: int) -> None:
    """校验 worker runtime 与本测试脚本的硬编码假设一致。"""

    num_envs = _runtime_num_envs(runtime)
    if num_envs != int(expected_num_envs):
        raise ValueError(f"{role} expected {expected_num_envs} envs, got {num_envs}")
    robot_names = set(getattr(runtime, "robot_names", ()))
    if {"left", "right"} - robot_names:
        raise ValueError(f"{role} runtime must contain left and right robots, got {robot_names}")
    for robot_name in ("left", "right"):
        view_runtime = runtime.scene.articulation_views[robot_name]
        command_width = int(np.asarray(view_runtime.command_joint_indices).reshape(-1).size)
        if command_width < ARM_WIDTH:
            raise ValueError(
                f"{role}.{robot_name} command width {command_width} is smaller than {ARM_WIDTH}"
            )


def _runtime_num_envs(runtime: object) -> int:
    """返回 tiled runtime env 数。"""

    return int(runtime.scene.config.num_envs)


def _close_runtime(runtime: object | None) -> None:
    """关闭 runtime，忽略空对象。"""

    if runtime is None:
        return
    close = getattr(runtime, "close", None)
    if callable(close):
        close()


def _error_response(exc: BaseException) -> dict[str, object]:
    """把 worker 异常转成可通过 Pipe 发送的错误响应。"""

    return {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(traceback.format_exception(exc)),
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"TILED_DUAL_WORLD_MPC_FAILED {type(exc).__name__}: {exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
