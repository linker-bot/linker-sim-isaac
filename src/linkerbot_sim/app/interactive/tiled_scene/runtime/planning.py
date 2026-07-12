"""Isaac TiledSceneRuntime 的轨迹回放、手部动作与异步规划服务。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.command_utils import (
    _apply_joint_targets,
    _normalize_env_ids,
    _positive_decimation,
)
from linkerbot_sim.app.interactive.tiled_scene.hand_messages import (
    load_interactive_hand_motion,
)
from linkerbot_sim.app.interactive.tiled_scene.plan_messages import (
    load_ready_planning_results,
    planning_request_from_message,
)
from linkerbot_sim.app.interactive.tiled_scene.selectors import RobotSelection
from linkerbot_sim.app.interactive.tiled_scene.trajectory_messages import (
    load_interactive_trajectory,
)
from linkerbot_sim.snapshots.transactions import require_runtime_mutable
from linkerbot_sim.tiled.planning.types import TiledPlanningResult

if TYPE_CHECKING:
    from linkerbot_sim.app.interactive.tiled_scene.runtime.core import (
        TiledSceneRuntime,
    )


def load_trajectory(
    runtime: "TiledSceneRuntime",
    trajectory: Mapping[str, object],
    *,
    env_ids: np.ndarray,
    robot_name: str | None = None,
) -> dict[str, object]:
    """把已规划好的关节轨迹载入回放缓冲。"""

    require_runtime_mutable(runtime, operation="load_trajectory")
    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    robot = runtime._single_selected_robot_name(robot_name)
    articulation = runtime.scene.articulation_views[robot]
    loaded = load_interactive_trajectory(
        runtime.trajectory_buffer,
        trajectory,
        env_ids=selected,
        robot_name=robot,
        current_positions=runtime.target_positions[robot][selected],
        command_joint_names=articulation.command_joint_names,
    )
    return {
        "event": "trajectory_loaded",
        "accepted": True,
        "backend": "isaac",
        **loaded,
        "step": runtime.step,
        "time_s": runtime.time_s,
    }


def step_trajectory(
    runtime: "TiledSceneRuntime",
    *,
    env_ids: np.ndarray,
    robot_names: RobotSelection = None,
    decimation: int | None = None,
) -> dict[str, object]:
    """按 physics tick 回放 ready trajectory，并同步推进真实 tiled scene。"""

    require_runtime_mutable(runtime, operation="step_trajectory")
    planner_results, planner_loaded, load_rejected = collect_planner_results(runtime)
    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    selected_robots = runtime._selected_runtime_items(
        robot_names, require_explicit=True
    )
    ticks = _positive_decimation(
        decimation, default_decimation=runtime.default_decimation
    )
    last_results: dict[str, object] = {}
    for _ in range(ticks):
        for name, articulation in selected_robots:
            result = runtime.trajectory_buffer.step(
                robot_name=name,
                current_positions=runtime.target_positions[name],
                dt_s=float(runtime.session.world.get_physics_dt()),
                env_ids=selected,
            )
            runtime.target_positions[name][:, :] = result.joint_positions
            _apply_joint_targets(
                articulation.view,
                runtime.target_positions[name],
                joint_indices=articulation.command_joint_indices,
            )
            last_results[name] = result.to_json()
        runtime._step_world(phase="trajectory")
        for name, _articulation in selected_robots:
            runtime._refresh_tcp_state(name, env_ids=selected)
    return {
        "event": "trajectory_step",
        "accepted": True,
        "backend": "isaac",
        "env_ids": selected.tolist(),
        "robots": [name for name, _ in selected_robots],
        "ticks": int(ticks),
        "step": runtime.step,
        "time_s": runtime.time_s,
        "episode_steps": runtime.episode_steps.tolist(),
        "trajectory": last_results,
        "planner_ready": [item.to_json() for item in planner_results],
        "planner_loaded": planner_loaded,
        "load_rejected": load_rejected,
    }


def submit_plan(
    runtime: "TiledSceneRuntime",
    message: Mapping[str, object],
    *,
    env_ids: np.ndarray,
    robot_name: str | None = None,
) -> dict[str, object]:
    """冻结当前 command state 并提交 tiled 异步规划请求。"""

    require_runtime_mutable(runtime, operation="submit_plan")
    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    robot = runtime._single_selected_robot_name(robot_name)
    articulation = runtime.scene.articulation_views[robot]
    request = planning_request_from_message(
        message,
        robot_name=robot,
        env_ids=selected,
        current_positions=runtime.target_positions[robot][selected],
        command_joint_names=articulation.command_joint_names,
        default_sample_dt_s=float(runtime.session.world.get_physics_dt()),
        default_tcp_frame_name=runtime._command_adapter(robot).tcp_frame_name,
        request_defaults=getattr(runtime, "planner_request_defaults", None),
        command_defaults=getattr(runtime, "command_defaults", None),
    )
    request_id = runtime.planner_manager.submit(request)
    return {
        "event": "plan_submitted",
        "accepted": True,
        "backend": "isaac",
        "request_id": request_id,
        "robot": robot,
        "env_ids": selected.tolist(),
        "duration_s": float(request.duration_s),
        "sample_dt_s": float(request.sample_dt_s),
        "segments": [segment.kind for segment in request.segments],
        "load_on_success": bool(request.load_on_success),
    }


def submit_hand_motion(
    runtime: "TiledSceneRuntime",
    message: Mapping[str, object],
    *,
    env_ids: np.ndarray,
    robot_name: str | None = None,
) -> dict[str, object]:
    """为一个明确 robot ID 解析出的机器人提交手部关节运动。"""

    require_runtime_mutable(runtime, operation="submit_hand_motion")
    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    if message.get("type") != "hand":
        raise ValueError("hand motion requires type='hand'")
    robot = runtime._single_selected_robot_name(robot_name)
    loaded = [
        load_hand_payload(
            runtime,
            message,
            robot_name=robot,
            env_ids=selected,
        )
    ]
    return {
        "event": "hand_motion_queued",
        "accepted": True,
        "backend": "isaac",
        "motions": loaded,
        "step": runtime.step,
        "time_s": runtime.time_s,
    }


def planner_status(
    runtime: "TiledSceneRuntime",
    *,
    wait_timeout_s: float = 0.0,
) -> dict[str, object]:
    """收集 ready result，并把成功结果自动载入 trajectory buffer。"""

    require_runtime_mutable(runtime, operation="planner_status")
    ready, loaded, load_rejected = collect_planner_results(
        runtime, timeout_s=wait_timeout_s
    )
    return {
        "event": "planner_status",
        "accepted": True,
        "backend": "isaac",
        "ready": [item.to_json() for item in ready],
        "loaded": loaded,
        "load_rejected": load_rejected,
        "planner": runtime.planner_manager.status(),
    }


def cancel_plan(
    runtime: "TiledSceneRuntime",
    *,
    request_id: str | None = None,
    env_ids: np.ndarray | None = None,
    robot_name: str | None = None,
) -> dict[str, object]:
    """按 request ID 或 robot/env selector 取消后台规划请求。"""

    require_runtime_mutable(runtime, operation="cancel_plan")
    robot = (
        None if robot_name is None else runtime._single_selected_robot_name(robot_name)
    )
    if request_id is not None:
        result: object = runtime.planner_manager.cancel(request_id)
    else:
        result = runtime.planner_manager.cancel_matching(
            robot_name=robot, env_ids=env_ids
        )
    return {
        "event": "plan_cancelled",
        "accepted": True,
        "backend": "isaac",
        "result": result,
    }


def clear_completed(
    runtime: "TiledSceneRuntime",
    *,
    request_ids: str | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """清理 planner completed result 缓存。"""

    require_runtime_mutable(runtime, operation="clear_completed")
    return {
        "event": "completed_cleared",
        "accepted": True,
        "backend": "isaac",
        "result": runtime.planner_manager.clear_completed(request_ids),
    }


def trajectory_status(
    runtime: "TiledSceneRuntime",
    *,
    env_ids: np.ndarray,
    robot_name: str | None = None,
) -> dict[str, object]:
    """返回 selected trajectory buffer 状态。"""

    robot = (
        None if robot_name is None else runtime._single_selected_robot_name(robot_name)
    )
    return {
        "event": "trajectory_status",
        "accepted": True,
        "backend": "isaac",
        "step": runtime.step,
        "time_s": runtime.time_s,
        "trajectory": runtime.trajectory_buffer.status(
            robot_name=robot, env_ids=env_ids
        ),
    }


def clear_trajectory(
    runtime: "TiledSceneRuntime",
    *,
    env_ids: np.ndarray,
    robot_name: str | None = None,
) -> dict[str, object]:
    """清理 selected trajectory buffer。"""

    require_runtime_mutable(runtime, operation="clear_trajectory")
    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    robot = (
        None if robot_name is None else runtime._single_selected_robot_name(robot_name)
    )
    cleared = runtime.trajectory_buffer.clear(robot_name=robot, env_ids=selected)
    return {
        "event": "trajectory_cleared",
        "accepted": True,
        "backend": "isaac",
        "cleared": cleared,
        "step": runtime.step,
        "time_s": runtime.time_s,
    }


def load_hand_payload(
    runtime: "TiledSceneRuntime",
    payload: Mapping[str, object],
    *,
    robot_name: str,
    env_ids: np.ndarray,
) -> dict[str, object]:
    """把单个 hand payload 写入 trajectory buffer。"""

    require_runtime_mutable(runtime, operation="load_hand_payload")
    articulation = runtime.scene.articulation_views[robot_name]
    return load_interactive_hand_motion(
        runtime.trajectory_buffer,
        payload,
        env_ids=env_ids,
        robot_name=robot_name,
        current_positions=runtime.target_positions[robot_name][env_ids],
        command_joint_names=articulation.command_joint_names,
    )


def collect_planner_results(
    runtime: "TiledSceneRuntime",
    *,
    timeout_s: float = 0.0,
) -> tuple[
    tuple[TiledPlanningResult, ...],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """收集 planner results，并把成功结果载入 trajectory buffer。"""

    require_runtime_mutable(runtime, operation="collect_planner_results")
    results = runtime.planner_manager.collect_ready(timeout_s=timeout_s)
    loaded, rejected = load_ready_planning_results(runtime.trajectory_buffer, results)
    return results, loaded, rejected
