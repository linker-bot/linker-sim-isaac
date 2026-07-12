"""Isaac TiledSceneRuntime 的主线程 world stepping、env reset 与同步动作执行。

本模块中的 PhysX/USD 读写全部发生在调用线程。动作先为所有选中机器人完整计算轨迹并
检查 IK 拒绝条件，确认可执行后才提交 adapter cache 和逐 tick 控制目标；reset 则在首次
setter 前捕获所有可回滚状态，并通过 mutation transaction 防止失败后继续使用污染 runtime。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.command_utils import (
    _action_decimation,
    _action_for_selected_envs,
    _action_width,
    _apply_joint_targets,
    _apply_runtime_mimic_targets,
    _jsonable_mapping,
    _linear_path_ik_steps,
    _normalize_env_ids,
    _selected_rows,
)
from linkerbot_sim.app.interactive.tiled_scene.selectors import RobotSelection
from linkerbot_sim.snapshots.transactions import (
    mutation_transaction,
    require_runtime_mutable,
)
from linkerbot_sim.tiled.control.adapter import (
    TiledIKRequestRejected,
    selected_ik_failure_env_ids,
)
from linkerbot_sim.tiled.control.types import TiledCommandAction
from linkerbot_sim.tiled.state.object_io import (
    capture_tiled_object_pose_snapshot,
    restore_tiled_object_pose_snapshot,
)

if TYPE_CHECKING:
    from linkerbot_sim.app.interactive.tiled_scene.runtime.core import (
        TiledSceneRuntime,
    )


def idle_step(runtime: "TiledSceneRuntime") -> None:
    """保持当前目标并推进一次 world，使 GUI 与 telemetry 持续刷新。"""

    require_runtime_mutable(runtime, operation="idle_step")
    for name, articulation in runtime._selected_runtime_items(None):
        _apply_joint_targets(
            articulation.view,
            runtime.target_positions[name],
            joint_indices=articulation.command_joint_indices,
        )
    step_world(runtime, phase="idle")
    for name, _articulation in runtime._selected_runtime_items(None):
        runtime._refresh_tcp_state(name)


def step_world(runtime: "TiledSceneRuntime", *, phase: str) -> None:
    """在调用线程推进一次 world，并同步采样 tiled sensor camera。

    mimic targets 必须先于 ``world.step`` 写入。全局 step 和各 env episode step 只在 world
    成功返回后递增；相机 observer 使用递增前的 ``sample_step``，与该帧控制输入对齐。
    """

    require_runtime_mutable(runtime, operation="step_world")
    for _name, articulation in runtime._selected_runtime_items(None):
        _apply_runtime_mimic_targets(articulation)
    sample_step = runtime.step
    runtime.session.world.step(render=runtime.render)
    runtime.step += 1
    runtime.episode_steps[:] += 1
    if runtime.camera_output is not None:
        runtime.camera_output.observer.observe(
            runtime.session.world,
            step=sample_step,
            phase=phase,
        )


def idle_period_s(runtime: "TiledSceneRuntime") -> float:
    """返回正有限的 rendering dt；不可用时使用 physics dt。

    两个 getter 都缺失、抛错或返回非法值时汇总原因并显式失败，避免事件循环以零周期
    busy-spin，或用未经配置的常量掩盖 runtime 错误。
    """

    errors: list[str] = []
    rendering_dt = getattr(runtime.session.world, "get_rendering_dt", None)
    if callable(rendering_dt):
        try:
            value = float(rendering_dt())
            if np.isfinite(value) and value > 0.0:
                return value
            errors.append(f"rendering dt is not positive and finite: {value!r}")
        except Exception as exc:
            errors.append(f"rendering dt failed: {exc}")
    physics_dt = getattr(runtime.session.world, "get_physics_dt", None)
    if callable(physics_dt):
        try:
            value = float(physics_dt())
            if np.isfinite(value) and value > 0.0:
                return value
            errors.append(f"physics dt is not positive and finite: {value!r}")
        except Exception as exc:
            errors.append(f"physics dt failed: {exc}")
    detail = "; ".join(errors) if errors else "dt getters are unavailable"
    raise ValueError(f"runtime idle period is unavailable: {detail}")


def reset(
    runtime: "TiledSceneRuntime",
    env_ids: np.ndarray,
) -> dict[str, object]:
    """事务式重置 selected env；无法完整回滚时让 runtime 永久 fail-stop。

    机器人全 DOF 状态、command/TCP cache、对象位姿和 episode counters 都在首次写入前
    捕获。trajectory clear 与 planner cancel 不可逆，之后若失败，transaction 会标记
    runtime 不可继续修改，防止旧轨迹与新物理状态混合执行。
    """

    require_runtime_mutable(runtime, operation="reset")
    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    # 在首次 reset setter 前捕获全部物理值和派生缓存；否则后续捕获可能读到半重置状态。
    robot_rollback = tuple(
        _capture_reset_robot(runtime, name, articulation, selected)
        for name, articulation in runtime._selected_runtime_items(None)
    )
    object_rollback = capture_tiled_object_pose_snapshot(
        stage=runtime.session.stage,
        object_prim_paths=runtime.scene.object_prim_paths,
        env_origins=runtime.scene.env_origins,
        env_ids=selected,
        object_pose_views=runtime.object_pose_views,
    )
    reset_object_names = set(runtime.initial_object_states)
    rollback_object_names = set(object_rollback)
    if reset_object_names != rollback_object_names:
        raise RuntimeError(
            "reset object state is not fully recoverable: "
            f"initial_only={sorted(reset_object_names - rollback_object_names)}, "
            f"current_only={sorted(rollback_object_names - reset_object_names)}"
        )
    original_episode_steps = runtime.episode_steps[selected].copy()
    original_episode_ids = runtime.episode_ids[selected].copy()

    objects_reset = 0
    with mutation_transaction(runtime, operation="reset") as transaction:
        for rollback in robot_rollback:
            name = rollback.name
            articulation = rollback.articulation
            transaction.add_rollback(
                f"robot {name} positions and TCP cache",
                lambda rollback=rollback: _restore_reset_robot_positions(
                    runtime,
                    rollback,
                    selected,
                ),
            )
            articulation.view.set_joint_positions(
                runtime.initial_joint_positions[name][selected],
                indices=selected,
            )
            transaction.add_rollback(
                f"robot {name} velocities",
                lambda rollback=rollback: (
                    rollback.articulation.view.set_joint_velocities(
                        rollback.velocities,
                        indices=selected,
                    )
                ),
            )
            articulation.view.set_joint_velocities(
                np.zeros_like(runtime.initial_joint_velocities[name][selected]),
                indices=selected,
            )
            transaction.add_rollback(
                f"robot {name} target cache",
                lambda rollback=rollback: _restore_reset_target_cache(
                    runtime,
                    rollback,
                    selected,
                ),
            )
            runtime.target_positions[name][selected, :] = (
                runtime.initial_joint_positions[name][selected][
                    :, articulation.command_joint_indices
                ]
            )
            transaction.add_rollback(
                f"robot {name} command cache",
                lambda rollback=rollback: _restore_reset_adapter_cache(rollback),
            )
            rollback.adapter.reset()
            runtime._refresh_tcp_state(name, env_ids=selected)
        if object_rollback:
            transaction.add_rollback(
                "tiled objects",
                lambda: _restore_reset_objects(
                    runtime,
                    object_rollback,
                    selected,
                ),
            )
        objects_reset = _restore_reset_objects(
            runtime,
            runtime.initial_object_states,
            selected,
        )
        transaction.mark_irreversible("trajectory buffer clear")
        runtime.trajectory_buffer.clear(env_ids=selected)
        transaction.mark_irreversible("planner cancellation")
        runtime.planner_manager.cancel_matching(env_ids=selected)
        transaction.add_rollback(
            "episode_steps",
            lambda: _restore_reset_episode_rows(
                runtime.episode_steps,
                selected,
                original_episode_steps,
            ),
        )
        runtime.episode_steps[selected] = 0
        transaction.add_rollback(
            "episode_ids",
            lambda: _restore_reset_episode_rows(
                runtime.episode_ids,
                selected,
                original_episode_ids,
            ),
        )
        runtime.episode_ids[selected] += 1
    return {
        "event": "reset",
        "accepted": True,
        "env_ids": selected.tolist(),
        "step": runtime.step,
        "time_s": runtime.time_s,
        "episode_steps": runtime.episode_steps.tolist(),
        "episode_ids": runtime.episode_ids.tolist(),
        "objects_reset": int(objects_reset),
    }


@dataclass(frozen=True)
class _ResetRobotRollbackState:
    """reset 开始前捕获的完整机器人状态和派生缓存。

    positions/velocities 覆盖全 DOF，而 targets 只覆盖 command joints；TCP 和 adapter cache
    允许不存在，此时回滚路径重新计算或保持缺失语义。
    """

    name: str
    articulation: object
    positions: np.ndarray
    velocities: np.ndarray
    targets: np.ndarray
    adapter: object
    adapter_target: object
    tcp_positions: np.ndarray | None
    tcp_orientations: np.ndarray | None


def _capture_reset_robot(
    runtime: "TiledSceneRuntime",
    name: str,
    articulation: object,
    selected: np.ndarray,
) -> _ResetRobotRollbackState:
    """读取单个机器人的全 DOF 状态及可变 command/TCP caches。

    返回数组全部复制，避免 Isaac view 或 numpy advanced indexing 的共享内存在 reset 期间
    改写回滚基线。
    """

    adapter = runtime._command_adapter(name)
    return _ResetRobotRollbackState(
        name=name,
        articulation=articulation,
        positions=np.asarray(
            articulation.view.get_joint_positions(indices=selected),
            dtype=float,
        ).copy(),
        velocities=np.asarray(
            articulation.view.get_joint_velocities(indices=selected),
            dtype=float,
        ).copy(),
        targets=runtime.target_positions[name][selected, :].copy(),
        adapter=adapter,
        adapter_target=_copy_reset_cache(adapter, "last_target"),
        tcp_positions=_copy_reset_named_rows(
            runtime,
            "tcp_positions_world",
            name,
            selected,
        ),
        tcp_orientations=_copy_reset_named_rows(
            runtime,
            "tcp_orientations_wxyz",
            name,
            selected,
        ),
    )


def _restore_reset_robot_positions(
    runtime: "TiledSceneRuntime",
    rollback: _ResetRobotRollbackState,
    selected: np.ndarray,
) -> None:
    """先恢复全 DOF 位置，再恢复与该位置对应的 TCP cache。

    若原 runtime 没有缓存，则从已恢复 articulation 状态重新计算，不能保留 reset 后派生值。
    """

    rollback.articulation.view.set_joint_positions(
        rollback.positions,
        indices=selected,
    )
    if rollback.tcp_positions is None or rollback.tcp_orientations is None:
        runtime._refresh_tcp_state(rollback.name, env_ids=selected)
        return
    runtime.tcp_positions_world[rollback.name][selected] = rollback.tcp_positions
    runtime.tcp_orientations_wxyz[rollback.name][selected] = rollback.tcp_orientations


def _restore_reset_target_cache(
    runtime: "TiledSceneRuntime",
    rollback: _ResetRobotRollbackState,
    selected: np.ndarray,
) -> None:
    """恢复 selected env 的 command target 行。"""

    runtime.target_positions[rollback.name][selected, :] = rollback.targets


def _copy_reset_cache(owner: object, name: str) -> object:
    """复制可选 reset cache，并用 sentinel 区分“属性缺失”和值为 ``None``。"""

    if not hasattr(owner, name):
        return _reset_missing
    value = getattr(owner, name)
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else value


def _restore_reset_adapter_cache(rollback: _ResetRobotRollbackState) -> None:
    """恢复 adapter reset 所清除的最后目标缓存。"""

    if rollback.adapter_target is _reset_missing:
        return
    value = rollback.adapter_target
    copy = getattr(value, "copy", None)
    setattr(
        rollback.adapter,
        "last_target",
        copy() if callable(copy) else value,
    )


def _copy_reset_named_rows(
    runtime: object,
    attribute: str,
    name: str,
    selected: np.ndarray,
) -> np.ndarray | None:
    """从可选的 robot-keyed cache 复制 selected env 行。"""

    cache = getattr(runtime, attribute, None)
    if not isinstance(cache, Mapping) or name not in cache:
        return None
    return np.asarray(cache[name][selected], dtype=float).copy()


def _restore_reset_objects(
    runtime: "TiledSceneRuntime",
    snapshot: Mapping[str, Mapping[str, object]],
    selected: np.ndarray,
) -> int:
    """恢复每个请求的 object/env 组合，并拒绝静默的部分 reset。

    底层返回实际写入数量；它必须等于对象数乘 env 数，否则无法证明所有对象已进入同一
    episode 边界，事务会尝试回滚并报告失败。
    """

    restored = restore_tiled_object_pose_snapshot(
        stage=runtime.session.stage,
        object_prim_paths=runtime.scene.object_prim_paths,
        snapshot=snapshot,
        env_ids=selected,
        env_origins=runtime.scene.env_origins,
        object_pose_views=runtime.object_pose_views,
    )
    expected = len(snapshot) * int(selected.size)
    if restored != expected:
        raise RuntimeError(
            f"tiled object reset was incomplete: restored={restored}, expected={expected}"
        )
    return int(restored)


def _restore_reset_episode_rows(
    values: np.ndarray,
    selected: np.ndarray,
    original: np.ndarray,
) -> None:
    """恢复 episode counter 行，并让任何赋值失败传播给事务。"""

    values[selected] = original


_reset_missing = object()


def step_action(
    runtime: "TiledSceneRuntime",
    action: TiledCommandAction,
    *,
    env_ids: np.ndarray,
    robot_names: RobotSelection = None,
) -> dict[str, object]:
    """为选中机器人准备并同步执行一条 command action。

    所有机器人轨迹和 IK 结果先完成准备；``reject_request`` 命中任一 env 时，在写入 target
    cache 或推进 world 前拒绝整个请求。执行阶段逐 tick 更新目标并调用 ``step_world``，
    最后只刷新选中 env 的 TCP cache。返回的 ``ticks`` 是实际推进的 physics step 数。
    """

    require_runtime_mutable(runtime, operation="step_action")
    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    physics_dt = float(runtime.session.world.get_physics_dt())
    ticks = _action_decimation(
        action,
        default_decimation=runtime.default_decimation,
        physics_dt=physics_dt,
    )
    ik_steps = (
        _linear_path_ik_steps(
            action,
            execution_ticks=ticks,
            physics_dt=physics_dt,
        )
        if action.kind == "ee_linear_path"
        else None
    )
    info: dict[str, object] = {}
    selected_robots = runtime._selected_runtime_items(
        robot_names, require_explicit=True
    )
    trajectories: dict[str, tuple[object, np.ndarray, np.ndarray]] = {}
    prepared_adapter_targets: dict[str, np.ndarray] = {}
    rejected_ik_env_ids: set[int] = set()
    for name, articulation in selected_robots:
        command_indices = articulation.command_joint_indices
        current = np.asarray(
            articulation.view.get_joint_positions(joint_indices=command_indices),
            dtype=float,
        )
        previous_target = runtime.target_positions[name].copy()
        if action.kind.startswith("ee_"):
            adapter = runtime._command_adapter(name)
            robot_action = runtime._action_for_robot_reference(
                action, robot_name=name, env_ids=selected
            )
            robot_action = _action_for_selected_envs(
                action=robot_action,
                env_ids=selected,
                current_positions=current,
                current_tcp_positions=runtime._tcp_positions(name),
                current_tcp_orientations_wxyz=runtime._tcp_orientations(name),
                env_origins=runtime.scene.env_origins,
            )
            if action.kind == "ee_linear_path":
                seeds = previous_target.copy()
                seeds[selected, :] = current[selected, :]
                path = adapter.linear_path_to_joint_trajectory(
                    robot_action,
                    steps=int(ik_steps),
                    execution_steps=ticks,
                    current_positions=seeds,
                    current_tcp_positions=runtime._tcp_positions(name),
                    current_tcp_orientations_wxyz=runtime._tcp_orientations(name),
                    env_origins=runtime.scene.env_origins,
                    active_env_ids=selected,
                    update_last_target=False,
                )
                trajectories[name] = (
                    articulation,
                    path.joint_positions,
                    command_indices,
                )
                prepared_adapter_targets[name] = np.asarray(
                    path.joint_positions[-1], dtype=float
                ).copy()
                failed_env_ids = selected_ik_failure_env_ids(
                    path.info,
                    env_ids=selected,
                    num_envs=runtime.scene.config.num_envs,
                )
                path_info = dict(path.info)
                path_info["failed_env_ids"] = np.asarray(failed_env_ids, dtype=int)
                if adapter.failure_policy == "reject_request":
                    rejected_ik_env_ids.update(failed_env_ids)
                info[name] = {
                    "command_width": int(command_indices.size),
                    "ik": _jsonable_mapping(path_info),
                    "ik_backend": getattr(
                        getattr(adapter, "ik_solver", None),
                        "tcp_frame_name",
                        "",
                    ),
                }
                continue
            target = adapter.action_to_joint_target(
                robot_action,
                current_positions=current,
                current_tcp_positions=runtime._tcp_positions(name),
                current_tcp_orientations_wxyz=runtime._tcp_orientations(name),
                env_origins=runtime.scene.env_origins,
                update_last_target=False,
            )
            targets = previous_target.copy()
            targets[selected, :] = target.joint_positions[selected, :]
            prepared_adapter_targets[name] = targets.copy()
            start = previous_target.copy()
            start[selected, :] = current[selected, :]
            trajectories[name] = (
                articulation,
                adapter.interpolate_to(
                    targets,
                    start=start,
                    action=robot_action,
                ),
                command_indices,
            )
            failed_env_ids = selected_ik_failure_env_ids(
                target.info,
                env_ids=selected,
                num_envs=runtime.scene.config.num_envs,
            )
            target_info = dict(target.info)
            target_info["failed_env_ids"] = np.asarray(failed_env_ids, dtype=int)
            if adapter.failure_policy == "reject_request":
                rejected_ik_env_ids.update(failed_env_ids)
            info[name] = {
                "command_width": int(command_indices.size),
                "ik": _jsonable_mapping(target_info),
                "ik_backend": getattr(
                    getattr(adapter, "ik_solver", None),
                    "tcp_frame_name",
                    "",
                ),
            }
            continue
        width = _action_width(action, default_width=command_indices.size)
        joint_indices = command_indices[:width]
        start = previous_target[:, :width].copy()
        start[selected, :] = current[selected, :width]
        if action.kind == "hold":
            targets = previous_target[:, :width].copy()
        elif action.kind == "joint_position_target":
            targets = previous_target[:, :width].copy()
            targets[selected, :] = _selected_rows(
                action.values, selected.size, width, f"{action.kind}.values"
            )
        elif action.kind == "joint_delta_pos":
            targets = previous_target[:, :width].copy()
            targets[selected, :] = current[selected, :width] + _selected_rows(
                action.values, selected.size, width, f"{action.kind}.values"
            )
        else:
            raise ValueError(f"unsupported Isaac interactive action: {action.kind}")
        trajectories[name] = (
            articulation,
            runtime._command_adapter(name).interpolate_to(
                targets,
                start=start,
                action=action,
            ),
            joint_indices,
        )
        info[name] = {"command_width": int(width)}
    if rejected_ik_env_ids:
        raise TiledIKRequestRejected(sorted(rejected_ik_env_ids))
    for name, prepared_target in prepared_adapter_targets.items():
        runtime._command_adapter(name).last_target = prepared_target
    for tick_index in range(ticks):
        for name, (articulation, trajectory, joint_indices) in trajectories.items():
            tick_targets = trajectory[tick_index]
            runtime.target_positions[name][:, : tick_targets.shape[1]] = tick_targets
            _apply_joint_targets(
                articulation.view,
                tick_targets,
                joint_indices=joint_indices,
            )
        step_world(runtime, phase="action")
    for name, _articulation in selected_robots:
        runtime._refresh_tcp_state(name, env_ids=selected)
    response = {
        "event": "step",
        "accepted": True,
        "backend": "isaac",
        "kind": action.kind,
        "env_ids": selected.tolist(),
        "robots": [name for name, _ in selected_robots],
        "ticks": int(ticks),
        "step": runtime.step,
        "time_s": runtime.time_s,
        "episode_steps": runtime.episode_steps.tolist(),
        "info": info,
    }
    if action.kind == "ee_linear_path":
        response["duration_s"] = float(ticks) * physics_dt
        response["sample_dt_s"] = float(
            physics_dt if action.sample_dt_s is None else action.sample_dt_s
        )
        response["ik_waypoints"] = int(ik_steps)
    return response
