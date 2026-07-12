"""Isaac TiledSceneRuntime 的批量状态读写、snapshot 与 env clone 服务。

``get_state``/``set_state`` 操作当前 runtime 的内部字段，snapshot API 则使用与 runtime
解耦的可移植模型。所有写入先完成字段、shape、env ID 和可回滚状态校验；清空轨迹与取消
planner 属于不可逆边界，此后失败会让 mutation transaction 将 runtime 标记为不可修改。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from linkerbot_sim.app.interactive.tiled_scene.command_utils import (
    _filter_isaac_state_fields,
    _jsonable,
    _normalize_env_ids,
    _selected_rows,
)
from linkerbot_sim.snapshots.tiled_scene_adapter import (
    clone_tiled_env_state,
    get_tiled_scene_snapshot,
    set_tiled_scene_snapshot,
)
from linkerbot_sim.snapshots.transactions import (
    mutation_transaction,
    require_runtime_mutable,
)
from linkerbot_sim.tiled.state.object_io import read_tiled_object_states

if TYPE_CHECKING:
    from linkerbot_sim.app.interactive.tiled_scene.runtime.core import (
        TiledSceneRuntime,
    )


def get_state(
    runtime: "TiledSceneRuntime",
    *,
    env_ids: np.ndarray,
    fields: tuple[str, ...] | None = None,
    include_efforts: bool = False,
) -> dict[str, object]:
    """读取 selected env 的机器人、对象和 episode 状态。

    机器人矩阵始终按请求 env 顺序和 command-joint 顺序返回。effort getter 不存在或返回
    shape 不符时省略对应可选字段，不用伪造零值；``fields`` 过滤在完整 payload 组装后执行。
    """

    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    robots = {}
    for name, articulation in runtime._selected_runtime_items(None):
        command_indices = articulation.command_joint_indices
        robot_state = {
            "joint_names": list(articulation.command_joint_names),
            "joint_positions": np.asarray(
                articulation.view.get_joint_positions(
                    indices=selected, joint_indices=command_indices
                ),
                dtype=float,
            ).tolist(),
            "joint_velocities": np.asarray(
                articulation.view.get_joint_velocities(
                    indices=selected, joint_indices=command_indices
                ),
                dtype=float,
            ).tolist(),
            "tcp_positions_world": runtime._tcp_positions(name)[selected].tolist(),
            "tcp_orientations_wxyz": runtime._tcp_orientations(name)[selected].tolist(),
        }
        if include_efforts:
            for field_name, method_name in (
                ("measured_efforts", "get_measured_joint_efforts"),
                ("applied_efforts", "get_applied_joint_efforts"),
            ):
                efforts = _optional_joint_efforts(
                    articulation.view,
                    method_name=method_name,
                    env_ids=selected,
                    joint_indices=command_indices,
                )
                if efforts is not None:
                    robot_state[field_name] = _jsonable(efforts)
        robots[name] = robot_state
    payload = {
        "robots": robots,
        "objects": object_states(runtime, selected),
        "episode_steps": runtime.episode_steps[selected].tolist(),
        "episode_ids": runtime.episode_ids[selected].tolist(),
    }
    if fields is not None:
        payload = _filter_isaac_state_fields(payload, fields)
    return {
        "event": "state",
        "accepted": True,
        "backend": "isaac",
        "env_ids": selected.tolist(),
        "step": runtime.step,
        "time_s": runtime.time_s,
        "state": payload,
    }


def _optional_joint_efforts(
    view: object,
    *,
    method_name: str,
    env_ids: np.ndarray,
    joint_indices: np.ndarray,
) -> np.ndarray | None:
    """读取 tiled effort matrix；wrapper 不支持时保持该模态缺失。"""

    method = getattr(view, method_name, None)
    if not callable(method):
        return None
    try:
        values = method(indices=env_ids, joint_indices=joint_indices)
    except (AttributeError, NotImplementedError, TypeError):
        return None
    array = np.asarray(values, dtype=float)
    expected = (int(env_ids.size), int(joint_indices.size))
    return array if array.shape == expected else None


def set_state(
    runtime: "TiledSceneRuntime",
    state: Mapping[str, object],
    *,
    env_ids: np.ndarray,
) -> dict[str, object]:
    """先完整校验 state，再事务式写回 selected env。

    本接口只接受机器人关节状态和 episode counters；对象状态必须使用 ``set_snapshot``。
    所有 articulation/caches 原值在第一个 setter 前捕获。位置写入后同步 target 与 TCP
    cache，随后 reset adapter；成功末尾清空相关轨迹并取消可能基于旧状态的规划请求。
    """

    require_runtime_mutable(runtime, operation="set_state")
    if not isinstance(state, Mapping):
        raise ValueError("set_state.state must be a JSON object")
    _validate_state_keys(state)
    selected = _normalize_env_ids(env_ids, runtime.scene.config.num_envs)
    robots = state.get("robots", {})
    if not isinstance(robots, Mapping):
        raise ValueError("set_state.state.robots must be a JSON object")
    runtime_items = dict(runtime._selected_runtime_items(None))
    unknown_robots = set(robots).difference(runtime_items)
    if unknown_robots:
        raise ValueError(
            "set_state.state.robots has unknown robots: "
            f"{sorted(str(name) for name in unknown_robots)}"
        )
    plans: list[_RobotStateWritePlan] = []
    for name, robot_state in robots.items():
        articulation = runtime_items[name]
        if not isinstance(robot_state, Mapping):
            raise ValueError(f"state.robots.{name} must be a JSON object")
        unknown_fields = set(robot_state).difference(
            {"joint_positions", "joint_velocities"}
        )
        if unknown_fields:
            raise ValueError(
                f"set_state.state.robots.{name} has unknown fields: "
                f"{sorted(str(field) for field in unknown_fields)}"
            )
        command_indices = articulation.command_joint_indices
        positions = (
            _selected_rows(
                robot_state["joint_positions"],
                selected.size,
                command_indices.size,
                f"robots.{name}.joint_positions",
            )
            if "joint_positions" in robot_state
            else None
        )
        velocities = (
            _selected_rows(
                robot_state["joint_velocities"],
                selected.size,
                command_indices.size,
                f"robots.{name}.joint_velocities",
            )
            if "joint_velocities" in robot_state
            else None
        )
        if positions is None and velocities is None:
            continue
        plans.append(
            _RobotStateWritePlan(
                name=name,
                articulation=articulation,
                command_indices=np.asarray(command_indices, dtype=int),
                positions=positions,
                velocities=velocities,
            )
        )

    episode_steps = (
        _selected_counter_rows(state["episode_steps"], selected.size, "episode_steps")
        if "episode_steps" in state
        else None
    )
    episode_ids = (
        _selected_counter_rows(state["episode_ids"], selected.size, "episode_ids")
        if "episode_ids" in state
        else None
    )
    if not plans and episode_steps is None and episode_ids is None:
        raise ValueError("set_state.state must contain at least one writable field")
    # 第一个 setter 前必须捕获全部物理状态与可变缓存，确保回滚基线来自同一提交前时刻。
    originals = _capture_original_robot_state(runtime, plans, selected)
    original_episode_steps = (
        runtime.episode_steps[selected].copy() if episode_steps is not None else None
    )
    original_episode_ids = (
        runtime.episode_ids[selected].copy() if episode_ids is not None else None
    )

    with mutation_transaction(runtime, operation="set_state") as transaction:
        for original in originals:
            plan = original.plan
            if plan.positions is not None:
                transaction.add_rollback(
                    f"robot {plan.name} positions and TCP cache",
                    lambda original=original: _restore_robot_positions(
                        runtime,
                        original,
                        selected,
                    ),
                )
                plan.articulation.view.set_joint_positions(
                    plan.positions,
                    indices=selected,
                    joint_indices=plan.command_indices,
                )
                transaction.add_rollback(
                    f"robot {plan.name} target cache",
                    lambda original=original: _restore_target_cache(
                        runtime,
                        original,
                        selected,
                    ),
                )
                runtime.target_positions[plan.name][selected, :] = plan.positions
            if plan.velocities is not None:
                transaction.add_rollback(
                    f"robot {plan.name} velocities",
                    lambda original=original: (
                        original.plan.articulation.view.set_joint_velocities(
                            original.velocities,
                            indices=selected,
                            joint_indices=original.plan.command_indices,
                        )
                    ),
                )
                plan.articulation.view.set_joint_velocities(
                    plan.velocities,
                    indices=selected,
                    joint_indices=plan.command_indices,
                )
            if plan.positions is not None:
                runtime._refresh_tcp_state(plan.name, env_ids=selected)
            transaction.add_rollback(
                f"robot {plan.name} command cache",
                lambda original=original: _restore_adapter_cache(original),
            )
            original.adapter.reset()
        if episode_steps is not None:
            assert original_episode_steps is not None
            transaction.add_rollback(
                "episode_steps",
                lambda: _restore_selected_array(
                    runtime.episode_steps,
                    selected,
                    original_episode_steps,
                ),
            )
            runtime.episode_steps[selected] = episode_steps
        if episode_ids is not None:
            assert original_episode_ids is not None
            transaction.add_rollback(
                "episode_ids",
                lambda: _restore_selected_array(
                    runtime.episode_ids,
                    selected,
                    original_episode_ids,
                ),
            )
            runtime.episode_ids[selected] = episode_ids
        transaction.mark_irreversible("trajectory buffer clear")
        runtime.trajectory_buffer.clear(env_ids=selected)
        transaction.mark_irreversible("planner cancellation")
        runtime.planner_manager.cancel_matching(env_ids=selected)
    return {
        "event": "set_state",
        "accepted": True,
        "backend": "isaac",
        "env_ids": selected.tolist(),
        "step": runtime.step,
        "time_s": runtime.time_s,
    }


@dataclass(frozen=True)
class _RobotStateWritePlan:
    """一个机器人经完整校验后的 selected-env 写入计划。"""

    name: str
    articulation: object
    command_indices: np.ndarray
    positions: np.ndarray | None
    velocities: np.ndarray | None


@dataclass(frozen=True)
class _OriginalRobotState:
    """提交前捕获的最小回滚状态。"""

    plan: _RobotStateWritePlan
    positions: np.ndarray | None
    velocities: np.ndarray | None
    targets: np.ndarray | None
    adapter: object
    adapter_target: object
    tcp_positions: np.ndarray | None
    tcp_orientations: np.ndarray | None


def _capture_original_robot_state(
    runtime: "TiledSceneRuntime",
    plans: list[_RobotStateWritePlan],
    selected: np.ndarray,
) -> tuple[_OriginalRobotState, ...]:
    """在首次 setter 前读取所有会被覆盖的 articulation/target 状态。"""

    originals = []
    for plan in plans:
        positions = (
            _selected_rows(
                plan.articulation.view.get_joint_positions(
                    indices=selected,
                    joint_indices=plan.command_indices,
                ),
                selected.size,
                plan.command_indices.size,
                f"runtime.robots.{plan.name}.joint_positions",
            )
            if plan.positions is not None
            else None
        )
        velocities = (
            _selected_rows(
                plan.articulation.view.get_joint_velocities(
                    indices=selected,
                    joint_indices=plan.command_indices,
                ),
                selected.size,
                plan.command_indices.size,
                f"runtime.robots.{plan.name}.joint_velocities",
            )
            if plan.velocities is not None
            else None
        )
        originals.append(
            _OriginalRobotState(
                plan=plan,
                positions=positions,
                velocities=velocities,
                targets=(
                    runtime.target_positions[plan.name][selected, :].copy()
                    if plan.positions is not None
                    else None
                ),
                adapter=(adapter := runtime._command_adapter(plan.name)),
                adapter_target=_copy_optional_cache(adapter, "last_target"),
                tcp_positions=_copy_selected_cache(
                    runtime,
                    "tcp_positions_world",
                    plan.name,
                    selected,
                ),
                tcp_orientations=_copy_selected_cache(
                    runtime,
                    "tcp_orientations_wxyz",
                    plan.name,
                    selected,
                ),
            )
        )
    return tuple(originals)


def _restore_robot_positions(
    runtime: "TiledSceneRuntime",
    original: _OriginalRobotState,
    selected: np.ndarray,
) -> None:
    """先恢复 articulation 位置，再恢复由它派生的 TCP cache。

    原 runtime 没有 TCP cache 时，回滚必须基于已恢复关节位置重新计算，不能继续使用失败
    写入期间产生的缓存。
    """

    assert original.positions is not None
    plan = original.plan
    plan.articulation.view.set_joint_positions(
        original.positions,
        indices=selected,
        joint_indices=plan.command_indices,
    )
    if original.tcp_positions is None or original.tcp_orientations is None:
        runtime._refresh_tcp_state(plan.name, env_ids=selected)
        return
    _restore_named_cache(
        runtime,
        "tcp_positions_world",
        plan.name,
        selected,
        original.tcp_positions,
    )
    _restore_named_cache(
        runtime,
        "tcp_orientations_wxyz",
        plan.name,
        selected,
        original.tcp_orientations,
    )


def _restore_target_cache(
    runtime: "TiledSceneRuntime",
    original: _OriginalRobotState,
    selected: np.ndarray,
) -> None:
    """恢复 ``set_state`` 改写的 command target 行。"""

    assert original.targets is not None
    runtime.target_positions[original.plan.name][selected, :] = original.targets


def _copy_optional_cache(owner: object, name: str) -> object:
    """复制可选 cache，并用 sentinel 保留“属性不存在”的区别。"""

    if not hasattr(owner, name):
        return _missing
    value = getattr(owner, name)
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else value


def _restore_adapter_cache(original: _OriginalRobotState) -> None:
    """在状态写入失败后恢复 adapter 的最后目标 cache。"""

    if original.adapter_target is _missing:
        return
    value = original.adapter_target
    copy = getattr(value, "copy", None)
    setattr(original.adapter, "last_target", copy() if callable(copy) else value)


def _copy_selected_cache(
    runtime: object,
    attribute: str,
    name: str,
    selected: np.ndarray,
) -> np.ndarray | None:
    """从可选 robot-keyed runtime cache 复制 selected env 行。"""

    values = getattr(runtime, attribute, None)
    if not isinstance(values, Mapping) or name not in values:
        return None
    return np.asarray(values[name][selected], dtype=float).copy()


def _restore_named_cache(
    runtime: object,
    attribute: str,
    name: str,
    selected: np.ndarray,
    values: np.ndarray,
) -> None:
    """在必需的 robot-keyed runtime cache 中恢复 selected env 行。"""

    cache = getattr(runtime, attribute)
    cache[name][selected] = values


def _restore_selected_array(
    array: np.ndarray,
    selected: np.ndarray,
    values: np.ndarray,
) -> None:
    """回滚 selected episode 行，并让赋值失败继续传播。"""

    array[selected] = values


def _selected_counter_rows(
    values: object,
    selected_count: int,
    label: str,
) -> np.ndarray:
    """只接受显式非负整数，并允许单元素广播到全部 selected env。

    bool 即使是 Python int 子类也会拒绝；浮点整数和数字字符串同样不做隐式转换，避免
    episode identity 被宽松 JSON 类型污染。
    """

    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D integer array")
    normalized: list[int] = []
    for index, value in enumerate(array):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(f"{label}[{index}] must be an integer")
        item = int(value)
        if item < 0:
            raise ValueError(f"{label}[{index}] must be nonnegative")
        normalized.append(item)
    result = np.asarray(normalized, dtype=int)
    if result.size == 1 and int(selected_count) != 1:
        return np.repeat(result, int(selected_count))
    if result.size != int(selected_count):
        raise ValueError(f"{label} length must be 1 or len(env_ids)")
    return result


def _validate_state_keys(state: Mapping[str, object]) -> None:
    """在任何读写前拒绝不支持或未知的内部状态字段。"""

    if "objects" in state:
        raise ValueError(
            "set_state.state.objects is unsupported; use set_snapshot for object state"
        )
    unknown = set(state).difference({"robots", "episode_steps", "episode_ids"})
    if unknown:
        raise ValueError(
            "set_state.state has unknown fields: "
            f"{sorted(str(field) for field in unknown)}"
        )


_missing = object()


def get_snapshot(
    runtime: "TiledSceneRuntime",
    *,
    env_id: int,
) -> dict[str, object]:
    """读取单个 env 的 runtime-neutral snapshot。"""

    snapshot = get_tiled_scene_snapshot(runtime, env_id=int(env_id))
    return {
        "event": "snapshot",
        "accepted": True,
        "backend": "isaac",
        "env_id": int(env_id),
        "step": runtime.step,
        "time_s": runtime.time_s,
        "snapshot": snapshot.as_dict(),
    }


def set_snapshot(
    runtime: "TiledSceneRuntime",
    snapshot: Mapping[str, object],
    *,
    env_ids: np.ndarray,
    label_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> dict[str, object]:
    """把一个与 runtime 解耦的 snapshot 广播写回 selected env。

    ``label_map`` 只负责把快照身份映射到当前场景身份；``strict=True`` 要求所有必需状态
    都可解析。事务、对象位姿和 planner 失效处理由统一 tiled snapshot adapter 执行。
    """

    result = set_tiled_scene_snapshot(
        runtime,
        snapshot,
        env_ids=env_ids,
        label_map=label_map,
        strict=bool(strict),
    )
    return {
        **result.as_dict(),
        "backend": "isaac",
        "step": runtime.step,
        "time_s": runtime.time_s,
    }


def clone_state(
    runtime: "TiledSceneRuntime",
    *,
    source_env_id: int,
    target_env_ids: np.ndarray,
    strict: bool = True,
) -> dict[str, object]:
    """通过 runtime-neutral snapshot 把一个源 env 克隆到多个目标 env。

    源状态只读取一次，随后以单次事务写入全部目标；目标 env 顺序原样保留在响应中。
    """

    result = clone_tiled_env_state(
        runtime,
        source_env_id=int(source_env_id),
        target_env_ids=target_env_ids,
        strict=bool(strict),
    )
    return {
        **result.as_dict(),
        "event": "state_cloned",
        "backend": "isaac",
        "source_env_id": int(source_env_id),
        "target_env_ids": [int(env_id) for env_id in target_env_ids],
        "step": runtime.step,
        "time_s": runtime.time_s,
    }


def object_states(
    runtime: "TiledSceneRuntime",
    env_ids: np.ndarray,
) -> dict[str, object]:
    """读取 selected env 中所有 runtime object 的 pose/state。"""

    return read_tiled_object_states(
        stage=runtime.session.stage,
        object_prim_paths=runtime.scene.object_prim_paths,
        env_origins=runtime.scene.env_origins,
        env_ids=env_ids,
        object_pose_views=runtime.object_pose_views,
    )
