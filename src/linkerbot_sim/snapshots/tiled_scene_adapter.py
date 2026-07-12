"""Isaac/debug Tiled Scene runtime 的快照读取、恢复、环境复制与目标描述。

canonical 快照始终描述单个 source env；写入时再将这一行状态广播到选中的多个 target
env。Isaac tensor 的首维固定是 env，机器人关节矩阵 shape 为
``(selected_env_count, command_joint_count)``，对象 body 矩阵则为
``(selected_env_count, body_count, 3|4)``。adapter 在这里显式完成单行与批量 shape 的
转换，schema 不携带 tiled 批次维。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from linkerbot_sim.snapshots.compatibility import (
    ObjectTargetDescriptor,
    RobotTargetDescriptor,
    SnapshotTargetDescriptor,
    require_snapshot_compatibility,
)
from linkerbot_sim.snapshots.debug_tiled_scene_adapter import (
    debug_tiled_scene_target_descriptor,
    get_debug_tiled_scene_snapshot,
)
from linkerbot_sim.snapshots.runtime_objects import (
    _asset_fingerprint_from_path,
    _object_profiles_by_name,
)
from linkerbot_sim.snapshots.schema import (
    ObjectSnapshot,
    RobotSnapshot,
    SimulationSnapshot,
    SnapshotMetadata,
    SnapshotRestoreResult,
)
from linkerbot_sim.snapshots.transactions import (
    mutation_transaction,
    require_runtime_mutable,
)
from linkerbot_sim.tiled.state.object_io import (
    read_tiled_object_states,
    restore_tiled_object_pose_snapshot,
)


def get_tiled_scene_snapshot(runtime: object, *, env_id: int) -> SimulationSnapshot:
    """从 TiledSceneRuntime 读取一个 env 的 runtime-neutral 快照。

    ``env_id`` 必须位于 ``[0, num_envs)``；返回对象会去掉 tiled batch 维，因此各机器人
    关节数组都是一维，metadata 的坐标系标记为 ``env-local``。
    """

    if _is_debug_tiled_scene_runtime(runtime):
        selected = int(_single_env_id(env_id, runtime.config.num_envs)[0])
        return get_debug_tiled_scene_snapshot(runtime, env_id=selected)
    return _get_isaac_tiled_scene_snapshot(runtime, env_id=env_id)


def set_tiled_scene_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot | Mapping[str, object],
    *,
    env_ids: Sequence[int] | np.ndarray,
    label_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """把一份单环境逻辑快照事务式广播到一个或多个 tiled env。

    ``env_ids`` 必须非空、无重复且全部在范围内。每个目标 env 的旧状态单独保存，因为
    它们在写入前通常互不相同；失败回滚不能拿 source 快照反向广播来替代这些旧值。
    debug 与 Isaac 路径共享兼容性、partial 结果和 fail-stop 语义。
    """

    require_runtime_mutable(runtime, operation="set_tiled_scene_snapshot")
    parsed = _snapshot_from_input(snapshot)
    if _is_debug_tiled_scene_runtime(runtime):
        selected = _env_ids(env_ids, runtime.config.num_envs)
        compatibility = require_snapshot_compatibility(
            parsed,
            debug_tiled_scene_target_descriptor(runtime),
            label_map=label_map,
            strict=strict,
        )
        # debug runtime 没有 PhysX view，但位置矩阵与 adapter target 仍必须一起回滚。
        original_positions = np.asarray(
            runtime.current_positions[selected], dtype=float
        ).copy()
        original_adapter_target = _copy_optional_attribute(
            runtime.adapter,
            "last_target",
        )
        with mutation_transaction(
            runtime,
            operation="set_tiled_scene_snapshot",
        ) as transaction:
            transaction.add_rollback(
                "debug joint positions",
                lambda: _restore_debug_positions(
                    runtime,
                    selected,
                    original_positions,
                ),
            )
            transaction.add_rollback(
                "debug command cache",
                lambda: _restore_optional_attribute(
                    runtime.adapter,
                    "last_target",
                    original_adapter_target,
                ),
            )
            restored = []
            for target_label, mapping in compatibility.robot_mappings.items():
                source_robot = parsed.robots[mapping.source_label]
                runtime.current_positions[
                    selected[:, None], mapping.joints.target_indices
                ] = source_robot.joint_positions[mapping.joints.source_indices][None, :]
                restored.append(target_label)
            runtime.adapter.reset()
            transaction.mark_irreversible("trajectory buffer clear")
            runtime.trajectory_buffer.clear(env_ids=selected)
            transaction.mark_irreversible("planner cancellation")
            runtime.planner_manager.cancel_matching(env_ids=selected)
            return SnapshotRestoreResult(
                accepted=True,
                robots=tuple(restored),
                env_ids=tuple(int(item) for item in selected),
                partial=compatibility.partial,
            )
    return _set_isaac_tiled_scene_snapshot(
        runtime,
        parsed,
        env_ids=env_ids,
        label_map=label_map,
        strict=strict,
    )


def clone_tiled_env_state(
    runtime: object,
    *,
    source_env_id: int,
    target_env_ids: Sequence[int] | np.ndarray,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """通过同一套 get/set 语义把 source env 状态复制到 target envs。

    该函数不走独立的快速复制路径，因而会保留普通恢复的兼容性校验、控制缓存同步、对象
    local-frame 语义及事务回滚保证。
    """

    snapshot = get_tiled_scene_snapshot(runtime, env_id=int(source_env_id))
    return set_tiled_scene_snapshot(
        runtime,
        snapshot,
        env_ids=target_env_ids,
        strict=strict,
    )


def tiled_scene_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """为 TiledSceneRuntime 构建快照恢复目标描述，不读取任一 env 的动态状态。"""

    if _is_debug_tiled_scene_runtime(runtime):
        return debug_tiled_scene_target_descriptor(runtime)
    return _isaac_tiled_scene_target_descriptor(runtime)


def _get_isaac_tiled_scene_snapshot(
    runtime: object,
    *,
    env_id: int,
) -> SimulationSnapshot:
    """从真实 Isaac TiledSceneRuntime 读取单个 env 的机器人和对象快照。

    Isaac view 的查询结果保留长度为 1 的 env 维；构造 ``RobotSnapshot`` 前显式取第零行，
    将 shape 从 ``(1, joint_count)`` 收敛为 schema 要求的 ``(joint_count,)``。
    """

    scene = runtime.scene
    selected = _single_env_id(env_id, scene.config.num_envs)
    robots: dict[str, RobotSnapshot] = {}
    for robot_index, name in enumerate(tuple(getattr(runtime, "robot_names", ()))):
        view_runtime = scene.articulation_views[name]
        command_indices = np.asarray(view_runtime.command_joint_indices, dtype=int)
        joint_names = tuple(str(item) for item in view_runtime.command_joint_names)
        positions = np.asarray(
            view_runtime.view.get_joint_positions(
                indices=selected,
                joint_indices=command_indices,
            ),
            dtype=float,
        ).reshape(1, -1)[0]
        velocities = np.asarray(
            view_runtime.view.get_joint_velocities(
                indices=selected,
                joint_indices=command_indices,
            ),
            dtype=float,
        ).reshape(1, -1)[0]
        target_positions = getattr(runtime, "target_positions", {}).get(name)
        command_targets = None
        if target_positions is not None:
            command_targets = np.asarray(target_positions, dtype=float).reshape(
                scene.config.num_envs,
                -1,
            )[int(env_id)]
        robot_summary = scene.robots.get(name)
        robots[name] = RobotSnapshot(
            robot_id=(
                robot_index if robot_summary is None else int(robot_summary.robot_id)
            ),
            label=(name if robot_summary is None else str(robot_summary.label)),
            robot_profile=(
                None if robot_summary is None else str(robot_summary.profile_name)
            ),
            asset_fingerprint=(
                None
                if robot_summary is None
                else _asset_fingerprint_from_path(robot_summary.asset_path)
            ),
            joint_names=joint_names,
            joint_positions=positions,
            joint_velocities=velocities,
            command_joint_names=joint_names,
            command_targets=command_targets,
        )
    objects = _object_snapshots_from_tiled_state(
        read_tiled_object_states(
            stage=runtime.session.stage,
            object_prim_paths=scene.object_prim_paths,
            env_origins=scene.env_origins,
            env_ids=selected,
            object_pose_views=getattr(runtime, "object_pose_views", {}),
        ),
        object_profiles=_object_profiles_by_name(runtime),
    )
    return SimulationSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="tiled_scene",
            source_env_id=int(env_id),
            step=int(getattr(runtime, "step", 0)),
            time_s=float(getattr(runtime, "time_s", 0.0)),
            coordinate_frame="env-local",
            info={"per_env": scene.config.metadata_for_env(int(env_id))},
        ),
        robots=robots,
        objects=objects,
    )


def _set_isaac_tiled_scene_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot,
    *,
    env_ids: Sequence[int] | np.ndarray,
    label_map: Mapping[str, str] | None,
    strict: bool,
) -> SnapshotRestoreResult:
    """把单环境快照事务式广播到真实 Isaac TiledSceneRuntime 的 selected envs。

    写入顺序为关节位置、关节速度、target cache、adapter cache、TCP cache 和对象状态；
    每一步写入前都登记对应补偿动作。轨迹清空与 planner 取消不可重建，因此放在所有可逆
    写入之后并标记为不可逆步骤。
    """

    scene = runtime.scene
    selected = _env_ids(env_ids, scene.config.num_envs)
    # 在写入任何 PhysX 状态前完成全部兼容性检查，避免半恢复。
    compatibility = require_snapshot_compatibility(
        snapshot,
        _isaac_tiled_scene_target_descriptor(runtime),
        label_map=label_map,
        strict=strict,
    )
    # 每个 env 的旧状态必须分别采集。单份广播 source 不能作为回滚状态，因为目标 env 的
    # 位置、速度和对象状态在写入前可能各不相同。
    originals = tuple(
        (int(env_id), _get_isaac_tiled_scene_snapshot(runtime, env_id=int(env_id)))
        for env_id in selected
    )
    expected_objects = tuple(compatibility.object_mappings)
    for env_id, original in originals:
        missing = set(expected_objects).difference(original.objects)
        if missing:
            raise RuntimeError(
                f"cannot capture rollback state for tiled env {env_id} objects: "
                f"{sorted(missing)}"
            )
    robot_rollback = {}
    for target_label in compatibility.robot_mappings:
        view_runtime = scene.articulation_views[target_label]
        command_indices = np.asarray(
            view_runtime.command_joint_indices,
            dtype=int,
        )
        adapter = runtime._command_adapter(target_label)
        robot_rollback[target_label] = _TiledRobotRollbackState(
            view_runtime=view_runtime,
            command_indices=command_indices,
            positions=np.vstack(
                [
                    original.robots[target_label].joint_positions
                    for _, original in originals
                ]
            ),
            velocities=np.vstack(
                [
                    original.robots[target_label].joint_velocities
                    for _, original in originals
                ]
            ),
            targets=runtime.target_positions[target_label][selected, :].copy(),
            adapter=adapter,
            adapter_target=_copy_optional_attribute(adapter, "last_target"),
            tcp_positions=_copy_named_rows(
                runtime,
                "tcp_positions_world",
                target_label,
                selected,
            ),
            tcp_orientations=_copy_named_rows(
                runtime,
                "tcp_orientations_wxyz",
                target_label,
                selected,
            ),
        )

    restored_robots: list[str] = []
    restored_objects: tuple[str, ...] = ()
    with mutation_transaction(
        runtime,
        operation="set_tiled_scene_snapshot",
    ) as transaction:
        for target_label, mapping in compatibility.robot_mappings.items():
            source_robot = snapshot.robots[mapping.source_label]
            rollback = robot_rollback[target_label]
            view_runtime = rollback.view_runtime
            command_indices = rollback.command_indices

            # 以每个 env 的旧矩阵为底，只替换 compatibility 选中的列。非 strict 模式下
            # 未映射列因此保持原值，而 source 一维向量通过 ``[None, :]`` 广播到所有行。
            q = rollback.positions.copy()
            dq = rollback.velocities.copy()
            q[:, mapping.joints.target_indices] = source_robot.joint_positions[
                mapping.joints.source_indices
            ][None, :]
            dq[:, mapping.joints.target_indices] = source_robot.joint_velocities[
                mapping.joints.source_indices
            ][None, :]
            transaction.add_rollback(
                f"robot {target_label} positions and TCP cache",
                lambda view_runtime=view_runtime, command_indices=command_indices, rollback=rollback, target_label=target_label,: (
                    _restore_tiled_robot_positions(
                        runtime,
                        view_runtime,
                        command_indices,
                        selected,
                        rollback.positions,
                        target_label=target_label,
                        tcp_positions=rollback.tcp_positions,
                        tcp_orientations=rollback.tcp_orientations,
                    )
                ),
            )
            view_runtime.view.set_joint_positions(
                q,
                indices=selected,
                joint_indices=command_indices,
            )
            transaction.add_rollback(
                f"robot {target_label} velocities",
                lambda view_runtime=view_runtime, command_indices=command_indices, rollback=rollback: (
                    view_runtime.view.set_joint_velocities(
                        rollback.velocities,
                        indices=selected,
                        joint_indices=command_indices,
                    )
                ),
            )
            view_runtime.view.set_joint_velocities(
                dq,
                indices=selected,
                joint_indices=command_indices,
            )
            transaction.add_rollback(
                f"robot {target_label} target cache",
                lambda target_label=target_label, rollback=rollback: (
                    _restore_target_rows(
                        runtime,
                        target_label,
                        selected,
                        rollback.targets,
                    )
                ),
            )
            _write_snapshot_command_targets(
                runtime,
                target_label,
                source_robot,
                mapping=mapping,
                env_ids=selected,
                restored_positions=q,
            )
            transaction.add_rollback(
                f"robot {target_label} command cache",
                lambda rollback=rollback: _restore_optional_attribute(
                    rollback.adapter,
                    "last_target",
                    rollback.adapter_target,
                ),
            )
            rollback.adapter.reset()
            runtime._refresh_tcp_state(target_label, env_ids=selected)
            restored_robots.append(target_label)
        if expected_objects:
            for env_id, original in originals:
                rollback_snapshot = SimulationSnapshot(
                    robots={},
                    objects={name: original.objects[name] for name in expected_objects},
                    metadata=original.metadata,
                )
                transaction.add_rollback(
                    f"tiled env {env_id} objects",
                    lambda env_id=env_id, rollback_snapshot=rollback_snapshot: (
                        _restore_snapshot_objects_to_tiled(
                            runtime,
                            rollback_snapshot,
                            env_ids=np.asarray([env_id], dtype=int),
                        )
                    ),
                )
            restored_objects = _restore_snapshot_objects_to_tiled(
                runtime,
                snapshot,
                env_ids=selected,
            )
        transaction.mark_irreversible("trajectory buffer clear")
        runtime.trajectory_buffer.clear(env_ids=selected)
        transaction.mark_irreversible("planner cancellation")
        runtime.planner_manager.cancel_matching(env_ids=selected)
    return SnapshotRestoreResult(
        accepted=True,
        robots=tuple(restored_robots),
        objects=tuple(restored_objects),
        env_ids=tuple(int(item) for item in selected),
        partial=compatibility.partial,
    )


@dataclass(frozen=True)
class _TiledRobotRollbackState:
    """一次 tiled 快照写入前捕获的单机器人完整补偿状态。

    ``positions``/``velocities``/``targets`` 的首维与 selected env 顺序一致；TCP cache
    保存 world-frame 派生值，避免回滚物理关节后仍暴露本次失败写入计算出的末端位姿。
    """

    view_runtime: object
    command_indices: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    targets: np.ndarray
    adapter: object
    adapter_target: object
    tcp_positions: np.ndarray | None
    tcp_orientations: np.ndarray | None


def _restore_snapshot_objects_to_tiled(
    runtime: object,
    snapshot: SimulationSnapshot,
    *,
    env_ids: np.ndarray,
) -> tuple[str, ...]:
    """把 ``SimulationSnapshot.objects`` 广播恢复到 TiledSceneRuntime。

    底层返回实际写入的 ``对象数 * env 数``；这里校验精确计数，禁止把少写某个 env 的
    partial mutation 当作成功。
    """

    if not snapshot.objects:
        return ()
    restore_payload = _tiled_restore_payload_from_snapshot(snapshot, env_ids=env_ids)
    restored_count = restore_tiled_object_pose_snapshot(
        stage=runtime.session.stage,
        object_prim_paths=runtime.scene.object_prim_paths,
        snapshot=restore_payload,
        env_ids=env_ids,
        env_origins=runtime.scene.env_origins,
        object_pose_views=getattr(runtime, "object_pose_views", {}),
    )
    expected_count = len(restore_payload) * int(np.asarray(env_ids).size)
    if restored_count != expected_count:
        raise RuntimeError(
            "tiled object restore was incomplete: "
            f"restored={restored_count}, expected={expected_count}"
        )
    return tuple(restore_payload.keys())


def _restore_debug_positions(
    runtime: object,
    env_ids: np.ndarray,
    positions: np.ndarray,
) -> None:
    """adapter/cache 写入失败后恢复 debug runtime 的 selected 行。"""

    runtime.current_positions[env_ids, :] = positions


def _copy_optional_attribute(owner: object, name: str) -> object:
    """复制可选缓存属性，并用 sentinel 保留“属性不存在”的语义。"""

    if not hasattr(owner, name):
        return _missing
    value = getattr(owner, name)
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else value


def _restore_optional_attribute(owner: object, name: str, value: object) -> None:
    """恢复可选缓存，但不在原本无此属性的 adapter 上凭空创建字段。"""

    if value is _missing:
        return
    copy = getattr(value, "copy", None)
    setattr(owner, name, copy() if callable(copy) else value)


def _copy_named_rows(
    runtime: object,
    attribute: str,
    name: str,
    env_ids: np.ndarray,
) -> np.ndarray | None:
    """从可选的 robot-keyed runtime cache 复制 selected env 行。"""

    cache = getattr(runtime, attribute, None)
    if not isinstance(cache, Mapping) or name not in cache:
        return None
    return np.asarray(cache[name][env_ids], dtype=float).copy()


def _restore_tiled_robot_positions(
    runtime: object,
    view_runtime: object,
    command_indices: np.ndarray,
    env_ids: np.ndarray,
    positions: np.ndarray,
    *,
    target_label: str,
    tcp_positions: np.ndarray | None,
    tcp_orientations: np.ndarray | None,
) -> None:
    """先恢复物理关节位置，再恢复与之对应的 world-frame TCP cache。"""

    view_runtime.view.set_joint_positions(
        positions,
        indices=env_ids,
        joint_indices=command_indices,
    )
    if tcp_positions is None or tcp_orientations is None:
        runtime._refresh_tcp_state(target_label, env_ids=env_ids)
        return
    getattr(runtime, "tcp_positions_world")[target_label][env_ids] = tcp_positions
    getattr(runtime, "tcp_orientations_wxyz")[target_label][env_ids] = tcp_orientations


def _restore_target_rows(
    runtime: object,
    target_label: str,
    env_ids: np.ndarray,
    values: np.ndarray,
) -> None:
    """恢复单机器人所有 selected env 的完整 command-target 行。"""

    runtime.target_positions[target_label][env_ids, :] = values


def _write_snapshot_command_targets(
    runtime: object,
    target_label: str,
    source_robot: RobotSnapshot,
    *,
    mapping: object,
    env_ids: np.ndarray,
    restored_positions: np.ndarray,
) -> None:
    """同步 command cache，优先使用快照显式保存的 targets。

    快照无 target 时以恢复后的关节位置作为 target，防止下一控制步被写入前的旧 target
    拉回。只更新兼容性映射命中的列，其余列保留目标 env 原值。
    """

    command_mapping = mapping.command_joints
    if source_robot.command_targets is not None and command_mapping is not None:
        target_indices = command_mapping.target_indices
        values = source_robot.command_targets[command_mapping.source_indices]
        rows = np.repeat(values.reshape(1, -1), env_ids.size, axis=0)
    else:
        target_indices = mapping.joints.target_indices
        rows = restored_positions[:, target_indices]
    runtime.target_positions[target_label][
        env_ids[:, None],
        target_indices,
    ] = rows


_missing = object()


def _tiled_restore_payload_from_snapshot(
    snapshot: SimulationSnapshot,
    *,
    env_ids: np.ndarray,
) -> dict[str, dict[str, object]]:
    """把单 source-env object 快照广播成 tiled object restore payload。

    根 pose 从 ``(3|4,)`` 扩展为 ``(env_count, 3|4)``；per-body pose 从
    ``(body_count, 3|4)`` 扩展为 ``(env_count, body_count, 3|4)``。``np.repeat`` 产生
    独立批次数组，避免底层原地修改共享 view。
    """

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    payload: dict[str, dict[str, object]] = {}
    for name, obj in snapshot.objects.items():
        entry: dict[str, object] = {
            "env_ids": selected.copy(),
            "positions_local": np.repeat(
                obj.positions_local.reshape(1, 3),
                selected.size,
                axis=0,
            ),
            "orientations_wxyz": np.repeat(
                obj.orientations_wxyz.reshape(1, 4),
                selected.size,
                axis=0,
            ),
        }
        for key in ("linear_velocities", "angular_velocities"):
            value = getattr(obj, key)
            if value is not None:
                entry[key] = np.repeat(
                    np.asarray(value, dtype=float).reshape(1, 3),
                    selected.size,
                    axis=0,
                )
        if obj.body_names:
            assert obj.body_positions_local is not None
            assert obj.body_orientations_wxyz is not None
            entry["body_names"] = tuple(obj.body_names)
            entry["body_positions_local"] = np.repeat(
                obj.body_positions_local.reshape(1, len(obj.body_names), 3),
                selected.size,
                axis=0,
            )
            entry["body_orientations_wxyz"] = np.repeat(
                obj.body_orientations_wxyz.reshape(1, len(obj.body_names), 4),
                selected.size,
                axis=0,
            )
            for key in ("body_linear_velocities", "body_angular_velocities"):
                value = getattr(obj, key)
                if value is not None:
                    entry[key] = np.repeat(
                        np.asarray(value, dtype=float).reshape(
                            1, len(obj.body_names), 3
                        ),
                        selected.size,
                        axis=0,
                    )
        payload[name] = entry
    return payload


def _object_snapshots_from_tiled_state(
    object_state: Mapping[str, object],
    *,
    object_profiles: Mapping[str, str | None],
) -> dict[str, ObjectSnapshot]:
    """把 tiled object state payload 的首个 env 行折叠成 ``ObjectSnapshot``。

    调用方读取时已用单元素 ``env_ids`` 选择器，因此这里取第零行不是丢弃其它环境，而是
    移除底层 tiled I/O 保留的 batch 维。
    """

    result: dict[str, ObjectSnapshot] = {}
    for name, state in object_state.items():
        if not isinstance(state, Mapping):
            continue
        positions = np.asarray(
            state.get("positions_local", ()),
            dtype=float,
        ).reshape(-1, 3)
        orientations = np.asarray(
            state.get("orientations_wxyz", ()),
            dtype=float,
        ).reshape(-1, 4)
        if positions.shape[0] < 1 or orientations.shape[0] < 1:
            continue
        body_names = tuple(str(item) for item in state.get("body_names", ()))
        kwargs: dict[str, object] = {
            key: value
            for key in ("linear_velocities", "angular_velocities")
            if (value := _first_optional_vector(state, key)) is not None
        }
        if body_names:
            body_positions = np.asarray(
                state.get("body_positions_local", ()),
                dtype=float,
            ).reshape(-1, len(body_names), 3)
            body_orientations = np.asarray(
                state.get("body_orientations_wxyz", ()),
                dtype=float,
            ).reshape(-1, len(body_names), 4)
            kwargs.update(
                {
                    "body_names": body_names,
                    "body_positions_local": body_positions[0],
                    "body_orientations_wxyz": body_orientations[0],
                }
            )
            kwargs.update(
                {
                    key: value
                    for key in (
                        "body_linear_velocities",
                        "body_angular_velocities",
                    )
                    if (
                        value := _first_optional_body_vectors(
                            state,
                            key,
                            body_count=len(body_names),
                        )
                    )
                    is not None
                }
            )
        result[str(name)] = ObjectSnapshot(
            name=str(name),
            object_profile=object_profiles.get(str(name)),
            positions_local=positions[0],
            orientations_wxyz=orientations[0],
            **kwargs,
        )
    return result


def _first_optional_vector(
    state: Mapping[str, object],
    key: str,
) -> np.ndarray | None:
    """读取 tiled state 中可选的第一行三维向量。"""

    value = state.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    return array.reshape(-1, 3)[0]


def _first_optional_body_vectors(
    state: Mapping[str, object],
    key: str,
    *,
    body_count: int,
) -> np.ndarray | None:
    """读取 tiled state 中可选的第一行 per-body 三维向量。"""

    value = state.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    return array.reshape(-1, body_count, 3)[0]


def _isaac_tiled_scene_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """根据真实 Isaac TiledSceneRuntime 构建兼容性检查 target descriptor。"""

    scene = runtime.scene
    robots = {}
    for name in tuple(getattr(runtime, "robot_names", ())):
        view_runtime = scene.articulation_views[name]
        robot_summary = scene.robots.get(name)
        joint_names = tuple(str(item) for item in view_runtime.command_joint_names)
        robots[name] = RobotTargetDescriptor(
            label=name,
            robot_profile=(
                None if robot_summary is None else str(robot_summary.profile_name)
            ),
            asset_fingerprint=(
                None
                if robot_summary is None
                else _asset_fingerprint_from_path(robot_summary.asset_path)
            ),
            joint_names=joint_names,
            command_joint_names=joint_names,
        )
    object_profiles = _object_profiles_by_name(runtime)
    objects = {}
    for name in scene.object_prim_paths:
        view = getattr(runtime, "object_pose_views", {}).get(str(name))
        body_names = ()
        if hasattr(view, "body_names"):
            body_names = tuple(str(item) for item in getattr(view, "body_names"))
        objects[str(name)] = ObjectTargetDescriptor(
            name=str(name),
            object_profile=object_profiles.get(str(name)),
            body_names=body_names,
        )
    return SnapshotTargetDescriptor(
        runtime_kind="tiled_scene",
        robots=robots,
        objects=objects,
    )


def _snapshot_from_input(
    snapshot: SimulationSnapshot | Mapping[str, object],
) -> SimulationSnapshot:
    """接受已解析 snapshot 或 canonical JSON mapping，并统一返回 schema object。"""

    if isinstance(snapshot, SimulationSnapshot):
        return snapshot
    if isinstance(snapshot, Mapping):
        return SimulationSnapshot.from_mapping(snapshot)
    raise ValueError("snapshot must be a SimulationSnapshot or JSON object")


def _is_debug_tiled_scene_runtime(runtime: object) -> bool:
    """识别不依赖 Isaac 的 debug Tiled Scene runtime adapter shape。"""

    return hasattr(runtime, "current_positions") and hasattr(runtime, "adapter")


def _single_env_id(env_id: int, num_envs: int) -> np.ndarray:
    """校验单个 source env ID，并包装为内部一维 selector。"""

    env = int(env_id)
    if env < 0 or env >= int(num_envs):
        raise ValueError("env_id is out of range")
    return np.asarray([env], dtype=int)


def _env_ids(env_ids: Sequence[int] | np.ndarray, num_envs: int) -> np.ndarray:
    """校验非空 target env IDs 的范围并复制为一维 int array。"""

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    if selected.size < 1:
        raise ValueError("env_ids cannot be empty")
    if np.unique(selected).size != selected.size:
        raise ValueError("env_ids cannot contain duplicates")
    if np.any(selected < 0) or np.any(selected >= int(num_envs)):
        raise ValueError("env_ids contains out-of-range env id")
    return selected.astype(int, copy=True)


__all__ = [
    "clone_tiled_env_state",
    "get_tiled_scene_snapshot",
    "set_tiled_scene_snapshot",
    "tiled_scene_target_descriptor",
]
