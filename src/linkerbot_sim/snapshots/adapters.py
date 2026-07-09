"""现有仿真 runtime 的 snapshot adapter。

schema/compatibility 层只描述数据和匹配关系；本模块负责访问真实 runtime，把
single、dual、tiled 的状态读成统一 ``SimulationSnapshot``，或把该 snapshot 写回
指定目标。所有 public 函数都尽量以普通 Python/JSON-compatible 对象作为边界，
便于交互协议层直接复用。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from linkerbot_sim.snapshots.compatibility import (
    ObjectTargetDescriptor,
    RobotTargetDescriptor,
    SnapshotTargetDescriptor,
    require_snapshot_compatibility,
)
from linkerbot_sim.snapshots.schema import (
    ObjectSnapshot,
    RobotSnapshot,
    SimulationSnapshot,
    SnapshotMetadata,
    SnapshotRestoreResult,
)


def get_tiled_snapshot(runtime: object, *, env_id: int) -> SimulationSnapshot:
    """从 tiled runtime 读取单个 env 的逻辑快照。

    tiled 场景中有多个 env，snapshot 的语义固定为“某一个 env 的状态”。要复制到多
    个 env 时，调用方应先读一个 source env，再用 ``set_tiled_snapshot`` 写到目标集合。
    """

    if _is_debug_tiled_runtime(runtime):
        return _get_debug_tiled_snapshot(runtime, env_id=env_id)
    return _get_isaac_tiled_snapshot(runtime, env_id=env_id)


def get_single_robot_snapshot(runtime: object) -> SimulationSnapshot:
    """从 single-arm runtime 读取当前 scene 的逻辑快照。

    single runtime 没有 env 维度，因此 role 固定为 ``single``；后续如要恢复到 dual
    或 tiled，可通过 ``robot_map`` 映射到目标 role/机器人名。
    """

    execution = runtime.execution
    robot = _robot_snapshot_from_execution(
        role="single",
        execution=execution,
        robot_profile=None,
        asset_fingerprint=_imported_asset_fingerprint(
            getattr(runtime, "imported_robot", None)
        ),
    )
    return SimulationSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="single",
            step=None,
            time_s=None,
            coordinate_frame="scene-local",
        ),
        robots={"single": robot},
        objects=_runtime_object_snapshots(
            stage=getattr(runtime.session, "stage", None),
            handles=getattr(runtime, "object_handles", ()),
        ),
    )


def get_dual_robot_snapshot(runtime: object) -> SimulationSnapshot:
    """从 dual-arm runtime 读取当前左右臂的逻辑快照。

    dual snapshot 同时包含 ``left`` 和 ``right`` 两个 role。恢复到 single 时必须通过
    ``robot_map`` 指定只取哪一侧，避免把双臂状态误写到单臂 runtime。
    """

    robots = {}
    for side in ("left", "right"):
        side_runtime = runtime.execution.side(side)
        robots[side] = _robot_snapshot_from_execution(
            role=side,
            execution=side_runtime,
            robot_profile=None,
            asset_fingerprint=_imported_asset_fingerprint(
                getattr(runtime, "imported", {}).get(side)
            ),
        )
    return SimulationSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="dual",
            coordinate_frame="scene-local",
        ),
        robots=robots,
        objects=_runtime_object_snapshots(
            stage=getattr(runtime.session, "stage", None),
            handles=getattr(runtime, "object_handles", ()),
        ),
    )


def get_snapshot(runtime: object, *, env_id: int | None = None) -> SimulationSnapshot:
    """根据 runtime 形状分发到对应 snapshot reader。"""

    if _looks_like_tiled_runtime(runtime):
        if env_id is None:
            raise ValueError("env_id is required for tiled snapshot reads")
        return get_tiled_snapshot(runtime, env_id=int(env_id))
    if _looks_like_dual_runtime(runtime):
        return get_dual_robot_snapshot(runtime)
    if _looks_like_single_runtime(runtime):
        return get_single_robot_snapshot(runtime)
    raise ValueError("unsupported runtime type for get_snapshot")


def set_tiled_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot | Mapping[str, object],
    *,
    env_ids: Sequence[int] | np.ndarray,
    robot_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """把逻辑快照恢复到 tiled runtime 的一个或多个 env。

    同一份 snapshot 会被广播到 ``env_ids`` 中的所有目标 env；对象位姿使用 env-local
    数据恢复，因此即使目标 env 有不同 world origin，也会保持与 source env 相同的局部
    场景布局。
    """

    parsed = _snapshot_from_input(snapshot)
    if _is_debug_tiled_runtime(runtime):
        return _set_debug_tiled_snapshot(
            runtime,
            parsed,
            env_ids=env_ids,
            robot_map=robot_map,
            strict=strict,
        )
    return _set_isaac_tiled_snapshot(
        runtime,
        parsed,
        env_ids=env_ids,
        robot_map=robot_map,
        strict=strict,
    )


def set_single_robot_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot | Mapping[str, object],
    *,
    robot_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """把逻辑快照恢复到 single-arm runtime。

    恢复前先做兼容性检查，确保目标单臂的 command joints 能按名字从 snapshot 中找到；
    通过检查后只写 command joints 和 runtime objects，不碰 unrelated runtime 配置。
    """

    parsed = _snapshot_from_input(snapshot)
    compatibility = require_snapshot_compatibility(
        parsed,
        _single_target_descriptor(runtime),
        robot_map=robot_map,
        strict=strict,
    )
    restored_robots: list[str] = []
    mapping = compatibility.robot_mappings.get("single")
    if mapping is not None:
        # single runtime 只有一个 execution；mapping.source_role 可能来自 snapshot 的
        # ``single``，也可能由 robot_map 指向 dual 的 ``left``/``right``。
        _restore_robot_snapshot_to_execution(
            runtime.execution,
            parsed.robots[mapping.source_role],
            mapping=mapping,
        )
        restored_robots.append("single")
    restored_objects = _restore_runtime_objects(
        runtime,
        parsed,
        compatibility=compatibility,
    )
    _reset_execution_observers(getattr(runtime, "execution", None))
    return SnapshotRestoreResult(
        accepted=True,
        robots=tuple(restored_robots),
        objects=restored_objects,
        partial=compatibility.partial,
    )


def set_dual_robot_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot | Mapping[str, object],
    *,
    robot_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """把逻辑快照恢复到 dual-arm runtime。

    dual runtime 对每个目标 role 分别取 side execution，支持 single snapshot 通过
    ``robot_map`` 写到某一侧，也支持完整 dual snapshot 恢复左右两侧。
    """

    parsed = _snapshot_from_input(snapshot)
    compatibility = require_snapshot_compatibility(
        parsed,
        _dual_target_descriptor(runtime),
        robot_map=robot_map,
        strict=strict,
    )
    restored_robots: list[str] = []
    for target_role, mapping in compatibility.robot_mappings.items():
        # target_role 已经由 compatibility 保证存在于 dual execution 中。
        side_runtime = runtime.execution.side(target_role)
        _restore_robot_snapshot_to_execution(
            side_runtime,
            parsed.robots[mapping.source_role],
            mapping=mapping,
        )
        restored_robots.append(target_role)
    restored_objects = _restore_runtime_objects(
        runtime,
        parsed,
        compatibility=compatibility,
    )
    _reset_execution_observers(getattr(runtime, "execution", None))
    return SnapshotRestoreResult(
        accepted=True,
        robots=tuple(restored_robots),
        objects=restored_objects,
        partial=compatibility.partial,
    )


def set_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot | Mapping[str, object],
    *,
    env_ids: Sequence[int] | np.ndarray | None = None,
    robot_map: Mapping[str, str] | None = None,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """根据 runtime 形状分发到对应 snapshot writer。"""

    if _looks_like_tiled_runtime(runtime):
        if env_ids is None:
            raise ValueError("env_ids is required for tiled snapshot restores")
        return set_tiled_snapshot(
            runtime,
            snapshot,
            env_ids=env_ids,
            robot_map=robot_map,
            strict=strict,
        )
    if _looks_like_dual_runtime(runtime):
        return set_dual_robot_snapshot(
            runtime,
            snapshot,
            robot_map=robot_map,
            strict=strict,
        )
    if _looks_like_single_runtime(runtime):
        return set_single_robot_snapshot(
            runtime,
            snapshot,
            robot_map=robot_map,
            strict=strict,
        )
    raise ValueError("unsupported runtime type for set_snapshot")


def clone_tiled_env_state(
    runtime: object,
    *,
    source_env_id: int,
    target_env_ids: Sequence[int] | np.ndarray,
    strict: bool = True,
) -> SnapshotRestoreResult:
    """把一个 tiled source env 的状态克隆到一个或多个 target env。

    clone 本质上是 ``get_tiled_snapshot`` + ``set_tiled_snapshot`` 的组合，保证行为
    与显式 get/set 完全一致，也让兼容性检查、对象恢复和缓存清理只维护一套逻辑。
    """

    snapshot = get_tiled_snapshot(runtime, env_id=int(source_env_id))
    return set_tiled_snapshot(
        runtime,
        snapshot,
        env_ids=target_env_ids,
        strict=strict,
    )


def tiled_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """为 tiled runtime 构建 snapshot 恢复目标描述。"""

    if _is_debug_tiled_runtime(runtime):
        return _debug_tiled_target_descriptor(runtime)
    return _isaac_tiled_target_descriptor(runtime)


def single_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """为 single-arm runtime 构建 snapshot 恢复目标描述。"""

    return _single_target_descriptor(runtime)


def dual_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """为 dual-arm runtime 构建 snapshot 恢复目标描述。"""

    return _dual_target_descriptor(runtime)


def _get_isaac_tiled_snapshot(runtime: object, *, env_id: int) -> SimulationSnapshot:
    """从真实 Isaac tiled runtime 读取单个 env 的机器人和对象快照。"""

    scene = runtime.scene
    selected = _single_env_id(env_id, scene.config.num_envs)
    robots: dict[str, RobotSnapshot] = {}
    for name in tuple(getattr(runtime, "robot_names", ())):
        # tiled articulation view 内可能包含完整 DOF；交互控制只操作 command_joint_indices，
        # 所以 snapshot 也只保存这部分，避免把非控制关节写回到不兼容 runtime。
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
            # command target 是控制器下一帧会继续保持/插值的目标；保存它可以让恢复后的
            # idle/trajectory 不会立刻把机器人拉回旧目标。
            command_targets = np.asarray(target_positions, dtype=float).reshape(
                scene.config.num_envs,
                -1,
            )[int(env_id)]
        robot_summary = scene.robots.get(name)
        robots[name] = RobotSnapshot(
            role=name,
            robot_profile=(
                None if robot_summary is None else str(robot_summary.profile_name)
            ),
            asset_fingerprint=(
                None if robot_summary is None else str(robot_summary.asset_path)
            ),
            joint_names=joint_names,
            joint_positions=positions,
            joint_velocities=velocities,
            command_joint_names=joint_names,
            command_targets=command_targets,
        )
    objects = _object_snapshots_from_tiled_state(
        _read_tiled_object_states_lazy()(
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
            source_runtime="tiled",
            source_env_id=int(env_id),
            step=int(getattr(runtime, "step", 0)),
            time_s=float(getattr(runtime, "time_s", 0.0)),
            coordinate_frame="env-local",
        ),
        robots=robots,
        objects=objects,
    )


def _set_isaac_tiled_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot,
    *,
    env_ids: Sequence[int] | np.ndarray,
    robot_map: Mapping[str, str] | None,
    strict: bool,
) -> SnapshotRestoreResult:
    """把快照写回真实 Isaac tiled runtime 的 selected envs。"""

    scene = runtime.scene
    selected = _env_ids(env_ids, scene.config.num_envs)
    # 先解析名字映射，再写任何 PhysX 状态；一旦不兼容就整体拒绝，避免只恢复一半机器人。
    compatibility = require_snapshot_compatibility(
        snapshot,
        _isaac_tiled_target_descriptor(runtime),
        robot_map=robot_map,
        strict=strict,
    )
    restored_robots: list[str] = []
    for target_role, mapping in compatibility.robot_mappings.items():
        source_robot = snapshot.robots[mapping.source_role]
        view_runtime = scene.articulation_views[target_role]
        command_indices = np.asarray(view_runtime.command_joint_indices, dtype=int)
        # 先读目标 env 的当前数组，再只覆盖映射到的列。这样非 strict 模式下未匹配的
        # command joints 会保留目标 runtime 原值。
        q = np.asarray(
            view_runtime.view.get_joint_positions(
                indices=selected,
                joint_indices=command_indices,
            ),
            dtype=float,
        ).reshape(selected.size, -1)
        dq = np.asarray(
            view_runtime.view.get_joint_velocities(
                indices=selected,
                joint_indices=command_indices,
            ),
            dtype=float,
        ).reshape(selected.size, -1)
        q[:, mapping.joints.target_indices] = source_robot.joint_positions[
            mapping.joints.source_indices
        ][None, :]
        dq[:, mapping.joints.target_indices] = source_robot.joint_velocities[
            mapping.joints.source_indices
        ][None, :]
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
        # target_positions 是 tiled command adapter 的控制目标缓存；如果只写 PhysX joint
        # state 而不更新它，下一次 idle/action 会把机器人又推回旧 target。
        runtime.target_positions[target_role][selected[:, None], mapping.joints.target_indices] = q[
            :, mapping.joints.target_indices
        ]
        # IK/插值 adapter 内部缓存基于旧 command，恢复后必须重置并刷新 TCP 缓存。
        runtime._command_adapter(target_role).reset()
        runtime._refresh_tcp_state(target_role, env_ids=selected)
        restored_robots.append(target_role)
    restored_objects = _restore_snapshot_objects_to_tiled(
        runtime,
        snapshot,
        env_ids=selected,
    )
    runtime.trajectory_buffer.clear(env_ids=selected)
    runtime.planner_manager.cancel_matching(env_ids=selected)
    return SnapshotRestoreResult(
        accepted=True,
        robots=tuple(restored_robots),
        objects=tuple(restored_objects),
        env_ids=tuple(int(item) for item in selected),
        partial=compatibility.partial,
    )


def _get_debug_tiled_snapshot(runtime: object, *, env_id: int) -> SimulationSnapshot:
    """从 debug tiled runtime 读取单个 env 的简化快照。"""

    # debug tiled runtime 是测试/无 Isaac 场景使用的轻量实现；它也走同一套 schema，
    # 这样协议测试可以覆盖 tiled snapshot 的主要行为。
    selected = int(_single_env_id(env_id, runtime.config.num_envs)[0])
    joint_names = tuple(f"joint_{index}" for index in range(runtime.adapter.command_dim))
    return SimulationSnapshot(
        metadata=SnapshotMetadata(
            source_runtime="tiled_debug",
            source_env_id=selected,
            step=int(runtime.step),
            time_s=float(runtime.time_s),
            coordinate_frame="env-local",
        ),
        robots={
            "debug": RobotSnapshot(
                role="debug",
                joint_names=joint_names,
                joint_positions=np.asarray(runtime.current_positions[selected], dtype=float),
                joint_velocities=np.zeros(runtime.adapter.command_dim, dtype=float),
                command_joint_names=joint_names,
                command_targets=np.asarray(runtime.current_positions[selected], dtype=float),
            )
        },
        objects={},
    )


def _set_debug_tiled_snapshot(
    runtime: object,
    snapshot: SimulationSnapshot,
    *,
    env_ids: Sequence[int] | np.ndarray,
    robot_map: Mapping[str, str] | None,
    strict: bool,
) -> SnapshotRestoreResult:
    """把快照写回 debug tiled runtime 的 selected envs。"""

    selected = _env_ids(env_ids, runtime.config.num_envs)
    # fake/debug runtime 没有对象和真实 PhysX，但仍使用兼容性检查验证关节名映射。
    compatibility = require_snapshot_compatibility(
        snapshot,
        _debug_tiled_target_descriptor(runtime),
        robot_map=robot_map,
        strict=strict,
    )
    restored = []
    for target_role, mapping in compatibility.robot_mappings.items():
        source_robot = snapshot.robots[mapping.source_role]
        runtime.current_positions[selected[:, None], mapping.joints.target_indices] = (
            source_robot.joint_positions[mapping.joints.source_indices][None, :]
        )
        restored.append(target_role)
    runtime.adapter.reset()
    runtime.trajectory_buffer.clear(env_ids=selected)
    runtime.planner_manager.cancel_matching(env_ids=selected)
    return SnapshotRestoreResult(
        accepted=True,
        robots=tuple(restored),
        env_ids=tuple(int(item) for item in selected),
        partial=compatibility.partial,
    )


def _restore_snapshot_objects_to_tiled(
    runtime: object,
    snapshot: SimulationSnapshot,
    *,
    env_ids: np.ndarray,
) -> tuple[str, ...]:
    """把 ``SimulationSnapshot.objects`` 恢复到 tiled runtime 中。"""

    if not snapshot.objects:
        return ()
    # object_states 模块的 restore API 仍沿用 tiled get_state 的 batched payload；
    # 这里把 runtime-neutral ObjectSnapshot 广播成该 payload，统一处理 rigid view 和
    # dynamic-chain view 的细节。
    restore_payload = _tiled_restore_payload_from_snapshot(snapshot, env_ids=env_ids)
    restored_count = _restore_tiled_object_pose_snapshot_lazy()(
        stage=runtime.session.stage,
        object_prim_paths=runtime.scene.object_prim_paths,
        snapshot=restore_payload,
        env_ids=env_ids,
        env_origins=runtime.scene.env_origins,
        object_pose_views=getattr(runtime, "object_pose_views", {}),
    )
    if restored_count <= 0:
        return ()
    return tuple(restore_payload.keys())


def _tiled_restore_payload_from_snapshot(
    snapshot: SimulationSnapshot,
    *,
    env_ids: np.ndarray,
) -> dict[str, dict[str, object]]:
    """把 runtime-neutral object snapshot 转成 tiled object restore payload。"""

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    payload: dict[str, dict[str, object]] = {}
    for name, obj in snapshot.objects.items():
        # ObjectSnapshot 保存的是单个 source env 的局部位姿；恢复到多个 env 时按行复制，
        # 真正转 world 坐标由 object_states 根据目标 env origin 完成。
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
        if obj.body_names:
            # dynamic-chain 必须携带每个 child body 的 local pose；否则只恢复 root 会让链条
            # 在 PhysX 中仍保持旧形状。
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
        payload[name] = entry
    return payload


def _object_snapshots_from_tiled_state(
    object_state: Mapping[str, object],
    *,
    object_profiles: Mapping[str, str | None],
) -> dict[str, ObjectSnapshot]:
    """把 tiled object state payload 折叠成单 env 的 ``ObjectSnapshot`` 集合。"""

    result: dict[str, ObjectSnapshot] = {}
    for name, state in object_state.items():
        if not isinstance(state, Mapping):
            continue
        positions = np.asarray(state.get("positions_local", ()), dtype=float).reshape(-1, 3)
        orientations = np.asarray(
            state.get("orientations_wxyz", ()), dtype=float
        ).reshape(-1, 4)
        if positions.shape[0] < 1 or orientations.shape[0] < 1:
            continue
        body_names = tuple(str(item) for item in state.get("body_names", ()))
        kwargs: dict[str, object] = {}
        if body_names:
            # tiled object reader 返回 batched env rows；get_snapshot 只请求单个 env，
            # 因此这里取第 0 行作为 ObjectSnapshot 的单实例状态。
            body_positions = np.asarray(
                state.get("body_positions_local", ()), dtype=float
            ).reshape(-1, len(body_names), 3)
            body_orientations = np.asarray(
                state.get("body_orientations_wxyz", ()), dtype=float
            ).reshape(-1, len(body_names), 4)
            kwargs.update(
                {
                    "body_names": body_names,
                    "body_positions_local": body_positions[0],
                    "body_orientations_wxyz": body_orientations[0],
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


def _isaac_tiled_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """根据真实 Isaac tiled runtime 构建兼容性检查用 target descriptor。"""

    scene = runtime.scene
    robots = {}
    for name in tuple(getattr(runtime, "robot_names", ())):
        # descriptor 只放恢复需要的信息：目标机器人名、profile/asset、command joint 名字。
        view_runtime = scene.articulation_views[name]
        robot_summary = scene.robots.get(name)
        robots[name] = RobotTargetDescriptor(
            role=name,
            robot_profile=(
                None if robot_summary is None else str(robot_summary.profile_name)
            ),
            asset_fingerprint=(
                None if robot_summary is None else str(robot_summary.asset_path)
            ),
            joint_names=tuple(str(item) for item in view_runtime.command_joint_names),
            command_joint_names=tuple(str(item) for item in view_runtime.command_joint_names),
        )
    object_profiles = _object_profiles_by_name(runtime)
    objects = {}
    for name in scene.object_prim_paths:
        view = getattr(runtime, "object_pose_views", {}).get(str(name))
        body_names = ()
        if hasattr(view, "body_names"):
            # dynamic-chain wrapper 暴露 body_names，普通 rigid view 没有该字段。
            body_names = tuple(str(item) for item in getattr(view, "body_names"))
        objects[str(name)] = ObjectTargetDescriptor(
            name=str(name),
            object_profile=object_profiles.get(str(name)),
            body_names=body_names,
        )
    return SnapshotTargetDescriptor(
        runtime_kind="tiled",
        robots=robots,
        objects=objects,
    )


def _debug_tiled_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """为 debug tiled runtime 构建最小 target descriptor。"""

    joint_names = tuple(f"joint_{index}" for index in range(runtime.adapter.command_dim))
    return SnapshotTargetDescriptor(
        runtime_kind="tiled_debug",
        robots={
            "debug": RobotTargetDescriptor(
                role="debug",
                joint_names=joint_names,
                command_joint_names=joint_names,
            )
        },
        objects={},
    )


def _single_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """为 single-arm runtime 构建 target descriptor。"""

    execution = runtime.execution
    return SnapshotTargetDescriptor(
        runtime_kind="single",
        robots={
            "single": _robot_target_from_execution(
                role="single",
                execution=execution,
                robot_profile=None,
                asset_fingerprint=_imported_asset_fingerprint(
                    getattr(runtime, "imported_robot", None)
                ),
            )
        },
        objects=_runtime_object_targets(getattr(runtime, "object_handles", ())),
    )


def _dual_target_descriptor(runtime: object) -> SnapshotTargetDescriptor:
    """为 dual-arm runtime 构建包含左右侧的 target descriptor。"""

    robots = {}
    for side in ("left", "right"):
        robots[side] = _robot_target_from_execution(
            role=side,
            execution=runtime.execution.side(side),
            robot_profile=None,
            asset_fingerprint=_imported_asset_fingerprint(
                getattr(runtime, "imported", {}).get(side)
            ),
        )
    return SnapshotTargetDescriptor(
        runtime_kind="dual",
        robots=robots,
        objects=_runtime_object_targets(getattr(runtime, "object_handles", ())),
    )


def _robot_snapshot_from_execution(
    *,
    role: str,
    execution: object,
    robot_profile: str | None,
    asset_fingerprint: str | None,
) -> RobotSnapshot:
    """从 single/dual 的 execution 对象读取 command-joint 快照。"""

    articulation = execution.articulation
    controller = execution.joint_controller
    command_indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    joint_names = _command_joint_names(articulation, controller)
    # single/dual articulation API 返回全 DOF；snapshot 只截取 controller 管理的 command
    # joints，保持与 tiled adapter 的语义一致。
    positions = np.asarray(articulation.get_joint_positions(), dtype=float).reshape(-1)
    velocities = np.asarray(articulation.get_joint_velocities(), dtype=float).reshape(-1)
    return RobotSnapshot(
        role=role,
        robot_profile=robot_profile,
        asset_fingerprint=asset_fingerprint,
        joint_names=joint_names,
        joint_positions=positions[command_indices],
        joint_velocities=velocities[command_indices],
        command_joint_names=joint_names,
        command_targets=positions[command_indices],
    )


def _robot_target_from_execution(
    *,
    role: str,
    execution: object,
    robot_profile: str | None,
    asset_fingerprint: str | None,
) -> RobotTargetDescriptor:
    """从 execution 对象提取目标机器人关节名字和资产指纹。"""

    return RobotTargetDescriptor(
        role=role,
        robot_profile=robot_profile,
        asset_fingerprint=asset_fingerprint,
        joint_names=_command_joint_names(execution.articulation, execution.joint_controller),
        command_joint_names=_command_joint_names(
            execution.articulation,
            execution.joint_controller,
        ),
    )


def _restore_robot_snapshot_to_execution(
    execution: object,
    source_robot: RobotSnapshot,
    *,
    mapping: object,
) -> None:
    """把一个 ``RobotSnapshot`` 写回 single/dual execution 的 articulation。"""

    articulation = execution.articulation
    controller = execution.joint_controller
    command_indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    q = np.asarray(articulation.get_joint_positions(), dtype=float).reshape(-1)
    dq = np.asarray(articulation.get_joint_velocities(), dtype=float).reshape(-1)
    # mapping.joints.target_indices 是 command-joint 空间的索引；先映射到真实 articulation
    # DOF index，再写入全量 q/dq，避免覆盖非 command DOF。
    target_command_indices = command_indices[mapping.joints.target_indices]
    q[target_command_indices] = source_robot.joint_positions[mapping.joints.source_indices]
    dq[target_command_indices] = source_robot.joint_velocities[mapping.joints.source_indices]
    articulation.set_joint_positions(q)
    articulation.set_joint_velocities(dq)
    if hasattr(controller, "last_commanded_efforts"):
        # 关节位置被外部强制改写后，上一帧 effort 缓存不再可信；置 NaN 让后续 observer/
        # telemetry 不把旧控制输出误认为当前命令。
        controller.last_commanded_efforts = np.full(q.size, np.nan, dtype=float)


def _command_joint_names(articulation: object, controller: object) -> tuple[str, ...]:
    """按 controller 配置返回 command joint 名字。

    新 controller 会直接提供 ``command_joint_names``；旧对象只提供索引时，则从
    articulation.dof_names 中反查。
    """

    names = getattr(controller, "command_joint_names", None)
    if names is not None:
        return tuple(str(name) for name in names)
    dof_names = tuple(str(name) for name in getattr(articulation, "dof_names", ()))
    command_indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    return tuple(dof_names[int(index)] for index in command_indices)


def _runtime_object_snapshots(
    *,
    stage: object | None,
    handles: Sequence[object],
) -> dict[str, ObjectSnapshot]:
    """读取 single/dual runtime object handles 对应的 scene-local object 快照。"""

    if stage is None:
        return {}
    result: dict[str, ObjectSnapshot] = {}
    for handle in handles:
        name = _runtime_object_name(handle)
        prim_path = _runtime_object_prim_path(handle)
        if not name or prim_path is None:
            continue
        pose = _read_prim_world_pose_lazy()(stage, prim_path)
        if pose is None:
            continue
        position, orientation = pose
        body_names, body_paths = _runtime_object_body_paths(handle)
        kwargs: dict[str, object] = {}
        if body_names:
            # single/dual 非 tiled 场景没有 env origin 的概念，因此对象 root 和 child body
            # 都按 scene-local pose 保存；恢复时同样写回 local/world 等价的 prim pose。
            body_positions = []
            body_orientations = []
            for body_path in body_paths:
                body_pose = _read_prim_world_pose_lazy()(stage, body_path)
                if body_pose is None:
                    break
                body_position, body_orientation = body_pose
                body_positions.append(body_position)
                body_orientations.append(body_orientation)
            if len(body_positions) == len(body_names):
                kwargs.update(
                    {
                        "body_names": body_names,
                        "body_positions_local": np.vstack(body_positions),
                        "body_orientations_wxyz": np.vstack(body_orientations),
                    }
                )
        result[name] = ObjectSnapshot(
            name=name,
            object_profile=_runtime_object_profile(handle),
            positions_local=position,
            orientations_wxyz=orientation,
            **kwargs,
        )
    return result


def _runtime_object_targets(handles: Sequence[object]) -> dict[str, ObjectTargetDescriptor]:
    """根据 single/dual object handles 构建 object target descriptors。"""

    result = {}
    for handle in handles:
        name = _runtime_object_name(handle)
        if not name:
            continue
        body_names, _ = _runtime_object_body_paths(handle)
        result[name] = ObjectTargetDescriptor(
            name=name,
            object_profile=_runtime_object_profile(handle),
            body_names=body_names,
        )
    return result


def _restore_runtime_objects(
    runtime: object,
    snapshot: SimulationSnapshot,
    *,
    compatibility: object,
) -> tuple[str, ...]:
    """把 snapshot objects 写回 single/dual runtime 中的 USD prim。"""

    stage = getattr(getattr(runtime, "session", None), "stage", None)
    if stage is None or not snapshot.objects:
        return ()
    handles_by_name = {
        _runtime_object_name(handle): handle
        for handle in getattr(runtime, "object_handles", ())
        if _runtime_object_name(handle)
    }
    restored: list[str] = []
    for target_name, mapping in compatibility.object_mappings.items():
        obj = snapshot.objects[mapping.source_name]
        handle = handles_by_name.get(target_name)
        if handle is None:
            continue
        prim_path = _runtime_object_prim_path(handle)
        if prim_path is not None and _apply_prim_local_pose_and_zero_velocity_lazy()(
            stage,
            prim_path,
            obj.positions_local,
            obj.orientations_wxyz,
        ):
            restored.append(target_name)
        if obj.body_names and mapping.bodies is not None:
            # 多刚体对象按 body name 映射，允许目标 body 顺序与 snapshot 不一致。
            body_names, body_paths = _runtime_object_body_paths(handle)
            body_path_by_name = dict(zip(body_names, body_paths, strict=True))
            assert obj.body_positions_local is not None
            assert obj.body_orientations_wxyz is not None
            for source_index, body_name in zip(
                mapping.bodies.source_indices,
                mapping.bodies.names,
                strict=True,
            ):
                body_path = body_path_by_name.get(body_name)
                if body_path is None:
                    continue
                _apply_prim_local_pose_and_zero_velocity_lazy()(
                    stage,
                    body_path,
                    obj.body_positions_local[int(source_index)],
                    obj.body_orientations_wxyz[int(source_index)],
                )
    return tuple(restored)


def _reset_execution_observers(execution: object | None) -> None:
    """恢复状态后重置 execution 上可能缓存旧采样的 observer。"""

    if execution is None:
        return
    for name in ("state_observer", "camera_observer"):
        observer = getattr(execution, name, None)
        reset = getattr(observer, "reset", None)
        if callable(reset):
            # snapshot restore 是一次状态跳变；清理 observer 缓存可以避免状态流继续沿用
            # 恢复前的 joint/object 采样。
            reset()


def _runtime_object_name(handle: object) -> str:
    """从 runtime object handle 中提取对外使用的稳定对象名。"""

    runtime_handle = getattr(handle, "runtime_handle", None)
    if runtime_handle is not None:
        return str(runtime_handle)
    name = getattr(handle, "name", None)
    return "" if name is None else str(name)


def _runtime_object_profile(handle: object) -> str | None:
    """读取 object profile 名称；缺失时返回 ``None`` 以允许旧 handle。"""

    config = getattr(handle, "config", None)
    if hasattr(config, "object_profile"):
        return str(getattr(config, "object_profile"))
    return None


def _runtime_object_prim_path(handle: object) -> str | None:
    """从 object handle 的 model/config 中解析 root prim path。"""

    for source in (getattr(handle, "model", None), getattr(handle, "config", None)):
        if source is None:
            continue
        prim_path = getattr(source, "prim_path", None)
        if prim_path is not None:
            return str(prim_path)
        if isinstance(source, Mapping):
            root = source.get("root")
            if root is not None and hasattr(root, "GetPath"):
                return str(root.GetPath())
            prim_path = source.get("prim_path")
            if prim_path is not None:
                return str(prim_path)
    return None


def _runtime_object_body_paths(handle: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """读取 dynamic/multi-body object 的 child body 名字和 prim paths。"""

    bodies = []
    model = getattr(handle, "model", None)
    if isinstance(model, Mapping):
        bodies = list(model.get("bodies", ()) or ())
    else:
        bodies = list(getattr(model, "bodies", ()) or ())
    names = []
    paths = []
    for body in bodies:
        path_getter = getattr(body, "GetPath", None)
        body_path = str(path_getter() if callable(path_getter) else body)
        name_getter = getattr(body, "GetName", None)
        body_name = str(name_getter() if callable(name_getter) else body_path.rsplit("/", 1)[-1])
        names.append(body_name)
        paths.append(body_path)
    return tuple(names), tuple(paths)


def _imported_asset_fingerprint(imported: object | None) -> str | None:
    """把导入资产路径作为轻量 asset fingerprint。"""

    if imported is None:
        return None
    asset_path = getattr(imported, "asset_path", None)
    if asset_path is None:
        return None
    return str(asset_path)


def _object_profiles_by_name(runtime: object) -> dict[str, str | None]:
    """按 object name 收集 tiled runtime 中的 object profile。"""

    result: dict[str, str | None] = {}
    for handle in getattr(runtime.scene, "object_handles", ()) or ():
        name = str(getattr(handle, "name", ""))
        profile = None
        config = getattr(handle, "config", None)
        if hasattr(config, "object_profile"):
            profile = str(getattr(config, "object_profile"))
        if name:
            result[name] = profile
    return result


def _snapshot_from_input(snapshot: SimulationSnapshot | Mapping[str, object]) -> SimulationSnapshot:
    """统一接受 dataclass 或 JSON dict 形式的 snapshot 输入。"""

    # 协议层通常传 JSON dict，内部调用/测试可以直接传 dataclass；统一在 adapter 边界解析。
    if isinstance(snapshot, SimulationSnapshot):
        return snapshot
    if isinstance(snapshot, Mapping):
        return SimulationSnapshot.from_mapping(snapshot)
    raise ValueError("snapshot must be a SimulationSnapshot or JSON object")


def _is_debug_tiled_runtime(runtime: object) -> bool:
    """粗略判断对象是否是 debug tiled runtime。"""

    return hasattr(runtime, "current_positions") and hasattr(runtime, "adapter")


def _looks_like_tiled_runtime(runtime: object) -> bool:
    """粗略判断对象是否暴露 tiled runtime 所需接口。"""

    return _is_debug_tiled_runtime(runtime) or hasattr(runtime, "scene")


def _looks_like_single_runtime(runtime: object) -> bool:
    """粗略判断对象是否是 single-arm runtime。"""

    execution = getattr(runtime, "execution", None)
    return execution is not None and hasattr(execution, "articulation")


def _looks_like_dual_runtime(runtime: object) -> bool:
    """粗略判断对象是否是 dual-arm runtime。"""

    execution = getattr(runtime, "execution", None)
    return execution is not None and hasattr(execution, "left") and hasattr(execution, "right")


def _read_tiled_object_states_lazy():
    """惰性导入 tiled object state reader，避免 snapshots 包强依赖 Isaac。"""

    # lazy import 避免 snapshots 包导入时强依赖 tiled/Isaac 相关模块。
    from linkerbot_sim.app.interactive.tiled.object_states import _read_tiled_object_states

    return _read_tiled_object_states


def _restore_tiled_object_pose_snapshot_lazy():
    """惰性导入 tiled object restore helper。"""

    from linkerbot_sim.app.interactive.tiled.object_states import (
        _restore_tiled_object_pose_snapshot,
    )

    return _restore_tiled_object_pose_snapshot


def _read_prim_world_pose_lazy():
    """惰性导入 USD prim world pose reader。"""

    from linkerbot_sim.app.interactive.tiled.object_states import _read_prim_world_pose

    return _read_prim_world_pose


def _apply_prim_local_pose_and_zero_velocity_lazy():
    """惰性导入 prim pose 写回和速度清零 helper。"""

    from linkerbot_sim.app.interactive.tiled.object_states import (
        _apply_prim_local_pose_and_zero_velocity,
    )

    return _apply_prim_local_pose_and_zero_velocity


def _single_env_id(env_id: int, num_envs: int) -> np.ndarray:
    """校验并返回只包含一个 env id 的 ndarray。"""

    env = int(env_id)
    if env < 0 or env >= int(num_envs):
        raise ValueError("env_id is out of range")
    return np.asarray([env], dtype=int)


def _env_ids(env_ids: Sequence[int] | np.ndarray, num_envs: int) -> np.ndarray:
    """校验并规范化 selected env ids。"""

    selected = np.asarray(env_ids, dtype=int).reshape(-1)
    if selected.size < 1:
        raise ValueError("env_ids cannot be empty")
    if np.any(selected < 0) or np.any(selected >= int(num_envs)):
        raise ValueError("env_ids contains out-of-range env id")
    return selected.astype(int, copy=True)
