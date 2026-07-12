"""snapshot adapter 共享的机器人、对象 descriptor、指纹与状态恢复 helper。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path

import numpy as np

from linkerbot_sim.objects.state_views import SceneObjectStateView
from linkerbot_sim.snapshots.compatibility import (
    ObjectTargetDescriptor,
    RobotTargetDescriptor,
)
from linkerbot_sim.snapshots.schema import (
    ObjectSnapshot,
    RobotSnapshot,
    SimulationSnapshot,
)
from linkerbot_sim.tiled.state.usd_pose import (
    apply_prim_local_pose_and_zero_velocity,
    read_prim_world_pose,
)


def _robot_snapshot_from_execution(
    *,
    label: str,
    robot_id: int,
    execution: object,
    robot_profile: str | None,
    asset_fingerprint: str | None,
) -> RobotSnapshot:
    """从一个 RobotRuntime execution 读取 command-joint 快照。"""

    articulation = execution.articulation
    controller = execution.joint_controller
    command_indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    joint_names = _command_joint_names(articulation, controller)
    # articulation API 返回全 DOF；snapshot 只截取 controller 管理的 command joints，
    # 保持与 tiled adapter 的语义一致。
    positions = np.asarray(articulation.get_joint_positions(), dtype=float).reshape(-1)
    velocities = np.asarray(articulation.get_joint_velocities(), dtype=float).reshape(
        -1
    )
    return RobotSnapshot(
        label=label,
        robot_id=robot_id,
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
    label: str,
    execution: object,
    robot_profile: str | None,
    asset_fingerprint: str | None,
) -> RobotTargetDescriptor:
    """从 execution 对象提取目标机器人关节名字和资产指纹。"""

    joint_names = _command_joint_names(
        execution.articulation,
        execution.joint_controller,
    )
    return RobotTargetDescriptor(
        label=label,
        robot_profile=robot_profile,
        asset_fingerprint=asset_fingerprint,
        joint_names=joint_names,
        command_joint_names=joint_names,
    )


def _restore_robot_snapshot_to_execution(
    execution: object,
    source_robot: RobotSnapshot,
    *,
    mapping: object,
) -> None:
    """把一个 ``RobotSnapshot`` 写回 RobotRuntime execution 的 articulation。"""

    articulation = execution.articulation
    controller = execution.joint_controller
    command_indices = np.asarray(controller.command_indices, dtype=int).reshape(-1)
    q = np.asarray(articulation.get_joint_positions(), dtype=float).reshape(-1)
    dq = np.asarray(articulation.get_joint_velocities(), dtype=float).reshape(-1)
    # mapping.joints.target_indices 是 command-joint 空间索引；先转换成 articulation DOF
    # index，再写入全量 q/dq，避免覆盖非 command DOF。
    target_command_indices = command_indices[mapping.joints.target_indices]
    q[target_command_indices] = source_robot.joint_positions[
        mapping.joints.source_indices
    ]
    dq[target_command_indices] = source_robot.joint_velocities[
        mapping.joints.source_indices
    ]
    articulation.set_joint_positions(q)
    articulation.set_joint_velocities(dq)
    if hasattr(controller, "last_commanded_efforts"):
        controller.last_commanded_efforts = np.full(q.size, np.nan, dtype=float)


def _command_joint_names(articulation: object, controller: object) -> tuple[str, ...]:
    """按 controller 配置返回 command joint 名字。"""

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
    state_views: Mapping[str, SceneObjectStateView] | None = None,
) -> dict[str, ObjectSnapshot]:
    """读取 SingleSceneRuntime object handles 对应的 scene-local 对象快照。"""

    if stage is None:
        return {}
    views = {} if state_views is None else state_views
    result: dict[str, ObjectSnapshot] = {}
    for handle in handles:
        name = _runtime_object_name(handle)
        prim_path = _runtime_object_prim_path(handle)
        if not name or prim_path is None:
            continue
        pose = read_prim_world_pose(stage, prim_path)
        if pose is None:
            continue
        position, orientation = pose
        body_names, body_paths = _runtime_object_body_paths(handle)
        state_view = views.get(name)
        if state_view is not None:
            state_view.require_velocity_support(object_name=name)
        kwargs: dict[str, object] = {}
        velocities = (
            state_view.root_velocities()
            if state_view is not None and state_view.root_view is not None
            else _read_prim_rigid_body_velocities(stage, prim_path)
        )
        if velocities is not None:
            kwargs["linear_velocities"], kwargs["angular_velocities"] = velocities
        if body_names:
            # SingleSceneRuntime 没有 env origin，root/child body 都按 scene-local pose 保存。
            body_positions = []
            body_orientations = []
            body_linear_velocities = []
            body_angular_velocities = []
            body_velocities_complete = True
            live_body_velocities = None
            if state_view is not None and state_view.body_view is not None:
                if state_view.body_names != body_names:
                    raise RuntimeError(
                        f"Scene object {name!r} body view names do not match snapshot paths"
                    )
                live_body_velocities = state_view.body_velocities()
            for body_index, body_path in enumerate(body_paths):
                body_pose = read_prim_world_pose(stage, body_path)
                if body_pose is None:
                    break
                body_position, body_orientation = body_pose
                body_positions.append(body_position)
                body_orientations.append(body_orientation)
                body_velocities = (
                    None
                    if live_body_velocities is None
                    else (
                        live_body_velocities[0][body_index],
                        live_body_velocities[1][body_index],
                    )
                )
                if state_view is None or state_view.body_view is None:
                    body_velocities = _read_prim_rigid_body_velocities(stage, body_path)
                if body_velocities is None:
                    body_velocities_complete = False
                else:
                    body_linear, body_angular = body_velocities
                    body_linear_velocities.append(body_linear)
                    body_angular_velocities.append(body_angular)
            if len(body_positions) == len(body_names):
                kwargs.update(
                    {
                        "body_names": body_names,
                        "body_positions_local": np.vstack(body_positions),
                        "body_orientations_wxyz": np.vstack(body_orientations),
                    }
                )
                if body_velocities_complete:
                    kwargs.update(
                        {
                            "body_linear_velocities": np.vstack(body_linear_velocities),
                            "body_angular_velocities": np.vstack(
                                body_angular_velocities
                            ),
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


def _runtime_object_targets(
    handles: Sequence[object],
) -> dict[str, ObjectTargetDescriptor]:
    """根据 SingleSceneRuntime object handles 构建 object target descriptors。"""

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
    """把 snapshot objects 写回 SingleSceneRuntime 中的 USD prim。"""

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
        state_view = getattr(runtime, "object_state_views", {}).get(target_name)
        if state_view is not None:
            state_view.require_velocity_support(object_name=target_name)
        if prim_path is not None and _apply_prim_local_pose_and_velocity(
            stage,
            prim_path,
            obj.positions_local,
            obj.orientations_wxyz,
            obj.linear_velocities,
            obj.angular_velocities,
            state_view=state_view,
        ):
            restored.append(target_name)
        if obj.body_names and mapping.bodies is not None:
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
                _apply_prim_local_pose_and_velocity(
                    stage,
                    body_path,
                    obj.body_positions_local[int(source_index)],
                    obj.body_orientations_wxyz[int(source_index)],
                    (
                        None
                        if obj.body_linear_velocities is None
                        else obj.body_linear_velocities[int(source_index)]
                    ),
                    (
                        None
                        if obj.body_angular_velocities is None
                        else obj.body_angular_velocities[int(source_index)]
                    ),
                    state_view=state_view,
                    body_index=(
                        None
                        if state_view is None or state_view.body_view is None
                        else state_view.body_names.index(body_name)
                    ),
                )
    return tuple(restored)


def _read_prim_rigid_body_velocities(
    stage: object,
    prim_path: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """读取 prim 自身的 USD 刚体速度；非刚体 root 返回 ``None``。"""

    from pxr import Sdf, UsdPhysics

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return None
    api = UsdPhysics.RigidBodyAPI(prim)
    values = []
    for getter_name in ("GetVelocityAttr", "GetAngularVelocityAttr"):
        attr = getattr(api, getter_name)()
        value = attr.Get() if attr is not None and attr.IsValid() else None
        values.append(
            np.zeros(3, dtype=float)
            if value is None
            else np.asarray(value, dtype=float).reshape(3)
        )
    # USD Physics authoring attr 使用 deg/s；canonical snapshot 和 live PhysX view 使用 rad/s。
    return values[0], np.deg2rad(values[1])


def _apply_prim_local_pose_and_velocity(
    stage: object,
    prim_path: str,
    position: np.ndarray,
    orientation_wxyz: np.ndarray,
    linear_velocity: np.ndarray | None,
    angular_velocity: np.ndarray | None,
    *,
    state_view: SceneObjectStateView | None = None,
    body_index: int | None = None,
) -> bool:
    """恢复 pose，并为 live rigid view 写回完整的速度对。"""

    if (linear_velocity is None) != (angular_velocity is None):
        raise ValueError("object velocity requires both linear and angular components")
    live_velocity_target = state_view is not None and (
        (body_index is None and state_view.root_view is not None)
        or (body_index is not None and state_view.body_view is not None)
    )
    if live_velocity_target and linear_velocity is None:
        raise ValueError("object snapshot is missing required velocity state")

    applied = apply_prim_local_pose_and_zero_velocity(
        stage,
        prim_path,
        position,
        orientation_wxyz,
    )
    if not applied:
        return applied
    if live_velocity_target:
        assert linear_velocity is not None
        assert angular_velocity is not None
        if body_index is None:
            state_view.set_root_velocities(linear_velocity, angular_velocity)
        else:
            state_view.set_body_velocities(
                body_index=body_index,
                linear=linear_velocity,
                angular=angular_velocity,
            )
        return True
    if linear_velocity is None and angular_velocity is None:
        return True
    from pxr import Gf, Sdf, UsdPhysics

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"object prim {prim_path!r} does not have RigidBodyAPI")
    api = UsdPhysics.RigidBodyAPI(prim)
    for getter_name, velocity, angular in (
        ("GetVelocityAttr", linear_velocity, False),
        ("GetAngularVelocityAttr", angular_velocity, True),
    ):
        if velocity is None:
            continue
        xyz = np.asarray(velocity, dtype=float).reshape(3)
        if angular:
            xyz = np.rad2deg(xyz)
        getattr(api, getter_name)().Set(
            Gf.Vec3f(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        )
    return True


def _reset_execution_observers(execution: object | None) -> None:
    """恢复状态后重置 execution 上可能缓存旧采样的 observer。"""

    if execution is None:
        return
    for name in ("state_observer", "camera_observer"):
        observer = getattr(execution, name, None)
        reset = getattr(observer, "reset", None)
        if callable(reset):
            reset()


def _runtime_object_name(handle: object) -> str:
    """从 runtime object handle 中提取对外使用的稳定对象名。"""

    runtime_handle = getattr(handle, "runtime_handle", None)
    if runtime_handle is not None:
        return str(runtime_handle)
    name = getattr(handle, "name", None)
    return "" if name is None else str(name)


def _runtime_object_profile(handle: object) -> str | None:
    """读取 object profile 名称；缺失时返回 ``None``。"""

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


def _runtime_object_body_paths(
    handle: object,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """读取 dynamic/multi-body object 的 child body 名字和 prim paths。"""

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
        body_name = str(
            name_getter() if callable(name_getter) else body_path.rsplit("/", 1)[-1]
        )
        names.append(body_name)
        paths.append(body_path)
    return tuple(names), tuple(paths)


def _imported_asset_fingerprint(imported: object | None) -> str | None:
    """计算已导入仿真资产的稳定指纹。"""

    if imported is None:
        return None
    asset_path = getattr(imported, "asset_path", None)
    if asset_path is None:
        return None
    return _asset_fingerprint_from_path(asset_path)


def _asset_fingerprint_from_path(asset_path: str | Path) -> str:
    """基于规范路径和文件内容返回 scene/tiled 共用的 SHA-256 指纹。"""

    path = Path(asset_path)
    digest = hashlib.sha256(str(path.resolve()).encode())
    if path.is_file():
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _object_profiles_by_name(runtime: object) -> dict[str, str | None]:
    """按 object name 收集 TiledSceneRuntime 中的 object profile。"""

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
