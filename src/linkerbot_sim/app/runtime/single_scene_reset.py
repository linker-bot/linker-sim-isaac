"""已创建 Isaac session 的轻量 reset 工具。

这里的 reset 目标是“复用现有 ``SimulationApp`` 和 ``World``”，只恢复机器人 root pose、
对象 root pose、物理世界状态和若干 observer/controller 缓存。它不是重新构建场景，也不重新
加载 USD/MJCF/URDF，因此适合交互式服务中的 ``reset`` 指令和 smoke test 循环。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from linkerbot_sim.objects.runtime import (
    RuntimeObjectConfig,
    runtime_objects_from_env_config,
)
from linkerbot_sim.envs.settings import EnvRuntimeSettings
from linkerbot_sim.assets.root_pose import (
    RootPoseConfig,
    apply_root_pose,
)
from linkerbot_sim.objects.physics import apply_root_pose_to_prim
from linkerbot_sim.snapshots.transactions import (
    MutationTransaction,
    mutation_transaction,
    require_runtime_mutable,
)
from linkerbot_sim.tiled.state.usd_pose import read_prim_world_pose


RobotRootPoseApplier = Callable[[object, str, RootPoseConfig], None]
ObjectRootPoseApplier = Callable[[object, str, RootPoseConfig], None]
RootPoseReader = Callable[[object, str], RootPoseConfig]


@dataclass(frozen=True)
class SingleSceneResetOptions:
    """轻量 reset 的可选行为。

    ``reset_single_scene_runtime`` 将 ``hold_after_reset`` 写入结果摘要；交互式 Single Scene
    主循环读取同一选项，并在 reset 成功后决定是否执行短 hold。底层 helper 本身不直接推进
    physics step。
    """

    hold_after_reset: bool = True


@dataclass(frozen=True)
class SingleSceneResetResult:
    """轻量 reset 的结果摘要。"""

    step: int = 0
    message: str = ""


@dataclass(frozen=True)
class _RootPoseResetPlan:
    """一项已校验的 root pose 写入及其 reset 前回滚值。

    plan 在任何 USD 写入前完整构造，确保事务开始后不再因读取旧位姿失败而留下半写状态。
    """

    label: str
    prim_path: str
    target: RootPoseConfig
    original: RootPoseConfig
    applier: Callable[[object, str, RootPoseConfig], None]


def reset_single_scene_runtime(
    runtime,
    *,
    options: SingleSceneResetOptions = SingleSceneResetOptions(),
    robot_root_pose_applier: RobotRootPoseApplier = apply_root_pose,
    object_root_pose_applier: ObjectRootPoseApplier = apply_root_pose_to_prim,
    root_pose_reader: RootPoseReader | None = None,
) -> SingleSceneResetResult:
    """围绕唯一一次 World reset 恢复任意数量已注册 robot。

    object/robot root pose 在 reset 前写回，controller、observer 和 collision registry 在
    reset 后重置，避免旧 velocity、采样状态或 collision snapshot 穿过 reset 边界。
    """

    require_runtime_mutable(runtime, operation="reset_single_scene_runtime")
    if root_pose_reader is None:
        root_pose_reader = _read_root_pose
    stage = runtime.session.stage
    robots = tuple(runtime.robots_by_id.values())
    object_configs = runtime_objects_from_env_config(runtime.env_config)
    target_gravity_z = EnvRuntimeSettings.from_env_config(runtime.env_config).gravity_z

    # 必须在第一次 USD 写入前捕获全部回滚位姿。World.reset 会修改无法枚举的 PhysX
    # 内部状态，这部分不能通过逐 prim 回写恢复，因此它之后的失败会使 runtime 失效。
    object_root_plans = _object_root_pose_plans(
        stage=stage,
        handles=runtime.object_handles,
        configs=object_configs,
        root_pose_reader=root_pose_reader,
        object_root_pose_applier=object_root_pose_applier,
    )
    robot_root_plans = _robot_root_pose_plans(
        stage=stage,
        robots=robots,
        root_pose_reader=root_pose_reader,
        robot_root_pose_applier=robot_root_pose_applier,
    )
    with mutation_transaction(
        runtime,
        operation="reset_single_scene_runtime",
    ) as transaction:
        for plan in object_root_plans:
            _apply_root_pose_plan(transaction, stage=stage, plan=plan)
        for plan in robot_root_plans:
            _apply_root_pose_plan(transaction, stage=stage, plan=plan)

        # 即使 World.reset 抛出异常，也可能已部分修改不透明的模拟器状态。从这里开始的
        # 任何失败都必须重建 runtime，事务通过 irreversible 标记对外保留这一事实。
        transaction.mark_irreversible("World.reset")
        _reset_world(runtime, gravity_z=target_gravity_z)
        for robot in robots:
            _reset_prepared_robot(robot.prepared)
            _reset_execution_observers(robot.execution)
        _reset_execution_observers(runtime)
        collision_registry = getattr(runtime, "collision_registry", None)
        mark_dirty = getattr(collision_registry, "mark_dirty", None)
        if callable(mark_dirty):
            mark_dirty()
    return SingleSceneResetResult(
        step=0,
        message=(
            "runtime reset completed; hold_after_reset="
            f"{bool(options.hold_after_reset)}"
        ),
    )


def _reset_world(runtime, *, gravity_z: float) -> None:
    """reset world，并恢复 env profile 中声明的场景重力。"""

    runtime.session.world.reset()
    runtime.session.world.get_physics_context().set_gravity(gravity_z)


def _reset_prepared_robot(prepared) -> None:
    """恢复 world reset 后容易丢失或变脏的机器人运行态设置。"""

    articulation = prepared.articulation
    if prepared.gravity_policy.disables_all_known_components():
        # Isaac reset 后部分 articulation 属性会回到默认值，需要重新应用机器人级重力策略。
        articulation.disable_gravity()
    _call_optional(articulation, "set_joint_velocities", _zeros(_num_dof(articulation)))
    controller = prepared.joint_controller
    configure_runtime = getattr(controller, "configure_runtime", None)
    if configure_runtime is not None:
        configure_runtime()
    if hasattr(controller, "last_commanded_efforts"):
        # NaN 表示 reset 后尚未发送过 effort，日志和调试面板可据此区分“真实 0 力矩”。
        controller.last_commanded_efforts = np.full(
            _num_dof(articulation), np.nan, dtype=float
        )


def _object_root_pose_plans(
    *,
    stage: object,
    handles: Sequence[object],
    configs: Sequence[RuntimeObjectConfig],
    root_pose_reader: RootPoseReader,
    object_root_pose_applier: ObjectRootPoseApplier,
) -> tuple[_RootPoseResetPlan, ...]:
    """为可定位的 runtime objects 构造 root pose 写入与回滚计划。

    配置可同时用稳定对象名和 ``runtime_handle`` 别名索引；没有对应配置或 prim path 的
    handle 不属于可重置对象，保持原状。任一旧位姿读取失败都会在首次写入前终止。
    """

    configs_by_key: dict[str, RuntimeObjectConfig] = {}
    for config in configs:
        configs_by_key[config.name] = config
        if config.runtime_handle is not None:
            # runtime_handle 是交互层可读写对象时使用的别名，优先级与配置名一致。
            configs_by_key[config.runtime_handle] = config
    plans = []
    for handle in handles:
        config = _runtime_object_config_for_handle(handle, configs_by_key)
        prim_path = _runtime_object_prim_path(handle)
        if config is None or prim_path is None:
            continue
        plans.append(
            _RootPoseResetPlan(
                label=f"object {config.name}",
                prim_path=prim_path,
                target=config.root_pose,
                original=_require_root_pose(
                    root_pose_reader(stage, prim_path),
                    label=f"object {config.name}",
                ),
                applier=object_root_pose_applier,
            )
        )
    return tuple(plans)


def _robot_root_pose_plans(
    *,
    stage: object,
    robots: Sequence[object],
    root_pose_reader: RootPoseReader,
    robot_root_pose_applier: RobotRootPoseApplier,
) -> tuple[_RootPoseResetPlan, ...]:
    """为全部已导入机器人构造 root pose 写入与回滚计划。

    尚无 ``imported_root_path`` 的条目没有可写 USD 根节点，因而跳过；其余机器人必须先
    成功捕获原位姿，随后才能进入事务提交阶段。
    """

    plans = []
    for robot in robots:
        root_path = getattr(robot.imported, "imported_root_path", None)
        if root_path is None:
            continue
        path = str(root_path)
        label = f"robot {getattr(robot, 'label', '?')}"
        plans.append(
            _RootPoseResetPlan(
                label=label,
                prim_path=path,
                target=robot.scene_instance.root_pose,
                original=_require_root_pose(
                    root_pose_reader(stage, path),
                    label=label,
                ),
                applier=robot_root_pose_applier,
            )
        )
    return tuple(plans)


def _apply_root_pose_plan(
    transaction: MutationTransaction,
    *,
    stage: object,
    plan: _RootPoseResetPlan,
) -> None:
    """先登记单项 root pose 回滚，再执行目标写入。

    登记顺序不能交换：applier 可能在部分修改 USD 后抛错，预先登记才能让事务尝试恢复。
    """

    transaction.add_rollback(
        f"{plan.label} root pose",
        lambda: plan.applier(stage, plan.prim_path, plan.original),
    )
    plan.applier(stage, plan.prim_path, plan.target)


def _reset_execution_observers(execution: object) -> None:
    """清除 ``World.reset`` 后已失效的状态与相机 observer 缓存。"""

    for name in ("state_observer", "camera_observer"):
        observer = getattr(execution, name, None)
        reset = getattr(observer, "reset", None)
        if not callable(reset):
            continue
        reset()


def _read_root_pose(stage: object, prim_path: str) -> RootPoseConfig:
    """读取 USD 世界位姿，并把 wxyz 四元数转换为 XYZ 欧拉角。

    ``RootPoseConfig`` 使用 ``xyz`` + ``rpy``，而统一 USD reader 返回 wxyz；转换前先重排
    为 SciPy 所需的 xyzw。prim 不存在时拒绝继续，以免事务缺少可靠回滚值。
    """

    pose = read_prim_world_pose(stage, prim_path)
    if pose is None:
        raise RuntimeError(f"Cannot capture root pose; prim not found: {prim_path}")
    position, quat_wxyz = pose
    quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
    rpy = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")
    return RootPoseConfig(
        xyz=tuple(float(value) for value in np.asarray(position).reshape(3)),
        rpy=tuple(float(value) for value in rpy),
    )


def _require_root_pose(value: object, *, label: str) -> RootPoseConfig:
    """拒绝未返回完整 ``RootPoseConfig`` 的自定义 reader。"""

    if not isinstance(value, RootPoseConfig):
        raise RuntimeError(f"{label} root pose reader returned invalid state")
    return value


def _runtime_object_config_for_handle(
    handle: object,
    configs_by_key: Mapping[str, RuntimeObjectConfig],
) -> RuntimeObjectConfig | None:
    """根据 runtime handle 找回创建它的 env 对象配置。"""

    for key in (getattr(handle, "runtime_handle", None), getattr(handle, "name", None)):
        if key is not None and str(key) in configs_by_key:
            return configs_by_key[str(key)]
    return None


def _runtime_object_prim_path(handle: object) -> str | None:
    """从类似 ``RuntimeObjectHandle`` 的对象中读取 root prim path。"""

    for source in (getattr(handle, "model", None), getattr(handle, "config", None)):
        if source is None:
            continue
        prim_path = getattr(source, "prim_path", None)
        if prim_path is not None:
            return str(prim_path)
        if isinstance(source, Mapping):
            # 部分测试 fake 使用 mapping 暴露 USD root prim，兼容它可以保持 reset 单元测试轻量。
            root = source.get("root")
            if root is not None and hasattr(root, "GetPath"):
                return str(root.GetPath())
            prim_path = source.get("prim_path")
            if prim_path is not None:
                return str(prim_path)
    return None


def _call_optional(source: object, method_name: str, *args: Any) -> None:
    """若对象存在可选方法则调用，适配 Isaac 对象和测试 fake 的差异。"""

    method = getattr(source, method_name, None)
    if method is not None:
        method(*args)


def _num_dof(articulation: object) -> int:
    """读取 articulation DOF 数量，兼容真实 Isaac 和测试 fake。"""

    if hasattr(articulation, "num_dof"):
        return int(getattr(articulation, "num_dof"))
    return len(getattr(articulation, "dof_names", ()))


def _zeros(size: int) -> np.ndarray:
    """返回 float 类型零向量。"""

    return np.zeros(int(size), dtype=float)
