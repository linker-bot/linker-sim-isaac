"""PhysX scene 和机器人刚体 solver 覆盖。

复杂接触、细小灵巧手和绳体交互都容易受 solver 迭代次数影响。
这里把 scene 级 solver type 和 robot 级 arm/hand 迭代次数分开配置，便于在不同实验中只覆盖
真正归属对应层级的 PhysX 属性。

这些覆盖发生在 USD/PhysX 属性层面，只影响求解稳定性和接触收敛，不改变关节目标、
控制器命令空间或动作脚本 API。arm/hand 分类依赖资产命名约定，未知 prim 会跳过。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from linkerbot_sim.robots.classification import (
    RobotComponentMapping,
    is_arm_name,
    is_hand_name,
)


@dataclass(frozen=True)
class SolverIterationConfig:
    """PhysX solver type 和机器人刚体可选迭代次数覆盖。

    position/velocity iteration 分别对应 PhysX 位置约束和速度约束求解次数。数值越大通常
    越稳定但越慢，适合在绳体接触、指尖夹持等局部不稳定时按部件提高。字段为 ``None``
    表示不覆盖该 PhysX 属性。

    输入字段:
        solver_type: 可选 ``PGS`` 或 ``TGS``，写到 physics scene，通常来自 env YAML。
        arm_position_iterations/arm_velocity_iterations: 可选机械臂刚体迭代次数。
        hand_position_iterations/hand_velocity_iterations: 可选灵巧手刚体迭代次数。
    输出:
        传给 ``apply_solver_iteration_overrides`` 后写入 stage。
    """

    solver_type: str | None = None
    arm_position_iterations: int | None = None
    arm_velocity_iterations: int | None = None
    hand_position_iterations: int | None = None
    hand_velocity_iterations: int | None = None


def scene_solver_settings(
    env_config: Mapping[str, object],
) -> SolverIterationConfig | None:
    """从环境配置构造 scene 级 PhysX solver 覆盖设置。

    env YAML 只允许声明 ``solver.type``，因为它写到 physics scene。机器人刚体 iteration
    属于 robot YAML 的 ``robot.physics.solver``。
    """

    solver = env_config.get("solver")
    if solver is None:
        return None
    if not isinstance(solver, Mapping):
        raise ValueError("solver config must be a mapping")
    _reject_solver_keys(
        solver,
        {"type"},
        "solver",
        extra_message=("arm/hand solver iterations belong under robot.physics.solver"),
    )
    config = SolverIterationConfig(
        solver_type=_optional_string(solver, "type", "solver")
    )
    return config if _has_overrides(config) else None


def robot_solver_settings(
    robot_solver_config: Mapping[str, object] | None, *, label: str
) -> SolverIterationConfig | None:
    """从 ``robot.physics.solver`` 构造机器人刚体 solver iteration 覆盖。"""

    if robot_solver_config is None:
        return None
    if not isinstance(robot_solver_config, Mapping):
        raise ValueError(f"{label} must be a mapping")
    _reject_solver_keys(
        robot_solver_config,
        {"arm", "hand"},
        label,
        extra_message="scene solver type belongs under env solver.type",
    )
    arm = _optional_group_solver_mapping(robot_solver_config, "arm", label)
    hand = _optional_group_solver_mapping(robot_solver_config, "hand", label)
    config = SolverIterationConfig(
        arm_position_iterations=(
            _optional_int(arm, "position_iterations", f"{label}.arm")
            if arm is not None
            else None
        ),
        arm_velocity_iterations=(
            _optional_int(arm, "velocity_iterations", f"{label}.arm")
            if arm is not None
            else None
        ),
        hand_position_iterations=(
            _optional_int(hand, "position_iterations", f"{label}.hand")
            if hand is not None
            else None
        ),
        hand_velocity_iterations=(
            _optional_int(hand, "velocity_iterations", f"{label}.hand")
            if hand is not None
            else None
        ),
    )
    return config if _has_iteration_overrides(config) else None


def merge_solver_configs(
    *configs: SolverIterationConfig | None,
) -> SolverIterationConfig | None:
    """合并 scene solver type 和 robot solver iteration 覆盖。"""

    merged = SolverIterationConfig()
    has_any = False
    for config in configs:
        if config is None:
            continue
        has_any = True
        merged = SolverIterationConfig(
            solver_type=(
                config.solver_type
                if config.solver_type is not None
                else merged.solver_type
            ),
            arm_position_iterations=(
                config.arm_position_iterations
                if config.arm_position_iterations is not None
                else merged.arm_position_iterations
            ),
            arm_velocity_iterations=(
                config.arm_velocity_iterations
                if config.arm_velocity_iterations is not None
                else merged.arm_velocity_iterations
            ),
            hand_position_iterations=(
                config.hand_position_iterations
                if config.hand_position_iterations is not None
                else merged.hand_position_iterations
            ),
            hand_velocity_iterations=(
                config.hand_velocity_iterations
                if config.hand_velocity_iterations is not None
                else merged.hand_velocity_iterations
            ),
        )
    return merged if has_any and _has_overrides(merged) else None


def _optional_string(
    data: Mapping[str, object], key: str, parent_label: str
) -> str | None:
    """读取可选非空字符串字段。"""

    value = data.get(key)
    if value is None:
        return None
    text = str(value)
    if not text:
        raise ValueError(f"{parent_label}.{key} cannot be empty")
    return text


def _optional_int(
    data: Mapping[str, object], key: str, parent_label: str
) -> int | None:
    """读取可选非负整数字段。"""

    value = data.get(key)
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{parent_label}.{key} must be non-negative")
    return parsed


def _optional_group_solver_mapping(
    data: Mapping[str, object], key: str, parent_label: str
) -> Mapping[str, object] | None:
    """读取 arm/hand solver 子分组，并限制只能写 iteration 字段。"""

    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    _reject_solver_keys(
        value,
        {"position_iterations", "velocity_iterations"},
        f"{parent_label}.{key}",
    )
    return value


def _reject_solver_keys(
    data: Mapping[str, object],
    allowed: set[str],
    label: str,
    *,
    extra_message: str | None = None,
) -> None:
    """拒绝 solver 配置中的未知 key，并可追加层级归属提示。"""

    unsupported = set(data) - allowed
    if not unsupported:
        return
    message = f"{label} contains unsupported keys: {', '.join(sorted(unsupported))}"
    if extra_message is not None:
        message = f"{message}; {extra_message}"
    raise ValueError(message)


def _has_overrides(config: SolverIterationConfig) -> bool:
    """判断配置是否包含 scene solver type 或任一刚体 iteration 覆盖。"""

    return config.solver_type is not None or _has_iteration_overrides(config)


def _has_iteration_overrides(config: SolverIterationConfig) -> bool:
    """判断配置是否包含 arm/hand position/velocity iteration 覆盖。"""

    return any(
        getattr(config, field_name) is not None for field_name in _ITERATION_FIELD_NAMES
    )


def is_hand_prim_name(name: str) -> bool:
    """判断 prim 名是否属于灵巧手。

    参数:
        name: USD prim 名称。
    返回:
        名称是否包含规范 category ``hand``。
    """

    return is_hand_name(name)


def is_arm_prim_name(name: str) -> bool:
    """判断 prim 名是否属于机械臂。

    参数:
        name: USD prim 名称。
    返回:
        名称是否包含规范 category ``arm``。
    """

    return is_arm_name(name)


def solver_iterations_for_prim_name(
    name: str,
    config: SolverIterationConfig,
    *,
    component_mapping: RobotComponentMapping | None = None,
) -> tuple[int | None, int | None, str] | None:
    """根据 prim 名和配置决定该刚体要写入的迭代次数。

    参数:
        name: USD prim 名称。
        config: solver 覆盖配置。
    返回:
        ``(position_iterations, velocity_iterations, group)``；该刚体没有任何字段需要覆盖时
        返回 ``None``。
    """

    component = (
        component_mapping.rigid_body_component(name)
        if component_mapping is not None
        else (
            "arm"
            if is_arm_prim_name(name)
            else "hand"
            if is_hand_prim_name(name)
            else "default"
        )
    )
    if component == "arm":
        position_iterations = config.arm_position_iterations
        velocity_iterations = config.arm_velocity_iterations
        group = "arm"
    elif component == "hand":
        position_iterations = config.hand_position_iterations
        velocity_iterations = config.hand_velocity_iterations
        group = "hand"
    else:
        return None
    if position_iterations is None and velocity_iterations is None:
        return None
    return position_iterations, velocity_iterations, group


def apply_solver_iteration_overrides(
    stage,
    articulation_root_path: str,
    config: SolverIterationConfig,
    *,
    component_mapping: RobotComponentMapping | None = None,
) -> dict[str, int]:
    """写入配置中显式指定的 PhysX solver 属性。

    参数:
        stage: 当前 USD stage。
        articulation_root_path: articulation root prim 路径。
        config: solver 覆盖配置。
    返回:
        统计字典，记录写入的刚体数量、分组数量、属性数量和 physics scene 数量。
    """

    from isaacsim.core.utils.prims import get_prim_at_path
    from pxr import PhysxSchema, Usd, UsdPhysics

    _validate_solver_config(config)

    # solver 类型写在 physics scene 上，是全局物理求解策略；迭代次数写在刚体/关节树上，
    # 可以只提高关键部件的稳定性，避免整场景成本过高。
    physics_scene_prims = [
        prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)
    ]
    if config.solver_type is not None:
        solver_type = str(config.solver_type).upper()
        for scene_prim in physics_scene_prims:
            scene_api = (
                PhysxSchema.PhysxSceneAPI(scene_prim)
                if scene_prim.HasAPI(PhysxSchema.PhysxSceneAPI)
                else PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
            )
            scene_api.CreateSolverTypeAttr().Set(solver_type)

    counts = {
        "rigid_bodies": 0,
        "arm_rigid_bodies": 0,
        "hand_rigid_bodies": 0,
        "skipped_rigid_bodies": 0,
        "physics_scenes": len(physics_scene_prims),
    }
    if not _has_iteration_overrides(config):
        return counts

    articulation_root = get_prim_at_path(articulation_root_path)
    for prim in Usd.PrimRange(articulation_root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        # 通过命名约定判断 arm/hand，未知刚体不写入，避免把环境或装饰 prim 意外纳入迭代次数覆盖。
        solver_iterations = solver_iterations_for_prim_name(
            prim.GetName(), config, component_mapping=component_mapping
        )
        if solver_iterations is None:
            counts["skipped_rigid_bodies"] += 1
            continue
        position_iterations, velocity_iterations, group = solver_iterations
        rigid_api = (
            PhysxSchema.PhysxRigidBodyAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            else PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        )
        if position_iterations is not None:
            rigid_api.CreateSolverPositionIterationCountAttr().Set(
                int(position_iterations)
            )
        if velocity_iterations is not None:
            rigid_api.CreateSolverVelocityIterationCountAttr().Set(
                int(velocity_iterations)
            )
        counts["rigid_bodies"] += 1
        if group == "arm":
            counts["arm_rigid_bodies"] += 1
        elif group == "hand":
            counts["hand_rigid_bodies"] += 1
    return counts


def _validate_solver_config(config: SolverIterationConfig) -> None:
    """校验 solver type 和 iteration 数值范围。"""

    if config.solver_type is not None:
        solver_type = str(config.solver_type).upper()
        if solver_type not in {"PGS", "TGS"}:
            raise ValueError(f"Unsupported solver_type: {solver_type}")
    for field_name in _ITERATION_FIELD_NAMES:
        value = getattr(config, field_name)
        if value is not None and int(value) < 0:
            raise ValueError(f"{field_name} must be non-negative")


_ITERATION_FIELD_NAMES = (
    "arm_position_iterations",
    "arm_velocity_iterations",
    "hand_position_iterations",
    "hand_velocity_iterations",
)
