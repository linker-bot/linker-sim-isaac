"""机器人刚体的 PhysX solver 覆盖。

复杂接触、细小灵巧手和绳体交互都容易受 solver 迭代次数影响。
这里把 solver type 和 arm/hand 迭代次数按字段独立配置，便于在不同实验中只覆盖
真正需要调整的 PhysX 属性。

这些覆盖发生在 USD/PhysX 属性层面，只影响求解稳定性和接触收敛，不改变关节目标、
控制器命令空间或动作脚本 API。arm/hand 分类依赖资产命名约定，未知 prim 会跳过。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from linkerbot_sim.robots.classification import is_arm_name, is_hand_name


@dataclass(frozen=True)
class SolverIterationConfig:
    """PhysX solver type 和每组刚体的可选迭代次数覆盖。

    position/velocity iteration 分别对应 PhysX 位置约束和速度约束求解次数。数值越大通常
    越稳定但越慢，适合在绳体接触、指尖夹持等局部不稳定时按部件提高。字段为 ``None``
    表示不覆盖该 PhysX 属性。

    输入字段:
        solver_type: 可选 ``PGS`` 或 ``TGS``，写到 physics scene。
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


def solver_settings(env_config: dict) -> SolverIterationConfig | None:
    """从环境配置构造 PhysX solver 覆盖设置。

    配置语义是“写了哪个字段，就覆盖哪个字段”。未出现在 YAML 中的字段保持 Isaac/资产默认值，
    不再用代码默认值补齐后隐式覆盖。
    """

    solver = env_config.get("solver")
    if solver is None:
        return None
    if not isinstance(solver, Mapping):
        raise ValueError("solver config must be a mapping")
    config = SolverIterationConfig(
        solver_type=_optional_string(solver, "type"),
        arm_position_iterations=_optional_int(solver, "arm_position_iterations"),
        arm_velocity_iterations=_optional_int(solver, "arm_velocity_iterations"),
        hand_position_iterations=_optional_int(solver, "hand_position_iterations"),
        hand_velocity_iterations=_optional_int(solver, "hand_velocity_iterations"),
    )
    return config if _has_overrides(config) else None


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    text = str(value)
    if not text:
        raise ValueError(f"solver.{key} cannot be empty")
    return text


def _optional_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"solver.{key} must be non-negative")
    return parsed


def _has_overrides(config: SolverIterationConfig) -> bool:
    return config.solver_type is not None or _has_iteration_overrides(config)


def _has_iteration_overrides(config: SolverIterationConfig) -> bool:
    return any(
        getattr(config, field_name) is not None
        for field_name in _ITERATION_FIELD_NAMES
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
    name: str, config: SolverIterationConfig
) -> tuple[int | None, int | None, str] | None:
    """根据 prim 名和配置决定该刚体要写入的迭代次数。

    参数:
        name: USD prim 名称。
        config: solver 覆盖配置。
    返回:
        ``(position_iterations, velocity_iterations, group)``；该刚体没有任何字段需要覆盖时
        返回 ``None``。
    """

    if is_arm_prim_name(name):
        position_iterations = config.arm_position_iterations
        velocity_iterations = config.arm_velocity_iterations
        group = "arm"
    elif is_hand_prim_name(name):
        position_iterations = config.hand_position_iterations
        velocity_iterations = config.hand_velocity_iterations
        group = "hand"
    else:
        return None
    if position_iterations is None and velocity_iterations is None:
        return None
    return position_iterations, velocity_iterations, group


def apply_solver_iteration_overrides(
    stage, articulation_root_path: str, config: SolverIterationConfig
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
        solver_iterations = solver_iterations_for_prim_name(prim.GetName(), config)
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
