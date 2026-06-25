"""机器人刚体的 PhysX solver 迭代次数覆盖。

复杂接触、细小灵巧手和绳体交互都容易受 solver 迭代次数影响。
这里把迭代次数按 arm/hand 分组配置，便于在不同实验中快速调整稳定性。

这些覆盖发生在 USD/PhysX 属性层面，只影响求解稳定性和接触收敛，不改变关节目标、
控制器命令空间或任务 API。arm/hand 分类依赖资产命名约定，未知 prim 会跳过或使用默认值。
"""

from __future__ import annotations

from dataclasses import dataclass

from manipulation_project.robots.classification import is_arm_name, is_hand_name


@dataclass(frozen=True)
class SolverIterationConfig:
    """PhysX solver 类型和每组刚体的迭代次数。

    position/velocity iteration 分别对应 PhysX 位置约束和速度约束求解次数。数值越大通常
    越稳定但越慢，适合在绳体接触、指尖夹持等局部不稳定时按部件提高。

    输入字段:
        solver_type: ``PGS`` 或 ``TGS``。
        arm_position_iterations/arm_velocity_iterations: 机械臂刚体迭代次数。
        hand_position_iterations/hand_velocity_iterations: 灵巧手刚体迭代次数。
        apply_scope: 应用范围，支持 ``arm``、``hand``、``arm_hand``、``articulation``。
    输出:
        传给 ``apply_solver_iteration_overrides`` 后写入 stage。
    """

    solver_type: str = "TGS"
    arm_position_iterations: int = 32
    arm_velocity_iterations: int = 4
    hand_position_iterations: int = 32
    hand_velocity_iterations: int = 4
    apply_scope: str = "arm_hand"


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
) -> tuple[int, int, str] | None:
    """根据 prim 名和配置决定该刚体要写入的迭代次数。

    参数:
        name: USD prim 名称。
        config: solver 覆盖配置。
    返回:
        ``(position_iterations, velocity_iterations, group)``；不在应用范围内时返回 ``None``。
    """

    if config.apply_scope == "articulation":
        return (
            config.hand_position_iterations,
            config.hand_velocity_iterations,
            "articulation",
        )
    if config.apply_scope in {"arm", "arm_hand"} and is_arm_prim_name(name):
        return config.arm_position_iterations, config.arm_velocity_iterations, "arm"
    if config.apply_scope in {"hand", "arm_hand"} and is_hand_prim_name(name):
        return config.hand_position_iterations, config.hand_velocity_iterations, "hand"
    return None


def apply_solver_iteration_overrides(
    stage, articulation_root_path: str, config: SolverIterationConfig
) -> dict[str, int]:
    """写入 PhysX solver 类型和刚体迭代次数。

    参数:
        stage: 当前 USD stage。
        articulation_root_path: articulation root prim 路径。
        config: solver 迭代次数配置。
    返回:
        统计字典，记录写入的刚体数量、分组数量和 physics scene 数量。
    """

    from isaacsim.core.utils.prims import get_prim_at_path
    from pxr import PhysxSchema, Usd, UsdPhysics

    # solver 类型写在 physics scene 上，是全局物理求解策略；迭代次数写在刚体/关节树上，
    # 可以只提高关键部件的稳定性，避免整场景成本过高。
    solver_type = str(config.solver_type).upper()
    if solver_type not in {"PGS", "TGS"}:
        raise ValueError(f"Unsupported solver_type: {solver_type}")
    if config.apply_scope not in {"arm", "hand", "arm_hand", "articulation"}:
        raise ValueError(f"Unsupported solver apply_scope: {config.apply_scope}")

    physics_scene_prims = [
        prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)
    ]
    for scene_prim in physics_scene_prims:
        scene_api = (
            PhysxSchema.PhysxSceneAPI(scene_prim)
            if scene_prim.HasAPI(PhysxSchema.PhysxSceneAPI)
            else PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        )
        scene_api.CreateSolverTypeAttr().Set(solver_type)

    articulation_root = get_prim_at_path(articulation_root_path)
    if config.apply_scope == "articulation":
        # articulation 级覆盖适合不想依赖命名分类的资产；它会给整棵 articulation 使用同一组
        # 迭代次数。arm/hand 模式则继续在刚体级别细分。
        articulation_api = (
            PhysxSchema.PhysxArticulationAPI(articulation_root)
            if articulation_root.HasAPI(PhysxSchema.PhysxArticulationAPI)
            else PhysxSchema.PhysxArticulationAPI.Apply(articulation_root)
        )
        articulation_api.CreateSolverPositionIterationCountAttr().Set(
            config.hand_position_iterations
        )
        articulation_api.CreateSolverVelocityIterationCountAttr().Set(
            config.hand_velocity_iterations
        )

    counts = {
        "rigid_bodies": 0,
        "arm_rigid_bodies": 0,
        "hand_rigid_bodies": 0,
        "skipped_rigid_bodies": 0,
    }
    for prim in Usd.PrimRange(articulation_root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        # 通过命名约定判断 arm/hand，未知刚体不写入，避免把环境或装饰 prim 意外纳入高迭代设置。
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
        rigid_api.CreateSolverPositionIterationCountAttr().Set(int(position_iterations))
        rigid_api.CreateSolverVelocityIterationCountAttr().Set(int(velocity_iterations))
        counts["rigid_bodies"] += 1
        if group == "arm":
            counts["arm_rigid_bodies"] += 1
        elif group == "hand":
            counts["hand_rigid_bodies"] += 1
    counts["physics_scenes"] = len(physics_scene_prims)
    return counts
