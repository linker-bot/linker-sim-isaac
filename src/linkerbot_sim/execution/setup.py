"""Isaac 执行侧机器人装配辅助工具。

这里封装动作脚本中重复的“导入机器人并创建控制器”流程，并明确分成 ``world.reset()`` 前后
两段：

* reset 前只能做 USD stage 级副作用，例如导入资产、应用 root pose、写入 USD/PhysX 覆盖、
  solver iteration 和刚体重力策略。
* reset 后 Isaac articulation view 才稳定，此时才能读取 DOF、清零速度、创建
  ``JointController`` 并写 runtime gain/mode。

保持这个边界能避免单臂/双臂脚本在导入顺序、重力策略或 controller 初始化时悄悄分叉。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linkerbot_sim.assets.robot_config import RobotGravityPolicy
from linkerbot_sim.assets.robot_import import import_robot_asset
from linkerbot_sim.assets.robot_instances import RobotExecutionConfig
from linkerbot_sim.assets.root_pose import apply_root_pose
from linkerbot_sim.assets.solver_overrides import (
    apply_solver_iteration_overrides,
    merge_solver_configs,
    scene_solver_settings,
)
from linkerbot_sim.assets.usd_overrides import (
    apply_robot_gravity_policy,
    apply_robot_usd_overrides,
)
from linkerbot_sim.controllers.config import (
    ControllerProfiles,
    joint_control_settings,
    physx_override_configs,
)
from linkerbot_sim.controllers.joint_controller import JointController
from linkerbot_sim.robots.classification import RobotComponentMapping


@dataclass(frozen=True)
class ImportedRobot:
    """reset 前已经导入 stage、但还没有创建控制器的机器人摘要。

    ``articulation`` 已经加入 ``world.scene``，但它的 DOF view 要等 ``world.reset()`` 后才可靠。
    ``solver_counts`` 和 ``gravity_counts`` 用于脚本打印诊断信息，确认 env/robot 配置实际命中。
    """

    articulation: object
    articulation_path: str
    imported_root_path: str
    asset_path: Path
    asset_type: str
    controlled_joints: tuple[str, ...]
    gravity_policy: RobotGravityPolicy
    component_mapping: RobotComponentMapping
    solver_counts: dict[str, int]
    gravity_counts: dict[str, int]

    @property
    def mimic_path(self) -> Path | None:
        """返回可由原生格式声明 follower 关系的资产路径。

        只有 MJCF 和 URDF 参与 mimic 解析；USD 的约束已经烘焙在 stage 中，不把资产路径
        交给基于 XML 的关系解析器。
        """

        return self.asset_path if self.asset_type in {"mjcf", "urdf"} else None


@dataclass(frozen=True)
class PreparedRobotRuntime:
    """reset 后可直接交给 execution step 使用的机器人运行时对象。"""

    articulation: object
    joint_controller: JointController
    asset_path: Path
    gravity_policy: RobotGravityPolicy


def import_execution_robot_to_stage(
    *,
    world: object,
    stage: object,
    single_articulation_type: object,
    robot_execution: RobotExecutionConfig,
    controller_profiles: ControllerProfiles,
    env_config: Mapping[str, object],
) -> ImportedRobot:
    """导入一个机器人 articulation，并写入 reset 前的 stage 级覆盖。

    参数中的 ``robot_execution`` 合并了 robot profile 的资产/物理属性和 env scene 的 root pose；
    robot profile 决定资产路径、受控关节选择、重力策略和刚体 solver iteration。
    env YAML 还提供世界级设置，例如 scene solver type。
    """

    articulation_path, asset_path, imported_root_path = import_robot_asset(
        robot_execution.robot
    )
    # root_pose 写在导入根 prim 上，保证 Isaac 执行模型和 cuRobo 双臂生成模型使用同一安装位姿。
    apply_root_pose(stage, imported_root_path, robot_execution.root_pose)
    controlled_joints = tuple(robot_execution.controlled_joints)
    # USD 层写入 joint friction 和 drive seed，并叠加 robot YAML 中的材料与刚体阻尼；
    # reset 后 controller 只更新运行时 gain、effort limit 和控制模式。
    physx_configs = robot_execution.robot.physx_overrides.apply_to_configs(
        physx_override_configs(controller_profiles)
    )
    apply_robot_usd_overrides(
        imported_root_path,
        physx_configs,
        driven_joint_names=controlled_joints,
        mjcf_path=asset_path if robot_execution.robot.asset_type == "mjcf" else None,
        mimic_path=(
            asset_path if robot_execution.robot.asset_type in {"mjcf", "urdf"} else None
        ),
        component_mapping=robot_execution.robot.component_mapping,
        native_mimic=robot_execution.robot.asset_type == "urdf",
    )
    solver_config = merge_solver_configs(
        scene_solver_settings(env_config),
        robot_execution.robot.solver_iterations,
    )
    solver_counts = (
        apply_solver_iteration_overrides(
            stage,
            articulation_path,
            solver_config,
            component_mapping=robot_execution.robot.component_mapping,
        )
        if solver_config is not None
        else {"configured": 0}
    )
    # 机器人重力策略只来自 robot YAML。这里写 USD disableGravity，reset 后还会按策略处理 runtime。
    gravity_policy = robot_execution.robot.gravity_policy
    gravity_counts = apply_robot_gravity_policy(
        imported_root_path,
        gravity_policy,
        component_mapping=robot_execution.robot.component_mapping,
    )
    articulation = world.scene.add(
        single_articulation_type(
            prim_path=articulation_path, name=robot_execution.robot.name
        )
    )
    return ImportedRobot(
        articulation=articulation,
        articulation_path=articulation_path,
        imported_root_path=imported_root_path,
        asset_path=asset_path,
        asset_type=robot_execution.robot.asset_type,
        controlled_joints=controlled_joints,
        gravity_policy=gravity_policy,
        component_mapping=robot_execution.robot.component_mapping,
        solver_counts=solver_counts,
        gravity_counts=gravity_counts,
    )


def finalize_robot_controller(
    *,
    imported: ImportedRobot,
    controller_profiles: ControllerProfiles,
    control_mode: str,
) -> PreparedRobotRuntime:
    """在 ``world.reset()`` 后创建并配置 ``JointController``。

    这个阶段可以安全读取 articulation 的 DOF 数量、名称和 runtime controller。若 robot YAML
    表示所有已知部件都关闭重力，则同步调用 Isaac runtime 的 ``disable_gravity()``，与 reset
    前写入的 USD ``disableGravity`` 保持一致。
    """

    gravity_policy = imported.gravity_policy
    if gravity_policy.disables_all_known_components():
        imported.articulation.disable_gravity()
    imported.articulation.set_joint_velocities(
        np.zeros(imported.articulation.num_dof, dtype=float)
    )
    controller = JointController(
        imported.articulation,
        joint_names=list(imported.controlled_joints),
        settings=joint_control_settings(controller_profiles, mode=control_mode),
        mimic_path=imported.mimic_path,
        component_mapping=imported.component_mapping,
        native_mimic=imported.asset_type == "urdf",
    )
    controller.configure_runtime()
    return PreparedRobotRuntime(
        articulation=imported.articulation,
        joint_controller=controller,
        asset_path=imported.asset_path,
        gravity_policy=gravity_policy,
    )
