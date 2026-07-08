"""cuMotion 后端入口。

本子包把 NVIDIA cuMotion 的 robot description、FK、IK、碰撞世界、运动规划和轨迹输出转换
为项目内部的轻量数据结构。约定单位与 Isaac/URDF 保持一致：长度为 m、角度为 rad，四元数
在项目边界使用 ``wxyz`` 顺序。cuMotion 返回的 C-space 关节数组不会自动等同于 Isaac
articulation 的完整 DOF，调用方需要按关节名映射回完整 DOF 后再下发控制。

模块职责:
    * ``context``: 解析 ``CuMotionConfig``、加载 cuMotion Python 模块和 robot description，
      维护当前环境 collision world，并作为 FK/IK/planner 的共享工厂。
    * ``collision_world``: 把规划层 ``CollisionObject`` 转成 cuMotion ``World`` obstacle，
      维护 obstacle handle/world view，并提供 world/robot 碰撞诊断 inspector。当前项目适配
      基础 primitive（cuboid/sphere/capsule）；SDF grid 需要另行扩展数据结构和写入逻辑。
    * ``forward_kinematics``: 封装 cuMotion kinematics FK，把后端 ``Pose3`` 转为项目使用的
      position、wxyz quaternion 和 rotation matrix。
    * ``inverse_kinematics``: 把 ``IKRequest`` 转成几何 IK 或 collision-free IK 调用，并把
      cuMotion 结果归一化为 ``IKResult``。
    * ``motion_planner``: 作为 facade 按配置分发到 trajectory optimization、graph search 或
      specified path pipeline，并统一返回 ``MotionResult``。
    * ``pose_adapter``: 在项目的 4x4 pose / position+quaternion 表示和 cuMotion ``Pose3`` 之间
      做边界转换。
    * ``path_spec_adapter`` / ``specified_path_planner``: 通过 cuMotion 官方 PathSpec API 处理
      调用方指定的 C-space/task-space/composite 路径。
    * ``trajectory_sampler``: 把 cuMotion time-parameterized trajectory 采样成项目
      ``JointTrajectory``，统一关节名、时间、位置、速度和 effort 数组。
    * ``tiled_ik``: 封装 cuMotion ``CollisionFreeIkSolver.solve_array``，为 tiled
      command-step runtime 提供真正 batch IK，不走 per-env planner/IK loop。
    * ``tcp_urdf_builder``: 根据 robot YAML 中的 ``custom_tcps`` 生成带 fixed TCP link/frame
      的派生 URDF，让 cuMotion 能直接以自定义 TCP frame 做 FK/IK。

入口文件只重新导出项目常用封装类和函数，不在导入时加载机器人模型。由于 cuMotion 可能只在
Isaac 环境中安装，具体模块会尽量在构造 ``CuMotionContext`` 时再导入第三方库，而不是在包
导入阶段失败。
"""

# collision_world.py: 负责把规划层 CollisionObject 转成 cuMotion World obstacle，
# 并提供 world/robot 碰撞距离查询和诊断 inspector。
from linkerbot_sim.backends.cumotion.collision_world import (
    CuMotionCollisionWorld,
    CuMotionRobotWorldInspector,
    CuMotionWorldInspector,
    make_collision_world,
)

# context.py: 负责加载 cuMotion、机器人描述、共享 kinematics 和当前环境 collision world，
# 并作为 FK、IK、MotionPlanner 等后端对象的工厂入口。
from linkerbot_sim.backends.cumotion.context import (
    CuMotionConfig,
    CuMotionContext,
)
# dual_urdf.py: 从双臂 robot YAML 生成运行时 cuMotion URDF/XRDF，并返回最终后端配置。
from linkerbot_sim.backends.cumotion.dual_urdf import (
    dual_cumotion_config_from_sides,
    prepare_cumotion_config_from_robot_config,
)

# forward_kinematics.py: 封装 cuMotion FK，把后端 Pose3 转成项目统一的
# position、wxyz quaternion 和 rotation matrix。
from linkerbot_sim.backends.cumotion.forward_kinematics import (
    CuMotionForwardKinematics,
)

# inverse_kinematics.py: 封装几何 IK 和 collision-free IK，
# 把 IKRequest 转成 cuMotion 调用并返回项目 IKResult。
from linkerbot_sim.backends.cumotion.inverse_kinematics import (
    CuMotionInverseKinematics,
)

# motion_planner.py: motion planning facade，按 MotionPlannerBackendConfig 分发到
# trajectory_optimization / graph_search / specified_path。
from linkerbot_sim.backends.cumotion.motion_planner import (
    CuMotionMotionPlanner,
)
# motion_planner_config.py: motion planner facade 使用的分组配置模型。
from linkerbot_sim.backends.cumotion.motion_planner_config import (
    GraphSearchConfig,
    MotionPlannerBackendConfig,
    SpecifiedPathConfig,
    TrajectoryGenerationConfig,
    TrajectoryOptimizationConfig,
)

# trajectory_sampler.py: 把 cuMotion time-parameterized trajectory 采样并转换成
# 项目 JointTrajectory。
from linkerbot_sim.backends.cumotion.trajectory_sampler import (
    joint_trajectory_from_cumotion,
)
# tiled_ik.py: tiled runtime 的 cuMotion batch IK 适配，放在 backend 层而不是 tiled 核心层。
from linkerbot_sim.backends.cumotion.tiled_ik import (
    BatchedCuMotionIKSolver,
    CuMotionJointMapping,
)
# tiled_planner.py: tiled async planner 的 cuMotion adapter，和 tiled 核心 manager 解耦。
from linkerbot_sim.backends.cumotion.tiled_planner import CuMotionJointPlannerBackend

__all__ = [
    "BatchedCuMotionIKSolver",
    "CuMotionJointPlannerBackend",
    "CuMotionJointMapping",
    "CuMotionConfig",
    "CuMotionContext",
    "CuMotionCollisionWorld",
    "CuMotionForwardKinematics",
    "CuMotionInverseKinematics",
    "CuMotionMotionPlanner",
    "CuMotionRobotWorldInspector",
    "CuMotionWorldInspector",
    "GraphSearchConfig",
    "MotionPlannerBackendConfig",
    "SpecifiedPathConfig",
    "TrajectoryGenerationConfig",
    "TrajectoryOptimizationConfig",
    "dual_cumotion_config_from_sides",
    "joint_trajectory_from_cumotion",
    "make_collision_world",
    "prepare_cumotion_config_from_robot_config",
]
