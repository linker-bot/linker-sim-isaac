"""cuMotion 后端入口。

本子包把 NVIDIA cuMotion 的 robot description、FK、IK、碰撞世界、运动规划和轨迹输出转换
为项目内部的轻量数据结构。约定单位与 Isaac/URDF 保持一致：长度为 m、角度为 rad，四元数
在项目边界使用 ``wxyz`` 顺序。cuMotion 返回的 C-space 关节数组不会自动等同于 Isaac
articulation 的完整 DOF，调用方需要按关节名映射回完整 DOF 后再下发控制。

模块职责:
    * ``context``: 解析 ``CuMotionConfig``、加载 cuMotion Python 模块和 robot description，
      并作为 FK/IK/planner 的共享工厂，避免重复解析 URDF/YAML。
    * ``collision_world``: 把规划层 ``CollisionObject`` 转成 cuMotion ``World`` obstacle，
      维护 obstacle handle/world view，并提供 world/robot 碰撞诊断 inspector。当前项目适配
      基础 primitive（cuboid/sphere/capsule）；SDF grid 需要另行扩展数据结构和写入逻辑。
    * ``forward_kinematics``: 封装 cuMotion kinematics FK，把后端 ``Pose3`` 转为项目使用的
      position、wxyz quaternion 和 rotation matrix。
    * ``inverse_kinematics``: 把 ``IKRequest`` 转成几何 IK 或 collision-free IK 调用，并把
      cuMotion 结果归一化为 ``IKResult``。
    * ``motion_planner``: 封装 cuMotion ``MotionPlanner``，处理 planner 参数、轨迹生成参数、
      C-space path/trajectory 输出和 ``MotionResult`` 诊断。
    * ``pose_adapter``: 在项目的 4x4 pose / position+quaternion 表示和 cuMotion ``Pose3`` 之间
      做边界转换。
    * ``trajectory_adapter``: 把 cuMotion time-parameterized trajectory 采样成项目
      ``JointTrajectory``，统一关节名、时间、位置、速度和 effort 数组。
    * ``tcp_line``: 生成直线 TCP waypoint，并通过 IK 串接成 C-space 关节路径；适合简单笛卡尔
      直线移动，不做全局避障路径规划。
    * ``tcp_urdf_builder``: 在临时 URDF 中追加 fixed TCP link/frame，让 cuMotion 能直接以自定义
      TCP frame 做 FK/IK。

入口文件只重新导出项目常用封装类和函数，不在导入时加载机器人模型。由于 cuMotion 可能只在
Isaac 环境中安装，具体模块会尽量在构造 ``CuMotionContext`` 时再导入第三方库，而不是在包
导入阶段失败。
"""

# collision_world.py: 负责把规划层 CollisionObject 转成 cuMotion World obstacle，
# 并提供 world/robot 碰撞距离查询和诊断 inspector。
from manipulation_project.backends.cumotion.collision_world import (
    CuMotionCollisionWorld,
    CuMotionRobotWorldInspector,
    CuMotionWorldInspector,
    make_collision_world,
)

# context.py: 负责加载 cuMotion、机器人描述和共享 kinematics，
# 并作为 FK、IK、MotionPlanner 等后端对象的工厂入口。
from manipulation_project.backends.cumotion.context import (
    CuMotionConfig,
    CuMotionContext,
)

# forward_kinematics.py: 封装 cuMotion FK，把后端 Pose3 转成项目统一的
# position、wxyz quaternion 和 rotation matrix。
from manipulation_project.backends.cumotion.forward_kinematics import (
    CuMotionForwardKinematics,
)

# inverse_kinematics.py: 封装几何 IK 和 collision-free IK，
# 把 IKRequest 转成 cuMotion 调用并返回项目 IKResult。
from manipulation_project.backends.cumotion.inverse_kinematics import (
    CuMotionInverseKinematics,
)

# motion_planner.py: 封装 cuMotion MotionPlanner，负责 C-space 路径规划、
# 可选时间参数化轨迹生成和 MotionResult 诊断整理。
from manipulation_project.backends.cumotion.motion_planner import (
    CuMotionMotionPlanner,
)

# tcp_line.py: 根据直线 TCP waypoint 串联 IK 解，生成简单笛卡尔直线移动的
# C-space 关节路径。
from manipulation_project.backends.cumotion.tcp_line import (
    plan_tcp_line_joint_path,
)

# trajectory_adapter.py: 把 cuMotion time-parameterized trajectory 采样并转换成
# 项目 JointTrajectory。
from manipulation_project.backends.cumotion.trajectory_adapter import (
    joint_trajectory_from_cumotion,
)

__all__ = [
    "CuMotionConfig",
    "CuMotionContext",
    "CuMotionCollisionWorld",
    "CuMotionForwardKinematics",
    "CuMotionInverseKinematics",
    "CuMotionMotionPlanner",
    "CuMotionRobotWorldInspector",
    "CuMotionWorldInspector",
    "joint_trajectory_from_cumotion",
    "make_collision_world",
    "plan_tcp_line_joint_path",
]
