"""cuMotion 后端入口。

本子包把 NVIDIA cuMotion 的 robot description、FK、IK、碰撞世界和轨迹输出转换为项目内部
的轻量数据结构。约定单位与 Isaac/URDF 保持一致：长度为 m、角度为 rad，四元数在项目边界
使用 ``wxyz`` 顺序。

入口文件只导出项目封装类，不在导入时加载机器人模型。由于 cuMotion 可能只在 Isaac 环境中
安装，具体模块会尽量在构造 ``CuMotionContext`` 时再导入第三方库，而不是在包导入阶段失败。
调用方拿到后端结果后仍需按关节名映射回 Isaac articulation 的完整 DOF 顺序。
"""

from manipulation_project.backends.cumotion.context import CuMotionConfig, CuMotionContext
from manipulation_project.backends.cumotion.forward_kinematics import CuMotionForwardKinematics
from manipulation_project.backends.cumotion.inverse_kinematics import CuMotionInverseKinematics
from manipulation_project.backends.cumotion.trajectory_adapter import joint_trajectory_from_cumotion

__all__ = [
    "CuMotionConfig",
    "CuMotionContext",
    "CuMotionForwardKinematics",
    "CuMotionInverseKinematics",
    "joint_trajectory_from_cumotion",
]
