"""cuRobo 后端入口。

当前包先提供可测试的配置、关节映射、轨迹适配和 tiled batch IK 适配骨架。真实 cuRobo
运行依赖 ``warp``、``cuda-core`` 等较重依赖，因此所有第三方导入都放到构造或调用边界，
避免普通单元测试因为环境未完整安装 cuRobo 而失败。
"""

from linkerbot_sim.backends.curobo.config import (
    CuroboConfig,
    CuroboDeviceConfig,
    CuroboIkConfig,
    CuroboMotionPlannerConfig,
    CuroboRobotConfig,
    CuroboTaskBundle,
    CuroboTcpFrame,
    SUPPORTED_CUROBO_DTYPES,
)
from linkerbot_sim.backends.curobo.collision_world import (
    CuroboCollisionWorld,
    make_curobo_scene_cfg,
)
from linkerbot_sim.backends.curobo.context import (
    CollisionCapability,
    CuroboContext,
)
from linkerbot_sim.backends.curobo.robot_model import (
    default_tcp_frame_name,
    materialize_curobo_config,
    resolve_curobo_cache_dir,
    resolve_tcp_frame_name,
)
from linkerbot_sim.backends.curobo.forward_kinematics import CuroboForwardKinematics
from linkerbot_sim.backends.curobo.runtime_imports import import_curobo_module
from linkerbot_sim.backends.curobo.inverse_kinematics import CuroboInverseKinematics
from linkerbot_sim.backends.curobo.joint_mapping import CuroboJointMapping
from linkerbot_sim.backends.curobo.linear_pose_path import plan_linear_pose_path
from linkerbot_sim.backends.curobo.motion_planner import CuroboMotionPlanner
from linkerbot_sim.backends.curobo.profile_merge import (
    load_curobo_profile,
    merged_robot_config_with_curobo_profile,
    robot_curobo_config,
    validate_curobo_profile,
)
from linkerbot_sim.backends.curobo.batch.ik import CuroboBatchIKSolver
from linkerbot_sim.backends.curobo.trajectory_adapter import (
    joint_trajectory_from_curobo,
)

__all__ = [
    "CuroboBatchIKSolver",
    "CuroboConfig",
    "CollisionCapability",
    "CuroboContext",
    "CuroboCollisionWorld",
    "CuroboDeviceConfig",
    "CuroboForwardKinematics",
    "CuroboIkConfig",
    "CuroboInverseKinematics",
    "CuroboJointMapping",
    "CuroboMotionPlanner",
    "CuroboMotionPlannerConfig",
    "CuroboRobotConfig",
    "CuroboTaskBundle",
    "CuroboTcpFrame",
    "SUPPORTED_CUROBO_DTYPES",
    "default_tcp_frame_name",
    "import_curobo_module",
    "joint_trajectory_from_curobo",
    "load_curobo_profile",
    "make_curobo_scene_cfg",
    "materialize_curobo_config",
    "resolve_curobo_cache_dir",
    "merged_robot_config_with_curobo_profile",
    "plan_linear_pose_path",
    "resolve_tcp_frame_name",
    "robot_curobo_config",
    "validate_curobo_profile",
]
