"""robot 资产 profile 与 cuRobo 数值 profile 的 typed composition。

配置 catalog 已分别把 ``configs/robots`` 和 ``configs/curobo`` 解析为冻结 dataclass。
本模块只做一次单向投影：注入 mode root 的 CUDA 设备并构造最终 ``CuroboConfig``；不会
还原 YAML mapping，也不会再次运行 backend mapping parser。
"""

from __future__ import annotations

from dataclasses import replace

from linkerbot_sim.backends.curobo.config import (
    CuroboConfig,
    CuroboDeviceConfig,
    CuroboIkConfig,
    CuroboMotionPlannerConfig,
)
from linkerbot_sim.configuration.curobo import CuroboProfileSettings
from linkerbot_sim.configuration.robots import RobotProfileSettings


def curobo_config_from_profiles(
    robot_profile: RobotProfileSettings,
    *,
    cuda_device: int,
    curobo_settings: CuroboProfileSettings | None = None,
) -> CuroboConfig:
    """把已解析 robot/model 与可选算法 profile 组合为后端配置。

    ``cuda_device`` 始终必填，避免诊断或测试路径重新引入隐式 ``cuda:0``。算法 profile
    缺省时使用后端 dataclass 的已验证默认值，适合只物化机器人模型的工具调用。
    """

    if not isinstance(robot_profile, RobotProfileSettings):
        raise TypeError("robot_profile must be RobotProfileSettings")
    if type(cuda_device) is not int or cuda_device < 0:
        raise ValueError("cuda_device must be a non-negative integer")
    if curobo_settings is not None and not isinstance(
        curobo_settings, CuroboProfileSettings
    ):
        raise TypeError("curobo_settings must be CuroboProfileSettings")

    binding = robot_profile.curobo.binding
    robot = robot_profile.curobo.robot
    if not binding.enabled or robot is None:
        raise ValueError(
            f"robot profile {robot_profile.name!r} does not enable a cuRobo model"
        )

    ik = CuroboIkConfig()
    planner = CuroboMotionPlannerConfig()
    if curobo_settings is not None:
        kinematics = curobo_settings.kinematics
        ik = replace(
            ik,
            num_seeds=kinematics.seed_count,
            seed_solver_num_seeds=kinematics.seed_count,
            max_batch_size=kinematics.max_batch_size,
            use_cuda_graph=kinematics.use_cuda_graph,
            self_collision_check=kinematics.collision_check,
            collision_cache=(
                kinematics.collision_cache.as_backend_mapping()
                if kinematics.collision_check and kinematics.collision_cache is not None
                else {}
            ),
        )
        settings = curobo_settings.motion_planner
        if settings is not None:
            planner = replace(
                planner,
                warmup=settings.warmup,
                num_ik_seeds=settings.ik_seed_count,
                num_trajopt_seeds=settings.trajectory_seed_count,
                use_cuda_graph=settings.use_cuda_graph,
                self_collision_check=settings.collision_check,
                collision_cache=(
                    settings.collision_cache.as_backend_mapping()
                    if settings.collision_check and settings.collision_cache is not None
                    else {}
                ),
            )

    config = CuroboConfig(
        robot=robot,
        device=CuroboDeviceConfig(device=f"cuda:{cuda_device}"),
        ik=ik,
        motion_planner=planner,
    )
    config.validate()
    return config


__all__ = ["curobo_config_from_profiles"]
