"""cuMotion profile 合并与解析工具。

本项目把 cuMotion 配置拆成两层：

* ``configs/cumotion/*.yaml`` 保存算法 profile，例如 IK 容差、planner pipeline 和后端参数。
* ``configs/robots/*.yaml`` 保存具体机器人资源，例如 XRDF/URDF 和 frame 名。
* ``configs/envs/*.yaml`` 保存 scene 中的机器人实例选择和 root pose。

动作脚本先按 env scene 选择机器人 profile，再把 cuMotion profile 默认值合到 robot 配置下面，
最后由这里解析成后端 dataclass。这样脚本只选择“使用哪个 scene”，不需要知道 cuMotion 配置内部层级。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from linkerbot_sim.backends.cumotion.context import CuMotionConfig
from linkerbot_sim.backends.cumotion.dual_urdf import (
    prepare_cumotion_config_from_robot_config,
)
from linkerbot_sim.backends.cumotion.motion_planner_config import (
    MotionPlannerBackendConfig,
)
from linkerbot_sim.utils.config import deep_merge


def merged_robot_config_with_cumotion_profile(
    robot_config: Mapping[str, Any], cumotion_profile: Mapping[str, Any]
) -> dict[str, Any]:
    """把 cuMotion profile 默认值合入 robot 配置。

    合并优先级为 ``cumotion profile < robot YAML``：profile 提供算法默认值，robot YAML 可以
    覆盖与具体机器人绑定的字段，例如 ``xrdf_path``、``urdf_path``、``flange_frame`` 或双臂
    资源分组。
    """

    profile_cumotion = cumotion_profile.get("cumotion")
    if profile_cumotion is None:
        return dict(robot_config)
    if not isinstance(profile_cumotion, Mapping):
        raise ValueError("cuMotion profile key 'cumotion' must be a mapping")
    return deep_merge({"cumotion": dict(profile_cumotion)}, dict(robot_config))


def robot_cumotion_config(
    robot_config: Mapping[str, Any],
    *,
    dual_root_poses: Mapping[str, object] | None = None,
) -> CuMotionConfig:
    """解析机器人级 cuMotion 资源；双臂配置会按 scene root pose 生成缓存资产。"""

    return prepare_cumotion_config_from_robot_config(
        robot_config,
        dual_root_poses=dual_root_poses,
    ).backend_config


def motion_planner_config_from_profile(
    cumotion_profile: Mapping[str, Any],
) -> MotionPlannerBackendConfig:
    """从 cuMotion profile 中解析 motion planner 默认配置。

    robot YAML 只负责机器人模型资源；planner pipeline、graph search、specified path 等算法
    默认值统一来自 profile，避免机器人配置文件混入动作/算法调参。
    """

    profile_cumotion = cumotion_profile.get("cumotion")
    if profile_cumotion is None:
        return MotionPlannerBackendConfig.from_mapping(None)
    if not isinstance(profile_cumotion, Mapping):
        raise ValueError("cuMotion profile key 'cumotion' must be a mapping")
    profile_motion_planner = profile_cumotion.get("motion_planner")
    if profile_motion_planner is not None and not isinstance(
        profile_motion_planner, Mapping
    ):
        raise ValueError(
            "cuMotion profile key 'cumotion.motion_planner' must be a mapping"
        )
    return MotionPlannerBackendConfig.from_mapping(profile_motion_planner)
