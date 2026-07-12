"""运行时 profile 及其完整配置依赖图的纯 Python 校验。

本模块从已解析的运行时配置出发，加载环境引用的机器人、对象、控制器、cuRobo 和日志
profile，验证各层 schema、能力绑定及 USD prim 路径所有权。它是启动前的组合校验边界：
会读取项目配置文件，但不会导入 Isaac/Omni、创建 stage、实例化控制器或启动规划后端。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from linkerbot_sim.assets.robot_instances import (
    RobotExecutionConfig,
    resolve_controller_profile,
    robot_instances_from_env_config,
)
from linkerbot_sim.assets.robot_config import load_robot_profile
from linkerbot_sim.backends.curobo.profile_merge import (
    merged_robot_config_with_curobo_profile,
    robot_curobo_config,
    validate_curobo_profile,
)
from linkerbot_sim.configs.instance_paths import validate_disjoint_instance_prim_paths
from linkerbot_sim.configs.profiles import profile_path
from linkerbot_sim.configs.runtime import ResolvedRuntimeConfig, RuntimeProfileConfig
from linkerbot_sim.controllers.config import load_controller_bundle
from linkerbot_sim.logging.config import load_joint_logging_profile
from linkerbot_sim.objects.config import (
    ObjectProfileConfig,
    object_scene_instances_from_env_config,
)
from linkerbot_sim.robots.capabilities import (
    PlanningBindingConfig,
    robot_kind_from_profile,
)
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.utils.config import load_yaml


@dataclass(frozen=True)
class ValidatedProfileGraph:
    """通过完整依赖校验的运行时配置快照。

    ``runtime_profile`` 是入口 profile 名，``profile`` 保留严格解析后的原 profile，
    ``resolved`` 是合并默认值和覆盖后的最终配置。``dependencies`` 按配置分组保存已验证的
    稳定名称 tuple，并通过只读 mapping 暴露；对象本身不持有已加载后端或文件句柄。
    """

    runtime_profile: str
    profile: RuntimeProfileConfig
    resolved: ResolvedRuntimeConfig
    dependencies: Mapping[str, tuple[str, ...]]


def validate_profile_graph(
    *,
    runtime_profile: str,
    profile: RuntimeProfileConfig,
    resolved: ResolvedRuntimeConfig,
    env_config: Mapping[str, object],
) -> ValidatedProfileGraph:
    """校验一份最终运行时配置能够到达的全部 profile。

    参数:
        runtime_profile: 当前运行时 profile 的稳定名称，用于结果追踪。
        profile: YAML 严格解析后的原始运行时 profile。
        resolved: 已完成默认值和覆盖合并的最终运行时配置。
        env_config: ``resolved.profiles.env`` 对应的完整环境 mapping。
    返回:
        冻结的 :class:`ValidatedProfileGraph`，依赖名称已排序以保证结果稳定。
    异常:
        FileNotFoundError: 任一引用 profile 或资产配置文件不存在。
        TypeError: 某层配置不是声明的结构。
        ValueError: 模式、schema、机器人能力、控制器绑定或 prim 路径不合法。
    副作用:
        读取并解析依赖 YAML；不启动仿真、规划线程或日志输出。
    """

    tiled = TiledEnvConfig.from_env_config(env_config)
    expected_tiled = resolved.mode == "tiled_scene"
    if tiled.enabled != expected_tiled:
        expected = "tiled.enabled=true" if expected_tiled else "tiled.enabled=false"
        raise ValueError(
            f"runtime.mode={resolved.mode!r} requires selected env {expected}"
        )

    robot_instances = robot_instances_from_env_config(env_config)
    robot_profiles: dict[str, Mapping[str, object]] = {}
    planning_robot_names: set[str] = set()
    controller_bundles: set[str] = set()

    curobo_name = resolved.profiles.curobo
    curobo_path = profile_path("curobo", curobo_name)
    curobo_profile = validate_curobo_profile(
        load_yaml(curobo_path), source=str(curobo_path)
    )
    # 同一机器人 profile 可能被多个场景实例复用。按名称缓存可确保只校验、合并一次，
    # 同时仍对每个实例单独解析 prim path 和 controller 归属。
    for instance in robot_instances:
        robot_name = instance.robot_profile
        robot_data = robot_profiles.get(robot_name)
        if robot_data is None:
            robot_path = profile_path("robot", robot_name)
            robot_data = load_robot_profile(robot_path)
            robot_profiles[robot_name] = robot_data
            kind = robot_kind_from_profile(robot_data)
            binding = PlanningBindingConfig.from_profile(robot_data, kind=kind)
            if binding.enabled:
                robot_curobo_config(
                    robot_data,
                    curobo_profile=curobo_profile,
                    robot_source=str(robot_path),
                    curobo_profile_source=str(curobo_path),
                )
                planning_robot_names.add(robot_name)
    merged_robot_profiles = {
        name: merged_robot_config_with_curobo_profile(
            data,
            curobo_profile,
            profile_source=str(curobo_path),
        )
        for name, data in robot_profiles.items()
    }
    robot_paths: dict[str, str] = {}
    for instance in robot_instances:
        robot_data = merged_robot_profiles[instance.robot_profile]
        execution = RobotExecutionConfig.from_mapping(
            robot_data,
            scene_instance=instance,
        )
        robot_paths[instance.label] = execution.robot.prim_path
        controller_bundles.add(
            resolve_controller_profile(
                instance,
                execution.robot,
                resolved.profiles.controller_bundle,
            )
        )

    object_instances = object_scene_instances_from_env_config(env_config)
    object_names = {instance.object_profile for instance in object_instances}
    object_profiles: dict[str, ObjectProfileConfig] = {}
    for object_name in sorted(object_names):
        object_profiles[object_name] = ObjectProfileConfig.from_profile(object_name)
    # 在创建 stage 前统一检查机器人和对象的 USD 子树所有权，避免后加载实例覆盖先前 prim。
    validate_disjoint_instance_prim_paths(
        robot_paths=robot_paths,
        object_paths={
            instance.name: instance.effective_prim_path for instance in object_instances
        },
    )

    for bundle_name in sorted(controller_bundles):
        load_controller_bundle(bundle_name)

    logging_name = resolved.profiles.logging
    load_joint_logging_profile(logging_name)

    dependencies = MappingProxyType(
        {
            "runtime": (runtime_profile,),
            "env": (resolved.profiles.env,),
            "robot": tuple(sorted(robot_profiles)),
            "planning_robot": tuple(sorted(planning_robot_names)),
            "object": tuple(sorted(object_names)),
            "controller": tuple(sorted(controller_bundles)),
            "curobo": (curobo_name,),
            "logging": (logging_name,),
        }
    )
    return ValidatedProfileGraph(
        runtime_profile=runtime_profile,
        profile=profile,
        resolved=resolved,
        dependencies=dependencies,
    )


__all__ = ["ValidatedProfileGraph", "validate_profile_graph"]
