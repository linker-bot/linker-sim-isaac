"""cuRobo profile 合并工具。

cuRobo 配置拆成两层：``configs/curobo/*.yaml`` 保存算法默认值，
``configs/robots/*.yaml`` 的 ``curobo.robot`` 保存机器人资源。后端入口只接受这一套
canonical cuRobo 配置语义。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from linkerbot_sim.backends.curobo.config import (
    CuroboConfig,
    CuroboDeviceConfig,
    CuroboIkConfig,
    CuroboMotionPlannerConfig,
    CuroboTaskBundle,
    DEFAULT_CUROBO_TASK_BUNDLE,
)
from linkerbot_sim.utils.config import deep_merge, load_yaml


_CUROBO_PROFILE_ROOT_KEYS = frozenset({"curobo"})
_CUROBO_ALGORITHM_KEYS = frozenset(
    {"task_bundle", "device", "kinematics", "motion_planner"}
)


def validate_curobo_profile(
    data: Mapping[str, Any], *, source: str = "<curobo profile>"
) -> dict[str, Any]:
    """严格校验项目侧当前 cuRobo 算法 profile。

    ``configs/curobo/task/**/*.yml`` 是受版本化 bundle 管理的第三方资源，不应传给
    本函数，也不使用项目 profile schema。
    """

    try:
        if not isinstance(data, Mapping):
            raise ValueError("cuRobo profile must be a mapping")
        canonical = dict(data)
        _reject_unknown_keys(canonical, _CUROBO_PROFILE_ROOT_KEYS, "profile")
        curobo = _required_mapping(canonical, "curobo", "profile")
        _reject_unknown_keys(curobo, _CUROBO_ALGORITHM_KEYS, "curobo")

        CuroboTaskBundle.named(curobo.get("task_bundle", DEFAULT_CUROBO_TASK_BUNDLE))
        CuroboDeviceConfig.from_mapping(_optional_mapping(curobo, "device", "curobo"))
        kinematics = _optional_mapping(curobo, "kinematics", "curobo")
        _reject_unknown_keys(kinematics, {"ik"}, "curobo.kinematics")
        CuroboIkConfig.from_mapping(
            _optional_mapping(kinematics, "ik", "curobo.kinematics")
        )
        CuroboMotionPlannerConfig.from_mapping(
            _optional_mapping(curobo, "motion_planner", "curobo")
        )
    except ValueError as exc:
        if str(exc).startswith(f"{source}:"):
            raise
        raise ValueError(f"{source}: {exc}") from exc
    return canonical


def load_curobo_profile(path: str | Path) -> dict[str, Any]:
    """从文件加载并严格校验项目侧 cuRobo 算法 profile。"""

    profile_path = Path(path)
    return validate_curobo_profile(load_yaml(profile_path), source=str(profile_path))


def _required_mapping(
    data: Mapping[str, Any], key: str, label: str
) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{key} must be a mapping")
    return value


def _optional_mapping(
    data: Mapping[str, Any], key: str, parent_label: str
) -> Mapping[str, Any]:
    if key not in data:
        return {}
    value = data[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{parent_label}.{key} must be a mapping")
    return value


def _reject_unknown_keys(
    data: Mapping[Any, Any], allowed: set[str] | frozenset[str], label: str
) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        paths = ", ".join(f"{label}.{key}" for key in unknown)
        raise ValueError(f"unsupported configuration field(s): {paths}")


def merged_robot_config_with_curobo_profile(
    robot_config: Mapping[str, Any],
    curobo_profile: Mapping[str, Any],
    *,
    profile_source: str = "<curobo profile>",
) -> dict[str, Any]:
    """把 cuRobo profile 默认值合入 robot 配置。

    合并优先级为 ``curobo profile < robot YAML``。profile 提供 batch size、collision cache、
    CUDA graph 等算法默认值；robot YAML 负责覆盖原生 ``curobo.robot`` 资源。
    """

    canonical_profile = validate_curobo_profile(curobo_profile, source=profile_source)
    profile_curobo = _required_mapping(canonical_profile, "curobo", "profile")
    return deep_merge({"curobo": dict(profile_curobo)}, dict(robot_config))


def robot_curobo_config(
    robot_config: Mapping[str, Any],
    *,
    curobo_profile: Mapping[str, Any] | None = None,
    robot_source: str = "<robot profile>",
    curobo_profile_source: str = "<curobo profile>",
) -> CuroboConfig:
    """解析机器人级 cuRobo 配置，并可选合入算法 profile。"""

    try:
        config = (
            dict(robot_config)
            if curobo_profile is None
            else merged_robot_config_with_curobo_profile(
                robot_config,
                curobo_profile,
                profile_source=curobo_profile_source,
            )
        )
        return CuroboConfig.from_mapping(config)
    except ValueError as exc:
        if str(exc).startswith(f"{robot_source}:"):
            raise
        raise ValueError(
            f"{robot_source} merged with {curobo_profile_source}: {exc}"
        ) from exc


__all__ = [
    "load_curobo_profile",
    "merged_robot_config_with_curobo_profile",
    "robot_curobo_config",
    "validate_curobo_profile",
]
