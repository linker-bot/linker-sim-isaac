"""环境 profile 与逐环境覆盖片段的严格纯 Python 校验。

本模块是 ``configs/envs`` 的 schema 边界：检查环境基础参数、solver、视觉与传感器设置、
机器人/对象实例、tiled 布局以及逐环境引用的一致性。校验会复用各领域的生产解析器，确保
预检和实际启动采用相同语义；返回值与原 mapping 脱离，调用方可以安全继续合并。

这里不会导入 Isaac/Omni、创建 stage 或加载资产。跨 profile 的控制器、规划器及 prim 路径
所有权由更高层的完整依赖图校验负责。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from linkerbot_sim.envs.settings import EnvRuntimeSettings
from linkerbot_sim.assets.robot_instances import robot_instances_from_env_config
from linkerbot_sim.assets.solver_overrides import scene_solver_settings
from linkerbot_sim.objects.config import object_scene_instances_from_env_config
from linkerbot_sim.tiled.config import TiledEnvConfig
from linkerbot_sim.tiled.scene.cameras import tiled_sensor_camera_settings


ENV_TOP_LEVEL_KEYS = frozenset(
    {
        "env",
        "solver",
        "visuals",
        "sensors",
        "robots",
        "objects",
        "tiled",
    }
)


def validate_env_profile(
    data: Mapping[str, object],
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """校验一份完整环境 profile，并返回脱离输入的 mapping。

    参数:
        data: 顶层环境配置，只接受当前 schema 声明的字段。
        source_path: 可选来源路径，用于在结构错误前附加定位信息。
    返回:
        递归复制后的普通字典；嵌套 mapping 和序列均不与输入共享可变容器。
    异常:
        ValueError: 顶层结构、字段类型/范围、实例姿态、tiled 作用域或引用不合法。
    副作用:
        无；不会读取资产、创建场景或修改传入 mapping。
    """

    source = "<mapping>" if source_path is None else str(source_path)
    if not isinstance(data, Mapping):
        raise ValueError(f"{source}: env profile must be a mapping")
    canonical = _copy_mapping(data)
    _validate_env_mapping(canonical)
    return canonical


def validate_per_env_fragment(
    data: Mapping[str, object],
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """校验目录式 tiled 环境中的单个逐环境覆盖片段。

    参数:
        data: 包含必需 ``env_id`` 以及可选机器人、对象、相机和 metadata 覆盖的 mapping。
        source_path: 可选来源路径，用于给生产解析器抛出的错误补充文件位置。
    返回:
        与输入脱离、仍保持 YAML payload 结构的字典。
    异常:
        ValueError: 缺少 ``env_id``、出现未知字段，或任一嵌套覆盖无法由
            ``TiledPerEnvConfig`` 严格解析。
    副作用:
        无；只构造临时解析对象进行验证。
    """

    source = "<mapping>" if source_path is None else str(source_path)
    try:
        if not isinstance(data, Mapping):
            raise ValueError("per-env profile must be a mapping")
        canonical = dict(data)
        _reject_keys(
            canonical,
            {"env_id", "robots", "objects", "cameras", "metadata"},
            "per-env profile",
        )

        payload = _copy_mapping(canonical)
        if "env_id" not in payload:
            raise ValueError("per-env profile.env_id is required")

        # 复用实际运行时解析器，避免预检单独维护一份容易漂移的逐环境字段规则。
        from linkerbot_sim.tiled.config import TiledPerEnvConfig

        TiledPerEnvConfig.from_mapping(payload, index=0)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc
    return payload


def _validate_env_mapping(
    data: Mapping[str, object],
) -> None:
    """按领域解析器校验完整环境，并执行 tiled 依赖的组合约束。"""

    _reject_keys(data, set(ENV_TOP_LEVEL_KEYS), "env profile")

    env = _required_mapping(data, "env", "env profile")
    _reject_keys(
        env,
        {
            "name",
            "description",
            "gravity_z",
            "add_ground",
            "ground_height",
            "physics_frequency",
            "render_frequency",
        },
        "env",
    )
    _validate_env_fields(env)

    settings = EnvRuntimeSettings.from_env_config(data)
    solver_data = data.get("solver")
    if solver_data is not None:
        if not isinstance(solver_data, Mapping):
            raise ValueError("solver must be a mapping")
        _reject_keys(solver_data, {"type"}, "solver")
    solver = scene_solver_settings(data)
    if solver is not None and solver.solver_type is not None:
        solver_type = solver.solver_type.upper()
        if solver_type not in {"PGS", "TGS"}:
            raise ValueError("solver.type must be one of PGS or TGS")

    _validate_instance_pose_paths(data)
    robots = robot_instances_from_env_config(data)
    objects = object_scene_instances_from_env_config(data)

    tiled = TiledEnvConfig.from_env_config(data)
    if tiled.enabled:
        tiled_sensor_camera_settings(
            settings.sensors,
            tiled_config=tiled,
        )
        _validate_per_env_references(
            tiled,
            robot_labels={item.label for item in robots},
            object_names={item.name for item in objects},
            camera_names={item.name for item in settings.sensors.cameras},
        )
    else:
        settings.sensors.validate_single_scene_camera_scope()
        if tiled.per_env:
            raise ValueError("tiled.per_env requires tiled.enabled=true")


def _validate_env_fields(env: Mapping[str, object]) -> None:
    name = env.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("env.name must be a non-empty string")
    if "description" in env and not isinstance(env["description"], str):
        raise ValueError("env.description must be a string")
    if "add_ground" in env and not isinstance(env["add_ground"], bool):
        raise ValueError("env.add_ground must be a boolean")
    for key in ("gravity_z", "ground_height"):
        if key in env:
            _finite_number(env[key], f"env.{key}")
    for key in ("physics_frequency", "render_frequency"):
        if key in env and _finite_number(env[key], f"env.{key}") <= 0.0:
            raise ValueError(f"env.{key} must be positive")


def _validate_instance_pose_paths(data: Mapping[str, object]) -> None:
    """校验实例列表只含当前字段，并保证每个根姿态是有限三维向量。"""

    robots = data.get("robots")
    if not isinstance(robots, Sequence) or isinstance(robots, (str, bytes)):
        raise ValueError("env profile.robots must be a sequence")
    for index, item in enumerate(robots):
        if not isinstance(item, Mapping):
            raise ValueError(f"robots[{index}] must be a mapping")
        _reject_keys(
            item,
            {
                "robot_profile",
                "root_pose",
                "label",
                "prim_path",
                "controller_profile",
            },
            f"robots[{index}]",
        )
        _validate_pose(item.get("root_pose"), f"robots[{index}].root_pose")

    objects = data.get("objects", ())
    if objects is None:
        return
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        raise ValueError("env profile.objects must be a sequence")
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise ValueError(f"objects[{index}] must be a mapping")
        _reject_keys(
            item,
            {"name", "object_profile", "runtime_handle", "prim_path", "root_pose"},
            f"objects[{index}]",
        )
        _validate_pose(item.get("root_pose"), f"objects[{index}].root_pose")


def _validate_pose(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    _reject_keys(value, {"xyz", "rpy"}, label)
    for key in ("xyz", "rpy"):
        vector = value.get(key, (0.0, 0.0, 0.0))
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise ValueError(f"{label}.{key} must be a length-3 sequence")
        if len(vector) != 3:
            raise ValueError(f"{label}.{key} must contain exactly 3 values")
        for index, item in enumerate(vector):
            _finite_number(item, f"{label}.{key}[{index}]")


def _validate_per_env_references(
    tiled: TiledEnvConfig,
    *,
    robot_labels: set[str],
    object_names: set[str],
    camera_names: set[str],
) -> None:
    """确保逐环境覆盖只引用基础场景中已声明的实例和相机。"""

    for index, item in enumerate(tiled.per_env):
        for configured, known, group in (
            (set(item.robot_root_poses), robot_labels, "robots"),
            (set(item.object_root_poses), object_names, "objects"),
            (set(item.camera_poses), camera_names, "cameras"),
        ):
            unknown = sorted(configured - known)
            if unknown:
                raise ValueError(
                    f"tiled.per_env[{index}].{group}.{unknown[0]} references an "
                    f"unknown {group[:-1]}"
                )


def _required_mapping(
    data: Mapping[str, object], key: str, label: str
) -> Mapping[str, object]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{key} must be a mapping")
    return value


def _optional_mapping(
    value: object, key: str, label: str
) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    nested = value.get(key)
    if nested is None:
        return None
    if not isinstance(nested, Mapping):
        raise ValueError(f"{label}.{key} must be a mapping")
    return nested


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _reject_keys(data: Mapping[str, object], allowed: set[str], label: str) -> None:
    unsupported = sorted(str(key) for key in data if key not in allowed)
    if unsupported:
        raise ValueError(f"{label}.{unsupported[0]} is not supported")


def _copy_mapping(data: Mapping[str, object]) -> dict[str, Any]:
    """递归复制 YAML mapping，并把 tuple 统一为可变 list payload。"""

    return {str(key): _copy_value(value) for key, value in data.items()}


def _copy_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return _copy_mapping(value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_value(item) for item in value]
    return value


__all__ = [
    "validate_env_profile",
    "validate_per_env_fragment",
]
