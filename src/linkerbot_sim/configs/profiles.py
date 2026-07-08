"""按稳定 profile 名称加载仓库内配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from linkerbot_sim.controllers.config import load_controller_profiles
from linkerbot_sim.utils.config import deep_merge, load_yaml
from linkerbot_sim.utils.paths import CONFIGS_ROOT


PROFILE_GROUP_DIRS = {
    "robot": CONFIGS_ROOT / "robots",
    "env": CONFIGS_ROOT / "envs",
    "object": CONFIGS_ROOT / "objects",
    "cumotion": CONFIGS_ROOT / "cumotion",
    "logging": CONFIGS_ROOT / "logging",
}


def profile_path(group: str, name: str) -> Path:
    """把 ``group/name`` 解析成仓库内 YAML profile 路径。"""

    if group not in PROFILE_GROUP_DIRS:
        raise ValueError(f"Unknown config profile group: {group!r}")
    _validate_profile_name(name)
    if group == "env":
        env_file = PROFILE_GROUP_DIRS[group] / f"{name}.yaml"
        env_dir_base = PROFILE_GROUP_DIRS[group] / name / "base.yaml"
        if env_dir_base.is_file() and not env_file.is_file():
            return env_dir_base
    return PROFILE_GROUP_DIRS[group] / f"{name}.yaml"


def load_profile_yaml(group: str, name: str) -> dict[str, Any]:
    """读取仓库内指定 profile YAML。"""

    if group == "env":
        return load_env_profile_yaml(name)
    return load_yaml(profile_path(group, name))


def load_env_profile_yaml(name: str) -> dict[str, Any]:
    """读取 env profile，兼容 ``scene.yaml`` 和目录型 ``scene/base.yaml``。

    目录型 env profile 用于 tiled envs:

    ``configs/envs/<name>/base.yaml``:
        保存机器人、相机、灯光、solver、共享对象集合等共通配置。
    ``configs/envs/<name>/<per_env_config_dir>/*.yaml``:
        保存每个 env 对已有对象的局部位姿覆盖。
    """

    _validate_profile_name(name)
    env_root = PROFILE_GROUP_DIRS["env"]
    yaml_path = env_root / f"{name}.yaml"
    if yaml_path.is_file():
        return load_yaml(yaml_path)

    profile_dir = env_root / name
    if profile_dir.is_dir():
        return load_env_profile_directory(profile_dir)

    # 保持旧错误形态：普通文件不存在时由 load_yaml 抛 FileNotFoundError。
    return load_yaml(yaml_path)


def load_env_profile_directory(profile_dir: Path) -> dict[str, Any]:
    """读取目录型 env profile 并把 per-env 文件合并到 ``tiled.per_env``。"""

    base_path = profile_dir / "base.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(f"Env profile directory missing base.yaml: {profile_dir}")
    base = load_yaml(base_path)
    tiled = base.get("tiled")
    if not isinstance(tiled, dict):
        raise ValueError(f"{base_path} must contain top-level tiled mapping")

    per_env_dir_name = _relative_dir_name(
        tiled.get("per_env_config_dir", "envs"),
        label=f"{base_path}:tiled.per_env_config_dir",
    )
    per_env_dir = profile_dir / per_env_dir_name
    if not per_env_dir.is_dir():
        if "per_env_config_dir" in tiled:
            raise FileNotFoundError(f"tiled per-env directory not found: {per_env_dir}")
        per_env_items: list[dict[str, Any]] = []
    else:
        per_env_items = [
            _load_per_env_yaml(path, index=index)
            for index, path in enumerate(sorted(per_env_dir.glob("*.yaml")))
        ]

    tiled_overlay: dict[str, Any] = {
        "enabled": True,
        "per_env_config_dir": per_env_dir_name,
        "per_env": sorted(per_env_items, key=lambda item: int(item["env_id"])),
    }
    if per_env_items and "num_envs" not in tiled:
        tiled_overlay["num_envs"] = max(int(item["env_id"]) for item in per_env_items) + 1
    return deep_merge(base, {"tiled": tiled_overlay})


def load_default_controller_profiles():
    """读取项目默认 controller 配置集合。"""

    return load_controller_profiles(CONFIGS_ROOT / "controllers")


def _load_per_env_yaml(path: Path, *, index: int) -> dict[str, Any]:
    """读取单个 per-env YAML；缺省 env_id 时从文件名推导。"""

    data = load_yaml(path)
    if "env_id" not in data:
        data["env_id"] = _env_id_from_stem(path.stem, index=index)
    return data


def _env_id_from_stem(stem: str, *, index: int) -> int:
    """从 ``env_000`` / ``000`` 文件名推导 env id。"""

    token = stem
    if token.startswith("env_"):
        token = token.removeprefix("env_")
    if token.isdigit():
        return int(token)
    raise ValueError(
        f"per-env YAML missing env_id and file name does not encode one: {stem!r} "
        f"(index {index})"
    )


def _relative_dir_name(value: object, *, label: str) -> str:
    """校验 profile 内相对目录名。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    normalized = value.replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part)
    if normalized.startswith("/") or not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"{label} must be a relative directory")
    return "/".join(parts)


def _validate_profile_name(name: str) -> None:
    """限制 profile 名称为单个 stem，避免从配置入口逃逸目录。"""

    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Profile name must be a simple file stem, got {name!r}")
