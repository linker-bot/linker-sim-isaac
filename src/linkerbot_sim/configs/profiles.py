"""按稳定 profile 名称加载仓库内配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from linkerbot_sim.controllers.config import load_controller_profiles
from linkerbot_sim.utils.config import load_yaml
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
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Profile name must be a simple file stem, got {name!r}")
    return PROFILE_GROUP_DIRS[group] / f"{name}.yaml"


def load_profile_yaml(group: str, name: str) -> dict[str, Any]:
    """读取仓库内指定 profile YAML。"""

    return load_yaml(profile_path(group, name))


def load_default_controller_profiles():
    """读取项目默认 controller 配置集合。"""

    return load_controller_profiles(CONFIGS_ROOT / "controllers")
