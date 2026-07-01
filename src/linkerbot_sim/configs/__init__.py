"""配置 profile 名称解析入口。"""

from linkerbot_sim.configs.profiles import (
    load_default_controller_profiles,
    load_profile_yaml,
    profile_path,
)

__all__ = [
    "load_default_controller_profiles",
    "load_profile_yaml",
    "profile_path",
]
