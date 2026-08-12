"""显式 Gymnasium registration；导入包本身不会修改全局 registry。"""

from __future__ import annotations

GYMNASIUM_ENV_ID = "linkerbot/TBlockPush-Kaleidoscope-v1"


def register_gymnasium_envs() -> None:
    """幂等注册项目 vector entry point。"""

    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "Gymnasium registration requires the project 'training' extra"
        ) from exc
    if GYMNASIUM_ENV_ID in gym.registry:
        return
    gym.register(
        id=GYMNASIUM_ENV_ID,
        entry_point=None,
        vector_entry_point="linkerbot_sim.kaleidoscope.bootstrap:make_gymnasium_env",
        disable_env_checker=True,
        order_enforce=False,
    )


__all__ = ["GYMNASIUM_ENV_ID", "register_gymnasium_envs"]
