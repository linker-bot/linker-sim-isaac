"""Kaleidoscope 唯一 composition root。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from linkerbot_sim.kaleidoscope.env import TorchKaleidoscopeEnv

if TYPE_CHECKING:
    from linkerbot_sim.configuration.modes.kaleidoscope import KaleidoscopeConfig
    from linkerbot_sim.kaleidoscope.adapters.gymnasium import (
        GymnasiumKaleidoscopeAdapter,
    )
    from linkerbot_sim.kaleidoscope.runtime import KaleidoscopeRuntime

RuntimeFactory = Callable[..., "KaleidoscopeRuntime"]


def make_torch_env(
    *,
    config: "KaleidoscopeConfig | None" = None,
    profile: str = "physx_cuda",
    num_envs: int | None = None,
    runtime_factory: RuntimeFactory | None = None,
) -> TorchKaleidoscopeEnv:
    """加载严格配置并构造 PhysX CUDA 或项目自管 Newton native 环境。

    ``runtime_factory`` 是真实 Isaac adapter 与 pure/fake 测试之间唯一的替换点；生产默认工厂
    位于 ``kaleidoscope.isaac_adapter``。后端在构造前由 profile 判定，二者共享同一套
    Torch/state/task API，且不会在 facade import 时加载 Isaac。
    """

    if config is None:
        from linkerbot_sim.configuration import load_kaleidoscope_config

        config = load_kaleidoscope_config(profile)
    physics_selection = (
        str(getattr(config.physics, "engine", "")),
        str(getattr(config.physics, "execution", "")),
    )
    if config.mode != "kaleidoscope" or physics_selection not in {
        ("physx", "cuda"),
        ("newton", "cuda"),
    }:
        raise ValueError(
            "Kaleidoscope bootstrap only accepts PhysX CUDA or Newton CUDA config"
        )
    if num_envs is not None and (type(num_envs) is not int or num_envs < 1):
        raise ValueError("num_envs override must be a positive int")
    if runtime_factory is None:
        from linkerbot_sim.kaleidoscope.isaac_adapter import (
            create_kaleidoscope_runtime,
        )

        runtime_factory = create_kaleidoscope_runtime
    runtime = runtime_factory(config=config, num_envs=num_envs)
    return TorchKaleidoscopeEnv(runtime)


def make_viewport_env(
    *,
    config: "KaleidoscopeConfig | None" = None,
    profile: str = "physx_cuda",
    viewport: object | None = None,
    viewport_profile: str = "kaleidoscope",
    num_envs: int | None = None,
    runtime_factory: RuntimeFactory | None = None,
) -> TorchKaleidoscopeEnv:
    """构造显式 human viewport 环境；默认训练入口仍保持 renderer-free。

    viewport 是独立冷配置，不参与 episode snapshot 的语义指纹。它只选择 Kit、
    viewport、可见 world 与 render cadence，不改变 task、physics 或 state schema。
    """

    if config is None or viewport is None:
        from linkerbot_sim.configuration import (
            load_kaleidoscope_config,
            load_kaleidoscope_viewport_config,
        )

        if config is None:
            config = load_kaleidoscope_config(profile)
        if viewport is None:
            viewport = load_kaleidoscope_viewport_config(viewport_profile)
    if num_envs is not None and (type(num_envs) is not int or num_envs < 1):
        raise ValueError("num_envs override must be a positive int")
    count = config.environments.num_envs if num_envs is None else num_envs
    selected_env = int(getattr(viewport, "selected_env"))
    if selected_env >= count:
        raise ValueError("viewport.selected_env must be below the final num_envs")
    if runtime_factory is None:
        from linkerbot_sim.kaleidoscope.isaac_adapter import (
            create_kaleidoscope_runtime,
        )

        runtime_factory = create_kaleidoscope_runtime
    runtime = runtime_factory(
        config=config,
        num_envs=num_envs,
        viewport=viewport,
    )
    return TorchKaleidoscopeEnv(runtime)


def make_gymnasium_env(
    *,
    config: "KaleidoscopeConfig | None" = None,
    profile: str = "physx_cuda",
    num_envs: int | None = None,
    autoreset_mode: str = "disabled",
    render_mode: str | None = None,
    viewport_profile: str = "kaleidoscope",
    runtime_factory: RuntimeFactory | None = None,
    **_vector_kwargs: object,
) -> "GymnasiumKaleidoscopeAdapter":
    """Gymnasium vector_entry_point；NumPy 转换只发生在返回的 adapter。"""

    # Gymnasium 是显式的 CPU/NumPy 边界。延迟导入保证 native Torch 与 skrl 热路径
    # 不因加载 composition root 而初始化 Gymnasium 或 NumPy 适配层。
    from linkerbot_sim.kaleidoscope.adapters.gymnasium import (
        GymnasiumKaleidoscopeAdapter,
    )

    if render_mode is None:
        env = make_torch_env(
            config=config,
            profile=profile,
            num_envs=num_envs,
            runtime_factory=runtime_factory,
        )
    elif render_mode == "human":
        env = make_viewport_env(
            config=config,
            profile=profile,
            viewport_profile=viewport_profile,
            num_envs=num_envs,
            runtime_factory=runtime_factory,
        )
    else:
        raise ValueError("render_mode must be None or 'human'")
    return GymnasiumKaleidoscopeAdapter(
        env,
        autoreset_mode=autoreset_mode,
        render_mode=render_mode,
    )


__all__ = [
    "RuntimeFactory",
    "make_gymnasium_env",
    "make_torch_env",
    "make_viewport_env",
]
