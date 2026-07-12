"""Isaac Sim ``SimulationApp`` 启动工具。

本模块是项目进入 Isaac/Omni 运行时的统一边界。所有依赖 Isaac/Omni 的导入都放在
函数内部，避免普通单元测试、配置解析或静态检查阶段误启动 Isaac Sim。

调用顺序约定:
    1. 脚本入口先解析 CLI/YAML 中的 GUI、分辨率等运行参数。
    2. 调用 ``launch_simulation_app`` 创建 ``SimulationApp``。
    3. 再导入需要 ``omni``、``pxr`` 或 Isaac Core 的模块。

这样可以减少 “module imported before app launch” 类错误，也让 headless 和 GUI 模式的
渲染、GPU、窗口参数集中在一个位置。返回对象由脚本入口负责关闭；本模块不注册
atexit 钩子，也不持有全局单例，便于测试用例按需构造和释放。
"""

from __future__ import annotations

import os
from pathlib import Path

from linkerbot_sim.configs.runtime import SimulationAppSettings


_EULA_ENV_VAR = "OMNI_KIT_ACCEPT_EULA"
_ACCEPTED_EULA_VALUES = frozenset({"y", "yes", "1"})


def _experience_path() -> Path:
    """返回仓库随附的 Isaac Sim experience 路径。"""

    return Path(__file__).resolve().parents[3] / "apps" / "linkerbot_sim.python.kit"


def _require_eula_acceptance() -> None:
    """要求部署环境显式接受 Kit EULA，项目自身不会代替用户设置该状态。"""

    value = os.environ.get(_EULA_ENV_VAR)
    if value is not None and value.lower() in _ACCEPTED_EULA_VALUES:
        return
    raise RuntimeError(
        "Isaac Sim EULA has not been accepted. Set "
        f"{_EULA_ENV_VAR}=Y (or YES/1) in the deployment environment before "
        "launching; linkerbot_sim will not set it automatically."
    )


def _kit_config(settings: SimulationAppSettings) -> dict[str, object]:
    """把强类型运行设置转换为 Kit ``SimulationApp`` 支持的启动字段。

    GUI 与 headless 使用各自分辨率；可空开关仅在未显式配置时根据运行模式选择默认值。
    """

    gui = settings.gui
    gpu = settings.gpu
    render = settings.render
    width, height = render.gui_size if gui else render.headless_size
    window_width, window_height = render.window_size

    hide_ui = render.hide_ui
    if hide_ui is None:
        hide_ui = False if gui else None
    disable_viewport_updates = render.disable_viewport_updates
    if disable_viewport_updates is None:
        disable_viewport_updates = not gui
    fast_shutdown = render.fast_shutdown
    if fast_shutdown is None:
        fast_shutdown = not gui

    extra_args = [
        f"--/rtx/materialDb/syncLoads={str(render.material_sync_loads).lower()}",
        "--/rtx/hydra/materialSyncLoads="
        f"{str(render.hydra_material_sync_loads).lower()}",
    ]
    # Auto mode suppresses the Kit window in headless runs without overriding an
    # explicit YAML choice.
    if not gui and render.hide_ui is None:
        extra_args.insert(0, "--/app/window/hideUi=1")

    return {
        "headless": not gui,
        "hide_ui": hide_ui,
        "disable_viewport_updates": disable_viewport_updates,
        "fast_shutdown": fast_shutdown,
        "multi_gpu": gpu.multi_gpu,
        "max_gpu_count": gpu.max_gpu_count,
        "active_gpu": gpu.active_gpu,
        "physics_gpu": gpu.physics_gpu,
        "width": width,
        "height": height,
        "window_width": window_width,
        "window_height": window_height,
        "renderer": render.renderer,
        "anti_aliasing": (
            render.anti_aliasing_gui if gui else render.anti_aliasing_headless
        ),
        "samples_per_pixel_per_frame": render.samples_per_pixel_per_frame,
        "denoiser": render.denoiser,
        "extra_args": extra_args,
    }


def launch_simulation_app(settings: SimulationAppSettings):
    """按 resolved runtime profile 启动 Isaac Sim。

    参数:
        settings: 已严格校验的 SimulationApp、GPU 和渲染设置。
    返回:
        Isaac Sim ``SimulationApp`` 实例。
    """

    _require_eula_acceptance()

    # 必须延迟导入：Isaac 的 Python 包在导入时会初始化大量插件状态，若在
    # ``SimulationApp`` 创建前由其它模块间接导入，容易触发扩展加载顺序问题。
    from isaacsim import SimulationApp

    experience_path = _experience_path()
    if not experience_path.is_file():
        raise RuntimeError(f"Isaac Sim experience not found: {experience_path}")
    return SimulationApp(_kit_config(settings), experience=str(experience_path))
