"""Isaac Sim SimulationApp 启动工具。

所有依赖 Isaac/Omni 的导入都放在函数内部，避免普通单元测试或
配置解析时误启动 Isaac Sim。
"""

from __future__ import annotations


def launch_simulation_app(*, gui: bool, width: int = 1280, height: int = 720):
    """按本项目默认参数启动 Isaac Sim。

    参数:
        gui: 是否启动带窗口的 GUI 模式；为假时使用 headless 配置。
        width: GUI viewport 宽度；headless 下会使用较小固定尺寸。
        height: GUI viewport 高度；headless 下会使用较小固定尺寸。
    返回:
        Isaac Sim ``SimulationApp`` 实例。
    """

    from isaacsim import SimulationApp

    return SimulationApp(
        {
            "headless": not gui,
            "hide_ui": False if gui else None,
            "disable_viewport_updates": not gui,
            "fast_shutdown": not gui,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "active_gpu": 0,
            "physics_gpu": 0,
            "width": width if gui else 640,
            "height": height if gui else 480,
            "window_width": 1440,
            "window_height": 900,
            "renderer": "RaytracedLighting",
            "anti_aliasing": 3 if gui else 0,
            "samples_per_pixel_per_frame": 1,
            "denoiser": False,
            "extra_args": (
                [
                    "--/rtx/materialDb/syncLoads=false",
                    "--/rtx/hydra/materialSyncLoads=false",
                ]
                if gui
                else [
                    "--/app/window/hideUi=1",
                    "--/rtx/materialDb/syncLoads=false",
                    "--/rtx/hydra/materialSyncLoads=false",
                ]
            ),
        }
    )
