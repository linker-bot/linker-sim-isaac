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


def launch_simulation_app(*, gui: bool, width: int = 1280, height: int = 720):
    """按本项目默认参数启动 Isaac Sim。

    参数:
        gui: 是否启动带窗口的 GUI 模式；为假时使用 headless 配置。
        width: GUI viewport 宽度；headless 下会使用较小固定尺寸。
        height: GUI viewport 高度；headless 下会使用较小固定尺寸。
    返回:
        Isaac Sim ``SimulationApp`` 实例。
    """

    # 必须延迟导入：Isaac 的 Python 包在导入时会初始化大量插件状态，若在
    # ``SimulationApp`` 创建前由其它模块间接导入，容易触发扩展加载顺序问题。
    from isaacsim import SimulationApp

    # 这里显式写出项目使用的渲染和 GPU 约定，而不是依赖 Isaac 默认值：
    # GUI 模式优先保证可视化质量；headless 模式关闭 viewport 更新和抗锯齿，
    # 让批量测试/轨迹生成更快，并避免无显示环境下创建窗口。
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
