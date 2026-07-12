"""GUI viewport 视角辅助函数。

动作脚本可以调用这里的函数快速设置一个适合观察机械臂、灵巧手和 rope 端点的默认视角。
本模块只处理 GUI 观察角度，不影响物理世界、机器人状态或日志输出。

Isaac viewport API 在函数内部延迟导入，保证导入 visualization 包本身不会要求 GUI 已启动。
headless 运行通常不需要调用该函数；如果调用方在无 viewport 环境执行，应由脚本入口决定
是否捕获 Isaac 侧异常。
"""

from __future__ import annotations

from linkerbot_sim.envs.visual_settings import ViewportViewSettings


def set_default_viewport_view(settings: ViewportViewSettings | None = None) -> None:
    """设置默认操作场景 GUI viewport 视角。

    参数:
        settings: viewport 的 eye、target 与 camera prim path；为 ``None`` 时使用默认值。
    返回:
        无返回值；副作用是修改当前 Isaac viewport 的 eye/target。
    """

    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Gf

    view = settings or ViewportViewSettings()
    set_camera_view(
        eye=Gf.Vec3d(*view.eye),
        target=Gf.Vec3d(*view.target),
        camera_prim_path=view.prim_path,
    )
