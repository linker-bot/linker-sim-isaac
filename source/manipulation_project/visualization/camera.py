"""摄像机视角辅助函数。

任务脚本可以调用这里的函数快速设置一个适合观察机械臂、灵巧手和 rope 端点的默认视角。
"""

from __future__ import annotations


def set_default_camera() -> None:
    """设置默认操作场景摄像机。

    参数:
        无。
    返回:
        无返回值；副作用是修改当前 Isaac viewport 的相机 eye/target。
    """

    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Gf

    set_camera_view(eye=Gf.Vec3d(1.35, -1.65, 1.05), target=Gf.Vec3d(0.0, -0.1, 0.42))
