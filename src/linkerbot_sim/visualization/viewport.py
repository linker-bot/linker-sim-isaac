"""GUI viewport 视角辅助函数。

动作脚本可以调用这里的函数快速设置一个适合观察机械臂、灵巧手和 rope 端点的默认视角。
本模块只处理 GUI 观察角度，不影响物理世界、机器人状态或日志输出。

Isaac viewport API 在函数内部延迟导入，保证导入 visualization 包本身不会要求 GUI 已启动。
headless 运行通常不需要调用该函数；如果调用方在无 viewport 环境执行，应由脚本入口决定
是否捕获 Isaac 侧异常。
"""

from __future__ import annotations

from linkerbot_sim.configuration.scenes import ViewportSettings


def set_default_viewport_view(settings: ViewportSettings | None = None) -> None:
    """设置默认操作场景 GUI viewport 视角。

    参数:
        settings: viewport 的 eye、target 与 camera prim path；为 ``None`` 时使用默认值。
    返回:
        无返回值；副作用是修改当前 Isaac viewport 的 eye/target。
    """

    view = settings or ViewportSettings()
    set_camera_view(
        eye=view.eye,
        target=view.target,
        camera_prim_path=view.prim_path,
    )


def set_camera_view(
    *,
    eye: object,
    target: object,
    camera_prim_path: str,
) -> None:
    """Set one viewport camera without importing the Isaac Core carrier."""

    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    from pxr import Gf

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("cannot configure camera view without an active viewport")
    eye_value = tuple(float(value) for value in eye)  # type: ignore[arg-type]
    target_value = tuple(float(value) for value in target)  # type: ignore[arg-type]
    if len(eye_value) != 3 or len(target_value) != 3:
        raise ValueError("viewport eye and target must contain three values")
    state = ViewportCameraState(str(camera_prim_path), viewport)
    state.set_position_world(Gf.Vec3d(*eye_value), True)
    state.set_target_world(Gf.Vec3d(*target_value), True)


def set_viewport_camera_navigation_enabled(
    viewport_window: object,
    *,
    enabled: bool,
) -> None:
    """只切换一个 ``ViewportWindow`` 的鼠标相机导航 layer。

    Isaac Sim 6.0.1 的 camera manipulator 是进程级 extension，但每个 viewport window
    都拥有独立的 ``Camera/manipulator`` layer。这里有意使用该固定版本提供的
    ``_find_viewport_layer`` 桥接点，使主观察视口可以交互，同时让 SyntheticData
    相机窗口保持只读；找不到 layer 时直接失败，避免传感器窗口静默变成可编辑视口。
    """

    if type(enabled) is not bool:
        raise TypeError("enabled must be a boolean")
    find_layer = getattr(viewport_window, "_find_viewport_layer", None)
    if not callable(find_layer):
        raise RuntimeError("viewport window does not expose per-window layers")
    camera_layer = find_layer("Camera", "manipulator")
    if camera_layer is None:
        raise RuntimeError("viewport camera manipulator layer is unavailable")
    camera_layer.visible = enabled
    if bool(getattr(camera_layer, "visible", not enabled)) != enabled:
        raise RuntimeError("viewport camera manipulator visibility did not change")


__all__ = [
    "set_camera_view",
    "set_default_viewport_view",
    "set_viewport_camera_navigation_enabled",
]
